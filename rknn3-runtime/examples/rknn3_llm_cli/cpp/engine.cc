// Shared inference engine for the rknn3 LLM demos.
//
// This implements the common runtime pipeline used by both front-ends:
//   - rknn3_llm_cli           (SESSION decode: one session_run per turn)
//   - rknn3_llm_step_decode   (EXTERNAL decode: one session_run per token)
//
// The decode strategy is selected once through SessionState::decode_mode and
// honored by run_turn(): everything else (init, callbacks, session setup,
// sampling, LoRA, rope, kvcache checkpoint, benchmark, status) is shared.

#include "engine.h"
#include "args.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

/*=============================================================================
   Global state
=============================================================================*/

SessionState g_state;
const char*  g_logits_name = nullptr;
StageResultState g_stage_result;

/* whether stdin is an interactive terminal (enables /stop during generation) */
bool g_stdin_tty = false;

/* SESSION-decode timing: first decode token timestamp captured by the result
   callback, polled by run_turn() to compute TTFT / prefill / decode times. */
static struct timeval g_run_start;
static struct timeval g_first_token;
static bool           g_first_decode = true;

/* SESSION-decode interrupt support: set by the generation worker, polled by
   the interrupt monitor. */
static std::atomic<bool> g_run_done(false);
static std::atomic<bool> g_interrupted(false);
static std::atomic<int>  g_run_ret(RKNN3_SUCCESS);

/*=============================================================================
   Callbacks
=============================================================================*/

static int result_callback(void* userdata, RKLLMResult* result, LLMCallState state)
{
  (void)userdata;
  Tokenizer* tokenizer = g_state.tokenizer;

  switch (state)
  {
    case RKLLM_RUN_NORMAL:
    {
      if (!result) break;

      /* capture the last produced token so the external decode loop can feed
         it back as the next input (no-op overhead in SESSION decode). */
      if (result->num_tokens > 0)
      {
        std::lock_guard<std::mutex> lock(g_stage_result.mutex);
        g_stage_result.next_token = result->token_ids[result->num_tokens - 1];
        g_stage_result.has_token = true;
      }

      /* the model (e.g. with a mismatched LoRA) may emit out-of-range token
         ids; guard before TokenToPiece/Decode to avoid crashing on them */
      bool valid = true;
      for (int i = 0; i < result->num_tokens; ++i)
      {
        if (result->token_ids[i] < 0 || result->token_ids[i] >= g_state.vocab_info.vocab_size)
        {
          valid = false;
          break;
        }
      }

      if (valid)
      {
        std::string piece;
        if (result->num_tokens == 1)
        {
          piece = tokenizer->TokenToPiece(result->token_ids[0]);
        }
        else
        {
          piece = tokenizer->Decode(result->token_ids, result->num_tokens);
        }
        printf("%s", piece.c_str());
      }
      else
      {
        VLOG("[result_callback] skipping out-of-range token id(s)\n");
      }
      fflush(stdout);

      if (g_first_decode)
      {
        gettimeofday(&g_first_token, NULL);
        g_first_decode = false;
      }
      break;
    }
    case RKLLM_RUN_WAITING:
      /* waiting for a complete UTF-8 sequence */
      break;
    case RKLLM_RUN_FINISH:
    case RKLLM_RUN_MAX_NEW_TOKEN_REACHED:
      /* SESSION decode terminates the token stream here; the REPL adds the
         turn separator. In EXTERNAL decode each step ends with one of these
         states, so the newline is printed by the front-end instead
         (SessionState::callback_print_newline). */
      if (g_state.callback_print_newline)
      {
        printf("\n");
        fflush(stdout);
      }
      break;
    case RKLLM_RUN_STOP:
      printf("\n[generation stopped by user]\n");
      fflush(stdout);
      break;
    case RKLLM_RUN_ERROR:
      printf("\n[error occurred during inference]\n");
      fflush(stdout);
      break;
    case RKLLM_RUN_PAUSE:
    case RKLLM_RUN_RESUME:
    default:
      break;
  }
  return 0;
}

static int tokenizer_callback(void* userdata, const char* text, int32_t text_len,
                              int32_t* tokens, int32_t n_tokens_max)
{
  Tokenizer* tokenizer = (Tokenizer*)userdata;
  VLOG("[tokenizer_callback] len=%d text='%.*s'\n", text_len, text_len, text);
  int n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);
  if (n_tokens <= 0)
  {
    LOGW("tokenizer failed for %s\n", text);
    return n_tokens;
  }
  return n_tokens;
}

static int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens,
                          void* embed, uint64_t len)
{
  embedding_info* embed_info = (embedding_info*)userdata;
  uint64_t row_size = (uint64_t)embed_info->embedding_dim * sizeof(float16);

  if (len != num_tokens * row_size)
  {
    LOGW("invalid embed buffer\n");
    return -1;
  }

  for (uint64_t i = 0; i < num_tokens; ++i)
  {
    int32_t token_id = tokens[i];
    if (token_id < 0 || token_id >= embed_info->vocab_size)
    {
      LOGW("invalid token id: %d\n", token_id);
      return -1;
    }
    memcpy((unsigned char*)embed + i * row_size,
           embed_info->embedding_data + (uint64_t)token_id * embed_info->embedding_dim,
           row_size);
  }
  return 0;
}

/*=============================================================================
   Initialization / teardown
=============================================================================*/

rknn3_context init_context_and_model(const char* model_path, const char* weight_path,
                                     uint32_t core_mask, const char* device_id)
{
  int ret = 0;
  rknn3_config config;
  rknn3_context ctx = 0;

  rknn3_init_extend init_extend;
  memset(&init_extend, 0, sizeof(init_extend));
  init_extend.device_id = (char*)device_id; /* NULL -> the runtime picks the default device */

  ret = rknn3_init(&ctx, &init_extend);
  if (ret < 0)
  {
    LOGR("rknn3_init fail! ret=%d\n", ret);
    return 0;
  }

  ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
  if (ret != RKNN3_SUCCESS)
  {
    LOGR("rknn3_load_model_from_path failed! ret=%d\n", ret);
    rknn3_destroy(ctx);
    return 0;
  }

  memset(&config, 0, sizeof(config));
  config.run_core_mask = core_mask;

  ret = rknn3_model_init(ctx, &config);
  if (ret != RKNN3_SUCCESS)
  {
    LOGR("rknn3_model_init failed! ret=%d\n", ret);
    rknn3_destroy(ctx);
    return 0;
  }

  return ctx;
}

int init_tokenizer_and_embedding(const char* tokenizer_path, const char* embedding_path)
{
  g_state.tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
  if (!g_state.tokenizer)
  {
    LOGR("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
    return -1;
  }
  if (!g_state.tokenizer->GetVocabInfo(&g_state.vocab_info))
  {
    LOGR("GetVocabInfo failed!\n");
    return -1;
  }
  embedding_info* ei = &g_state.embed_info;
  struct stat st;
  memset(ei, 0, sizeof(*ei));
  ei->fd = open(embedding_path, O_RDONLY);
  if (ei->fd == -1)
  {
    LOGR("Failed to open embedding file: %s\n", embedding_path);
    return -1;
  }
  if (fstat(ei->fd, &st) == -1)
  {
    LOGR("Failed to get embedding file size\n");
    close(ei->fd);
    ei->fd = -1;
    return -1;
  }
  ei->embedding_data = (float16*)mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, ei->fd, 0);
  if (ei->embedding_data == MAP_FAILED)
  {
    LOGR("Failed to mmap embedding file\n");
    close(ei->fd);
    ei->fd = -1;
    ei->embedding_data = nullptr;
    return -1;
  }
  ei->data_size = (size_t)st.st_size;
  ei->vocab_size = g_state.vocab_info.vocab_size;

  /* embed_callback indexes embedding_data[token_id * embedding_dim]: reject
     truncated / mismatched files here instead of reading OOB there */
  bool valid = ei->vocab_size > 0 &&
               (uint64_t)st.st_size >= (uint64_t)ei->vocab_size * sizeof(float16) &&
               st.st_size % ((off_t)ei->vocab_size * (off_t)sizeof(float16)) == 0;
  if (valid)
  {
    ei->embedding_dim = (int)(((uint64_t)st.st_size / (uint64_t)ei->vocab_size) / sizeof(float16));
    valid = ei->embedding_dim > 0;
  }
  if (!valid)
  {
    LOGR("Embedding file '%s' (size %lld) does not match vocab_size %d: "
         "expected a positive multiple of %d * %zu bytes\n",
         embedding_path, (long long)st.st_size, ei->vocab_size,
         ei->vocab_size, sizeof(float16));
    munmap(ei->embedding_data, (size_t)st.st_size);
    close(ei->fd);
    ei->fd = -1;
    ei->embedding_data = nullptr;
    ei->vocab_size = 0;
    return -1;
  }

  VLOG("vocab_size=%d embedding_dim=%d\n", ei->vocab_size, ei->embedding_dim);
  return 0;
}

static void destroy_context()
{
  if (g_state.session)
  {
    if (g_state.lora.enabled)
      lora_detach_session(&g_state.lora, g_state.session);
    rknn3_session_destroy(g_state.session);
    g_state.session = nullptr;
  }

  if (g_state.ctx)
  {
    rknn3_destroy(g_state.ctx);
    g_state.ctx = 0;
  }
}

void release_resources()
{
  /* disable + unload LoRA while the session/context are still alive */
  lora_teardown(&g_state.lora);
  rope_teardown(&g_state.rope);

  destroy_context();

  if (g_state.embed_info.embedding_data)
  {
    munmap(g_state.embed_info.embedding_data, g_state.embed_info.data_size);
    g_state.embed_info.embedding_data = nullptr;
  }
  if (g_state.embed_info.fd != -1)
  {
    close(g_state.embed_info.fd);
    g_state.embed_info.fd = -1;
  }
  if (g_state.tokenizer)
  {
    delete g_state.tokenizer;
    g_state.tokenizer = nullptr;
  }
}

int require_rope_cache(const char* rope_tensor_path)
{
  bool host_storage = g_state.llm_config.rope_cache_host_storage != 0;
  bool has_cache = rope_tensor_path && rope_tensor_path[0] != '\0';

  if (!host_storage)
  {
    /* model stores rope caches on device: an explicitly provided cache is
       superfluous. Warn and ignore it rather than failing the run. */
    if (has_cache)
      LOGW("model does not require host-stored rope cache (rope_cache_host_storage=0); "
           "--rope-tensor '%s' ignored\n", rope_tensor_path);
    return 0;
  }

  /* host-stored rope is mandatory for this model */
  if (!has_cache)
  {
    LOGR("model requires host-stored rope cache (rope_cache_host_storage=%u), "
         "please pass --rope-tensor <path>\n",
         g_state.llm_config.rope_cache_host_storage);
    return -1;
  }

  if (rope_setup(&g_state.rope, g_state.ctx, rope_tensor_path) != 0)
  {
    LOGR("rope_setup failed for '%s'; cannot run model without its rope cache\n",
         rope_tensor_path);
    return -1;
  }
  LOGI("external rope cache enabled\n");
  return 0;
}

/* apply the current chat template + system prompt to the session; if no
   template component was explicitly configured, the runtime keeps using the
   model's built-in chat template (which also drives thinking via
   enable_thinking). */
static int apply_chat_template()
{
  if (g_state.system_prompt.empty() && g_state.prompt_prefix.empty() && g_state.prompt_postfix.empty())
    return 0;

  int ret = rknn3_session_set_chat_template(g_state.session,
                                            g_state.system_prompt.c_str(),
                                            g_state.prompt_prefix.c_str(),
                                            g_state.prompt_postfix.c_str());
  if (ret != RKNN3_SUCCESS)
  {
    LOGR("rknn3_session_set_chat_template failed, ret=%d\n", ret);
    return ret;
  }
  return 0;
}

int apply_llm_param()
{
  g_state.params.vocab_info.ignore_eos_token = g_state.ignore_eos ? 1 : 0;
  int ret = rknn3_session_set_llm_param(g_state.session, &g_state.params, 1);
  if (ret != RKNN3_SUCCESS)
  {
    LOGR("rknn3_session_set_llm_param failed, ret=%d\n", ret);
    return ret;
  }
  return 0;
}

int apply_kvcache_checkpoint_policy()
{
  if (!g_state.checkpoint_enabled)
    return 0;

  rknn3_session_clear_kvcache(g_state.session, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
  int ret = rknn3_session_set_kvcache_policy(g_state.session,
                                             RKNN3_KVCACHE_POLICY_SAVE_CHECKPOINT,
                                             &g_state.kvcache_policy);
  if (ret != RKNN3_SUCCESS)
  {
    LOGR("rknn3_session_set_kvcache_policy failed, ret=%d\n", ret);
    return ret;
  }
  return 0;
}

/* create a fresh session on the initialized context, with callbacks and the
   current chat template applied. Used at startup and by /thinking to fully
   reset the session state (kvcache clear alone is not enough). */
int create_session()
{
  bool lora_was_enabled = g_state.lora.enabled;

  if (g_state.session)
  {
    /* detach an active LoRA from the old session before destroying it, so
       the adapter is not left bound to a destroyed session */
    if (g_state.lora.enabled)
      lora_detach_session(&g_state.lora, g_state.session);
    rknn3_session_destroy(g_state.session);
    g_state.session = nullptr;
  }

  g_state.session = rknn3_session_init(g_state.ctx, &g_state.params, 1);
  if (!g_state.session)
  {
    LOGR("rknn3_session_init failed\n");
    return -1;
  }

  RKLLMCallback callback;
  memset(&callback, 0, sizeof(callback));
  callback.result_callback = result_callback;
  callback.result_userdata = g_state.tokenizer;
  callback.embed_callback = embed_callback;
  callback.embed_userdata = &g_state.embed_info;
  callback.tokenizer_callback = tokenizer_callback;
  callback.tokenizer_userdata = g_state.tokenizer;
  if (g_state.rope.enabled)
  {
    callback.input_callback = rope_input_callback;
    callback.input_userdata = &g_state.rope;
    callback.input_tensors_index = g_state.rope.ext_input_indices;
    callback.n_input_tensors = (uint32_t)g_state.rope.n_ext_inputs;
  }

  int ret = rknn3_session_set_callback(g_state.session, &callback);
  if (ret != RKNN3_SUCCESS)
  {
    LOGR("rknn3_session_set_callback failed, ret=%d\n", ret);
    return -1;
  }

  ret = apply_chat_template();
  if (ret != RKNN3_SUCCESS) return ret;

  /* apply the kvcache checkpoint policy (step_decode / linear-attention) */
  ret = apply_kvcache_checkpoint_policy();
  if (ret != RKNN3_SUCCESS) return ret;

  /* a recreated session loses the LoRA binding: re-apply it if it was enabled */
  g_state.lora.session = g_state.session;
  if (g_state.lora.loaded && lora_was_enabled)
  {
    int lret = lora_reapply_session(&g_state.lora, g_state.session);
    if (lret != RKNN3_SUCCESS)
    {
      LOGR("lora_reapply_session failed, ret=%d\n", lret);
      g_state.lora.enabled = false;
    }
  }

  return 0;
}

/* fill rknn3_llm_param from the options + queried vocab info, then create a
   fresh session on the initialized context. */
int init_session(const CommandLineOptions& opt)
{
  rknn3_llm_param* params = &g_state.params;
  memset(params, 0, sizeof(*params));

  params->logits_name = (char*)(g_logits_name ? g_logits_name : "output");
  params->max_context_len = (int32_t)g_state.llm_config.max_ctx_len;
  params->sampling_param.top_k = opt.top_k;
  params->sampling_param.top_p = opt.top_p;
  params->sampling_param.temperature = opt.temperature;
  params->sampling_param.repeat_penalty = opt.repeat_penalty;
  params->sampling_param.frequency_penalty = opt.frequency_penalty;
  params->sampling_param.presence_penalty = opt.presence_penalty;

  params->vocab_info.vocab_size = g_state.vocab_info.vocab_size;
  params->vocab_info.n_special_bos_id = g_state.vocab_info.n_special_bos_id;
  params->vocab_info.n_special_eos_id = g_state.vocab_info.n_special_eos_id;
  memcpy(params->vocab_info.special_bos_id, g_state.vocab_info.special_bos_id,
         sizeof(g_state.vocab_info.special_bos_id));
  memcpy(params->vocab_info.special_eos_id, g_state.vocab_info.special_eos_id,
         sizeof(g_state.vocab_info.special_eos_id));
  params->vocab_info.linefeed_id = g_state.vocab_info.linefeed_id;
  params->vocab_info.ignore_eos_token = g_state.ignore_eos ? 1 : 0;

  /* in EXTERNAL decode each single step ends with RKLLM_RUN_FINISH /
     MAX_NEW_TOKEN_REACHED; the callback must not append a newline per step,
     the front-end prints it once at the end of the turn. */
  g_state.callback_print_newline = (g_state.decode_mode == DecodeMode::SESSION);

  return create_session();
}

/*=============================================================================
   Inference
=============================================================================*/

void reset_stage_result()
{
  std::lock_guard<std::mutex> lock(g_stage_result.mutex);
  g_stage_result.has_token = false;
  g_stage_result.next_token = -1;
}

bool get_stage_token(int32_t* token)
{
  std::lock_guard<std::mutex> lock(g_stage_result.mutex);
  if (!g_stage_result.has_token)
  {
    return false;
  }
  *token = g_stage_result.next_token;
  return true;
}

/* one inference step: prefill the full prompt / token list, or decode a
   single token. max_new_tokens is forced to 1 so the runtime performs a
   single step and the caller drives the loop (EXTERNAL decode). */
bool run_inference_once(const char* prompt, const std::vector<int32_t>* input_tokens,
                        bool keep_history, uint64_t* step_input_tokens)
{
  rknn3_llm_infer_param infer_param;
  memset(&infer_param, 0, sizeof(infer_param));
  infer_param.keep_history = keep_history ? 1 : 0;
  infer_param.max_new_tokens = 1;
  infer_param.prefill_only = true;
  infer_param.disable_sampling = false;

  reset_stage_result();

  rknn3_llm_input input;
  memset(&input, 0, sizeof(input));
  input.llm_input.enable_thinking = g_state.enable_thinking;

  if (prompt)
  {
    input.input_type = RKNN3_LLM_INPUT_PROMPT;
    input.llm_input.prompt = prompt;
  }
  else
  {
    if (!input_tokens || input_tokens->empty())
    {
      return false;
    }
    input.input_type = RKNN3_LLM_INPUT_TOKEN;
    input.llm_input.tokens = const_cast<int32_t*>(input_tokens->data());
    input.llm_input.n_tokens = input_tokens->size();
  }

  int ret = rknn3_session_run(g_state.session, &input, 1, &infer_param);

  if (step_input_tokens)
  {
    RKLLMRunState state;
    memset(&state, 0, sizeof(state));
    if (rknn3_session_query_state(g_state.session, &state) != RKNN3_SUCCESS)
    {
      VLOG("rknn3_session_query_state failed\n");
    }
    *step_input_tokens = state.n_input_tokens;
  }

  return ret == RKNN3_SUCCESS;
}

/* SESSION decode: one rknn3_session_run(max_new_tokens=N) that performs
   prefill + all decode internally. Supports /stop interruption on a TTY. */
static int run_session_decode(const char* prompt_text, const std::vector<int32_t>* tokens,
                              bool keep_history, bool allow_interrupt, RunStats* stats_out)
{
  int ret = 0;

  rknn3_llm_infer_param infer_param;
  memset(&infer_param, 0, sizeof(infer_param));
  infer_param.keep_history = keep_history ? 1 : 0;
  infer_param.max_new_tokens = g_state.max_new_tokens;

  rknn3_llm_input input;
  memset(&input, 0, sizeof(input));

  if (tokens && !tokens->empty())
  {
    input.input_type = RKNN3_LLM_INPUT_TOKEN;
    input.llm_input.tokens = const_cast<int32_t*>(tokens->data());
    input.llm_input.n_tokens = tokens->size();
    input.llm_input.enable_thinking = g_state.enable_thinking;
  }
  else
  {
    input.input_type = RKNN3_LLM_INPUT_PROMPT;
    input.llm_input.prompt = prompt_text;
    input.llm_input.enable_thinking = g_state.enable_thinking;
    VLOG("[run_session_decode] keep_history=%d thinking=%d prompt='%s'\n",
         keep_history, g_state.enable_thinking, prompt_text ? prompt_text : "");
  }

  gettimeofday(&g_run_start, NULL);
  g_first_decode = true;
  g_interrupted = false;
  g_run_done = false;

  /* interrupt monitor only makes sense on an interactive terminal; for
     piped / redirected stdin it would consume buffered lines, so fall back
     to a plain synchronous run. */
  bool can_interrupt = allow_interrupt && g_stdin_tty;

  if (can_interrupt)
  {
    /* run generation on a worker thread while the main thread watches stdin
       so that '/stop' can interrupt a running generation. The session_run
       return value is carried back via g_run_ret so failures are not lost. */
    g_run_ret.store(RKNN3_SUCCESS);
    std::thread worker([&]() {
      int r = rknn3_session_run(g_state.session, &input, 1, &infer_param);
      g_run_ret.store(r);
      if (r != RKNN3_SUCCESS)
      {
        LOGR("rknn3_session_run failed, ret=%d\n", r);
      }
      g_run_done = true;
    });

    /* temporary non-blocking reads: select() on a canonical-mode tty only
       fires when a complete line is available, so no partial-line blocking */
    int saved_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, saved_flags | O_NONBLOCK);

    while (!g_run_done.load())
    {
      struct timeval tv;
      tv.tv_sec = 0;
      tv.tv_usec = 100000; /* 100 ms */
      fd_set rfds;
      FD_ZERO(&rfds);
      FD_SET(STDIN_FILENO, &rfds);

      int sel = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
      if (sel > 0 && FD_ISSET(STDIN_FILENO, &rfds))
      {
        char buf[512];
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n > 0)
        {
          std::string chunk(buf, (size_t)n);
          size_t pos;
          while ((pos = chunk.find('\n')) != std::string::npos)
          {
            std::string line = trim(chunk.substr(0, pos));
            chunk.erase(0, pos + 1);
            if (line == "/stop" || line == "/cancel" || line == "/interrupt")
            {
              g_interrupted = true;
              rknn3_session_stop(g_state.session);
              VLOG("[interrupt monitor] stop requested\n");
              chunk.clear();
              break;
            }
            VLOG("[interrupt monitor] input during generation ignored: '%s'\n", line.c_str());
          }
          if (g_interrupted.load()) break;
        }
        else if (n == 0)
        {
          break; /* EOF */
        }
      }
    }

    fcntl(STDIN_FILENO, F_SETFL, saved_flags);
    worker.join();
    ret = g_run_ret.load();
  }
  else
  {
    ret = rknn3_session_run(g_state.session, &input, 1, &infer_param);
    if (ret != RKNN3_SUCCESS)
    {
      LOGR("rknn3_session_run failed, ret=%d\n", ret);
      return ret;
    }
  }

  if (ret != RKNN3_SUCCESS)
  {
    LOGR("session run failed, ret=%d\n", ret);
    return ret;
  }

  struct timeval run_end;
  gettimeofday(&run_end, NULL);

  RKLLMRunState state;
  memset(&state, 0, sizeof(state));

  if (rknn3_session_query_state(g_state.session, &state) != RKNN3_SUCCESS)
  {
    VLOG("rknn3_session_query_state failed\n");
  }

  RunStats s;
  memset(&s, 0, sizeof(s));

  double prefill_us;
  if (g_first_decode)
  {
    /* no token was produced (e.g. immediate EOS / error): use whole run as prefill */
    prefill_us = elapsed_ms(g_run_start, run_end) * 1e3;
    s.ttft_us = (int64_t)prefill_us;
    s.n_prefill_tokens = state.n_input_tokens;
    s.n_decode_tokens = 0;
  }
  else
  {
    prefill_us = (double)(g_first_token.tv_sec - g_run_start.tv_sec) * 1e6 +
                 (double)(g_first_token.tv_usec - g_run_start.tv_usec);
    s.ttft_us = (int64_t)prefill_us;
    s.n_prefill_tokens = state.n_prefill_tokens;
    s.n_decode_tokens = state.n_decode_tokens;
  }

  s.prefill_ms = prefill_us / 1e3;
  s.decode_ms = g_first_decode ? 0.0 : elapsed_ms(g_first_token, run_end);
  s.prefill_tps = s.prefill_ms > 0.0 ? s.n_prefill_tokens / (s.prefill_ms / 1e3) : 0.0;
  s.decode_tps = s.decode_ms > 0.0 ? s.n_decode_tokens / (s.decode_ms / 1e3) : 0.0;
  s.prefill_tpt_ms = s.n_prefill_tokens > 0 ? s.prefill_ms / s.n_prefill_tokens : 0.0;
  s.decode_tpt_ms = s.n_decode_tokens > 0 ? s.decode_ms / s.n_decode_tokens : 0.0;
  s.stopped_by_user = g_interrupted.load() ? 1 : 0;
  s.valid = true;

  if (stats_out) *stats_out = s;
  return 0;
}

/* EXTERNAL decode: prefill the prompt (which already produces the first
   token), then loop single-token decode steps, feeding the previous token
   back until max_new_tokens / EOS / interrupt. */
static int run_external_decode(const char* prompt_text, const std::vector<int32_t>* tokens,
                               bool keep_history, RunStats* out)
{
  RunStats s;
  memset(&s, 0, sizeof(s));
  int ret = 0;

  timeval turn_start;
  gettimeofday(&turn_start, NULL);

  uint64_t prefill_input = 0;
  if (!run_inference_once(prompt_text, tokens, keep_history, &prefill_input))
  {
    return -1;
  }

  /* TTFT: time from turn start to end of prefill (the prefill step also
     produces the first token in this external single-step decode scheme). */
  timeval prefill_end;
  gettimeofday(&prefill_end, NULL);
  s.ttft_us = (int64_t)elapsed_us(turn_start, prefill_end);
  s.prefill_ms = elapsed_us(turn_start, prefill_end) / 1e3;
  s.n_prefill_tokens = prefill_input;

  int next_token = -1;
  get_stage_token(&next_token);

  uint64_t decode_steps = g_state.max_new_tokens > 0 ? (uint64_t)g_state.max_new_tokens : 0;
  double decode_total_ms = 0.0;
  uint64_t n_decode = 0;

  for (uint64_t step = 1; step < decode_steps && next_token >= 0; ++step)
  {
    std::vector<int32_t> token_vec(1, next_token);
    VLOG("[Decode %llu] token=%d\n", (unsigned long long)(step + 1), next_token);

    timeval d0;
    timeval d1;
    gettimeofday(&d0, NULL);
    bool ok = run_inference_once(nullptr, &token_vec, /*keep_history=*/true, nullptr);
    gettimeofday(&d1, NULL);
    decode_total_ms += elapsed_us(d0, d1) / 1e3;
    if (!ok)
    {
      printf("decode step %llu failed\n", (unsigned long long)(step + 1));
      ret = -1;
      break;
    }

    if (!get_stage_token(&next_token))
    {
      printf("decode step %llu did not return token from result_callback\n",
             (unsigned long long)(step + 1));
      break;
    }
    n_decode++;
    if (!g_state.ignore_eos && next_token == g_state.vocab_info.special_eos_id[0])
    {
      VLOG("\ndecode step %llu reached EOS token\n", (unsigned long long)(step + 1));
      break;
    }
  }

  s.decode_ms = decode_total_ms;
  s.n_decode_tokens = n_decode;
  s.prefill_tps = s.prefill_ms > 0.0 ? s.n_prefill_tokens / (s.prefill_ms / 1e3) : 0.0;
  s.decode_tps = s.decode_ms > 0.0 ? s.n_decode_tokens / (s.decode_ms / 1e3) : 0.0;
  s.prefill_tpt_ms = s.n_prefill_tokens > 0 ? s.prefill_ms / s.n_prefill_tokens : 0.0;
  s.decode_tpt_ms = s.n_decode_tokens > 0 ? s.decode_ms / s.n_decode_tokens : 0.0;
  s.valid = true;

  if (out) *out = s;
  return ret;
}

int run_turn(const char* prompt_text, const std::vector<int32_t>* tokens,
             bool keep_history, bool allow_interrupt, RunStats* stats_out)
{
  if (g_state.decode_mode == DecodeMode::EXTERNAL)
  {
    return run_external_decode(prompt_text, tokens, keep_history, stats_out);
  }
  return run_session_decode(prompt_text, tokens, keep_history, allow_interrupt, stats_out);
}

/*=============================================================================
   Benchmark
=============================================================================*/

static void build_random_tokens(uint64_t n, std::vector<int32_t>* tokens)
{
  tokens->clear();
  tokens->reserve((size_t)n);
  for (uint64_t i = 0; i < n; ++i)
  {
    tokens->push_back(rand() % g_state.embed_info.vocab_size);
  }
}

int run_bench(uint64_t n_in, uint64_t n_out)
{
  if (n_in == 0) n_in = 1;
  if (n_out == 0) n_out = 1;

  /* n_in must leave room for at least one decode token: without this a huge
     n_in (e.g. --bench 999999999999) would both overflow the sum below and
     make build_random_tokens allocate a massive vector (OOM). */
  if (n_in >= g_state.llm_config.max_ctx_len)
  {
    LOGW("n_in (%llu) >= max context len (%u), capping to %u\n",
         (unsigned long long)n_in, g_state.llm_config.max_ctx_len,
         g_state.llm_config.max_ctx_len - 1);
    n_in = g_state.llm_config.max_ctx_len - 1;
  }

  /* safe sum: n_in < max_ctx_len is already guaranteed, so n_in + n_out can
     only overflow if n_out is huge; compare against the remaining budget. */
  if (n_out > g_state.llm_config.max_ctx_len - n_in)
  {
    LOGW("n_in + n_out exceeds max context len (%u), capping n_out to %llu\n",
         g_state.llm_config.max_ctx_len,
         (unsigned long long)(g_state.llm_config.max_ctx_len - n_in));
    n_out = g_state.llm_config.max_ctx_len - n_in;
  }
  if (n_out == 0) n_out = 1;

  LOGI("benchmark: %llu input tokens -> %llu output tokens\n",
       (unsigned long long)n_in, (unsigned long long)n_out);

  /* clean slate for a fair measurement */
  rknn3_session_clear_kvcache(g_state.session, RKNN3_KVCACHE_CLEAR_ALL, nullptr);

  std::vector<int32_t> tokens;
  srand(42);
  build_random_tokens(n_in, &tokens);

  /* output is random, disable printing to keep the benchmark clean */
  int32_t saved_max = g_state.max_new_tokens;
  bool saved_ie = g_state.ignore_eos;
  bool saved_thinking = g_state.enable_thinking;
  g_state.max_new_tokens = (int32_t)n_out;
  g_state.ignore_eos = true;
  g_state.enable_thinking = false;
  apply_llm_param();

  RunStats s;
  int ret = run_turn(nullptr, &tokens, false, false, &s);
  if (ret == 0) print_perf(s);

  /* always restore the caller's run parameters, even on failure, so a failed
     benchmark cannot leak its temporary settings into later turns */
  g_state.max_new_tokens = saved_max;
  g_state.ignore_eos = saved_ie;
  g_state.enable_thinking = saved_thinking;
  apply_llm_param();
  return ret;
}

/*=============================================================================
   Status printing
=============================================================================*/

void print_perf(const RunStats& s)
{
  if (!s.valid)
  {
    LOGW("No run recorded yet.\n");
    return;
  }
  printf("\n-----------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-16s | %-8s | %-20s | %-20s\n", "Stage", "Total Time (ms)", "Tokens",
         "Time/Token (ms)", "Tokens per Second");
  printf("-----------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-16.2f | %-8llu | %-20.3f | %-20.2f\n", "Prefill",
         s.prefill_ms, (unsigned long long)s.n_prefill_tokens, s.prefill_tpt_ms, s.prefill_tps);
  printf(" %-10s | %-16.2f | %-8llu | %-20.3f | %-20.2f\n", "Generate",
         s.decode_ms, (unsigned long long)s.n_decode_tokens, s.decode_tpt_ms, s.decode_tps);
  printf("-----------------------------------------------------------------------------------------\n");
  printf(" TTFT: %.2f ms   |   Total: %.2f ms\n", s.ttft_us / 1e3, s.prefill_ms + s.decode_ms);
  if (s.stopped_by_user) LOGW(" (stopped by user)\n");
  printf("-----------------------------------------------------------------------------------------\n");
}

void print_model_info()
{
  const rknn3_llm_config& c = g_state.llm_config;
  printf("\n");
  printf("=============================================================\n");
  printf("%-32s: %-8u\n", "Max Context Length", c.max_ctx_len);
  printf("%-32s: %-8u\n", "Max Position Embeddings", c.max_position_embeddings);
  printf("%-32s: %s\n", "Model Type", c.model_type ? c.model_type : "unknown");
  printf("%-32s: %s\n", "Task Type",
         c.task_type == RKNN3_LLM_TASK_GENERATE ? "RKNN3_LLM_TASK_GENERATE" : "RKNN3_LLM_TASK_EMBEDDING");
  printf("%-32s: %-8d\n", "Vocab Size", g_state.vocab_info.vocab_size);
  printf("%-32s: %-8d\n", "Embedding Dim", g_state.embed_info.embedding_dim);
  printf("%-32s: %-8u\n", "Max New Tokens", (uint32_t)g_state.max_new_tokens);
  printf("=============================================================\n");
}

void print_status()
{
  print_model_info();

  rknn3_sampling_params sp = g_state.params.sampling_param;
  printf("-- Sampling -------------------------------------------------\n");
  printf("  temperature      : %.3f\n", sp.temperature);
  printf("  top_k            : %d\n", sp.top_k);
  printf("  top_p            : %.3f\n", sp.top_p);
  printf("  repeat_penalty   : %.3f\n", sp.repeat_penalty);
  printf("  frequency_penalty: %.3f\n", sp.frequency_penalty);
  printf("  presence_penalty : %.3f\n", sp.presence_penalty);
  printf("-- Session --------------------------------------------------\n");
  printf("  max_new_tokens   : %d\n", g_state.max_new_tokens);
  printf("  keep_history     : %s\n", g_state.keep_history ? "on" : "off");
  printf("  enable_thinking  : %s\n", g_state.enable_thinking ? "on" : "off");
  printf("  ignore_eos       : %s\n", g_state.ignore_eos ? "on" : "off");
  printf("  verbose          : %s\n", g_state.verbose ? "on" : "off");
  if (g_state.lora.enabled)
    printf("  lora             : %s (enabled)\n", g_state.lora.lora.lora_name);
  else if (g_state.lora.loaded)
    printf("  lora             : %s (loaded)\n", g_state.lora.lora.lora_name);
  else
    printf("  lora             : none\n");

  bool custom_template = !g_state.system_prompt.empty() || !g_state.prompt_prefix.empty() ||
                         !g_state.prompt_postfix.empty();
  const char* ct = g_state.llm_config.chat_template;
  printf("  chat template    : %s\n", custom_template ? "custom override" : "model built-in");
  if (!custom_template && ct && ct[0])
  {
    std::string s(ct);
    printf("                    %s%s\n", s.substr(0, 60).c_str(), s.size() > 60 ? "..." : "");
  }
  printf("  system_prompt    : \"%s\"\n", g_state.system_prompt.c_str());
  printf("  prompt_prefix    : \"%s\"\n", g_state.prompt_prefix.c_str());
  printf("  prompt_postfix   : \"%s\"\n", g_state.prompt_postfix.c_str());

  RKLLMRunState st;
  memset(&st, 0, sizeof(st));

  if (rknn3_session_query_state(g_state.session, &st) == RKNN3_SUCCESS)
  {
    printf("-- Context --------------------------------------------------\n");
    printf("  n_total_tokens   : %llu\n", (unsigned long long)st.n_total_tokens);
    printf("  n_max_tokens     : %llu\n", (unsigned long long)st.n_max_tokens);
    printf("  n_reuse_tokens   : %llu\n", (unsigned long long)st.n_reuse_tokens);
    printf("  context usage    : %.1f%%\n",
           st.n_max_tokens > 0 ? 100.0 * (double)st.n_total_tokens / (double)st.n_max_tokens : 0.0);
  }

  if (g_state.checkpoint_enabled)
  {
    const auto& cp = g_state.kvcache_policy.save_checkpoint;
    printf("-- kvcache checkpoint ----------------------------------------\n");
    printf("  checkpoint start pos : %lld\n", (long long)cp.checkpoint_start_pos);
    printf("  checkpoint interval  : %lld\n", (long long)cp.checkpoint_interval);
    printf("  max checkpoint count : %lld\n", (long long)cp.max_checkpoint_count);
    printf("  tail overwrite       : %s\n", cp.checkpoint_tail_overwrite ? "on" : "off");
  }
  printf("=============================================================\n");
}
