// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Common helpers shared by the rknn3_session test demos. Implementation
// file for rknn3_session_test_common.h; see the header for declarations.

#include "rknn3_session_test_common.h"

/* ------------------------------------------------------------------ *
 * Timing
 *
 * Demos historically keep these as file-scope globals; each demo is a
 * single TU so giving them internal linkage here preserves that.
 * ------------------------------------------------------------------ */
void rknn3_test_timing_mark_start(rknn3_test_timing_t* t)
{
  gettimeofday(&t->start, NULL);
  t->first_decode = true;
}

void rknn3_test_timing_mark_end(rknn3_test_timing_t* t) { gettimeofday(&t->end, NULL); }

/* The result callback stamps first_token on the first decoded token. */
void rknn3_test_timing_mark_first_token(rknn3_test_timing_t* t)
{
  if (t->first_decode) {
    gettimeofday(&t->first_token, NULL);
    t->first_decode = false;
  }
}

/* ------------------------------------------------------------------ *
 * Embedding table (mmap'd, read-only)
 * ------------------------------------------------------------------ */
void embedding_info_release(struct embedding_info* info, struct stat* emb_st)
{
  if (info->embedding_data && info->embedding_data != MAP_FAILED) {
    if (emb_st) {
      munmap(info->embedding_data, emb_st->st_size);
    }
    info->embedding_data = nullptr;
  }
  if (info->fd != -1) {
    close(info->fd);
    info->fd = -1;
  }
}

/* ------------------------------------------------------------------ *
 * Math helpers
 * ------------------------------------------------------------------ */
int argmax_fp16(const float16* data, int size)
{
  if (size <= 0 || !data) {
    return -1;
  }
  int max_id = 0;
  for (int i = 1; i < size; i++) {
    if (fp16_to_fp32(data[i]) > fp16_to_fp32(data[max_id])) {
      max_id = i;
    }
  }
  return max_id;
}

/* ------------------------------------------------------------------ *
 * Standard LLM callbacks
 *
 * result/tokenizer/embed/sampling are wired up by every demo; output is
 * only used when a demo opts in-> They all read their userdata from the
 * same pair of objects (Tokenizer* + embedding_info*).
 * ------------------------------------------------------------------ */
int default_result_callback(void* userdata, RKLLMResult* result, LLMCallState state)
{
  rknn3_test_session_userdata_t* session_userdata = (rknn3_test_session_userdata_t*)userdata;
  Tokenizer*                     tokenizer        = session_userdata->tokenizer;

  switch (state) {
  case RKLLM_RUN_NORMAL: {
    std::string piece;
    if (result->num_tokens == 1) {
      piece = tokenizer->TokenToPiece(result->token_ids[0]);
    } else {
      piece = tokenizer->Decode(result->token_ids, result->num_tokens);
    }
    printf("%s", piece.c_str());
    rknn3_test_timing_mark_first_token(&session_userdata->timing);
    fflush(stdout);
    break;
  }
  case RKLLM_RUN_WAITING:
    printf("\n\nWaiting for UTF-8 encoded character\n");
    fflush(stdout);
    break;
  case RKLLM_RUN_FINISH:
    printf("\n\n--------------------Finished-------------------- \n");
    fflush(stdout);
    break;
  case RKLLM_RUN_STOP:
    printf("\n\n-----------------------Stop--------------------- \n");
    fflush(stdout);
    break;
  case RKLLM_RUN_MAX_NEW_TOKEN_REACHED:
    printf("\n\n--------------Max new token reached------------- \n");
    fflush(stdout);
    break;
  case RKLLM_RUN_PAUSE:
    printf("\n\n-----------------------Pause-------------------- \n");
    fflush(stdout);
    break;
  case RKLLM_RUN_RESUME:
    printf("\n\n----------------------Resume-------------------- \n");
    fflush(stdout);
    break;
  case RKLLM_RUN_ERROR:
    printf("\n\nError occurred during inference\n");
    fflush(stdout);
    break;
  default:
    printf("\n\nUnknown LLM call state: %d\n", (int)state);
    fflush(stdout);
    break;
  }
  return 0;
}

int default_tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max)
{
  Tokenizer* tokenizer = (Tokenizer*)userdata;
  int        n_tokens  = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);
  if (n_tokens <= 0) {
    printf("tokenizer failed for %s\n", text);
  }
  return n_tokens;
}

int default_embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{
  struct embedding_info* info = (struct embedding_info*)userdata;
  if (len != num_tokens * info->embedding_dim * sizeof(float16)) {
    printf("invalid embed buffer\n");
    return -1;
  }
  for (uint64_t n = 0; n < num_tokens; n++) {
    memcpy((unsigned char*)embed + n * info->embedding_dim * sizeof(float16), info->embedding_data + tokens[n] * info->embedding_dim,
           info->embedding_dim * sizeof(float16));
  }
  return 0;
}

int default_sampling_callback(void* userdata, float16* logits, char* logits_name)
{
  (void)logits_name;
  struct embedding_info* info = (struct embedding_info*)userdata;
  return argmax_fp16(logits, info->vocab_size);
}

/* Prints the first 10 fp16 elements of each output tensor. */
int default_output_callback(void* userdata, rknn3_tensor* output_tensors, uint32_t n_output_tensors, LLMOutputCallbackState state)
{
  (void)userdata;
  printf("\noutput_callback: state = %d\n", state);
  for (uint32_t i = 0; i < n_output_tensors; i++) {
    printf("output_callback: output[%d]->attr->index = %d\n", i, output_tensors[i].attr->index);
    printf("output_callback: output[%d]->attr->name = %s\n", i, output_tensors[i].attr->name);
    printf("output_callback: output[%d]->mem->size = %zu\n", i, output_tensors[i].mem->size);
    for (int j = 0; j < 10; j++) {
      printf("output_callback: output[%d]->mem->virt_addr[%d] = %f\n", i, j, fp16_to_fp32(((float16*)output_tensors[i].mem->virt_addr)[j]));
    }
  }
  return 0;
}

/* ------------------------------------------------------------------ *
 * Context + model init
 *
 * Returns RKNN3_SUCCESS / error code via ret, ctx via out-param. The old
 * per-demo helpers returned rknn3_context with -1 on failure, which (since
 * rknn3_context is unsigned) made the `if (ctx < 0)` checks in callers
 * always false; routing through an int return fixes that without changing
 * observable behaviour on the happy path.
 * ------------------------------------------------------------------ */
int init_context_and_model(rknn3_context* ctx_out, const char* model_path, const char* weight_path, uint32_t core_mask,
                           const char* key_path, const char* device_id)
{
  int           ret = 0;
  rknn3_config  config;
  rknn3_context ctx = 0;

  rknn3_init_extend init_extend;
  memset(&init_extend, 0, sizeof(init_extend));
  init_extend.device_id = (char*)device_id;
  ret                   = rknn3_init(&ctx, &init_extend);
  if (ret < 0) {
    printf("rknn3_init fail! ret=%d\n", ret);
    return ret;
  }

  if (key_path != nullptr && strlen(key_path) > 0) {
    printf("Setting decrypt key from: %s\n", key_path);
    ret = rknn3_set_decrypt_key_from_path(ctx, key_path);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_set_decrypt_key_from_path failed! ret=%d\n", ret);
      rknn3_destroy(ctx);
      return ret;
    }
    printf("rknn3_set_decrypt_key_from_path success\n");
  }

  ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_load_model_from_path failed! ret=%d\n", ret);
    rknn3_destroy(ctx);
    return ret;
  }

  memset(&config, 0, sizeof(config));
  config.run_core_mask = core_mask;

  ret = rknn3_model_init(ctx, &config);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_model_init failed! ret=%d\n", ret);
    rknn3_destroy(ctx);
    return ret;
  }

  *ctx_out = ctx;
  return RKNN3_SUCCESS;
}

/* ------------------------------------------------------------------ *
 * Tokenizer + embedding bootstrap
 * ------------------------------------------------------------------ */

/* Loads just the tokenizer (no embedding file). Sets vocab_info. */
int get_tokenizer(const char* tokenizer_path, VocabInfo* vocab_info, Tokenizer** tokenizer)
{
  *tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
  (*tokenizer)->GetVocabInfo(vocab_info);
  return 0;
}

/* Loads just the embedding table (mmap). Caller must have already loaded
 * the tokenizer to know vocab_size. */
int get_embedding(struct embedding_info* embedding_info, const char* embedding_path, struct stat* emb_st, int vocab_size)
{
  memset(embedding_info, 0x00, sizeof(struct embedding_info));
  embedding_info->fd = open(embedding_path, O_RDONLY);
  if (embedding_info->fd == -1) {
    printf("Failed to open embedding file: %s\n", embedding_path);
    return -1;
  }
  if (fstat(embedding_info->fd, emb_st) == -1) {
    printf("Failed to get file size\n");
    close(embedding_info->fd);
    embedding_info->fd = -1;
    return -1;
  }
  embedding_info->embedding_data = (float16*)mmap(NULL, emb_st->st_size, PROT_READ, MAP_PRIVATE, embedding_info->fd, 0);
  if (embedding_info->embedding_data == MAP_FAILED) {
    printf("Failed to mmap file\n");
    close(embedding_info->fd);
    embedding_info->fd = -1;
    return -1;
  }
  embedding_info->vocab_size    = vocab_size;
  embedding_info->embedding_dim = (emb_st->st_size / vocab_size) / sizeof(float16);
  return 0;
}

/* Convenience wrapper: loads tokenizer then embedding. */
int get_tokenizer_and_embedding(const char* tokenizer_path, VocabInfo* vocab_info, Tokenizer** tokenizer,
                                struct embedding_info* embedding_info, const char* embedding_path, struct stat* emb_st)
{
  int ret = get_tokenizer(tokenizer_path, vocab_info, tokenizer);
  if (ret != 0) {
    return ret;
  }
  ret = get_embedding(embedding_info, embedding_path, emb_st, vocab_info->vocab_size);
  if (ret != 0) {
    delete *tokenizer;
    *tokenizer = nullptr;
  }
  return ret;
}

/* ------------------------------------------------------------------ *
 * Default param / callback assembly
 * ------------------------------------------------------------------ */
void build_default_llm_params(rknn3_llm_param* params, const VocabInfo* vocab_info, int32_t max_context_len, bool ignore_eos_token)
{
  memset(params, 0, sizeof(*params));
  params->logits_name                      = (char*)"output";
  params->max_context_len                  = max_context_len;
  params->sampling_param.temperature       = 1.0f;
  params->sampling_param.top_k             = 1; // topk=1 sampling
  params->sampling_param.top_p             = 0.9f;
  params->sampling_param.repeat_penalty    = 1.1f; // repetition penalty
  params->sampling_param.frequency_penalty = 0.0f;
  params->sampling_param.presence_penalty  = 0.0f;
  params->vocab_info.vocab_size            = vocab_info->vocab_size;
  params->vocab_info.n_special_eos_id      = vocab_info->n_special_eos_id;
  params->vocab_info.n_special_bos_id      = vocab_info->n_special_bos_id;
  memcpy(params->vocab_info.special_eos_id, vocab_info->special_eos_id, sizeof(vocab_info->special_eos_id));
  memcpy(params->vocab_info.special_bos_id, vocab_info->special_bos_id, sizeof(vocab_info->special_bos_id));
  params->vocab_info.linefeed_id      = vocab_info->linefeed_id;
  params->vocab_info.ignore_eos_token = ignore_eos_token;
}

/* Wires result/embed/tokenizer to the standard callbacks. Demos that need
 * sampling/output/input callbacks set those themselves after this call. */
void build_default_callback(RKLLMCallback* callback, Tokenizer* tokenizer, struct embedding_info* emb,
                            rknn3_test_session_userdata_t* session_userdata)
{
  memset(callback, 0, sizeof(*callback));
  session_userdata->tokenizer = tokenizer;
  memset(&session_userdata->timing, 0, sizeof(session_userdata->timing));
  session_userdata->timing.first_decode = true;
  callback->result_callback             = default_result_callback;
  callback->result_userdata             = session_userdata;
  callback->embed_callback              = default_embed_callback;
  callback->embed_userdata              = emb;
  callback->tokenizer_callback          = default_tokenizer_callback;
  callback->tokenizer_userdata          = tokenizer;
}

/* ------------------------------------------------------------------ *
 * LLM input construction
 *
 * Demos historically hand-rolled an rknn3_llm_input + rknn3_llm_infer_param
 * for each run, with the same keep_history/max_new_tokens/enable_thinking
 * defaults. These helpers build the three single-tensor input flavours
 * (PROMPT / TOKEN / EMBED) and a default infer param, so a demo's run loop
 * shrinks to a couple of lines.
 *
 * build_llm_input_* return the input by value (cheap POD copy). For EMBED
 * the caller owns the malloc'd embed buffer and must free it (the helper
 * returns it via *embed so the caller can release it after the run).
 * ------------------------------------------------------------------ */

/* Builds the default infer param. keep_history defaults to 1; prefill_only
 * and disable_sampling default to false (normal generate). Pass prefill_only=
 * true (typically with disable_sampling=true) for embedding/reranker runs. */
rknn3_llm_infer_param build_default_infer_param(int32_t max_new_tokens, int keep_history, bool prefill_only, bool disable_sampling)
{
  rknn3_llm_infer_param p;
  memset(&p, 0, sizeof(p));
  p.keep_history     = keep_history;
  p.max_new_tokens   = max_new_tokens;
  p.prefill_only     = prefill_only;
  p.disable_sampling = disable_sampling;
  return p;
}

/* PROMPT input: text prompt, no tokens/embed. */
rknn3_llm_input* build_llm_input_prompt(const char* prompt, bool enable_thinking, bool print_input)
{
  if (print_input && prompt) {
    printf("%s\n", prompt);
  }
  rknn3_llm_input* in = new rknn3_llm_input;
  memset(in, 0, sizeof(*in));
  in->input_type                = RKNN3_LLM_INPUT_PROMPT;
  in->llm_input.name            = NULL;
  in->llm_input.prompt          = prompt;
  in->llm_input.embed           = NULL;
  in->llm_input.tokens          = NULL;
  in->llm_input.n_tokens        = 0;
  in->llm_input.enable_thinking = enable_thinking;
  return in;
}

/* TOKEN input: caller-supplied token id array (must outlive the run). */
rknn3_llm_input* build_llm_input_tokens(const int32_t* tokens, uint64_t n_tokens, bool enable_thinking, bool print_input)
{
  if (print_input) {
    printf("token input: n_tokens=%llu, tokens=[", (unsigned long long)n_tokens);
    for (uint64_t i = 0; i < n_tokens; i++) {
      printf("%d%s", tokens[i], (i + 1 < n_tokens) ? ", " : "");
    }
    printf("]\n");
  }
  rknn3_llm_input* in = new rknn3_llm_input;
  memset(in, 0, sizeof(*in));
  in->input_type                = RKNN3_LLM_INPUT_TOKEN;
  in->llm_input.name            = NULL;
  in->llm_input.prompt          = NULL;
  in->llm_input.embed           = NULL;
  in->llm_input.tokens          = (int32_t*)tokens;
  in->llm_input.n_tokens        = n_tokens;
  in->llm_input.enable_thinking = enable_thinking;
  return in;
}

/* EMBED input: looks up embeddings for `tokens` from the mmap'd embedding
 * Wraps a caller-provided embed buffer into an rknn3_llm_input. The caller
 * is responsible for filling the buffer (e.g. via default_embed_callback)
 * and freeing it after the run. */
rknn3_llm_input* build_llm_input_embed(float16* embed, uint64_t n_tokens, bool enable_thinking, bool print_input)
{
  if (print_input) {
    printf("embed input: n_tokens=%llu\n", (unsigned long long)n_tokens);
  }
  rknn3_llm_input* in = new rknn3_llm_input;
  memset(in, 0, sizeof(*in));

  in->input_type                = RKNN3_LLM_INPUT_EMBED;
  in->llm_input.name            = NULL;
  in->llm_input.prompt          = NULL;
  in->llm_input.embed           = embed;
  in->llm_input.tokens          = NULL;
  in->llm_input.n_tokens        = n_tokens;
  in->llm_input.enable_thinking = enable_thinking;
  return in;
}

/* MULTIMODAL image input: caller supplies a pre-allocated image_embed
 * buffer (size = n_image * n_image_tokens * embedding_dim * sizeof(float16))
 * plus the per-image geometry and prompt. The Qwen-VL image tags default to
 * the standard <|vision_start|>/<|vision_end|>/<|image_pad|> set; pass your
 * own if the model needs different tags. The returned input borrows
 * image_embed (caller frees it). */
rknn3_llm_input* build_llm_input_multimodal_image(const char* prompt, const int32_t* tokens, uint64_t n_tokens, bool enable_thinking,
                                                  float16* image_embed, uint64_t n_image_tokens, uint64_t n_image, const char* image_start,
                                                  const char* image_end, const char* image_content, uint64_t image_width,
                                                  uint64_t image_height, bool print_input)
{
  if (print_input) {
    if (tokens) {
      printf("tokens: [");
      for (uint64_t i = 0; i < n_tokens; i++) {
        printf("%d, ", tokens[i]);
      }
      printf("]\n");
    }
    if (prompt) {
      printf("%s\n", prompt);
    }
  }
  rknn3_llm_input* in = new rknn3_llm_input;
  memset(in, 0, sizeof(*in));
  in->input_type = RKNN3_LLM_INPUT_MULTIMODAL;

  rknn3_llm_multimodal_tensor& m = in->multimodal_input;
  m.name                         = NULL;
  m.prompt                       = prompt;
  m.tokens                       = (int32_t*)tokens;
  m.n_tokens                     = n_tokens;
  m.enable_thinking              = enable_thinking;

  m.image.image_embed    = image_embed;
  m.image.n_image_tokens = n_image_tokens;
  m.image.n_image        = n_image;
  m.image.image_start    = image_start;
  m.image.image_end      = image_end;
  m.image.image_content  = image_content;
  m.image.image_width    = image_width;
  m.image.image_height   = image_height;
  return in;
}

/* AUX input (deepstack-style): allocates a cacheable device mem of `size`
 * bytes on `core_id`, copies `data` into it, and wraps it as an AUX input.
 * The returned input owns the mem; release it via release_llm_input() (the
 * AUX case frees the mem). Returns a zeroed input on failure. */
rknn3_llm_input* build_llm_input_aux(rknn3_context ctx, int32_t core_id, uint64_t size, const void* data, bool print_input)
{
  rknn3_llm_input* in = new rknn3_llm_input;
  memset(in, 0, sizeof(*in));

  rknn3_tensor_mem* mem = rknn3_create_mem(ctx, size, core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
  if (!mem) {
    printf("fail to create aux mem! size=%llu\n", (unsigned long long)size);
    delete in;
    return nullptr;
  }
  if (data) {
    memcpy(mem->virt_addr, data, size);
  }

  in->input_type     = RKNN3_LLM_INPUT_AUX;
  in->aux_input.mem  = mem;
  in->aux_input.attr = nullptr;
  return in;
}

/* Releases any heap memory owned by an input built by the build_llm_input_*
 * helpers. AUX inputs own device mem. EMBED/PROMPT/TOKEN inputs borrow
 * caller-owned memory and are a no-op. Safe to call on a zero-initialised
 * input. */
void release_llm_input(rknn3_context ctx, rknn3_llm_input*& in)
{
  if (!in) {
    return;
  }
  if (in->input_type == RKNN3_LLM_INPUT_AUX && in->aux_input.mem) {
    /* AUX mem was allocated via rknn3_create_mem in build_llm_input_aux. */
    rknn3_destroy_mem(ctx, in->aux_input.mem);
    in->aux_input.mem = nullptr;
  }
  delete in;
  in = nullptr;
}

/* ------------------------------------------------------------------ *
 * Pretty-printers
 * ------------------------------------------------------------------ */
void dump_tensor_attr(const rknn3_tensor_attr* attr)
{
  std::string shape_str;
  for (uint32_t j = 0; j < attr->n_dims; j++) {
    shape_str += std::to_string(attr->shape[j]);
    if (j < attr->n_dims - 1) {
      shape_str += ", ";
    }
  }
  std::string stride_str;
  for (uint32_t j = 0; j < attr->n_stride; j++) {
    stride_str += std::to_string(attr->stride[j]);
    if (j < attr->n_stride - 1) {
      stride_str += ", ";
    }
  }
  printf("  name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, qnt_type=%s, scale=%.5f, zero_point=%d\n",
         attr->name, attr->n_dims, shape_str.c_str(), stride_str.c_str(), attr->aligned_size, rknn3_get_layout_string(attr->layout),
         rknn3_get_type_string(attr->dtype), rknn3_get_qnt_type_string(attr->qnt_type), attr->qnt_info.scale, attr->qnt_info.zero_point);
}

/* Prints every field of rknn3_llm_config. Enum fields without a dedicated
 * string helper are printed as their numeric value. */
const char* rknn3_test_task_type_string(rknn3_llm_task_type t)
{
  switch (t) {
  case RKNN3_LLM_TASK_GENERATE:
    return "RKNN3_LLM_TASK_GENERATE";
  case RKNN3_LLM_TASK_EMBEDDING:
    return "RKNN3_LLM_TASK_EMBEDDING";
  case RKNN3_LLM_TASK_RERANKER:
    return "RKNN3_LLM_TASK_RERANKER";
  default:
    return "UNKNOWN";
  }
}

const char* rknn3_test_attention_type_string(rknn3_attention_type t)
{
  switch (t) {
  case RKNN3_ATTENTION_TYPE_FULL_ATTENTION:
    return "FULL_ATTENTION";
  case RKNN3_ATTENTION_TYPE_SLIDING_ATTENTION:
    return "SLIDING_ATTENTION";
  case RKNN3_ATTENTION_TYPE_LINEAR_ATTENTION:
    return "LINEAR_ATTENTION";
  default:
    return "UNKNOWN";
  }
}

void print_llm_config(const rknn3_llm_config* cfg)
{
  printf("=============================================================\n");
  // printf("%-32s: %s\n",                "Chat Template",          cfg->chat_template ? cfg->chat_template : "(null)");
  printf("%-32s: %u\n", "Vocab Size", cfg->vocab_size);
  printf("%-32s: %u\n", "Embedding Dim", cfg->embedding_dim);
  printf("%-32s: %u\n", "Max Context Length", cfg->max_ctx_len);
  printf("%-32s: %u\n", "Max Position Embeddings", cfg->max_position_embeddings);
  printf("%-32s: %d\n", "Kvcache Store Method", (int)cfg->kvcache_store_method);
  printf("%-32s: %d\n", "Kvcache Dtype", (int)cfg->kvcache_dtype);
  printf("%-32s: %u\n", "Kvcache Group Size", cfg->kvcache_group_size);
  printf("%-32s: %u\n", "Kvcache Residual Depth", cfg->kvcache_residual_depth);
  printf("%-32s: %s\n", "Model Type", cfg->model_type ? cfg->model_type : "(null)");
  printf("%-32s: %s\n", "Task Type", rknn3_test_task_type_string(cfg->task_type));
  printf("%-32s: %u\n", "Rope Cache Host Storage", cfg->rope_cache_host_storage);
  printf("%-32s: %u\n", "N Attention Kvcache Lens", cfg->n_attention_kvcache_lens);
  for (uint32_t i = 0; i < cfg->n_attention_kvcache_lens && i < RKNN3_MAX_ATTENTION_TYPE_NUM; i++) {
    const rknn3_attention_kvcache_lens* a = &cfg->attention_kvcache_lens[i];
    printf("  [%u] %-26s: n_kvcache_buffer_lens=%u, lens=[", i, rknn3_test_attention_type_string(a->attention_type),
           a->n_kvcache_buffer_lens);
    for (uint32_t j = 0; j < a->n_kvcache_buffer_lens; j++) {
      printf("%d%s", a->kvcache_buffer_lens[j], (j + 1 < a->n_kvcache_buffer_lens) ? ", " : "");
    }
    printf("]\n");
  }
  printf("=============================================================\n\n");
}

const char* rknn3_test_kvcache_policy_string(rknn3_kvcache_policy p)
{
  switch (p) {
  case RKNN3_KVCACHE_POLICY_DEFAULT:
    return "RKNN3_KVCACHE_POLICY_DEFAULT";
  case RKNN3_KVCACHE_POLICY_RECURRENT:
    return "RKNN3_KVCACHE_POLICY_RECURRENT";
  case RKNN3_KVCACHE_POLICY_NORMAL:
    return "RKNN3_KVCACHE_POLICY_NORMAL";
  case RKNN3_KVCACHE_POLICY_SAVE_CHECKPOINT:
    return "RKNN3_KVCACHE_POLICY_SAVE_CHECKPOINT";
  default:
    return "UNKNOWN";
  }
}

/* Dumps every field of an RKLLMRunState (the struct returned by
 * rknn3_session_query_state). Uses %llu for the uint64_t token counts (the
 * demos used to print them with %lu/%d, which truncated on 32-bit). Fields are
 * printed in an aligned %-32s column to match print_llm_config. */
void print_run_state(const RKLLMRunState* state, bool print_tokens)
{
  printf("\nRun State:\n");
  printf("-------------------------------------------------------------\n");
  printf("%-32s: %llu\n", "N Total Tokens", (unsigned long long)state->n_total_tokens);
  printf("%-32s: %llu\n", "N Reuse Tokens", (unsigned long long)state->n_reuse_tokens);
  printf("%-32s: %llu\n", "N Max Tokens", (unsigned long long)state->n_max_tokens);
  printf("%-32s: %llu\n", "N Prefill Tokens", (unsigned long long)state->n_prefill_tokens);
  printf("%-32s: %llu\n", "N Decode Tokens", (unsigned long long)state->n_decode_tokens);
  printf("%-32s: %llu\n", "N Input Tokens", (unsigned long long)state->n_input_tokens);
  printf("%-32s: %llu\n", "N Output Tokens", (unsigned long long)state->n_output_tokens);
  printf("%-32s: %s\n", "Kvcache Policy", rknn3_test_kvcache_policy_string(state->kvcache_policy));
  printf("%-32s: %d\n", "N Loras Enabled", (int)state->n_loras_enabled);
  for (int i = 0; i < state->n_loras_enabled; i++) {
    printf("  [%d] %-26s: scale=%.6f\n", i, state->loras_enabled[i].lora_name, state->loras_enabled[i].scale);
  }

  if (print_tokens) {
    if (state->input_tokens) {
      printf("%-32s: ", "Input Tokens");
      for (uint64_t i = 0; i < state->n_input_tokens; i++) {
        printf("%d ", state->input_tokens[i]);
      }
      printf("\n");
    }
    if (state->output_tokens) {
      printf("%-32s: ", "Output Tokens");
      for (uint64_t i = 0; i < state->n_output_tokens; i++) {
        printf("%d ", state->output_tokens[i]);
      }
      printf("\n");
    }
  }
  printf("-------------------------------------------------------------\n");
  fflush(stdout);
}

/* Prints prefill/decode throughput. Uses the timing struct stamped by
 * the timing helpers + default_result_callback. */
void print_perf_stats(const RKLLMRunState* state, const rknn3_test_timing_t* t)
{
  int   prefill_n_tokens = (int)state->n_prefill_tokens;
  float prefill_us       = (t->first_token.tv_sec - t->start.tv_sec) * 1e6f + (t->first_token.tv_usec - t->start.tv_usec);
  float prefill_ms       = prefill_n_tokens == 0 ? 0.0f : prefill_us / 1e3f;
  float prefill_s        = prefill_us / 1e6f;
  float prefill_tpt      = prefill_n_tokens == 0 ? 0.0f : prefill_ms / prefill_n_tokens;
  float prefill_tps      = prefill_n_tokens == 0 ? 0.0f : prefill_n_tokens / prefill_s;

  int   decode_n_tokens = (int)state->n_decode_tokens;
  float decode_time_us  = ((t->end.tv_sec - t->first_token.tv_sec) * 1e6f) + (t->end.tv_usec - t->first_token.tv_usec);
  float decode_ms       = decode_n_tokens == 0 ? 0.0f : decode_time_us / 1e3f;
  float decode_s        = decode_time_us / 1e6f;
  float decode_tpt      = decode_n_tokens == 0 ? 0.0f : decode_ms / decode_n_tokens;
  float decode_tps      = decode_n_tokens == 0 ? 0.0f : decode_n_tokens / decode_s;

  printf("\nPerformance Statistics: ");
  printf("\n-----------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-16s | %-8s | %-20s | %-20s \n", "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
  printf("-----------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f \n", "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);
  printf(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f \n", "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps);
  printf("-----------------------------------------------------------------------------------------\n\n");
  fflush(stdout);
}

/* ------------------------------------------------------------------ *
 * Model input/output tensor attr query.
 *
 * Wraps the RKNN3_QUERY_IN_OUT_NUM + RKNN3_QUERY_{INPUT,OUTPUT}_ATTR dance
 * that multimodal / split-run demos need (e.g. to pick the AUX core_id from
 * input_attrs[i]). input_attrs/output_attrs may be NULL to skip that side;
 * max_n_* cap the loop to the caller's array capacity. io_num_out optionally
 * returns the queried counts.
 * ------------------------------------------------------------------ */
int query_input_output_attrs(rknn3_context ctx, rknn3_tensor_attr* input_attrs, int max_n_input, rknn3_tensor_attr* output_attrs,
                             int max_n_output, rknn3_input_output_num* io_num_out)
{
  rknn3_input_output_num io_num;
  int                    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    printf("rknn_query fail! ret=%d\n", ret);
    return ret;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
  if (io_num_out) {
    *io_num_out = io_num;
  }

  if (input_attrs) {
    printf("input tensors:\n");
    for (int i = 0; i < (int)io_num.n_input && i < max_n_input; i++) {
      input_attrs[i].index = i;
      ret                  = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn3_tensor_attr));
      if (ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
      }
      dump_tensor_attr(&input_attrs[i]);
    }
  }

  if (output_attrs) {
    printf("output tensors:\n");
    for (int i = 0; i < (int)io_num.n_output && i < max_n_output; i++) {
      output_attrs[i].index = i;
      ret                   = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn3_tensor_attr));
      if (ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
      }
      dump_tensor_attr(&output_attrs[i]);
    }
  }
  return RKNN3_SUCCESS;
}

/* ------------------------------------------------------------------ *
 * Output-tensor setup / teardown for the optional output_callback path.
 *
 * Every demo that opts into output_callback repeats the same block: query
 * IN_OUT_NUM, then for each requested output index query OUTPUT_ATTR and
 * allocate an rknn3_tensor_mem of aligned_size. The matching teardown frees
 * the attr and destroys the mem. io_num_out lets a caller (e.g. the
 * input_callback demo) reuse the queried input/output counts afterwards.
 * ------------------------------------------------------------------ */
int setup_output_tensors(rknn3_context ctx, rknn3_tensor* output_tensors, int n_output_tensors, const int* output_tensors_index,
                         rknn3_input_output_num* io_num_out)
{
  rknn3_input_output_num io_num;
  int                    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    printf("rknn_query fail! ret=%d\n", ret);
    return ret;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
  if (io_num_out) {
    *io_num_out = io_num;
  }

  for (int i = 0; i < n_output_tensors; i++) {
    output_tensors[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
    if (!output_tensors[i].attr) {
      printf("malloc output_tensors[%d].attr failed\n", i);
      return -1;
    }
    output_tensors[i].attr->index = output_tensors_index[i];
    ret                           = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, output_tensors[i].attr, sizeof(rknn3_tensor_attr));
    if (ret < 0) {
      printf("rknn_query fail! ret=%d\n", ret);
      return ret;
    }
    output_tensors[i].mem =
      rknn3_create_mem(ctx, output_tensors[i].attr->aligned_size, output_tensors[i].attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
    if (!output_tensors[i].mem) {
      printf("rknn3_create_mem failed for output tensor[%d]\n", i);
      free(output_tensors[i].attr);
      output_tensors[i].attr = nullptr;
      return -1;
    }
    printf("output_callback output tensor[%d]: %s\n", i, output_tensors[i].attr->name);
  }
  return RKNN3_SUCCESS;
}

/* Frees attr + mem allocated by setup_output_tensors. Safe to call even if
 * setup was never invoked or aborted partway (slots stay null). */
void cleanup_output_tensors(rknn3_context ctx, rknn3_tensor* output_tensors, int n_output_tensors)
{
  for (int i = 0; i < n_output_tensors; i++) {
    if (output_tensors[i].attr) {
      free(output_tensors[i].attr);
      output_tensors[i].attr = nullptr;
    }
    if (output_tensors[i].mem) {
      rknn3_destroy_mem(ctx, output_tensors[i].mem);
      output_tensors[i].mem = nullptr;
    }
  }
}

/* ------------------------------------------------------------------ *
 * Binary file I/O for multimodal / token fixtures
 * ------------------------------------------------------------------ */
uint64_t read_data_from_bin(const char* bin_path, float16* data)
{
  FILE* fp = fopen(bin_path, "rb");
  if (!fp) {
    printf("Fail to open file: %s\n", bin_path);
    return 0;
  }
  fseek(fp, 0, SEEK_END);
  long file_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  uint64_t n_elements = file_size / sizeof(float16);
  size_t   read_size  = fread(data, sizeof(float16), n_elements, fp);
  fclose(fp);

  if (read_size != n_elements) {
    printf("Fail to read file, only read %zu elements\n", read_size);
    return 0;
  }
  return n_elements;
}

uint64_t read_data_from_bin(const char* bin_path, float* data)
{
  FILE* fp = fopen(bin_path, "rb");
  if (!fp) {
    printf("Fail to open file: %s\n", bin_path);
    return 0;
  }
  fseek(fp, 0, SEEK_END);
  long file_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  uint64_t n_elements = file_size / sizeof(float);
  size_t   read_size  = fread(data, sizeof(float), n_elements, fp);
  fclose(fp);

  if (read_size != n_elements) {
    printf("Fail to read file, only read %zu elements\n", read_size);
    return 0;
  }
  return n_elements;
}

uint64_t save_data_to_bin(const char* bin_path, const float16* data, uint64_t n_elements)
{
  FILE* fp = fopen(bin_path, "wb");
  if (!fp) {
    printf("Fail to open file for writing: %s\n", bin_path);
    return 0;
  }
  size_t write_size = fwrite(data, sizeof(float16), n_elements, fp);
  fclose(fp);
  if (write_size != n_elements) {
    printf("Fail to write file, only wrote %zu elements\n", write_size);
    return 0;
  }
  printf("Successfully saved %llu elements to %s\n", (unsigned long long)n_elements, bin_path);
  return n_elements;
}

uint64_t read_tokens_from_bin(const char* bin_path, int32_t* tokens)
{
  FILE* fp = fopen(bin_path, "rb");
  if (!fp) {
    printf("Failed to open file: %s\n", bin_path);
    return 0;
  }
  fseek(fp, 0, SEEK_END);
  long file_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  uint64_t n_tokens  = file_size / sizeof(int32_t);
  size_t   read_size = fread(tokens, sizeof(int32_t), n_tokens, fp);
  if (read_size != n_tokens) {
    printf("Failed to read file, only read %zu elements\n", read_size);
    fclose(fp);
    return 0;
  }
  fclose(fp);
  return n_tokens;
}

/* ------------------------------------------------------------------ *
 * KV cache save / load helpers.
 *
 * save_kvcache_to_path: thin wrapper around rknn3_session_save_kvcache.
 *
 * load_kvcache_from_data: reads the file into memory, then calls
 *   rknn3_session_load_kvcache_from_data.
 *
 * load_kvcache_from_path: thin wrapper around
 *   rknn3_session_load_kvcache_from_path.
 * ------------------------------------------------------------------ */
int load_kvcache_from_data(rknn3_session* session, const char* path)
{
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    printf("fopen %s failed\n", path);
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  int64_t kvcache_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  uint8_t* kvcache_data = (uint8_t*)malloc(kvcache_size);
  if (!kvcache_data) {
    printf("malloc kvcache_data failed\n");
    fclose(fp);
    return -1;
  }

  size_t read_size = fread(kvcache_data, 1, kvcache_size, fp);
  if (read_size != (size_t)kvcache_size) {
    printf("fread %s failed, only read %zu bytes, expected %zu bytes\n", path, read_size, kvcache_size);
    free(kvcache_data);
    fclose(fp);
    return -1;
  }

  fclose(fp);

  int ret = rknn3_session_load_kvcache_from_data(session, kvcache_data, kvcache_size);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_load_kvcache_from_data failed, ret=%d\n", ret);
    free(kvcache_data);
    return ret;
  }

  free(kvcache_data);
  
  printf("\n------------------------------------------------\n");
  printf("Successfully loaded kvcache data from %s\n", path);
  printf("------------------------------------------------\n");
  return ret;
}

int load_kvcache_from_path(rknn3_session* session, const char* path)
{
  int ret = rknn3_session_load_kvcache_from_path(session, path);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_load_kvcache_from_path failed, ret=%d\n", ret);
    return ret;
  }
  printf("\n------------------------------------------------\n");
  printf("Successfully loaded kvcache from %s\n", path);
  printf("------------------------------------------------\n");
  return ret;
}

int save_kvcache_to_path(rknn3_session* session, const char* path)
{
  int ret = rknn3_session_save_kvcache(session, path);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_save_kvcache failed, ret=%d\n", ret);
    return ret;
  }
  printf("\n------------------------------------------------\n");
  printf("Successfully saved kvcache to %s\n", path);
  printf("------------------------------------------------\n");
  return ret;
}
