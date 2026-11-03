// rknn3_llm_cli : interactive command line chat client for RKNN3 LLM models.
//
// A llama.cpp / ollama style interactive CLI that runs a RKNN3-converted LLM
// model on an RK1820 / RK1828 device. Besides plain chat it supports a set of
// slash-commands to control the session at runtime (clear kvcache, adjust
// sampling parameters, switch chat template, benchmark, save/load kvcache...).
//
// This front-end shares the whole inference engine with rknn3_llm_step_decode
// (see engine.h / engine.cc); the only difference is the decode strategy
// (DecodeMode::SESSION - one session_run per turn).
//
// Usage:
//   ./rknn3_llm_cli --model <rknn_path> [options]
//
// Type '/help' inside the REPL for the full command list.

#include "args.h"
#include "engine.h"

#include <getopt.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <thread>
#include <vector>

/*=============================================================================
   Slash command helpers
=============================================================================*/

static bool parse_bool(const std::string& v, bool& out)
{
  if (iequals(v, "on") || iequals(v, "1") || iequals(v, "true") || iequals(v, "yes"))
  {
    out = true;
    return true;
  }
  if (iequals(v, "off") || iequals(v, "0") || iequals(v, "false") || iequals(v, "no"))
  {
    out = false;
    return true;
  }
  return false;
}

/* extract the text after the command (everything, preserving spaces) */
static std::string rest_of_line(const std::string& line, size_t cmd_len)
{
  size_t p = line.find_first_of(' ', cmd_len);
  if (p == std::string::npos) return "";
  return trim(line.substr(p + 1));
}

/*=============================================================================
   Status / help
=============================================================================*/

static void print_ctx()
{
  RKLLMRunState st;
  memset(&st, 0, sizeof(st));

  if (rknn3_session_query_state(g_state.session, &st) != RKNN3_SUCCESS)
  {
    LOGR("rknn3_session_query_state failed\n");
    return;
  }
  printf("context: %llu / %llu tokens (%.1f%%)\n",
         (unsigned long long)st.n_total_tokens, (unsigned long long)st.n_max_tokens,
         st.n_max_tokens > 0 ? 100.0 * (double)st.n_total_tokens / (double)st.n_max_tokens : 0.0);
}

static void print_stats()
{
  const RunStats& t = g_state.total;
  double ptps = t.prefill_ms > 0.0 ? t.n_prefill_tokens / (t.prefill_ms / 1e3) : 0.0;
  double dtps = t.decode_ms > 0.0 ? t.n_decode_tokens / (t.decode_ms / 1e3) : 0.0;
  printf("\n== Cumulative statistics ==\n");
  printf("  turns            : %llu\n", (unsigned long long)g_state.turn_count);
  printf("  prefill tokens   : %llu\n", (unsigned long long)t.n_prefill_tokens);
  printf("  decode tokens    : %llu\n", (unsigned long long)t.n_decode_tokens);
  printf("  prefill time     : %.2f ms (%.2f tokens/s)\n", t.prefill_ms, ptps);
  printf("  decode time      : %.2f ms (%.2f tokens/s)\n", t.decode_ms, dtps);
  printf("  total time       : %.2f ms\n", t.prefill_ms + t.decode_ms);
}

static void print_help()
{
  printf("\n");
  printf("== rknn3_llm_cli commands ==\n");
  printf("  /help, /?                    show this help\n");
  printf("  /exit, /quit, /bye           exit the program\n");
  printf("\n-- conversation --\n");
  printf("  /clear                       clear KV cache (keep system prompt), start a new conversation\n");
  printf("  /reset                       clear all KV cache (including system prompt)\n");
  printf("  /prompt <text>               send raw text without chat template (one-shot)\n");
  printf("  /stop, /cancel               interrupt the ongoing generation\n");
  printf("\n-- sampling / generation --\n");
  printf("  /temp <v>                    set temperature\n");
  printf("  /top-k <n>                   set top-k\n");
  printf("  /top-p <v>                   set top-p\n");
  printf("  /repeat-penalty <v>          set repeat penalty\n");
  printf("  /frequency-penalty <v>       set frequency penalty\n");
  printf("  /presence-penalty <v>        set presence penalty\n");
  printf("  /sampling <top_k> <top_p> <temp> <repeat> <freq> <presence>\n");
  printf("                               set all sampling parameters at once\n");
  printf("  /predict <n>, /max-tokens <n>  set max new tokens per turn\n");
  printf("  /thinking on|off             toggle thinking mode (re-initializes session)\n");
  printf("  /ignore-eos on|off           toggle ignoring EOS\n");
  printf("  /keep-history on|off         toggle history retention (resets conversation)\n");
  printf("\n-- prompt / template --\n");
  printf("  /system <prompt>             set system prompt (clears KV cache)\n");
  printf("  /template <prefix> <postfix> set chat template prefix / postfix\n");
  printf("\n-- info / performance --\n");
  printf("  /status, /ctx                show model / session / context info\n");
  printf("  /perf                        show last turn performance\n");
  printf("  /stats                       show cumulative statistics\n");
  printf("  /bench [n_in] [n_out]        run a benchmark with random tokens\n");
  printf("  /verbose on|off              toggle verbose logging\n");
  printf("\n-- kvcache persistence --\n");
  printf("  /save <path>                 save current KV cache to file\n");
  printf("  /load <path>                 load KV cache from file\n");
  printf("\n-- lora --\n");
  printf("  /lora <path>                 load a LoRA adapter onto the context\n");
  printf("  /lora-enable                 enable the loaded adapter on the session\n");
  printf("  /lora-disable                disable the adapter on the session\n");
  printf("  /lora-unload                 unload the adapter from the context\n");
  printf("  /lora-off, /disable-lora     disable + unload in one step\n");
  printf("\n");
}

/*=============================================================================
   Slash command dispatch
=============================================================================*/

static void record_turn(const RunStats& s)
{
  g_state.last_run = s;
  g_state.turn_count++;
  g_state.total.n_prefill_tokens += s.n_prefill_tokens;
  g_state.total.n_decode_tokens += s.n_decode_tokens;
  g_state.total.prefill_ms += s.prefill_ms;
  g_state.total.decode_ms += s.decode_ms;
}

static int cmd_clear(bool keep_system_prompt)
{
  int policy = keep_system_prompt ? RKNN3_KVCACHE_KEEP_SYSTEM_PROMPT : RKNN3_KVCACHE_CLEAR_ALL;
  int ret = rknn3_session_clear_kvcache(g_state.session, (rknn3_kvcache_clear_policy)policy, nullptr);
  if (ret != RKNN3_SUCCESS)
  {
    LOGR("rknn3_session_clear_kvcache failed, ret=%d\n", ret);
    return -1;
  }
  if (keep_system_prompt) LOGI("KV cache cleared (system prompt kept). New conversation started.\n");
  else                    LOGI("KV cache fully cleared. New conversation started.\n");
  return 0;
}

static int cmd_bench(const std::vector<std::string>& args)
{
  uint64_t n_in = 128;
  uint64_t n_out = 128;
  if (args.size() >= 2)
  {
    if (!parse_positive_u64(args[1].c_str(), &n_in))
    {
      LOGW("/bench n_in must be a positive integer\n");
      return -1;
    }
  }
  if (args.size() >= 3)
  {
    if (!parse_positive_u64(args[2].c_str(), &n_out))
    {
      LOGW("/bench n_out must be a positive integer\n");
      return -1;
    }
  }
  return run_bench(n_in, n_out);
}

static bool dispatch_command(const std::string& line)
{
  std::vector<std::string> args = split_args(line);
  if (args.empty()) return false;

  const std::string& cmd = args[0];

  if (cmd == "/help" || cmd == "/?")
  {
    print_help();
  }
  else if (cmd == "/exit" || cmd == "/quit" || cmd == "/bye" || cmd == "/q")
  {
    return true; /* signal exit */
  }
  else if (cmd == "/clear")
  {
    cmd_clear(true);
  }
  else if (cmd == "/reset")
  {
    cmd_clear(false);
  }
  else if (cmd == "/prompt")
  {
    std::string text = rest_of_line(line, 7);
    if (text.empty())
    {
      LOGW("usage: /prompt <text>\n");
      return false;
    }
    /* one-shot prompt with no history. The text is passed through the
       configured template components; when none are set the model's
       built-in template still wraps it (no truly raw mode). */
    std::string full = g_state.system_prompt + g_state.prompt_prefix + text + g_state.prompt_postfix;
    RunStats s;
    if (run_turn(full.c_str(), nullptr, false, true, &s) != 0)
    {
      LOGR("/prompt run failed\n");
      return false;
    }
    record_turn(s);
  }
  else if (cmd == "/temp")
  {
    if (args.size() < 2) { LOGW("usage: /temp <value>\n"); return false; }
    g_state.params.sampling_param.temperature = strtof(args[1].c_str(), nullptr);
    apply_llm_param();
    LOGI("temperature = %.3f\n", g_state.params.sampling_param.temperature);
  }
  else if (cmd == "/top-k")
  {
    if (args.size() < 2) { LOGW("usage: /top-k <n>\n"); return false; }
    g_state.params.sampling_param.top_k = atoi(args[1].c_str());
    apply_llm_param();
    LOGI("top_k = %d\n", g_state.params.sampling_param.top_k);
  }
  else if (cmd == "/top-p")
  {
    if (args.size() < 2) { LOGW("usage: /top-p <value>\n"); return false; }
    g_state.params.sampling_param.top_p = strtof(args[1].c_str(), nullptr);
    apply_llm_param();
    LOGI("top_p = %.3f\n", g_state.params.sampling_param.top_p);
  }
  else if (cmd == "/repeat-penalty")
  {
    if (args.size() < 2) { LOGW("usage: /repeat-penalty <value>\n"); return false; }
    g_state.params.sampling_param.repeat_penalty = strtof(args[1].c_str(), nullptr);
    apply_llm_param();
    LOGI("repeat_penalty = %.3f\n", g_state.params.sampling_param.repeat_penalty);
  }
  else if (cmd == "/frequency-penalty")
  {
    if (args.size() < 2) { LOGW("usage: /frequency-penalty <value>\n"); return false; }
    g_state.params.sampling_param.frequency_penalty = strtof(args[1].c_str(), nullptr);
    apply_llm_param();
    LOGI("frequency_penalty = %.3f\n", g_state.params.sampling_param.frequency_penalty);
  }
  else if (cmd == "/presence-penalty")
  {
    if (args.size() < 2) { LOGW("usage: /presence-penalty <value>\n"); return false; }
    g_state.params.sampling_param.presence_penalty = strtof(args[1].c_str(), nullptr);
    apply_llm_param();
    LOGI("presence_penalty = %.3f\n", g_state.params.sampling_param.presence_penalty);
  }
  else if (cmd == "/sampling")
  {
    if (args.size() < 7)
    {
      LOGW("usage: /sampling <top_k> <top_p> <temp> <repeat> <freq> <presence>\n");
      return false;
    }
    g_state.params.sampling_param.top_k = atoi(args[1].c_str());
    g_state.params.sampling_param.top_p = strtof(args[2].c_str(), nullptr);
    g_state.params.sampling_param.temperature = strtof(args[3].c_str(), nullptr);
    g_state.params.sampling_param.repeat_penalty = strtof(args[4].c_str(), nullptr);
    g_state.params.sampling_param.frequency_penalty = strtof(args[5].c_str(), nullptr);
    g_state.params.sampling_param.presence_penalty = strtof(args[6].c_str(), nullptr);
    apply_llm_param();
    LOGI("sampling params updated\n");
  }
  else if (cmd == "/predict" || cmd == "/max-tokens")
  {
    if (args.size() < 2) { LOGW("usage: /%s <n>\n", cmd.c_str()); return false; }
    int n = atoi(args[1].c_str());
    if (n <= 0) { LOGW("max_new_tokens must be > 0\n"); return false; }
    g_state.max_new_tokens = n;
    LOGI("max_new_tokens = %d\n", g_state.max_new_tokens);
  }
  else if (cmd == "/thinking")
  {
    bool v;
    if (args.size() < 2 || !parse_bool(args[1], v)) { LOGW("usage: /thinking on|off\n"); return false; }
    g_state.enable_thinking = v;
    /* fully re-create the session: the thinking markers live in the chat
       template, and a prior generation in the other mode leaves state that a
       kvcache clear does not remove. */
    if (create_session() != 0)
    {
      LOGR("session re-initialization failed after /thinking\n");
      return false;
    }
    LOGI("enable_thinking = %s, session re-initialized\n", v ? "on" : "off");
  }
  else if (cmd == "/ignore-eos")
  {
    bool v;
    if (args.size() < 2 || !parse_bool(args[1], v)) { LOGW("usage: /ignore-eos on|off\n"); return false; }
    g_state.ignore_eos = v;
    apply_llm_param();
    LOGI("ignore_eos = %s\n", v ? "on" : "off");
  }
  else if (cmd == "/keep-history")
  {
    bool v;
    if (args.size() < 2 || !parse_bool(args[1], v)) { LOGW("usage: /keep-history on|off\n"); return false; }
    g_state.keep_history = v;
    rknn3_session_clear_kvcache(g_state.session, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
    LOGI("keep_history = %s, KV cache cleared (new conversation)\n", v ? "on" : "off");
  }
  else if (cmd == "/verbose")
  {
    bool v;
    if (args.size() < 2 || !parse_bool(args[1], v)) { LOGW("usage: /verbose on|off\n"); return false; }
    g_state.verbose = v;
    LOGI("verbose = %s\n", v ? "on" : "off");
  }
  else if (cmd == "/system")
  {
    std::string text = rest_of_line(line, 7);
    if (text.empty()) { LOGW("usage: /system <prompt>\n"); return false; }
    g_state.system_prompt = text;
    if (create_session() != 0)
    {
      LOGR("session re-initialization failed after /system\n");
      return false;
    }
    LOGI("system prompt updated, session re-initialized.\n");
  }
  else if (cmd == "/template")
  {
    std::string rest = rest_of_line(line, 9);
    std::vector<std::string> parts = split_args(rest);
    if (parts.size() < 2)
    {
      LOGW("usage: /template <prefix> <postfix>\n");
      return false;
    }
    g_state.prompt_prefix = parts[0];
    g_state.prompt_postfix = parts[1];
    if (create_session() != 0)
    {
      LOGR("session re-initialization failed after /template\n");
      return false;
    }
    LOGI("chat template updated, session re-initialized.\n");
  }
  else if (cmd == "/status")
  {
    print_status();
  }
  else if (cmd == "/ctx")
  {
    print_ctx();
  }
  else if (cmd == "/stop" || cmd == "/cancel" || cmd == "/interrupt")
  {
    LOGW("no generation in progress to stop\n");
  }
  else if (cmd == "/perf")
  {
    print_perf(g_state.last_run);
  }
  else if (cmd == "/stats")
  {
    print_stats();
  }
  else if (cmd == "/bench")
  {
    if (cmd_bench(args) != 0)
      LOGR("/bench failed\n");
  }
  else if (cmd == "/save")
  {
    std::string path = rest_of_line(line, 5);
    if (path.empty()) { LOGW("usage: /save <path>\n"); return false; }
    int ret = rknn3_session_save_kvcache(g_state.session, path.c_str());
    if (ret == RKNN3_SUCCESS) LOGI("KV cache saved to '%s'\n", path.c_str());
    else LOGR("rknn3_session_save_kvcache failed, ret=%d\n", ret);
  }
  else if (cmd == "/load")
  {
    std::string path = rest_of_line(line, 5);
    if (path.empty()) { LOGW("usage: /load <path>\n"); return false; }
    int ret = rknn3_session_load_kvcache_from_path(g_state.session, path.c_str());
    if (ret == RKNN3_SUCCESS) LOGI("KV cache loaded from '%s'\n", path.c_str());
    else LOGR("rknn3_session_load_kvcache_from_path failed, ret=%d\n", ret);
  }
  else if (cmd == "/lora")
  {
    /* load the adapter onto the context only (not enabled yet) */
    std::string path = rest_of_line(line, 5);
    if (path.empty()) { LOGW("usage: /lora <path>  (then /lora-enable)\n"); return false; }
    if (lora_load(&g_state.lora, g_state.ctx, path.c_str()) == 0)
      LOGI("LoRA loaded: %s\n", g_state.lora.lora.lora_name);
    else
      LOGR("LoRA load failed for '%s'\n", path.c_str());
  }
  else if (cmd == "/lora-enable")
  {
    if (lora_enable(&g_state.lora, g_state.session) == 0)
      LOGI("LoRA enabled: %s (scale=%.3f)\n", g_state.lora.lora.lora_name, g_state.lora.lora.scale);
    else
      LOGR("LoRA enable failed\n");
  }
  else if (cmd == "/lora-disable")
  {
    int r = lora_disable(&g_state.lora, g_state.session);
    if (r == 0) LOGI("LoRA disabled\n");
    else if (r == 1) LOGW("LoRA not enabled\n");
    else LOGR("LoRA disable failed\n");
  }
  else if (cmd == "/lora-unload")
  {
    if (lora_unload(&g_state.lora) == 0)
      LOGI("LoRA unloaded\n");
    else
      LOGR("LoRA unload failed\n");
  }
  else if (cmd == "/lora-off" || cmd == "/disable-lora")
  {
    /* full teardown: disable (if enabled) + unload */
    if (lora_teardown(&g_state.lora) == 0)
      LOGI("LoRA disabled and unloaded\n");
    else
      LOGR("LoRA teardown failed\n");
  }
  else
  {
    LOGW("unknown command '%s' (type /help for the command list)\n", cmd.c_str());
  }
  return false;
}

/*=============================================================================
   Interactive REPL
=============================================================================*/

static int run_interactive(const CommandLineOptions& opt)
{
  /* initial prompt (if any) */
  std::string initial_prompt;
  const char* initial = opt.prompt;
  if (initial)
  {
    /* a prompt is read from a file when it starts with '@' (explicit @file)
       or when the value itself names an existing regular file; otherwise it
       is used verbatim as prompt text. */
    const char* file = initial[0] == '@' ? initial + 1
                     : file_exists(initial) ? initial : nullptr;
    if (file)
    {
      if (!read_file(file, initial_prompt))
      {
        LOGR("Failed to read prompt file: %s\n", file);
        return -1;
      }
      initial = initial_prompt.c_str();
    }
  }

  if (initial && initial[0] != '\0')
  {
    printf("\033[1;32m> \033[0m%s\n", initial);
    RunStats s;
    if (run_turn(initial, nullptr, g_state.keep_history, true, &s) != 0)
    {
      LOGR("initial prompt run failed\n");
      return -1;
    }
    record_turn(s);
    if (opt.once)
    {
      return 0;
    }
    printf("\n");
  }
  else if (opt.once)
  {
    LOGW("--once requires --prompt\n");
    return -1;
  }

  /* REPL loop */
  bool quit = false;
  while (!quit)
  {
    printf("\033[1;32m> \033[0m");
    fflush(stdout);

    std::string line;
    if (read_line(line) < 0)
    {
      printf("\nbye.\n");
      break; /* EOF (Ctrl+D) */
    }

    std::string trimmed = trim(line);
    if (trimmed.empty()) continue;

    if (trimmed[0] == '/')
    {
      if (dispatch_command(trimmed))
      {
        printf("bye.\n");
        break;
      }
      continue;
    }

    /* normal chat turn */
    RunStats s;
    int r = run_turn(line.c_str(), nullptr, g_state.keep_history, true, &s);
    if (r != 0)
    {
      LOGR("generation failed, ret=%d\n", r);
      continue;
    }
    record_turn(s);
    printf("\n");
  }

  return 0;
}

/*=============================================================================
   Main
=============================================================================*/

int main(int argc, char** argv)
{
  g_stdin_tty = isatty(STDIN_FILENO) != 0;

  CommandLineOptions opt;
  if (!parse_command_line(argc, argv, &opt))
  {
    print_usage(argv[0]);
    return -1;
  }

  /* --dry-run: print the resolved model paths and exit without touching any
     device. Useful to verify path auto-derivation. */
  if (opt.dry_run)
  {
    printf("model     : %s\n", opt.model.c_str());
    printf("weight    : %s\n", opt.weight.c_str());
    printf("tokenizer : %s\n", opt.tokenizer.c_str());
    printf("embedding : %s\n", opt.embedding.c_str());
    return 0;
  }

  int ret = 0;
  bool bench_mode = opt.bench_in > 0;

  rknn3_devices devs;
  memset(&devs, 0, sizeof(devs));
  ret = rknn3_find_devices(&devs);
  if (ret != RKNN3_SUCCESS || devs.n_devices == 0)
  {
    LOGR("rknn3_find_devices fail! ret=%d\n", ret);
    return -1;
  }
  printf("found %u RKNN3 device(s)\n", devs.n_devices);

  /* resolve the device to run on: an explicit --device-id must match one of
     the found devices; otherwise fall back to the first device. */
  const char* device_id = opt.device_id;
  if (device_id)
  {
    uint32_t i = 0;
    for (; i < devs.n_devices; ++i)
    {
      if (strcmp(devs.devices[i].id, device_id) == 0)
      {
        printf("using device[%u]: id=%s\n", i, devs.devices[i].id);
        break;
      }
    }
    if (i == devs.n_devices)
    {
      LOGR("--device-id '%s' not found among %u device(s)\n", device_id, devs.n_devices);
      return -1;
    }
  }
  else
  {
    device_id = devs.devices[0].id;
  }

  /* decode strategy: SESSION (one session_run per turn) */
  g_state.decode_mode = DecodeMode::SESSION;

  /* init model */
  g_state.ctx = init_context_and_model(opt.model.c_str(), opt.weight.c_str(), opt.core_mask, device_id);
  if (!g_state.ctx)
  {
    LOGR("init_context_and_model fail\n");
    return -1;
  }

  ret = rknn3_query(g_state.ctx, RKNN3_QUERY_LLM_CONFIG, &g_state.llm_config,
                    sizeof(rknn3_llm_config));
  if (ret != RKNN3_SUCCESS)
  {
    LOGR("rknn3_query llm config failed! ret=%d\n", ret);
    goto exit;
  }

  /* tokenizer + embedding */
  if (init_tokenizer_and_embedding(opt.tokenizer.c_str(), opt.embedding.c_str()) != 0)
  {
    ret = -1;
    goto exit;
  }

  /* validate / apply context length */
  if (opt.ctx_size > 0 && (uint32_t)opt.ctx_size != g_state.llm_config.max_ctx_len)
  {
    if ((uint32_t)opt.ctx_size < g_state.llm_config.max_ctx_len)
    {
      LOGW("Warning: ctx-size (%d) < model max_ctx_len (%u). Using %d.\n",
           opt.ctx_size, g_state.llm_config.max_ctx_len, opt.ctx_size);
      g_state.llm_config.max_ctx_len = (uint32_t)opt.ctx_size;
    }
    else
    {
      LOGR("Error: ctx-size (%d) > model max_ctx_len (%u).\n",
           opt.ctx_size, g_state.llm_config.max_ctx_len);
      ret = -1;
      goto exit;
    }
  }

  /* session defaults */
  g_state.max_new_tokens = opt.predict > 0 ? opt.predict : 512;
  g_state.ignore_eos = opt.ignore_eos;
  g_state.enable_thinking = opt.thinking;
  g_state.verbose = opt.verbose;
  g_state.keep_history = opt.keep_history;
  /* template components are only set when explicitly given; otherwise the
     session keeps using the model's built-in chat template (queried from
     the rknn model), so no hard-coded default is needed. */
  if (opt.system) g_state.system_prompt = opt.system;
  if (opt.prompt_prefix) g_state.prompt_prefix = opt.prompt_prefix;
  if (opt.prompt_postfix) g_state.prompt_postfix = opt.prompt_postfix;

  g_logits_name = opt.logits_name;

  /* --rope-tensor: enforce the model's rope_cache_host_storage contract.
     Must run BEFORE init_session() so that create_session() registers the
     input_callback (it checks g_state.rope.enabled); only the context is
     needed here, not the session. */
  ret = require_rope_cache(opt.rope_tensor);
  if (ret != 0)
  {
    LOGR("rope cache contract check failed, ret=%d\n", ret);
    goto exit;
  }

  ret = init_session(opt);
  if (ret != 0)
  {
    LOGR("init_session failed, ret=%d\n", ret);
    goto exit;
  }

  /* --lora: enable a LoRA adapter on startup (failure keeps the base model) */
  if (opt.lora)
  {
    if (lora_setup(&g_state.lora, g_state.ctx, g_state.session, opt.lora) == 0)
      LOGI("LoRA enabled: %s (scale=%.3f)\n", g_state.lora.lora.lora_name, g_state.lora.lora.scale);
    else
      LOGR("LoRA setup failed for '%s', continuing with base model\n", opt.lora);
  }

  if (bench_mode)
  {
    ret = run_bench(opt.bench_in, opt.bench_out);
    if (ret != 0)
      LOGR("benchmark run failed, ret=%d\n", ret);
    goto exit;
  }

  /* --load: resume a previously saved conversation */
  if (opt.load)
  {
    ret = rknn3_session_load_kvcache_from_path(g_state.session, opt.load);
    if (ret == RKNN3_SUCCESS) LOGI("KV cache loaded from '%s'\n", opt.load);
    else LOGR("rknn3_session_load_kvcache_from_path failed, ret=%d\n", ret);
  }

  /* banner */
  printf("\n");
  LOGG("  rknn3-llm-cli\n");
  printf("  model            : %s\n", opt.model.c_str());
  printf("  weight           : %s\n", opt.weight.c_str());
  printf("  tokenizer        : %s\n", opt.tokenizer.c_str());
  printf("  embedding        : %s\n", opt.embedding.c_str());
  print_model_info();
  printf("  Type '/help' for commands, type your message to chat, Ctrl+D to exit.\n\n");

  ret = run_interactive(opt);
  if (ret != 0)
  {
    LOGR("run_interactive failed, ret=%d\n", ret);
  }

exit:
  /* --save: persist the conversation kvcache before tearing down the session.
     best-effort: a save failure must not turn the run into an error exit. */
  if (!bench_mode && opt.save && g_state.session)
  {
    int sret = rknn3_session_save_kvcache(g_state.session, opt.save);
    if (sret == RKNN3_SUCCESS) LOGI("KV cache saved to '%s'\n", opt.save);
    else LOGR("rknn3_session_save_kvcache failed, ret=%d\n", sret);
  }
  release_resources();
  return ret;
}
