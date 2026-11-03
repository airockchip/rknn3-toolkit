// rknn3_llm_step_decode : per-token (external) decode demo for RKNN3 LLM models.
//
// Unlike the interactive CLI (which lets one rknn3_session_run(max_new_tokens=N)
// perform prefill + all decode internally), this demo drives the decode loop
// step by step from the caller: every rknn3_session_run is issued with
// max_new_tokens=1, and the token produced by the result callback is fed back
// as the next input until max_new_tokens / EOS / interrupt.
//
// This front-end shares the whole inference engine with rknn3_llm_cli (see
// engine.h / engine.cc); the only difference is the decode strategy
// (DecodeMode::EXTERNAL).
//
//   run_turn():                                  // engine.cc
//     run_inference_once(prefill)                // prefill the whole prompt, produces token[0]
//     for step in 0 .. max_new_tokens:
//       run_inference_once({token[step]})        // decode one token per session_run
//
// Usage:
//   ./rknn3_llm_step_decode --model <rknn_path> [options]

#include "args.h"
#include "engine.h"

#include <sys/time.h>
#include <unistd.h>

#include <vector>

/*=============================================================================
   Interactive loop
=============================================================================*/

static int run(const CommandLineOptions& opt)
{
  /* initial prompt (if any) */
  std::string initial_prompt;
  const char* initial = opt.prompt;
  if (initial && initial[0] == '@')
  {
    if (!read_file(initial + 1, initial_prompt))
    {
      LOGR("Failed to read prompt file: %s\n", initial + 1);
      return -1;
    }
    initial = initial_prompt.c_str();
  }

  if (initial && initial[0] != '\0')
  {
    printf("\033[1;32m> \033[0m%s\n", initial);
    RunStats s;
    if (run_turn(initial, nullptr, g_state.keep_history, false, &s) != 0)
    {
      return -1;
    }
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

  /* loop */
  while (1)
  {
    printf("\033[1;32m> \033[0m");
    fflush(stdout);

    std::string line;
    if (read_line(line) < 0)
    {
      break;
    }

    std::string trimmed = trim(line);
    if (trimmed.empty()) continue;

    if (trimmed == "/exit")
    {
      break;
    }

    /* normal chat turn */
    RunStats s;
    int r = run_turn(line.c_str(), nullptr, g_state.keep_history, false, &s);
    if (r != 0)
    {
      LOGR("generation failed, ret=%d\n", r);
      continue;
    }
    printf("\n");
    if (opt.perf) print_perf(s);
  }

  return 0;
}

/*=============================================================================
   Main
=============================================================================*/

int main(int argc, char** argv)
{
  CommandLineOptions options;
  if (!parse_command_line(argc, argv, &options))
  {
    print_usage(argv[0]);
    return -1;
  }

  const char* model_path = options.model.c_str();
  const char* weight_path = options.weight.c_str();
  const char* tokenizer_path = options.tokenizer.c_str();
  const char* embedding_path = options.embedding.c_str();
  uint32_t run_core_mask = options.core_mask;

  /* --dry-run: print the resolved model paths and exit without touching any
     device. Useful to verify path auto-derivation. */
  if (options.dry_run)
  {
    printf("model     : %s\n", model_path);
    printf("weight    : %s\n", weight_path);
    printf("tokenizer : %s\n", tokenizer_path);
    printf("embedding : %s\n", embedding_path);
    return 0;
  }

  /* 校验模型/权重文件存在 */
  struct stat st;
  if (stat(model_path, &st) != 0)
  {
    printf("ERROR: model file not found: %s\n", model_path);
    return -1;
  }
  if (stat(weight_path, &st) != 0)
  {
    printf("ERROR: weight file not found: %s\n", weight_path);
    return -1;
  }

  /* 查找设备，单卡使用第一个可用设备（或用户指定的 --device-id） */
  rknn3_devices devs;
  memset(&devs, 0, sizeof(devs));
  int ret = rknn3_find_devices(&devs);
  if (ret != RKNN3_SUCCESS)
  {
    printf("find devices failed: ret=%d\n", ret);
    return -1;
  }
  printf("found %d devices:\n", devs.n_devices);
  for (int i = 0; i < devs.n_devices; ++i)
  {
    printf("  [%d] type=%s, id=%s\n", i, devs.devices[i].type, devs.devices[i].id);
  }

  if (devs.n_devices < 1)
  {
    printf("no devices found\n");
    return -1;
  }

  const char* device_id = nullptr;
  if (options.device_id)
  {
    bool found = false;
    for (int i = 0; i < devs.n_devices; ++i)
    {
      if (strcmp(devs.devices[i].id, options.device_id) == 0)
      {
        device_id = devs.devices[i].id;
        found = true;
        break;
      }
    }
    if (!found)
    {
      printf("device-id %s not found in available devices\n", options.device_id);
      return -1;
    }
  }
  else
  {
    device_id = devs.devices[0].id;
    printf("device-id unspecified, use the first device: %s \n", device_id);
  }

  /* decode strategy: EXTERNAL (one session_run per token) */
  g_state.decode_mode = DecodeMode::EXTERNAL;

  /* 初始化 RKNN3 */
  g_state.ctx = init_context_and_model(model_path, weight_path, run_core_mask, device_id);
  if (!g_state.ctx)
  {
    return -1;
  }

  ret = rknn3_query(g_state.ctx, RKNN3_QUERY_LLM_CONFIG, &g_state.llm_config,
                    sizeof(rknn3_llm_config));
  if (ret != RKNN3_SUCCESS)
  {
    printf("query llm config failed, ret=%d\n", ret);
    release_resources();
    return -1;
  }

  /* 加载tokenizer和embedding */
  if (init_tokenizer_and_embedding(tokenizer_path, embedding_path) != 0)
  {
    release_resources();
    return -1;
  }

  /* 校验、应用上下文长度 */
  if (options.ctx_size > 0 && (uint32_t)options.ctx_size != g_state.llm_config.max_ctx_len)
  {
    if ((uint32_t)options.ctx_size < g_state.llm_config.max_ctx_len)
    {
      LOGW("Warning: ctx-size (%d) < model max_ctx_len (%u). Using %d.\n",
           options.ctx_size, g_state.llm_config.max_ctx_len, options.ctx_size);
      g_state.llm_config.max_ctx_len = (uint32_t)options.ctx_size;
    }
    else
    {
      LOGR("Error: ctx-size (%d) > model max_ctx_len (%u).\n",
           options.ctx_size, g_state.llm_config.max_ctx_len);
      release_resources();
      return -1;
    }
  }

  /* --rope-tensor: enforce the model's rope_cache_host_storage contract
     (Qwen3.5 / Gemma-4). Must run BEFORE init_session() so create_session()
     registers the input_callback. */
  if (require_rope_cache(options.rope_tensor) != 0)
  {
    release_resources();
    return -1;
  }

  /* session 参数 */
  g_state.max_new_tokens = options.predict > 0 ? options.predict : 512;
  g_state.ignore_eos = options.ignore_eos;
  g_state.enable_thinking = options.thinking;
  g_state.keep_history = options.keep_history;
  g_state.verbose = options.verbose;
  if (options.system) g_state.system_prompt = options.system;
  if (options.prompt_prefix) g_state.prompt_prefix = options.prompt_prefix;
  if (options.prompt_postfix) g_state.prompt_postfix = options.prompt_postfix;
  g_logits_name = options.logits_name;

  /* sampling 参数由 init_session() 从 options 填充 */

  /* kvcache checkpoint 策略 */
  if (options.has_checkpoint)
  {
    memset(&g_state.kvcache_policy, 0, sizeof(g_state.kvcache_policy));
    g_state.kvcache_policy.save_checkpoint.checkpoint_start_pos = options.checkpoint_start_pos;
    g_state.kvcache_policy.save_checkpoint.checkpoint_interval = options.checkpoint_interval;
    g_state.kvcache_policy.save_checkpoint.max_checkpoint_count = options.checkpoint_max_count;
    g_state.kvcache_policy.save_checkpoint.checkpoint_tail_overwrite = options.checkpoint_tail_overwrite;
    g_state.checkpoint_enabled = true;
  }

  ret = init_session(options);
  if (ret != 0)
  {
    LOGR("init_session failed, ret=%d\n", ret);
    release_resources();
    return -1;
  }

  /* --lora: 启动时启用LoRA */
  if (options.lora)
  {
    if (lora_setup(&g_state.lora, g_state.ctx, g_state.session, options.lora) == 0)
      LOGI("LoRA enabled: %s (scale=%.3f)\n", g_state.lora.lora.lora_name, g_state.lora.lora.scale);
    else
      LOGR("LoRA setup failed for '%s', continuing with base model\n", options.lora);
  }

  /* --bench: 运行基准测试后退出 */
  if (options.bench_in > 0)
  {
    int bret = run_bench(options.bench_in, options.bench_out);
    rknn3_session_clear_kvcache(g_state.session, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
    release_resources();
    return bret == 0 ? 0 : -1;
  }

  /* --load: 通过导入kvcache恢复以前保存的对话 */
  if (options.load)
  {
    ret = rknn3_session_load_kvcache_from_path(g_state.session, options.load);
    if (ret == RKNN3_SUCCESS) LOGI("KV cache loaded from '%s'\n", options.load);
    else LOGR("rknn3_session_load_kvcache_from_path failed, ret=%d\n", ret);
  }

  /* banner */
  printf("\n");
  LOGI("==============================INFO==============================\n");
  printf("  model            : %s\n", model_path);
  printf("  weight           : %s\n", weight_path);
  printf("  tokenizer        : %s\n", tokenizer_path);
  printf("  embedding        : %s\n", embedding_path);
  print_status();
  LOGI("Type your message, input /exit to quit.\n");

  ret = run(options);
  if (ret != 0)
  {
    LOGR("run failed, ret=%d\n", ret);
  }

  /* --save: 在终止会话之前，先保存当前会话的kvcache */
  if (options.save && g_state.session)
  {
    int sret = rknn3_session_save_kvcache(g_state.session, options.save);
    if (sret == RKNN3_SUCCESS) LOGI("KV cache saved to '%s'\n", options.save);
    else LOGR("rknn3_session_save_kvcache failed, ret=%d\n", sret);
  }

  release_resources();
  return ret;
}
