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

#include "rknn3_session_test_common.h"

// Perf callback: suppresses all text output, only records first-token time.
static int perf_result_callback(void* userdata, RKLLMResult* result, LLMCallState state)
{
  rknn3_test_session_userdata_t* session_userdata = (rknn3_test_session_userdata_t*)userdata;
  (void)result;
  if (state == RKLLM_RUN_NORMAL) {
    rknn3_test_timing_mark_first_token(&session_userdata->timing);
  }
  fflush(stdout);
  return 0;
}

int main(int argc, char** argv)
{
  // Usage: %s <rknn> <weight> <tokenizer> <embedding> <max_context_len> <n_input_tokens> <max_new_tokens> <core_mask> [keep_history]
  // [ignore_eos_token] [key_path]
  if (argc < 9 || argc > 12) {
    LOGW(
      "Usage: %s <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <n_input_tokens> <max_new_tokens> <core_mask> [keep_history] [ignore_eos_token] [key_path]\n ",
      argv[0]);
    LOGW(
      "Such as: %s Qwen2.5-0.5B.rknn Qwen2.5-0.5B.weight Qwen2.5-0.5B.tokenizer.gguf Qwen2.5-0.5B.embed.bin 1024 128 128 0xff 0 0 ./model/key.env\n",
      argv[0]);
    return -1;
  }

  int         ret              = 0;
  char*       model_path       = argv[1];
  char*       weight_path      = argv[2];
  const char* tokenizer_path   = argv[3];
  const char* embedding_path   = argv[4];
  int32_t     max_context_len  = atoi(argv[5]);
  uint64_t    n_input_tokens   = atoi(argv[6]);
  int32_t     max_new_tokens   = atoi(argv[7]);
  uint32_t    core_mask        = strtoul(argv[8], nullptr, 16);
  int         keep_history     = (argc > 9) ? atoi(argv[9]) : 0;
  bool        ignore_eos_token = (argc > 10) ? (atoi(argv[10]) != 0) : true;
  const char* key_path         = (argc > 11) ? argv[11] : nullptr;

  rknn3_devices                 devs;
  rknn3_context                 ctx = 0;
  VocabInfo                     vocab_info;
  Tokenizer*                    tokenizer = nullptr;
  struct stat                   emb_st;
  struct embedding_info         embedding_info;
  RKLLMCallback                 callback;
  rknn3_test_session_userdata_t session_userdata;
  rknn3_llm_param               params;
  RKLLMRunState                 state;
  rknn3_lora                    loras_enabled[RKNN3_MAX_LORA_NUM];
  rknn3_llm_config              llm_config;
  rknn3_kvcache_policy          kvcache_policy = RKNN3_KVCACHE_POLICY_NORMAL;
  rknn3_kvcache_policy_param    kvcache_policy_param;
  int                           llm_n_inputs = 1;
  rknn3_llm_input*              llm_inputs[1];
  rknn3_llm_infer_param         llm_infer_param;
  int                           n_params = 1;

  memset(&devs, 0, sizeof(devs));
  memset(&vocab_info, 0, sizeof(vocab_info));
  memset(&emb_st, 0, sizeof(emb_st));
  memset(&embedding_info, 0, sizeof(embedding_info));
  memset(&callback, 0, sizeof(callback));
  memset(&params, 0, sizeof(params));
  memset(&state, 0, sizeof(state));
  memset(&llm_config, 0, sizeof(llm_config));
  memset(loras_enabled, 0, sizeof(loras_enabled));
  memset(&kvcache_policy_param, 0, sizeof(kvcache_policy_param));
  memset(llm_inputs, 0, sizeof(llm_inputs));
  memset(&llm_infer_param, 0, sizeof(llm_infer_param));
  memset(&session_userdata, 0, sizeof(session_userdata));
  state.loras_enabled = loras_enabled;

  int test_num = 3;

  printf("*******************************NEW TEST**********************************\n");

  // find devices
  const char* device_id = nullptr;
  ret                   = rknn3_find_devices(&devs);
  if (ret != RKNN3_SUCCESS || devs.n_devices == 0) {
    printf("rknn3_find_devices fail! ret=%d\n", ret);
    goto exit;
  }

  // init context and model
  device_id = devs.devices[0].id;
  ret       = init_context_and_model(&ctx, model_path, weight_path, core_mask, key_path, device_id);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_init_model fail! ret=%d\n", ret);
    goto exit;
  }

  // get tokenizer and embedding
  ret = get_tokenizer_and_embedding(tokenizer_path, &vocab_info, &tokenizer, &embedding_info, embedding_path, &emb_st);
  if (ret < 0) {
    printf("get_tokenizer_and_embedding fail! ret=%d\n", ret);
    goto exit;
  }

  ret = rknn3_query(ctx, RKNN3_QUERY_LLM_CONFIG, &llm_config, sizeof(rknn3_llm_config));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query llm config failed! ret=%d", ret);
    goto exit;
  }

  // Set basic session parameters (ignore_eos_token=true for perf testing)
  build_default_llm_params(&params, &vocab_info, max_context_len, ignore_eos_token);

  // init session
  session_userdata.session = rknn3_session_init(ctx, &params, n_params);
  if (!session_userdata.session) {
    printf("Failed to initialize test session_userdata.session, ret=%d\n", ret);
    goto exit;
  }

  // set callback: start from defaults, then override result_callback
  // with the perf variant that suppresses text output.
  build_default_callback(&callback, tokenizer, &embedding_info, &session_userdata);
  callback.result_callback = perf_result_callback;

  ret = rknn3_session_set_callback(session_userdata.session, &callback);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_set_callback failed, ret=%d\n", ret);
    goto exit;
  }

  print_llm_config(&llm_config);
  printf("=============================================================\n");
  printf("%-32s: %-8llu\n", "Number of Input Tokens", (unsigned long long)n_input_tokens);
  printf("%-32s: %-8d\n", "Max New Tokens", max_new_tokens);
  printf("=============================================================\n\n");

  // Test the performance evaluation workflow: each iteration uses the same
  // fixed input data so that TPS, TTFT, and related metrics can be compared
  // under consistent and repeatable test conditions.
  for (int i = 0; i < test_num; i++) {
    // generate fixed tokens for performance testing
    std::vector<int32_t> tokens(n_input_tokens);
    srand(42); // use fixed seed
    for (uint64_t j = 0; j < n_input_tokens; j++) {
      tokens[j] = rand() % 1024;
    }

    llm_infer_param = build_default_infer_param(max_new_tokens, keep_history);
    llm_inputs[0]   = build_llm_input_tokens(tokens.data(), n_input_tokens, false, false);

    rknn3_test_timing_mark_start(&session_userdata.timing);
    ret = rknn3_session_run(session_userdata.session, llm_inputs[0], llm_n_inputs, &llm_infer_param);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_session_run failed, ret=%d\n", ret);
      goto exit;
    }
    rknn3_test_timing_mark_end(&session_userdata.timing);

    ret = rknn3_session_query_state(session_userdata.session, &state);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_session_query_state failed, ret=%d\n", ret);
      goto exit;
    }

    ret = rknn3_session_clear_kvcache(session_userdata.session, RKNN3_KVCACHE_KEEP_SYSTEM_PROMPT);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_session_clear_kvcache failed, ret=%d\n", ret);
      goto exit;
    }

    printf("Run %d:", i);
    print_perf_stats(&state, &session_userdata.timing);

    if (i == test_num - 1) {
      // Summary table on the last run
      uint64_t prefill_n  = state.n_prefill_tokens;
      uint64_t decode_n   = state.n_decode_tokens;
      float    prefill_us = (session_userdata.timing.first_token.tv_sec - session_userdata.timing.start.tv_sec) * 1e6f +
                         (session_userdata.timing.first_token.tv_usec - session_userdata.timing.start.tv_usec);
      float prefill_ms     = prefill_us / 1e3f;
      float prefill_s      = prefill_us / 1e6f;
      float prefill_tps    = prefill_n == 0 ? 0.0f : prefill_n / prefill_s;
      float decode_time_us = ((session_userdata.timing.end.tv_sec - session_userdata.timing.first_token.tv_sec) * 1e6f) +
                             (session_userdata.timing.end.tv_usec - session_userdata.timing.first_token.tv_usec);
      float decode_ms  = decode_time_us / 1e3f;
      float decode_tpt = decode_n == 0 ? 0.0f : decode_ms / decode_n;
      float decode_s   = decode_time_us / 1e6f;
      float decode_tps = decode_n == 0 ? 0.0f : decode_n / decode_s;

      printf("\n\n====================================================================================================================\n");
      printf("%*s\n", 60 + (int)strlen("Performance Summary") / 2, "Performance Summary");
      printf("====================================================================================================================\n");
      printf("--------------------------------------------------------------------------------------------------------------------\n");
      printf(" %-15s | %-11s | %-13s | %-11s | %-9s | %-10s | %-13s | %-10s \n", "Model Context", "Rope Length", "Input Tokens",
             "Output Tokens", "TTFT(ms)", "TPOT(ms)", "Prefill TPS", "Decode TPS");
      printf("--------------------------------------------------------------------------------------------------------------------\n");
      printf(" %-15d | %-11d | %-13llu | %-11llu | %-9.2f | %-10.2f | %-13.2f | %-10.2f \n", llm_config.max_ctx_len,
             llm_config.max_position_embeddings, (unsigned long long)n_input_tokens, (unsigned long long)(decode_n + 1), prefill_ms,
             decode_tpt, prefill_tps, decode_tps);
      printf("--------------------------------------------------------------------------------------------------------------------\n");
    }

    for (int j = 0; j < llm_n_inputs; j++) {
      release_llm_input(ctx, llm_inputs[j]);
    }
    fflush(stdout);
  }

exit:
  for (int j = 0; j < llm_n_inputs; j++) {
    release_llm_input(ctx, llm_inputs[j]);
  }

  if (session_userdata.session) {
    rknn3_session_destroy(session_userdata.session);
  }

  if (ctx) {
    rknn3_destroy(ctx);
  }

  embedding_info_release(&embedding_info, &emb_st);

  if (tokenizer) {
    delete tokenizer;
  }

  printf("\n*******************************END TEST**********************************\n");

  fflush(stdout);

  return 0;
}
