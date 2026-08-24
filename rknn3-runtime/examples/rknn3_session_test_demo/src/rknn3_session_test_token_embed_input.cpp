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

int main(int argc, char** argv)
{
  // Usage: %s <rknn> <weight> <tokenizer> <embedding> <max_context_len> <max_new_tokens> <core_mask> <test_type> [keep_history]
  // [ignore_eos_token] [key_path]
  if (argc < 9 || argc > 12) {
    LOGW(
      "Usage: %s <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask> <test_type> [keep_history] [ignore_eos_token] [key_path]\n ",
      argv[0]);
    LOGW(
      "Such as: %s ./model/Qwen2.5-0.5B.rknn ./model/Qwen2.5-0.5B.weight ./model/Qwen2.5-0.5B.tokenizer.gguf ./model/Qwen2.5-0.5B-embedding.bin 1024 256 0xff 0 0 0 ./model/key.env\n",
      argv[0]);

    return -1;
  }

  int         ret              = 0;
  char*       model_path       = argv[1];
  char*       weight_path      = argv[2];
  const char* tokenizer_path   = argv[3];
  const char* embedding_path   = argv[4];
  int32_t     max_context_len  = atoi(argv[5]);
  int32_t     max_new_tokens   = atoi(argv[6]);
  uint32_t    core_mask        = strtoul(argv[7], nullptr, 16);
  int         test_type        = atoi(argv[8]);
  int         keep_history     = (argc > 9) ? atoi(argv[9]) : 0;
  bool        ignore_eos_token = (argc > 10) ? (atoi(argv[10]) != 0) : false;
  const char* key_path         = (argc > 11) ? argv[11] : nullptr;

  rknn3_devices                 devs;
  rknn3_context                 ctx = 0;
  rknn3_input_output_num        io_num;
  VocabInfo                     vocab_info;
  Tokenizer*                    tokenizer = nullptr;
  struct stat                   emb_st;
  struct embedding_info         embedding_info;
  RKLLMCallback                 callback;
  rknn3_test_session_userdata_t session_userdata;
  rknn3_llm_param               params;
  int                           n_params = 1;
  RKLLMRunState                 state;
  rknn3_lora                    loras_enabled[RKNN3_MAX_LORA_NUM];
  rknn3_llm_config              llm_config;
  rknn3_kvcache_policy          kvcache_policy = RKNN3_KVCACHE_POLICY_NORMAL;
  rknn3_kvcache_policy_param    kvcache_policy_param;
  int                           llm_n_inputs = 1;
  rknn3_llm_input*              llm_inputs[llm_n_inputs];
  rknn3_llm_infer_param         llm_infer_param;

  memset(&devs, 0, sizeof(devs));
  memset(&io_num, 0, sizeof(io_num));
  memset(&vocab_info, 0, sizeof(vocab_info));
  memset(&emb_st, 0, sizeof(emb_st));
  memset(&embedding_info, 0, sizeof(embedding_info));
  memset(&callback, 0, sizeof(callback));
  memset(&params, 0, sizeof(params));
  memset(&state, 0, sizeof(state));
  memset(&llm_config, 0, sizeof(llm_config));
  memset(loras_enabled, 0, sizeof(loras_enabled));
  memset(&kvcache_policy_param, 0, sizeof(kvcache_policy_param));
  state.loras_enabled = loras_enabled;
  memset(llm_inputs, 0, sizeof(llm_inputs));
  memset(&llm_infer_param, 0, sizeof(llm_infer_param));
  memset(&session_userdata, 0, sizeof(session_userdata));

  // system prompt kept in the KV cache for the TOKEN / EMBED paths.
  std::string system_prompt_for_keep =
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n";
  uint64_t             system_prompt_for_keep_n_tokens         = 0;
  uint64_t             system_prompt_for_keep_n_tokens_aligned = 0;
  std::vector<int32_t> system_prompt_for_keep_tokens;

  std::vector<std::string> random_prompts  = {"请解释一下相对论的基本概念。", "Please explain the basic concept of relativity"};
  std::vector<std::string> random_prompts2 = {
    "<|im_start|>user\n请解释一下相对论的基本概念。<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nPlease explain the basic concept of relativity<|im_end|>\n<|im_start|>assistant\n",
  };

  int test_num = random_prompts.size();

  float16* embed = nullptr;

  printf("*******************************NEW TEST**********************************\n");

  // find devices
  const char* device_id = nullptr;
  ret                   = rknn3_find_devices(&devs);
  if (ret != RKNN3_SUCCESS || devs.n_devices == 0) {
    printf("rknn3_find_devices failed, ret = %d\n", ret);
    goto exit;
  }

  // init context and model
  device_id = devs.devices[0].id;
  ret       = init_context_and_model(&ctx, model_path, weight_path, core_mask, key_path, device_id);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_init_model fail! ret = %d\n", ret);
    goto exit;
  }

  // query llm config
  ret = rknn3_query(ctx, RKNN3_QUERY_LLM_CONFIG, &llm_config, sizeof(rknn3_llm_config));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query llm config failed! ret=%d", ret);
    goto exit;
  }

  // get tokenizer and embedding
  ret = get_tokenizer_and_embedding(tokenizer_path, &vocab_info, &tokenizer, &embedding_info, embedding_path, &emb_st);
  if (ret < 0) {
    printf("get_tokenizer_and_embedding failed, ret = %d\n", ret);
    goto exit;
  }

  // Set basic session parameters
  build_default_llm_params(&params, &vocab_info, max_context_len, ignore_eos_token);

  // init session
  session_userdata.session = rknn3_session_init(ctx, &params, n_params);
  if (!session_userdata.session) {
    printf("rknn3_session_init failed, ret = %d\n", ret);
    goto exit;
  }

  // set session callback
  build_default_callback(&callback, tokenizer, &embedding_info, &session_userdata);

  ret = rknn3_session_set_callback(session_userdata.session, &callback);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_set_callback failed, ret = %d\n", ret);
    goto exit;
  }

  // set chat template and kvcache policy
  if (test_type == 0) {
    // set chat template for prompt input
    std::string system_prompt  = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n";
    std::string prompt_prefix  = "<|im_start|>user\n";
    std::string prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n";
    ret = rknn3_session_set_chat_template(session_userdata.session, system_prompt.c_str(), prompt_prefix.c_str(), prompt_postfix.c_str());
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_session_set_chat_template failed, ret = %d\n", ret);
      goto exit;
    }

    // set kvcache policy for prompt input
    kvcache_policy = RKNN3_KVCACHE_POLICY_RECURRENT;
    ret            = rknn3_session_set_kvcache_policy(session_userdata.session, kvcache_policy, nullptr);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_session_set_kvcache_policy failed, ret=%d\n", ret);
      goto exit;
    }
  } else if (test_type == 1 || test_type == 2) {
    // set kvcache policy for token/embed input, i.e. set system prompt in kvcache
    int32_t system_prompt_for_keep_len        = system_prompt_for_keep.length();
    int32_t system_prompt_for_keep_max_tokens = system_prompt_for_keep_len + 2;
    system_prompt_for_keep_tokens.resize(system_prompt_for_keep_max_tokens, 0);

    system_prompt_for_keep_n_tokens = default_tokenizer_callback(tokenizer, system_prompt_for_keep.c_str(), system_prompt_for_keep_len,
                                                                 system_prompt_for_keep_tokens.data(), system_prompt_for_keep_max_tokens);
    if (system_prompt_for_keep_n_tokens <= 0) {
      printf("get system_prompt_for_keep tokens failed\n");
      goto exit;
    }

    system_prompt_for_keep_n_tokens_aligned = ALIGNED_SIZE(system_prompt_for_keep_n_tokens, llm_config.kvcache_group_size);
    system_prompt_for_keep_tokens.resize(system_prompt_for_keep_n_tokens_aligned, 0);

    // set kvcache policy for token/embed input
    kvcache_policy                                = RKNN3_KVCACHE_POLICY_RECURRENT;
    kvcache_policy_param.recurrent.n_keep         = system_prompt_for_keep_n_tokens;
    kvcache_policy_param.recurrent.n_keep_aligned = system_prompt_for_keep_n_tokens_aligned;
    ret = rknn3_session_set_kvcache_policy(session_userdata.session, kvcache_policy, &kvcache_policy_param);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_session_set_kvcache_policy failed, ret=%d\n", ret);
      goto exit;
    }
  } else {
    printf("invalid test type\n");
    goto exit;
  }

  print_llm_config(&llm_config);

  // Test the token/embed input: each iteration builds a request in the selected input format,
  // runs one round of generation to show the standard prompt/token/embed input workflow.
  for (int i = 0; i < test_num; i++) {
    printf("\n--------------------Input[%d]-------------------- \n", i);

    llm_infer_param = build_default_infer_param(max_new_tokens, keep_history);

    std::string prompt = (test_type == 0) ? random_prompts[i % random_prompts.size()] : random_prompts2[i % random_prompts2.size()];

    // Tokenise the prompt once; TOKEN/EMBED paths prepend the system-prompt
    // tokens and reuse the resulting array.
    int32_t              prompt_len = prompt.length();
    int32_t              max_tokens = prompt_len + 2;
    std::vector<int32_t> tokens(max_tokens, 0);
    uint64_t             n_tokens = default_tokenizer_callback(tokenizer, prompt.c_str(), prompt_len, tokens.data(), max_tokens);
    if (n_tokens <= 0) {
      printf("get prompt tokens failed\n");
      goto exit;
    }
    tokens.resize(n_tokens);

    switch (test_type) {
    case 0: // prompt input:
      // session applies the chat template itself, so the system prompt is prepended automatically -- no manual splice.
      llm_inputs[0] = build_llm_input_prompt(prompt.c_str());
      break;
    case 1: { // token input
      // With keep_history=1 the system prompt already lives in the KV cache
      // from the first turn, so only prepend it on the first iteration (or
      // whenever history is discarded).
      if (keep_history != 1 || i == 0) {
        n_tokens += system_prompt_for_keep_n_tokens_aligned;
        tokens.insert(tokens.begin(), system_prompt_for_keep_tokens.begin(), system_prompt_for_keep_tokens.end());
      }
      llm_inputs[0] = build_llm_input_tokens(tokens.data(), n_tokens);
      break;
    }
    case 2: { // embed input
      if (keep_history != 1 || i == 0) {
        n_tokens += system_prompt_for_keep_n_tokens_aligned;
        tokens.insert(tokens.begin(), system_prompt_for_keep_tokens.begin(), system_prompt_for_keep_tokens.end());
      }
      embed = (float16*)malloc(n_tokens * embedding_info.embedding_dim * sizeof(float16));
      default_embed_callback(&embedding_info, tokens.data(), n_tokens, embed, n_tokens * embedding_info.embedding_dim * sizeof(float16));
      llm_inputs[0] = build_llm_input_embed(embed, n_tokens);
      break;
    }
    default:
      break;
    }

    printf("\n--------------------Output---------------------- \n");

    rknn3_test_timing_mark_start(&session_userdata.timing);
    ret = rknn3_session_run(session_userdata.session, llm_inputs[0], llm_n_inputs, &llm_infer_param);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_session_run failed, ret = %d\n", ret);
      goto exit;
    }
    rknn3_test_timing_mark_end(&session_userdata.timing);

    ret = rknn3_session_query_state(session_userdata.session, &state);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_session_query_state failed, ret=%d\n", ret);
      goto exit;
    }

    print_run_state(&state, true);

    // clear kvcache
    if (state.n_total_tokens >= (state.n_max_tokens - max_new_tokens)) {
      ret = rknn3_session_clear_kvcache(session_userdata.session, RKNN3_KVCACHE_CLEAR_ALL);
      if (ret != RKNN3_SUCCESS) {
        printf("rknn3_session_clear_kvcache failed, ret=%d\n", ret);
        goto exit;
      }
    }

    print_perf_stats(&state, &session_userdata.timing);

    for (int j = 0; j < llm_n_inputs; j++) {
      release_llm_input(ctx, llm_inputs[j]);
    }
    if (embed) {
      free(embed);
      embed = nullptr;
    }
  }

exit:
  for (int j = 0; j < llm_n_inputs; j++) {
    release_llm_input(ctx, llm_inputs[j]);
  }

  if (embed) {
    free(embed);
    embed = nullptr;
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

  printf("*******************************END TEST**********************************\n");

  fflush(stdout);

  return 0;
}
