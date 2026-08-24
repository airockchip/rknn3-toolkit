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
  // Usage: %s <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask>
  // <checkpoint_tail_overwrite> [keep_history] [ignore_eos_token] [key_path]
  if (argc < 9 || argc > 12) {
    LOGW(
      "Usage: %s <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask> <checkpoint_tail_overwrite> [keep_history] [ignore_eos_token] [key_path]\n ",
      argv[0]);
    LOGW(
      "Such as: %s ./model/Qwen2.5-0.5B.rknn ./model/Qwen2.5-0.5B.weight ./model/Qwen2.5-0.5B.tokenizer.gguf ./model/Qwen2.5-0.5B-embedding.bin 1024 256 0xff 0 0 0 ./model/key.env\n",
      argv[0]);
    return -1;
  }

  int         ret                       = 0;
  char*       model_path                = argv[1];
  char*       weight_path               = argv[2];
  const char* tokenizer_path            = argv[3];
  const char* embedding_path            = argv[4];
  int32_t     max_context_len           = atoi(argv[5]);
  int32_t     max_new_tokens            = atoi(argv[6]);
  uint32_t    core_mask                 = strtoul(argv[7], nullptr, 16);
  bool        checkpoint_tail_overwrite = atoi(argv[8]) != 0;
  int         keep_history              = (argc > 9) ? atoi(argv[9]) : 0;
  bool        ignore_eos_token          = (argc > 10) ? (atoi(argv[10]) != 0) : false;
  const char* key_path                  = (argc > 11) ? argv[11] : nullptr;

  const char* kvcache_file = "kvcache.bin";

  rknn3_devices                 devs;
  rknn3_context                 ctx = 0;
  VocabInfo                     vocab_info;
  Tokenizer*                    tokenizer = nullptr;
  struct stat                   emb_st;
  struct embedding_info         embedding_info;
  rknn3_llm_config              llm_config;
  RKLLMCallback                 callback;
  rknn3_test_session_userdata_t session_userdata;
  rknn3_llm_param               params;
  RKLLMRunState                 state;
  rknn3_lora                    loras_enabled[RKNN3_MAX_LORA_NUM];
  rknn3_kvcache_policy          kvcache_policy = RKNN3_KVCACHE_POLICY_SAVE_CHECKPOINT;
  rknn3_kvcache_policy_param    kvcache_policy_param;
  int                           n_params     = 1;
  int                           llm_n_inputs = 1;
  rknn3_llm_input*              llm_inputs[1];
  rknn3_llm_infer_param         llm_infer_param;

  memset(&devs, 0, sizeof(devs));
  memset(&vocab_info, 0, sizeof(vocab_info));
  memset(&emb_st, 0, sizeof(emb_st));
  memset(&embedding_info, 0, sizeof(embedding_info));
  memset(&llm_config, 0, sizeof(llm_config));
  memset(&callback, 0, sizeof(callback));
  memset(&params, 0, sizeof(params));
  memset(&state, 0, sizeof(state));
  memset(loras_enabled, 0, sizeof(loras_enabled));
  memset(&kvcache_policy_param, 0, sizeof(kvcache_policy_param));
  memset(llm_inputs, 0, sizeof(llm_inputs));
  memset(&llm_infer_param, 0, sizeof(llm_infer_param));
  memset(&session_userdata, 0, sizeof(session_userdata));
  state.loras_enabled = loras_enabled;

  kvcache_policy_param.save_checkpoint.checkpoint_start_pos      = 0;
  kvcache_policy_param.save_checkpoint.checkpoint_interval       = 128;
  kvcache_policy_param.save_checkpoint.max_checkpoint_count      = keep_history ? 100 : 3;
  kvcache_policy_param.save_checkpoint.checkpoint_tail_overwrite = checkpoint_tail_overwrite;

  std::vector<std::string> random_prompts = {
    "Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?\nLet's think step by step\nIn 2005, 60/2=30 kids came to the cookout.\nIn 2006, 30/3*2=20 kids came to the cookout.\nThe answer is 20\n\nQuestion: Zilla spent 7% of her monthly earnings on rent, half of it on her other monthly expenses, and put the rest in her savings. If she spent $133 on her rent, how much does she deposit into her savings account in a month?\nLet's think step by step\nSince $133 is equal to 7% of her earnings, then 1% is equal to $133/7 = $19.\nThe total monthly earning of Zilla is represented by 100%, so $19 x 100 = $1900 is her monthly earnings.\nSo, $1900/2 = $950 is spent on her other monthly expenses.\nThe total amount spent on the rent and other monthly expenses is $133 + $950 = $1083.\nHence, she saves $1900 - $1083 = $817 per month.\nThe answer is 817\n\nQuestion: If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?\nLet's think step by step\nThe total ratio representing the slices of pizza that Buzz bought is 5+8=13\nIf he shared the slices of pizza with the waiter, the waiter received a fraction of 8/13 of the total number of slices, which totals 8/13 * 78 = 48 slices\nTwenty less the number of slices of pizza that the waiter ate is 48-20 = 28\nThe answer is 28\n\nQuestion: Jame gets a raise to $20 per hour and works 40 hours a week.  His old job was $16 an hour for 25 hours per week.  How much more money does he make per year in his new job than the old job if he works 52 weeks a year?\nLet's think step by step\nHe makes 20*40=$800 per week\nHe used to make 16*25=$400 per week\nSo his raise was 800-400=$400 per week\nSo he makes 400*52=$20,800 per year more\nThe answer is 20800\nQuestion: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?\nLet's think step by step\n",
  };
  int test_num = 2;

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
    printf("init_context_and_model fail! ret = %d\n", ret);
    goto exit;
  }

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

  build_default_llm_params(&params, &vocab_info, max_context_len, ignore_eos_token);
  params.sampling_param.repeat_penalty = 1.1f;

  session_userdata.session = rknn3_session_init(ctx, &params, n_params);
  if (!session_userdata.session) {
    printf("rknn3_session_init failed, ret = %d\n", ret);
    goto exit;
  }

  build_default_callback(&callback, tokenizer, &embedding_info, &session_userdata);
  ret = rknn3_session_set_callback(session_userdata.session, &callback);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_set_callback failed, ret = %d\n", ret);
    goto exit;
  }

  // set kvcache policy
  ret = rknn3_session_set_kvcache_policy(session_userdata.session, kvcache_policy, &kvcache_policy_param);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_set_kvcache_policy failed, ret=%d\n", ret);
    goto exit;
  }

  print_llm_config(&llm_config);

  // Test KV-cache checkpoint save/load during inference: each iteration
  // builds a prompt request, runs one round of generation, and then saves the
  // checkpoint. The next iteration reloads the checkpoint and runs inference
  // again to verify that resumable inference works for checkpoint-capable models.
  for (int i = 0; i < 3; i++) {
    printf("\n--------------------Input[%d]-------------------- \n", i);

    llm_infer_param = build_default_infer_param(max_new_tokens, keep_history);

    std::string cur_prompt = random_prompts[0];
    llm_inputs[0]          = build_llm_input_prompt(cur_prompt.c_str());

    // Load checkpoint for iterations > 0
    if (i > 0) {
      ret = load_kvcache_from_path(session_userdata.session, kvcache_file);
      if (ret != RKNN3_SUCCESS) {
        goto exit;
      }
    }

    printf("\n--------------------Output---------------------- \n");

    rknn3_test_timing_mark_start(&session_userdata.timing);
    ret = rknn3_session_run(session_userdata.session, llm_inputs[0], llm_n_inputs, &llm_infer_param);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_session_run failed, ret = %d\n", ret);
      goto exit;
    }
    rknn3_test_timing_mark_end(&session_userdata.timing);

    // Save kvcache after each run
    ret = save_kvcache_to_path(session_userdata.session, kvcache_file);
    if (ret != RKNN3_SUCCESS) {
      goto exit;
    }

    ret = rknn3_session_query_state(session_userdata.session, &state);
    if (ret != RKNN3_SUCCESS) {
      printf("rknn3_session_query_state failed, ret=%d\n", ret);
      goto exit;
    }

    print_run_state(&state, true);

    print_perf_stats(&state, &session_userdata.timing);

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

  printf("*******************************END TEST**********************************\n");

  fflush(stdout);

  return 0;
}
