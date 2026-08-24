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
#include <json.hpp>

#include <regex>

static std::string model_answer;
static bool        tool_call = false;

typedef struct
{
  double      temperature;
  std::string location;
  std::string date;
  std::string unit;
} TemperatureResult;

static TemperatureResult get_current_temperature(const char* location, const char* unit)
{
  TemperatureResult r;
  r.temperature = 26.1;
  r.location    = location;
  r.unit        = (unit && strlen(unit) > 0) ? unit : "celsius";
  r.date        = "";
  return r;
}

static TemperatureResult get_temperature_date(const char* location, const char* date, const char* unit)
{
  TemperatureResult r;
  r.temperature = 25.9;
  r.location    = location;
  r.unit        = (unit && strlen(unit) > 0) ? unit : "celsius";
  r.date        = date;
  return r;
}

static void* get_function_by_name(const char* name)
{
  if (strcmp(name, "get_current_temperature") == 0)
    return (void*)get_current_temperature;
  if (strcmp(name, "get_temperature_date") == 0)
    return (void*)get_temperature_date;
  return NULL;
}

static std::vector<std::string> extract_tool_call_list(const std::string& answer)
{
  std::vector<std::string> list;
  std::string::size_type   pos = 0;
  while (true) {
    std::string::size_type s = answer.find("<tool_call>", pos);
    if (s == std::string::npos)
      break;
    s += std::string("<tool_call>").length();
    std::string::size_type e = answer.find("</tool_call>", s);
    if (e == std::string::npos)
      break;
    std::string block = answer.substr(s, e - s);
    while (!block.empty() && (block[0] == '\n' || block[0] == '\r' || block[0] == ' '))
      block.erase(0, 1);
    while (!block.empty() && (block.back() == '\n' || block.back() == '\r' || block.back() == ' '))
      block.pop_back();
    list.push_back(block);
    pos = e + std::string("</tool_call>").length();
  }
  return list;
}

static std::string get_tool_call_results_json(const std::vector<std::string>& tool_call_list)
{
  std::vector<std::string> json_results;
  for (const auto& tc : tool_call_list) {
    std::string func_name, location, date, unit;
    std::smatch m;
    std::regex_search(tc, m, std::regex("\"name\"\\s*:\\s*\"([^\"]*)\""));
    if (m.size() > 1)
      func_name = m[1].str();
    std::regex_search(tc, m, std::regex("\"location\"\\s*:\\s*\"([^\"]*)\""));
    if (m.size() > 1)
      location = m[1].str();
    std::regex_search(tc, m, std::regex("\"date\"\\s*:\\s*\"([^\"]*)\""));
    if (m.size() > 1)
      date = m[1].str();
    std::regex_search(tc, m, std::regex("\"unit\"\\s*:\\s*\"([^\"]*)\""));
    if (m.size() > 1)
      unit = m[1].str();

    typedef TemperatureResult (*TempFuncType)(const char*, const char*, const char*);
    void* func_ptr = get_function_by_name(func_name.c_str());
    if (func_ptr) {
      TemperatureResult      result = ((TempFuncType)func_ptr)(location.c_str(), date.c_str(), unit.c_str());
      nlohmann::ordered_json j;
      j["temperature"] = result.temperature;
      if (!result.location.empty())
        j["location"] = result.location;
      if (!result.date.empty())
        j["date"] = result.date;
      if (!result.unit.empty())
        j["unit"] = result.unit;
      json_results.push_back(j.dump());
    } else {
      printf("[ToolCall] Function '%s' not found!\n", func_name.c_str());
    }
  }
  std::string result = "[";
  for (size_t i = 0; i < json_results.size(); ++i) {
    result += json_results[i];
    if (i + 1 != json_results.size())
      result += ",";
  }
  result += "]";
  return result;
}

static int fc_result_callback(void* userdata, RKLLMResult* result, LLMCallState state)
{
  rknn3_test_session_userdata_t* session_userdata = (rknn3_test_session_userdata_t*)userdata;
  Tokenizer*                     tokenizer        = session_userdata->tokenizer;
  switch (state) {
  case RKLLM_RUN_NORMAL: {
    std::string piece;
    if (result->num_tokens == 1)
      piece = tokenizer->TokenToPiece(result->token_ids[0]);
    else
      piece = tokenizer->Decode(result->token_ids, result->num_tokens);
    if (tool_call)
      printf("%s", piece.c_str());
    model_answer += piece;
    rknn3_test_timing_mark_first_token(&session_userdata->timing);
    fflush(stdout);
    break;
  }
  case RKLLM_RUN_FINISH:
    if (tool_call)
      printf("\n\n--------------------Finished-------------------- \n");
    fflush(stdout);
    break;
  case RKLLM_RUN_MAX_NEW_TOKEN_REACHED:
    if (tool_call)
      printf("\n\n--------------Max new token reached------------- \n");
    fflush(stdout);
    break;
  case RKLLM_RUN_STOP:
    if (tool_call)
      printf("\n\n-----------------------Stop--------------------- \n");
    fflush(stdout);
    break;
  default:
    default_result_callback(userdata, result, state);
    break;
  }
  return 0;
}

int main(int argc, char** argv)
{
  if (argc < 8 || argc > 11) {
    LOGW(
      "Usage: %s <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask> [keep_history] [ignore_eos_token] [key_path]\n ",
      argv[0]);
    LOGW(
      "Such as: %s ./model/Qwen2.5-0.5B.rknn ./model/Qwen2.5-0.5B.weight ./model/Qwen2.5-0.5B.tokenizer.gguf ./model/Qwen2.5-0.5B-embedding.bin 1024 256 0xff 0 0 ./model/key.env\n",
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
  int         keep_history     = (argc > 8) ? atoi(argv[8]) : 0;
  bool        ignore_eos_token = (argc > 9) ? (atoi(argv[9]) != 0) : false;
  const char* key_path         = (argc > 10) ? argv[10] : nullptr;

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

  std::string function_tools =
    R"([{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, State, Country\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \"celsius\"."}}, "required": ["location"]}}},{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, State, Country\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \"Year-Month-Day\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \"celsius\"."}}, "required": ["location", "date"]}}}])";

  std::vector<std::string> random_prompts = {
    "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30.",
    "What's the temperature in Shanghai now? How about tomorrow? Current Date: 2024-09-30.",
  };
  int test_num = random_prompts.size();

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

  // get tokenizer and embedding
  ret = get_tokenizer_and_embedding(tokenizer_path, &vocab_info, &tokenizer, &embedding_info, embedding_path, &emb_st);
  if (ret < 0) {
    printf("get_tokenizer_and_embedding failed, ret = %d\n", ret);
    goto exit;
  }

  ret = rknn3_query(ctx, RKNN3_QUERY_LLM_CONFIG, &llm_config, sizeof(rknn3_llm_config));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query llm config failed! ret=%d", ret);
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

  // set session callback: start from defaults, then override result_callback
  // with the fc variant that accumulates model_answer.
  build_default_callback(&callback, tokenizer, &embedding_info, &session_userdata);
  callback.result_callback = fc_result_callback;

  ret = rknn3_session_set_callback(session_userdata.session, &callback);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_set_callback failed, ret = %d\n", ret);
    goto exit;
  }

  // set kvcache policy
  ret = rknn3_session_set_kvcache_policy(session_userdata.session, kvcache_policy, nullptr);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_set_kvcache_policy failed, ret=%d\n", ret);
    goto exit;
  }

  ret = rknn3_session_set_function_tools(session_userdata.session, function_tools.c_str());
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_set_function_tools failed, ret = %d\n", ret);
    goto exit;
  }

  print_llm_config(&llm_config);

  // Test a practical function calling scenario: each iteration first runs the
  // user prompt to collect tool calls, then feeds back the tool result so the
  // model can finish the response.
  for (int i = 0; i < test_num; i++) {
    printf("\n--------------------Input[%d]-------------------- \n", i);

    // ---- Run 1: user prompt (role="user") ----
    std::string cur_prompt = random_prompts[i % random_prompts.size()];

    llm_infer_param     = build_default_infer_param(max_new_tokens, keep_history);
    llm_inputs[0]       = build_llm_input_prompt(cur_prompt.c_str());
    llm_inputs[0]->role = "user";

    printf("\n--------------------Output---------------------- \n");

    rknn3_test_timing_mark_start(&session_userdata.timing);
    model_answer = "";
    tool_call    = false;
    ret          = rknn3_session_run(session_userdata.session, llm_inputs[0], llm_n_inputs, &llm_infer_param);
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

    if (state.n_total_tokens >= (state.n_max_tokens - max_new_tokens)) {
      ret = rknn3_session_clear_kvcache(session_userdata.session, RKNN3_KVCACHE_CLEAR_ALL);
      if (ret != RKNN3_SUCCESS) {
        printf("rknn3_session_clear_kvcache failed, ret=%d\n", ret);
        goto exit;
      }
    }

    release_llm_input(ctx, llm_inputs[0]);

    // ---- Run 2: tool result (role="tool") ----
    std::vector<std::string> tool_call_list = extract_tool_call_list(model_answer);
    cur_prompt                              = get_tool_call_results_json(tool_call_list);

    llm_infer_param     = build_default_infer_param(max_new_tokens, keep_history);
    llm_inputs[0]       = build_llm_input_prompt(cur_prompt.c_str(), false, false);
    llm_inputs[0]->role = "tool";

    rknn3_test_timing_mark_start(&session_userdata.timing);
    model_answer = "";
    tool_call    = true;
    ret          = rknn3_session_run(session_userdata.session, llm_inputs[0], llm_n_inputs, &llm_infer_param);
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

    if (state.n_total_tokens >= (state.n_max_tokens - max_new_tokens)) {
      ret = rknn3_session_clear_kvcache(session_userdata.session, RKNN3_KVCACHE_CLEAR_ALL);
      if (ret != RKNN3_SUCCESS) {
        printf("rknn3_session_clear_kvcache failed, ret=%d\n", ret);
        goto exit;
      }
    }

    print_perf_stats(&state, &session_userdata.timing);
    release_llm_input(ctx, llm_inputs[0]);

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
