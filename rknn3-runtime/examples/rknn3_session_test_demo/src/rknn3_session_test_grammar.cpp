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

// XGrammar evaluation demo: for each grammar type (JSON Schema / Builtin JSON /
// Regex / EBNF / Structural Tag) it runs the same prompt twice -- once with the grammar constraint
// and once without -- and reports (a) whether the grammar-constrained output
// actually satisfies the constraint (accuracy) and (b) the per-token decode
// overhead introduced by the grammar (performance). Shared boilerplate comes
// from rknn3_session_test_common.h.

#include "rknn3_session_test_common.h"
#include "json.hpp"

#include <regex.h>
#include <string>
#include <vector>

static rknn3_session* session       = nullptr;
static std::string    g_output_text;

// Result callback: appends decoded text to g_output_text and stamps the
// first-token timestamp (used for TTFT / TPOT).
static int eval_result_callback(void* userdata, RKLLMResult* result, LLMCallState state)
{
  rknn3_test_session_userdata_t* session_userdata = (rknn3_test_session_userdata_t*)userdata;
  Tokenizer*                     tokenizer        = session_userdata->tokenizer;
  if (state == RKLLM_RUN_NORMAL) {
    std::string piece;
    if (result->num_tokens == 1) {
      piece = tokenizer->TokenToPiece(result->token_ids[0]);
    } else {
      piece = tokenizer->Decode(result->token_ids, result->num_tokens);
    }
    g_output_text += piece;
    printf("%s", piece.c_str());
    fflush(stdout);
    rknn3_test_timing_mark_first_token(&session_userdata->timing);
  }
  return 0;
}

/* ----------------------------- validators ----------------------------- */

static bool parse_json(const std::string& text, nlohmann::json* parsed)
{
  if (!parsed) return false;
  size_t start = text.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return false;
  size_t end = text.find_last_not_of(" \t\n\r");
  std::string trimmed = text.substr(start, end - start + 1);

  if (trimmed.empty()) return false;
  *parsed = nlohmann::json::parse(trimmed, nullptr, false);
  return !parsed->is_discarded();
}

static bool is_valid_json_object(const std::string& text)
{
  nlohmann::json parsed;
  return parse_json(text, &parsed) && parsed.is_object();
}

static bool json_has_field(const std::string& text, const char* field)
{
  nlohmann::json parsed;
  return field && parse_json(text, &parsed) && parsed.is_object() && parsed.contains(field);
}

static bool match_regex(const std::string& text, const char* pattern)
{
  regex_t    re;
  regmatch_t match;
  if (regcomp(&re, pattern, REG_EXTENDED) != 0) return false;

  size_t start = text.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    regfree(&re);
    return false;
  }
  size_t end = text.find_last_not_of(" \t\n\r");
  std::string trimmed = text.substr(start, end - start + 1);

  bool ok = (regexec(&re, trimmed.c_str(), 1, &match, 0) == 0);
  if (ok) {
    ok = (match.rm_so == 0 && (size_t)match.rm_eo == trimmed.size());
  }
  regfree(&re);
  return ok;
}

static bool contains_only_words(const std::string& text, const std::vector<std::string>& words)
{
  size_t start = text.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return false;
  size_t end = text.find_last_not_of(" \t\n\r");
  std::string trimmed = text.substr(start, end - start + 1);
  for (const auto& w : words) {
    if (trimmed == w) return true;
  }
  return false;
}

static bool validate_json_has_name_age(const std::string& t)
{
  return is_valid_json_object(t) && json_has_field(t, "name") && json_has_field(t, "age");
}
static bool validate_json_value(const std::string& t)
{
  nlohmann::json parsed;
  return parse_json(t, &parsed);
}
static bool validate_yes_no(const std::string& t)       { return contains_only_words(t, {"yes", "no", "maybe"}); }
static bool validate_date(const std::string& t)         { return match_regex(t, "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"); }
static bool validate_uuid(const std::string& t)         { return match_regex(t, "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"); }
static bool validate_time(const std::string& t)         { return match_regex(t, "^[0-9]{2}:[0-9]{2}$"); }
static bool validate_json_has_title_author_pages(const std::string& t)
{
  return is_valid_json_object(t) && json_has_field(t, "title") && json_has_field(t, "author") &&
         json_has_field(t, "pages");
}
static bool validate_json_has_street_city_zip(const std::string& t)
{
  return is_valid_json_object(t) && json_has_field(t, "street") && json_has_field(t, "city") &&
         json_has_field(t, "zip");
}
static bool validate_json_has_name_price_tags(const std::string& t)
{
  return is_valid_json_object(t) && json_has_field(t, "name") && json_has_field(t, "price") &&
         json_has_field(t, "tags");
}
static bool validate_json_has_student_grade_courses(const std::string& t)
{
  return is_valid_json_object(t) && json_has_field(t, "student") && json_has_field(t, "grade") &&
         json_has_field(t, "courses");
}
static bool validate_qwen35_tool_call(const std::string& t)
{
  // Qwen3.5 / Qwen3-Coder: <tool_call>\n<function=name>\n{json}\n</function>\n</tool_call>
  return t.find("<tool_call>") != std::string::npos && t.find("<function=") != std::string::npos &&
         t.find("</function>") != std::string::npos && t.find("</tool_call>") != std::string::npos;
}
static bool validate_qwen3_tool_call(const std::string& t)
{
  // Qwen3: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
  return t.find("<tool_call>") != std::string::npos && t.find("\"name\"") != std::string::npos &&
         t.find("\"arguments\"") != std::string::npos && t.find("</tool_call>") != std::string::npos;
}

static bool extract_tag_content(const std::string& text, const char* begin, const char* end,
                                std::string* content)
{
  if (!begin || !end || !content) return false;

  size_t start = text.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return false;
  size_t finish = text.find_last_not_of(" \t\n\r");
  std::string trimmed = text.substr(start, finish - start + 1);

  size_t begin_len = strlen(begin);
  size_t end_len = strlen(end);
  if (trimmed.size() < begin_len + end_len || trimmed.compare(0, begin_len, begin) != 0 ||
      trimmed.compare(trimmed.size() - end_len, end_len, end) != 0) {
    return false;
  }

  *content = trimmed.substr(begin_len, trimmed.size() - begin_len - end_len);
  return true;
}

static bool validate_json_answer(const std::string& t)
{
  nlohmann::json parsed;
  return parse_json(t, &parsed) && parsed.is_object() && parsed.contains("answer") &&
         parsed["answer"].is_number_integer();
}

static bool validate_native_tag(const std::string& t)
{
  std::string content;
  if (!extract_tag_content(t, "<answer>", "</answer>", &content)) return false;

  nlohmann::json parsed;
  return parse_json(content, &parsed) && parsed.is_object() && parsed.contains("value") &&
         parsed["value"].is_number_integer();
}

static bool validate_triggered_tags(const std::string& t)
{
  std::string content;
  if (!extract_tag_content(t, "<function=get_weather>", "</function>", &content)) return false;

  nlohmann::json parsed;
  return parse_json(content, &parsed) && parsed.is_object() && parsed.contains("city") &&
         parsed["city"].is_string() && parsed["city"] == "Shanghai";
}

static bool validate_output(const std::string& text, bool enable_thinking,
                            bool (*validator)(const std::string&))
{
  if (!validator) return false;
  if (!enable_thinking) return validator(text);

  static const char* transition = "</think>\n\n";
  size_t transition_pos = text.find(transition);
  if (transition_pos == std::string::npos || transition_pos == 0) return false;

  size_t final_answer_pos = transition_pos + strlen(transition);
  if (text.find(transition, final_answer_pos) != std::string::npos) return false;
  return validator(text.substr(final_answer_pos));
}

/* --------------------------- perf measurement --------------------------- */

struct PerfMetrics {
  uint64_t n_prefill;
  uint64_t n_decode;
  double   ttft_ms;
  double   tpot_ms;
  double   decode_tps;
};

static void compute_perf(const RKLLMRunState* state, const rknn3_test_timing_t* timing, PerfMetrics* m)
{
  m->n_prefill = state->n_prefill_tokens;
  m->n_decode  = state->n_decode_tokens;

  double prefill_us = (timing->first_token.tv_sec - timing->start.tv_sec) * 1e6 +
                      (timing->first_token.tv_usec - timing->start.tv_usec);
  double decode_us = (timing->end.tv_sec - timing->first_token.tv_sec) * 1e6 +
                     (timing->end.tv_usec - timing->first_token.tv_usec);

  m->ttft_ms   = prefill_us / 1e3;
  m->tpot_ms   = m->n_decode == 0 ? 0.0 : (decode_us / 1e3) / m->n_decode;
  m->decode_tps = m->n_decode == 0 ? 0.0 : m->n_decode / (decode_us / 1e6);
}

/* Run the prompt, measure timing, and capture the output text. */
static int run_measured(rknn3_context ctx, rknn3_session* session, const char* prompt, int max_new_tokens,
                        bool enable_thinking, RKLLMRunState* out_state, rknn3_test_timing_t* timing)
{
  g_output_text.clear();

  rknn3_llm_infer_param infer = build_default_infer_param(max_new_tokens, 0);
  rknn3_llm_input*      input = build_llm_input_prompt(prompt, enable_thinking, false);

  rknn3_test_timing_mark_start(timing);
  int ret = rknn3_session_run(session, input, 1, &infer);
  rknn3_test_timing_mark_end(timing);
  release_llm_input(ctx, input);
  if (ret != RKNN3_SUCCESS) {
    return ret;
  }
  return rknn3_session_query_state(session, out_state);
}

/* ----------------------------- test driver ----------------------------- */

struct EvalCase {
  const char*         name;
  const char*         prompt;
  const char*         grammar_str;
  rknn3_grammar_type  grammar_type;
  bool                (*validator)(const std::string&);
};

/*
 * This SDK variant has no separate thinking-grammar API. Build one complete
 * Structural Tag for the generated stream:
 *   reasoning ... </think>\n\n + final format
 */
static bool build_thinking_grammar(const EvalCase& eval_case, std::string* grammar_str)
{
  if (!grammar_str) return false;

  nlohmann::json final_format;
  switch (eval_case.grammar_type) {
    case RKNN3_GRAMMAR_JSON_SCHEMA: {
      if (!eval_case.grammar_str) return false;
      nlohmann::json schema = nlohmann::json::parse(eval_case.grammar_str, nullptr, false);
      if (schema.is_discarded()) return false;
      final_format["type"] = "json_schema";
      final_format["json_schema"] = schema;
      break;
    }
    case RKNN3_GRAMMAR_EBNF:
      if (!eval_case.grammar_str) return false;
      final_format["type"] = "grammar";
      final_format["grammar"] = eval_case.grammar_str;
      break;
    case RKNN3_GRAMMAR_BUILTIN_JSON:
      final_format["type"] = "json_schema";
      final_format["json_schema"] = nlohmann::json::object();
      break;
    case RKNN3_GRAMMAR_REGEX:
      if (!eval_case.grammar_str) return false;
      final_format["type"] = "regex";
      final_format["pattern"] = eval_case.grammar_str;
      break;
    case RKNN3_GRAMMAR_STRUCTURAL_TAG: {
      if (!eval_case.grammar_str) return false;
      nlohmann::json structural_tag = nlohmann::json::parse(eval_case.grammar_str, nullptr, false);
      if (structural_tag.is_discarded() || !structural_tag.is_object() ||
          structural_tag.value("type", "") != "structural_tag" || !structural_tag.contains("format")) {
        return false;
      }
      final_format = structural_tag["format"];
      break;
    }
    default:
      return false;
  }

  nlohmann::json thinking_format;
  thinking_format["type"] = "tag";
  thinking_format["begin"] = "";
  thinking_format["content"] = {{"type", "any_text"}};
  thinking_format["end"] = "</think>";

  nlohmann::json separator_format;
  separator_format["type"] = "const_string";
  separator_format["value"] = "\n\n";

  nlohmann::json sequence_format;
  sequence_format["type"] = "sequence";
  sequence_format["elements"] = nlohmann::json::array();
  sequence_format["elements"].push_back(thinking_format);
  sequence_format["elements"].push_back(separator_format);
  sequence_format["elements"].push_back(final_format);

  nlohmann::json structural_tag;
  structural_tag["type"] = "structural_tag";
  structural_tag["format"] = sequence_format;
  *grammar_str = structural_tag.dump();
  return true;
}

int main(int argc, char** argv)
{
  if (argc != 8) {
    LOGW("Usage: %s <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_token> <core_mask>", argv[0]);
    LOGW("Such as: %s ./model/Qwen2.5-0.5B.rknn ./model/Qwen2.5-0.5B.weight ./model/Qwen2.5-0.5B.tokenizer.gguf "
         "./model/Qwen2.5-0.5B-embedding.bin 1024 256 0xff", argv[0]);
    return -1;
  }

  const char* model_path      = argv[1];
  const char* weight_path     = argv[2];
  const char* tokenizer_path  = argv[3];
  const char* embedding_path  = argv[4];
  int32_t     max_context_len = atoi(argv[5]);
  int32_t     max_new_token   = atoi(argv[6]);
  uint32_t    core_mask       = strtoul(argv[7], nullptr, 16);

  int                  ret = 0;
  bool                 had_failure = false;
  rknn3_context          ctx = 0;
  VocabInfo              vocab_info;
  Tokenizer*             tokenizer = nullptr;
  struct stat            emb_st;
  struct embedding_info  embedding_info;
  RKLLMCallback          callback;
  rknn3_test_session_userdata_t session_userdata;
  rknn3_llm_param        params;
  RKLLMRunState          state;
  rknn3_llm_config       llm_config;

  memset(&vocab_info, 0, sizeof(vocab_info));
  memset(&emb_st, 0, sizeof(emb_st));
  memset(&embedding_info, 0, sizeof(embedding_info));
  memset(&callback, 0, sizeof(callback));
  memset(&session_userdata, 0, sizeof(session_userdata));
  memset(&params, 0, sizeof(params));
  memset(&state, 0, sizeof(state));
  memset(&llm_config, 0, sizeof(llm_config));

  printf("*******************************XGRAMMAR EVAL*******************************\n");

  ret = init_context_and_model(&ctx, model_path, weight_path, core_mask);
  if (ret != RKNN3_SUCCESS) {
    printf("init_context_and_model fail! ret=%d\n", ret);
    goto exit;
  }

  ret = get_tokenizer_and_embedding(tokenizer_path, &vocab_info, &tokenizer, &embedding_info, embedding_path, &emb_st);
  if (ret < 0) {
    printf("get_tokenizer_and_embedding fail! ret=%d\n", ret);
    goto exit;
  }

  ret = rknn3_query(ctx, RKNN3_QUERY_LLM_CONFIG, &llm_config, sizeof(rknn3_llm_config));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query llm config failed! ret=%d\n", ret);
    goto exit;
  }

  build_default_llm_params(&params, &vocab_info, max_context_len, /*ignore_eos_token=*/false);

  session_userdata.session = rknn3_session_init(ctx, &params, 1);
  if (!session_userdata.session) {
    printf("rknn3_session_init failed!\n");
    ret = -1;
    goto exit;
  }
  session = session_userdata.session;

  build_default_callback(&callback, tokenizer, &embedding_info, &session_userdata);
  callback.result_callback = eval_result_callback;

  ret = rknn3_session_set_callback(session, &callback);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_set_callback failed, ret=%d\n", ret);
    goto exit;
  }

  // Build the encoded-vocab blob and initialize the Host-side grammar engine.
  {
    int vocab_size = vocab_info.vocab_size;
    std::vector<uint8_t> vocab_blob;
    vocab_blob.reserve(vocab_size * 8);
    for (int i = 0; i < vocab_size; i++) {
      std::string piece = tokenizer->TokenToPiece(i);
      int32_t     len   = (int32_t)piece.size();
      vocab_blob.insert(vocab_blob.end(), (uint8_t*)&len, (uint8_t*)&len + sizeof(int32_t));
      vocab_blob.insert(vocab_blob.end(), (uint8_t*)piece.data(), (uint8_t*)piece.data() + piece.size());
    }
    std::vector<int32_t> stop_tokens;
    for (int i = 0; i < vocab_info.n_special_eos_id; i++) {
      stop_tokens.push_back(vocab_info.special_eos_id[i]);
    }

    rknn3_grammar_config grammar_config = {};
    grammar_config.vocab_size          = vocab_size;
    grammar_config.encoded_vocab       = vocab_blob.data();
    grammar_config.encoded_vocab_len   = (int32_t)vocab_blob.size();
    grammar_config.n_stop_tokens       = (int32_t)stop_tokens.size();
    grammar_config.stop_tokens         = stop_tokens.data();

    ret = rknn3_init_grammar(ctx, &grammar_config);
    if (ret != 0) {
      printf("rknn3_init_grammar failed, ret=%d\n", ret);
      goto exit;
    }
    printf("rknn3_init_grammar OK (vocab_size=%d)\n", vocab_size);
  }

  {
  const char* json_schema_str =
      "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},"
      "\"age\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":150}},"
      "\"required\":[\"name\",\"age\"]}";
  const char* book_schema_str =
      "{\"type\":\"object\",\"properties\":{\"title\":{\"type\":\"string\"},"
      "\"author\":{\"type\":\"string\"},\"pages\":{\"type\":\"integer\",\"minimum\":1}},"
      "\"required\":[\"title\",\"author\",\"pages\"]}";
  const char* address_schema_str =
      "{\"type\":\"object\",\"properties\":{\"street\":{\"type\":\"string\"},"
      "\"city\":{\"type\":\"string\"},\"zip\":{\"type\":\"string\"}},"
      "\"required\":[\"street\",\"city\",\"zip\"]}";
  const char* product_schema_str =
      "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},"
      "\"price\":{\"type\":\"number\",\"minimum\":0},"
      "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},"
      "\"required\":[\"name\",\"price\",\"tags\"]}";
  const char* student_schema_str =
      "{\"type\":\"object\",\"properties\":{\"student\":{\"type\":\"object\"},"
      "\"grade\":{\"type\":\"integer\"},"
      "\"courses\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},"
      "\"required\":[\"student\",\"grade\",\"courses\"]}";
  const char* date_regex_str  = "[0-9]{4}-[0-9]{2}-[0-9]{2}";
  const char* uuid_regex_str  = "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}";
  const char* time_regex_str  = "[0-9]{2}:[0-9]{2}";
  // Qwen3.5 / Qwen3-Coder XML tool call:
  // <tool_call>\n<function=name>\n{json}\n</function>\n</tool_call>
  const char* structural_tag_str =
      "{\"type\":\"structural_tag\",\"format\":{\"type\":\"sequence\",\"elements\":["
      "{\"type\":\"const_string\",\"value\":\"<tool_call>\\n<function=\"},"
      "{\"type\":\"regex\",\"pattern\":\"[A-Za-z0-9_-]{1,64}\"},"
      "{\"type\":\"const_string\",\"value\":\">\\n\"},"
      "{\"type\":\"const_string\",\"value\":\"<parameter=location>\\nShanghai\\n</parameter>\"},"
      "{\"type\":\"const_string\",\"value\":\"\\n</function>\\n</tool_call>\"}"
      "]}}";
  // Qwen3 JSON tool call:
  // <tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "Shanghai"}}\n</tool_call>
  const char* qwen3_triggered_tags_str =
      "{\"type\":\"structural_tag\",\"format\":{\"type\":\"triggered_tags\","
      "\"triggers\":[\"<tool_call>\\n\"],\"tags\":[{\"type\":\"tag\","
      "\"begin\":\"<tool_call>\\n{\\\"name\\\": \\\"get_current_temperature\\\", \\\"arguments\\\": \","
      "\"content\":{\"type\":\"json_schema\",\"json_schema\":{\"type\":\"object\","
      "\"additionalProperties\":true}},"
      "\"end\":\"}\\n</tool_call>\"}],\"at_least_one\":true,\"stop_after_first\":true}}";
  const char* native_tag_str =
      "{\"type\":\"structural_tag\",\"format\":{\"type\":\"tag\","
      "\"begin\":\"<answer>\","
      "\"content\":{\"type\":\"json_schema\",\"json_schema\":{\"type\":\"object\","
      "\"properties\":{\"value\":{\"type\":\"integer\"}},\"required\":[\"value\"],"
      "\"additionalProperties\":false}},\"end\":\"</answer>\"}}";
  const char* triggered_tags_str =
      "{\"type\":\"structural_tag\",\"format\":{\"type\":\"triggered_tags\","
      "\"triggers\":[\"<function=\"],\"tags\":[{\"type\":\"tag\","
      "\"begin\":\"<function=get_weather>\","
      "\"content\":{\"type\":\"json_schema\",\"json_schema\":{\"type\":\"object\","
      "\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],"
      "\"additionalProperties\":false}},\"end\":\"</function>\"}],"
      "\"at_least_one\":true,\"stop_after_first\":true}}";
  const char* thinking_answer_schema_str =
      "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"integer\"}},"
      "\"required\":[\"answer\"],\"additionalProperties\":false}";
  const char* ebnf_str  = "root ::= answer\nanswer ::= \"yes\" | \"no\" | \"maybe\"\n";

  EvalCase cases[] = {
    // --- JSON Schema ---
    {"JSON Schema (name+age)", "Generate a JSON with name and age fields. Reply only with JSON.", json_schema_str,
     RKNN3_GRAMMAR_JSON_SCHEMA, validate_json_has_name_age},
    {"JSON Schema (book)", "Generate a JSON object describing a book with title, author, and pages. Reply only with JSON.",
     book_schema_str, RKNN3_GRAMMAR_JSON_SCHEMA, validate_json_has_title_author_pages},
    {"JSON Schema (address)", "Generate a JSON object with street, city, and zip code. Reply only with JSON.",
     address_schema_str, RKNN3_GRAMMAR_JSON_SCHEMA, validate_json_has_street_city_zip},
    {"JSON Schema (product)", "Generate a JSON object for a product with name, price, and tags array. Reply only with JSON.",
     product_schema_str, RKNN3_GRAMMAR_JSON_SCHEMA, validate_json_has_name_price_tags},
    {"JSON Schema (student)", "Generate a JSON object with a student, grade, and courses array. Reply only with JSON.",
     student_schema_str, RKNN3_GRAMMAR_JSON_SCHEMA, validate_json_has_student_grade_courses},
    {"Builtin JSON (any valid JSON)", "Generate any valid JSON value. Reply only with JSON.", nullptr,
     RKNN3_GRAMMAR_BUILTIN_JSON, validate_json_value},
    {"JSON Schema (answer)",
     "What is 2 + 2? Think briefly, then reply with JSON containing integer field answer.",
     thinking_answer_schema_str, RKNN3_GRAMMAR_JSON_SCHEMA, validate_json_answer},
    // --- Regex ---
    {"Regex (date)", "Reply with a date in YYYY-MM-DD format.", date_regex_str, RKNN3_GRAMMAR_REGEX, validate_date},
    {"Regex (UUID)", "Reply with a UUID.", uuid_regex_str, RKNN3_GRAMMAR_REGEX, validate_uuid},
    {"Regex (time)", "Reply with a time in HH:MM format.", time_regex_str, RKNN3_GRAMMAR_REGEX, validate_time},
    // --- EBNF ---
    {"EBNF (yes/no/maybe)", "Answer yes, no, or maybe.", ebnf_str, RKNN3_GRAMMAR_EBNF, validate_yes_no},
    // --- Structural Tag ---
    {"Structural Tag (Native XML)", "Call the get_weather function for Shanghai. Reply only with the function call.",
     structural_tag_str, RKNN3_GRAMMAR_STRUCTURAL_TAG, validate_qwen35_tool_call},
    {"Structural Tag (JSON-in-XML / Hermes, triggered_tags)",
     "Call get_current_temperature with location=Shanghai. Reply only with the tool call.",
     qwen3_triggered_tags_str, RKNN3_GRAMMAR_STRUCTURAL_TAG, validate_qwen3_tool_call},
    {"Structural Tag (tag)", "Reply exactly as <answer>{\"value\":4}</answer>.",
     native_tag_str, RKNN3_GRAMMAR_STRUCTURAL_TAG, validate_native_tag},
    {"Structural Tag (triggered_tags)",
     "Call get_weather for Shanghai. Reply exactly with <function=get_weather>{\"city\":\"Shanghai\"}</function>.",
     triggered_tags_str, RKNN3_GRAMMAR_STRUCTURAL_TAG, validate_triggered_tags},
  };
  int n_cases = sizeof(cases) / sizeof(cases[0]);

  printf("\n==================== Summary ====================\n");
  printf("%-48s | %-8s | %-8s | %-9s | %-9s | %-10s | %-10s | %-9s\n", "Case", "Mode", "Acc", "n_dec",
         "TTFT(ms)", "TPOT(ms)", "TPOT(ms) base", "overhead");
  printf("--------------------------------------------------------------------------------------------------------------------------\n");

  int n_acc_pass = 0;
  int n_runs = 0;
  const bool thinking_modes[] = {false, true};

  for (int i = 0; i < n_cases; i++) {
    EvalCase& c = cases[i];
    printf("\n==================== Case %d: %s ====================\n", i + 1, c.name);

    for (bool enable_thinking : thinking_modes) {
      const char* mode = enable_thinking ? "THINKING" : "DIRECT";
      n_runs++;
      printf("\n-------------------- Mode: %s --------------------\n", mode);

      std::string effective_grammar_storage;
      const char* effective_grammar_str = c.grammar_str;
      rknn3_grammar_type effective_grammar_type = c.grammar_type;
      if (enable_thinking) {
        if (!build_thinking_grammar(c, &effective_grammar_storage)) {
          printf("[FAIL][%s] failed to build the complete thinking grammar\n", mode);
          had_failure = true;
          rknn3_session_clear_grammar(session);
          rknn3_session_clear_kvcache(session, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
          continue;
        }
        effective_grammar_str = effective_grammar_storage.c_str();
        effective_grammar_type = RKNN3_GRAMMAR_STRUCTURAL_TAG;
      }

      // ---- grammar-constrained run ----
      ret = rknn3_session_set_grammar(session, effective_grammar_str, effective_grammar_type, NULL);
      if (ret != RKNN3_SUCCESS) {
        printf("[FAIL][%s] set_grammar failed, ret=%d\n", mode, ret);
        had_failure = true;
        rknn3_session_clear_grammar(session);
        rknn3_session_clear_kvcache(session, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
        continue;
      }

      printf("[Grammar][%s] Prompt: %s\n", mode, c.prompt);
      printf("[Grammar][%s] Output: ", mode);
      ret = run_measured(ctx, session, c.prompt, max_new_token, enable_thinking, &state,
                         &session_userdata.timing);
      printf("\n");
      if (ret != RKNN3_SUCCESS) {
        printf("[FAIL][%s] grammar run failed, ret=%d\n", mode, ret);
        had_failure = true;
        rknn3_session_clear_grammar(session);
        rknn3_session_clear_kvcache(session, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
        continue;
      }

      PerfMetrics gm;
      compute_perf(&state, &session_userdata.timing, &gm);
      bool valid = validate_output(g_output_text, enable_thinking, c.validator);
      printf("[Grammar][%s] Output text: \"%s\"\n", mode, g_output_text.c_str());
      printf("[Grammar][%s] %s: n_decode=%llu, TTFT=%.2fms, TPOT=%.2fms/token (%.1f tok/s)\n",
             mode, valid ? "PASS" : "FAIL", (unsigned long long)gm.n_decode, gm.ttft_ms, gm.tpot_ms,
             gm.decode_tps);
      if (valid) n_acc_pass++;
      else had_failure = true;

      rknn3_session_clear_grammar(session);
      rknn3_session_clear_kvcache(session, RKNN3_KVCACHE_CLEAR_ALL, nullptr);

      // ---- baseline (no grammar), using the same thinking mode ----
      printf("[Baseline][%s] Prompt: %s\n", mode, c.prompt);
      printf("[Baseline][%s] Output: ", mode);
      ret = run_measured(ctx, session, c.prompt, max_new_token, enable_thinking, &state,
                         &session_userdata.timing);
      printf("\n");
      if (ret != RKNN3_SUCCESS) {
        printf("[WARN][%s] baseline run failed, ret=%d\n", mode, ret);
        had_failure = true;
        rknn3_session_clear_kvcache(session, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
        continue;
      }

      PerfMetrics bm;
      compute_perf(&state, &session_userdata.timing, &bm);
      printf("[Baseline][%s] Output text: \"%s\"\n", mode, g_output_text.c_str());
      printf("[Baseline][%s] n_decode=%llu, TTFT=%.2fms, TPOT=%.2fms/token (%.1f tok/s)\n",
             mode, (unsigned long long)bm.n_decode, bm.ttft_ms, bm.tpot_ms, bm.decode_tps);

      double overhead_ms = gm.tpot_ms - bm.tpot_ms;
      double overhead_pct = bm.tpot_ms > 0 ? (overhead_ms / bm.tpot_ms) * 100.0 : 0.0;

      printf("%-48s | %-8s | %-8s | %-9llu | %-9.2f | %-10.2f | %-10.2f | %+8.2fms (%+5.1f%%)\n",
             c.name, mode, valid ? "PASS" : "FAIL", (unsigned long long)gm.n_decode, gm.ttft_ms,
             gm.tpot_ms, bm.tpot_ms, overhead_ms, overhead_pct);

      rknn3_session_clear_kvcache(session, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
    }
  }

  printf("\n==================== Result ====================\n");
  int expected_runs = n_cases * (int)(sizeof(thinking_modes) / sizeof(thinking_modes[0]));
  printf("Accuracy: %d/%d runs passed (%d cases x 2 modes)\n", n_acc_pass, expected_runs, n_cases);
  if (n_runs != expected_runs || n_acc_pass != expected_runs) had_failure = true;
  }

exit:
  if (session) {
    rknn3_session_destroy(session);
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
  return (ret == RKNN3_SUCCESS && !had_failure) ? 0 : 1;
}
