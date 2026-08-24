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

// Common helpers shared by the rknn3_session test demos.

#ifndef RKNN3_SESSION_TEST_COMMON_H
#define RKNN3_SESSION_TEST_COMMON_H

#include "Tokenizer.h"
#include "float16.h"
#include "rknn3_api.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <string>
#include <vector>

#define LOGW(fmt, ...) printf("\033[33m" fmt "\033[0m", ##__VA_ARGS__)
#define ALIGNED_SIZE(x, bytes) (((x) + (bytes - 1)) & (~(bytes - 1)))

/* ------------------------------------------------------------------ *
 * Timing helpers
 *
 * Timing state is per-session (not global) so that multi-session demos
 * don't stomp on each other's timestamps.
 * ------------------------------------------------------------------ */
typedef struct rknn3_test_timing
{
  struct timeval start;
  struct timeval first_token;
  struct timeval end;
  bool           first_decode;
} rknn3_test_timing_t;

/* Bundles the Tokenizer pointer and per-session timing state that the
 * result callback needs.  Passed as result_userdata. */
typedef struct rknn3_test_session_userdata
{
  Tokenizer*          tokenizer;
  rknn3_session*      session;
  rknn3_test_timing_t timing;
} rknn3_test_session_userdata_t;

void rknn3_test_timing_mark_start(rknn3_test_timing_t* t);
void rknn3_test_timing_mark_end(rknn3_test_timing_t* t);
void rknn3_test_timing_mark_first_token(rknn3_test_timing_t* t);

/* ------------------------------------------------------------------ *
 * Embedding table (mmap'd, read-only)
 * ------------------------------------------------------------------ */
struct embedding_info
{
  int      fd             = -1;
  float16* embedding_data = nullptr;
  int      embedding_dim  = 0;
  int      vocab_size     = 0;
};

void embedding_info_release(struct embedding_info* info, struct stat* emb_st);

/* ------------------------------------------------------------------ *
 * Math helpers
 * ------------------------------------------------------------------ */
int argmax_fp16(const float16* data, int size);

/* ------------------------------------------------------------------ *
 * Standard LLM callbacks
 * ------------------------------------------------------------------ */
int default_result_callback(void* userdata, RKLLMResult* result, LLMCallState state);
int default_tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max);
int default_embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len);
int default_sampling_callback(void* userdata, float16* logits, char* logits_name);
int default_output_callback(void* userdata, rknn3_tensor* output_tensors, uint32_t n_output_tensors, LLMOutputCallbackState state);

/* ------------------------------------------------------------------ *
 * Context + model init
 * ------------------------------------------------------------------ */
int init_context_and_model(rknn3_context* ctx_out, const char* model_path, const char* weight_path, uint32_t core_mask,
                           const char* key_path = nullptr, const char* device_id = nullptr);

/* ------------------------------------------------------------------ *
 * Tokenizer + embedding bootstrap
 * ------------------------------------------------------------------ */
int get_tokenizer(const char* tokenizer_path, VocabInfo* vocab_info, Tokenizer** tokenizer);
int get_embedding(struct embedding_info* embedding_info, const char* embedding_path, struct stat* emb_st, int vocab_size);
int get_tokenizer_and_embedding(const char* tokenizer_path, VocabInfo* vocab_info, Tokenizer** tokenizer,
                                struct embedding_info* embedding_info, const char* embedding_path, struct stat* emb_st);

/* ------------------------------------------------------------------ *
 * Default param / callback assembly
 * ------------------------------------------------------------------ */
void build_default_llm_params(rknn3_llm_param* params, const VocabInfo* vocab_info, int32_t max_context_len, bool ignore_eos_token = false);
void build_default_callback(RKLLMCallback* callback, Tokenizer* tokenizer, struct embedding_info* emb,
                            rknn3_test_session_userdata_t* session_userdata);

/* ------------------------------------------------------------------ *
 * LLM input construction
 * ------------------------------------------------------------------ */
rknn3_llm_infer_param build_default_infer_param(int32_t max_new_tokens, int keep_history = 0, bool prefill_only = false,
                                                bool disable_sampling = false);

rknn3_llm_input* build_llm_input_prompt(const char* prompt, bool enable_thinking = false, bool print_input = true);
rknn3_llm_input* build_llm_input_tokens(const int32_t* tokens, uint64_t n_tokens, bool enable_thinking = false, bool print_input = true);
rknn3_llm_input* build_llm_input_embed(float16* embed, uint64_t n_tokens, bool enable_thinking = false, bool print_input = true);
rknn3_llm_input* build_llm_input_multimodal_image(const char* prompt, const int32_t* tokens, uint64_t n_tokens,
                                                  bool enable_thinking = false, float16* image_embed = nullptr,
                                                  uint64_t n_image_tokens = 144, uint64_t n_image = 1,
                                                  const char* image_start = "<|vision_start|>", const char* image_end = "<|vision_end|>",
                                                  const char* image_content = "<|image_pad|>", uint64_t image_width = 320,
                                                  uint64_t image_height = 320, bool print_input = true);
rknn3_llm_input* build_llm_input_aux(rknn3_context ctx, int32_t core_id, uint64_t size, const void* data, bool print_input = true);
void             release_llm_input(rknn3_context ctx, rknn3_llm_input*& in);

/* ------------------------------------------------------------------ *
 * Pretty-printers
 * ------------------------------------------------------------------ */
void        dump_tensor_attr(const rknn3_tensor_attr* attr);
const char* rknn3_test_task_type_string(rknn3_llm_task_type t);
const char* rknn3_test_attention_type_string(rknn3_attention_type t);
void        print_llm_config(const rknn3_llm_config* cfg);
const char* rknn3_test_kvcache_policy_string(rknn3_kvcache_policy p);
void        print_run_state(const RKLLMRunState* state, bool print_tokens = false);
void        print_perf_stats(const RKLLMRunState* state, const rknn3_test_timing_t* t);

/* ------------------------------------------------------------------ *
 * Model input/output tensor attr query
 * ------------------------------------------------------------------ */
int  query_input_output_attrs(rknn3_context ctx, rknn3_tensor_attr* input_attrs, int max_n_input, rknn3_tensor_attr* output_attrs,
                              int max_n_output, rknn3_input_output_num* io_num_out = nullptr);
int  setup_output_tensors(rknn3_context ctx, rknn3_tensor* output_tensors, int n_output_tensors, const int* output_tensors_index,
                          rknn3_input_output_num* io_num_out = nullptr);
void cleanup_output_tensors(rknn3_context ctx, rknn3_tensor* output_tensors, int n_output_tensors);

/* ------------------------------------------------------------------ *
 * Binary file I/O for multimodal / token fixtures
 * ------------------------------------------------------------------ */
uint64_t read_data_from_bin(const char* bin_path, float16* data);
uint64_t read_data_from_bin(const char* bin_path, float* data);
uint64_t save_data_to_bin(const char* bin_path, const float16* data, uint64_t n_elements);
uint64_t read_tokens_from_bin(const char* bin_path, int32_t* tokens);

/* ------------------------------------------------------------------ *
 * KV cache save / load helpers
 * ------------------------------------------------------------------ */
int load_kvcache_from_data(rknn3_session* session, const char* path);
int save_kvcache_to_path(rknn3_session* session, const char* path);
int load_kvcache_from_path(rknn3_session* session, const char* path);

#endif /* RKNN3_SESSION_TEST_COMMON_H */
