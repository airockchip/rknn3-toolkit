// Shared inference engine for the rknn3 LLM demos.
//
// One SessionState `g_state` holds every runtime object (context / session /
// tokenizer / embedding / lora / rope) plus the run configuration. The two
// front-ends share the whole engine (init, callbacks, run_turn, benchmark,
// status printing) and only differ in the decode strategy:
//
//   DecodeMode::SESSION  - one rknn3_session_run(max_new_tokens=N) call
//                          performs prefill + all decode internally (CLI)
//   DecodeMode::EXTERNAL - rknn3_session_run(max_new_tokens=1) is called once
//                          per token; the caller feeds the last token back in
//                          a loop until EOS / max_new_tokens (step_decode)
#ifndef RKNN3_LLM_CLI_ENGINE_H
#define RKNN3_LLM_CLI_ENGINE_H

#include "helpers.h"
#include "lora.h"
#include "rope.h"

#include <atomic>
#include <string>

struct CommandLineOptions;   /* defined in args.h */

/* which party drives the decode loop */
enum class DecodeMode
{
  SESSION  = 0,   /* runtime decodes internally (one session_run per turn) */
  EXTERNAL = 1,   /* caller loops token by token (session_run max_new_tokens=1) */
};

/* runtime state shared across callbacks */
struct SessionState
{
  rknn3_context    ctx = 0;
  rknn3_session*   session = nullptr;

  DecodeMode       decode_mode = DecodeMode::SESSION;

  Tokenizer*       tokenizer = nullptr;
  VocabInfo        vocab_info;
  embedding_info   embed_info;
  rknn3_llm_config llm_config;
  rknn3_llm_param  params;

  /* run configuration, changeable at runtime through slash commands */
  int32_t max_new_tokens = 512;
  bool    keep_history = true;
  bool    enable_thinking = false;
  bool    ignore_eos = false;
  bool    verbose = false;

  /* result callback prints turn-ending newline only in SESSION decode;
     EXTERNAL decode front-end prints it instead. */
  bool    callback_print_newline = true;

  std::string system_prompt;
  std::string prompt_prefix;
  std::string prompt_postfix;

  /* performance bookkeeping */
  RunStats last_run;
  RunStats total;
  uint64_t turn_count = 0;

  /* external rope cache state */
  RopeState rope;
  /* active LoRA adapter */
  LoraState lora;

  /* kvcache checkpoint policy (SAVE_CHECKPOINT) */
  rknn3_kvcache_policy_param kvcache_policy;
  bool checkpoint_enabled = false;
};

/* single shared runtime state (defined in engine.cc) */
extern SessionState g_state;

/* output logits tensor name (NULL -> default "output") */
extern const char* g_logits_name;

/* whether stdin is an interactive terminal (enables /stop during generation) */
extern bool g_stdin_tty;

/* last-token result of a single inference step (external decode) */
extern StageResultState g_stage_result;

/*=============================================================================
   Initialization / teardown
=============================================================================*/

/* init rknn3 + load model + model_init; returns ctx (0 on failure) */
rknn3_context init_context_and_model(const char* model_path, const char* weight_path,
                                     uint32_t core_mask, const char* device_id);

/* load tokenizer + mmap the embedding file into g_state */
int init_tokenizer_and_embedding(const char* tokenizer_path, const char* embedding_path);

/* create a fresh session on the initialized context, with callbacks, the
   current chat template, and the kvcache checkpoint policy applied. A LoRA
   enabled on the previous session is re-applied. */
int create_session();

/* fill rknn3_llm_param from the options + queried vocab info, then create a
   fresh session on the initialized context. */
int init_session(const CommandLineOptions& opt);

/* push the current sampling / eos parameters to the session */
int apply_llm_param();

/* apply the kvcache checkpoint (SAVE_CHECKPOINT) policy (step_decode) */
int apply_kvcache_checkpoint_policy();

/* release everything (lora / rope / session / ctx / embedding / tokenizer) */
void release_resources();

/* validate the model's rope_cache_host_storage contract against the given
   --rope-tensor option:
     - host_storage == 0 and no option:            OK
     - host_storage == 0 and option given:         OK (extra cache ignored, warn)
     - host_storage != 0 and option given:         load via rope_setup(); any
                                                   failure is fatal
     - host_storage != 0 and no option:            fatal (cache is required)
   Must be called after llm_config is queried and before init_session().
   Returns 0 on success, -1 on a fatal contract violation. */
int require_rope_cache(const char* rope_tensor_path);

/*=============================================================================
   Inference
=============================================================================*/

/* run a full generation turn: prefill the prompt (or given tokens), then
   decode according to g_state.decode_mode until max_new_tokens / EOS.
   Returns 0 on success; fills stats_out when non-null. */
int run_turn(const char* prompt_text, const std::vector<int32_t>* tokens,
             bool keep_history, bool allow_interrupt, RunStats* stats_out);

/* one session_run step: prefill (prompt/tokens) or single-token decode.
   Used by the EXTERNAL decode strategy. max_new_tokens is forced to 1. */
bool run_inference_once(const char* prompt, const std::vector<int32_t>* input_tokens,
                        bool keep_history, uint64_t* step_input_tokens);

/* reset the last-token stage result (called before each inference step) */
void reset_stage_result();

/* read the last produced token (external decode) */
bool get_stage_token(int32_t* token);

/*=============================================================================
   Benchmark / status
=============================================================================*/

int run_bench(uint64_t n_in, uint64_t n_out);

void print_perf(const RunStats& s);
void print_model_info();
void print_status();

#endif /* RKNN3_LLM_CLI_ENGINE_H */
