// Command-line options shared by the rknn3 LLM demos.
// This is the superset of every option used by both front-ends:
//   - rknn3_llm_cli           (interactive CLI)
//   - rknn3_llm_step_decode   (per-token external decode)
#ifndef RKNN3_LLM_CLI_OPTIONS_H
#define RKNN3_LLM_CLI_OPTIONS_H

#include <cstdint>
#include <string>

struct CommandLineOptions
{
  /* model paths */
  std::string model;
  std::string weight;
  std::string tokenizer;
  std::string embedding;
  bool        has_model = false;

  int32_t  ctx_size = 0;        /* 0 -> use model config */
  uint32_t core_mask = 0xff;
  bool     has_core_mask = false;

  /* run / generation */
  const char* prompt = nullptr;
  bool        once = false;
  int32_t     predict = 512;
  bool        verbose = false;
  bool        ignore_eos = false;
  bool        thinking = false;
  bool        keep_history = true;
  bool        perf = false;              /* step_decode: print perf after each turn */
  const char* logits_name = nullptr;
  const char* device_id = nullptr;

  /* sampling */
  float temperature = 1.0f;
  int   top_k = 1;
  float top_p = 0.9f;
  float repeat_penalty = 1.2f;
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;

  /* chat template */
  const char* system = nullptr;
  const char* prompt_prefix = nullptr;
  const char* prompt_postfix = nullptr;

  /* kvcache persistence / benchmark */
  const char* load = nullptr;
  const char* save = nullptr;
  uint64_t    bench_in = 0;
  uint64_t    bench_out = 0;

  /* lora / rope */
  const char* lora = nullptr;
  const char* rope_tensor = nullptr;

  /* CLI only */
  bool dry_run = false;

  /* kvcache checkpoint policy (SAVE_CHECKPOINT), step_decode */
  int64_t checkpoint_start_pos = 0;
  int64_t checkpoint_interval = 0;
  int64_t checkpoint_max_count = 0;
  bool    checkpoint_tail_overwrite = false;
  bool    has_checkpoint = false;   /* set when any checkpoint option is given */
};

/* parse argv into options; returns true on success, false on error / help */
bool parse_command_line(int argc, char** argv, CommandLineOptions* options);

/* print usage / help text */
void print_usage(const char* program);

/* strip the extension from the file name portion of a path,
   e.g. "model/foo.rknn" -> "model/foo" (a dot inside a dir name is kept) */
std::string strip_extension(const std::string& path);

/* derive weight / tokenizer / embedding from the rknn path when not given:
   foo.rknn -> foo.weight, foo.tokenizer.gguf, foo.embed.bin */
void resolve_paths(CommandLineOptions* opt);

/* strict unsigned parse: whole string must be digits, errno-checked, result
   > 0. Used by --bench / /bench to reject garbage and overflow. */
bool parse_positive_u64(const char* value, uint64_t* result);

#endif /* RKNN3_LLM_CLI_OPTIONS_H */
