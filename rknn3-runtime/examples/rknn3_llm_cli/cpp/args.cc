// Command-line parsing shared by the rknn3 LLM demos.

#include "args.h"
#include "helpers.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <strings.h>

/*=============================================================================
   Helpers
=============================================================================*/

static bool take_option_value(int argc, char** argv, int* index,
                              const char* option, const char** value)
{
  if (!index || !value || *index + 1 >= argc || argv[*index + 1][0] == '\0' ||
      strncmp(argv[*index + 1], "--", 2) == 0) {
    printf("%s requires a value\n", option);
    return false;
  }
  *value = argv[++(*index)];
  return true;
}

bool parse_positive_u64(const char* value, uint64_t* result)
{
  if (!value || !result || value[0] == '\0' || value[0] == '-') {
    return false;
  }

  errno = 0;
  char* end = nullptr;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
    return false;
  }

  *result = (uint64_t)parsed;
  return true;
}

static bool parse_positive_i32(const char* value, int32_t* result)
{
  uint64_t parsed = 0;
  if (!result || !parse_positive_u64(value, &parsed) || parsed > 0x7fffffffU) {
    return false;
  }
  *result = (int32_t)parsed;
  return true;
}

/* parse a non-negative 64-bit integer (allows 0, e.g. checkpoint_start_pos) */
static bool parse_nonneg_i64(const char* value, int64_t* result)
{
  if (!value || !result || value[0] == '\0' || value[0] == '-') {
    return false;
  }

  errno = 0;
  char* end = nullptr;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed > 0x7fffffffffffffffULL) {
    return false;
  }

  *result = (int64_t)parsed;
  return true;
}

static bool parse_core_mask(const char* value, uint32_t* result)
{
  if (!value || !result || value[0] == '\0' || value[0] == '-') {
    return false;
  }

  errno = 0;
  char* end = nullptr;
  unsigned long long parsed = strtoull(value, &end, 0);
  if (errno != 0 || end == value || *end != '\0') {
    errno = 0;
    end = nullptr;
    parsed = strtoull(value, &end, 16);
  }
  if (errno != 0 || end == value || *end != '\0' || parsed > 0xffffffffULL) {
    return false;
  }

  *result = (uint32_t)parsed;
  return true;
}

/*=============================================================================
   Named-option parsing
   (only named options are accepted; positional arguments are rejected)
=============================================================================*/

static bool parse_named_command_line(int argc, char** argv, CommandLineOptions* options)
{
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    const char* value = nullptr;

    if (strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->model = value;
      options->has_model = true;
    } else if (strcmp(arg, "-w") == 0 || strcmp(arg, "--weight") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->weight = value;
    } else if (strcmp(arg, "--vocab") == 0 || strcmp(arg, "--tokenizer") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->tokenizer = value;
    } else if (strcmp(arg, "--embed") == 0 || strcmp(arg, "--embedding") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->embedding = value;
    } else if (strcmp(arg, "-c") == 0 || strcmp(arg, "--ctx-size") == 0 ||
               strcmp(arg, "--max-context-len") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value) ||
          !parse_positive_i32(value, &options->ctx_size)) {
        printf("%s requires a positive integer\n", arg);
        return false;
      }
    } else if (strcmp(arg, "--core-mask") == 0 || strcmp(arg, "--run-core-mask") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value) ||
          !parse_core_mask(value, &options->core_mask)) {
        printf("%s requires a valid hexadecimal mask\n", arg);
        return false;
      }
      options->has_core_mask = true;
    } else if (strcmp(arg, "-n") == 0 || strcmp(arg, "--predict") == 0 ||
               strcmp(arg, "--n-predict") == 0 || strcmp(arg, "--max-new-tokens") == 0) {
      int32_t parsed = 0;
      if (!take_option_value(argc, argv, &i, arg, &value) ||
          !parse_positive_i32(value, &parsed)) {
        printf("%s requires a positive integer\n", arg);
        return false;
      }
      options->predict = parsed;
    } else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--prompt") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->prompt = value;
    } else if (strcmp(arg, "--once") == 0) {
      options->once = true;
    } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--temperature") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->temperature = strtof(value, nullptr);
    } else if (strcmp(arg, "--top-k") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->top_k = atoi(value);
    } else if (strcmp(arg, "--top-p") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->top_p = strtof(value, nullptr);
    } else if (strcmp(arg, "--repeat-penalty") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->repeat_penalty = strtof(value, nullptr);
    } else if (strcmp(arg, "--frequency-penalty") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->frequency_penalty = strtof(value, nullptr);
    } else if (strcmp(arg, "--presence-penalty") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->presence_penalty = strtof(value, nullptr);
    } else if (strcmp(arg, "-s") == 0 || strcmp(arg, "--system") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->system = value;
    } else if (strcmp(arg, "--prompt-prefix") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->prompt_prefix = value;
    } else if (strcmp(arg, "--prompt-postfix") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->prompt_postfix = value;
    } else if (strcmp(arg, "--verbose") == 0) {
      options->verbose = true;
    } else if (strcmp(arg, "--no-verbose") == 0) {
      options->verbose = false;
    } else if (strcmp(arg, "--ignore-eos") == 0) {
      options->ignore_eos = true;
    } else if (strcmp(arg, "--no-ignore-eos") == 0) {
      options->ignore_eos = false;
    } else if (strcmp(arg, "--thinking") == 0) {
      options->thinking = true;
    } else if (strcmp(arg, "--keep-history") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      if (strcmp(value, "on") == 0 || strcasecmp(value, "1") == 0 ||
          strcasecmp(value, "true") == 0) {
        options->keep_history = true;
      } else if (strcmp(value, "off") == 0 || strcasecmp(value, "0") == 0 ||
                 strcasecmp(value, "false") == 0) {
        options->keep_history = false;
      } else {
        printf("%s requires 0|1 (or on|off)\n", arg);
        return false;
      }
    } else if (strcmp(arg, "--perf") == 0) {
      options->perf = true;
    } else if (strcmp(arg, "--logits-name") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->logits_name = value;
    } else if (strcmp(arg, "--rope-tensor") == 0 || strcmp(arg, "--rope") == 0 ||
               strcmp(arg, "--rope-path") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &options->rope_tensor)) return false;
    } else if (strcmp(arg, "--device-id") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &options->device_id)) return false;
    } else if (strcmp(arg, "--dry-run") == 0) {
      options->dry_run = true;
    } else if (strcmp(arg, "--checkpoint-start-pos") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value) ||
          !parse_nonneg_i64(value, &options->checkpoint_start_pos)) {
        printf("%s requires a non-negative integer\n", arg);
        return false;
      }
      options->has_checkpoint = true;
    } else if (strcmp(arg, "--checkpoint-interval") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value) ||
          !parse_nonneg_i64(value, &options->checkpoint_interval)) {
        printf("%s requires a non-negative integer\n", arg);
        return false;
      }
      options->has_checkpoint = true;
    } else if (strcmp(arg, "--checkpoint-max-count") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value) ||
          !parse_nonneg_i64(value, &options->checkpoint_max_count)) {
        printf("%s requires a non-negative integer\n", arg);
        return false;
      }
      options->has_checkpoint = true;
    } else if (strcmp(arg, "--checkpoint-tail-overwrite") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      if (strcmp(value, "1") == 0 || strcasecmp(value, "on") == 0 ||
          strcasecmp(value, "true") == 0) {
        options->checkpoint_tail_overwrite = true;
      } else if (strcmp(value, "0") == 0 || strcasecmp(value, "off") == 0 ||
                 strcasecmp(value, "false") == 0) {
        options->checkpoint_tail_overwrite = false;
      } else {
        printf("%s requires 0|1 (or on|off)\n", arg);
        return false;
      }
      options->has_checkpoint = true;
    } else if (strcmp(arg, "--bench") == 0) {
      /* --bench without args defaults to 128 in / 128 out (matching the
         runtime /bench command); otherwise parse n_in (required) and an
         optional n_out. */
      options->bench_out = 128;
      if (i + 1 >= argc || argv[i + 1][0] == '-')
      {
        options->bench_in = 128;
        continue;
      }
      const char* v = argv[++i];
      if (!parse_positive_u64(v, &options->bench_in))
      {
        printf("--bench requires a positive integer for n_in\n");
        return false;
      }
      /* optional n_out: either embedded after a space in the same arg
         (--bench "32 16") or as a separate numeric argument (--bench 32 16) */
      char* end = nullptr;
      (void)strtoull(v, &end, 10);
      if (end && *end != '\0')
      {
        uint64_t out = 0;
        if (!parse_positive_u64(end, &out))
        {
          printf("--bench n_out must be a positive integer\n");
          return false;
        }
        options->bench_out = out;
      }
      else if (i + 1 < argc && argv[i + 1][0] != '-')
      {
        uint64_t out = 0;
        if (parse_positive_u64(argv[i + 1], &out))
        {
          options->bench_out = out;
          ++i;
        }
        else
        {
          printf("--bench n_out must be a positive integer\n");
          return false;
        }
      }
    } else if (strcmp(arg, "--load") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->load = value;
    } else if (strcmp(arg, "--save") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->save = value;
    } else if (strcmp(arg, "--lora") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->lora = value;
    } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
      return false;
    } else {
      printf("unknown or positional argument in named mode: %s\n", arg);
      return false;
    }
  }

  if (!options->has_model || options->model.empty())
  {
    printf("--model is required\n");
    return false;
  }

  resolve_paths(options);
  return true;
}

bool parse_command_line(int argc, char** argv, CommandLineOptions* options)
{
  if (!options || argc < 2) {
    return false;
  }
  return parse_named_command_line(argc, argv, options);
}

/*=============================================================================
   Path derivation
=============================================================================*/

std::string strip_extension(const std::string& path)
{
  size_t slash = path.find_last_of('/');
  size_t dot = path.find_last_of('.');
  if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
  {
    return path.substr(0, dot);
  }
  return path;
}

void resolve_paths(CommandLineOptions* opt)
{
  if (opt->model.empty()) return;
  std::string base = strip_extension(opt->model);
  if (opt->weight.empty()) opt->weight = base + ".weight";
  if (opt->tokenizer.empty()) opt->tokenizer = base + ".tokenizer.gguf";
  if (opt->embedding.empty()) opt->embedding = base + ".embed.bin";
}

/*=============================================================================
   Usage
=============================================================================*/

void print_usage(const char* program)
{
  printf("Usage:\n");
  printf("  %s --model <rknn_path> [options]\n", program);
  printf("\nModel paths (only --model is required; the rest are auto-derived\n");
  printf("from the rknn base name: foo.rknn -> foo.weight / foo.tokenizer.gguf /\n");
  printf("foo.embed.bin):\n");
  printf("  -m, --model <path>        RKNN model file (required)\n");
  printf("      --weight <path>       weight file\n");
  printf("      --vocab <path>        tokenizer file\n");
  printf("      --embed <path>        embedding weight file\n");
  printf("\nRun / generation:\n");
  printf("  -c, --ctx-size <n>        max context length (default: from model config)\n");
  printf("      --core-mask <mask>    NPU core mask, hex or dec (default: 0xff)\n");
  printf("  -n, --predict <n>         max new tokens per turn (default: 512)\n");
  printf("  -p, --prompt <text|@file> send one prompt on startup\n");
  printf("      --once                run the --prompt once and exit\n");
  printf("      --keep-history        keep multi-turn history (default: on)\n");
  printf("      --perf                print performance after each turn\n");
  printf("      --ignore-eos          ignore EOS during generation\n");
  printf("      --thinking            enable thinking mode\n");
  printf("      --logits-name <name>  output logits name (default: output)\n");
  printf("      --device-id <id>      RKNN3 device ID to use (default: first device)\n");
  printf("      --verbose             enable verbose logging\n");
  printf("\nSampling:\n");
  printf("  -t, --temperature <v>     sampling temperature (default: 1.0)\n");
  printf("      --top-k <n>           top-k (default: 1)\n");
  printf("      --top-p <v>           top-p (default: 0.9)\n");
  printf("      --repeat-penalty <v>  repeat penalty (default: 1.2)\n");
  printf("      --frequency-penalty <v> (default: 0.0)\n");
  printf("      --presence-penalty <v>  (default: 0.0)\n");
  printf("\nChat template:\n");
  printf("  -s, --system <prompt>     system prompt (default: model built-in)\n");
  printf("      --prompt-prefix <p>   chat template prefix (default: model built-in)\n");
  printf("      --prompt-postfix <p>  chat template postfix (default: model built-in)\n");
  printf("\nKV cache / benchmark:\n");
  printf("      --load <path>         load a saved kvcache file before chatting\n");
  printf("      --save <path>         save the kvcache to a file on exit\n");
  printf("      --bench [n_in] [n_out] run a benchmark (random tokens) and exit\n");
  printf("                              (default: 128 in / 128 out)\n");
  printf("      --dry-run             print resolved model paths and exit\n");
  printf("\nLoRA / rope:\n");
  printf("      --lora <path>         enable a LoRA adapter on startup\n");
  printf("      --rope-tensor <path>  external rope cache file (Qwen3.5 / Gemma-4)\n");
  printf("\nKV cache checkpoint (step_decode, linear/sliding-attention models):\n");
  printf("      --checkpoint-start-pos <n>    checkpoint start position\n");
  printf("      --checkpoint-interval <n>     checkpoint save interval\n");
  printf("      --checkpoint-max-count <n>    max checkpoint count\n");
  printf("      --checkpoint-tail-overwrite <0|1>  rolling tail overwrite\n");
  printf("\n  -h, --help                show this help\n");
}
