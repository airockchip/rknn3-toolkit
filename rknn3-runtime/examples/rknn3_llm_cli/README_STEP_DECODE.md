# rknn3_llm_step_decode

> 中文版: [README_STEP_DECODE_CN.md](README_STEP_DECODE_CN.md)

`rknn3_llm_step_decode` is a **per-token (externally driven) decode** demo: every
`rknn3_session_run` performs a single step (`max_new_tokens=1`), and an external
loop feeds the token returned by `result_callback` back as the next input until
`max_new_tokens` / EOS is reached.

The difference from `rknn3_llm_cli` (where one `rknn3_session_run` runs prefill +
the whole decode internally) is that **the decode loop is driven by the caller**:

```
one-shot inference:  one rknn3_session_run(max_new_tokens=N) runs prefill + all decode internally
step_decode:      run_turn() {
                      run_inference_once(prefill)        // consumes the whole prompt, samples token[0]
                      for step in 1 .. max_new_tokens:
                          run_inference_once({last_token}) // feeds 1 token per step, produces the next
                  }
```

- `run_inference_once()` issues `rknn3_session_run` with `max_new_tokens=1`,
  `prefill_only=true`, `disable_sampling=false`, i.e. a single inference step;
  the input is the prompt (prefill phase, still samples and emits the 1st
  token) or a single token (decode phase).
- `run_turn()` owns the outer loop: after prefill it keeps feeding the token
  from `result_callback` back until `max_new_tokens` or EOS is hit (unless
  `--ignore-eos`), and collects TTFT / prefill / decode timings. The decode
  loop starts at the 2nd token (`step=1`) because the 1st token is already
  produced by the prefill phase, avoiding double-counting.

## 1. Code structure & reuse

`rknn3_llm_cli` and `rknn3_llm_step_decode` **share the same inference engine**
(`cpp/` directory); each demo only keeps its own front-end entry:

```
cpp/
├── helpers.h      # shared types / utilities (RunStats, embedding_info, log macros, string helpers)
├── args.h/.cc     # command-line parsing (superset of both demos' options)
├── engine.h/.cc   # shared inference engine: session init, callbacks, run_turn, bench, status
├── rope.h/.cc     # external rope caches (Qwen3.5 / Gemma-4)
├── lora.h/.cc     # LoRA adapter lifecycle
├── cli_main.cc    # rknn3_llm_cli front-end: REPL + slash commands + interrupt monitor
└── step_main.cc   # rknn3_llm_step_decode front-end: per-token loop entry
```

Both targets are built from the same `cpp/CMakeLists.txt`; the decode strategy
is selected via `SessionState::decode_mode` (`SESSION` / `EXTERNAL`) and
everything else is shared. `engine.cc::run_turn()` dispatches to
`run_session_decode()` (one-shot) or `run_external_decode()` (per-token loop)
based on that mode.

## 2. Build

Use the repo's unified build scripts (one build produces both binaries):

```bash
cd examples/rknn3_llm_cli/

# Linux (export GCC_COMPILER first for cross-compilation)
./build-linux.sh -t rk3588 -a aarch64 -b Release
# output: install/rknn3_llm_cli_rk3588_linux_aarch64/
#   rknn3_llm_cli          interactive CLI
#   rknn3_llm_step_decode  per-token decode
```

## 3. Deploy & run

```bash
adb push install/rknn3_llm_cli_rk3588_linux_aarch64/ /data/
adb shell
cd /data/rknn3_llm_cli_rk3588_linux_aarch64
export LD_LIBRARY_PATH=./lib

# start (--model is required; the rest are auto-derived from the same base name)
./rknn3_llm_step_decode --model model/Qwen3.5-0.8B.rknn --core-mask 0xff

# or specify all paths explicitly
./rknn3_llm_step_decode \
    --model model/Qwen3.5-0.8B.rknn \
    --weight model/Qwen3.5-0.8B.weight \
    --vocab model/Qwen3.5-0.8B.tokenizer.gguf \
    --embed model/Qwen3.5-0.8B.embed.bin \
    -c 2048 --core-mask 0xff
```

> Note: `--model` is required; the other paths are auto-derived as
> `foo.rknn` → `foo.weight` / `foo.tokenizer.gguf` / `foo.embed.bin` (verify
> with `--dry-run`). Model files are exported by the corresponding model
> example, e.g. `examples/Qwen3_5/python` (`export_llm.py` / `export_rknn.py`).
> For models with host-stored rope, pass `--rope-tensor`.

### 3.1 Command-line arguments

The demo is configured via **startup options**: model paths, sampling
parameters, chat template, KV cache policy, LoRA, external rope and
benchmarking; it also supports a basic line-loop chat.

| Argument | Description | Default |
|----------|-------------|---------|
| `--model <path>` | RKNN model file | required |
| `--weight <path>` | weight file (auto-derived when omitted) | derived |
| `--vocab <path>` | tokenizer file (auto-derived when omitted) | derived |
| `--embed <path>` | embedding weight file (auto-derived when omitted) | derived |
| `-c, --ctx-size <n>` | max context length (tokens) | model config |
| `-n, --predict <n>` | max new tokens per turn | 512 |
| `--core-mask <mask>` | NPU core mask, hex/dec, e.g. `0xff` | `0xff` |
| `--temperature <v>` | sampling temperature | 1.0 |
| `--top-k <n>` | top-k | 1 |
| `--top-p <v>` | top-p | 0.9 |
| `--repeat-penalty <v>` | repeat penalty | 1.2 |
| `--frequency-penalty <v>` | frequency penalty | 0.0 |
| `--presence-penalty <v>` | presence penalty | 0.0 |
| `--keep-history <on/off>` | keep multi-turn history | `on` |
| `-s, --system <prompt>` | system prompt | model built-in |
| `--prompt-prefix <p>` | chat template prefix | model built-in |
| `--prompt-postfix <p>` | chat template postfix | model built-in |
| `-p, --prompt <text\|@file>` | send one message on startup (`@` prefix reads from a file) | none |
| `--logits-name <name>` | output logits name (for models with multiple outputs) | `output` |
| `--once` | run `--prompt` once and exit | off |
| `--perf` | print performance after each turn | off |
| `--verbose` | enable verbose logging | off |
| `--ignore-eos` | ignore EOS during generation | off |
| `--thinking` | enable thinking mode | off |
| `--dry-run` | print the resolved model paths and exit (verify auto-derivation) | off |
| `--rope-tensor <path>` | external rope cache | none |
| `--device-id <id>` | device ID (default: first device) | auto |
| `--checkpoint-start-pos <n>` | kvcache checkpoint start position (align to 128) | disabled |
| `--checkpoint-interval <n>` | checkpoint save interval (align to 128) | disabled |
| `--checkpoint-max-count <n>` | max checkpoint count (auto-trimmed when exceeded) | disabled |
| `--checkpoint-tail-overwrite <0\|1>` | use the tail slot as a rolling overwrite slot | disabled |
| `--bench [n_in] [n_out]` | random-token benchmark (defaults to 128 in / 128 out when omitted) | none |
| `--load <path>` | load a saved kvcache on startup | none |
| `--save <path>` | save the kvcache to a file on exit | none |
| `--lora <path>` | load and enable a LoRA adapter on startup | none |
| `-h, --help` | show help | - |

### 3.2 Batch / session persistence / benchmark examples

```bash
# send one message and exit
./rknn3_llm_step_decode --model Qwen3.5-0.8B.rknn --core-mask 0xff \
    --prompt "Introduce yourself" --once

# session persistence: save after running, resume later
./rknn3_llm_step_decode --model Qwen3.5-0.8B.rknn --core-mask 0xff \
    --prompt "hi" --once --save /data/kvcache.bin

./rknn3_llm_step_decode --model Qwen3.5-0.8B.rknn --core-mask 0xff \
    --prompt "What can you do?" --once --load /data/kvcache.bin

# benchmark (random tokens)
./rknn3_llm_step_decode --model Qwen3.5-0.8B.rknn --core-mask 0xff \
    --bench               # omitted args default to 128 in / 128 out
./rknn3_llm_step_decode --model Qwen3.5-0.8B.rknn --core-mask 0xff \
    --bench 1024 128    # 1024 input tokens -> 128 output tokens
./rknn3_llm_step_decode --model Qwen3.5-0.8B.rknn --core-mask 0xff \
    --bench 256        # n_in only, n_out defaults to 128
```

**`--bench` notes**: generates `n_in` input tokens with a fixed random seed,
clears the KV cache, runs one prefill + per-token decode turn with
`ignore_eos=1` and thinking off, prints the performance table (Prefill /
Generate / TTFT / tokens-per-second) and exits. If `n_in` exceeds
`max_ctx_len` it is first capped to `max_ctx_len - 1` (keeping at least one
decode-token slot), then `n_out` is capped to `max_ctx_len - n_in` when it
exceeds the remaining budget.

## 4. Basic interaction

Without `--once`, the demo loops on stdin for chat (multi-turn history is kept
by default; turn it off with `--keep-history off`), and `/exit` quits.

```
==============================INFO==============================
  model            : Qwen3.5-0.8B.rknn

...
=============================================================
  Type your message, input /exit to quit.

> hi
Hello! How can I help you?
> who are you?
I am Qwen3.5...
> /exit
```

## 5. LoRA usage

Supports `.lora_weight` files (see `examples/Qwen3_VL_LoRA` for the export
flow). Load and enable on startup via `--lora <path>`:

```bash
./rknn3_llm_step_decode --model Qwen2.5-0.5B-Instruct.rknn --core-mask 0xff \
    --lora Qwen2.5-0.5B-Instruct.lora_weight
```

- Call flow (`lora.cc`): `rknn3_lora_init` → `query(LORA_NUM)` →
  `query(LORA_INFO)` → `rknn3_lora_load` → `rknn3_session_enable_lora`.
- The LoRA is enabled with the session at startup and automatically
  disabled + unloaded on exit.
- The status banner shows the LoRA state: `<name> (enabled)` /
  `<name> (loaded)` / `none`.

## 6. External rope

Some models keep the rope cos/sin caches on the host and feed them per
generation step through the session's `input_callback`. Enable with
`--rope-tensor <path>`:

```bash
./rknn3_llm_step_decode --model Qwen3.5-4B.rknn --core-mask 0xff \
    --rope-tensor Qwen3.5-4B.safetensors
```

- After loading, the model's rope input tensors (`rope_cos_cache` /
  `rope_sin_cache` / `per_layer_inputs`) are discovered and an
  `input_callback` is registered.
- Two formats are auto-detected: Qwen3.5 (`rope_cos_cache` /
  `rope_sin_cache`) and Gemma-4 (`rope_cos_cache_0/1`,
  `rope_sin_cache_0/1`).
- **Enforced contract**: when the model reports `rope_cache_host_storage != 0`,
  startup **aborts** if `--rope-tensor` is missing or fails to load; when the
  model does not need host-stored rope but `--rope-tensor` is given, the
  argument is ignored with a warning.

## 7. KV cache checkpoint policy

For linear / sliding-attention models (Qwen3.5 / Gemma-4 families), a
checkpoint-style KV cache policy can be configured via startup options:

```bash
./rknn3_llm_step_decode --model Qwen3.5-4B.rknn --core-mask 0xff \
    --checkpoint-start-pos 0 --checkpoint-interval 128 --checkpoint-max-count 8 --checkpoint-tail-overwrite 1
```

## 8. Implementation notes

- Session init / callbacks / inference: `rknn3_init` →
  `rknn3_load_model_from_path` → `rknn3_model_init` → `rknn3_session_init` →
  `rknn3_session_set_callback` → `rknn3_session_run`.
- Multi-turn chat is controlled by `--keep-history <on/off>` (default `on`)
  which drives `rknn3_llm_infer_param.keep_history`; `--once` batch mode exits
  after one prompt.
- Sampling parameters are set at startup via `rknn3_llm_param.sampling_param`
  and pushed with `rknn3_session_set_llm_param`; the checkpoint policy is
  applied via `rknn3_session_set_kvcache_policy(SAVE_CHECKPOINT)`.
- The inference engine shared with `rknn3_llm_cli` lives in
  `cpp/engine.{h,cc}`; this demo's only specific logic is `step_main.cc` (the
  per-token loop is implemented in `engine.cc::run_external_decode()`) plus the
  checkpoint parameters.
