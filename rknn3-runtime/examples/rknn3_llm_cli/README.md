# rknn3_llm_cli

`rknn3_llm_cli` is an interactive command-line chat client for RKNN3 LLM models
(llama.cpp / ollama style). It is built on the same session API as
`examples/rknn3_session_test_demo` and provides a REPL with a rich set of slash
commands for adjusting sampling parameters, clearing the KV cache, switching the
chat template, running benchmarks, and saving/loading the KV cache at runtime.

> This demo shares the whole inference engine with `rknn3_llm_step_decode`
> (per-token external decode) under `cpp/`; one build produces both binaries.
> See [`README_STEP_DECODE.md`](README_STEP_DECODE.md) for the per-token
> demo.

## 1. Build

### 1.1 Standalone build scripts

```bash
cd examples/rknn3_llm_cli/

# Linux (on an x86 host use -t x86 -a x86_64; export GCC_COMPILER for cross)
./build-linux.sh -t rk3588 -a aarch64 -b Release
# output: install/rknn3_llm_cli_rk3588_linux_aarch64/
#   rknn3_llm_cli          interactive CLI (this demo)
#   rknn3_llm_step_decode  per-token decode demo

# Android (export ANDROID_NDK_PATH first)
./build-android.sh -t rk3588 -a arm64-v8a -b Release
```

## 2. Deploy & Run

```bash
adb push install/rknn3_llm_cli_rk3588_linux_aarch64/ /data/

adb shell
cd /data/rknn3_llm_cli_rk3588_linux_aarch64
export LD_LIBRARY_PATH=./lib

# interactive chat (only --model is required; the rest are auto-derived)
./rknn3_llm_cli --model model/Qwen2.5-0.5B-Instruct.rknn --core-mask 0xff

# or specify all four paths explicitly
./rknn3_llm_cli \
    --model model/Qwen2.5-0.5B-Instruct.rknn \
    --weight model/Qwen2.5-0.5B-Instruct.weight \
    --vocab model/Qwen2.5-0.5B-Instruct.tokenizer.gguf \
    --embed model/Qwen2.5-0.5B-Instruct.embed.bin \
    --core-mask 0xff

# send one message on startup then enter the REPL
./rknn3_llm_cli \
    --model model/Qwen2.5-0.5B-Instruct.rknn \
    --ctx-size 1024 --core-mask 0xff --temperature 0.8 \
    --prompt "Introduce yourself" --once
```

> Model files (`.rknn` / `.weight` / `.tokenizer.gguf` / `.embed.bin`) must be
> prepared first; see the export/convert flow of the corresponding model
> example (e.g. `examples/Qwen2_5`). When only `--model` is given, the other
> paths are derived from the same base name (`foo.rknn` -> `foo.weight` /
> `foo.tokenizer.gguf` / `foo.embed.bin`).

### 2.1 Command-line arguments

> Only named options are accepted (`--model` is required); positional
> arguments are rejected.

| Argument | Description | Default |
|----------|-------------|---------|
| `-m, --model <path>` | RKNN model file | required |
| `--weight <path>` | weight file (auto-derived from `--model` when omitted) | derived |
| `--vocab <path>` | tokenizer file (auto-derived when omitted) | derived |
| `--embed <path>` | embedding weight file (auto-derived when omitted) | derived |
| `-c, --ctx-size <n>` | max context length (tokens) | model config |
| `-n, --predict <n>` | max new tokens per turn | 512 |
| `--core-mask <mask>` | NPU core mask, hex or decimal, e.g. `0xff` | `0xff` |
| `--device-id <id>` | RKNN3 device ID to use (default: first device) | first device |
| `-t, --temperature <v>` | sampling temperature | 1.0 |
| `--top-k <n>` | top-k | 1 |
| `--top-p <v>` | top-p | 0.9 |
| `--repeat-penalty <v>` | repeat penalty | 1.2 |
| `--frequency-penalty <v>` | frequency penalty | 0.0 |
| `--presence-penalty <v>` | presence penalty | 0.0 |
| `-s, --system <prompt>` | system prompt | model built-in template |
| `--prompt-prefix <p>` | chat template prefix | model built-in |
| `--prompt-postfix <p>` | chat template postfix | model built-in |
| `-p, --prompt <text\|@file\|path>` | send one message on startup (`@file` forces reading from a file; an existing file path is also read automatically) | none |
| `--once` | run `--prompt` once and exit (batch) | off |
| `--keep-history <on/off>` | keep multi-turn history (default: on) | on |
| `--ignore-eos` | ignore EOS during generation | off |
| `--thinking` | enable thinking mode | off |
| `--logits-name <name>` | output logits name (only for models with multiple outputs) | `output` |
| `--dry-run` | print the resolved model paths and exit (no device) | off |
| `--load <path>` | load a saved kvcache file on startup (resume session) | none |
| `--save <path>` | save the kvcache to a file on exit | none |
| `--bench [n_in] [n_out]` | run a random-token benchmark on startup and exit (defaults to 128 in / 128 out when omitted) | none |
| `--lora <path>` | load and enable a LoRA adapter on startup | none |
| `--rope-tensor <path>` | external rope cache file (for Qwen3.5 / Gemma-4 etc. models that need external rope) | none |
| `--verbose` | enable verbose logging | off |
| `-h, --help` | show help | - |

> Different models use different chat templates (e.g. Gemma uses
> `<|turn|>user\n...`). Switch with `--system` / `--prompt-prefix` /
> `--prompt-postfix` or the runtime `/system` / `/template` commands.

### 2.2 Session save / restore / benchmark

```bash
# run one message, then persist the kvcache
./rknn3_llm_cli --model foo.rknn --prompt "hi" --once --save /data/kvcache.bin

# resume a previously saved conversation
./rknn3_llm_cli --model foo.rknn --load /data/kvcache.bin

# run a benchmark on startup (`--bench [n_in] [n_out]`, defaults to 128 in / 128 out)
./rknn3_llm_cli --model foo.rknn --bench
./rknn3_llm_cli --model foo.rknn --bench 32 16
```

> `--save` / `--load` are equivalent to the runtime `/save` / `/load`
> commands but happen at process exit / startup; `--bench [n_in] [n_out]` is
> equivalent to the runtime `/bench` (both default to 128 in / 128 out when
> arguments are omitted).

## 3. Interactive usage

After startup you are in a REPL: type text to chat (multi-turn history is kept
by default, i.e. `keep_history=1`):

```
  rknn3-llm-cli
  model            : model/Qwen2.5-0.5B-Instruct.rknn
=============================================================
Max Context Length      : 1024
...
  Type '/help' for commands, type your message to chat, Ctrl+D to exit.

> Hello
Hello! How can I help you?
> Who are you?
I am Qwen, a helpful assistant.
> /clear
KV cache cleared (system prompt kept). New conversation started.
>
```

`/stop` interrupts the ongoing generation.

> ⚠️ Real-time `/stop` only works on an interactive terminal (TTY). When stdin
> is a pipe/script (e.g. non-interactive `adb shell`), `/stop` during
> generation cannot be received (it falls through to the REPL as "no
> generation in progress"). Also, text typed during generation on a TTY is
> discarded by the interrupt monitor.

### 3.1 Slash commands

| Command | Description |
|---------|-------------|
| `/help` or `/?` | show help |
| `/exit` / `/quit` / `/bye` / `/q` | exit |
| **conversation** | |
| `/clear` | clear KV cache (keep system prompt), new conversation |
| `/reset` | clear all KV cache (including system prompt) |
| `/prompt <text>` | one-shot raw prompt (no history, `keep_history=0`) |
| `/stop` / `/cancel` / `/interrupt` | interrupt current generation |
| **sampling / generation** | |
| `/temp <v>` | set temperature |
| `/top-k <n>` | set top-k |
| `/top-p <v>` | set top-p |
| `/repeat-penalty <v>` | set repeat penalty |
| `/frequency-penalty <v>` | set frequency penalty |
| `/presence-penalty <v>` | set presence penalty |
| `/sampling <top_k> <top_p> <temp> <repeat> <freq> <presence>` | set all sampling params at once |
| `/predict <n>` or `/max-tokens <n>` | set max new tokens per turn |
| `/thinking on\|off` | toggle thinking mode (re-initializes the session so the runtime applies thinking via its built-in template) |
| `/ignore-eos on\|off` | toggle ignoring EOS |
| `/keep-history on\|off` | toggle history retention (clears KV cache) |
| **prompt / template** | |
| `/system <prompt>` | set system prompt (re-initializes the session) |
| `/template <prefix> <postfix>` | set chat template prefix/postfix (re-initializes the session) |
| **info / performance** | |
| `/status` | show model / sampling / session / context info |
| `/ctx` | show current context usage |
| `/perf` | show last-turn performance (Prefill / Generate / TTFT) |
| `/stats` | show cumulative statistics |
| `/bench [n_in] [n_out]` | random-token benchmark (default 128 in / 128 out) |
| `/verbose on\|off` | toggle verbose logging |
| **kvcache persistence** | |
| `/save <path>` | save current KV cache to file |
| `/load <path>` | load KV cache from file |
| **LoRA** | |
| `/lora <path>` | load a LoRA adapter onto the context (not enabled yet) |
| `/lora-enable` | enable the loaded adapter on the session |
| `/lora-disable` | disable the adapter on the session (stays loaded) |
| `/lora-unload` | unload the adapter from the context |
| `/lora-off` or `/disable-lora` | disable + unload in one step |

## 4. LoRA usage

LoRA adapters use `.lora_weight` files (see `examples/Qwen3_VL_LoRA` for the
export flow). Load and enable (and disable and unload) are separate commands so
the lifecycle can be driven step by step:

```bash
# load and enable on startup (one step)
./rknn3_llm_cli --model foo.rknn --lora adapter.lora_weight

# step by step at runtime
> /lora adapter.lora_weight     # load onto the context only
LoRA loaded: <adapter_name>
> /lora-enable                  # enable on the session
LoRA enabled: <adapter_name> (scale=1.000)
> /lora-disable                 # disable only (stays loaded)
LoRA disabled
> /lora-unload                  # unload from the context
LoRA unloaded
> /lora-off                     # or disable + unload in one step
LoRA disabled and unloaded
```

- Call flow (`lora.cc`): `rknn3_lora_init` -> `query(LORA_NUM)` ->
  `query(LORA_INFO)` -> `rknn3_lora_load` (`/lora`);
  `rknn3_session_enable_lora` (`/lora-enable`);
  `rknn3_session_disable_lora` (`/lora-disable`);
  `rknn3_lora_unload` (`/lora-unload`).
- `/thinking`, `/system`, `/template` re-create the session; an enabled LoRA is
  re-bound automatically. On exit the adapter is disabled + unloaded.
- `/status` shows the LoRA state: `<name> (enabled)` / `<name> (loaded)` /
  `none`.

## 4.1 External rope

Some models (Qwen3.5 / Gemma-4 with `rope_cache_host_storage=1`) keep the rope
cos/sin caches on the host and feed them per generation step through the
session's `input_callback`. The CLI supports this via `--rope-tensor <path>`:

```bash
# models that need an external rope cache (Qwen3.5 / Gemma-4)
./rknn3_llm_cli --model foo.rknn --rope-tensor rope_cos_sin.bin
```

- After loading, the model's rope input tensors (`rope_cos_cache` /
  `rope_sin_cache`) are discovered and an `input_callback` is registered
  (see `examples/multicard`).
- Two formats are auto-detected: Qwen3.5 (`rope_cos_cache` /
  `rope_sin_cache`) and Gemma-4 (`rope_cos_cache_0/1`,
  `rope_sin_cache_0/1`).
- Implementation lives in `rope.cc` (`rope_setup` -> `load_rope_cache` +
  `query_ext_input_indices`; `rope_input_callback` copies the slice for
  `param.pos` into the input tensors).
- **Enforced contract**: when the model reports `rope_cache_host_storage != 0`,
  startup **aborts** if `--rope-tensor` is missing or fails to load (no more
  warn-and-continue, which could start with a broken model). When the model
  does not need host-stored rope (`rope_cache_host_storage=0`) but
  `--rope-tensor` is given, the argument is ignored with a warning.

## 5. Thinking modes

Thinking is controlled by the runtime through the `enable_thinking` field on
the input tensor (`rknn3_llm_tensor`). Every model carries its own built-in
chat template, which the runtime renders with the `enable_thinking` flag, so
no CLI-side handling is needed. The CLI no longer maintains any thinking
postfix logic.

> Note: `/thinking off` reliably suppresses thinking for ordinary questions,
> but "reasoning-inclined" models may still emit reasoning for complex
> step-by-step tasks (e.g. math) — that is model behavior.

## 6. Implementation notes

- Session init / callbacks / inference follow `examples/rknn3_session_test_demo`:
  `rknn3_init` -> `rknn3_load_model_from_path` -> `rknn3_model_init` ->
  `rknn3_session_init` -> `rknn3_session_set_callback` -> `rknn3_session_run`.
- Multi-turn chat uses `rknn3_llm_infer_param.keep_history = 1`; the session
  wraps each input with the chat template set via
  `rknn3_session_set_chat_template` and accumulates history; `/clear` calls
  `rknn3_session_clear_kvcache`.
- Runtime sampling changes go through `rknn3_session_set_llm_param`.
- `/stop` runs `rknn3_session_run` on a worker thread while the main thread
  watches stdin and calls `rknn3_session_stop` to interrupt.
- The generated token stream is guarded against out-of-range token ids (e.g.
  from a mismatched LoRA) so the CLI never crashes on corrupt model output.
- Source layout (`cpp/`, shared with `rknn3_llm_step_decode`): `helpers.h`
  (shared types / utilities), `args.{h,cc}` (command-line parsing),
  `engine.{h,cc}` (shared inference engine; `run_turn()` dispatches to one-shot
  or per-token decode via `SessionState::decode_mode`), `rope.{h,cc}` (external
  rope), `lora.{h,cc}` (LoRA), `cli_main.cc` (this demo front-end),
  `step_main.cc` (step_decode front-end).

