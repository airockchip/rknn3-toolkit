#!/bin/bash
# ============================================================
# On-device integration tests for rknn3_llm_cli
#
# This script does NOT build the binary and has no local (host-only)
# tests: it only runs tests on a real RKNN3 device via adb, expecting
# a pre-compiled rknn3_llm_cli deployed on the target.
#
# Usage:
#   ./test.sh -m <model_path> -d <adb_serial> [-o <results_log>]
#
#   -m <model_path>   RKNN model path ON THE DEVICE (required)
#   -d <adb_serial>   adb serial of the device (required)
#   -o <results_log>  append per-test results to this file
#                     (default: ./test_results.log)
#
# Environment:
#   DEVICE_BIN    cli binary path on the device
#                 (default: /data/llm_cli_test/rknn3_llm_cli)
#   DEVICE_LD_PRELOAD  optional library to LD_PRELOAD when running the cli on
#                 the device (e.g. /usr/lib/libasan.so.3 when librknn3_api_rkcp.so
#                 is ASan-instrumented); each run also exports
#                 RKNN3_SKIP_PROTO_CHECK=1
#   LORA_WEIGHT   optional .lora_weight file path on the device; when set,
#                 the LoRA happy-path tests (--lora, /lora load/enable,
#                 disable/unload, session-recreation re-apply) are also run
#   DEVICE_WORK   work directory on the device for test artifacts
#                 (default: /data/llm_cli_test)
#
# Every test case runs one real inference and its model output is
# recorded in the results log. At the end a summary list of every test
# item with its PASS/FAIL/SKIP status is printed, so different runs can
# be compared later.
# ============================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_FILE="$SCRIPT_DIR/test_results.log"

PASS=0
FAIL=0
SKIP=0

# per-test tracking for the final summary list
declare -a TESTS=()
declare -a TSTATUS=()

report() { printf '%s\n' "$*"; }

# ---- argument parsing: -m and -d are required (unless -l) ----
MODEL=""
DEVICE=""
SELECTED=""
LIST_ONLY=0
usage() { report "Usage: $0 -m <model_path> -d <adb_serial> [-t <name[,name...]>] [-l] [-o <results_log>]"; exit 1; }
while getopts "m:d:o:t:l" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    d) DEVICE="$OPTARG" ;;
    o) RESULTS_FILE="$OPTARG" ;;
    t) SELECTED="$OPTARG" ;;
    l) LIST_ONLY=1 ;;
    *) usage ;;
  esac
done

# selectable test cases: every real maybe_case "name" call in this script (in
# file order), plus positional_rejected which is handled by a dedicated block.
# The grep anchors on leading whitespace so comments / the function definition
# / this very line can never be mistaken for a test case. ALL_CASES is a real
# array so membership checks below are exact, not substring/whitespace based.
mapfile -t ALL_CASES < <(grep -oE '^[[:space:]]+maybe_case "[^"]+"' "$0" | sed 's/^[[:space:]]*maybe_case "//; s/"$//')
ALL_CASES+=("positional_rejected")
# comma-joined view of ALL_CASES for exact membership tests
ALL_CASES_CSV=",${ALL_CASES[*]},"
ALL_CASES_CSV="${ALL_CASES_CSV// /,}"

if [ "$LIST_ONLY" -eq 1 ]; then
  report "Available test cases:"
  for t in "${ALL_CASES[@]}"; do report "  $t"; done
  exit 0
fi

if [ -z "$MODEL" ] || [ -z "$DEVICE" ]; then
  report "ERROR: -m <model_path> and -d <adb_serial> are required"
  usage
fi
if [ -n "$SELECTED" ]; then
  # a mistyped test name must fail the run, not silently skip to a green exit
  for t in ${SELECTED//,/ }; do
    case "$ALL_CASES_CSV" in *",$t,"*) ;; *) report "ERROR: unknown test case '$t' (see -l)"; exit 1 ;; esac
  done
fi

# ---- results log header for this run ----
if [ -n "$RESULTS_FILE" ]; then
  mkdir -p "$(dirname "$RESULTS_FILE")"
  printf '\n== %s | model=%s | device=%s ==\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$MODEL" "$DEVICE" >> "$RESULTS_FILE"
fi

log_result() { [ -n "$RESULTS_FILE" ] && printf '[%s] %s\n' "$1" "$2" >> "$RESULTS_FILE"; }
log_output() {
  [ -n "$RESULTS_FILE" ] || return 0
  {
    printf '    -- output (%s) --\n' "$1"
    printf '%s\n' "$2" | sed 's/^/    /'
  } >> "$RESULTS_FILE"
}

# true when the given case should run under the current -t selection
case_selected() {
  [ -z "$SELECTED" ] && return 0
  case ",$SELECTED," in *",$1,"*) return 0 ;; esac
  return 1
}

# run dev_case only if the case is selected
maybe_case() {
  local name="$1"
  if case_selected "$name"; then dev_case "$@"; fi
}

# record one test result: <name> <PASS|FAIL|SKIP> <detail message>
EXECUTED=0   # number of cases that actually ran (PASS+FAIL, not SKIP)
test_done() {
  local name="$1" status="$2" msg="${3:-}"
  TESTS+=("$name")
  TSTATUS+=("$status")
  case "$status" in
    PASS) PASS=$((PASS+1)); EXECUTED=$((EXECUTED+1)) ;;
    FAIL) FAIL=$((FAIL+1)); EXECUTED=$((EXECUTED+1)) ;;
    SKIP) SKIP=$((SKIP+1)) ;;
  esac
  log_result "$status" "$name"
  report "  $status: $name$msg"
}

# ------------------------------------------------------------
# On-device tests: every case runs one real inference
# ------------------------------------------------------------
report ""
report "== On-device tests via adb -s $DEVICE (model: $MODEL) =="

  DEVICE_MODEL="$MODEL"
  DEVICE_BIN="${DEVICE_BIN:-/data/llm_cli_test/rknn3_llm_cli}"
  DEVICE_WORK="${DEVICE_WORK:-/data/llm_cli_test}"
  DEVICE_LD_PRELOAD="${DEVICE_LD_PRELOAD:-}"
  LORA_WEIGHT="${LORA_WEIGHT:-}"
  DEV_READY=0

if ! adb -s "$DEVICE" get-state >/dev/null 2>&1; then
  test_done "device_reachable" "FAIL" " (device $DEVICE not reachable)"
elif ! adb -s "$DEVICE" shell "test -f '$DEVICE_MODEL' && echo MODEL_OK" 2>/dev/null | grep -q "MODEL_OK"; then
  test_done "model_exists" "FAIL" " (model '$DEVICE_MODEL' not found on device)"
elif ! adb -s "$DEVICE" shell "test -x '$DEVICE_BIN' && echo BIN_OK" 2>/dev/null | grep -q "BIN_OK"; then
  test_done "device_binary" "SKIP" " (cli binary not found on device: $DEVICE_BIN)"
else
  report "  using device binary: $DEVICE_BIN"
  DEV_READY=1
fi

# the CLI uses the model's built-in chat template by default (queried from
# the rknn model); no --system/--prompt-prefix/--prompt-postfix needed
DEV_ARGS="--model $DEVICE_MODEL --core-mask 0xff"

# strip ANSI colors and loader noise from the raw CLI output
strip_noise() {
  printf '%s' "$1" | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -v -E 'llama_model_loader|print_info|GetVocabInfo|special_|^load:|init_tokenizer|^W RKNN|^E RKNN|found [0-9]+ RKNN3|tokenizer\.ggml'
}

# dev_case <name> <cli_args> <stdin> <sleep> [needle...] -- one real inference
dev_case() {
  local name="$1" cli="$2" stdin="$3" slp="$4"; shift 4
  local out clean ok needle detail="" cli_exit=""
  case "$stdin" in *"'"*) report "  WARN: $name stdin contains a single quote (may break)" ;; esac
  out="$(adb -s "$DEVICE" shell "export RKNN3_SKIP_PROTO_CHECK=1${DEVICE_LD_PRELOAD:+; export LD_PRELOAD=$DEVICE_LD_PRELOAD}; (printf '$stdin'; sleep $slp) | \"$DEVICE_BIN\" $cli --predict 20; echo __CLI_EXIT=\$?" 2>&1)"
  # adb shell does not propagate the remote exit code, so read it from the marker
  cli_exit="$(printf '%s' "$out" | sed -n 's/.*__CLI_EXIT=\(-\{0,1\}[0-9]*\).*/\1/p' | tail -1)"
  clean="$(strip_noise "$out")"
  log_output "$name" "$clean"
  ok=1
  # a nonzero exit code (init / --once / bench failure) must never PASS
  if [ -n "$cli_exit" ] && [ "$cli_exit" -ne 0 ]; then
    ok=0; detail="$detail (cli exit=$cli_exit)"
  fi
  # a run that produced a hard failure must never PASS, even if the expected
  # needles happen to be present in the output
  if printf '%s' "$clean" | grep -qE "rknn3_session_run failed|generation failed|session run failed|benchmark run failed|/prompt run failed"; then
    ok=0; detail="$detail (run reported a failure)"
  fi
  for needle in "$@"; do
    if ! printf '%s' "$clean" | grep -qF -- "$needle"; then
      ok=0; detail="$detail (output missing: '$needle')"
    fi
  done
  if [ "$ok" -eq 1 ]; then test_done "$name" "PASS";
  else test_done "$name" "FAIL" "$detail"; fi
}

if [ "$DEV_READY" -eq 1 ]; then
  # auto-derived paths + one inference
  maybe_case "gen_derived" "$DEV_ARGS" '1+1=?\n' 5 \
    "$(dirname "$DEVICE_MODEL")/$(basename "$DEVICE_MODEL" .rknn).weight" "bye."
  # explicit paths + one inference
  EXPLICIT_ARGS="--model $DEVICE_MODEL --weight $(dirname "$DEVICE_MODEL")/$(basename "$DEVICE_MODEL" .rknn).weight --vocab $(dirname "$DEVICE_MODEL")/$(basename "$DEVICE_MODEL" .rknn).tokenizer.gguf --embed $(dirname "$DEVICE_MODEL")/$(basename "$DEVICE_MODEL" .rknn).embed.bin --core-mask 0xff --system \"\" --prompt-prefix \"<s><|im_start|>user\\n\" --prompt-postfix \"<|im_end|>\\n<|im_start|>assistant\\n\""
  maybe_case "gen_explicit" "$EXPLICIT_ARGS" '2+2=?\n' 5 \
    "$DEVICE_MODEL" "bye."
  # /help then a chat message (inference)
  maybe_case "help_cmd" "$DEV_ARGS" '/help\nhi\n' 5 "/thinking" "bye."
  # unknown command then a chat message (inference)
  maybe_case "unknown_cmd" "$DEV_ARGS" '/bogus\nhi\n' 5 "unknown command" "bye."
  # /thinking off then a chat message (inference)
  maybe_case "thinking_off" "$DEV_ARGS" '/thinking off\nWhat is the capital of France? Answer in one word.\n' 6 \
    "enable_thinking = off" "bye."
  # /thinking on then a chat message (inference)
  maybe_case "thinking_on" "$DEV_ARGS" '/thinking on\nSolve 17*3.\n' 6 \
    "enable_thinking = on" "bye."
  # /clear then a chat message (inference)
  maybe_case "clear_cmd" "$DEV_ARGS" 'hello\n/clear\n1+1=?\n' 6 "KV cache cleared" "bye."
  # /status then a chat message (inference)
  maybe_case "status_cmd" "$DEV_ARGS" '/status\nhi\n' 5 "Max Context Length" "bye."
  # a chat message then /stats (records cumulative stats)
  maybe_case "stats_cmd" "$DEV_ARGS" 'hi\n/stats\n' 6 "turns" "tokens/s"
  # a chat message then /perf (records prefill/decode table)
  maybe_case "perf_cmd" "$DEV_ARGS" 'hi\n/perf\n' 6 "Prefill" "Generate"
  # /ctx then a chat message
  maybe_case "ctx_cmd" "$DEV_ARGS" '/ctx\nhi\n' 5 "context:" "bye."
  # /reset then a chat message
  maybe_case "reset_cmd" "$DEV_ARGS" '/reset\nhi\n' 5 "KV cache fully cleared" "bye."
  # /exit / /quit / /bye / /q all terminate
  maybe_case "exit_cmd" "$DEV_ARGS" '/exit\n' 2 "bye."
  maybe_case "quit_cmd" "$DEV_ARGS" '/quit\n' 2 "bye."
  maybe_case "bye_cmd" "$DEV_ARGS" '/bye\n' 2 "bye."
  maybe_case "q_cmd" "$DEV_ARGS" '/q\n' 2 "bye."
  # /stop / /cancel / /interrupt recognized outside generation
  maybe_case "stop_cmd" "$DEV_ARGS" '/stop\nhi\n' 5 "no generation in progress" "bye."
  maybe_case "cancel_cmd" "$DEV_ARGS" '/cancel\nhi\n' 5 "no generation in progress" "bye."
  # /prompt runs one raw (untemplated) inference
  maybe_case "prompt_cmd" "$DEV_ARGS" '/prompt What is 2+2?\n' 6 "bye."

  # sampling parameters
  maybe_case "temp_cmd" "$DEV_ARGS" '/temp 0.8\nhi\n' 5 "temperature = 0.800" "bye."
  maybe_case "topk_cmd" "$DEV_ARGS" '/top-k 3\nhi\n' 5 "top_k = 3" "bye."
  maybe_case "topp_cmd" "$DEV_ARGS" '/top-p 0.7\nhi\n' 5 "top_p = 0.700" "bye."
  maybe_case "repeat_cmd" "$DEV_ARGS" '/repeat-penalty 1.3\nhi\n' 5 "repeat_penalty = 1.300" "bye."
  maybe_case "freq_cmd" "$DEV_ARGS" '/frequency-penalty 0.1\nhi\n' 5 "frequency_penalty = 0.100" "bye."
  maybe_case "presence_cmd" "$DEV_ARGS" '/presence-penalty 0.2\nhi\n' 5 "presence_penalty = 0.200" "bye."
  maybe_case "sampling_cmd" "$DEV_ARGS" '/sampling 5 0.8 0.9 1.1 0.1 0.2\nhi\n' 5 "sampling params updated" "bye."
  # generation limits
  maybe_case "predict_cmd" "$DEV_ARGS" '/predict 10\nhi\n' 5 "max_new_tokens = 10" "bye."
  maybe_case "max_tokens_cmd" "$DEV_ARGS" '/max-tokens 15\nhi\n' 5 "max_new_tokens = 15" "bye."
  # toggles
  maybe_case "ignore_eos_cmd" "$DEV_ARGS" '/ignore-eos on\nhi\n' 5 "ignore_eos = on" "bye."
  maybe_case "keep_history_cmd" "$DEV_ARGS" '/keep-history off\nhi\n' 5 "keep_history = off" "KV cache cleared" "bye."
  maybe_case "verbose_cmd" "$DEV_ARGS" '/verbose on\nhi\n' 5 "verbose = on" "bye."
  # prompt / template
  maybe_case "system_cmd" "$DEV_ARGS" '/system You are a helpful assistant.\nhi\n' 6 "system prompt updated" "bye."
  maybe_case "template_cmd" "$DEV_ARGS" '/template PX PY\nhi\n' 6 "chat template updated" "bye."
  # benchmark (random-token inference, no chat needed)
  maybe_case "bench_cmd" "$DEV_ARGS" '/bench 32 16\n' 6 "benchmark: 32 input tokens -> 16 output tokens" "bye."
  # kvcache persistence: save then load
  maybe_case "save_load_cmd" "$DEV_ARGS" "hi\n/save $DEVICE_WORK/test_kvcache.bin\n/load $DEVICE_WORK/test_kvcache.bin\nhi\n" 8 \
    "KV cache saved" "KV cache loaded" "bye."
  # startup --bench <n_in> <n_out> (batch, no REPL)
  maybe_case "bench_cli" "$DEV_ARGS --bench 32 16" '' 6 \
    "benchmark: 32 input tokens -> 16 output tokens"
  # startup --save: persist the conversation kvcache on exit
  adb -s "$DEVICE" shell "rm -f $DEVICE_WORK/test_cli_kvcache.bin" >/dev/null 2>&1
  maybe_case "save_cli" "$DEV_ARGS --save $DEVICE_WORK/test_cli_kvcache.bin" 'hi\n' 6 "KV cache saved"
  # startup --load: resume the saved kvcache
  maybe_case "load_cli" "$DEV_ARGS --load $DEVICE_WORK/test_cli_kvcache.bin" 'continue\n' 6 "KV cache loaded"
  # startup --lora with a missing weight file: must fail gracefully and keep the base model
  maybe_case "lora_cli" "$DEV_ARGS --lora /nonexistent/lora.weight" 'hi\n' 6 \
    "LoRA setup failed" "bye."
  # granular slash commands on the error path: load fails, enable/disable/unload are no-ops
  maybe_case "lora_cmd" "$DEV_ARGS" '/lora /nonexistent/lora.weight\n/lora-enable\n/lora-disable\n/lora-unload\n/lora-off\nhi\n' 6 \
    "LoRA load failed" "LoRA enable failed" "LoRA not enabled" "LoRA unloaded" "LoRA disabled and unloaded" "bye."

  # LoRA happy path: only when a real .lora_weight file is provided
  if [ -n "$LORA_WEIGHT" ] && adb -s "$DEVICE" shell "test -f '$LORA_WEIGHT' && echo LORA_OK" 2>/dev/null | grep -q "LORA_OK"; then
    maybe_case "lora_cli_on" "$DEV_ARGS --lora \"$LORA_WEIGHT\"" 'hi\n' 6 "LoRA enabled" "bye."
    # full lifecycle: load -> enable -> disable -> unload
    maybe_case "lora_cmd_on" "$DEV_ARGS" "/lora \"$LORA_WEIGHT\"\n/lora-enable\n/lora-disable\n/lora-unload\nhi\n" 8 \
      "LoRA loaded" "LoRA enabled" "LoRA disabled" "LoRA unloaded" "bye."
    # session recreation (/thinking) must re-apply the enabled LoRA
    maybe_case "lora_reenable" "$DEV_ARGS" "/lora \"$LORA_WEIGHT\"\n/lora-enable\n/thinking off\nhi\n" 8 \
      "LoRA loaded" "LoRA enabled" "bye."
  else
    case_selected "lora_cli_on" && test_done "lora_cli_on" "SKIP" " (need LORA_WEIGHT on device)"
    case_selected "lora_cmd_on" && test_done "lora_cmd_on" "SKIP" " (need LORA_WEIGHT on device)"
    case_selected "lora_reenable" && test_done "lora_reenable" "SKIP" " (need LORA_WEIGHT on device)"
  fi

  # positional args are rejected on the command line
  if case_selected "positional_rejected"; then
    out=$(adb -s "$DEVICE" shell "export RKNN3_SKIP_PROTO_CHECK=1${DEVICE_LD_PRELOAD:+; export LD_PRELOAD=$DEVICE_LD_PRELOAD}; \"$DEVICE_BIN\" $DEV_ARGS xyz" 2>&1)
    out="$(strip_noise "$out")"
    log_output "positional_rejected" "$out"
    if printf '%s' "$out" | grep -qF "unexpected positional"; then
      test_done "positional_rejected" "PASS"
    else
      test_done "positional_rejected" "FAIL" " (missing: 'unexpected positional')"
    fi
  fi
fi

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
report ""
report "=========================================="
report "  PASS: $PASS   FAIL: $FAIL   SKIP: $SKIP"
report "=========================================="

report ""
report "== All test cases =="
for i in "${!TESTS[@]}"; do
  report "  ${TSTATUS[$i]}: ${TESTS[$i]}"
done

if [ -n "$RESULTS_FILE" ]; then
  printf -- '-- summary: PASS=%d FAIL=%d SKIP=%d --\n' "$PASS" "$FAIL" "$SKIP" >> "$RESULTS_FILE"
  report ""
  report "  results appended to: $RESULTS_FILE"
fi

# with an explicit -t selection, at least one selected case must actually have
# run; otherwise (e.g. device unreachable) the run is a silent no-op that must
# not report success
if [ -n "$SELECTED" ] && [ "$EXECUTED" -eq 0 ]; then
  report "ERROR: no selected test case actually ran (device/model/binary not ready?)"
  exit 1
fi

[ "$FAIL" -eq 0 ]
