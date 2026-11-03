import os

# Follow rknn_session_test.py style: configure environment before importing runtime packages.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")
# tie-word model must use the embedding weights tied inside the model/weight file.
os.environ["RKNN_ENABLE_TIE_WORD_EMBEDDINGS"] = "1"

from transformers import AutoTokenizer
import ctypes
import time

from rknn3lite.api import RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMTokenizerCallback, LLMGetEmbedCallback
from rknn3lite.api.rknn3_types import (
    RKNN3QueryCmd,
    RKNN3KVCachePolicy,
    RKNN3KVCacheClearPolicy,
    RKNN3KVCachePolicyParam,
    dump_tensor_attr,
)

# ============================================= Default Config =============================================

RKNN_MODEL = "Qwen2.5-0.5B-Instruct-tie-word.rknn"
WEIGHT_MODEL = "Qwen2.5-0.5B-Instruct-tie-word.weight"
TOKENIZER_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 512
CORE_MASK = "0xff"

CHECKPOINT_START_POS = 0
CHECKPOINT_INTERVAL = 128
MAX_CHECKPOINT_COUNT = 100

# The C++ demo contains one long prompt and runs it twice (test_num = 2).
RANDOM_PROMPTS = [
    """Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?
Let's think step by step
In 2005, 60/2=30 kids came to the cookout.
In 2006, 30/3*2=20 kids came to the cookout.
The answer is 20

Question: Zilla spent 7% of her monthly earnings on rent, half of it on her other monthly expenses, and put the rest in her savings. If she spent $133 on her rent, how much does she deposit into her savings account in a month?
Let's think step by step
Since $133 is equal to 7% of her earnings, then 1% is equal to $133/7 = $19.
The total monthly earning of Zilla is represented by 100%, so $19 x 100 = $1900 is her monthly earnings.
So, $1900/2 = $950 is spent on her other monthly expenses.
The total amount spent on the rent and other monthly expenses is $133 + $950 = $1083.
Hence, she saves $1900 - $1083 = $817 per month.
The answer is 817

Question: If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?
Let's think step by step
The total ratio representing the slices of pizza that Buzz bought is 5+8=13
If he shared the slices of pizza with the waiter, the waiter received a fraction of 8/13 of the total number of slices, which totals 8/13 * 78 = 48 slices
Twenty less the number of slices of pizza that the waiter ate is 48-20 = 28
The answer is 28

Question: Jame gets a raise to $20 per hour and works 40 hours a week.  His old job was $16 an hour for 25 hours per week.  How much more money does he make per year in his new job than the old job if he works 52 weeks a year?
Let's think step by step
He makes 20*40=$800 per week
He used to make 16*25=$400 per week
So his raise was 800-400=$400 per week
So he makes 400*52=$20,800 per year more
The answer is 20800
Question: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?
Let's think step by step
"""
]

tokenizer = None
first_token = None

# Keep ctypes callback objects/userdata alive for the whole session lifetime.
_result_cb = None
_tokenizer_cb = None
_tok_ud = None
_embed_cb = None
_embed_ud = None


# ============================================= Callbacks =============================================

def result_callback(userdata, result_ptr, state):
    global tokenizer, first_token

    if not hasattr(result_callback, "accumulated_tokens"):
        result_callback.accumulated_tokens = []
        result_callback.last_output_text = ""

    def decode_safe(tokens):
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text.split("\ufffd", 1)[0] if "\ufffd" in text else text

    # ERROR
    if state == 5:
        print("\n\nError occurred during inference")
        return 0

    # FINISH / STOP / MAX_TOKEN
    if state in (2, 3, 4):
        if result_callback.accumulated_tokens:
            try:
                safe_text = decode_safe(result_callback.accumulated_tokens)
                new_part = safe_text[len(result_callback.last_output_text):]
                if new_part:
                    print(new_part, end="", flush=True)
            except Exception as e:
                print(f"\n[Decode error: {e}]", flush=True)

        result_callback.accumulated_tokens.clear()
        result_callback.last_output_text = ""
        msg = {2: "Finished", 3: "Stop", 4: "Max new token reached"}.get(state, "Unknown")
        print(f"\n\n--------------------{msg}--------------------")
        return 0

    # WAITING
    if state == 1:
        print("\n\nWaiting for UTF-8 encoded character")
        return 0

    # NORMAL
    if state == 0:
        n = result_ptr.contents.num_tokens
        new_tokens = [result_ptr.contents.token_ids[i] for i in range(n)]
        result_callback.accumulated_tokens.extend(new_tokens)

        if first_token is None:
            first_token = time.perf_counter()

        try:
            safe_text = decode_safe(result_callback.accumulated_tokens)
            new_part = safe_text[len(result_callback.last_output_text):]
            if new_part:
                print(new_part, end="", flush=True)
                result_callback.last_output_text += new_part
        except Exception as e:
            print(f"\n[Temp decode error: {e}], waiting for more tokens", flush=True)
            return 0

    return 0


def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    text = text_ptr.decode("utf-8")
    inputs = tokenizer(text, return_tensors="np", truncation=True)
    tokens = inputs["input_ids"][0][:n_tokens_max]
    n_tokens = len(tokens)

    if n_tokens <= 0:
        print(f"Tokenizer failed for {text}")
        return n_tokens

    for i in range(n_tokens):
        tokens_ptr[i] = tokens[i]

    return n_tokens


# ============================================= Helpers =============================================

def print_run_state(state):
    print("\n--------------------Run State-------------------")
    for name in (
        "n_total_tokens",
        "n_max_tokens",
        "n_prefill_tokens",
        "n_decode_tokens",
        "n_cache_tokens",
    ):
        if hasattr(state, name):
            print(f"{name}: {getattr(state, name)}")
    print("------------------------------------------------")


def printf_perf(first_token_time, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time):
    print("\n-----------------------------------------------------------------------------------------")
    print(" %-10s | %-16s | %-8s | %-20s | %-20s " % (
        "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("-----------------------------------------------------------------------------------------")

    prefill_ms = (first_token_time - llm_start_time) * 1000.0
    if n_prefill_tokens == 0 or prefill_ms <= 0:
        prefill_tpt, prefill_tps = 0.0, 0.0
    else:
        prefill_tpt = prefill_ms / n_prefill_tokens
        prefill_tps = (n_prefill_tokens * 1000.0) / prefill_ms
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Prefill", prefill_ms, n_prefill_tokens, prefill_tpt, prefill_tps))

    decode_ms = (llm_end_time - first_token_time) * 1000.0
    if n_decode_tokens == 0 or decode_ms <= 0:
        decode_tpt, decode_tps = 0.0, 0.0
    else:
        decode_tpt = decode_ms / n_decode_tokens
        decode_tps = (n_decode_tokens * 1000.0) / decode_ms
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Generate", decode_ms, n_decode_tokens, decode_tpt, decode_tps))

    print("-----------------------------------------------------------------------------------------")


# ============================================= Main =============================================

def main():
    global tokenizer, first_token, _result_cb, _tokenizer_cb, _tok_ud, _embed_cb, _embed_ud

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="RKNN3 tie-word Session Test - Python port of rknn3_session_test_tie_word.cpp"
    )
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL, help="tie-word rknn model path")
    parser.add_argument("--weight_path", type=str, default=WEIGHT_MODEL, help="tie-word weight model path")
    parser.add_argument(
        "--tokenizer_path", type=str, default=TOKENIZER_PATH,
        help="Hugging Face tokenizer name or local tokenizer directory",
    )
    parser.add_argument("--max_new_token", type=int, default=MAX_NEW_TOKENS, help="max new tokens")
    parser.add_argument("--core_mask", type=str, default=CORE_MASK, help="NPU core mask in hex")
    parser.add_argument("--keep_history", type=int, choices=(0, 1), default=0, help="keep history, 0 or 1")
    parser.add_argument(
        "--ignore_eos_token", type=int, choices=(0, 1), default=0,
        help="ignore EOS token, 0 or 1",
    )
    parser.add_argument(
        "--decrypt_key_path", type=str, default=None,
        help="decrypt key/license file path for encrypted RKNN model",
    )
    parser.add_argument("--test_num", type=int, default=2, help="number of inference rounds")
    args = parser.parse_args()

    if args.max_new_token <= 0:
        print("max_new_token must be greater than 0")
        return -1
    if args.test_num <= 0:
        print("test_num must be greater than 0")
        return -1

    try:
        core_mask = int(args.core_mask, 16)
    except ValueError:
        print(f"Invalid core_mask: {args.core_mask}")
        return -1

    print("*******************************NEW TEST**********************************")
    print("RKNN_ENABLE_TIE_WORD_EMBEDDINGS=1")

    # tie-word: load tokenizer only; no embedding.bin is needed.
    print("--> Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print("done")

    rknn = RKNN3Lite(llm_mode=True, verbose=True)

    try:
        # Step 1: Load model
        print("--> Loading model")
        ret = rknn.load_rknn(args.rknn_path, args.weight_path)
        if ret != 0:
            print("Load model failed!")
            return ret
        print("done")

        # Step 2: Init runtime (without llm_args, separated flow)
        print("--> Init runtime environment")
        ret = rknn.init_runtime(
            target="rk1820",
            core_mask=core_mask,
            decrypt_key_path=args.decrypt_key_path,
        )
        if ret != 0:
            print("Init runtime environment failed!")
            return ret
        print("done")

        # Step 3: Query model info before init_llm_session
        print("--> Query model info")
        io_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
        if io_num is None:
            print("Query input/output num failed!")
            return -1
        print(f"model input num: {io_num.n_input}, output num: {io_num.n_output}")

        for i in range(io_num.n_output):
            output_attr = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=i)
            if output_attr is not None:
                dump_tensor_attr(output_attr, prefix="output")

        llm_config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
        if llm_config is None:
            print("Query LLM config failed!")
            return -1

        vocab_size = llm_config.vocab_size
        print(f"vocab_size={vocab_size}")
        print(
            f"max_ctx_len={llm_config.max_ctx_len}, "
            f"max_position_embeddings={llm_config.max_position_embeddings}"
        )


        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
        special_eos_id = -1 if args.ignore_eos_token else eos_token_id

        # Step 4: Build LLM args. This follows rknn_session_test.py style,
        # but keep_history / ignore_eos_token follow the C++ demo arguments.
        LLM_ARGS = [{
            "max_new_tokens": args.max_new_token,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 1.0,
            "repeat_penalty": 1.1,
            "vocab_size": vocab_size,
            "special_eos_id": special_eos_id,
            "max_context_len": llm_config.max_ctx_len,
            "keep_history": args.keep_history,
            "logits_name": b"output",
        }]

        # Step 5: Build callback.
        # tie-word uses tied weights for embeddings (RKNN_ENABLE_TIE_WORD_EMBEDDINGS=1),
        # so no external embedding.bin is needed.  However, the C API still
        callback = RKLLMCallback()

        _result_cb = LLMResultCallback(result_callback)
        callback.result_callback = _result_cb
        callback.result_userdata = None

        _tokenizer_cb = LLMTokenizerCallback(tokenizer_callback)
        callback.tokenizer_callback = _tokenizer_cb
        _tok_ud = ctypes.py_object(tokenizer)
        callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(_tok_ud), ctypes.c_void_p)

        # Step 6: Init LLM session (separated from init_runtime)
        print("--> Init LLM session")
        ret = rknn.init_llm_session(llm_args=LLM_ARGS, llm_callback=callback)
        if ret != 0:
            print("Init LLM session failed!")
            return ret
        print("done")

        # Step 7: Set SAVE_CHECKPOINT KV cache policy, matching the C++ demo.
        print("--> Set KV cache policy: SAVE_CHECKPOINT")
        kvcache_policy_param = RKNN3KVCachePolicyParam()
        kvcache_policy_param.save_checkpoint.checkpoint_start_pos = CHECKPOINT_START_POS
        kvcache_policy_param.save_checkpoint.checkpoint_interval = CHECKPOINT_INTERVAL
        kvcache_policy_param.save_checkpoint.max_checkpoint_count = MAX_CHECKPOINT_COUNT

        ret = rknn.set_kvcache_policy(
            RKNN3KVCachePolicy.RKNN3_KVCACHE_POLICY_SAVE_CHECKPOINT,
            kvcache_policy_param,
        )
        if ret != 0:
            print("Set kvcache policy failed!")
            return ret
        print(
            f"checkpoint_start_pos={CHECKPOINT_START_POS}, "
            f"checkpoint_interval={CHECKPOINT_INTERVAL}, "
            f"max_checkpoint_count={MAX_CHECKPOINT_COUNT}"
        )
        print("done")

        # Step 8: Run inference
        for i in range(args.test_num):
            prompt = RANDOM_PROMPTS[i % len(RANDOM_PROMPTS)]

            print(f"\n--------------------Input[{i}]--------------------")
            print(prompt)
            print("\n--------------------Output----------------------")

            first_token = None
            ret, perf = rknn.session_run(
                prompt=prompt,
                keep_history=bool(args.keep_history),
                max_new_tokens=args.max_new_token,
            )
            if ret != 0:
                print("RKNN llm inference failed!")
                return ret

            n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time = perf

            state = rknn.session_query_state()
            if state is None:
                print("Query session state failed!")
                return -1

            print_run_state(state)

            # Match the C++ behavior: clear all KV cache when the current context is
            # close enough to the maximum that another max_new_token decode may overflow.
            n_total_tokens = getattr(state, "n_total_tokens", None)
            n_max_tokens = getattr(state, "n_max_tokens", None)
            if (
                n_total_tokens is not None
                and n_max_tokens is not None
                and n_total_tokens >= (n_max_tokens - args.max_new_token)
            ):
                print("--> Context is near limit, clear all KV cache")
                ret = rknn.clear_kvcache(RKNN3KVCacheClearPolicy.RKNN3_KVCACHE_CLEAR_ALL)
                if ret != 0:
                    print("Clear kvcache failed!")
                    return ret
                print("done")

            if first_token is not None:
                printf_perf(
                    first_token,
                    n_decode_tokens,
                    n_prefill_tokens,
                    llm_start_time,
                    llm_end_time,
                )
            first_token = None

        print("done")
        return 0

    finally:
        # Step 9: Release
        rknn.release()
        print("*******************************END TEST**********************************")


if __name__ == "__main__":
    raise SystemExit(main())