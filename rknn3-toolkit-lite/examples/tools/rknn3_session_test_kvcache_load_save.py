import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

import ctypes
import time
from argparse import ArgumentParser

import numpy as np
from transformers import AutoTokenizer

from rknn3lite.api import (
    RKNN3Lite,
    RKLLMCallback,
    LLMResultCallback,
    LLMGetEmbedCallback,
    LLMTokenizerCallback,
)
from rknn3lite.api.rknn3_types import (
    RKNN3QueryCmd,
    RKNN3KVCachePolicy,
    RKNN3KVCacheClearPolicy,
    dump_tensor_attr,
)

# ============================================= Default Config =============================================

RKNN_MODEL = "Qwen2.5-0.5B-Instruct.rknn"
WEIGHT_MODEL = "Qwen2.5-0.5B-Instruct.weight"
EMBED_PATH = "Qwen2.5-0.5B-Instruct.embed.bin"
TOKENIZER_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
KVCACHE_PATH = "kvcache.bin"

# C++ demo sets chat template to empty strings.
# Keep the same behavior here so the input prompt is passed as-is.
system_prompt = ""
prompt_prefix = ""
prompt_postfix = ""

tokenizer = None
embeds_data = None
first_token = None


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
    if isinstance(text_ptr, bytes):
        text = text_ptr[:text_len].decode("utf-8", errors="ignore") if text_len > 0 else text_ptr.decode("utf-8", errors="ignore")
    else:
        text = ctypes.string_at(text_ptr, text_len).decode("utf-8", errors="ignore") if text_len > 0 else text_ptr.decode("utf-8")

    inputs = tokenizer(text, return_tensors="np", truncation=True)
    tokens = inputs["input_ids"][0][:n_tokens_max]
    n_tokens = len(tokens)
    if n_tokens <= 0:
        print(f"Tokenizer failed for {text}")
        return n_tokens

    for i in range(n_tokens):
        tokens_ptr[i] = int(tokens[i])
    return n_tokens


def embed_callback(userdata, tokens_ptr, num_tokens, embed, length):
    global embeds_data

    embedding_dim = embeds_data.shape[1]
    expected_len = num_tokens * embedding_dim * np.dtype(np.float16).itemsize
    if length != expected_len:
        print("invalid embed buffer")
        return -1

    dst = np.ctypeslib.as_array(
        ctypes.cast(embed, ctypes.POINTER(ctypes.c_uint16)),
        shape=(num_tokens * embedding_dim,),
    ).view(np.float16)
    tokens = [tokens_ptr[i] for i in range(num_tokens)]
    dst[:] = embeds_data[tokens].ravel()
    return 0


# ============================================= Helpers =============================================

def printf_perf(first_token_time, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time):
    if first_token_time is None:
        print("\n[Perf] first token time is unavailable, skip performance statistics")
        return

    print("\nPerformance Statistics: ")
    print("-----------------------------------------------------------------------------------------")
    print(" %-10s | %-16s | %-8s | %-20s | %-20s " % (
        "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("-----------------------------------------------------------------------------------------")

    prefill_ms = max((first_token_time - llm_start_time) * 1000.0, 0.0)
    if n_prefill_tokens == 0 or prefill_ms == 0:
        prefill_tpt, prefill_tps = 0.0, 0.0
    else:
        prefill_tpt = prefill_ms / n_prefill_tokens
        prefill_tps = (n_prefill_tokens * 1000.0) / prefill_ms
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Prefill", prefill_ms, n_prefill_tokens, prefill_tpt, prefill_tps))

    decode_ms = max((llm_end_time - first_token_time) * 1000.0, 0.0)
    if n_decode_tokens == 0 or decode_ms == 0:
        decode_tpt, decode_tps = 0.0, 0.0
    else:
        decode_tpt = decode_ms / n_decode_tokens
        decode_tps = (n_decode_tokens * 1000.0) / decode_ms
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Generate", decode_ms, n_decode_tokens, decode_tpt, decode_tps))
    print("-----------------------------------------------------------------------------------------")


def build_callback():
    callback = RKLLMCallback()

    callback.result_callback = LLMResultCallback(result_callback)
    callback.result_userdata = None

    callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    _tok_ud = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(_tok_ud), ctypes.c_void_p)

    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    _emb_ud = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(_emb_ud), ctypes.c_void_p)

    # Keep userdata Python objects alive for the lifetime of callback.
    callback._py_userdata_refs = (_tok_ud, _emb_ud)
    return callback


def load_kvcache(rknn, kvcache_path, load_from_data=True):
    if load_from_data:
        with open(kvcache_path, "rb") as fp:
            kvcache_bytes = fp.read()
        if len(kvcache_bytes) == 0:
            print(f"kvcache file is empty: {kvcache_path}")
            return -1

        buf = ctypes.create_string_buffer(kvcache_bytes)
        return rknn.load_kvcache(
            kvcache_data=ctypes.cast(buf, ctypes.c_void_p),
            size=len(kvcache_bytes),
        )

    return rknn.load_kvcache(kvcache_path=kvcache_path)


def get_state_value(state, name, default=0):
    return getattr(state, name, default) if state is not None else default


# ============================================= Main =============================================

if __name__ == "__main__":
    parser = ArgumentParser(
        description="RKNN3 Session Test - KV cache load/save demo, Python version of "
                    "rknn3_session_test_kvcache_load_save.cpp"
    )
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL, help="rknn model path")
    parser.add_argument("--weight_path", type=str, default=None, help="weight path, default: replace .rknn with .weight")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH, help="huggingface tokenizer path")
    parser.add_argument("--embed_path", type=str, default=EMBED_PATH, help="embedding bin path")
    parser.add_argument("--kvcache_path", type=str, default=KVCACHE_PATH, help="kv cache save/load file path")
    parser.add_argument("--max_context_len", type=int, default=0, help="expected max context len; 0 means use model config")
    parser.add_argument("--max_new_token", type=int, default=256, help="max new tokens")
    parser.add_argument("--core_mask", type=str, default="0xff", help="NPU core mask in hex")
    parser.add_argument("--load_kvcache_from_path", action="store_true", help="load kvcache by path instead of reading file into memory")
    args = parser.parse_args()

    weight_path = args.weight_path or args.rknn_path.replace(".rknn", ".weight")
    core_mask = int(args.core_mask, 16)
    rknn = None

    random_prompts = [
        "请解释一下相对论的基本概念。",
        "请解释一下相对论的基本概念。 相对论是物理学中的一个重要分支，它研究了时间和空间在不同参考系（即不同的观察者）下的性质和规律。相对论",
        "请解释一下相对论的基本概念。 相对论是物理学中的一个重要分支，它研究了时间和空间在不同参考系（即不同的观察者）下的性质和规律。相对论",
        "请解释一下相对论的基本概念。",
        "的基本概念包括：\n\n1. 独立性：在相对论中，时间和空间是独立的物理量。这意味着无论观察者选择什么样的参考",
    ]

    print("*******************************NEW TEST**********************************")

    try:
        # Load tokenizer & embedding
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        embeds_data = np.fromfile(args.embed_path, dtype=np.float16)

        # Create RKNN object
        rknn = RKNN3Lite(llm_mode=True, verbose=True)

        # Step 1: Load model
        print("--> Loading model")
        ret = rknn.load_rknn(args.rknn_path, weight_path)
        if ret != 0:
            print("Load model failed!")
            raise SystemExit(ret)
        print("done")

        # Step 2: Init runtime without llm_args, same separated flow as rknn_session_test.py
        print("--> Init runtime environment")
        ret = rknn.init_runtime(target="rk1820", core_mask=core_mask)
        if ret != 0:
            print("Init runtime environment failed!")
            raise SystemExit(ret)
        print("done")

        # Step 3: Query model info before init_llm_session
        print("--> Query model info")
        io_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
        print(f"model input num: {io_num.n_input}, output num: {io_num.n_output}")

        for i in range(io_num.n_output):
            output_attr = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=i)
            dump_tensor_attr(output_attr, prefix="output")

        llm_config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
        vocab_size = llm_config.vocab_size
        embedding_dim = embeds_data.size // vocab_size
        embeds_data = embeds_data.reshape(vocab_size, embedding_dim)
        print(f"vocab_size={vocab_size}, embedding_dim={embedding_dim}")

        print("\n=============================================================")
        print("%-32s: %-8d" % ("Max Context Length", llm_config.max_ctx_len))
        print("%-32s: %-8d" % ("Max Position Embeddings", llm_config.max_position_embeddings))
        print("=============================================================\n")

        expected_ctx_len = args.max_context_len or llm_config.max_ctx_len
        if expected_ctx_len != llm_config.max_ctx_len:
            print(
                f"\033[33mmax_context_len != llm_config.max_ctx_len, "
                f"max_context_len {expected_ctx_len}, llm_config.max_ctx_len {llm_config.max_ctx_len}\033[0m"
            )
            print(f"\033[33mplease set --max_context_len to {llm_config.max_ctx_len}\033[0m")
            raise SystemExit(-1)

        # Step 4: Build LLM args
        LLM_ARGS = [{
            "max_new_tokens": args.max_new_token,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 1.0,
            "repeat_penalty": 1.1,
            "vocab_size": vocab_size,
            "special_eos_id": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1,
            "max_context_len": llm_config.max_ctx_len,
            "keep_history": 1,
            "logits_name": b"output",
        }]

        # Step 5: Init LLM session
        callback = build_callback()
        print("--> Init LLM session")
        ret = rknn.init_llm_session(llm_args=LLM_ARGS, llm_callback=callback)
        if ret != 0:
            print("Init LLM session failed!")
            raise SystemExit(ret)
        print("done")

        # Step 6: Empty chat template, same as C++ demo
        ret = rknn.set_chat_template(system_prompt, prompt_prefix, prompt_postfix)
        if ret != 0:
            print("Set chat template failed!")
            raise SystemExit(ret)

        # Step 7: Set KV cache policy
        ret = rknn.set_kvcache_policy(RKNN3KVCachePolicy.RKNN3_KVCACHE_POLICY_NORMAL)
        if ret != 0:
            print("Set kvcache policy failed!")
            raise SystemExit(ret)

        # Step 8: Run KV cache load/save test
        for i, prompt in enumerate(random_prompts):
            keep_history = False if i < 3 else True

            print(f"\n--------------------Input[{i}]--------------------")
            print(prompt)
            print(f"[keep_history={int(keep_history)}]")
            print("\n--------------------Output----------------------")

            # C++ demo loads kvcache before i == 1 or i == 4.
            if i in (1, 4):
                if not os.path.exists(args.kvcache_path):
                    print(f"kvcache file not found: {args.kvcache_path}")
                    raise SystemExit(-1)

                print(f"\n--> Load kvcache: {args.kvcache_path}")
                ret = load_kvcache(
                    rknn,
                    args.kvcache_path,
                    load_from_data=not args.load_kvcache_from_path,
                )
                if ret != 0:
                    print(f"load kvcache failed, ret={ret}")
                    raise SystemExit(ret)
                print("done")

            first_token = None
            ret, perf = rknn.session_run(
                prompt=prompt,
                keep_history=keep_history,
                max_new_tokens=args.max_new_token,
                enable_thinking=False,
            )
            if ret != 0:
                print(f"RKNN llm inference failed, ret={ret}")
                raise SystemExit(ret)

            n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time = perf

            # C++ demo saves kvcache after i == 0 or i == 3.
            if i in (0, 3):
                print(f"\n--> Save kvcache: {args.kvcache_path}")
                ret = rknn.save_kvcache(args.kvcache_path)
                if ret != 0:
                    print(f"save kvcache failed, ret={ret}")
                    raise SystemExit(ret)
                print("done")

            state = rknn.session_query_state()
            if state is not None:
                state_prefill_tokens = get_state_value(state, "n_prefill_tokens", n_prefill_tokens)
                state_decode_tokens = get_state_value(state, "n_decode_tokens", n_decode_tokens)
                n_total_tokens = get_state_value(state, "n_total_tokens", 0)
                n_max_tokens = get_state_value(state, "n_max_tokens", llm_config.max_ctx_len)

                # Prefer state counters, consistent with the C++ demo.
                n_prefill_tokens = state_prefill_tokens
                n_decode_tokens = state_decode_tokens

                print(
                    f"\n[State] n_total_tokens={n_total_tokens}, "
                    f"n_max_tokens={n_max_tokens}, "
                    f"n_prefill_tokens={n_prefill_tokens}, "
                    f"n_decode_tokens={n_decode_tokens}"
                )

                # C++ demo clears when nearly full, or after i == 1 / i == 2.
                if n_total_tokens >= (n_max_tokens - args.max_new_token) or i in (1, 2):
                    print("--> Clear kvcache")
                    ret = rknn.clear_kvcache(RKNN3KVCacheClearPolicy.RKNN3_KVCACHE_CLEAR_ALL)
                    if ret != 0:
                        print(f"clear kvcache failed, ret={ret}")
                        raise SystemExit(ret)
                    print("done")
            else:
                print("\n[Warn] session_query_state failed, skip clear-by-state check")
                if i in (1, 2):
                    print("--> Clear kvcache")
                    ret = rknn.clear_kvcache(RKNN3KVCacheClearPolicy.RKNN3_KVCACHE_CLEAR_ALL)
                    if ret != 0:
                        print(f"clear kvcache failed, ret={ret}")
                        raise SystemExit(ret)
                    print("done")

            printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time)
            first_token = None

    finally:
        if rknn is not None:
            rknn.release()
        print("*******************************END TEST**********************************")
