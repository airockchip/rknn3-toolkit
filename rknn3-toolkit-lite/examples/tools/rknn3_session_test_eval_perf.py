import os
import sys
import ctypes
import numpy as np
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
from transformers import AutoTokenizer
from rknn3lite.api import RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMGetEmbedCallback, LLMTokenizerCallback
from rknn3lite.api.rknn3_types import RKNN3QueryCmd, RKNN3KVCachePolicy, RKNN3KVCacheClearPolicy, RKNN3LLMTaskType, dump_tensor_attr

# ============================================= Default Config =============================================

RKNN_MODEL     = 'Qwen2.5-0.5B-Instruct.rknn'
WEIGHT_MODEL   = 'Qwen2.5-0.5B-Instruct.weight'
EMBED_PATH     = 'Qwen2.5-0.5B-Instruct.embed.bin'
TOKENIZER_PATH = 'Qwen/Qwen2.5-0.5B-Instruct'

tokenizer   = None
embeds_data = None
first_token = None

# ============================================= Callbacks =============================================

def result_callback(userdata, result_ptr, state):
    """Result callback - only records first token time for performance measurement, no printing."""
    global first_token

    # ERROR
    if state == 5:
        return 0

    # FINISH / STOP / MAX_TOKEN
    if state in (2, 3, 4):
        return 0

    # WAITING
    if state == 1:
        return 0

    # NORMAL
    if state == 0:
        if first_token is None:
            first_token = time.perf_counter()

    return 0


def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    global tokenizer
    text = text_ptr.decode('utf-8')
    inputs = tokenizer(text, return_tensors='np', truncation=True)
    tokens = inputs['input_ids'][0][:n_tokens_max]
    n_tokens = len(tokens)
    if n_tokens <= 0:
        print(f"Tokenizer failed for {text}")
        return n_tokens
    for i in range(n_tokens):
        tokens_ptr[i] = tokens[i]
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
        shape=(num_tokens * embedding_dim,)
    ).view(np.float16)
    tokens = [tokens_ptr[i] for i in range(num_tokens)]
    dst[:] = embeds_data[tokens].ravel()
    return 0

# ============================================= Performance =============================================

def print_perf(first_token_time, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time, run_index):
    """Print performance statistics for a single run."""
    prefill_ms = (first_token_time - llm_start_time) * 1000.0
    if n_prefill_tokens == 0:
        prefill_tpt, prefill_tps = 0.0, 0.0
    else:
        prefill_tpt = prefill_ms / n_prefill_tokens
        prefill_s = prefill_ms / 1000.0
        prefill_tps = n_prefill_tokens / prefill_s if prefill_s > 0 else 0.0

    decode_ms = (llm_end_time - first_token_time) * 1000.0
    if n_decode_tokens == 0:
        decode_tpt, decode_tps = 0.0, 0.0
    else:
        decode_tpt = decode_ms / n_decode_tokens
        decode_s = decode_ms / 1000.0
        decode_tps = n_decode_tokens / decode_s if decode_s > 0 else 0.0

    print(f"\nPerformance Statistics for Run {run_index}: ")
    print("-----------------------------------------------------------------------------------------")
    print(" %-10s | %-16s | %-8s | %-20s | %-20s " % (
        "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("-----------------------------------------------------------------------------------------")
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Prefill", prefill_ms, n_prefill_tokens, prefill_tpt, prefill_tps))
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Generate", decode_ms, n_decode_tokens, decode_tpt, decode_tps))
    print("-----------------------------------------------------------------------------------------")

    return prefill_ms, prefill_tps, decode_tpt, decode_tps


def print_summary(llm_config, n_input_tokens, n_decode_tokens, prefill_ms, prefill_tps, decode_tpt, decode_tps):
    """Print performance summary table."""
    print("\n\n====================================================================================================================")
    print("%*s" % (60 + len("Performance Summary") // 2, "Performance Summary"))
    print("====================================================================================================================")
    print("--------------------------------------------------------------------------------------------------------------------")
    print(" %-15s | %-11s | %-13s | %-11s | %-9s | %-10s | %-13s | %-10s " % (
        "Model Context", "Rope Length", "Input Tokens",
        "New Tokens", "TTFT(ms)", "TPOT(ms)", "Prefill TPS", "Decode TPS"))
    print("--------------------------------------------------------------------------------------------------------------------")
    print(" %-15d | %-11d | %-13d | %-11d | %-9.2f | %-10.2f | %-13.2f | %-10.2f " % (
        llm_config.max_ctx_len, llm_config.max_position_embeddings,
        n_input_tokens, n_decode_tokens + 1,
        prefill_ms, decode_tpt, prefill_tps, decode_tps))
    print("--------------------------------------------------------------------------------------------------------------------")

# ============================================= Main =============================================

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="RKNN3 Session Eval Perf - Performance evaluation using token-based input.\n"
                    "Example: python rknn3_session_test_eval_perf.py "
                    "--rknn_path Qwen2.5-0.5B.rknn --tokenizer_path Qwen/Qwen2.5-0.5B-Instruct "
                    "--embed_path Qwen2.5-0.5B.embed.bin --max_context_len 1024 "
                    "--n_input_tokens 128 --max_new_tokens 128 --core_mask 0xff")
    parser.add_argument("--rknn_path",       type=str, default=RKNN_MODEL,     help="rknn model path")
    parser.add_argument("--tokenizer_path",  type=str, default=TOKENIZER_PATH, help="huggingface tokenizer path")
    parser.add_argument("--embed_path",      type=str, default=EMBED_PATH,     help="embedding bin path")
    parser.add_argument("--max_context_len", type=int, default=1024,           help="max context length")
    parser.add_argument("--n_input_tokens",  type=int, default=128,            help="number of input tokens")
    parser.add_argument("--max_new_tokens",  type=int, default=128,            help="max new tokens to generate")
    parser.add_argument("--core_mask",       type=str, default="0xff",         help="NPU core mask in hex")
    args = parser.parse_args()

    core_mask = int(args.core_mask, 16)
    weight_path = args.rknn_path.replace('.rknn', '.weight')
    n_input_tokens = args.n_input_tokens
    max_new_tokens = args.max_new_tokens
    test_num = 3

    # Load tokenizer & embedding
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    embeds_data = np.fromfile(args.embed_path, dtype=np.float16)

    print("*******************************NEW TEST**********************************")

    # Create RKNN object
    rknn = RKNN3Lite(llm_mode=True, verbose=True)

    # Step 1: Load model
    print('--> Loading model')
    ret = rknn.load_rknn(args.rknn_path, weight_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Step 2: Init runtime (without llm_args, separated flow)
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk1820', core_mask=core_mask)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Step 3: Query model info (before init_llm_session)
    print('--> Query model info')
    io_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
    print(f"model input num: {io_num.n_input}, output num: {io_num.n_output}")

    for i in range(io_num.n_output):
        output_attr = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=i)
        dump_tensor_attr(output_attr, prefix="output")

    llm_config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
    vocab_size    = llm_config.vocab_size
    embedding_dim = embeds_data.size // vocab_size
    embeds_data   = embeds_data.reshape(vocab_size, embedding_dim)
    print(f"vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    print(f"max_ctx_len={llm_config.max_ctx_len}, max_position_embeddings={llm_config.max_position_embeddings}")

    # Validate max_context_len
    max_context_len = args.max_context_len
    if max_context_len != llm_config.max_ctx_len:
        if max_context_len < llm_config.max_ctx_len:
            print(f"\033[33mWarning: max_context_len ({max_context_len}) is less than "
                  f"llm_config.max_ctx_len ({llm_config.max_ctx_len}).\033[0m")
            print(f"\033[33mIt's recommended to set <max_context_len> to {llm_config.max_ctx_len}.\033[0m")
        elif max_context_len > llm_config.max_ctx_len:
            print(f"\033[33mError: max_context_len ({max_context_len}) is greater than "
                  f"llm_config.max_ctx_len ({llm_config.max_ctx_len}).\033[0m")
            print(f"\033[33mPlease set <max_context_len> to {llm_config.max_ctx_len}.\033[0m")
            rknn.release()
            exit(-1)

    # Step 4: Build LLM args (using queried info, with ignore_eos_token for perf test)
    LLM_ARGS = [{
        "max_new_tokens":    max_new_tokens,
        "top_k":             1,
        "top_p":             0.9,
        "temperature":       1.0,
        "repeat_penalty":    1.1,
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "vocab_size":        vocab_size,
        "special_eos_id":    tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1,
        "max_context_len":   llm_config.max_ctx_len,
        "ignore_eos_token":  True,       # Force generating max_new_tokens for performance testing
        "keep_history":      0,
        "logits_name":       b"output",
    }]

    # Step 5: Build callback
    callback = RKLLMCallback()
    callback.result_callback = LLMResultCallback(result_callback)
    callback.result_userdata = None

    callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    _tok_ud = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(_tok_ud), ctypes.c_void_p)

    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    _emb_ud = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(_emb_ud), ctypes.c_void_p)

    # Step 6: Init LLM session (separated from init_runtime)
    print('--> Init LLM session')
    ret = rknn.init_llm_session(llm_args=LLM_ARGS, llm_callback=callback)
    if ret != 0:
        print('Init LLM session failed!')
        exit(ret)
    print('done')

    # Print test configuration
    task_type_str = ("RKNN3_LLM_TASK_GENERATE"
                     if llm_config.task_type == RKNN3LLMTaskType.RKNN3_LLM_TASK_GENERATE
                     else "RKNN3_LLM_TASK_EMBEDDING")
    model_type_str = (llm_config.model_type.decode('utf-8', errors='ignore')
                      if llm_config.model_type else "Unknown")

    print()
    print("=============================================================")
    print("%-32s: %-8d" % ("Max Context Length",       llm_config.max_ctx_len))
    print("%-32s: %-8d" % ("Max Position Embeddings",  llm_config.max_position_embeddings))
    print("%-32s: %s"   % ("Model Type",               model_type_str))
    print("%-32s: %s"   % ("Task Type",                task_type_str))
    print("%-32s: %-8d" % ("Number of Input Tokens",   n_input_tokens))
    print("%-32s: %-8d" % ("Max New Tokens",           max_new_tokens))
    print("=============================================================")
    print()

    # Step 7: Generate fixed random tokens for performance testing (same as C++ with seed 42)
    np.random.seed(42)
    input_tokens = np.random.randint(0, 1024, size=(n_input_tokens,), dtype=np.int32)

    # Pre-compute embedding for the fixed tokens: shape (1, n_input_tokens, embedding_dim)
    input_embeds = embeds_data[input_tokens].reshape(1, n_input_tokens, embedding_dim).astype(np.float16)

    # Step 8: Run performance test loop
    for i in range(test_num):
        first_token = None

        # Run inference with embed input (pre-computed from fixed random tokens)
        ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = rknn.session_run(
            inputs=None,
            prompt=None,
            embeds=input_embeds,
            keep_history=False,
            max_new_tokens=max_new_tokens,
        )
        if ret != 0:
            print(f'rknn3_session_run failed, ret={ret}')
            break

        # Clear KV cache (keep system prompt, matching C++ RKNN3_KVCACHE_KEEP_SYSTEM_PROMPT)
        ret = rknn.clear_kvcache(RKNN3KVCacheClearPolicy.RKNN3_KVCACHE_KEEP_SYSTEM_PROMPT)
        if ret != 0:
            print(f'rknn3_session_clear_kvcache failed, ret={ret}')
            break

        # Print performance statistics
        if first_token is not None:
            prefill_ms, prefill_tps, decode_tpt, decode_tps = print_perf(
                first_token, n_decode_tokens, n_prefill_tokens,
                llm_start_time, llm_end_time, i)
        else:
            print(f"\nRun {i}: Warning - first token time not captured")
            prefill_ms, prefill_tps, decode_tpt, decode_tps = 0.0, 0.0, 0.0, 0.0

        # Print summary at the last iteration
        if i == test_num - 1:
            print_summary(llm_config, n_input_tokens, n_decode_tokens,
                          prefill_ms, prefill_tps, decode_tpt, decode_tps)

        sys.stdout.flush()

    print('done')

    # Step 9: Release
    rknn.release()

    print("\n*******************************END TEST**********************************")
    sys.stdout.flush()
