import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
from transformers import AutoTokenizer
import ctypes
import numpy as np
import time
from rknn3lite.api import RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMGetEmbedCallback, LLMTokenizerCallback
from rknn3lite.api.rknn3_types import RKNN3QueryCmd, RKNN3KVCachePolicy, RKNN3KVCacheClearPolicy, dump_tensor_attr

# ============================================= Default Config =============================================

RKNN_MODEL     = 'Qwen2.5-0.5B-Instruct.rknn'
WEIGHT_MODEL   = 'Qwen2.5-0.5B-Instruct.weight'
EMBED_PATH     = 'Qwen2.5-0.5B-Instruct.embed.bin'
TOKENIZER_PATH = 'Qwen/Qwen2.5-0.5B-Instruct'

system_prompt  = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
prompt_prefix  = "<|im_start|>user\n"
prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"

tokenizer   = None
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
        return text.split('\ufffd', 1)[0] if '\ufffd' in text else text

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

def printf_perf(first_token_time, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time):
    print("\n-----------------------------------------------------------------------------------------")
    print(" %-10s | %-16s | %-8s | %-20s | %-20s " % (
        "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("-----------------------------------------------------------------------------------------")

    prefill_ms = (first_token_time - llm_start_time) * 1000.0
    if n_prefill_tokens == 0:
        prefill_tpt, prefill_tps = 0.0, 0.0
    else:
        prefill_tpt = prefill_ms / n_prefill_tokens
        prefill_tps = (n_prefill_tokens * 1000.0) / prefill_ms
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Prefill", prefill_ms, n_prefill_tokens, prefill_tpt, prefill_tps))

    decode_ms = (llm_end_time - first_token_time) * 1000.0
    if n_decode_tokens == 0:
        decode_tpt, decode_tps = 0.0, 0.0
    else:
        decode_tpt = decode_ms / n_decode_tokens
        decode_tps = (n_decode_tokens * 1000.0) / decode_ms
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Generate", decode_ms, n_decode_tokens, decode_tpt, decode_tps))

    print("-----------------------------------------------------------------------------------------")

# ============================================= Main =============================================

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="RKNN3 Session Test - Demonstrates separated init flow: "
                            "init_runtime -> rknn3_query -> init_llm_session -> session_run")
    parser.add_argument("--rknn_path",      type=str, default=RKNN_MODEL,     help="rknn model path")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH, help="huggingface tokenizer path")
    parser.add_argument("--embed_path",     type=str, default=EMBED_PATH,     help="embedding bin path")
    parser.add_argument("--max_new_token",  type=int, default=256,            help="max new tokens")
    parser.add_argument("--core_mask",      type=str, default="0xff",         help="NPU core mask in hex")
    parser.add_argument("--decrypt_key_path", type=str, default=None,         help="decrypt key/license file path for encrypted RKNN model")
    args = parser.parse_args()

    core_mask = int(args.core_mask, 16)

    # Load tokenizer & embedding
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    embeds_data = np.fromfile(args.embed_path, dtype=np.float16)

    # Create RKNN object
    rknn = RKNN3Lite(llm_mode=True, verbose=True)

    # Step 1: Load model
    print('--> Loading model')
    ret = rknn.load_rknn(args.rknn_path, args.rknn_path.replace('.rknn', '.weight'))
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Step 2: Init runtime (without llm_args, separated flow)
    print('--> Init runtime environment')
    # ret = rknn.init_runtime(target='rk1820', core_mask=core_mask, decrypt_key_data=key_data) # key_data可从文件读取
    ret = rknn.init_runtime(target='rk1820', core_mask=core_mask, decrypt_key_path=args.decrypt_key_path)
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

    # Step 4: Build LLM args (using queried info)
    LLM_ARGS = [{
        "max_new_tokens":    args.max_new_token,
        "top_k":             1,
        "top_p":             0.9,
        "temperature":       1.0,
        "repeat_penalty":    1.1,
        "vocab_size":        vocab_size,
        "special_eos_id":    tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1,
        "max_context_len":   llm_config.max_ctx_len,
        "keep_history":      1,
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

    # Step 7: Set chat template
    ret = rknn.set_chat_template(system_prompt, prompt_prefix, prompt_postfix)
    if ret != 0:
        print('Set chat template failed!')
        exit(ret)

    # Step 8: Set KV cache policy
    ret = rknn.set_kvcache_policy(RKNN3KVCachePolicy.RKNN3_KVCACHE_POLICY_NORMAL)
    if ret != 0:
        print('Set kvcache policy failed!')
        exit(ret)

    # Step 9: Run inference
    prompts = [
        "请解释一下相对论的基本概念。",
        "归纳上面内容",
        "Please explain the basic concept of relativity",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n--------------------Input[{i}]--------------------")
        print(prompt)
        print("\n--------------------Output----------------------")

        first_token = None
        ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = rknn.session_run(prompt=prompt, keep_history=True)
        if ret != 0:
            print('RKNN llm inference failed!')
            exit(ret)

        if first_token is not None:
            printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time)
        first_token = None

    print('done')

    # Step 10: Release
    rknn.release()
