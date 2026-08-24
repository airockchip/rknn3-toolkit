import os
import ctypes
import numpy as np
import time
from argparse import ArgumentParser
from transformers import AutoTokenizer

from rknn3lite.api import RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMGetEmbedCallback, LLMTokenizerCallback
from rknn3lite.api.rknn3_types import RKNN3QueryCmd, RKNN3KVCachePolicy, RKNN3KVCacheClearPolicy

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
        msg = {2: "Base Finished", 3: "Base Stop", 4: "Base max new token reached"}.get(state, "Unknown")
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

def result_callback_lora(userdata, result_ptr, state):
    global tokenizer, first_token

    if not hasattr(result_callback_lora, "accumulated_tokens"):
        result_callback_lora.accumulated_tokens = []
        result_callback_lora.last_output_text = ""

    def decode_safe(tokens):
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text.split('\ufffd', 1)[0] if '\ufffd' in text else text

    # ERROR
    if state == 5:
        print("\n\nError occurred during inference")
        return 0

    # FINISH / STOP / MAX_TOKEN
    if state in (2, 3, 4):
        if result_callback_lora.accumulated_tokens:
            try:
                safe_text = decode_safe(result_callback_lora.accumulated_tokens)
                new_part = safe_text[len(result_callback_lora.last_output_text):]
                if new_part:
                    print(new_part, end="", flush=True)
            except Exception as e:
                print(f"\n[Decode error: {e}]", flush=True)
        result_callback_lora.accumulated_tokens.clear()
        result_callback_lora.last_output_text = ""
        msg = {2: "LoRA Finished", 3: "LoRA Stop", 4: "LoRA max new token reached"}.get(state, "Unknown")
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
        result_callback_lora.accumulated_tokens.extend(new_tokens)
        if first_token is None:
            first_token = time.perf_counter()
        try:
            safe_text = decode_safe(result_callback_lora.accumulated_tokens)
            new_part = safe_text[len(result_callback_lora.last_output_text):]
            if new_part:
                print(new_part, end="", flush=True)
                result_callback_lora.last_output_text += new_part
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
        print(f"tokenizer failed for {text}")
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
    print("\nPerformance Statistics: ")
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
    import sys; sys.stdout.flush()

# ============================================= Main =============================================

if __name__ == '__main__':
    parser = ArgumentParser(description="RKNN3 Session LoRA Test")
    parser.add_argument("--rknn_path",      type=str, help="rknn model path")
    parser.add_argument("--weight_path",    type=str, help="rknn weight path")
    parser.add_argument("--lora_path",      type=str, help="lora weight path")
    parser.add_argument("--tokenizer_path", type=str, help="huggingface tokenizer path")
    parser.add_argument("--embed_path",     type=str, help="embedding bin path")
    parser.add_argument("--max_context_len",type=int, help="max context len")
    parser.add_argument("--max_new_token",  type=int, help="max new tokens")
    parser.add_argument("--core_mask",      type=str, help="NPU core mask in hex")
    args = parser.parse_args()

    core_mask = int(args.core_mask, 16)

    # 1. Load Tokenizer & Embedding
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    embeds_data = np.fromfile(args.embed_path, dtype=np.float16)

    random_prompts = [
        "请解释一下相对论的基本概念。",
        "请解释一下相对论的基本概念。"
    ]

    print("*******************************NEW TEST**********************************")

    # 2. Init Base Runtime
    rknn = RKNN3Lite(llm_mode=True, verbose=False)
    ret = rknn.load_rknn(args.rknn_path, args.weight_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)

    ret = rknn.init_runtime(target='rk1820', core_mask=core_mask)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)

    # 3. Model info queries mapping
    io_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
    print(f"model input num: {io_num.n_input}, output num: {io_num.n_output}")

    for i in range(io_num.n_output):
        output_attr = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=i)
        print(f"output_callback output tensor[{i}]: {output_attr.name.decode('utf-8')}")

    llm_config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
    vocab_size = llm_config.vocab_size
    embedding_dim = embeds_data.size // vocab_size
    embeds_data = embeds_data.reshape(vocab_size, embedding_dim)

    LLM_ARGS = [{
        "max_new_tokens":    args.max_new_token,
        "top_k":             1,
        "top_p":             0.9,
        "temperature":       1.0,
        "repeat_penalty":    1.1,
        "frequency_penalty": 0.0,
        "presence_penalty":  0.0,
        "vocab_size":        vocab_size,
        "special_eos_id":    tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1,
        "max_context_len":   llm_config.max_ctx_len,
        "keep_history":      1,
        "logits_name":       b"output",
    }]

    # 4. Build 2 sets of callbacks
    cb_base = RKLLMCallback()
    cb_base.result_callback = LLMResultCallback(result_callback)
    cb_base.result_userdata = None
    cb_base.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    _tok_ud = ctypes.py_object(tokenizer)
    cb_base.tokenizer_userdata = ctypes.cast(ctypes.pointer(_tok_ud), ctypes.c_void_p)
    cb_base.embed_callback = LLMGetEmbedCallback(embed_callback)
    _emb_ud = ctypes.py_object(embeds_data)
    cb_base.embed_userdata = ctypes.cast(ctypes.pointer(_emb_ud), ctypes.c_void_p)

    cb_lora = RKLLMCallback()
    cb_lora.result_callback = LLMResultCallback(result_callback_lora)
    cb_lora.result_userdata = None
    cb_lora.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    cb_lora.tokenizer_userdata = ctypes.cast(ctypes.pointer(_tok_ud), ctypes.c_void_p)
    cb_lora.embed_callback = LLMGetEmbedCallback(embed_callback)
    cb_lora.embed_userdata = ctypes.cast(ctypes.pointer(_emb_ud), ctypes.c_void_p)

    # 5. 测试释放内部分配的kvcache mem (Match C++ Logic)
    ret = rknn.init_llm_session(llm_args=LLM_ARGS, llm_callback=cb_base, session_index=0)
    if ret != 0:
        print("dummy rknn.init_llm_session failed")
        exit(ret)
    rknn.release(session_index=0)

    # 6. LoRA Initialization (必须在主 Session 循环前建立，支持从内存加载)
    use_lora_from_data = True
    if use_lora_from_data:
        with open(args.lora_path, "rb") as f:
            lora_data = f.read()
        ret = rknn.init_lora(weight_data=lora_data, weight_size=len(lora_data))
        if ret < 0:
            print("rknn.init_lora_from_data failed!")
            exit(ret)
        print("rknn3_lora_init_from_data success!") # Match cpp print
    else:
        ret = rknn.init_lora(lora_weight_path=args.lora_path)
        if ret < 0:
            print("rknn.init_lora failed!")
            exit(ret)

    lora_list, n_lora = rknn.query_lora()
    print(f"Found {n_lora} LoRA adapter(s):")
    for i in range(n_lora):
        print(f"  LoRA[{i}]: name={lora_list[i].lora_name.decode('utf-8')}, scale={lora_list[i].scale:.2f}")

    cur_lora = lora_list[0]
    ret = rknn.load_lora(cur_lora)
    if ret < 0:
        print("rknn.load_lora failed!")
        exit(ret)
    lora_loaded = True

    # 7. 循环建立核心 Session，设置 KV 缓存并将 LoRA 动态挂载到指定的 Session 1
    lora_enabled_session1 = False
    for s in range(2):
        active_cb = cb_lora if s == 1 else cb_base
        # 指定 session_index 进行初始化，无需手动重置 rknn.llm
        ret = rknn.init_llm_session(llm_args=LLM_ARGS, llm_callback=active_cb, session_index=s)
        if ret != 0:
            print(f"rknn.init_llm_session failed for session {s}")
            exit(ret)

        rknn.set_kvcache_policy(RKNN3KVCachePolicy.RKNN3_KVCACHE_POLICY_NORMAL, session_index=s)
        
        if s == 1:
            # 测试修改scale值
            cur_lora.scale = 0.0
            rknn.enable_lora(cur_lora, session_index=s)
            cur_lora.scale = 1.0
            rknn.enable_lora(cur_lora, session_index=s)
            lora_enabled_session1 = True


    if args.max_context_len != llm_config.max_ctx_len:
        print(f"max_context_len != llm_config.max_ctx_len, max_context_len {args.max_context_len}, llm_config.max_ctx_len {llm_config.max_ctx_len}")
        print(f"\nplease set <max_context_len> to {llm_config.max_ctx_len} \n")
        exit(-1)

    print("\n=============================================================")
    print(f"{'Max Context Length':<32}: {llm_config.max_ctx_len:<8}")
    print(f"{'Max Position Embeddings':<32}: {llm_config.max_position_embeddings:<8}")
    print("=============================================================\n")

    # 8. 推理环节：通过传入 session_index=s 指定运行通道
    for s in range(2):

        print(f"\n--------------------Input[0]-------------------- ")
        cur_prompt = random_prompts[s]
        print(cur_prompt)
        print("\n--------------------Output---------------------- ")

        if s == 0:
            print(f"Session[{s}]: Running base model...")
            print("expect result:\n相对论是20世纪初由爱因斯坦提出的一系列理论，包括狭义相对论和广义相对论。狭义相对论主要研究在静止或缓慢运动的参考系中，物体之间的相互作用以及光速与速度的关系；而广义相对论则将这些概念扩展到高速运动的参考系中。\n\n狭义相对论的主要内容包括：\n\n1. 光速不变原理：光的速度在不同的参考系中是恒定的，即光速c（约299794530千米/秒）是一个常数。这个速度\n\n")
        else:
            print(f"Session[{s}]: Running lora model...")
            print("expect result:\n相对论是物理学的一个重要理论，它描述了宇宙的运动和结构。相对论的基本概念包括：\n\n1. 宇宙：宇宙是一个无限大的空间，包含所有物质和能量。\n\n2. 现代化：现代化的宇宙是由一个特定的物理定律定义的，这个定律称为爱因斯坦定律。\n\n3. 没有绝对速度：在宇宙中，没有绝对速度，只有相对速度。相对速度是指物体相对于其他物体的速度。\n\n4. 宇宙的运动：宇宙的运动是通过引力和物质运动来解释的。引力是一种力，它使物体向\n\n")
        
        print("real result:\n")
        
        first_token = None
        ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = rknn.session_run(
            prompt=cur_prompt, keep_history=True, max_new_tokens=args.max_new_token, session_index=s)
        if ret != 0:
            print(f"rknn.session_run failed for session {s}")
            break
        
        # clear kvcache 指定通道清理
        rknn.clear_kvcache(RKNN3KVCacheClearPolicy.RKNN3_KVCACHE_CLEAR_ALL, session_index=s)

        if first_token is not None:
            printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time)

    # 9. Cleanup
    if lora_loaded:
        if lora_enabled_session1:
            rknn.disable_lora(cur_lora, session_index=1)
        rknn.unload_lora(cur_lora)

        
    # rknn.release() 默认不传参数会连同内置包含的所有 Session 以及 Runtime 全部释放
    rknn.release()
    print("*******************************END TEST**********************************")