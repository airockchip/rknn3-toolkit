# -*- coding: utf-8 -*-
"""
Paged KV cache test for RKNN3 Toolkit Lite.

This is a Python port of the C reference implementation in
    rknn-runtime3/examples/rknn3_model_test_demo/src/rknn3_test_utils.cpp
        - rknn3_paged_cache_page_count()
        - setup_kvcache_mems()
        - cleanup_kvcache_mems()

It exercises the paged KV cache path added to rknn3_api.h:
    - RKNN3_QUERY_KVCACHE_PAGE_INFO     (per-core page geometry)
    - RKNN3_QUERY_ATTENTION_CONFIG      (tokens_per_page etc.)
    - rknn3_set_kvcache_pages()         (bind / unbind page tables)

Flow (mirrors the C model-test ordering):
    load -> model_init -> query -> setup paged kvcache -> LLM session -> run
    ... -> unbind pages -> destroy pages -> release

Non-paged (legacy) models report n_cache_types == 0; in that case this script
falls back to the classic rknn3_set_kvcache_mem() path, exactly like the C code,
so the same script works for both model generations.

Usage:
    python rknn3_paged_kvcache_test.py --rknn_path Qwen2.5-0.5B-Instruct.rknn \
        --embed_path Qwen2.5-0.5B-Instruct.embed.bin \
        --tokenizer_path ./Qwen2.5-0.5B-Instruct
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

import ctypes
import time
from ctypes import POINTER

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
    RKNN3MemAllocFlags,
    RKNN3MemSyncMode,
    RKNN3PageCacheType,
    RKNN3AttentionType,
    RKNN3KvCachePageBinding,
    RKNN3TensorMemory,
    RKNN3_MAX_CACHE_PAGE_TYPES,
    dump_tensor_attr,
)

# ============================================= Default Config =============================================

RKNN_MODEL     = 'Qwen2.5-0.5B-Instruct.rknn'
EMBED_PATH     = 'Qwen2.5-0.5B-Instruct.embed.bin'
TOKENIZER_PATH = './Qwen2.5-0.5B-Instruct'

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

    if state == 5:  # RKLLM_RUN_ERROR
        print("\n\nError occurred during inference")
        return 0

    if state in (2, 3, 4):  # FINISH / STOP / MAX_TOKEN
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

    if state == 1:  # RKLLM_RUN_WAITING
        print("\n\nWaiting for UTF-8 encoded character")
        return 0

    if state == 0:  # RKLLM_RUN_NORMAL
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

# ============================================= Paged KV cache helpers =============================================

def dump_kvcache_page_info(info):
    """Port of dump_kvcache_page_info() in rknn3_test_utils.cpp."""
    print("KV cache page info: core_id=%d, max_request_num=%u, n_cache_types=%u" % (
        info.core_id, info.max_request_num, info.n_cache_types))
    for i in range(min(info.n_cache_types, RKNN3_MAX_CACHE_PAGE_TYPES)):
        cache = info.caches[i]
        print("  cache[%u]: type=%d, page_size=%d, max_page_num=%u" % (
            i, cache.cache_type, cache.page_size, cache.max_page_num))


def paged_cache_page_count(attr, attention_config, tokens):
    """Port of rknn3_paged_cache_page_count() in rknn3_test_utils.cpp.

    Returns the number of pages required for one (core, cache_type) pair, or
    None when the C helper would have returned RKNN3_ERR_ARGUMENT_INVALID.
    """
    if attr is None or attr.page_size == 0 or attr.max_page_num == 0:
        return None
    if attr.cache_type != RKNN3PageCacheType.RKNN3_PAGE_CACHE_FULL_ATTENTION:
        return attr.max_page_num
    if (attention_config is None
            or attention_config.attention_type != RKNN3AttentionType.RKNN3_ATTENTION_TYPE_FULL_ATTENTION
            or tokens == 0
            or attention_config.tokens_per_page == 0):
        return None
    count = tokens // attention_config.tokens_per_page
    if tokens % attention_config.tokens_per_page:
        count += 1
    if count > attr.max_page_num:
        return None
    return count


def destroy_kvcache_mems(rknn, mems):
    """Port of destroy_kvcache_mems()."""
    for mem in mems:
        if mem:
            rknn.destroy_mem(mem)
    mems.clear()


def cleanup_kvcache_mems(rknn, mems, paged):
    """Port of cleanup_kvcache_mems(): unbind every key, then free the pages."""
    if not mems:
        return

    if not paged:
        # Legacy path: memory is owned as a whole per core, just free it.
        destroy_kvcache_mems(rknn, mems)
        return

    cores = []
    unbinds = []
    for mem in mems:
        if not mem:
            continue
        core_id = mem.contents.core_id
        if core_id in cores:
            continue

        info = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_KVCACHE_PAGE_INFO, index=core_id)
        if info is None:
            print("RKNN3_QUERY_KVCACHE_PAGE_INFO during cleanup failed; pages remain allocated")
            return
        cores.append(core_id)
        for request in range(info.max_request_num):
            for t in range(info.n_cache_types):
                binding = RKNN3KvCachePageBinding()
                binding.core_id = info.core_id
                binding.cache_type = info.caches[t].cache_type
                binding.request_index = request
                binding.n_pages = 0  # 0 unbinds this entry
                binding.pages = None
                unbinds.append(binding)

    if unbinds:
        ret = rknn.set_kvcache_pages(unbinds)
        if ret != 0:
            print("rknn3_set_kvcache_pages unbind failed, ret=%d; pages remain allocated" % ret)
            return

    destroy_kvcache_mems(rknn, mems)


def setup_kvcache_mems(rknn, kvcache_mems):
    """Port of setup_kvcache_mems() in rknn3_test_utils.cpp.

    Returns (ret, paged). On success `kvcache_mems` holds every allocated
    rknn3_tensor_mem* that must stay alive until cleanup_kvcache_mems().
    """
    core_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_CORE_NUMBER)
    if core_num is None:
        print("RKNN3_QUERY_CORE_NUMBER failed")
        return -1, False
    if core_num == 0:
        return 0, False

    alloc_infos = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_ALLOCATION_INFO)
    if alloc_infos is None:
        print("RKNN3_QUERY_ALLOCATION_INFO failed")
        return -1, False

    # ---- Probe the first core to decide paged vs legacy ----
    first_page_info = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_KVCACHE_PAGE_INFO,
                                       index=alloc_infos[0].core_id)
    if first_page_info is None:
        print("RKNN3_QUERY_KVCACHE_PAGE_INFO failed")
        return -1, False
    dump_kvcache_page_info(first_page_info)

    # =====================================================================
    # Paged path (n_cache_types != 0)
    # =====================================================================
    if first_page_info.n_cache_types != 0:
        llm_config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
        if llm_config is None:
            return -1, True

        attention_configs = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_ATTENTION_CONFIG)
        if not attention_configs:
            print("Invalid model: paged KV cache is enabled, but no attention config was found")
            return -1, True

        page_config = None
        for config in attention_configs:
            if config.attention_type == RKNN3AttentionType.RKNN3_ATTENTION_TYPE_FULL_ATTENTION:
                page_config = config
                break
        if page_config is None:
            print("Invalid model: paged KV cache requires a FULL_ATTENTION config; found attention types:",
                  [int(c.attention_type) for c in attention_configs])
            return -1, True

        print("KV cache page config: tokens_per_page=%u, max_ctx_len=%u, residual_depth=%d, rollback_tokens_align=%u" % (
            page_config.tokens_per_page, llm_config.max_ctx_len,
            page_config.kvcache_residual_depth, page_config.rollback_tokens_align))

        bindings = []
        # Keep every page-pointer array alive until set_kvcache_pages() returns;
        # the C struct stores a raw pointer to the first element.
        page_arrays_keepalive = []

        for allocation in alloc_infos:
            info = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_KVCACHE_PAGE_INFO,
                                    index=allocation.core_id)
            if info is None or info.n_cache_types == 0:
                print("Invalid model: inconsistent paged cache info across cores")
                destroy_kvcache_mems(rknn, kvcache_mems)
                return -1, True
            dump_kvcache_page_info(info)

            for request in range(info.max_request_num):
                for t in range(info.n_cache_types):
                    cache = info.caches[t]
                    page_count = paged_cache_page_count(cache, page_config, llm_config.max_ctx_len)
                    if page_count is None:
                        print("rknn3_paged_cache_page_count failed")
                        destroy_kvcache_mems(rknn, kvcache_mems)
                        return -1, True

                    page_mems = []
                    for _ in range(page_count):
                        mem = rknn.create_mem(
                            size=cache.page_size, core_id=info.core_id,
                            flags=RKNN3MemAllocFlags.RKNN3_FLAG_MEMORY_CACHEABLE)
                        if not mem or not mem.contents.virt_addr:
                            if mem:
                                rknn.destroy_mem(mem)
                            destroy_kvcache_mems(rknn, kvcache_mems)
                            return -1, True
                        kvcache_mems.append(mem)
                        ctypes.memset(mem.contents.virt_addr, 0, mem.contents.size)
                        ret = rknn.mem_sync(mem, RKNN3MemSyncMode.RKNN3_MEMORY_SYNC_TO_DEVICE)
                        if ret != 0:
                            destroy_kvcache_mems(rknn, kvcache_mems)
                            return ret, True
                        page_mems.append(mem)

                    page_array = (POINTER(RKNN3TensorMemory) * page_count)(*page_mems)
                    page_arrays_keepalive.append(page_array)

                    binding = RKNN3KvCachePageBinding()
                    binding.core_id = info.core_id
                    binding.cache_type = cache.cache_type
                    binding.request_index = request
                    binding.n_pages = page_count
                    binding.pages = page_array
                    bindings.append(binding)

                    print("KV cache binding: core_id=%d, request=%u, cache_type=%d, n_pages=%u, page_size=%d" % (
                        info.core_id, request, cache.cache_type, page_count, cache.page_size))

        ret = rknn.set_kvcache_pages(bindings)
        if ret != 0:
            print("rknn3_set_kvcache_pages failed, ret=%d" % ret)
            destroy_kvcache_mems(rknn, kvcache_mems)
            return ret, True
        print("rknn3_set_kvcache_pages success, set %d page memories" % len(kvcache_mems))
        return 0, True

    # =====================================================================
    # Legacy path (non-paged models)
    # =====================================================================
    core_indices = []
    for allocation in alloc_infos:
        kvcache_size = allocation.kvcache_mem.size
        core_id = allocation.core_id
        print("Core %d allocation info: kvcache_size=%d" % (core_id, kvcache_size))
        if kvcache_size == 0:
            continue

        mem = rknn.create_mem(size=kvcache_size, core_id=core_id,
                              flags=RKNN3MemAllocFlags.RKNN3_FLAG_MEMORY_CACHEABLE)
        if not mem:
            print("rknn3_create_mem for kvcache failed, core=%d, size=%d" % (core_id, kvcache_size))
            cleanup_kvcache_mems(rknn, kvcache_mems, paged=False)
            return -1, False

        kvcache_mems.append(mem)
        core_indices.append(core_id)
        print("Created KV cache memory: core_id=%d, size=%d, virt=0x%x" % (
            core_id, mem.contents.size, mem.contents.virt_addr or 0))

    if not kvcache_mems:
        print("No KV cache memory is required by this model")
        return 0, False

    ret = rknn.set_kvcache_mem(kvcache_mems, core_indices)
    if ret != 0:
        print("rknn3_set_kvcache_mem failed, ret=%d" % ret)
        cleanup_kvcache_mems(rknn, kvcache_mems, paged=False)
        return ret, False

    print("rknn3_set_kvcache_mem success, set %d KV cache memories" % len(kvcache_mems))
    return 0, False

# ============================================= Main =============================================

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="RKNN3 paged KV cache test (port of the C "
                            "setup_kvcache_mems / cleanup_kvcache_mems reference)")
    parser.add_argument("--rknn_path",      type=str, default=RKNN_MODEL,     help="rknn model path")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH, help="huggingface tokenizer path")
    parser.add_argument("--embed_path",     type=str, default=EMBED_PATH,     help="embedding bin path")
    parser.add_argument("--max_new_token",  type=int, default=256,            help="max new tokens")
    parser.add_argument("--core_mask",      type=str, default="0xff",         help="NPU core mask in hex")
    parser.add_argument("--decrypt_key_path", type=str, default=None,         help="decrypt key/license file path for encrypted RKNN model")
    parser.add_argument("--skip_infer",     action="store_true",              help="only exercise page setup/cleanup, skip LLM inference")
    parser.add_argument("--loop",           type=int, default=1,              help="repeat the prompt list this many times (use >1 to exercise the clear-kvcache guard)")
    args = parser.parse_args()

    core_mask = int(args.core_mask, 16)
    kvcache_mems = []

    # Load tokenizer & embedding
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    embeds_data = np.fromfile(args.embed_path, dtype=np.float16)

    rknn = RKNN3Lite(llm_mode=True, verbose=True)

    print('--> Loading model')
    ret = rknn.load_rknn(args.rknn_path, args.rknn_path.replace('.rknn', '.weight'))
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk1820', core_mask=core_mask,
                            decrypt_key_path=args.decrypt_key_path)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # ---------------------------------------------------------------
    # Step: set up paged KV cache (the actual feature under test)
    # ---------------------------------------------------------------
    print('--> Setup paged KV cache')
    ret, paged = setup_kvcache_mems(rknn, kvcache_mems)
    if ret != 0:
        print('Setup paged KV cache failed!')
        rknn.release()
        exit(ret)
    print('paged KV cache: %s' % ('ENABLED' if paged else 'not used by this model (legacy path)'))
    print('done')

    # ---------------------------------------------------------------
    # Query model info
    # ---------------------------------------------------------------
    print('--> Query model info')
    io_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
    print("model input num: %d, output num: %d" % (io_num.n_input, io_num.n_output))

    for i in range(io_num.n_output):
        output_attr = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=i)
        dump_tensor_attr(output_attr, prefix="output")

    llm_config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
    vocab_size    = llm_config.vocab_size
    embedding_dim = embeds_data.size // vocab_size
    embeds_data   = embeds_data.reshape(vocab_size, embedding_dim)
    print("vocab_size=%d, embedding_dim=%d" % (vocab_size, embedding_dim))
    print("max_ctx_len=%d, max_position_embeddings=%d" % (
        llm_config.max_ctx_len, llm_config.max_position_embeddings))
    print("n_attention_configs=%d" % llm_config.n_attention_configs)

    if args.skip_infer:
        print('--> skip_infer set, tearing down paged KV cache')
        cleanup_kvcache_mems(rknn, kvcache_mems, paged)
        rknn.release()
        print("*******************************END TEST**********************************")
        exit(0)

    # ---------------------------------------------------------------
    # Build LLM args + callback, then init the session
    # ---------------------------------------------------------------
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

    callback = RKLLMCallback()
    callback.result_callback = LLMResultCallback(result_callback)
    callback.result_userdata = None

    callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    _tok_ud = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(_tok_ud), ctypes.c_void_p)

    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    _emb_ud = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(_emb_ud), ctypes.c_void_p)

    print('--> Init LLM session')
    ret = rknn.init_llm_session(llm_args=LLM_ARGS, llm_callback=callback)
    if ret != 0:
        print('Init LLM session failed!')
        cleanup_kvcache_mems(rknn, kvcache_mems, paged)
        rknn.release()
        exit(ret)
    print('done')

    ret = rknn.set_chat_template(system_prompt, prompt_prefix, prompt_postfix)
    if ret != 0:
        print('Set chat template failed!')
        cleanup_kvcache_mems(rknn, kvcache_mems, paged)
        rknn.release()
        exit(ret)

    # Cache length groups belong to the legacy (non-paged) cache layout.
    if not paged:
        ret = rknn.set_kvcache_policy(RKNN3KVCachePolicy.RKNN3_KVCACHE_POLICY_NORMAL)
        if ret != 0:
            print('Set kvcache policy failed!')
            cleanup_kvcache_mems(rknn, kvcache_mems, paged)
            rknn.release()
            exit(ret)
    else:
        print('paged model: skipping set_kvcache_policy (page table is bound instead)')

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------
    prompts = [
        "请解释一下相对论的基本概念。",
        "归纳上面内容",
        "Please explain the basic concept of relativity",
    ]

    for i in range(len(prompts) * args.loop):
        prompt = prompts[i % len(prompts)]
        print(f"\n--------------------Input[{i}]--------------------")
        print(prompt)
        print("\n--------------------Output----------------------")

        first_token = None
        ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = \
            rknn.session_run(prompt=prompt, keep_history=True)
        if ret != 0:
            print('RKNN llm inference failed!')
            cleanup_kvcache_mems(rknn, kvcache_mems, paged)
            rknn.release()
            exit(ret)

        if first_token is not None:
            printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time)
        first_token = None

        # Same guard as the C session test: drop the cache once it is nearly
        # full so the next round starts from a clean context.
        state = rknn.session_query_state()
        if state is not None:
            print("run state: n_total_tokens=%d, n_max_tokens=%d" % (
                state.n_total_tokens, state.n_max_tokens))
            if state.n_total_tokens >= (state.n_max_tokens - args.max_new_token):
                print("kvcache nearly full, clearing (RKNN3_KVCACHE_CLEAR_ALL)")
                ret = rknn.clear_kvcache(RKNN3KVCacheClearPolicy.RKNN3_KVCACHE_CLEAR_ALL)
                if ret != 0:
                    print('Clear kvcache failed!')
                    cleanup_kvcache_mems(rknn, kvcache_mems, paged)
                    rknn.release()
                    exit(ret)

    # ---------------------------------------------------------------
    # Teardown: unbind pages, then free them (mirrors C cleanup order)
    # ---------------------------------------------------------------
    print('\n--> Cleanup paged KV cache')
    cleanup_kvcache_mems(rknn, kvcache_mems, paged)
    print('done')

    rknn.release()
    print("*******************************END TEST**********************************")
