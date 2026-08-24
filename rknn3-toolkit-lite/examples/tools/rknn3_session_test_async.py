#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNN3 LLM async session test, Python version of rknn3_session_test_async.cpp.

This script follows the same high-level flow as the C++ demo:
  1. load RKNN model + weight
  2. init RKNN3 runtime / model
  3. query LLM config
  4. build callbacks: result / tokenizer / embedding
  5. init LLM session
  6. set chat template
  7. set recurrent KV-cache policy
  8. submit prompts with session_run_async
  9. keep process alive until terminal callbacks are received

Notes:
  - The Python tokenizer path follows rknn_session_test.py and uses transformers.AutoTokenizer.
    Pass a local HuggingFace tokenizer directory or model id, not a C++ llama tokenizer.gguf file.
  - One RKNN3 session is used. This mirrors the C++ sample, but it should be treated as async
    submission / queueing, not true safe multi-request concurrency. For real parallelism, use
    multiple sessions if your SDK/runtime supports it.
"""

import ctypes
import os
import sys
import time
import threading
from argparse import ArgumentParser
from typing import Iterable, List, Optional, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")

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
    LLMOutputCallback,
    RKNN3QueryCmd,
    RKNN3KVCachePolicy,
    RKNN3Tensor,
    dump_tensor_attr,
)


# ============================================= Defaults =============================================

RKNN_MODEL = "Qwen2.5-0.5B-Instruct.rknn"
WEIGHT_MODEL = "Qwen2.5-0.5B-Instruct.weight"
EMBED_PATH = "Qwen2.5-0.5B-Instruct.embed.bin"
TOKENIZER_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
PROMPT_PREFIX = "<|im_start|>user\n"
PROMPT_POSTFIX = "<|im_end|>\n<|im_start|>assistant\n"

# Global callback state. Native callbacks can be invoked from SDK threads, so protect shared counters.
tokenizer = None
embeds_data: Optional[np.ndarray] = None
first_token_time: Optional[float] = None
_callback_keepalive = []


class AsyncRunTracker:
    def __init__(self) -> None:
        self.expected = 0
        self.finished = 0
        self.lock = threading.Lock()
        self.done_event = threading.Event()

    def reset(self, expected: int) -> None:
        with self.lock:
            self.expected = expected
            self.finished = 0
            self.done_event.clear()

    def mark_terminal(self) -> None:
        with self.lock:
            self.finished += 1
            if self.finished >= self.expected:
                self.done_event.set()

    def snapshot(self) -> Tuple[int, int]:
        with self.lock:
            return self.finished, self.expected

    def wait(self, poll_interval: float = 0.5, timeout: Optional[float] = None, poll_state_fn=None) -> bool:
        start = time.perf_counter()
        last_poll_print = 0.0
        while True:
            finished, expected = self.snapshot()
            if finished >= expected:
                return True

            now = time.perf_counter()
            if poll_state_fn is not None and now - last_poll_print >= 1.0:
                last_poll_print = now
                try:
                    state = poll_state_fn()
                    if state is not None:
                        print(
                            f"\n[query_state] total={state.n_total_tokens}, "
                            f"max={state.n_max_tokens}, decode={state.n_decode_tokens}, "
                            f"prefill={state.n_prefill_tokens}, kvcache_policy={state.kvcache_policy}",
                            flush=True,
                        )
                except Exception as exc:
                    print(f"\n[query_state failed] {exc}", flush=True)

            if timeout is not None and now - start > timeout:
                return False

            self.done_event.wait(poll_interval)


run_tracker = AsyncRunTracker()


# ============================================= Helpers =============================================


def _decode_c_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    try:
        return ctypes.string_at(value).decode("utf-8", errors="ignore")
    except Exception:
        return str(value)


def _read_callback_text(text_ptr, text_len: int) -> str:
    if text_ptr is None:
        return ""
    if isinstance(text_ptr, (bytes, bytearray)):
        raw = bytes(text_ptr[:text_len]) if text_len and text_len > 0 else bytes(text_ptr)
    else:
        raw = ctypes.string_at(text_ptr, text_len)
    return raw.decode("utf-8", errors="ignore")


def _check_file(path: str, desc: str) -> None:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{desc} not found: {path}")


def _shape_embedding(embed_path: str, vocab_size: int, llm_embedding_dim: int = 0) -> np.ndarray:
    _check_file(embed_path, "embedding file")
    raw = np.memmap(embed_path, dtype=np.float16, mode="r")

    if vocab_size <= 0:
        raise ValueError(f"invalid vocab_size from LLM config: {vocab_size}")

    if raw.size % vocab_size != 0:
        raise ValueError(
            f"embedding size mismatch: fp16_count={raw.size}, vocab_size={vocab_size}, "
            f"remainder={raw.size % vocab_size}"
        )

    embedding_dim = raw.size // vocab_size
    if llm_embedding_dim and llm_embedding_dim != embedding_dim:
        print(
            f"Warning: embedding_dim from file ({embedding_dim}) != llm_config.embedding_dim ({llm_embedding_dim}). "
            f"The file-derived value will be used.",
            flush=True,
        )

    return raw.reshape(vocab_size, embedding_dim)


def _get_tokenizer_eos_id(tok) -> int:
    eos_id = getattr(tok, "eos_token_id", None)
    if eos_id is None:
        return -1
    if isinstance(eos_id, list):
        return int(eos_id[0]) if eos_id else -1
    return int(eos_id)


def print_llm_config(llm_config) -> None:
    print(f"llm_config.chat_template: {_decode_c_string(llm_config.chat_template)}")
    print(f"llm_config.vocab_size: {llm_config.vocab_size}")
    print(f"llm_config.embedding_dim: {llm_config.embedding_dim}")
    print(f"llm_config.max_ctx_len: {llm_config.max_ctx_len}")
    print(f"llm_config.max_position_embeddings: {llm_config.max_position_embeddings}")
    print(f"llm_config.kvcache_store_method: {llm_config.kvcache_store_method}")
    print(f"llm_config.kvcache_dtype: {llm_config.kvcache_dtype}")
    print(f"llm_config.kvcache_group_size: {llm_config.kvcache_group_size}")
    print(f"llm_config.kvcache_residual_depth: {llm_config.kvcache_residual_depth}")
    print(f"llm_config.model_type: {_decode_c_string(llm_config.model_type)}")
    print(f"llm_config.task_type: {llm_config.task_type}")

    task_name = "RKNN3_LLM_TASK_GENERATE" if int(llm_config.task_type) == 0 else "RKNN3_LLM_TASK_EMBEDDING"
    print("\n=============================================================")
    print(f"{'Model Config':>38}")
    print("=============================================================")
    print(f"{'Max Context Length':<32}: {llm_config.max_ctx_len:<8}")
    print(f"{'Max Position Embeddings':<32}: {llm_config.max_position_embeddings:<8}")
    print(f"{'Model Type':<32}: {_decode_c_string(llm_config.model_type)}")
    print(f"{'Task Type':<32}: {task_name}")
    print("=============================================================\n")


# ============================================= Callbacks =============================================


def result_callback(userdata, result_ptr, state):
    """Equivalent to C++ result_callback, with safer UTF-8 streaming decode."""
    global tokenizer, first_token_time

    if not hasattr(result_callback, "accumulated_tokens"):
        result_callback.accumulated_tokens = []
        result_callback.last_output_text = ""

    def decode_safe(tokens: List[int]) -> str:
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # Avoid printing half of a multi-byte UTF-8 replacement chunk.
        return text.split("\ufffd", 1)[0] if "\ufffd" in text else text

    try:
        state = int(state)

        # RKLLM_RUN_ERROR
        if state == 5:
            run_tracker.mark_terminal()
            print("\n\nError occurred during inference", flush=True)
            return 0

        # RKLLM_RUN_FINISH / RKLLM_RUN_STOP / RKLLM_RUN_MAX_NEW_TOKEN_REACHED
        if state in (2, 3, 4):
            if result_callback.accumulated_tokens:
                try:
                    safe_text = decode_safe(result_callback.accumulated_tokens)
                    new_part = safe_text[len(result_callback.last_output_text):]
                    if new_part:
                        print(new_part, end="", flush=True)
                except Exception as exc:
                    print(f"\n[Decode error: {exc}]", flush=True)

            result_callback.accumulated_tokens.clear()
            result_callback.last_output_text = ""
            first_token_time = None
            msg = {2: "Finished", 3: "Stop", 4: "Max new token reached"}.get(state, "Unknown")
            print(f"\n\n--------------------{msg}--------------------", flush=True)
            run_tracker.mark_terminal()
            return 0

        # RKLLM_RUN_WAITING
        if state == 1:
            print("\n\nWaiting for UTF-8 encoded character", flush=True)
            return 0

        # RKLLM_RUN_NORMAL
        if state == 0:
            n = int(result_ptr.contents.num_tokens)
            new_tokens = [int(result_ptr.contents.token_ids[i]) for i in range(n)]
            result_callback.accumulated_tokens.extend(new_tokens)

            if first_token_time is None:
                first_token_time = time.perf_counter()

            try:
                safe_text = decode_safe(result_callback.accumulated_tokens)
                new_part = safe_text[len(result_callback.last_output_text):]
                if new_part:
                    print(new_part, end="", flush=True)
                    result_callback.last_output_text += new_part
            except Exception as exc:
                print(f"\n[Temp decode error: {exc}], waiting for more tokens", flush=True)
                return 0

    except Exception as exc:
        # Do not throw across the ctypes callback boundary.
        print(f"\n[result_callback exception] {exc}", flush=True)

    return 0


def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    global tokenizer
    try:
        text = _read_callback_text(text_ptr, int(text_len))
        inputs = tokenizer(text, return_tensors="np", truncation=True)
        tokens = inputs["input_ids"][0][: int(n_tokens_max)]
        n_tokens = int(len(tokens))
        if n_tokens <= 0:
            print(f"Tokenizer failed for {text}", flush=True)
            return n_tokens

        for i, token in enumerate(tokens):
            tokens_ptr[i] = int(token)
        return n_tokens
    except Exception as exc:
        print(f"tokenizer_callback failed: {exc}", flush=True)
        return -1


def embed_callback(userdata, tokens_ptr, num_tokens, embed, length):
    global embeds_data
    try:
        if embeds_data is None:
            print("embedding data is not loaded", flush=True)
            return -1

        num_tokens = int(num_tokens)
        embedding_dim = int(embeds_data.shape[1])
        expected_len = num_tokens * embedding_dim * np.dtype(np.float16).itemsize
        if int(length) != expected_len:
            print(f"invalid embed buffer: len={length}, expected={expected_len}", flush=True)
            return -1

        dst = np.ctypeslib.as_array(
            ctypes.cast(embed, ctypes.POINTER(ctypes.c_uint16)),
            shape=(num_tokens * embedding_dim,),
        ).view(np.float16)

        tokens = np.fromiter((int(tokens_ptr[i]) for i in range(num_tokens)), dtype=np.int64, count=num_tokens)
        if np.any(tokens < 0) or np.any(tokens >= embeds_data.shape[0]):
            print(f"token id out of embedding range: min={tokens.min()}, max={tokens.max()}", flush=True)
            return -1

        dst[:] = embeds_data[tokens].reshape(-1)
        return 0
    except Exception as exc:
        print(f"embed_callback failed: {exc}", flush=True)
        return -1


def output_callback(userdata, output_tensors, n_output_tensors, state):
    """Optional equivalent of the disabled C++ output_callback block."""
    try:
        print(f"output_callback: state = {int(state)}", flush=True)
        n_output_tensors = int(n_output_tensors)
        for i in range(n_output_tensors):
            tensor: RKNN3Tensor = output_tensors[i]
            attr = tensor.attr.contents
            mem = tensor.mem.contents
            name = attr.name.decode("utf-8", errors="ignore")
            print(f"\noutput_callback: output[{i}]->attr->index = {attr.index}")
            print(f"output_callback: output[{i}]->attr->name = {name}")
            print(f"output_callback: output[{i}]->mem->size = {mem.size}")
            if mem.virt_addr:
                count = min(10, int(mem.size) // np.dtype(np.float16).itemsize)
                vals = np.ctypeslib.as_array(
                    ctypes.cast(mem.virt_addr, ctypes.POINTER(ctypes.c_uint16)),
                    shape=(count,),
                ).view(np.float16)
                for j, value in enumerate(vals):
                    print(f"output_callback: output[{i}][{j}] = {float(value):.6f}")
    except Exception as exc:
        print(f"output_callback failed: {exc}", flush=True)
    return 0


# ============================================= Main flow =============================================


def build_callback(enable_output_callback: bool, rknn: RKNN3Lite):
    callback = RKLLMCallback()

    cb_result = LLMResultCallback(result_callback)
    cb_tokenizer = LLMTokenizerCallback(tokenizer_callback)
    cb_embed = LLMGetEmbedCallback(embed_callback)

    callback.result_callback = cb_result
    callback.result_userdata = None
    callback.tokenizer_callback = cb_tokenizer
    callback.embed_callback = cb_embed

    # Keep Python objects and CFUNCTYPE wrappers alive for the full native session lifetime.
    _callback_keepalive.extend([cb_result, cb_tokenizer, cb_embed])

    tok_ud = ctypes.py_object(tokenizer)
    emb_ud = ctypes.py_object(embeds_data)
    _callback_keepalive.extend([tok_ud, emb_ud])
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(tok_ud), ctypes.c_void_p)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(emb_ud), ctypes.c_void_p)

    output_tensors = None
    n_output_tensors = 0
    if enable_output_callback:
        output_tensors, n_output_tensors = rknn.create_output_tensors(output_indices=[0])
        if output_tensors is None or n_output_tensors <= 0:
            raise RuntimeError("create_output_tensors failed")
        cb_output = LLMOutputCallback(output_callback)
        callback.output_callback = cb_output
        callback.output_tensors = output_tensors
        callback.n_output_tensors = n_output_tensors
        _callback_keepalive.extend([cb_output, output_tensors])

    return callback, output_tensors, n_output_tensors


def destroy_output_tensors(rknn: RKNN3Lite, output_tensors, n_output_tensors: int) -> None:
    if output_tensors is None:
        return
    for i in range(int(n_output_tensors)):
        try:
            if output_tensors[i].mem:
                rknn.destroy_mem(output_tensors[i].mem)
        except Exception as exc:
            print(f"destroy output tensor[{i}] mem failed: {exc}", flush=True)


def load_prompts(prompt_args: Optional[List[str]], prompt_file: Optional[str]) -> List[str]:
    if prompt_args:
        return prompt_args
    if prompt_file:
        _check_file(prompt_file, "prompt file")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
        if not prompts:
            raise ValueError(f"prompt file is empty: {prompt_file}")
        return prompts
    return [
        "请解释一下相对论的基本概念。",
        "Please explain the basic concept of relativity",
    ]


def main(argv: Optional[Iterable[str]] = None) -> int:
    global tokenizer, embeds_data, first_token_time

    parser = ArgumentParser(
        description="RKNN3 Session Test Async - Python version of rknn3_session_test_async.cpp"
    )
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL, help="RKNN model path")
    parser.add_argument(
        "--weight_path",
        type=str,
        default=None,
        help="RKNN weight path. Default: replace .rknn with .weight",
    )
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH, help="HF tokenizer path/model id")
    parser.add_argument("--embed_path", type=str, default=EMBED_PATH, help="embedding .bin path")
    parser.add_argument("--max_context_len", type=int, default=1024, help="expected max context length")
    parser.add_argument("--max_new_tokens", "--max_new_token", dest="max_new_tokens", type=int, default=256)
    parser.add_argument("--core_mask", type=str, default="0xff", help="NPU core mask in hex, e.g. 0xff")
    parser.add_argument("--target", type=str, default="rk1820", help="target device, e.g. rk1820/rk1828")
    parser.add_argument("--decrypt_key_path", type=str, default=None, help="decrypt key/license file path")
    parser.add_argument("--prompt", action="append", help="prompt text. Can be specified multiple times")
    parser.add_argument("--prompt_file", type=str, default=None, help="one prompt per line")
    parser.add_argument("--wait_timeout", type=float, default=None, help="seconds to wait for async completion")
    parser.add_argument("--poll_state", action="store_true", help="poll and print session state while waiting")
    parser.add_argument("--enable_output_callback", action="store_true", help="enable output tensor callback dump")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=True, help="enable RKNN verbose log")
    parser.add_argument("--quiet", dest="verbose", action="store_false", help="disable RKNN verbose log")
    args = parser.parse_args(list(argv) if argv is not None else None)

    weight_path = args.weight_path or args.rknn_path.replace(".rknn", ".weight")
    core_mask = int(args.core_mask, 16)
    prompts = load_prompts(args.prompt, args.prompt_file)

    rknn = None
    output_tensors = None
    n_output_tensors = 0

    try:
        print("*******************************NEW TEST**********************************")

        _check_file(args.rknn_path, "RKNN model")
        _check_file(weight_path, "RKNN weight")

        print("--> Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        print("done")

        print("--> Loading model")
        rknn = RKNN3Lite(llm_mode=True, verbose=args.verbose)
        ret = rknn.load_rknn(args.rknn_path, weight_path)
        if ret != 0:
            print("Load model failed!")
            return int(ret)
        print("done")

        print("--> Init runtime environment")
        ret = rknn.init_runtime(
            target=args.target,
            core_mask=core_mask,
            decrypt_key_path=args.decrypt_key_path,
        )
        if ret != 0:
            print("Init runtime environment failed!")
            return int(ret)
        print("done")

        print("--> Query model info")
        io_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
        if io_num is None:
            print("Query IO number failed!")
            return -1
        print(f"model input num: {io_num.n_input}, output num: {io_num.n_output}")
        for i in range(io_num.n_output):
            output_attr = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=i)
            if output_attr is not None:
                dump_tensor_attr(output_attr, prefix="output")

        llm_config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
        if llm_config is None:
            print("rknn3_query LLM config failed!")
            return -1
        print_llm_config(llm_config)

        embeds_data = _shape_embedding(args.embed_path, int(llm_config.vocab_size), int(llm_config.embedding_dim))
        embedding_dim = int(embeds_data.shape[1])
        print(f"vocab_size={llm_config.vocab_size}, embedding_dim={embedding_dim}")

        if args.max_context_len != int(llm_config.max_ctx_len):
            if args.max_context_len < int(llm_config.max_ctx_len):
                print(
                    f"Warning: max_context_len ({args.max_context_len}) is less than "
                    f"llm_config.max_ctx_len ({llm_config.max_ctx_len}).",
                    flush=True,
                )
                print(f"It's recommended to set <max_context_len> to {llm_config.max_ctx_len}.", flush=True)
            else:
                print(
                    f"Error: max_context_len ({args.max_context_len}) is greater than "
                    f"llm_config.max_ctx_len ({llm_config.max_ctx_len}).",
                    flush=True,
                )
                print(f"Please set <max_context_len> to {llm_config.max_ctx_len}.", flush=True)
                return -1

        llm_args = [{
            "max_new_tokens": args.max_new_tokens,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 1.0,
            "repeat_penalty": 1.1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "vocab_size": int(llm_config.vocab_size),
            "special_eos_id": _get_tokenizer_eos_id(tokenizer),
            "max_context_len": int(llm_config.max_ctx_len),
            "keep_history": 1,
            "logits_name": b"output",
        }]

        callback, output_tensors, n_output_tensors = build_callback(args.enable_output_callback, rknn)

        print("--> Init LLM session")
        ret = rknn.init_llm_session(llm_args=llm_args, llm_callback=callback)
        if ret != 0:
            print("Init LLM session failed!")
            return int(ret)
        print("done")

        print("--> Set chat template")
        ret = rknn.set_chat_template(SYSTEM_PROMPT, PROMPT_PREFIX, PROMPT_POSTFIX)
        if ret != 0:
            print("Set chat template failed!")
            return int(ret)
        print("done")

        print("--> Set KV cache policy: RKNN3_KVCACHE_POLICY_RECURRENT")
        ret = rknn.set_kvcache_policy(RKNN3KVCachePolicy.RKNN3_KVCACHE_POLICY_RECURRENT)
        if ret != 0:
            print(f"Set kvcache policy failed! ret={ret}")
            return int(ret)
        print("done")

        run_tracker.reset(len(prompts))
        for i, prompt in enumerate(prompts):
            print(f"\n--------------------Input[{i}]--------------------")
            print(prompt)
            print("\n--------------------Output----------------------")

            first_token_time = None
            ret, perf = rknn.session_run_async(
                prompt=prompt,
                keep_history=True,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=False,
                session_index=0,
            )
            if ret != 0:
                print(f"rknn3_session_run_async failed, ret = {ret}")
                return int(ret)
            # perf is returned by the wrapper, but async end time is usually not meaningful until callback finishes.

        ok = run_tracker.wait(
            poll_interval=0.5,
            timeout=args.wait_timeout,
            poll_state_fn=(lambda: rknn.session_query_state()) if args.poll_state else None,
        )
        if not ok:
            finished, expected = run_tracker.snapshot()
            print(f"\nTimeout waiting async runs: finished={finished}, expected={expected}")
            return -1

        print("done")
        return 0

    finally:
        if rknn is not None:
            destroy_output_tensors(rknn, output_tensors, n_output_tensors)
            rknn.release()
        print("*******************************END TEST**********************************")


if __name__ == "__main__":
    sys.exit(main())
