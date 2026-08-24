#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 Embedding RKNN3 Python demo.

This script is a Python version of examples/Qwen3_Embedding/cpp/main.cc.
It uses rknn3lite's separated LLM flow:
    load_rknn -> init_runtime -> query IO/config -> init_llm_session -> session_run

It captures the embedding model output from output_callback when
RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED is reached, converts fp16 output memory
to float32, and saves it to .npy. By default it keeps the raw C++ behavior;
"""

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Optional: use hf-mirror by default, consistent with the reference Python demos.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")

import numpy as np
from transformers import AutoTokenizer

try:
    from rknn3lite.api import (
        RKNN3Lite,
        RKLLMCallback,
        LLMResultCallback,
        LLMTokenizerCallback,
        LLMGetEmbedCallback,
        LLMOutputCallback,
        RKNN3Tensor,
        RKNN3QueryCmd,
        LLMOutputCallbackState,
        dump_tensor_attr,
    )
except Exception as exc:  # pragma: no cover - only happens off target board / env
    raise RuntimeError(
        "import rknn3lite failed. Please run this script in the RKNN3Lite environment."
    ) from exc


# ============================= LLM state constants =============================
# Mirror the SDK's LLMCallState enum values for readability.
RKLLM_STATE_NORMAL = 0
RKLLM_STATE_WAITING = 1
RKLLM_STATE_FINISH = 2
RKLLM_STATE_STOP = 3
RKLLM_STATE_MAX_NEW_TOKEN = 4
RKLLM_STATE_ERROR = 5


# ============================= Global callback state =============================

tokenizer = None
embeds_data: Optional[np.ndarray] = None
first_token_time: Optional[float] = None
model_outputs: List[np.ndarray] = []
verbose_output = True

# Keep ctypes py_object references alive for the whole session.
_user_data_refs = []


# ============================= Callback functions =============================

def result_callback(userdata, result_ptr, state):
    """Mostly unused by embedding model, but kept for compatibility/debug."""
    global tokenizer, first_token_time

    if not hasattr(result_callback, "accumulated_tokens"):
        result_callback.accumulated_tokens = []
        result_callback.last_output_text = ""

    def decode_safe(tokens: Sequence[int]) -> str:
        text = tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return text.split("\ufffd", 1)[0] if "\ufffd" in text else text

    # RKLLM_STATE_ERROR
    if state == RKLLM_STATE_ERROR:
        print("\n\nError occurred during inference")
        return 0

    # RKLLM_STATE_FINISH / RKLLM_STATE_STOP / RKLLM_STATE_MAX_NEW_TOKEN
    if state in (RKLLM_STATE_FINISH, RKLLM_STATE_STOP, RKLLM_STATE_MAX_NEW_TOKEN):
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
        msg = {
            RKLLM_STATE_FINISH: "Finished",
            RKLLM_STATE_STOP: "Stop",
            RKLLM_STATE_MAX_NEW_TOKEN: "Max new token reached",
        }.get(state, "Unknown")
        print(f"\n\n--------------------{msg}--------------------")
        return 0

    # RKLLM_STATE_WAITING
    if state == RKLLM_STATE_WAITING:
        print("\n\nWaiting for UTF-8 encoded character")
        return 0

    # RKLLM_STATE_NORMAL
    if state == RKLLM_STATE_NORMAL:
        n = result_ptr.contents.num_tokens
        new_tokens = [result_ptr.contents.token_ids[i] for i in range(n)]
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

    return 0


def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    """Tokenize prompt text for RKLLM."""
    global tokenizer

    # Always use ctypes.string_at with explicit length for safety and accuracy.
    raw = ctypes.string_at(text_ptr, text_len)
    text = raw.decode("utf-8", errors="replace")

    inputs = tokenizer(text, return_tensors="np", truncation=True)
    tokens = inputs["input_ids"][0][:n_tokens_max]
    n_tokens = int(len(tokens))

    if n_tokens <= 0:
        print(f"Tokenizer failed for {text}")
        return n_tokens

    for i, tok in enumerate(tokens):
        tokens_ptr[i] = int(tok)

    return n_tokens


def embed_callback(userdata, tokens_ptr, num_tokens, embed, length):
    """Fill RKLLM embedding input buffer from *.embed.bin."""
    global embeds_data

    if embeds_data is None:
        print("embeds_data is not initialized")
        return -1

    embedding_dim = int(embeds_data.shape[1])
    expected_len = int(num_tokens) * embedding_dim * np.dtype(np.float16).itemsize
    if int(length) != expected_len:
        print(f"invalid embed buffer: length={int(length)}, expected={expected_len}")
        return -1

    dst = np.ctypeslib.as_array(
        ctypes.cast(embed, ctypes.POINTER(ctypes.c_uint16)),
        shape=(int(num_tokens) * embedding_dim,),
    ).view(np.float16)

    tokens = np.fromiter((int(tokens_ptr[i]) for i in range(int(num_tokens))), dtype=np.int64)
    if tokens.size > 0:
        min_token = int(tokens.min())
        max_token = int(tokens.max())
        if min_token < 0 or max_token >= embeds_data.shape[0]:
            print(f"token id out of range: min={min_token}, max={max_token}, vocab={embeds_data.shape[0]}")
            return -1

    dst[:] = embeds_data[tokens].reshape(-1)
    return 0


def output_callback(userdata, output_tensors_ptr, n_output_tensors, state):
    """Capture output tensor at PREFILL_FINISHED, matching C++ output_callback."""
    global first_token_time, model_outputs, verbose_output

    if verbose_output:
        print(f"\noutput_callback: state = {state}")

    if state != LLMOutputCallbackState.RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED:
        return 0

    if first_token_time is None:
        first_token_time = time.perf_counter()

    if n_output_tensors == 0 or not output_tensors_ptr:
        print("output_callback: empty output_tensors")
        return 0

    try:
        for i in range(int(n_output_tensors)):
            tensor = output_tensors_ptr[i]
            if not tensor.mem or not tensor.attr:
                continue

            mem = tensor.mem.contents
            attr = tensor.attr.contents
            n_elems = int(attr.n_elems)

            name = attr.name.decode("utf-8", errors="ignore") if attr.name else ""
            if verbose_output:
                print(f"output_callback: output[{i}]->attr->index = {attr.index}")
                print(f"output_callback: output[{i}]->attr->name = {name}")
                print(f"output_callback: output[{i}]->mem->size = {mem.size}")

            if not mem.virt_addr or n_elems <= 0:
                continue

            # Validate memory size: fp16 output requires at least n_elems * 2 bytes.
            if int(mem.size) < n_elems * 2:
                print(f"output_callback: mem.size={mem.size} < expected={n_elems * 2} for output[{i}]")
                continue

            data = np.ctypeslib.as_array(
                ctypes.cast(mem.virt_addr, ctypes.POINTER(ctypes.c_uint16)),
                shape=(n_elems,),
            ).view(np.float16).astype(np.float32, copy=True)

            model_outputs.append(data)

            if verbose_output:
                for j in range(min(10, n_elems)):
                    print(f"output_callback: output[{i}][{j}] = {data[j]:.6f}")

    except Exception as exc:
        print(f"Error in output_callback: {exc}")
        import traceback
        traceback.print_exc()
        return -1

    return 0


# ============================= Helper functions =============================

def _get_first_id(value, default: int = -1) -> int:
    """Convert tokenizer special token id/list into one scalar id for rknn3lite args."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else default
    return int(value)


def _as_bytes_name(name: str):
    return name.encode("utf-8") if isinstance(name, str) else name


def load_embedding_table(embed_path: str, vocab_size: int, use_memmap: bool = True) -> np.ndarray:
    """Load or mmap fp16 embedding table and reshape to [vocab_size, embedding_dim]."""
    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"embedding file not found: {embed_path}")

    file_size = os.path.getsize(embed_path)
    itemsize = np.dtype(np.float16).itemsize
    if file_size % itemsize != 0:
        raise ValueError(f"embedding file size {file_size} is not aligned to fp16 itemsize")

    total_elems = file_size // itemsize
    if vocab_size <= 0:
        raise ValueError(f"invalid vocab_size: {vocab_size}")
    if total_elems % vocab_size != 0:
        raise ValueError(
            f"embedding fp16 elements {total_elems} cannot be divided by vocab_size {vocab_size}"
        )

    embedding_dim = total_elems // vocab_size
    if use_memmap:
        table = np.memmap(embed_path, dtype=np.float16, mode="r", shape=(vocab_size, embedding_dim))
    else:
        table = np.fromfile(embed_path, dtype=np.float16).reshape(vocab_size, embedding_dim)

    print(f"embedding: vocab_size={vocab_size}, embedding_dim={embedding_dim}, file={embed_path}")
    return table


def printf_perf(first_token: Optional[float], n_decode_tokens: int, n_prefill_tokens: int,
                llm_start_time: float, llm_end_time: float):
    """Print timing table similar to the C++ demo / Python references."""
    print("\n--------------------------------------------------------------------------------------")
    print(" %-12s  %-15s  %-8s  %-23s  %-23s" % (
        "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("--------------------------------------------------------------------------------------")

    if first_token is None:
        # If output callback did not run, avoid crashing; use end time as fallback.
        first_token = llm_end_time

    prefill_ms = max((first_token - llm_start_time) * 1000.0, 0.0)
    if n_prefill_tokens == 0 or prefill_ms == 0.0:
        prefill_tpt, prefill_tps = 0.0, 0.0
    else:
        prefill_tpt = prefill_ms / n_prefill_tokens
        prefill_tps = n_prefill_tokens * 1000.0 / prefill_ms
    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" % (
        "Prefill", prefill_ms, n_prefill_tokens, prefill_tpt, prefill_tps))

    decode_ms = max((llm_end_time - first_token) * 1000.0, 0.0)
    if n_decode_tokens == 0 or decode_ms == 0.0:
        decode_tpt, decode_tps = 0.0, 0.0
    else:
        decode_tpt = decode_ms / n_decode_tokens
        decode_tps = n_decode_tokens * 1000.0 / decode_ms
    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" % (
        "Generate", decode_ms, n_decode_tokens, decode_tpt, decode_tps))
    print("--------------------------------------------------------------------------------------")


def run_session(rknn: RKNN3Lite, prompt: str, keep_history: bool = False):
    """Call session_run while staying compatible with multiple rknn3lite versions."""
    # Some versions may not have enable_thinking keyword. Try the closest C++ behavior first.
    try:
        return rknn.session_run(prompt=prompt, keep_history=keep_history, enable_thinking=False)
    except TypeError:
        try:
            return rknn.session_run(prompt=prompt, keep_history=keep_history)
        except TypeError:
            return rknn.session_run(prompt=prompt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference Qwen3 Embedding RKNN model and save output.npy"
    )
    parser.add_argument("--rknn_path", required=True, help="Qwen3 embedding .rknn path")
    parser.add_argument("--weight_path", default=None, help=".weight path; default: rknn_path with .weight suffix")
    parser.add_argument("--tokenizer_path", required=True, help="HF tokenizer dir/name, e.g. Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embed_path", required=True, help="*.embed.bin path")
    parser.add_argument("--prompt", required=True, help="input text to embed")
    parser.add_argument("--output_path", default="output.npy", help="saved embedding .npy path")
    parser.add_argument("--target", default="rk1820", help="target for init_runtime; default: rk1820")
    parser.add_argument("--core_mask", default="0xff", help="NPU core mask, e.g. 0xff")
    parser.add_argument("--max_new_tokens", type=int, default=1, help="match C++ MAX_NEW_TOKENS, default: 1")
    parser.add_argument("--output_index", type=int, default=0, help="output tensor index to capture, default: 0")
    parser.add_argument("--logits_name", default="output", help="LLM logits/output name, default: output")
    parser.add_argument("--load_embed_to_ram", action="store_true", help="load embed.bin fully into RAM instead of np.memmap")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="allow execution of remote code from tokenizer repo (security risk)")
    parser.add_argument("--local_files_only", action="store_true", help="do not download tokenizer from HF")
    parser.add_argument("--quiet", action="store_true", help="less output_callback debug logging")
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    """Validate parsed arguments and return user-friendly errors."""
    # Validate core_mask is a valid hex string.
    try:
        int(args.core_mask, 0)
    except ValueError:
        raise ValueError(f"Invalid --core_mask value: {args.core_mask}. Must be a hex string like '0xff'.")

    # Validate max_new_tokens is positive.
    if args.max_new_tokens <= 0:
        raise ValueError(f"--max_new_tokens must be > 0, got {args.max_new_tokens}")

    # Validate rknn_path exists.
    if not os.path.isfile(args.rknn_path):
        raise FileNotFoundError(f"--rknn_path not found: {args.rknn_path}")

    # Validate weight_path exists.
    weight_path = args.weight_path or str(Path(args.rknn_path).with_suffix(".weight"))
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"--weight_path not found: {weight_path}")

    # Validate output_path parent directory exists.
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir and not os.path.isdir(output_dir):
        raise NotADirectoryError(f"Output directory does not exist: {output_dir}")

    # Validate tokenizer_path exists.
    if not os.path.isdir(args.tokenizer_path):
        raise NotADirectoryError(f"--tokenizer_path not found: {args.tokenizer_path}")

    # Validate embed_path exists.
    if not os.path.isfile(args.embed_path):
        raise FileNotFoundError(f"--embed_path not found: {args.embed_path}")


def main() -> int:
    global tokenizer, embeds_data, first_token_time, model_outputs, verbose_output

    args = parse_args()
    validate_args(args)
    core_mask = int(args.core_mask, 0)
    weight_path = args.weight_path or str(Path(args.rknn_path).with_suffix(".weight"))
    verbose_output = not bool(args.quiet)

    print("--> Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    print("done")

    rknn = RKNN3Lite(llm_mode=True, verbose=not args.quiet)

    try:
        print("--> Loading model")
        ret = rknn.load_rknn(args.rknn_path, weight_path)
        if ret != 0:
            print(f"Load model failed! ret={ret}")
            return int(ret)
        print("done")

        print("--> Init runtime environment")
        ret = rknn.init_runtime(
            target=args.target,
            core_mask=core_mask,
        )
        if ret != 0:
            print(f"Init runtime environment failed! ret={ret}")
            return int(ret)
        print("done")

        print("--> Query model info")
        io_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
        print(f"model input num: {io_num.n_input}, output num: {io_num.n_output}")

        for i in range(int(io_num.n_output)):
            output_attr = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=i)
            dump_tensor_attr(output_attr, prefix="output")

        llm_config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
        config_vocab_size = int(getattr(llm_config, "vocab_size", 0) or 0)
        tokenizer_vocab_size = len(tokenizer)
        vocab_size = config_vocab_size if config_vocab_size > 0 else tokenizer_vocab_size
        print(
            f"llm_config: max_ctx_len={llm_config.max_ctx_len}, "
            f"max_position_embeddings={getattr(llm_config, 'max_position_embeddings', 'N/A')}, "
            f"vocab_size={vocab_size}"
        )

        embeds_data = load_embedding_table(
            args.embed_path,
            vocab_size=vocab_size,
            use_memmap=not args.load_embed_to_ram,
        )

        if args.output_index < 0 or args.output_index >= int(io_num.n_output):
            print(f"invalid --output_index {args.output_index}, model has {io_num.n_output} outputs")
            return -1

        # Allocate exactly one output tensor like the C++ demo: output_tensors_index[0] = {0}.
        n_output_tensors = 1
        OutputTensorArray = RKNN3Tensor * n_output_tensors
        output_tensors = OutputTensorArray()

        attr = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=args.output_index)
        if attr is None:
            print(f"rknn3_query output attr failed for index {args.output_index}")
            return -1
        if rknn.create_mem(attr, output_tensors[0]) is None:
            print(f"Failed to create memory for output tensor {args.output_index}")
            return -1
        print(f"capture output tensor[{args.output_index}]: {attr.name.decode('utf-8', errors='ignore')}")

        llm_args = [{
            "max_new_tokens": int(args.max_new_tokens),
            "top_k": 1,
            "top_p": 0.0,
            "temperature": 1.0,
            "repeat_penalty": 1.0,
            "vocab_size": int(vocab_size),
            "special_bos_id": _get_first_id(getattr(tokenizer, "bos_token_id", None), -1),
            "special_eos_id": _get_first_id(getattr(tokenizer, "eos_token_id", None), -1),
            "max_context_len": int(llm_config.max_ctx_len),
            "keep_history": 0,
            "logits_name": _as_bytes_name(args.logits_name),
        }]

        print("llm_args:", llm_args[0])

        callback = RKLLMCallback()
        callback.result_callback = LLMResultCallback(result_callback)
        callback.result_userdata = None

        callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
        tok_ud = ctypes.py_object(tokenizer)
        _user_data_refs.append(tok_ud)
        callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(tok_ud), ctypes.c_void_p)

        callback.embed_callback = LLMGetEmbedCallback(embed_callback)
        emb_ud = ctypes.py_object(embeds_data)
        _user_data_refs.append(emb_ud)
        callback.embed_userdata = ctypes.cast(ctypes.pointer(emb_ud), ctypes.c_void_p)

        callback.output_callback = LLMOutputCallback(output_callback)
        out_ud = ctypes.py_object(model_outputs)
        _user_data_refs.append(out_ud)
        callback.output_userdata = ctypes.cast(ctypes.pointer(out_ud), ctypes.c_void_p)
        callback.output_tensors = ctypes.cast(output_tensors, ctypes.POINTER(RKNN3Tensor))
        callback.n_output_tensors = n_output_tensors

        print("--> Init LLM session")
        ret = rknn.init_llm_session(llm_args=llm_args, llm_callback=callback)
        if ret != 0:
            print(f"Init LLM session failed! ret={ret}")
            return int(ret)
        print("done")


        print("\n--------------------Input--------------------")
        if not args.quiet:
            print(args.prompt)
        print("\n--------------------Run----------------------")

        first_token_time = None
        model_outputs.clear()
        ret, perf_values = run_session(rknn, args.prompt, keep_history=False)
        if ret != 0:
            print(f"RKNN Qwen3 embedding inference failed! ret={ret}")
            return int(ret)

        n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time = perf_values
        if first_token_time is not None:
            printf_perf(first_token_time, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time)
        else:
            printf_perf(None, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time)

        if not model_outputs:
            print("No output tensor captured. Please check output_callback state/output_index/logits_name.")
            return -1

        output = model_outputs[0]
        if args.output_path.endswith(".npy"):
            np.save(args.output_path, output)
        else:
            np.save(args.output_path + ".npy", output)
            args.output_path = args.output_path + ".npy"

        print(f"Saved embedding: {args.output_path}, shape={output.shape}, dtype={output.dtype}")
        print("done")
        return 0

    finally:
        # Release output memory before releasing RKNN context.
        try:
            if "output_tensors" in locals():
                for tensor in output_tensors:
                    if tensor.mem:
                        rknn.destroy_mem(tensor.mem)
        except Exception as exc:
            print(f"destroy_mem warning: {exc}")
        try:
            rknn.release()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())