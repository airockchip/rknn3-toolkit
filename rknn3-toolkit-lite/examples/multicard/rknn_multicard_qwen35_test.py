import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

from argparse import ArgumentParser
from collections import deque
from dataclasses import dataclass, field
import ctypes
import re
import threading
import time
from typing import Deque, List, Optional

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
    RKNN3Tensor,
    LLMOutputCallback,
    LLMOutputCallbackState,
    dump_tensor_attr,
)


# ============================================= Default Config =============================================

RKNN_MODEL = "/userdata/model/Qwen3.5-0.8B-llm_seg0.rknn"
TOKENIZER_PATH = "./qwen3_5_0_8B"
EMBED_PATH = "/userdata/model/Qwen3.5-0.8B-llm.embed.bin"

PROMPT_PREFIX = "<|im_start|>user\n"
PROMPT_POSTFIX = "<|im_end|>\n<|im_start|>assistant\n"

QWEN35_FALLBACK_EOS_ID = 151645
QWEN35_FALLBACK_BOS_ID = 151643


SCRIPT_VERSION = "v5"

tokenizer = None
embeds_data = None


# ============================================= Pipeline Data =============================================

@dataclass
class TensorBlob:
    name: str
    dtype: int
    n_elems: int
    aligned_size: int
    data: bytes


@dataclass
class StageBatch:
    tensors: List[TensorBlob]
    n_tokens: int = 0


@dataclass
class StageSlot:
    batches: Deque[StageBatch] = field(default_factory=deque)
    expected_tokens: int = 0
    emitted_tokens: int = 0
    active_input_tokens: int = 0
    producer_done: bool = False
    failed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.cv = threading.Condition(self.lock)

    def reset(self):
        with self.cv:
            self.batches.clear()
            self.expected_tokens = 0
            self.emitted_tokens = 0
            self.active_input_tokens = 0
            self.producer_done = False
            self.failed = False

    def close(self):
        with self.cv:
            self.producer_done = True
            self.cv.notify_all()

    def mark_failed(self):
        with self.cv:
            self.failed = True
            self.cv.notify_all()

    def wait_batch(self) -> Optional[StageBatch]:
        with self.cv:
            while not self.batches and not self.producer_done and not self.failed:
                self.cv.wait()
            if self.failed or not self.batches:
                return None
            return self.batches.popleft()


class PipelineState:
    def __init__(self, stage_count: int, bucket_size: int, verbose: bool = False):
        self.slots = [StageSlot() for _ in range(stage_count)]
        self.bucket_size = bucket_size
        self.verbose = verbose

    def reset(self):
        for slot in self.slots:
            slot.reset()

    def fail(self):
        for slot in self.slots:
            slot.mark_failed()

    def failed(self) -> bool:
        return any(slot.failed for slot in self.slots)


class LastStageResultState:
    def __init__(self):
        self.lock = threading.Lock()
        self.token_event = threading.Event()
        self.has_token = False
        self.next_token = -1
        self.source_stage = -1
        self.generated_tokens: List[int] = []
        self.last_output_text = ""

    def reset_next_token(self):
        with self.lock:
            self.has_token = False
            self.next_token = -1
            self.source_stage = -1
            self.token_event.clear()

    def reset_generation(self):
        with self.lock:
            self.has_token = False
            self.next_token = -1
            self.source_stage = -1
            self.generated_tokens.clear()
            self.last_output_text = ""
            self.token_event.clear()

    def set_tokens(self, tokens: List[int], source_stage: int):
        if not tokens:
            return
        with self.lock:
            self.next_token = int(tokens[-1])
            self.has_token = True
            self.source_stage = int(source_stage)
            self.generated_tokens.extend(int(x) for x in tokens)
            self.token_event.set()

    def get_next_token(self) -> Optional[int]:
        with self.lock:
            if not self.has_token:
                return None
            return self.next_token

    def get_source_stage(self) -> int:
        with self.lock:
            return self.source_stage

    def wait_next_token(self, timeout: float = 2.0) -> Optional[int]:
        self.token_event.wait(timeout)
        return self.get_next_token()


LAST_RESULT = LastStageResultState()


@dataclass
class StageCallbackContext:
    pipeline: PipelineState
    stage_index: int
    embedding_dim: int


@dataclass
class ResultCallbackContext:
    stage_index: int
    is_last_stage: bool


@dataclass
class EmbedCallbackContext:
    pipeline: PipelineState


class StageRuntime:
    def __init__(self, name: str, model_path: str, weight_path: str):
        self.name = name
        self.model_path = model_path
        self.weight_path = weight_path
        self.rknn: Optional[RKNN3Lite] = None
        self.embedding_dim = 0
        self.vocab_size = 0
        self.max_ctx_len = 0
        self.max_position_embeddings = 0
        self.model_type = ""
        self.output_tensors = None
        self.n_output_tensors = 0
        self.callback = None
        self.callback_refs = []
        self.userdata_refs = []

    def release(self):
        # 1. destroy explicit output memory allocated by create_output_tensors()
        if self.rknn is not None and self.output_tensors is not None:
            for i in range(self.n_output_tensors):
                try:
                    mem = self.output_tensors[i].mem
                    if mem:
                        self.rknn.destroy_mem(mem)
                        self.output_tensors[i].mem = None
                except Exception:
                    pass
            self.output_tensors = None
            self.n_output_tensors = 0

        # 2. release native runtime first — after this the C side will no
        #    longer invoke any callback, so it is safe to drop Python refs.
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

        # 3. clear callback / userdata refs (order matters: native must be
        #    gone before the Python holders are garbage-collected).
        self.callback = None
        self.callback_refs.clear()
        self.userdata_refs.clear()


# ============================================= Helpers =============================================

def vlog(pipeline: PipelineState, msg: str):
    if pipeline.verbose:
        print(msg, flush=True)


def make_userdata(obj, keepalive: list):
    """Wrap a Python object as a void-ptr userdata for C callbacks.

    Returns a c_void_p value to pass to the C side as userdata.

    The ctypes.py_object *holder* that keeps *obj* alive is appended to
    *keepalive* (typically StageRuntime.userdata_refs).  The caller is
    responsible for keeping that list alive until after the native runtime
    is released.
    """
    holder = ctypes.py_object(obj)
    ptr = ctypes.cast(ctypes.pointer(holder), ctypes.c_void_p)
    keepalive.extend([obj, holder])
    return ptr


def userdata_value(userdata):
    return ctypes.cast(userdata, ctypes.POINTER(ctypes.py_object)).contents.value


def replace_seg_suffix(path: str, seg_idx: int) -> str:
    match = re.search(r"_seg\d+(?=\.[^.]+$|$)", path)
    if match:
        return path[:match.start()] + f"_seg{seg_idx}" + path[match.end():]

    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}_seg{seg_idx}{ext}"
    return f"{path}_seg{seg_idx}"


def resolve_weight_path(rknn_path: str, weight_path: Optional[str]) -> str:
    if weight_path:
        return weight_path
    root, _ = os.path.splitext(rknn_path)
    return root + ".weight"


def load_prompt(prompt_arg: Optional[str]) -> str:
    if not prompt_arg:
        return "请解释一下相对论的基本概念。"
    return prompt_arg


def normalize_device_id(device_id) -> bytes:
    """RKNN3InitExtend.device_id is c_char_p, so runtime must receive bytes."""
    if isinstance(device_id, bytes):
        return device_id
    if isinstance(device_id, bytearray):
        return bytes(device_id)
    return str(device_id).encode("utf-8")


def format_device_id(device_id) -> str:
    if isinstance(device_id, (bytes, bytearray)):
        return bytes(device_id).decode("utf-8", errors="ignore")
    return str(device_id)


def decode_c_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def resolve_special_token_ids(tok, eos_override=None, bos_override=None):
    """Qwen3.5 expects BOS/EOS arrays in RKNN3 LLM params, not scalar ids."""
    if eos_override:
        eos_ids = [int(x) for x in eos_override]
    else:
        eos = tok.eos_token_id
        eos_ids = [int(eos if eos is not None else QWEN35_FALLBACK_EOS_ID)]

    if bos_override:
        bos_ids = [int(x) for x in bos_override]
    else:
        bos = tok.bos_token_id
        bos_ids = [int(bos if bos is not None else QWEN35_FALLBACK_BOS_ID)]

    return eos_ids, bos_ids


def pick_embed_tensor(tensors: List[TensorBlob]) -> Optional[TensorBlob]:
    if not tensors:
        return None
    for tensor in tensors:
        name = tensor.name.lower()
        if "hidden" in name or "last_hidden" in name or "output" in name:
            return tensor
    return tensors[0]


# ============================================= Callbacks =============================================

def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    global tokenizer
    try:
        # Match the C++ multicard demo: tokenizer is carried by tokenizer_userdata.
        # Fall back to the module-global tokenizer only for compatibility.
        tok = userdata_value(userdata) if userdata else tokenizer
        if tok is None:
            print("tokenizer_callback: tokenizer userdata is None")
            return -1
        text = text_ptr.decode("utf-8")
        inputs = tok(text, return_tensors="np", truncation=True)
        tokens = inputs["input_ids"][0][:n_tokens_max]
        n_tokens = int(len(tokens))
        if n_tokens <= 0:
            print(f"Tokenizer failed for {text}")
            return n_tokens
        for i in range(n_tokens):
            tokens_ptr[i] = int(tokens[i])
        return n_tokens
    except Exception as e:
        print(f"tokenizer_callback failed: {e}")
        return -1


def embed_callback(userdata, tokens_ptr, num_tokens, embed, length):
    global embeds_data
    try:
        ctx: EmbedCallbackContext = userdata_value(userdata)
        embedding_dim = int(embeds_data.shape[1])
        expected_len = int(num_tokens) * embedding_dim * np.dtype(np.float16).itemsize
        if int(length) != expected_len:
            print(f"invalid embed buffer: length={length}, expected={expected_len}")
            return -1

        token_ids = np.ctypeslib.as_array(tokens_ptr, shape=(int(num_tokens),)).astype(np.int64, copy=False)
        if token_ids.size and (token_ids.min() < 0 or token_ids.max() >= embeds_data.shape[0]):
            print("embed_callback token id out of range")
            return -1

        dst = np.ctypeslib.as_array(
            ctypes.cast(embed, ctypes.POINTER(ctypes.c_uint16)),
            shape=(int(num_tokens) * embedding_dim,),
        ).view(np.float16)
        dst[:] = embeds_data[token_ids].reshape(-1)

        # Stage0 input token count; this is also how the C++ pipeline determines the final bucket.
        slot = ctx.pipeline.slots[0]
        with slot.cv:
            slot.expected_tokens += int(num_tokens)
            total = slot.expected_tokens
        vlog(ctx.pipeline, f"[embed_callback] num_tokens={num_tokens}, total={total}")
        return 0
    except Exception as e:
        print(f"embed_callback failed: {e}")
        return -1


def result_callback(userdata, result_ptr, state):
    """Minimal native callback.

    RKNN3 requires a result callback on every segmented session, but only the
    final stage owns logits/sampling.  Never let an intermediate stage overwrite
    the token used by the decode loop.  Also avoid calling HuggingFace tokenizer
    from the RKNN native callback thread; decoding is done later on the main
    Python thread.
    """
    try:
        ctx: ResultCallbackContext = userdata_value(userdata) if userdata else None
        if ctx is None or not ctx.is_last_stage:
            return 0

        if int(state) == 5:
            print(f"\n[last-stage result_callback] RKNN error state on stage{ctx.stage_index}")
            return 0
        if int(state) != 0 or not result_ptr:
            return 0

        n = int(result_ptr.contents.num_tokens)
        if n <= 0:
            return 0
        tokens = [int(result_ptr.contents.token_ids[i]) for i in range(n)]
        LAST_RESULT.set_tokens(tokens, source_stage=ctx.stage_index)
        return 0
    except Exception as e:
        print(f"result_callback failed: {e}")
        return 0


def flush_generated_text():
    """Decode/print generated tokens on the main Python thread only."""
    global tokenizer
    with LAST_RESULT.lock:
        if not LAST_RESULT.generated_tokens:
            return
        try:
            text = tokenizer.decode(
                LAST_RESULT.generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            safe_text = text.split("\ufffd", 1)[0] if "\ufffd" in text else text
            new_part = safe_text[len(LAST_RESULT.last_output_text):]
            if new_part:
                print(new_part, end="", flush=True)
                LAST_RESULT.last_output_text += new_part
        except Exception as e:
            print(f"\n[Decode error: {e}]", flush=True)


def stage_output_callback(userdata, output_tensors, n_output_tensors, state):
    try:
        cb_ctx: StageCallbackContext = userdata_value(userdata)
        pipeline = cb_ctx.pipeline
        slot = pipeline.slots[cb_ctx.stage_index]

        tensors = []
        for i in range(int(n_output_tensors)):
            tensor = output_tensors[i]
            if not tensor.attr or not tensor.mem:
                continue
            attr = tensor.attr.contents
            mem = tensor.mem.contents
            if not mem.virt_addr or int(attr.aligned_size) <= 0:
                continue

            name = bytes(attr.name).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
            data = ctypes.string_at(mem.virt_addr, int(attr.aligned_size))
            blob = TensorBlob(
                name=name,
                dtype=int(attr.dtype),
                n_elems=int(attr.n_elems),
                aligned_size=int(attr.aligned_size),
                data=data,
            )
            tensors.append(blob)
            vlog(
                pipeline,
                f"[stage{cb_ctx.stage_index}] captured output[{i}] name={name}, "
                f"dtype={blob.dtype}, n_elems={blob.n_elems}, aligned_size={blob.aligned_size}",
            )

        if not tensors:
            return 0

        with slot.cv:
            remaining = max(0, slot.expected_tokens - slot.emitted_tokens)
            if slot.active_input_tokens > 0:
                n_tokens = slot.active_input_tokens
            elif int(state) == int(LLMOutputCallbackState.RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED):
                n_tokens = remaining
            else:
                n_tokens = min(remaining, pipeline.bucket_size)

            if n_tokens == 0:
                embed = pick_embed_tensor(tensors)
                if embed is not None and embed.n_elems > 0 and cb_ctx.embedding_dim > 0:
                    n_tokens = embed.n_elems // cb_ctx.embedding_dim

            slot.emitted_tokens += int(n_tokens)
            slot.batches.append(StageBatch(tensors=tensors, n_tokens=int(n_tokens)))
            slot.cv.notify()

        vlog(
            pipeline,
            f"[stage{cb_ctx.stage_index}] output_callback state={int(state)}, "
            f"batch_tokens={n_tokens}, emitted={slot.emitted_tokens}/{slot.expected_tokens}",
        )
        return 0
    except Exception as e:
        print(f"stage_output_callback failed: {e}")
        try:
            userdata_value(userdata).pipeline.fail()
        except Exception:
            pass
        return -1


# ============================================= Stage Init =============================================

def init_stage(
    stage: StageRuntime,
    pipeline: PipelineState,
    stage_idx: int,
    device_id: str,
    target: str,
    core_mask: int,
    max_context_len: int,
    max_new_token: int,
    decrypt_key_path: Optional[str],
    ignore_eos: bool,
    special_eos_ids: List[int],
    special_bos_ids: List[int],
    presence_penalty: float,
    kvcache_policy: str,
    verbose: bool,
) -> bool:
    global tokenizer

    print(
        f"[{stage.name}] init stage: model={stage.model_path}, weight={stage.weight_path}, "
        f"device_id={format_device_id(device_id)}"
    )

    stage.rknn = RKNN3Lite(llm_mode=True, verbose=verbose)
    ret = stage.rknn.load_rknn(stage.model_path, stage.weight_path)
    if ret != 0:
        print(f"[{stage.name}] load_rknn failed")
        return False

    ret = stage.rknn.init_runtime(
        target=target,
        core_mask=core_mask,
        device_id=device_id,
        decrypt_key_path=decrypt_key_path,
    )
    if ret != 0:
        print(f"[{stage.name}] init_runtime failed")
        return False

    io_num = stage.rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
    llm_config = stage.rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
    if io_num is None or llm_config is None:
        print(f"[{stage.name}] query model info failed")
        return False

    stage.embedding_dim = int(llm_config.embedding_dim)
    stage.vocab_size = int(llm_config.vocab_size)
    if stage.vocab_size <= 0:
        stage.vocab_size = len(tokenizer)
    stage.max_ctx_len = int(llm_config.max_ctx_len)
    stage.max_position_embeddings = int(getattr(llm_config, "max_position_embeddings", 0))
    stage.model_type = decode_c_string(getattr(llm_config, "model_type", b""))

    print(f"[{stage.name}] model input num: {io_num.n_input}, output num: {io_num.n_output}")
    print(
        f"[{stage.name}] embedding_dim={stage.embedding_dim}, vocab_size={stage.vocab_size}, "
        f"max_ctx_len={stage.max_ctx_len}, max_position_embeddings={stage.max_position_embeddings}, "
        f"model_type={stage.model_type}"
    )

    if max_context_len > stage.max_ctx_len:
        print(
            f"[{stage.name}] ERROR: max_context_len ({max_context_len}) is greater than "
            f"model max_ctx_len ({stage.max_ctx_len})"
        )
        return False
    if 0 < max_context_len < stage.max_ctx_len:
        print(
            f"[{stage.name}] Warning: max_context_len={max_context_len} < model max_ctx_len={stage.max_ctx_len}. "
            "This is intentional only if you want a smaller KV-cache window."
        )

    if verbose:
        for i in range(int(io_num.n_input)):
            attr = stage.rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_INPUT_ATTR, index=i)
            dump_tensor_attr(attr, prefix=f"{stage.name}_input")
        for i in range(int(io_num.n_output)):
            attr = stage.rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=i)
            dump_tensor_attr(attr, prefix=f"{stage.name}_output")

    is_last_stage = stage_idx == len(pipeline.slots) - 1
    if not is_last_stage:
        output_tensors, n_output_tensors = stage.rknn.create_output_tensors()
        if output_tensors is None or n_output_tensors <= 0:
            print(f"[{stage.name}] create_output_tensors failed")
            return False
        stage.output_tensors = output_tensors
        stage.n_output_tensors = int(n_output_tensors)

    ctx_len = max_context_len if max_context_len > 0 else stage.max_ctx_len
    llm_args = [{
        "max_new_tokens": 1,
        "top_k": 1,
        "top_p": 0.9,
        "temperature": 1.0,
        "repeat_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": presence_penalty,
        "vocab_size": stage.vocab_size,
        # Qwen3.5 example passes BOS/EOS as arrays. Keep the same ABI-facing form.
        "special_eos_id": special_eos_ids,
        "special_bos_id": special_bos_ids,
        "max_context_len": ctx_len,
        # IMPORTANT: multicard decode is split into many prefill_only calls.
        # keep_history must stay enabled so every stage keeps its own KV state across token steps.
        "keep_history": 1,
        "ignore_eos_token": ignore_eos,
        "logits_name": b"output",
    }]

    callback = RKLLMCallback()
    stage.callback = callback

    # Tokenizer + embedding callback. Only stage0 needs them for prompt/token input,
    # but registering on every stage keeps behavior aligned with the C++ demo.
    tok_cb = LLMTokenizerCallback(tokenizer_callback)
    emb_cb = LLMGetEmbedCallback(embed_callback)
    callback.tokenizer_callback = tok_cb
    callback.embed_callback = emb_cb
    stage.callback_refs.extend([tok_cb, emb_cb])

    # IMPORTANT: the C++ multicard demo registers tokenizer_userdata on every
    # stage.  The RKNN3 runtime treats the tokenizer callback as incomplete
    # without its userdata in this flow. Keep the ctypes holder alive.
    callback.tokenizer_userdata = make_userdata(tokenizer, stage.userdata_refs)

    embed_ctx = EmbedCallbackContext(pipeline=pipeline)
    callback.embed_userdata = make_userdata(embed_ctx, stage.userdata_refs)

    # RKNN3 expects a result callback on every stage, but only the final stage
    # is allowed to publish sampled tokens to LAST_RESULT.  This prevents an
    # intermediate stage callback from poisoning the next-token decode input.
    res_cb = LLMResultCallback(result_callback)
    result_ctx = ResultCallbackContext(stage_index=stage_idx, is_last_stage=is_last_stage)
    callback.result_callback = res_cb
    callback.result_userdata = make_userdata(result_ctx, stage.userdata_refs)
    stage.callback_refs.append(res_cb)

    if not is_last_stage:
        out_cb = LLMOutputCallback(stage_output_callback)
        callback.output_callback = out_cb
        cb_ctx = StageCallbackContext(
            pipeline=pipeline,
            stage_index=stage_idx,
            embedding_dim=stage.embedding_dim,
        )
        callback.output_userdata = make_userdata(cb_ctx, stage.userdata_refs)
        callback.output_tensors = ctypes.cast(stage.output_tensors, ctypes.POINTER(RKNN3Tensor))
        callback.n_output_tensors = stage.n_output_tensors
        stage.callback_refs.append(out_cb)

    if verbose:
        print(
            f"[{stage.name}] callbacks: "
            f"tokenizer={bool(callback.tokenizer_callback)}, "
            f"tokenizer_userdata={bool(callback.tokenizer_userdata)}, "
            f"embed={bool(callback.embed_callback)}, "
            f"result={bool(callback.result_callback)}, "
            f"output={bool(callback.output_callback) if not is_last_stage else False}"
        )

    ret = stage.rknn.init_llm_session(llm_args=llm_args, llm_callback=callback)
    if ret != 0:
        print(f"[{stage.name}] init_llm_session failed")
        return False

    # Qwen3.5 reference example leaves KV-cache policy at runtime default.
    # Do not force NORMAL unless explicitly requested. This matters for hybrid/
    # sliding/linear-attention models where recurrent/default policy can differ.
    if kvcache_policy != "default":
        policy_map = {
            "normal": RKNN3KVCachePolicy.RKNN3_KVCACHE_POLICY_NORMAL,
            "recurrent": RKNN3KVCachePolicy.RKNN3_KVCACHE_POLICY_RECURRENT,
        }
        ret = stage.rknn.set_kvcache_policy(policy_map[kvcache_policy])
        if ret != 0:
            print(f"[{stage.name}] set_kvcache_policy({kvcache_policy}) failed")
            return False

    print(f"[{stage.name}] init done: outputs={stage.n_output_tensors}")
    return True


# ============================================= Pipeline Run =============================================

def run_stage_worker(stage_idx: int, stages: List[StageRuntime], pipeline: PipelineState):
    stage = stages[stage_idx]
    input_slot = pipeline.slots[stage_idx - 1]
    output_slot = pipeline.slots[stage_idx] if stage_idx + 1 < len(stages) else None
    is_last_stage = stage_idx == len(stages) - 1
    consumed_tokens = 0

    try:
        while True:
            batch = input_slot.wait_batch()
            if batch is None:
                with input_slot.cv:
                    clean_end = input_slot.producer_done and not input_slot.failed and not input_slot.batches
                if clean_end:
                    break
                if output_slot is not None:
                    output_slot.close()
                return

            embed = pick_embed_tensor(batch.tensors)
            if embed is None or batch.n_tokens <= 0 or stage.embedding_dim <= 0:
                print(f"[stage{stage_idx}] invalid pipeline batch")
                pipeline.fail()
                return

            embed_elems = batch.n_tokens * stage.embedding_dim
            embed_bytes = embed_elems * np.dtype(np.float16).itemsize
            if embed_bytes > len(embed.data):
                print(
                    f"[stage{stage_idx}] embed buffer too small: "
                    f"need={embed_bytes}, got={len(embed.data)}"
                )
                pipeline.fail()
                return

            # Copy only the logical [n_tokens, embedding_dim] part, not aligned padding.
            embed_array = np.frombuffer(embed.data, dtype=np.float16, count=embed_elems).reshape(
                batch.n_tokens, stage.embedding_dim
            ).copy()

            if output_slot is not None:
                with output_slot.cv:
                    output_slot.expected_tokens += batch.n_tokens
                    output_slot.active_input_tokens = batch.n_tokens

            disable_sampling = True
            if is_last_stage:
                with pipeline.slots[0].cv:
                    total_tokens = pipeline.slots[0].expected_tokens
                disable_sampling = consumed_tokens + batch.n_tokens < total_tokens

            consumed_tokens += batch.n_tokens
            vlog(
                pipeline,
                f"[stage{stage_idx}] consume batch: n_tokens={batch.n_tokens}, "
                f"bytes={embed_bytes}, disable_sampling={int(disable_sampling)}",
            )

            # Critical multicard invariant: the LLM EMBED input must report
            # n_tokens=batch.n_tokens, not embedding_dim. rknn3_session.py v5
            # fixes this by mapping a [T, D] numpy array to n_tokens=T.
            vlog(
                pipeline,
                f"[stage{stage_idx}] session_run EMBED shape={embed_array.shape}, "
                f"logical_n_tokens={batch.n_tokens}, hidden={stage.embedding_dim}, "
                f"nbytes={embed_array.nbytes}",
            )

            ret, _perf = stage.rknn.session_run(
                embeds=embed_array,
                keep_history=True,
                max_new_tokens=1,
                prefill_only=True,
                disable_sampling=disable_sampling,
            )

            if output_slot is not None:
                with output_slot.cv:
                    output_slot.active_input_tokens = 0

            if ret != 0:
                print(f"[stage{stage_idx}] run failed ret={ret}")
                pipeline.fail()
                return
    except Exception as e:
        print(f"[stage{stage_idx}] worker exception: {e}")
        pipeline.fail()
        return
    finally:
        if output_slot is not None:
            output_slot.close()


def run_pipeline_once(
    stages: List[StageRuntime],
    pipeline: PipelineState,
    prompt: Optional[str] = None,
    input_tokens: Optional[List[int]] = None,
    enable_thinking: bool = False,
    pipeline_mode: str = "threaded",
):
    if not stages:
        return False, 0

    pipeline.reset()
    LAST_RESULT.reset_next_token()

    workers = []
    if pipeline_mode == "threaded":
        for i in range(1, len(stages)):
            t = threading.Thread(
                target=run_stage_worker,
                args=(i, stages, pipeline),
                name=f"stage{i}-worker",
            )
            t.start()
            workers.append(t)

    stage0 = stages[0]
    try:
        stage0_disable_sampling = len(stages) > 1
        if prompt is not None:
            # Apply chat template internally instead of via set_chat_template,
            # because set_chat_template with a non-empty system_prompt triggers
            # internal alignment prefill that corrupts multicard KV-cache.
            prompt = PROMPT_PREFIX + prompt + PROMPT_POSTFIX
            ret, _perf = stage0.rknn.session_run(
                prompt=prompt,
                keep_history=True,
                max_new_tokens=1,
                prefill_only=True,
                disable_sampling=stage0_disable_sampling,
                enable_thinking=enable_thinking,
            )
        else:
            if not input_tokens:
                pipeline.fail()
                ret = -1
            else:
                ret, _perf = stage0.rknn.session_run(
                    tokens=np.asarray(input_tokens, dtype=np.int32),
                    keep_history=True,
                    max_new_tokens=1,
                    prefill_only=True,
                    disable_sampling=stage0_disable_sampling,
                    enable_thinking=enable_thinking,
                )

        if ret != 0:
            print(f"[stage0] run failed ret={ret}")
            pipeline.fail()
    except Exception as e:
        print(f"[stage0] run exception: {e}")
        ret = -1
        pipeline.fail()
    finally:
        pipeline.slots[0].close()

    if pipeline_mode == "threaded":
        for worker in workers:
            worker.join()
    else:
        # Functional-validation mode for Python/ctypes: execute each downstream
        # card on the caller thread after stage0 has copied all output buckets.
        # This preserves multicard correctness while removing Python worker-thread
        # lifetime/re-entry as a source of native segfaults.
        for i in range(1, len(stages)):
            run_stage_worker(i, stages, pipeline)
            if pipeline.failed():
                break

    with pipeline.slots[0].cv:
        stage0_input_tokens = pipeline.slots[0].expected_tokens

    return ret == 0 and not pipeline.failed(), stage0_input_tokens


# ============================================= Performance =============================================

def printf_perf(prefill_tokens, prefill_ms, decode_tokens, decode_ms):
    print("\n\nPerformance Statistics:")
    print("-----------------------------------------------------------------------------------------")
    print(" %-10s | %-16s | %-8s | %-20s | %-20s " % (
        "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("-----------------------------------------------------------------------------------------")

    if prefill_tokens == 0 or prefill_ms <= 0:
        prefill_tpt, prefill_tps = 0.0, 0.0
    else:
        prefill_tpt = prefill_ms / prefill_tokens
        prefill_tps = prefill_tokens * 1000.0 / prefill_ms
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Prefill", prefill_ms, prefill_tokens, prefill_tpt, prefill_tps))

    if decode_tokens == 0 or decode_ms <= 0:
        decode_tpt, decode_tps = 0.0, 0.0
    else:
        decode_tpt = decode_ms / decode_tokens
        decode_tps = decode_tokens * 1000.0 / decode_ms
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Generate", decode_ms, decode_tokens, decode_tpt, decode_tps))
    print("-----------------------------------------------------------------------------------------")


# ============================================= Main =============================================

def main():
    global tokenizer, embeds_data

    parser = ArgumentParser(
        description=(
            "RKNN3 multicard segmented LLM pipeline. Python version of main.cc, "
            "with CLI/callback style aligned to rknn_session_test.py"
        )
    )
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL,
                        help="stage0 rknn path; _segN is generated automatically")
    parser.add_argument("--weight_path", type=str, default=None,
                        help="stage0 weight path; default derives from rknn_path")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH,
                        help="HuggingFace tokenizer path")
    parser.add_argument("--embed_path", type=str, default=EMBED_PATH,
                        help="embedding bin path (float16)")
    parser.add_argument("--target", type=str, default="rk1820",
                        help="RKNN3 target, e.g. rk1820/rk1828")
    parser.add_argument("--core_mask", type=str, default="0xff",
                        help="NPU core mask in hex")
    parser.add_argument("--stage_count", type=int, default=2,
                        help="number of segmented stages/devices")
    parser.add_argument("--bucket_size", type=int, default=128,
                        help="token bucket size used to account output_callback chunks")
    parser.add_argument("--pipeline_mode", choices=["sequential", "threaded"], default="threaded",
                        help="Python multicard execution mode; sequential is safer for ctypes validation, threaded enables overlap")
    parser.add_argument("--max_context_len", type=int, default=0,
                        help="0 means use queried model max_ctx_len")
    parser.add_argument("--max_new_token", "--max_new_tokens", dest="max_new_token", type=int, default=1024,
                        help="decode loop count; --max_new_tokens is kept for Qwen3.5 example compatibility")
    parser.add_argument("--prompt", type=str, default=None,
                        help="prompt string")
    parser.add_argument("--ignore_eos", action="store_true",
                        help="continue decoding after all configured special EOS ids")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="enable Qwen3.5 thinking mode on stage0 input")
    parser.add_argument("--special_eos_id", action="append", type=int, default=None,
                        help="special EOS id; repeat for multiple ids (Qwen3.5 default falls back to 151645)")
    parser.add_argument("--special_bos_id", action="append", type=int, default=None,
                        help="special BOS id; repeat for multiple ids (Qwen3.5 default falls back to 151643)")
    parser.add_argument("--presence_penalty", type=float, default=1.5,
                        help="sampling presence penalty; Qwen3.5 example uses 1.5")
    parser.add_argument("--kvcache_policy", choices=["default", "normal", "recurrent"], default="default",
                        help="KV-cache policy. For Qwen3.5 default is recommended unless you intentionally override it")
    parser.add_argument("--device_id", action="append", default=[],
                        help="device id for each stage; repeat this option in stage order")
    parser.add_argument("--decrypt_key_path", type=str, default=None,
                        help="decrypt key/license file path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"rknn_multicard_qwen35 {SCRIPT_VERSION}")

    if args.stage_count < 1:
        raise ValueError(f"stage_count must be >= 1, got {args.stage_count}")
    if args.bucket_size <= 0:
        raise ValueError(f"bucket_size must be > 0, got {args.bucket_size}")

    core_mask = int(args.core_mask, 16)
    prompt = load_prompt(args.prompt)

    # Generate all segment paths.
    base_weight_path = resolve_weight_path(args.rknn_path, args.weight_path)
    model_paths = [replace_seg_suffix(args.rknn_path, i) for i in range(args.stage_count)]
    weight_paths = [replace_seg_suffix(base_weight_path, i) for i in range(args.stage_count)]
    for path in model_paths + weight_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    print("--> Loading tokenizer & embedding")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    special_eos_ids, special_bos_ids = resolve_special_token_ids(
        tokenizer, args.special_eos_id, args.special_bos_id
    )
    print(f"len(tokenizer)={len(tokenizer)}, special_eos_id={special_eos_ids}, special_bos_id={special_bos_ids}")
    embeds_flat = np.memmap(args.embed_path, dtype=np.float16, mode="r")
    print("done")

    # Discover devices using the same RKNN3 runtime wrapper.
    probe = RKNN3Lite(verbose=args.verbose)
    try:
        detected = [normalize_device_id(x) for x in probe.get_devices_id(target=args.target)]
    finally:
        probe.release()

    print(f"found {len(detected)} devices:")
    for i, dev_id in enumerate(detected):
        print(f"  [{i}] id={format_device_id(dev_id)}")

    if args.device_id:
        if len(args.device_id) < args.stage_count:
            raise RuntimeError(
                f"not enough --device_id arguments: need={args.stage_count}, got={len(args.device_id)}"
            )
        device_ids = [normalize_device_id(x) for x in args.device_id[:args.stage_count]]
        missing = [x for x in device_ids if x not in detected]
        if missing:
            raise RuntimeError(
                f"specified device_id not found: {[format_device_id(x) for x in missing]}"
            )
        print("using external device_id list:")
        for dev_id in device_ids:
            print(f"  {format_device_id(dev_id)}")
    else:
        if len(detected) < args.stage_count:
            raise RuntimeError(
                f"auto-detect failed: found={len(detected)} devices, need={args.stage_count}"
            )
        device_ids = detected[:args.stage_count]
        print(f"auto-assigning {args.stage_count} devices")

    pipeline = PipelineState(args.stage_count, args.bucket_size, args.verbose)
    stages = [
        StageRuntime(f"stage{i}", model_paths[i], weight_paths[i])
        for i in range(args.stage_count)
    ]

    try:
        # Initialize stages one-by-one, each stage bound to a different card/device.
        for i, stage in enumerate(stages):
            ok = init_stage(
                stage=stage,
                pipeline=pipeline,
                stage_idx=i,
                device_id=device_ids[i],
                target=args.target,
                core_mask=core_mask,
                max_context_len=args.max_context_len,
                max_new_token=args.max_new_token,
                decrypt_key_path=args.decrypt_key_path,
                ignore_eos=args.ignore_eos,
                special_eos_ids=special_eos_ids,
                special_bos_ids=special_bos_ids,
                presence_penalty=args.presence_penalty,
                kvcache_policy=args.kvcache_policy,
                verbose=args.verbose,
            )
            if not ok:
                raise RuntimeError(f"init {stage.name} failed")

        # Qwen3.5 segmented stages must agree on the LLM-facing dimensions.
        ref = stages[0]
        for stage in stages[1:]:
            if stage.embedding_dim != ref.embedding_dim:
                raise RuntimeError(
                    f"embedding_dim mismatch: {ref.name}={ref.embedding_dim}, "
                    f"{stage.name}={stage.embedding_dim}"
                )
            if stage.vocab_size != ref.vocab_size:
                print(
                    f"WARNING: vocab_size differs across stages: "
                    f"{ref.name}={ref.vocab_size}, {stage.name}={stage.vocab_size}"
                )
            if stage.max_ctx_len != ref.max_ctx_len:
                print(
                    f"WARNING: max_ctx_len differs across stages: "
                    f"{ref.name}={ref.max_ctx_len}, {stage.name}={stage.max_ctx_len}"
                )

        # Embedding shape is determined from stage0's queried vocabulary size,
        # matching the Qwen3.5 reference example.
        vocab_size = stages[0].vocab_size
        if vocab_size <= 0 or embeds_flat.size % vocab_size != 0:
            raise RuntimeError(
                f"invalid embedding file size: elems={embeds_flat.size}, vocab_size={vocab_size}"
            )
        if len(tokenizer) != vocab_size:
            print(
                f"Warning: tokenizer vocab size ({len(tokenizer)}) != "
                f"llm_config.vocab_size ({vocab_size}); use llm_config.vocab_size for embeddings"
            )
        embedding_dim = embeds_flat.size // vocab_size
        embeds_data = embeds_flat.reshape(vocab_size, embedding_dim)
        print(f"embedding file: vocab_size={vocab_size}, embedding_dim={embedding_dim}")
        print(f"pipeline_mode={args.pipeline_mode}")

        if embedding_dim != stages[0].embedding_dim:
            print(
                f"WARNING: embedding.bin dim={embedding_dim}, "
                f"stage0 queried embedding_dim={stages[0].embedding_dim}"
            )

        LAST_RESULT.reset_generation()
        print("\n=== Prefill Pipeline ===")
        print(f"Input: {prompt}")
        print("\nOutput: ", end="", flush=True)

        prefill_start = time.perf_counter()
        ok, prefill_tokens = run_pipeline_once(
            stages, pipeline, prompt=prompt, enable_thinking=args.enable_thinking,
            pipeline_mode=args.pipeline_mode
        )
        prefill_end = time.perf_counter()
        if not ok:
            raise RuntimeError("prefill failed")

        next_token = LAST_RESULT.wait_next_token(timeout=2.0)
        if next_token is None:
            raise RuntimeError(
                "prefill finished but LAST stage did not return a sampled token; "
                "refuse to enter decode with an intermediate-stage token"
            )
        expected_last_stage = len(stages) - 1
        source_stage = LAST_RESULT.get_source_stage()
        if source_stage != expected_last_stage:
            raise RuntimeError(
                f"invalid sampled-token source: stage{source_stage}, expected stage{expected_last_stage}"
            )
        flush_generated_text()
        print(f"\n[pipeline] prefill sampled token={next_token} from stage{source_stage}")

        print("\n=== Decode Loop ===")
        decode_start = time.perf_counter()
        decode_tokens = 0

        # Keep C++ behavior: the token sampled by prefill becomes the first decode input;
        # max_new_token counts subsequent pipeline decode iterations.
        for step in range(args.max_new_token):
            vlog(pipeline, f"[Decode {step + 1}] token={next_token}")
            ok, _ = run_pipeline_once(
                stages, pipeline, input_tokens=[int(next_token)], enable_thinking=args.enable_thinking,
                pipeline_mode=args.pipeline_mode
            )
            if not ok:
                print(f"\ndecode step {step + 1} failed")
                break

            next_token = LAST_RESULT.wait_next_token(timeout=2.0)
            if next_token is None:
                print(f"\ndecode step {step + 1} did not return token from LAST stage result_callback")
                break
            source_stage = LAST_RESULT.get_source_stage()
            if source_stage != len(stages) - 1:
                print(
                    f"\ndecode step {step + 1} got token from stage{source_stage}, "
                    f"expected stage{len(stages) - 1}; stop"
                )
                break
            flush_generated_text()

            decode_tokens += 1
            if not args.ignore_eos and int(next_token) in set(special_eos_ids):
                vlog(pipeline, f"decode step {step + 1} reached EOS token {next_token}")
                break

        decode_end = time.perf_counter()
        print()

        prefill_ms = (prefill_end - prefill_start) * 1000.0
        decode_ms = (decode_end - decode_start) * 1000.0
        printf_perf(prefill_tokens, prefill_ms, decode_tokens, decode_ms)

        # Clear every card/stage KV cache after the test, like the C++ demo.
        for stage in stages:
            try:
                stage.rknn.clear_kvcache(RKNN3KVCacheClearPolicy.RKNN3_KVCACHE_CLEAR_ALL)
            except Exception:
                pass

        print("done")
        return 0

    finally:
        for stage in stages:
            stage.release()


if __name__ == "__main__":
    raise SystemExit(main())