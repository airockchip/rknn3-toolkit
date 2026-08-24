import os
import gc
import cv2
import time
import mmap
import json
import struct
import ctypes
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
from transformers import AutoTokenizer
from transformers.utils.versions import require_version

require_version(
    "transformers>=5.5.0",
    "The remote code requires transformers>=5.5.0, please upgrade: pip install -U transformers"
)

from rknn3lite.api import (
    RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMGetEmbedCallback,
    LLMTokenizerCallback, LLMInputCallback, LLMOutputCallback,
    RKNN3Image, Float16
)
from rknn3lite.api.rknn3_types import RKNN3QueryCmd, RKNN3KVCacheClearPolicy


# ----------------- 默认路径配置 -----------------
RKNN_VISION_MODEL = 'gemma-4-vision.rknn'
WEIGHT_VISION_MODEL = 'gemma-4-vision.weight'

RKNN_LLM_MODEL = 'gemma-4-e2b-it.rknn'
WEIGHT_LLM_MODEL = 'gemma-4-e2b-it.weight'

TOKENIZER_PATH = 'gemma-4-e2b-it'
EMBED_PATH = 'gemma-4-e2b-it.embed.bin'
PER_LAYER_EMBED_PATH = 'gemma-4-e2b-it_per_layer_inputs.embed.bin'
SAFETENSORS_PATH = 'rope_caches.safetensors'
VIDEO_PATH = 'demo.mp4'
MAX_FRAMES = 8  # 最大帧数限制

# 参考 test.py：prompt 只保留用户内容，chat template 由 set_chat_template 统一添加。
system_prompt = ""
prompt_prefix = "<bos><|turn>user\n"
prompt_postfix = "<turn|>\n<|turn>model\n"
DEFAULT_PROMPT = '<video>请用中文描述这个视频。'

# Gemma4 视频逐帧 token 配置：<image> 会展开为 <|image><|image|>...<image|>。
IMAGE_START = b"<|image>"
IMAGE_END = b"<image|>"
IMAGE_CONTENT = b"<|image|>"

# ----------------- 全局变量 -----------------
ROPE_CACHE_NAMES = [
    "rope_cos_cache_0", "rope_sin_cache_0",
    "rope_cos_cache_1", "rope_sin_cache_1"
]

tokenizer = None
embeds_data = None
per_layer_embeds_data = None
first_token = None
rope_mmap = None
rope_file = None
rope_mmap_addr = 0
rope_mmap_base_obj = None
rope_caches = {}
llm_ext_input_indices = None
callback_refs = []

DTYPE_ELEM_SIZE = {
    0: 4,   # FLOAT32
    1: 2,   # FLOAT16
    2: 1,   # INT8
    3: 1,   # UINT8
    4: 2,   # INT16
    5: 2,   # UINT16
    6: 4,   # INT32
    7: 4,   # UINT32
    8: 8,   # INT64
    9: 8,   # UINT64
    10: 1,  # BOOL
    11: 1,  # INT4
    12: 1,  # FLOAT8E4M3FN
    13: 2,  # BFLOAT16
    14: 1,  # FLOAT8E8M0
    15: 1,  # FLOAT4E2M1
}


def get_dtype_elem_size(dtype: int) -> int:
    return DTYPE_ELEM_SIZE.get(int(dtype), 1)


def tensor_name(attr) -> str:
    name = attr.name
    if isinstance(name, bytes):
        return name.split(b'\0', 1)[0].decode('utf-8', errors='ignore')
    if isinstance(name, str):
        return name.split('\0', 1)[0]
    try:
        return ctypes.string_at(name).split(b'\0', 1)[0].decode('utf-8', errors='ignore')
    except Exception:
        return bytes(name).split(b'\0', 1)[0].decode('utf-8', errors='ignore')


def tensor_shape(attr):
    n_dims = int(getattr(attr, 'n_dims', 0))
    if n_dims <= 0:
        return []
    return [int(attr.shape[i]) for i in range(n_dims)]


def tensor_dtype(attr):
    # RKNN3 tensor attr 常见字段是 type；不同版本也可能叫 dtype。
    for field in ('type', 'dtype'):
        if hasattr(attr, field):
            try:
                return int(getattr(attr, field))
            except Exception:
                pass
    return None


# ----------------- 回调函数：基本保持 test.py 的行为 -----------------
def result_callback(userdata, result_ptr, state):
    global tokenizer, first_token
    if not hasattr(result_callback, "token_buffer"):
        result_callback.token_buffer = []

    if state == 5:
        print("\n\nError occurred during inference", flush=True)
        result_callback.token_buffer.clear()
        return 0

    # State 2, 3, 4: FINISH / STOP / MAX_TOKEN
    if state in (2, 3, 4):
        if result_callback.token_buffer:
            try:
                final_text = tokenizer.decode(result_callback.token_buffer, skip_special_tokens=True)
                print(final_text.replace('\ufffd', ''), end="", flush=True)
            except Exception as e:
                print(f"\n[Decode error: {e}]", flush=True)

        result_callback.token_buffer.clear()
        import sys
        sys.stdout.flush()
        return 0

    if state == 1:
        return 0

    if state == 0:
        if first_token is None:
            first_token = time.perf_counter()

        n = result_ptr.contents.num_tokens
        new_tokens = [int(result_ptr.contents.token_ids[i]) for i in range(n)]
        result_callback.token_buffer.extend(new_tokens)

        try:
            text = tokenizer.decode(result_callback.token_buffer, skip_special_tokens=True)
            if '\ufffd' not in text:
                print(text, end="", flush=True)
                result_callback.token_buffer.clear()
        except Exception as e:
            print(f"\n[Temp decode error: {e}], waiting for more tokens", flush=True)

    return 0


def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    if isinstance(text_ptr, (bytes, bytearray)):
        text = text_ptr[:text_len].decode('utf-8', errors='ignore')
    else:
        text = ctypes.string_at(text_ptr, text_len).decode('utf-8', errors='ignore')

    inputs = tokenizer(text, return_tensors='np', truncation=True, add_special_tokens=False)
    tokens = inputs['input_ids'][0][:n_tokens_max]
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
        shape=(num_tokens * embedding_dim,)
    ).view(np.float16)

    tokens = [int(tokens_ptr[i]) for i in range(num_tokens)]
    for token_id in tokens:
        if token_id < 0 or token_id >= embeds_data.shape[0]:
            print(f"invalid token id: {token_id}")
            return -1
    dst[:] = embeds_data[tokens].ravel()

    return 0


def input_callback(userdata, input_tensors, n_input_tensors, param):
    global per_layer_embeds_data, rope_mmap_addr, rope_caches

    p = param.contents if hasattr(param, "contents") else param
    num_tokens = int(p.num_tokens)
    pos = int(p.pos)
    tokens = [int(p.tokens[i]) for i in range(num_tokens)]
    embedding_dim = per_layer_embeds_data.shape[1]

    for i in range(n_input_tensors):
        tensor = input_tensors[i]
        attr = tensor.attr.contents if hasattr(tensor.attr, "contents") else tensor.attr
        mem = tensor.mem.contents if hasattr(tensor.mem, "contents") else tensor.mem

        name = tensor_name(attr)
        addr = mem.virt_addr
        if hasattr(addr, "value"):
            addr = addr.value

        if name in rope_caches:
            cache = rope_caches[name]

            elem_sz = get_dtype_elem_size(cache["dtype"])
            c1 = int(cache["shape"][1])
            c2_bytes = int(cache["shape"][4]) * elem_sz
            src_stride = int(cache["shape"][3]) * c2_bytes
            dst_stride = int(attr.shape[3]) * c2_bytes
            src_base = int(cache["offset"]) + pos * c2_bytes

            # 直接用 mmap 的虚拟地址做地址偏移，避免 rope_mmap[...] 切片产生 bytes 临时拷贝。
            for c in range(c1):
                src_addr = rope_mmap_addr + src_base + c * src_stride
                dst_addr = addr + c * dst_stride
                ctypes.memmove(dst_addr, src_addr, dst_stride)

            continue

        if name != "per_layer_inputs":
            continue

        dst = np.ctypeslib.as_array(
            ctypes.cast(ctypes.c_void_p(addr), ctypes.POINTER(ctypes.c_uint16)),
            shape=(num_tokens * embedding_dim,)
        ).view(np.float16)

        for t, token_id in enumerate(tokens):
            begin = t * embedding_dim
            end = begin + embedding_dim

            # VLM 场景可能出现 image/audio 占位 token，per_layer_inputs 没有对应词表行时填 0，
            # 避免越界；真正的图像 embedding 由 RKNN3Image.image_embed 注入。
            if token_id in (258881, 258880):
                token_id = 0

            if 0 <= token_id < per_layer_embeds_data.shape[0]:
                dst[begin:end] = per_layer_embeds_data[token_id]
            else:
                dst[begin:end] = 0

    return 0


def output_callback(userdata, output_tensors, n_output_tensors, state):
    return 0


def printf_perf(first_token_time, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time, vision_latency, num_frames):
    print("\n--------------------------------------------------------------------------------------")
    print(" %-12s  %-15s  %-8s  %-23s  %-23s" %
          ("Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("--------------------------------------------------------------------------------------")

    if first_token_time is None:
        print(" %-12s  %-15s  %-8d  %-23s  %-23s" %
              ("Prefill", "N/A", n_prefill_tokens, "N/A", "N/A"))
        print(" %-12s  %-15s  %-8d  %-23s  %-23s" %
              ("Generate", "N/A", n_decode_tokens, "N/A", "N/A"))
    else:
        # 和 test.py 一致：这里使用 session_run 返回的 llm_start_time/llm_end_time，单位按秒处理。
        prefill_time_sec = first_token_time - llm_start_time
        prefill_ms = prefill_time_sec * 1000.0
        prefill_n_tokens = n_prefill_tokens

        if prefill_n_tokens == 0 or prefill_ms <= 0:
            prefill_tpt = 0.0
            prefill_tps = 0.0
        else:
            prefill_tpt = prefill_ms / prefill_n_tokens
            prefill_tps = (prefill_n_tokens * 1000.0) / prefill_ms

        print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
              ("Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps))

        decode_time_sec = llm_end_time - first_token_time
        decode_ms = decode_time_sec * 1000.0
        decode_n_tokens = n_decode_tokens

        if decode_n_tokens == 0 or decode_ms <= 0:
            decode_tpt = 0.0
            decode_tps = 0.0
        else:
            decode_tpt = decode_ms / decode_n_tokens
            decode_tps = (decode_n_tokens * 1000.0) / decode_ms

        print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
              ("Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps))

    print("--------------------------------------------------------------------------------------")
    vision_latency_ms = vision_latency * 1000.0
    per_frame_ms = vision_latency_ms / num_frames if num_frames > 0 else 0.0
    per_frame_fps = 1000.0 / per_frame_ms if per_frame_ms > 0 else 0.0
    print(f" Vision latency = {vision_latency_ms:.2f} ms ({num_frames} frames), "
          f"Avg per frame = {per_frame_ms:.2f} ms, Per-frame FPS = {per_frame_fps:.2f}\n")


def prepare_vision_input(ori_img, input_attr):
    shape = tensor_shape(input_attr)
    dtype = tensor_dtype(input_attr)

    if len(shape) == 4 and shape[-1] == 3:          # NHWC: [1,H,W,3]
        img_h, img_w = shape[1], shape[2]
        layout = 'NHWC'
    elif len(shape) == 4 and shape[1] == 3:         # NCHW: [1,3,H,W]
        img_h, img_w = shape[2], shape[3]
        layout = 'NCHW'
    else:
        # Gemma4 vision 常见输入是 [1,384,384,3] UINT8。
        img_h, img_w = 384, 384
        layout = 'NHWC'

    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_w, img_h))

    if layout == 'NCHW':
        feature = np.transpose(img, (2, 0, 1))[None, ...]
    else:
        feature = img.reshape(1, img_h, img_w, 3)

    # 按模型输入 dtype 转换。若 attr 取不到 dtype，默认 UINT8，符合当前 Gemma4 vision 导出日志。
    if dtype == 0:
        feature = feature.astype(np.float32)
    elif dtype == 1:
        feature = feature.astype(np.float16)
    else:
        feature = feature.astype(np.uint8)

    return np.ascontiguousarray(feature), img_w, img_h


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Open video failed: {video_path}")
        exit(-1)

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        print("Warning: Unable to read FPS. Assuming 30 FPS.")
        orig_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / orig_fps if total_frames > 0 else 0.0
    num_frames = max(1, int(np.ceil(duration)))
    if num_frames > MAX_FRAMES:
        num_frames = MAX_FRAMES

    print(f"Video: {total_frames} frames, {duration:.2f}s, FPS={orig_fps:.2f}")
    print(f"Extracting {num_frames} frames (1 frame per second, max {MAX_FRAMES} frames)")

    frames = []
    timestamps = []
    for timestamp in range(num_frames):
        target_frame = int(round(timestamp * orig_fps))
        if total_frames > 0:
            target_frame = min(target_frame, total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Read video frame failed at {timestamp}s")
            continue

        frames.append(frame)
        timestamps.append(timestamp)

    cap.release()

    if len(frames) == 0:
        print(f"No valid frame extracted from video: {video_path}")
        exit(-1)

    return frames, timestamps


def build_video_prompt(prompt, timestamps):
    video_tokens = " ".join(
        f"{timestamp // 60:02d}:{timestamp % 60:02d} <image>"
        for timestamp in timestamps
    )
    return prompt.replace("<video>", video_tokens + " ", 1)


def build_llm_callback(ext_indices):
    global llm_ext_input_indices, callback_refs

    callback_refs.clear()
    callback = RKLLMCallback()

    cb_result = LLMResultCallback(result_callback)
    callback.result_callback = cb_result
    callback.result_userdata = None
    callback_refs.append(cb_result)

    cb_tokenizer = LLMTokenizerCallback(tokenizer_callback)
    tokenizer_userdata = ctypes.py_object(tokenizer)
    callback.tokenizer_callback = cb_tokenizer
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(tokenizer_userdata), ctypes.c_void_p)
    callback_refs.extend([cb_tokenizer, tokenizer_userdata])

    cb_embed = LLMGetEmbedCallback(embed_callback)
    embed_userdata = ctypes.py_object(embeds_data)
    callback.embed_callback = cb_embed
    callback.embed_userdata = ctypes.cast(ctypes.pointer(embed_userdata), ctypes.c_void_p)
    callback_refs.extend([cb_embed, embed_userdata])

    cb_input = LLMInputCallback(input_callback)
    input_userdata = ctypes.py_object(per_layer_embeds_data)
    callback.input_callback = cb_input
    callback.input_userdata = ctypes.cast(ctypes.pointer(input_userdata), ctypes.c_void_p)
    callback_refs.extend([cb_input, input_userdata])

    cb_output = LLMOutputCallback(output_callback)
    output_userdata = ctypes.py_object(embeds_data)
    callback.output_callback = cb_output
    callback.output_userdata = ctypes.cast(ctypes.pointer(output_userdata), ctypes.c_void_p)
    callback_refs.extend([cb_output, output_userdata])

    llm_ext_input_indices = (ctypes.c_int * len(ext_indices))(*ext_indices)
    callback.input_tensors_index = llm_ext_input_indices
    callback.n_input_tensors = len(ext_indices)
    callback_refs.append(llm_ext_input_indices)

    return callback


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Inference Gemma4 Vision+LLM model of RKNN")
    parser.add_argument("--rknn_vision_path", type=str, help="vision rknn model path", required=True, default=RKNN_VISION_MODEL)
    parser.add_argument("--rknn_llm_path", type=str, help="llm rknn model path", required=True, default=RKNN_LLM_MODEL)
    parser.add_argument("--tokenizer_path", type=str, help="huggingface tokenizer path or tokenizer.json", required=True, default=TOKENIZER_PATH)
    parser.add_argument("--embed_path", type=str, help="token embedding path", required=False, default=EMBED_PATH)
    parser.add_argument("--per_layer_embed_path", type=str, help="per_layer_inputs embedding path", required=False, default=PER_LAYER_EMBED_PATH)
    parser.add_argument("--safetensors_path", type=str, help="rope_caches.safetensors path", required=False, default=SAFETENSORS_PATH)
    parser.add_argument("--video_path", type=str, help="input video path", required=False, default=VIDEO_PATH)
    parser.add_argument("--prompt", type=str, help="prompt content, chat template will be added automatically", required=False, default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, help="max new tokens", required=False, default=1024)
    parser.add_argument("--llm_core_mask", type=lambda x: int(x, 0), help="npu core mask, e.g. 0xff", required=False, default=0xff)
    parser.add_argument("--vision_core_mask", type=lambda x: int(x, 0), help="npu core mask, e.g. 0xff", required=False, default=0xff)
    parser.add_argument("--dump_vision_output", type=str, help="optional path to save vision output npy", required=False, default="")
    args = parser.parse_args()

    if LLMInputCallback is None:
        print("Current rknn3lite.api has no LLMInputCallback, please use the RKNN3Lite package that supports Gemma4 input callback")
        exit(-1)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    rknn_vision = RKNN3Lite()
    rknn_llm = RKNN3Lite(llm_mode=True, verbose=True)

    try:
        # ----------------- 加载模型 -----------------
        print('--> Loading Vision model')
        ret = rknn_vision.load_rknn(args.rknn_vision_path, args.rknn_vision_path.replace(".rknn",".weight"))
        if ret != 0:
            print('Load vision model failed!')
            exit(ret)
        print('done')

        print('--> Loading LLM model')
        ret = rknn_llm.load_rknn(args.rknn_llm_path, args.rknn_llm_path.replace(".rknn",".weight"))
        if ret != 0:
            print('Load llm model failed!')
            exit(ret)
        print('done')

        # ----------------- 初始化 LLM runtime 并查询配置 -----------------
        print('--> Init LLM runtime environment')
        ret = rknn_llm.init_runtime(target='rk1820', core_mask=args.llm_core_mask)
        if ret != 0:
            print('Init LLM runtime environment failed!')
            exit(ret)
        print('done')

        print('--> Query LLM model info')
        llm_config = rknn_llm.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
        print("llm_config", llm_config)
        if llm_config is None:
            print('Query RKNN3_QUERY_LLM_CONFIG failed!')
            exit(-1)

        io_num = rknn_llm.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
        if io_num is None:
            print('Query RKNN3_QUERY_IN_OUT_NUM failed!')
            exit(-1)

        print("len(tokenizer)=", len(tokenizer))
        vocab_size = int(getattr(llm_config, 'vocab_size', 0))
        if vocab_size <= 0:
            vocab_size = len(tokenizer)
        if len(tokenizer) != vocab_size:
            print(f"Warning: tokenizer vocab size ({len(tokenizer)}) != llm_config.vocab_size ({vocab_size}), use llm_config.vocab_size for embeddings")

        embeds_data = np.memmap(args.embed_path, dtype=np.float16, mode='r')
        embedding_dim = embeds_data.size // vocab_size
        embeds_data = embeds_data.reshape(vocab_size, embedding_dim)

        per_layer_embeds_data = np.memmap(args.per_layer_embed_path, dtype=np.float16, mode='r')
        per_layer_embedding_dim = per_layer_embeds_data.size // vocab_size
        per_layer_embeds_data = per_layer_embeds_data.reshape(vocab_size, per_layer_embedding_dim)

        print(f"vocab_size={vocab_size}, embedding_dim={embedding_dim}, per_layer_embedding_dim={per_layer_embedding_dim}")
        print(f"max_ctx_len={llm_config.max_ctx_len}, max_position_embeddings={llm_config.max_position_embeddings}")

        # ----------------- 找到 per_layer_inputs / rope cache 外部输入 -----------------
        need_rope_cache = getattr(llm_config, 'rope_cache_host_storage', 0) != 0
        ext_indices = []
        for i in range(io_num.n_input):
            attr = rknn_llm.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_INPUT_ATTR, index=i)
            if attr is None:
                print(f'Query RKNN3_QUERY_INPUT_ATTR failed! index={i}')
                exit(-1)

            name = tensor_name(attr)
            if name == "per_layer_inputs":
                ext_indices.append(i)
            elif need_rope_cache and ("rope_cos_cache" in name or "rope_sin_cache" in name):
                ext_indices.append(i)

        if len(ext_indices) == 0:
            print('no ext input tensors found: per_layer_inputs or rope cache')
            exit(-1)

        # ----------------- 加载 Rope cache -----------------
        if need_rope_cache:
            if not os.path.exists(args.safetensors_path):
                print('model requires rope_caches.safetensors, but safetensors_path is invalid')
                exit(-1)

            rope_file = open(args.safetensors_path, 'rb')
            # ACCESS_COPY 便于 ctypes.from_buffer 拿到底层地址；不写入文件，也不会把整文件提前复制到 Python 堆。
            rope_mmap = mmap.mmap(rope_file.fileno(), 0, access=mmap.ACCESS_COPY)
            rope_mmap_base_obj = ctypes.c_char.from_buffer(rope_mmap)
            rope_mmap_addr = ctypes.addressof(rope_mmap_base_obj)

            header_size = struct.unpack('<Q', rope_mmap[:8])[0]
            header = json.loads(rope_mmap[8:8 + header_size].decode('utf-8'))
            meta_index = json.loads(header['__metadata__']['index'])
            data_base = 8 + header_size

            for name in ROPE_CACHE_NAMES:
                meta_t = meta_index[name]
                t = header[name]
                shape = t['shape']
                offsets = t['data_offsets']
                if len(shape) != 5:
                    print(f"Tensor {name}: expected 5-D NC1HWC2")
                    exit(-1)
                rope_caches[name] = {
                    "dtype": int(meta_t['dtype']),
                    "layout": int(meta_t['layout']),
                    "shape": shape,
                    "offset": data_base + int(offsets[0])
                }
                print("Loaded %-24s  dtype=%-2d  shape=%s" % (name, rope_caches[name]["dtype"], shape))

        special_eos_id = [1, 50, 106]  # Gemma4 拥有多个特殊 EOS，可从 tokenizer_config.json 查询到。
        special_bos_id = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 2]

        LLM_ARGS = [{"max_new_tokens": args.max_new_tokens,
                     "top_k": 1, "top_p": 0.9,
                     "temperature": 1.0,
                     "repeat_penalty": 1.0,
                     "frequency_penalty": 0.0,
                     "presence_penalty": 0.0,
                     "vocab_size": vocab_size,
                     "special_eos_id": special_eos_id,
                     "special_bos_id": special_bos_id,
                     "max_context_len": llm_config.max_ctx_len,
                     "keep_history": 0,
                     "logits_name": b"logits_gathered"}
                    ]

        print("\n=============================================================")
        print("%-32s: %-8d" % ("Max Context Length", getattr(llm_config, 'max_ctx_len', 0)))
        print("%-32s: %-8d" % ("Max Position Embeddings", getattr(llm_config, 'max_position_embeddings', 0)))
        print("%-32s: %s" % ("Model Type", getattr(llm_config, 'model_type', b'')))
        print("%-32s: %-8d" % ("Max New Tokens", args.max_new_tokens))
        print("=============================================================\n")

        callback = build_llm_callback(ext_indices)

        print('--> Init LLM session')
        ret = rknn_llm.init_llm_session(llm_args=LLM_ARGS, llm_callback=callback)
        if ret != 0:
            print('Init llm session failed!')
            exit(ret)
        print('done')

        ret = rknn_llm.set_chat_template(system_prompt, prompt_prefix, prompt_postfix)
        if ret != 0:
            print('Set chat template failed!')
            exit(ret)

        # ----------------- 初始化 Vision runtime -----------------
        print('--> Init Vision runtime environment')
        ret = rknn_vision.init_runtime(target='rk1820', core_mask=args.vision_core_mask)
        if ret != 0:
            print('Init Vision runtime environment failed!')
            exit(ret)
        print('done')

        # ----------------- 视频抽帧、图像预处理与 Vision 推理 -----------------
        vis_input_attr = rknn_vision.get_inputs_tensor_attr()[0]
        frames, timestamps = read_video_frames(args.video_path)

        print('--> Running Vision model')
        vision_start = time.perf_counter()
        features = []
        for frame in frames:
            feature, img_w, img_h = prepare_vision_input(frame, vis_input_attr)
            outputs = rknn_vision.inference(inputs=[feature])[0]

            # 每张视频帧输出通常为 [64,1536]，补成 [1,64,1536] 后沿帧维拼接。
            if len(outputs.shape) == 2:
                outputs = np.expand_dims(outputs, 0)
            features.append(outputs)

        outputs = np.float16(np.concatenate(features, axis=0))
        outputs = np.ascontiguousarray(outputs)
        vision_latency = time.perf_counter() - vision_start

        if args.dump_vision_output:
            np.save(args.dump_vision_output, outputs)

        # ----------------- 组装 LLM 多模态输入 -----------------
        llm_input = RKNN3Image()
        llm_input.image_embed = outputs.ctypes.data_as(ctypes.POINTER(Float16))
        llm_input.n_image_tokens = outputs.shape[1]
        llm_input.n_image = outputs.shape[0]
        llm_input.image_width = img_w
        llm_input.image_height = img_h
        llm_input.image_start = IMAGE_START
        llm_input.image_end = IMAGE_END
        llm_input.image_content = IMAGE_CONTENT

        prompt = build_video_prompt(args.prompt, timestamps)
        print("Video prompt:\n" + prompt)

        # ----------------- 运行 LLM 推理 -----------------
        print('--> inference Gemma4 Vision+LLM')
        first_token = None
        ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = rknn_llm.session_run(
            inputs=[llm_input],
            prompt=prompt,
            keep_history=0,
            max_new_tokens=args.max_new_tokens
        )
        if ret != 0:
            print('RKNN Gemma4 Vision+LLM inference failed!')
            exit(ret)

        ret = rknn_llm.clear_kvcache(RKNN3KVCacheClearPolicy.RKNN3_KVCACHE_CLEAR_ALL)
        if ret != 0:
            print(f'Clear kvcache failed! ret={ret}')

        printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time, vision_latency, len(frames))
        print('done')

    finally:
        try:
            rknn_vision.release()
        except Exception:
            pass
        try:
            rknn_llm.release()
        except Exception:
            pass

        rope_caches.clear()
        rope_mmap_addr = 0

        if rope_mmap_base_obj is not None:
            del rope_mmap_base_obj
            rope_mmap_base_obj = None

        gc.collect()

        if rope_mmap is not None:
            rope_mmap.close()
            rope_mmap = None

        if rope_file is not None:
            rope_file.close()
            rope_file = None