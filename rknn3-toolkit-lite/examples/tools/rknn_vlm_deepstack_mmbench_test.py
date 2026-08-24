#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RKNN VLM Deepstack MMBench Test Script

This script evaluates a VLM (Vision-Language Model) with deepstack auxiliary tensors
on the MMBench dataset. The vision model produces multiple outputs: the main image
embedding and additional deepstack tensors that are passed to the LLM as auxiliary inputs.

Usage:
    python rknn_vlm_deepstack_mmbench_test.py \
        --rknn_llm_path <llm_model.rknn> \
        --rknn_vision_path <vision_model.rknn> \
        --tokenizer_path <tokenizer_dir> \
        --embed_path <embed.bin> \
        --mmbench_json <mmbench_dev_en.json> \
        --output_jsonl <results_en.jsonl> \
        --vocab_size 151936 \
        --max_context_len 1024 \
        --special_eos_id 151645 \
        --vision_core_mask 0xff \
        --llm_core_mask 0xff
"""

import os
import sys
import json
import cv2
import time
import ctypes
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
from transformers import AutoTokenizer

from rknn3lite.api import (
    RKNN3Lite,
    RKLLMCallback,
    LLMResultCallback,
    LLMTokenizerCallback,
    LLMGetEmbedCallback,
    RKNN3Image,
    RKNN3AuxTensorWrapper,
    Float16,
    dump_tensor_attr,
)

# ============================================= Global State =============================================

tokenizer = None
embeds_data = None
first_token = None
model_answer = ""  # 累积模型输出的完整文本（类似 C++ model_answer）

# ============================================= Chat Template =============================================

system_prompt  = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
prompt_prefix  = "<|im_start|>user\n"
prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"

# ============================================= Callback Functions =============================================

def result_callback(userdata, result_ptr, state):
    """LLM 结果回调：累积 token 并解码为文本"""
    global tokenizer, first_token, model_answer

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
                    model_answer += new_part
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
                model_answer += new_part
        except Exception as e:
            print(f"\n[Temp decode error: {e}], waiting for more tokens", flush=True)
            return 0

    return 0


def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    """Tokenizer 回调：将文本编码为 token"""
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
    """Embedding 回调：根据 token ID 查表获取 embedding"""
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

# ============================================= Utility Functions =============================================

def printf_perf(first_token_time, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time, vision_latency):
    """打印性能指标"""
    print("\n--------------------------------------------------------------------------------------")
    print(" %-12s  %-15s  %-8s  %-23s  %-23s" %
          ("Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("--------------------------------------------------------------------------------------")

    # Prefill 阶段
    prefill_time_sec = first_token_time - llm_start_time
    prefill_ms = prefill_time_sec * 1000.0
    prefill_n_tokens = n_prefill_tokens

    if prefill_n_tokens == 0:
        prefill_tpt = 0.0
        prefill_tps = 0.0
    else:
        prefill_tpt = prefill_ms / prefill_n_tokens
        prefill_tps = (prefill_n_tokens * 1000.0) / prefill_ms

    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
          ("Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps))

    # Decode/Generate 阶段
    decode_time_sec = llm_end_time - first_token_time
    decode_ms = decode_time_sec * 1000.0
    decode_n_tokens = n_decode_tokens

    if decode_n_tokens == 0:
        decode_tpt = 0.0
        decode_tps = 0.0
    else:
        decode_tpt = decode_ms / decode_n_tokens
        decode_tps = (decode_n_tokens * 1000.0) / decode_ms

    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
          ("Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps))

    print("--------------------------------------------------------------------------------------")

    vision_latency_ms = vision_latency * 1000.0
    fps = 1.0 / vision_latency if vision_latency > 0 else 0.0
    print(f" Vision latency = {vision_latency_ms:.2f} ms, FPS = {fps:.2f}")


def prune_model_img_process(img):
    """裁剪版视觉模型的图像预处理（patch_size=16）"""
    img = np.float32(img)
    img[0, 2, ...] = (img[0, 2, ...] - 127.5) / 127.5
    img[0, 1, ...] = (img[0, 1, ...] - 127.5) / 127.5
    img[0, 0, ...] = (img[0, 0, ...] - 127.5) / 127.5
    patches = np.concatenate([img, img], axis=1)
    h = img.shape[2]
    w = img.shape[3]
    patches = patches.reshape(1, 2, 3, h // 2 // 16, 2, 16, w // 2 // 16, 2, 16)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    feature = patches.reshape(1 * h // 16 * w // 16, 3 * 2 * 16 * 16)
    return feature


def dump_all_tensor_attr(rknn_vision, rknn_llm):
    """打印视觉模型和 LLM 模型的张量属性"""
    for input_attr in rknn_vision.get_inputs_tensor_attr():
        dump_tensor_attr(input_attr, prefix="rknn_vision input")
    for output_attr in rknn_vision.get_outputs_tensor_attr():
        dump_tensor_attr(output_attr, prefix="rknn_vision output")

    for input_attr in rknn_llm.get_inputs_tensor_attr():
        dump_tensor_attr(input_attr, prefix="rknn_llm input")
    for output_attr in rknn_llm.get_outputs_tensor_attr():
        dump_tensor_attr(output_attr, prefix="rknn_llm output")


def get_image_index(img_path):
    """从图片路径中提取文件名（不含扩展名）作为索引"""
    basename = os.path.basename(img_path)
    stem, _ = os.path.splitext(basename)
    return stem

# ============================================= Main =============================================

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="RKNN VLM Deepstack MMBench Test - Batch evaluation on MMBench dataset with deepstack auxiliary tensors")
    parser.add_argument("--rknn_llm_path", type=str, required=True, help="Path to RKNN LLM model (.rknn)")
    parser.add_argument("--rknn_vision_path", type=str, required=True, help="Path to RKNN vision model (.rknn)")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to HuggingFace tokenizer directory")
    parser.add_argument("--embed_path", type=str, required=True, help="Path to embedding binary file (.embed.bin)")
    parser.add_argument("--mmbench_json", type=str, required=True, help="Path to MMBench JSON file (e.g. mmbench_dev_en.json)")
    parser.add_argument("--output_jsonl", type=str, default="results_en.jsonl", help="Output JSONL file path")
    parser.add_argument("--vocab_size", type=int, default=151936, help="Vocabulary size (from config.json)")
    parser.add_argument("--max_context_len", type=int, default=1024, help="Maximum context length")
    parser.add_argument("--special_eos_id", type=int, default=151645, help="Special EOS token ID (from config.json)")
    parser.add_argument("--vision_core_mask", type=str, default="0xff", help="Vision model NPU core mask (hex)")
    parser.add_argument("--llm_core_mask", type=str, default="0xff", help="LLM model NPU core mask (hex)")
    parser.add_argument("--image_size", type=int, default=384, help="Image resize dimension (width=height)")
    parser.add_argument("--n_deepstack", type=int, default=3, help="Number of deepstack auxiliary tensors from vision model")
    parser.add_argument("--deepstack_start_index", type=int, default=2, help="Starting LLM input index for deepstack tensors")
    args = parser.parse_args()

    vision_core_mask = int(args.vision_core_mask, 16)
    llm_core_mask = int(args.llm_core_mask, 16)

    VOCAB_SIZE = args.vocab_size
    MAX_CONTEXT_LEN = args.max_context_len

    ARGS = [{"max_new_tokens": 1024,
             "top_k": 1,
             "top_p": 0.9,
             "temperature": 1.0,
             "repeat_penalty": 1.2,
             "vocab_size": VOCAB_SIZE,
             "special_eos_id": args.special_eos_id,
             "max_context_len": MAX_CONTEXT_LEN,
             "keep_history": 0}]

    # ---------------------- Load Tokenizer & Embedding ----------------------
    print('--> Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print('done')

    print('--> Loading embedding')
    embeds_data = np.fromfile(args.embed_path, dtype=np.float16)
    embeds_data = embeds_data.reshape(VOCAB_SIZE, -1)
    print(f'done, embedding shape: {embeds_data.shape}')

    # ---------------------- Load MMBench Data ----------------------
    print(f'--> Loading MMBench data from {args.mmbench_json}')
    with open(args.mmbench_json, 'r', encoding='utf-8') as f:
        mmbench_data = json.load(f)
    print(f'done, total items: {len(mmbench_data)}')

    # ---------------------- Create RKNN Objects ----------------------
    rknn_vision = RKNN3Lite()
    rknn_llm = RKNN3Lite(llm_mode=True, verbose=True)

    # ---------------------- Load Models ----------------------
    print('--> Loading vision model')
    ret = rknn_vision.load_rknn(args.rknn_vision_path,
                                 args.rknn_vision_path.replace(".rknn", ".weight"))
    if ret != 0:
        print('Load vision model failed!')
        exit(ret)
    print('done')

    print('--> Loading LLM model')
    ret = rknn_llm.load_rknn(args.rknn_llm_path,
                              args.rknn_llm_path.replace(".rknn", ".weight"))
    if ret != 0:
        print('Load LLM model failed!')
        exit(ret)
    print('done')

    # ---------------------- Setup Callbacks ----------------------
    callback = RKLLMCallback()
    callback.result_callback = LLMResultCallback(result_callback)
    callback.result_userdata = None

    callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    userdata_tok = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(userdata_tok), ctypes.c_void_p)

    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    userdata_emb = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(userdata_emb), ctypes.c_void_p)

    # ---------------------- Init Runtime ----------------------
    print('--> Init runtime environment')
    ret = rknn_vision.init_runtime(target='rk1820', core_mask=vision_core_mask)
    if ret != 0:
        print('Init vision runtime environment failed!')
        exit(ret)

    ret = rknn_llm.init_runtime(target='rk1820', core_mask=llm_core_mask,
                                 llm_args=ARGS, llm_callback=callback)
    if ret != 0:
        print('Init LLM runtime environment failed!')
        exit(ret)
    print('done')

    dump_all_tensor_attr(rknn_vision, rknn_llm)

    ret = rknn_llm.set_chat_template(system_prompt, prompt_prefix, prompt_postfix)
    if ret != 0:
        print('Set chat template failed!')
        exit(ret)

    # ---------------------- 判断视觉模型是完整版还是裁剪版 ----------------------
    vision_input_attr = rknn_vision.get_inputs_tensor_attr()[0]
    is_full_vision_model = (vision_input_attr.n_dims == 4)
    img_size = args.image_size

    # ---------------------- 获取 vision 输出的 aligned_size，用于 deepstack 对齐 ----------------------
    output0_attr = rknn_vision.get_outputs_tensor_attr()[0]
    deepstack_aligned_size = output0_attr.aligned_size

    # ---------------------- Batch Inference ----------------------
    print(f'\n{"="*80}')
    print(f' Starting MMBench evaluation (deepstack): {len(mmbench_data)} items')
    print(f'{"="*80}\n')

    with open(args.output_jsonl, 'w', encoding='utf-8') as ofs:
        for idx, item in enumerate(mmbench_data):
            prompt_text = item["prompt"]
            img_path = item["image"]
            index_name = get_image_index(img_path)

            print(f'\n[{idx + 1}/{len(mmbench_data)}] index={index_name}, image={img_path}')

            # ---------- Read & Preprocess Image ----------
            ori_img = cv2.imread(img_path)
            if ori_img is None:
                print(f'Failed to read image: {img_path}, skipping...')
                obj = {"index": index_name, "prediction": ""}
                ofs.write(json.dumps(obj, ensure_ascii=False) + '\n')
                continue

            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))

            if is_full_vision_model:
                # 完整版：直接 reshape 为 (1, H, W, 3)
                feature = np.float16(img.reshape(1, img_size, img_size, 3))
            else:
                # 裁剪版：需要额外的图像预处理
                feature = prune_model_img_process(
                    img.transpose(2, 0, 1).reshape(1, 3, img_size, img_size)
                )
                feature = np.float16(feature)

            # ---------- Run Vision Model ----------
            vision_start = time.perf_counter()
            outputs = rknn_vision.inference(inputs=[feature])
            vision_latency = time.perf_counter() - vision_start

            # outputs[0] 是 image embedding，后续 outputs[1:] 是 deepstack 辅助张量
            # 有的模型输出是2维，需要补一个 batch 维度
            outputs[0] = np.float16(np.expand_dims(outputs[0], 0)) if outputs[0].ndim == 2 else np.float16(outputs[0])

            # ---------- Build LLM Multimodal Input (with deepstack) ----------
            inputs = []

            # 1. Image embedding 输入
            llm_input = RKNN3Image()
            llm_input.image_embed = outputs[0].ctypes.data_as(ctypes.POINTER(Float16))
            llm_input.n_image_tokens = outputs[0].shape[1]
            llm_input.n_image = outputs[0].shape[0]
            llm_input.image_width = img_size
            llm_input.image_height = img_size
            llm_input.image_start = "<|vision_start|>".encode('utf-8')
            llm_input.image_end = "<|vision_end|>".encode('utf-8')
            llm_input.image_content = "<|image_pad|>".encode('utf-8')
            inputs.append(llm_input)

            # 2. Deepstack 辅助张量输入
            #    deepstack 的 index 可通过 rknn3_query 查询所有 input_attrs 定位
            #    默认 index 为 2, 3, 4（由 --deepstack_start_index 和 --n_deepstack 控制）
            for i in range(args.n_deepstack):
                deepstack_tensor = RKNN3AuxTensorWrapper()
                deepstack_tensor.index = args.deepstack_start_index + i
                deepstack_tensor.aux_data = outputs[1 + i]
                deepstack_tensor.align_size = deepstack_aligned_size
                inputs.append(deepstack_tensor)

            # ---------- Run LLM Inference ----------
            prompt_with_image = "<image> " + prompt_text

            # 重置全局状态
            model_answer = ""
            first_token = None

            ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = \
                rknn_llm.session_run(inputs=inputs, prompt=prompt_with_image)

            if ret != 0:
                print(f'RKNN LLM inference failed for item {index_name}!')
                obj = {"index": index_name, "prediction": ""}
                ofs.write(json.dumps(obj, ensure_ascii=False) + '\n')
                continue

            if first_token is not None:
                printf_perf(first_token, n_decode_tokens, n_prefill_tokens,
                            llm_start_time, llm_end_time, vision_latency)

            # ---------- Save Result ----------
            try:
                obj = {"index": index_name, "prediction": model_answer}
                ofs.write(json.dumps(obj, ensure_ascii=False) + '\n')
            except Exception as e:
                # 模型回答中可能出现非法字符
                print(f'[Warning] Failed to serialize result for {index_name}: {e}')
                obj = {"index": index_name, "prediction": ""}
                ofs.write(json.dumps(obj, ensure_ascii=False) + '\n')

            ofs.flush()

    print(f'\n{"="*80}')
    print(f' MMBench evaluation (deepstack) finished. Results saved to: {args.output_jsonl}')
    print(f'{"="*80}')

    # ---------------------- Release ----------------------
    rknn_vision.release()
    rknn_llm.release()
