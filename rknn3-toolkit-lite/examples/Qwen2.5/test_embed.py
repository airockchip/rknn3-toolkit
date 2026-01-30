import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
from transformers import AutoTokenizer
import ctypes
import numpy as np
import time
from rknn3lite.api import RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMGetEmbedCallback, LLMTokenizerCallback

RKNN_MODEL = '/data/rknn_Qwen2_5_demo/model/Qwen2.5-3B-Instruct.rknn'
WEIGHT_MODEL = '/data/rknn_Qwen2_5_demo/model/Qwen2.5-3B-Instruct.weight'
EMBED_PATH = '/data/rknn_Qwen2_5_demo/model/Qwen2.5-3B-Instruct.embed.bin'
TOKENIZER_PATH = 'Qwen/Qwen2.5-0.5B-Instruct'


VOCAB_SIZE = 151936
MAX_CONTEXT_LEN = 1024 

ARGS = [{"max_new_tokens":256, 
        "top_k":1, 
        "top_p":0.9, 
        "temperature":1.0, 
        "repeat_penalty":1.1, 
        "vocab_size": VOCAB_SIZE, 
        "special_eos_id": 151645, # 可从config.json查询 eos_token_id 字段
        "max_context_len": MAX_CONTEXT_LEN},
        ]

tokenizer = None
embeds_data = None
first_token = None

def result_callback(userdata, result_ptr, state):
    global tokenizer, first_token

    # 函数静态变量（首次调用时初始化）
    if not hasattr(result_callback, "accumulated_tokens"):
        result_callback.accumulated_tokens = []
        result_callback.last_output_text = ""

    def decode_safe(tokens):
        """用 tokenizer 解码并移除可能的半截替换字符（�）以确保不打印不完整 UTF-8。"""
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # 遇到替换字符则截到第一个替换字符前（避免打印半截字符）
        return text.split('�', 1)[0] if '�' in text else text

    # ERROR
    if state == 5:
        print("\n\n[错误] 推理过程中发生错误")
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
                print(f"\n[Decode错误: {e}] 剩余tokens: {len(result_callback.accumulated_tokens)}", flush=True)
        result_callback.accumulated_tokens.clear()
        result_callback.last_output_text = ""
        msg = {2: "Finished", 3: "Stop", 4: "Max new token reached"}.get(state, "Unknown")
        print(f"\n\n--------------------{msg}--------------------")
        return 0

    # WAITING
    if state == 1:
        print("\n\nWaiting for UTF-8 encoded character")
        return 0

    # NORMAL（累积并尝试打印安全前缀）
    if state == 0:
        n = result_ptr.contents.num_tokens
        # 取新 tokens（保留原有索引访问方式以兼容 C 结构）
        new_tokens = [result_ptr.contents.token_ids[i] for i in range(n)]
        result_callback.accumulated_tokens.extend(new_tokens)
        if first_token == None:
           first_token = time.perf_counter()
        try:
            safe_text = decode_safe(result_callback.accumulated_tokens)
            new_part = safe_text[len(result_callback.last_output_text):]
            if new_part:
                print(new_part, end="", flush=True)
                result_callback.last_output_text += new_part
        except Exception as e:
            print(f"\n[临时Decode错误: {e}]，等待更多tokens", flush=True)
            return 0

    return 0


def embed_callback(userdata, tokens_ptr, num_tokens, embded, length):
    global embeds_data
    embedding_dim = embeds_data.shape[1]

    expected_len = num_tokens * embedding_dim * np.dtype(np.float16).itemsize
    if length != expected_len:
        print("invalid embded buffer")
        return -1

    dst = np.ctypeslib.as_array(
        ctypes.cast(embded, ctypes.POINTER(ctypes.c_uint16)),
        shape=(num_tokens * embedding_dim,)
    ).view(np.float16)

    tokens = [tokens_ptr[i] for i in range(num_tokens)]

    dst[:] = embeds_data[tokens].ravel()

    return 0

def print_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time):
    print("\n--------------------------------------------------------------------------------------")
    print(" %-12s  %-15s  %-8s  %-23s  %-23s" % 
          ("Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("--------------------------------------------------------------------------------------")

    # Prefill 阶段：从 llm_start_time 到 first_token
    prefill_time_sec = first_token - llm_start_time
    prefill_ms = prefill_time_sec * 1000.0
    prefill_n_tokens = n_prefill_tokens

    if prefill_n_tokens == 0:
        prefill_tpt = 0.0
        prefill_tps = 0.0
    else:
        prefill_tpt = prefill_ms / prefill_n_tokens
        prefill_tps = (prefill_n_tokens * 1000.0) / prefill_ms  # tokens per second

    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
          ("Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps))

    # Decode/Generate 阶段：从 first_token 到 llm_end_time
    decode_time_sec = llm_end_time - first_token
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

def prune_model_img_process(img):
    img = np.float32(img)
    img[0,2,...] = (img[0,2,...] - 104.09)/70.3 
    img[0,1,...] = (img[0,1,...] - 116.74)/66.6 
    img[0,0,...] = (img[0,0,...] - 122.70)/68.5 
    patches = np.concatenate([img, img], axis=1)
    h = img.shape[2]
    w = img.shape[3]
    patches = patches.reshape(1, 2, 3, h // 2 // 14, 2, 14, w // 2 // 14, 2, 14)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    feature = patches.reshape(1 * h // 14 * w // 14, 3 * 2 * 14 * 14)
    return feature


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Inference Qwen/Qwen2.5 3B llm model of RKNN") 
    parser.add_argument("--rknn_llm_path", type=str, help="rknn model path", required=False, default=RKNN_LLM_MODEL)
    parser.add_argument("--tokenizer_path", type=str, help="huggingface tokenizer path or name", required=False, default=TOKENIZER_PATH)
    parser.add_argument("--embed_path", type=str, help="embed path or name", required=False, default=EMBED_PATH)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    embeds_data = np.fromfile(args.embed_path, dtype=np.float16)
    embeds_data = embeds_data.reshape(VOCAB_SIZE, -1)

    # Create RKNN object
    rknn_llm = RKNN3Lite(llm_mode=True, verbose=True)

    print('--> Loading model')
    ret = rknn_llm.load_rknn(args.rknn_llm_path, args.rknn_llm_path.replace(".rknn",".weight"))
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Callback
    callback = RKLLMCallback()
    callback.result_callback = LLMResultCallback(result_callback)
    callback.result_userdata = None
    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    userdata = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(userdata), ctypes.c_void_p)
    
    # Init runtime environment
    ret = rknn_llm.init_runtime(target='rk1820', core_mask=0xff, llm_args=ARGS, llm_callback=callback)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # LLM Inference
    prompts = ["请解释一下相对论的基本概念？", "你是谁？", "介绍一下LLM模型的工作原理。"]
    for prompt in prompts:
        # 使用tokenizer构建消息并生成文本
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 使用tokenizer处理文本并获取图像位置
        inputs = tokenizer(text, return_tensors='np', truncation=True)

        # 获取embed
        embeds = embeds_data[inputs.input_ids]

        # 运行LLM推理,注意目前rknn3的embeds模式只适合普通rope的文本模型，如果是像qwen2.5/3 vl这种mrope会不支持embeds模式，mrope只支持prompt模式
        ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = rknn_llm.session_run(embeds=embeds)
        if ret != 0:
            print('RKNN LLM inference failed!')
            exit(ret)
        print_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time)
        first_token = None
    print('done')

    rknn_llm.release()

