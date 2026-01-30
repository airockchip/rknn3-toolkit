import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
from transformers import AutoTokenizer
import ctypes
import numpy as np
import time
from rknn3lite.api import RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMGetEmbedCallback, LLMTokenizerCallback

RKNN_MODEL = 'Qwen2.5-0.5B-Instruct.rknn'
WEIGHT_MODEL = 'Qwen2.5-0.5B-Instruct.weight'
EMBED_PATH = 'Qwen2.5-0.5B-Instruct.embed.bin'

# RKNN_MODEL = '/data/rknn_Qwen2_5_demo/model/Qwen2.5-3B-Instruct.rknn'
# WEIGHT_MODEL = '/data/rknn_Qwen2_5_demo/model/Qwen2.5-3B-Instruct.weight'
# EMBED_PATH = '/data/rknn_Qwen2_5_demo/model/Qwen2.5-3B-Instruct.embed.bin'

TOKENIZER_PATH = 'Qwen/Qwen2.5-0.5B-Instruct'


VOCAB_SIZE = 151936

ARGS = [{"max_new_tokens":512, 
         "top_k":1, "top_p":0.8, 
         "temperature":0.8, 
         "repeat_penalty":1.1, 
         "vocab_size": VOCAB_SIZE, 
         "special_eos_id": 151645, 
         "max_context_len": 1024,
         "keep_history": 0,
         "max_new_tokens": 1024}
        ]

system_prompt  = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
prompt_prefix  = "<|im_start|>user\n";
prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n";

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

def printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time):
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


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Inference Qwen/Qwen2.5 llm model of RKNN") 
    parser.add_argument("--rknn_path", type=str, help="rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--weight_path", type=str, help="rknn weight path", required=False, default=WEIGHT_MODEL)
    parser.add_argument("--tokenizer_path", type=str, help="huggingface tokenizer path or name", required=False, default=TOKENIZER_PATH)
    parser.add_argument("--embed_path", type=str, help="embed path or name", required=False, default=EMBED_PATH)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    embeds_data = np.fromfile(args.embed_path, dtype=np.float16)
    embeds_data = embeds_data.reshape(VOCAB_SIZE, -1)

    # Create RKNN object
    rknn = RKNN3Lite(llm_mode=True, verbose=True)

    # Load model
    print('--> Loading model')
    ret = rknn.load_rknn(args.rknn_path, args.weight_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Callback
    callback = RKLLMCallback()
    callback.result_callback = LLMResultCallback(result_callback)
    callback.result_userdata = None

    callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    userdata = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(userdata), ctypes.c_void_p)

    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    userdata = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(userdata), ctypes.c_void_p)


    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk1820', core_mask=0xff, llm_args=ARGS, llm_callback=callback)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    ret = rknn.set_chat_template(system_prompt, prompt_prefix, prompt_postfix)
    if ret != 0:
        print('Set chat template failed!')
        exit(ret)

    # LLM Inference
    prompts = ["请解释一下相对论的基本概念？", "你是谁？", "介绍一下LLM模型的工作原理。"]
    for prompt in prompts:
        ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = rknn.session_run(prompt=prompt)
        if ret != 0:
            print('RKNN llm inference failed!')
            exit(ret)
        printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time)
        first_token = None
    print('done')

    rknn.release()

