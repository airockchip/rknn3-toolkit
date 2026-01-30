import os
import cv2
import time
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
from transformers import AutoTokenizer
import ctypes
import numpy as np
from rknn3lite.api import RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMTokenizerCallback, LLMGetEmbedCallback, RKNN3Video, Float16, RKNN3AuxTensorWrapper, dump_tensor_attr


RKNN_LLM_MODEL = '/userdata/rknn_Qwen3_VL_demo/model/Qwen3-VL-2B-llm.rknn'
RKNN_VISION_MODEL = '/userdata/rknn_Qwen3_VL_demo/model/Qwen3-VL-2B-vision.rknn'
EMBED_PATH = '/userdata/rknn_Qwen3_VL_demo/model/Qwen3-VL-2B-llm.embed.bin'
TOKENIZER_PATH = './Qwen3VL/qwen3_VL_2B'


VOCAB_SIZE = 151936
MAX_CONTEXT_LEN = 1024 

ARGS = [{"max_new_tokens":256, 
        "top_k":1, 
        "top_p":0.9, 
        "temperature":0.7, 
        "repeat_penalty":1.0, 
        "vocab_size": VOCAB_SIZE, 
        "special_eos_id": 151645, # 可从config.json查询 eos_token_id 字段
        "max_context_len": MAX_CONTEXT_LEN,
        'keep_history': 0,
        'max_new_tokens': 1024}]

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

def printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time, vision_latency):
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

    vision_latency_ms = vision_latency * 1000  # 转为毫秒
    fps = 1.0 / vision_latency if vision_latency > 0 else 0.0
    print(f" Vision latency = {vision_latency_ms:.2f} ms, FPS = {fps:.2f}")

def prune_model_img_process(img1,img2):
    img1 = np.float32(img1)
    img1[0,2,...] = (img1[0,2,...] - 127.5)/127.5 
    img1[0,1,...] = (img1[0,1,...] - 127.5)/127.5 
    img1[0,0,...] = (img1[0,0,...] - 127.5)/127.5 
    img2 = np.float32(img2)
    img2[0,2,...] = (img2[0,2,...] - 127.5)/127.5 
    img2[0,1,...] = (img2[0,1,...] - 127.5)/127.5 
    img2[0,0,...] = (img2[0,0,...] - 127.5)/127.5 
    patches = np.concatenate([img1, img2], axis=1)
    h = img1.shape[2]
    w = img1.shape[3]
    patches = patches.reshape(1, 2, 3, h // 2 // 16, 2, 16, w // 2 // 16, 2, 16)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    feature = patches.reshape(1 * h // 16 * w // 16, 3 * 2 * 16 * 16)
    return feature

def dump_all_tensor_attr(rknn_vision, rknn_llm):
    
    for input_attr in rknn_vision.get_inputs_tensor_attr():
        dump_tensor_attr(input_attr, prefix = "rknn_vision input")
    for output_attr in rknn_vision.get_outputs_tensor_attr():
        dump_tensor_attr(output_attr, prefix = "rknn_vision output")

    for input_attr in rknn_llm.get_inputs_tensor_attr():
        dump_tensor_attr(input_attr, prefix = "rknn_llm input")
    for output_attr in rknn_llm.get_outputs_tensor_attr():
        dump_tensor_attr(output_attr, prefix = "rknn_llm output")


def vision_inference(img1,img2,rknn_vision_lite): # prune版 qwen3vl-2b
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (384, 384)).transpose(2,0,1).reshape(1,3,384, 384)

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (384, 384)).transpose(2,0,1).reshape(1,3,384, 384)

    feature = prune_model_img_process(img1,img2)
    feature = np.float16(feature)
    outputs = rknn_vision.inference(inputs=[feature])
    return outputs





if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Inference Qwen/Qwen2.5VL 3B llm model of RKNN") 
    parser.add_argument("--rknn_llm_path", type=str, help="rknn model path", required=False, default=RKNN_LLM_MODEL)
    parser.add_argument("--rknn_vision_path", type=str, help="rknn model path", required=False, default=RKNN_VISION_MODEL)
    parser.add_argument("--tokenizer_path", type=str, help="huggingface tokenizer path or name", required=False, default=TOKENIZER_PATH)
    parser.add_argument("--embed_path", type=str, help="embed path or name", required=False, default=EMBED_PATH)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    embeds_data = np.fromfile(args.embed_path, dtype=np.float16)
    embeds_data = embeds_data.reshape(VOCAB_SIZE, -1)

    # Create RKNN object
    rknn_vision = RKNN3Lite()
    rknn_llm = RKNN3Lite(llm_mode=True, verbose=True)

    print('--> Loading model')
    ret = rknn_vision.load_rknn(args.rknn_vision_path,args.rknn_vision_path.replace(".rknn",".weight"))
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

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

    callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    userdata = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(userdata), ctypes.c_void_p)

    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    userdata = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(userdata), ctypes.c_void_p)
    
    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn_vision.init_runtime(target='rk1820', core_mask=0xff)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)

    ret = rknn_llm.init_runtime(target='rk1820', core_mask=0xff, llm_args=ARGS, llm_callback=callback)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    dump_all_tensor_attr(rknn_vision, rknn_llm)


    ret = rknn_llm.set_chat_template(system_prompt, prompt_prefix, prompt_postfix)
    if ret != 0:
        print('Set chat template failed!')
        exit(ret)

    video_path = "test.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video.")

    # 获取原始视频的帧率
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        print("Warning: Unable to read FPS. Assuming 30 FPS.")
        orig_fps = 30.0

    features = []
    deepstack_tensors_0 = []
    deepstack_tensors_1 = []
    deepstack_tensors_2 = []

    for i in range(5):
        # 计算目标时间（秒）
        target_time_sec = 2 * i # 第0秒、第1秒、第2秒...        
        target_frame = int(round(target_time_sec * orig_fps))        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame1 = cap.read()

        target_time_sec = 2 * i + 1 # 第0秒、第1秒、第2秒...        
        target_frame = int(round(target_time_sec * orig_fps))        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame2 = cap.read()

        #视频推理目前vision_inference仅支持裁剪版，完整版需要有两个图片接口
        outputs = vision_inference(frame1,frame2,rknn_vision) 
        outputs[0] = np.expand_dims(outputs[0], 0)
        features.append(outputs[0])
        deepstack_tensors_0.append(outputs[1])
        deepstack_tensors_1.append(outputs[2])
        deepstack_tensors_2.append(outputs[3])

    features = np.float16(np.concatenate(features, axis=0))
    deepstack_tensors_0 =  np.float16(np.concatenate(deepstack_tensors_0, axis=0))
    deepstack_tensors_1 =  np.float16(np.concatenate(deepstack_tensors_1, axis=0))
    deepstack_tensors_2 =  np.float16(np.concatenate(deepstack_tensors_2, axis=0))


    prompts = ["<video>请介绍这个rockchip相关的视频"]
    for prompt in prompts:
        output0_attr = rknn_vision.get_outputs_tensor_attr()[0]  # 获取vision输出的attr，用于对齐deepstack的size
        deepstack_aligned_size = output0_attr.aligned_size * features.shape[0]

        inputs = []
        llm_input = RKNN3Video()
        llm_input.video_embed = features.ctypes.data_as(ctypes.POINTER(Float16))
        llm_input.n_frame_per_video = features.shape[0] # 注意qwenvl系列2张图片出一帧feature
        llm_input.n_frame_tokens = features.shape[1]
        llm_input.n_video = 1

        llm_input.video_start = "<|vision_start|>".encode('utf-8')
        llm_input.video_end = "<|vision_end|>".encode('utf-8')
        llm_input.video_content = "<|video_pad|>".encode('utf-8')
        llm_input.frame_width = 384
        llm_input.frame_height = 384
        
        inputs.append(llm_input)

        #deepstack的index为2、3、 4,可以通过rknn3_query查询所有input_attrs定位到deepstack的index
        for i in range(3):
            deepstack_tensor = RKNN3AuxTensorWrapper() 
            deepstack_tensor.index = 2 + i
            if i == 0:
                deepstack_tensor.aux_data = deepstack_tensors_0
            elif i == 1:
                deepstack_tensor.aux_data = deepstack_tensors_1
            else:
                deepstack_tensor.aux_data = deepstack_tensors_2
            deepstack_tensor.align_size = deepstack_aligned_size
            inputs.append(deepstack_tensor)

        # 运行LLM推理
        ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = rknn_llm.session_run(inputs=inputs, prompt=prompt)
        if ret != 0:
            print('RKNN LLM inference failed!')
            exit(ret)
        first_token = None
    print('done')

    rknn_vision.release()
    rknn_llm.release()

