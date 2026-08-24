[简体中文](README_CN.md) | [English](README.md)

# RKLLM3 HTTP Server

rkllm3-server是基于RKNPU3实现的一套基本的LLM Server。

本项目基于 [llama.cpp](https://github.com/ggml-org/llama.cpp)。

**功能:**
 * 基于RKNPU3的LLM推理
 * OpenAI API 兼容的对话模板

## 使用方法

**通用参数​**

| 参数 | 说明 |
| -------- | ----------- |
| `-a, --alias STRING` | 设置模型别名（供REST API使用） |
| `-c, --ctx-size N` | 提示词上下文大小（默认：4096，0 = 从模型加载） |
| `-n, --predict, --n-predict N` | 要预测的token数量（默认：-1，-1 = 上下文大小） |
| `--stage-count N` | 模型分段数（多卡推理下需设置） |
| `-m, --model FNAME` | LLM模型路径（多卡推理下设置为首段模型路径，命名规范为xxx_seg0.rknn） |
| `--weight FNAME` | LLM权重路径（可不设置，如有设置，在多卡推理下需设置为首段权重路径，命名规范为xxx_seg0.weight） |
| `--model2 FNAME` | 多模态模型时的vision模型路径|
| `--weight2 FNAME` | 多模态模型时的vision权重路径（可不设置） |
| `--model3 FNAME` | 多模态模型时的audio模型路径|
| `--weight3 FNAME` | 多模态模型时的audio权重路径（可不设置） |
| `--core-mask MASK` | LLM模型执行的NPU核心掩码，支持十六进制或十进制。每一位控制一个NPU核心（例如 `0x01` = 仅core 0, `0x03` = core 0+1, `0xff` = 全部核心）。默认：自动（使用全部可用核心） |
| `--core-mask2 MASK` | vision模型（model2）执行的NPU核心掩码。默认：自动（使用全部可用核心） |
| `--core-mask3 MASK` | audio模型（model3）执行的NPU核心掩码。默认：自动（使用全部可用核心） |
| `--vocab FNAME` | 词汇表路径 |
| `--embed FNAME` | embed的bin文件路径 |
| `--embed-ple FNAME` | gemma4模型的per_layer_inputs embed的bin文件路径 |
| `--rope-tensor FNAME` | gemma4模型的rope cache的safetensors文件路径 |
| `--mel-filter FNAME` | mel filters路径（用于audio模型） |
| `--lora-weight FNAME` | LLM lora权重路径。单个文件可包含多个适配器，所有会话共享；可通过 `/lora-adapters` API 为各会话独立设置缩放比例 |
| `--decrypt-key, --key FNAME` | RSA 加密的 LLM 模型用户密钥包路径（加载加密模型时必须配置） |
| `--decrypt-key2, --key2 FNAME` | RSA 加密的 Vision 模型用户密钥包路径（针对加密的多模态视觉模型） |
| `--decrypt-key3, --key3 FNAME` | RSA 加密的 Audio 模型用户密钥包路径（针对加密的多模态语音模型） |
| `--img-start STRING` | 多模态模型的图像输入前缀 |
| `--img-end STRING` | 多模态模型的图像输入后缀 |
| `--img-content STRING` | 多模态模型的图像输入的占位符 |
| `--audio-start STRING` | 多模态模型的语音输入前缀 |
| `--audio-end STRING` | 多模态模型的语音输入后缀 |
| `--audio-content STRING` | 多模态模型的语音输入的占位符 |
| `--video-start STRING` | qwen3_vl deepstack 视频模式的视频输入前缀 |
| `--video-end STRING` | qwen3_vl deepstack 视频模式的视频输入后缀 |
| `--video-content STRING` | qwen3_vl deepstack 视频模式的视频输入的占位符 |
| `--img-width N` | 多模态模型的输入图像的宽（部分裁减过的模型需要设置, 如qwen3_vl）|
| `--img-height N` | 多模态模型的输入图像的高（部分裁减过的模型需要设置, 如qwen3_vl）|
| `--chat-template-file JINJA_TEMPLATE_FILE` | 设置自定义Jinja聊天模板（默认：使用模型元数据中的模板）|
| `--embedding` | 是否是词嵌入模型 |
| `--reasoning-format FORMAT` | 控制是否允许在响应中包含思考标签，以及如何返回该标签，格式可为：<br/>- none：思考内容保留在`message.content`中<br/>- deepseek：将思考内容放入`message.reasoning_content`<br/>（默认：none） |
| `--reasoning [on\|off\|auto]` | 在对话中使用reasoning/thinking ('on', 'off', 或 'auto', 默认: 'auto' (从模板中获取)) |
| `--record-path STRING` | 设置保存用户请求的存放目录（默认为空, 不保存） |
| `--n-session` | session个数, 用于多session的场景 (默认为1) |
| `--bucket-size` | 多卡推理下，每次推理处理的 token 桶大小，通常设为 128 (默认为128) |

**采样参数**

| 参数 | 说明 |
| -------- | ----------- |
| `--temp N` | 温度（默认：0.8）|
| `--top-k N` | top-k采样（默认：40，0 = 禁用） |
| `--top-p N` | top-p采样（默认：0.9，1.0 = 禁用） |
| `--repeat-penalty N` | 惩罚重复token序列（默认：1.0，1.0 = 禁用） |
| `--presence-penalty N` | 重复alpha存在惩罚（默认：0.0，0.0 = 禁用） |
| `--frequency-penalty N` | 重复alpha频率惩罚（默认：0.0，0.0 = 禁用） |

**专用参数**

| 参数 | 说明 |
| -------- | ----------- |
| `-h, --help, --usage` | 打印使用说明并退出 |
| `--host HOST` | 监听IP地址（默认：127.0.0.1） |
| `--port PORT` | 监听端口（默认：8080） |
| `-to, --timeout N` | 服务器读写超时时间（秒）（默认：600） |
| `--device-id STRING` | 设备ID（可不设置，多卡推理下采用"#"分隔多个设备ID） |
| `--log-level N` | 日志等级（默认：0） |

## 快速上手

运行以下命令开启服务进程:

```bash
# LLM模型（标准模式，需要embed文件）
./rkllm3-server -m qwen2.5-3b.rknn --vocab qwen2.5-3b.tokenizer.gguf --embed qwen2.5-3b.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8

# LLM模型（tie-word模式，不需要embed文件）
./rkllm3-server -m Qwen3-VL-2B-llm-tie-word.rknn --weight Qwen3-VL-2B-llm-tie-word.weight --vocab Qwen3-VL-2B-llm_quant.tokenizer.gguf --tie-word-embedding 1 --host 0.0.0.0 --port 8080 -c 4096 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8

./rkllm3-server -m gemma-4.rknn --vocab gemma-4.tokenizer.gguf --embed gemma-4.embed.bin --embed-ple gemma-4_per_layer_inputs.embed.bin --rope-tensor gemma-4.safetensors --host 0.0.0.0 --port 8080 -c -1 --n_predict 2048  --repeat-penalty 1.0 --presence-penalty 0 --frequency-penalty 0 --top-k 1 --top-p 0.8 --temp 0.8

# LLM模型（多卡级联推理模式，以双卡级联推理为例）
./rkllm3-server --stage-count 2 -m Qwen3.5-9B-llm_seg0.rknn --weight Qwen3.5-9B-llm_seg0.weight --vocab Qwen3.5-9B-llm_seg0.gguf --embed Qwen3.5-9B-llm_seg0.embed.bin --host 0.0.0.0 --port 8080 -c 4096 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8

# 多模态模型 (LLM+VISION)
./rkllm3-server -m MiniCPM-3o-llm.rknn --model2 MiniCPM-3o-vision.rknn --vocab MiniCPM-3o-llm.tokenizer.gguf --embed MiniCPM-3o-llm.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<image>" --img-end "</image>" --img-content "<unk>"

./rkllm3-server -m Qwen2.5-VL-3B-llm.rknn --model2 Qwen2.5-VL-3B-vision.rknn --vocab Qwen2.5-VL-3B-llm.tokenizer.gguf --embed Qwen2.5-VL-3B-llm.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --img-width 392 --img-height 392

./rkllm3-server -m Qwen3-VL-2B-llm.rknn --model2 Qwen3-VL-2B-vision.rknn --vocab Qwen3-VL-2B-llm.tokenizer.gguf --embed Qwen3-VL-2B-llm.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --img-width 384 --img-height 384

# 视频模式 (qwen3_vl deepstack)
./rkllm3-server -m Qwen3-VL-2B-llm.rknn --model2 Qwen3-VL-2B-vision.rknn --vocab Qwen3-VL-2B-llm.tokenizer.gguf --embed Qwen3-VL-2B-llm.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --img-width 384 --img-height 384 --video-start "<|vision_start|>" --video-end "<|vision_end|>" --video-content "<|video_pad|>"

./rkllm3-server -m Qwen3.5-4B.rknn --model2 Qwen3.5-4B-vision.rknn --vocab Qwen3.5-4B.tokenizer.gguf --embed Qwen3.5-4B.embed.bin --host 0.0.0.0 --port 8080 -c 0 --n_predict 2048 --repeat-penalty 1.0 --presence-penalty 1.5 --frequency-penalty 0 --top-k 1 --top-p 0.9 --temp 1.0 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --img-width 384 --img-height 384

# 多模态模型 (LLM+AUDIO)
./rkllm3-server -m gemma-4.rknn --model3 gemma-4-audio.rknn --vocab gemma-4.tokenizer.gguf --embed gemma-4.embed.bin --embed-ple gemma-4_per_layer_inputs.embed.bin --rope-tensor gemma-4.safetensors --host 0.0.0.0 --port 8080 -c -1 --n_predict 2048  --repeat-penalty 1.0 --presence-penalty 0 --frequency-penalty 0 --top-k 1 --top-p 0.8 --temp 0.8 --audio-start "<|audio>" --audio-end "<audio|>" --audio-content "<|audio|>"

# 多模态模型 (LLM+VISION+AUDIO)
./rkllm3-server -m Qwen2.5-Omni-3B-llm.rknn --model2 Qwen2.5-Omni-3B-vision.rknn --model3 Qwen2.5-Omni-3B-audio.rknn --vocab Qwen2.5-Omni-3B-llm.tokenizer.gguf --embed Qwen2.5-Omni-3B-llm.embed.bin --mel-filter mel_128_filters.txt --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_bos|>" --img-end "<|vision_eos|>" --img-content "<|IMAGE|>" --audio-start "<|audio_bos|>" --audio-end "<|audio_eos|>" --audio-content "<|AUDIO|>"

# 词嵌入模型
./rkllm3-server -m Qwen3-Embedding-4B.rknn --vocab Qwen3-Embedding-4B.tokenizer.gguf --embed Qwen3-Embedding-4B.embed.bin --embedding

# 加载加密模型 (LLM 纯文本)
./rkllm3-server -m Qwen3-4B.rknn.enc --decrypt-key Qwen3-4B.key --vocab Qwen3-4B.tokenizer.gguf --embed Qwen3-4B.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512 --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8

# 加载加密模型 (LLM+VISION 多模态)
./rkllm3-server -m Qwen3-VL-4B-llm.rknn.enc --model2 Qwen3-VL-4B-vision.rknn.enc --decrypt-key Qwen3-VL-4B-llm.key --decrypt-key2 Qwen3-VL-4B-vision.key --vocab Qwen3-VL-4B-llm.tokenizer.gguf --embed Qwen3-VL-4B-llm.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512 --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --img-width 384 --img-height 384 --video-start "<|vision_start|>" --video-end "<|vision_end|>" --video-content "<|video_pad|>"

# 加载加密模型 (LLM+VISION+AUDIO 全模态)
./rkllm3-server -m Qwen3-Omni-4B-llm.rknn.enc --model2 Qwen3-Omni-4B-vision.rknn.enc --model3 Qwen3-Omni-4B-audio.rknn.enc --decrypt-key Qwen3-Omni-4B-llm.key --decrypt-key2 Qwen3-Omni-4B-vision.key --decrypt-key3 Qwen3-Omni-4B-audio.key --vocab Qwen3-Omni-4B-llm.tokenizer.gguf --embed Qwen3-Omni-4B-llm.embed.bin --mel-filter mel_128_filters.txt --host 0.0.0.0 --port 8080 -c 768 --n_predict 512 --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --audio-start "<|audio_start|>" --audio-end "<|audio_end|>" --audio-content "<|audio_pad|>"

# 同时加载多个模型 (需要确保内存足够加载多个模型)
./rkllm3-server --params-file params.json
```

加载多个模型时，params.json的基本格式如下：
```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "timeout": 30,
  "models": {
    "Qwen2.5-0.5B": {
      "alias": "Qwen2.5-0.5B",
      "model": "Qwen2.5-0.5B-Instruct.rknn",
      "weight": "Qwen2.5-0.5B-Instruct.weight",
      "model2": "",
      "weight2": "",
      "model3": "",
      "weight3": "",
      "core-mask": 0,
      "core-mask2": 0,
      "core-mask3": 0,
      "vocab": "Qwen2.5-0.5B-Instruct.tokenizer.gguf",
      "embed": "Qwen2.5-0.5B-Instruct.embed.bin",
      "device-id": "0001:11:00.0",
      "mel-filter": "",
      "lora-weight": "",
      "decrypt-key": "",
      "decrypt-key2": "",
      "decrypt-key3": "",
      "ctx-size": 1024,
      "predict": 512,
      "temp": 0.8,
      "top-k": 1,
      "top-p": 0.8,
      "repeat-penalty": 1.1,
      "presence-penalty": 1.0,
      "frequency-penalty": 1.0,
      "img-start": "",
      "img-end": "",
      "img-content": "",
      "audio-start": "",
      "audio-end": "",
      "audio-content": "",
      "video-start": "",
      "video-end": "",
      "video-content": "",
      "img-width": 0,
      "img-height": 0,
      "chat-template-file": "",
      "embedding": false,
      "record-path": "",
      "reasoning-format": "none",
      "reasoning": "auto",
      "n-session": 1,
    },
    "Qwen3-0.6B": {
      "alias": "Qwen3-0.6B",
      "model": "Qwen3-0.6B.rknn",
      "weight": "Qwen3-0.6B.weight",
      "model2": "",
      "weight2": "",
      "model3": "",
      "weight3": "",
      "core-mask": 0,
      "core-mask2": 0,
      "core-mask3": 0,
      "vocab": "Qwen3-0.6B.tokenizer.gguf",
      "embed": "Qwen3-0.6B.embed.bin",
      "device-id": "0001:11:00.0",
      "mel-filter": "",
      "lora-weight": "",
      "decrypt-key": "",
      "decrypt-key2": "",
      "decrypt-key3": "",
      "ctx-size": 1024,
      "predict": 512,
      "temp": 0.8,
      "top-k": 1,
      "top-p": 0.8,
      "repeat-penalty": 1.1,
      "presence-penalty": 1.0,
      "frequency-penalty": 1.0,
      "img-start": "",
      "img-end": "",
      "img-content": "",
      "audio-start": "",
      "audio-end": "",
      "audio-content": "",
      "video-start": "",
      "video-end": "",
      "video-content": "",
      "img-width": 0,
      "img-height": 0,
      "chat-template-file": "",
      "embedding": false,
      "record-path": "",
      "reasoning-format": "none",
      "reasoning": "auto",
      "n-session": 1,
    }
  }
}
```

加载多卡推理模型时，params.json的基本格式如下：
```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "timeout": 30,
  "models": {
    "Qwen3.5-9B": {
      "alias": "Qwen3.5-9B",
      "stage-count": 2,
      "model": "Qwen3.5-9B-llm_seg0.rknn",
      "weight": "Qwen3.5-9B-llm_seg0.weight",
      "model2": "",
      "weight2": "",
      "model3": "",
      "weight3": "",
      "vocab": "Qwen3.5-9B-llm.tokenizer.gguf",
      "embed": "Qwen3.5-9B-llm.embed.bin",
      "device-id": ["0001:11:00.0", "0003:31:00.0"],
      "mel-filter": "",
      "lora-weight": "",
      "ctx-size": 4096,
      "predict": 512,
      "temp": 0.8,
      "top-k": 1,
      "top-p": 0.8,
      "repeat-penalty": 1.1,
      "presence-penalty": 1.0,
      "frequency-penalty": 1.0,
      "img-start": "",
      "img-end": "",
      "img-content": "",
      "audio-start": "",
      "audio-end": "",
      "audio-content": "",
      "img-width": 0,
      "img-height": 0,
      "chat-template-file": "",
      "embedding": false,
      "record-path": "",
      "reasoning-format": "none",
      "reasoning": "auto",
      "n-session": 1,
      "bucket-size": 128,
    }
  }
}
```

注: 使用--params-file参数, 将忽略命令行的其余参数, 仅params.json内的参数有效。

另外, RKLLM3 Server除了提供常规的rkllm3-server二进制可执行程序外, 还提供了librkllm3-server.so的使用方式, 方便将server集成到应用中, 用法如下:
```c
#include "rkllm3-server.h"
#include <thread>
#include <stdio.h>

const char *json = R"({
  "host": "127.0.0.1",
  "port": 8080,
  "timeout": 30,
  "models": {
    "Qwen2.5-0.5B": {
      "alias": "Qwen2.5-0.5B",
      "model": "Qwen2.5-0.5B-Instruct.rknn",
      "weight": "Qwen2.5-0.5B-Instruct.weight",
      "model2": "",
      "weight2": "",
      "model3": "",
      "weight3": "",
      "core-mask": 0,
      "core-mask2": 0,
      "core-mask3": 0,
      "vocab": "Qwen2.5-0.5B-Instruct.tokenizer.gguf",
      "embed": "Qwen2.5-0.5B-Instruct.embed.bin",
      "mel-filter": "",
      "lora-weight": "",
      "decrypt-key": "",
      "decrypt-key2": "",
      "decrypt-key3": "",
      "ctx-size": 1024,
      "predict": 512,
      "temp": 0.8,
      "top-k": 1,
      "top-p": 0.8,
      "repeat-penalty": 1.1,
      "presence-penalty": 1.0,
      "frequency-penalty": 1.0,
      "img-start": "",
      "img-end": "",
      "img-content": "",
      "audio-start": "",
      "audio-end": "",
      "audio-content": "",
      "video-start": "",
      "video-end": "",
      "video-content": "",
      "img-width": 0,
      "img-height": 0,
      "chat-template-file": "",
      "embedding": false,
      "record-path": "",
      "reasoning-format": "none",
      "reasoning": "auto",
      "n-session": 1,
    }
  }
})";

// server子线程的状态回调函数
void status_callback(void* userdata, ServerStatus status) {
    switch (status) {
    case SERVER_MODEL_INITED:
        printf("SERVER_MODEL_INITED\n");
        break;
    case SERVER_EXITED:
        printf("SERVER_EXITED\n");
        break;
    case SERVER_ERROR:
        printf("SERVER_ERROR\n");
        break;
    default:
        printf("UNKNOW\n");
        break;
    }
}

int main() {
    StatusCallback callback   = {0};
    callback.status_callback = status_callback;
    callback.status_userdata = NULL;

    int server_result = 0;
    std::thread server_thread([&]() {
        // start_server会一直阻塞直至stop_server被调用,
        // 因此这边要创建子线程来执行start_server
        server_result = start_server(json, &callback);
    });

    printf("Main thread: Waiting 300 seconds ...\n");
    std::this_thread::sleep_for(std::chrono::seconds(60*5));

    printf("Main thread: Stopping server...\n");
    stop_server();

    printf("Main thread: server_thread.join ...\n");
    server_thread.join();

    return 0;
}
```
const char *json字符串内的"model", "weight", "model2", "weight2", "model3", "weight3", "vocab", "embed", "mel-filter", "lora-weight"即可以是文件路径, 也可以是文件的fd句柄 (解决Android系统下文件权限的问题), 如下:
```c
const char *json = R"({
  "host": "127.0.0.1",
  "port": 8080,
  "timeout": 30,
  "models": {
    "Qwen2.5-0.5B": {
      "alias": "Qwen2.5-0.5B",
      "model": "3",
      "weight": "4",
      "model2": "",
      "weight2": "",
      "model3": "",
      "weight3": "",
      "vocab": "5",
      "embed": "6",
      "mel-filter": "",
      "lora-weight": "",
       ...
    }
  }
})";
```

## 用CURL进行测试

使用 [curl](https://curl.se/)。

```sh
curl --request POST \
    --url http://localhost:8080/v1/chat/completions \
    --header "Content-Type: application/json" \
    --data '{"messages": [{"role": "user", "content": "Hello!"}],"n_predict": 128}'
```

## 多Session

`--n-session` 参数用于创建多个独立的会话（Slot），每个会话拥有独立维护的 KV Cache 和对话上下文，支持处理多个用户的对话请求。

当 `--n-session` 大于 1 时，服务端会创建对应数量的 slot，客户端在请求中通过 `id_slot` 参数指定使用哪个 slot（id 从 0 开始，取值范围 `[0, --n-session - 1]`，默认为 0）。

### 启动服务

```bash
# 创建 3 个 session
./rkllm3-server -m qwen2.5-3b.rknn --vocab qwen2.5-3b.tokenizer.gguf --embed qwen2.5-3b.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512 --n-session 3
```

### curl 示例

```bash
# Slot 0 对话
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
  "messages": [{"role": "user", "content": "你好，我叫小明"}],
  "id_slot": 0
}'

# Slot 1 对话（独立上下文，不知道 Slot 0 的对话内容）
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
  "messages": [{"role": "user", "content": "你好，我叫小红"}],
  "id_slot": 1
}'
```

### 使用场景

- **多用户并发服务**：多个用户同时对话，各自独立的 KV Cache，互不干扰
- **对话隔离**：不同任务/话题使用不同 slot，避免上下文混淆

### 各 Slot 独立上下文大小

当使用 `--n-session N`（多 session 模式）时，每个 slot 可以拥有**不同的**上下文大小（ctx-size），满足不同场景对内存和上下文长度的差异化需求。可通过 CLI 的 `#` 分隔符或 `params.json` 的 JSON 数组来分别设置。

**CLI 示例**（3 个 slot，各自不同的 ctx-size）：

```bash
./rkllm3-server -m qwen2.5-3b.rknn --vocab qwen2.5-3b.tokenizer.gguf --embed qwen2.5-3b.embed.bin \
  --host 0.0.0.0 --port 8080 --n-session 3 --n_predict 512 \
  -c 4096#2048#1024
```

**params.json 示例**（相同配置，使用 JSON 数组）：

```json
{
  "host": "0.0.0.0",
  "port": 8080,
  "models": [{
    "model": "./qwen2.5-3b.rknn",
    "vocab": "./qwen2.5-3b.tokenizer.gguf",
    "embed": "./qwen2.5-3b.embed.bin",
    "ctx-size": [4096, 2048, 1024],
    "predict": 512,
    "n-session": 3
  }]
}
```

`"n-session": 3` 时，各 slot 的分配如下：

| Slot | ctx-size | 适用场景 |
|------|----------|----------|
| 0    | 4096     | 长文档/多轮对话 |
| 1    | 2048     | 常规对话 |
| 2    | 1024     | 短问答/资源受限 |

如果数组元素个数少于 `n-session`，剩余 slot 将使用标量默认值。也支持标量值（如 `"ctx-size": 4096`），向后兼容地为所有 slot 设置相同的值。

> **注意**：各 slot 的 ctx-size 会影响 NPU 内存分配，较大的 ctx-size 会消耗更多内存。请根据实际内存情况合理配置。

### 各 Slot 独立预测长度

多 session 模式下，每个 slot 也可以拥有**不同的**预测 token 数（n-predict），满足不同场景对生成长度的差异化需求。同样通过 CLI 的 `#` 分隔符或 `params.json` 的 JSON 数组来分别设置。

**CLI 示例**（3 个 slot，各自不同的 n-predict）：

```bash
./rkllm3-server -m qwen2.5-3b.rknn --vocab qwen2.5-3b.tokenizer.gguf --embed qwen2.5-3b.embed.bin \
  --host 0.0.0.0 --port 8080 --n-session 3 -c 4096 \
  -n 512#256#-1
```

**params.json 示例**（相同配置，使用 JSON 数组）：

```json
{
  "host": "0.0.0.0",
  "port": 8080,
  "models": [{
    "model": "./qwen2.5-3b.rknn",
    "vocab": "./qwen2.5-3b.tokenizer.gguf",
    "embed": "./qwen2.5-3b.embed.bin",
    "ctx-size": 4096,
    "predict": [512, 256, -1],
    "n-session": 3
  }]
}
```

`"n-session": 3` 时，各 slot 的分配如下：

| Slot | n-predict | 含义 |
|------|-----------|------|
| 0    | 512       | 最多生成 512 个 token |
| 1    | 256       | 最多生成 256 个 token |
| 2    | -1        | 设置为上下文的长度 |

如果数组元素个数少于 `n-session`，剩余 slot 将使用标量默认值。也支持标量值（如 `"predict": 512`），向后兼容地为所有 slot 设置相同的值。

> **注意**：每个请求还可以通过 `extra_body` 中的 `n_predict` 字段进一步覆盖当前 slot 的默认值，提供更灵活的控制。

## API 端点

### GET `/health`: 返回健康检查结果

**响应格式**

- HTTP 状态码 503
  - Body: `{"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}`
  - 说明: 模型正在被加载
- HTTP 状态码 200
  - Body: `{"status": "ok" }`
  - 说明: 该模型已成功加载，并且服务器已准备就绪

### GET `/lora-adapters`：获取所有LoRA适配器列表

此端点返回已加载的LoRA适配器列表。您可以在启动服务器时通过 `--lora-weight` 参数添加适配器，例如：`--lora-weight Qwen2.5-3B-Instruct.lora_weight`

单个LoRA权重文件可以包含多个适配器。所有从该文件加载的适配器在所有会话（slot）之间共享，但每个会话可以在运行时独立配置各适配器的缩放比例。

默认情况下，所有适配器的缩放比例（scale）将设置为0。

请注意，此适配器中的scale会被每个用户请求中的 `lora` 字段值覆盖。

如果某个适配器已被禁用，其scale将被设置为0。

**请求格式**
```json
{
    "model": "Qwen2.5-3B-Instruct",
    "id_slot": 0
}
```

`id_slot` 为可选参数，默认值为 `0`，用于指定要查询哪个会话的LoRA配置。当 `--n-session` 大于 1 时，可通过此参数获取指定会话的配置。

**响应格式**

```json
[
  {"name": "lora0_pattern0", "scale": 0.0},
  {"name": "lora1_pattern0", "scale": 0.0}
]
```

### POST `/lora-adapters`：设置LoRA适配器列表

此端点用于为指定会话（slot）设置LoRA适配器的scale。请注意，该scale会被每个请求中的 `lora` 字段值覆盖。

若要禁用某个适配器，可以将其从下方列表中移除，或者将其scale设置为0。

**请求格式**

要获取适配器的 `name`，请使用 GET `/lora-adapters`

```json
{
    "model": "Qwen2.5-3B-Instruct",
    "id_slot": 0,
    "lora": [
        {"name": "lora0_pattern0", "scale": 0.2},
        {"name": "lora1_pattern0", "scale": 0.8}
    ]
}
```

`id_slot` 为可选参数，默认值为 `0`，用于指定要更新哪个会话的LoRA配置。当 `--n-session` 大于 1 时，可通过此参数独立配置各会话。所有会话共享同一组从 `--lora-weight` 加载的适配器，但每个会话可以拥有各自的scale值。

## Slot 暂停/继续

Pause/Resume 允许控制指定 slot 的推理状态。Pause 在**不销毁 KV Cache** 的前提下暂停推理，Resume 从暂停位置恢复。适用于资源调度、多 slot 间动态分配推理时间片等场景。

> **注意**: Pause/Resume 仅在 `--n-session 2`（或更高）时可用

### POST `/slots/{slot_id}/pause`：暂停 slot

暂停指定 slot 的推理，slot 的 KV Cache 将被保留。

**请求格式**

```json
{
    "model": "model_alias"
}
```

`model` 字段为可选，仅多模型加载时需要指定。

**成功响应**（HTTP 200）：

```json
{
    "success": true,
    "id_slot": 0,
    "state": "paused"
}
```

**错误响应**（HTTP 400）：

| 错误信息 | 原因 |
|----------|------|
| `Slot is already paused` | 该 slot 已经处于暂停状态 |
| `Slot is not in a pausable state` | slot 处于 IDLE 或其他不可暂停的状态 |
| `Invalid slot id: N` | slot ID 超出范围 |

### POST `/slots/{slot_id}/resume`：恢复 slot

恢复已暂停 slot 的推理。

**请求格式**

```json
{
    "model": "model_alias"
}
```

`model` 字段为可选，仅多模型加载时需要指定。

**成功响应**（HTTP 200）：

```json
{
    "success": true,
    "id_slot": 0,
    "state": "resumed"
}
```

**错误响应**（HTTP 400）：

| 错误信息 | 原因 |
|----------|------|
| `Slot is not paused, cannot resume` | slot 当前不在暂停状态 |
| `Invalid slot id: N` | slot ID 超出范围 |

## Slot 打断（Interrupt）

Interrupt 允许直接终止指定 slot 的推理并将 slot 重置为 IDLE 状态。与 Pause 不同，Interrupt 会直接终止推理并释放 slot，可通过 `save_kvcache_path` 参数在打断后自动保存 KV Cache（一步完成打断+保存，无需单独调用 save 接口）。结合 KV Cache Save/Load 能力，可实现优先级打断等高级任务调度场景。

### POST `/slots/{slot_id}/interrupt`：打断 slot

打断指定 slot 的推理，将 slot 重置为 IDLE 状态。

**请求格式**

```json
{
    "model": "model_alias",
    "save_kvcache_path": "/path/to/kvcache.kv"
}
```

- `model` 字段为可选，仅多模型加载时需要指定
- `save_kvcache_path` 为可选，指定后会在打断成功后自动保存 KV Cache 和推理状态（状态文件路径自动派生：`xxx.kv` -> `xxx.kv.json`）

**成功响应**（HTTP 200）：

```json
{
    "success": true,
    "id_slot": 0,
    "state": "idle",
    "kvcache_saved": true,
    "kvcache_path": "/path/to/kvcache.kv"
}
```

如果未指定 `save_kvcache_path`，则响应中不包含 `kvcache_saved` 和 `kvcache_path` 字段。

**错误响应**（HTTP 400）：

| 错误信息 | 原因 |
|----------|------|
| `Invalid slot id: N` | slot ID 超出范围 |
| `Slot is not processing, cannot interrupt` | slot 当前不在推理状态 |

*示例:*

```bash
# 打断 slot 0
curl -X POST http://localhost:8080/slots/0/interrupt \
  -H "Content-Type: application/json" \
  -d '{}'

# 打断 slot 0 并自动保存 KV Cache
curl -X POST http://localhost:8080/slots/0/interrupt \
  -H "Content-Type: application/json" \
  -d '{"save_kvcache_path": "/data/kvcache/task.kv"}'
```

### 优先级打断（Priority Interrupt）

优先级打断功能允许在推理过程中打断一个正在执行的低优先级任务，执行高优先级任务，之后恢复低优先级任务继续推理。该功能结合了 Slot Interrupt 和 KV Cache Save/Load 的能力，适用于多任务调度场景。

#### 典型流程

1. **启动低优先级任务**：在 slot 0 上以流式模式启动一个长文本推理任务
2. **打断低优先级任务并保存**：调用 `/slots/0/interrupt` 并指定 `save_kvcache_path`，一步完成打断 + KV Cache 保存（状态文件路径自动从 KV cache 路径派生：`xxx.kv` -> `xxx.kv.json`）
3. **断开连接**：断开低优先级任务的客户端连接，服务器检测到连接关闭后自动清理残留连接
4. **执行高优先级任务**：在 slot 0 上执行高优先级任务，等待其完成
5. **恢复低优先级任务**：重新发送与步骤 1 完全相同的 prompt，并在 `extra_body` 中指定 `load_kvcache_path`，服务器将自动加载 KV cache 和状态（从 cache 路径派生），从打断位置继续推理

#### Python 示例

以下示例展示了完整的优先级打断流程：

```python
import requests
import json

BASE_URL = "http://127.0.0.1:8080"
SLOT_ID = 0

# 步骤 1：启动低优先级任务（长文本，流式模式）
# 使用子进程或独立线程发送请求，以便在收到若干 token 后打断

# 步骤 2：打断 slot 0 并自动保存 KV Cache
# 一步完成打断 + 保存，状态文件路径自动从 KV cache 路径派生：
#   /data/kvcache/task_low_priority.kv -> /data/kvcache/task_low_priority.kv.json
resp = requests.post(
    f"{BASE_URL}/slots/{SLOT_ID}/interrupt",
    json={
        "model": "",
        "save_kvcache_path": "/data/kvcache/task_low_priority.kv"
    },
    timeout=60
)
print(resp.json())  # {"success": true, "id_slot": 0, "state": "idle", "kvcache_saved": true, "kvcache_path": "/data/kvcache/task_low_priority.kv"}

# 步骤 3：断开低优先级任务的客户端连接（kill 子进程 / 关闭 socket）
# 服务器检测到 is_connection_closed() → cancel_tasks() → 清理残留连接

# 步骤 4：执行高优先级任务
resp = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "请用中文写一首关于春天的五言绝句。"}],
        "max_tokens": 256,
        "stream": False,
        "id_slot": SLOT_ID
    },
    timeout=120
)
print(resp.json()["choices"][0]["message"]["content"])

# 步骤 5：恢复低优先级任务
# 使用与步骤 1 完全相同的 prompt，并通过 extra_body 指定加载路径
# 状态文件自动从 KV cache 路径派生：
#   /data/kvcache/task_low_priority.kv -> /data/kvcache/task_low_priority.kv.json
resp = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "请用中文详细写一篇关于人工智能发展历史的综述文章..."}],
        "max_tokens": 4096,
        "stream": True,
        "id_slot": SLOT_ID,
        "extra_body": {
            "load_kvcache_path": "/data/kvcache/task_low_priority.kv"
        }
    },
    stream=True,
    timeout=300
)

# 读取流式输出，将从停止位置继续生成
for line in resp.iter_lines(decode_unicode=True):
    if line and line.startswith("data: "):
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        data = json.loads(data_str)
        content = data["choices"][0]["delta"].get("content", "")
        if content:
            print(content, end="", flush=True)
```

#### 使用场景

- **多任务调度**：当有高优先级请求到达时，打断当前低优先级任务，执行高优先级任务后恢复
- **用户中断**：用户在等待长回复时发送新问题，打断旧任务，处理新问题后再恢复
- **资源管理**：动态分配 NPU 算力，确保关键任务优先完成

## KV Cache 保存/加载

KV Cache 保存/加载允许将指定 slot 的 KV Cache 保存到文件，或从文件加载恢复，实现**对话 KV Cache 持久化**。适用于以下场景：

- **对话续接**：先保存当前对话的 KV Cache，下次启动服务器后加载，无需重新输入历史消息即可续接对话
- **预热加速**：预先构造提示词生成 KV Cache 文件，避免每次冷启动时重复计算前缀

> **注意**: `save_kvcache_path` 和 `load_kvcache_path` 是每个请求的 `extra_body` 参数，两者**不能在同一请求中同时使用**。保存操作在本次生成结束后执行，加载操作在本次生成开始前执行。加载 KV Cache 后，请求中的 `messages` 仍会被处理（可用于追加新消息）。

### 使用方法

#### 保存 KV Cache

在请求中设置 `save_kvcache_path` 参数，生成结束后会自动将 KV Cache 保存到指定路径。请确保目标路径的**父目录已存在**。

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required"
)

# 对话并保存 KV Cache
completion = client.chat.completions.create(
    model="Qwen2.5-3B",
    messages=[
        {"role": "user", "content": "你好，我叫小明，请记住我的名字"}
    ],
    extra_body={"save_kvcache_path": "/data/kvcache/user_xiaoming.kv"}
)
print(completion.choices[0].message.content)
```

#### 加载 KV Cache

在请求中设置 `load_kvcache_path` 参数，生成之前会从指定路径加载 KV Cache。请确保**文件已存在**。

```python
# 加载 KV Cache 后续接对话
completion = client.chat.completions.create(
    model="Qwen2.5-3B",
    messages=[
        {"role": "user", "content": "我叫什么名字？"}
    ],
    extra_body={"load_kvcache_path": "/data/kvcache/user_xiaoming.kv"}
)
print(completion.choices[0].message.content)
# 输出: 你叫小明
```

### 结合多 Session 使用

KV Cache 保存/加载与 `id_slot` 配合使用时，每个 slot 可以独立持久化各自的 KV Cache：

```python
# Slot 0: 保存小明的 KV Cache
client.chat.completions.create(
    model="Qwen2.5-3B",
    messages=[{"role": "user", "content": "你好，我叫小明"}],
    extra_body={"id_slot": 0, "save_kvcache_path": "/data/kvcache/slot0.kv"}
)

# Slot 1: 保存小红的 KV Cache
client.chat.completions.create(
    model="Qwen2.5-3B",
    messages=[{"role": "user", "content": "你好，我叫小红"}],
    extra_body={"id_slot": 1, "save_kvcache_path": "/data/kvcache/slot1.kv"}
)

# 后续分别加载，各自独立 KV Cache
completion = client.chat.completions.create(
    model="Qwen2.5-3B",
    messages=[{"role": "user", "content": "我叫什么名字？"}],
    extra_body={"id_slot": 0, "load_kvcache_path": "/data/kvcache/slot0.kv"}
)
print(completion.choices[0].message.content)  # 输出: 你叫小明
```

### 错误说明

| 错误信息                                                     | 原因                         |
| ------------------------------------------------------------ | ---------------------------- |
| `'save_kvcache_path' and 'load_kvcache_path' cannot both be set in the same request` | 同一请求中同时设置了两个参数 |
| `directory for 'save_kvcache_path' does not exist: ...`      | 保存路径的父目录不存在       |
| `file for 'load_kvcache_path' does not exist: ...`           | 加载的文件不存在             |

### KV Cache 独立 API 接口

除了通过 `extra_body` 参数在 `/v1/chat/completions` 请求中触发保存/加载外，还提供了独立的 REST API 接口，可直接对指定 slot 的 KV Cache 进行保存或加载操作。

> **注意**：这些接口要求目标 slot 处于 **空闲状态**（IDLE），即该 slot 没有正在进行的推理任务。如果 slot 正在处理中，接口将返回错误。

#### POST `/slots/{slot_id}/kvcache/save`：保存指定 slot 的 KV Cache

将指定 slot 的 KV Cache 保存到文件。

**请求格式**

```json
{
    "model": "model_alias",
    "path": "/data/kvcache/backup.kv"
}
```

| 字段 | 说明 |
|------|------|
| `model` | 可选，模型别名或路径。仅多模型加载时需要指定 |
| `path` | **必填**，保存 KV Cache 的文件路径。请确保目标路径的**父目录已存在** |

**成功响应**（HTTP 200）：

```json
{
    "success": true,
    "id_slot": 0,
    "path": "/data/kvcache/backup.kv"
}
```

**错误响应**（HTTP 400 / 500）：

| 错误信息 | 原因 |
|----------|------|
| `'path' is required for kvcache save` | 请求中缺少 `path` 字段 |
| `Invalid slot id: N` | slot ID 超出范围 |
| `Failed to save kvcache for slot` | 保存失败（可能 slot 正在处理中，或路径无效） |

#### POST `/slots/{slot_id}/kvcache/load`：加载 KV Cache 到指定 slot

从文件加载 KV Cache 到指定 slot。

> **注意**：加载操作会**覆盖**目标 slot 当前的 KV Cache。加载后 slot 将继承文件中的 KV Cache 状态，可用于跨 slot 迁移或恢复之前的对话上下文。

**请求格式**

```json
{
    "model": "model_alias",
    "path": "/data/kvcache/backup.kv"
}
```

| 字段 | 说明 |
|------|------|
| `model` | 可选，模型别名或路径。仅多模型加载时需要指定 |
| `path` | **必填**，加载 KV Cache 的文件路径。请确保**文件已存在** |

**成功响应**（HTTP 200）：

```json
{
    "success": true,
    "id_slot": 1,
    "path": "/data/kvcache/backup.kv"
}
```

**错误响应**（HTTP 400 / 500）：

| 错误信息 | 原因 |
|----------|------|
| `'path' is required for kvcache load` | 请求中缺少 `path` 字段 |
| `Invalid slot id: N` | slot ID 超出范围 |
| `Failed to load kvcache for slot` | 加载失败（可能 slot 正在处理中，或文件不存在） |

#### 使用示例

```bash
# 先在 slot 0 上完成一次对话（通过 chat/completions）
# 然后将 slot 0 的 KV Cache 保存到文件
curl -X POST http://localhost:8080/slots/0/kvcache/save \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/kvcache/slot0_backup.kv"}'

# 将保存的 KV Cache 加载到 slot 1（跨 slot 迁移）
curl -X POST http://localhost:8080/slots/1/kvcache/load \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/kvcache/slot0_backup.kv"}'

# 加载后，在 slot 1 上续接对话
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "我叫什么名字？"}],
    "id_slot": 1
  }'
```

#### 典型使用场景

- **对话备份与恢复**：在对话关键节点保存 KV Cache，服务器重启后加载恢复
- **跨 Slot 上下文迁移**：将 slot A 的对话上下文迁移到 slot B，释放 slot A 用于新任务
- **预计算加速**：提前构造提示词在某个 slot 上推理并保存 KV Cache，后续在其他 slot 上加载复用

## KV Cache 检查点（Checkpoint）

KV Cache 检查点功能允许在推理过程中保存 KV Cache 快照（适用于线性注意力或滑动窗口注意力模型，如 Qwen3.5、Gemma4 系列），后续可恢复快照以继续生成。主要用途：

- **多轮对话**：在关键对话节点保存 KV Cache，节省 prefill 时间
- **多 Session 场景**：每个并发 slot 可拥有独立的检查点策略
- **每次请求定制**：每次 API 请求可动态调整检查点参数

> **注意**：检查点参数仅对线性注意力或滑动窗口注意力模型生效（如 Qwen3.5、Gemma4 系列），对其他模型无影响。`checkpoint_start_pos` 和 `checkpoint_interval` 的值会被 NPU 运行时自动对齐到 128。`max_checkpoint_count` 若超过 `(ctx_size - checkpoint_start_pos) / checkpoint_interval`，将被自动 clamp 到此上限。

### 参数说明

| 参数 | CLI | JSON (`params.json`) | 每次请求 (`extra_body`) | 说明 |
|------|-----|----------------------|------------------------|------|
| `checkpoint_start_pos` | `--checkpoint-start-pos N` 或 `N#N#...` | `"checkpoint-start-pos": N` 或 `[N, N, ...]` | `"checkpoint_start_pos": N` | 开始保存检查点的 token 位置（≥ 0）；会自动对齐到 128 |
| `checkpoint_interval` | `--checkpoint-interval N` 或 `N#N#...` | `"checkpoint-interval": N` 或 `[N, N, ...]` | `"checkpoint_interval": N` | 保存检查点的间隔 token 数（0 = 禁用）；会自动对齐到 128 |
| `max_checkpoint_count` | `--max-checkpoint-count N` 或 `N#N#...` | `"max-checkpoint-count": N` 或 `[N, N, ...]` | `"max_checkpoint_count": N` | 从 start_pos 开始最多保存的检查点数量；应 ≤ (ctx_size - start_pos) / interval；超出时自动调整 |
| `checkpoint_tail_overwrite` | `--checkpoint-tail-overwrite 0\|1` 或 `0\|1#...` | `"checkpoint-tail-overwrite": 0\|1` 或 `[0, 1, ...]` | `"checkpoint_tail_overwrite": 0\|1` | 尾部覆盖模式。若为 `1`，则最后一个检查点 checkpoint`max_checkpoint_count`）将作为循环覆盖位，其他检查点保持固定位置不变（默认：`0`） |

以上四个参数默认值均为 `0`，即默认不启用检查点保存。

### 配置方式

检查点参数可以在三个层级设置，优先级从高到低：

1. **每次请求**（最高优先级）：在 `/v1/chat/completions` 请求的 `extra_body` 中设置
2. **服务端 CLI**：`--checkpoint-start-pos`、`--checkpoint-interval`、`--max-checkpoint-count`、`--checkpoint-tail-overwrite`（使用 `#` 分隔符设置每个 slot 的值，如 `--checkpoint-start-pos 128#256#0`）
3. **服务端 params.json**：`"checkpoint-start-pos"`、`"checkpoint-interval"`、`"max-checkpoint-count"`、`"checkpoint-tail-overwrite"`（使用 JSON 数组设置每个 slot 的值）

### 单 Session 示例

启动服务时设置检查点参数：

```bash
./rkllm3-server -m Qwen3.5-4B.rknn --vocab Qwen3.5-4B.tokenizer.gguf --embed Qwen3.5-4B.embed.bin \
  --host 0.0.0.0 --port 8080 -c -1 --n_predict 2048 \
  --checkpoint-start-pos 128 --checkpoint-interval 128 --max-checkpoint-count 4 --checkpoint-tail-overwrite 1
```

或使用 `params.json`：

```json
{
  "host": "0.0.0.0",
  "port": 8080,
  "log-level": 1,
  "models": [{
    "model": "./Qwen3.5-4B.rknn",
    "vocab": "./Qwen3.5-4B.tokenizer.gguf",
    "embed": "./Qwen3.5-4B.embed.bin",
    "ctx-size": -1,
    "predict": 2048,
    "checkpoint-start-pos": 128,
    "checkpoint-interval": 128,
    "max-checkpoint-count": 4,
    "checkpoint-tail-overwrite": 1
  }]
}
```

每次 API 请求可覆盖检查点参数：

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好！"}],
    "extra_body": {
      "checkpoint_start_pos": 1024,
      "checkpoint_interval": 512,
      "max_checkpoint_count": 4,
      "checkpoint_tail_overwrite": 1
    }
  }'
```

如果请求中**未指定**这些检查点参数，slot 已有的 checkpoint 设置将被保留，不会重置为服务端默认值。

### 多 Session + 各 Session 独立 Checkpoint 配置

当使用 `--n-session N`（多 session 模式）时，每个 session 可以拥有**不同的**检查点配置。可通过 CLI 的 `#` 分隔符或 `params.json` 的 JSON 数组来分别设置。

**CLI 示例**（3 个 slot，各自不同的检查点配置）：

```bash
./rkllm3-server -m Qwen3.5-4B.rknn --vocab Qwen3.5-4B.tokenizer.gguf --embed Qwen3.5-4B.embed.bin \
  --host 0.0.0.0 --port 8080 -c -1 --n-session 3 --n_predict 2048 \
  --checkpoint-start-pos 128#256#0 --checkpoint-interval 128#256#0 --max-checkpoint-count 2#3#0 --checkpoint-tail-overwrite 1#0#0
```

**params.json 示例**（相同配置，使用 JSON 数组）：

```json
{
  "host": "0.0.0.0",
  "port": 8080,
  "models": [{
    "model": "./Qwen3.5-4B.rknn",
    "vocab": "./Qwen3.5-4B.tokenizer.gguf",
    "embed": "./Qwen3.5-4B.embed.bin",
    "ctx-size": -1,
    "predict": 2048,
    "n-session": 3,
    "checkpoint-start-pos": [128, 256, 0],
    "checkpoint-interval": [128, 256, 0],
    "max-checkpoint-count": [8, 4, 0],
    "checkpoint-tail-overwrite": [1, 0, 0]
  }]
}
```

`"n-session": 3` 时，各 slot 的分配如下：

| Slot | `start_pos` | `interval` | `max_count` | `tail_overwrite` | 效果 |
|------|-------------|------------|-------------|------------------|------|
| 0    | 128         | 128        | 8         | 1                | 高频检查点，尾部循环覆盖 |
| 1    | 256         | 256        | 4         | 0                | 中等频率检查点 |
| 2    | 0           | 0          | 0           | 0                | 禁用检查点 |

如果数组元素个数少于 `n-session`，剩余 slot 将使用标量默认值（或 `0`）。也支持标量值（如 `"checkpoint-interval": 128`），向后兼容地为所有 slot 设置相同的值。

## OpenAI兼容的API说明

### GET `/v1/models`: OpenAI兼容的模型信息API

返回已加载模型的相关信息。 详见[OpenAI Models API documentation](https://platform.openai.com/docs/api-reference/models).

返回的列表可以有多个元素，对应多个模型。

默认情况下，模型`id`字段是模型文件的路径，通过`-m`指定。您可以通过`--alias`参数为模型`id`字段设置自定义值。例如，`--alias Qwen2.5-3B`。

示例:

```json
{
    "object": "list",
    "data": [
        {
            "id": "Qwen2.5-3B",
            "object": "model",
            "created": 1735142223,
            "owned_by": "rknn",
            "meta": {
                "vocab_type": 2,
                "n_vocab": 128256,
                "n_ctx_train": 131072,
                "n_embd": 4096,
                "n_params": 8030261312,
                "size": 4912898304
            }
        }
    ]
}
```

### POST `/v1/chat/completions`: OpenAI兼容的聊天补全API


给定`messages`中的CHATML形式的JSON描述，返回预测的补全。支持同步和流模式。尽管没有完全实现OpenAI API规格, 但已经足够支持许多应用程序了。只有具有 [聊天模板](https://github.com/ggml-org/llama.cpp/wiki/templates-supported-by-llama_chat_apply_template) 的模型才可以在此端点下较为正常使用。默认情况下，将使用CHATML模板。

*选项:*

详见 [OpenAI Chat Completions API documentation](https://platform.openai.com/docs/api-reference/chat)。 


*示例:*

您可以使用Python的`openai`库：

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)

completion = client.chat.completions.create(
  model="Qwen2.5-3B",
  messages=[
    {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
    {"role": "user", "content": "Write a limerick about python exceptions"}
  ]
)

print(completion.choices[0].message)
```

... 或者原始的HTTP请求:

```shell
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
"messages": [
{
    "role": "system",
    "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
},
{
    "role": "user",
    "content": "Write a limerick about python exceptions"
}
]
}'
```

另外，多模态模型建议使用openai接口，示例如下:
```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key = "sk-no-key-required"
)

# Function to encode the image
def encode_base64(media_path):
    with open(media_path, "rb") as media_file:
        return base64.b64encode(media_file.read()).decode("utf-8")

image_path = "demo.jpg"
audio_path = "demo.wav"     # 如存在音频输入

# Getting the Base64 string
base64_image = encode_base64(image_path)
base64_audio = encode_base64(audio_path)    # 如存在音频输入

completion = client.chat.completions.create(
    model="Qwen2.5-VL",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"{base64_audio}",
                        "format": "wav",
                    }
                },  # 如存在音频输入
                { "type": "text", "text": "请描述一下图片?" },
            ],
        }
    ],
    stream=True,
    extra_body={
        "n_predict": 256,
        "chat_template_kwargs": { "enable_thinking": False }     # Thinking类模型(如Qwen3)可以通过此方式关闭thinking输出
        "id_slot": 0,           # 当--n-session参数大于1时, 可以通过此参数来切换不同的session

        # save_kvcache_path 和 load_kvcache_path 不能同时使用
        "save_kvcache_path": "/data/kvcache/1.kv",       # 保存kvcache到文件中
        "load_kvcache_path": "/data/kvcache/1.kv",       # 加载kvcache文件

        # 采样参数
        "top_k": 40,
        "top_p": 0.9,
        "temperature": 0.8,
        "repeat_penalty": 1.1,
        "frequency_penalty": 1.0,
        "presence_penalty": 1.0,
    }
)

for chunk in completion:
    delta = chunk.choices[0].delta
    if delta.content:
        delta.content = delta.content.replace(f'\n', '<br/>')
        # yield f"data: {delta.content}\n\n"
    if chunk.choices[0].finish_reason == "stop":
        break
    print(delta.content, end='', flush=True)
print('')

```

注: 上述代码里的chunk.choices[0].finish_reason为结束原因，"stop"为正常结束，"length"为达到max_tokens上限被截断，"tool_calls"为触发了工具调用。

`extra_body`支持的选项（非OpenAI兼容）：

- `n_predict`：当前请求要预测的token数量

- `chat_template_kwargs`：当前请求的对话模板参数，如`{ "enable_thinking": False }`用于控制thinking输出

- `id_slot`：指定当前slot的id (当--n-session参数大于1时, 可以通过此参数来切换不同的session, 该值需小于--n-session, 默认为0)

- `save_kvcache_path`：保存kvcache到文件中 (与load_kvcache_path不能同时使用)

- `load_kvcache_path`：加载kvcache文件 (与save_kvcache_path不能同时使用)

- `top_k`：top-k采样

- `top_p`：top-p采样

- `temperature`：温度采样

- `repeat_penalty`：惩罚重复token序列

- `frequency_penalty`：重复alpha频率惩罚

- `presence_penalty`：重复alpha存在惩罚

- `lora`：用于设定LoRA适配器列表。列表中的每个对象必须包含`name`和`scale`字段。例如：`[{"name": "lora0_pattern0", "scale": 0.5}, {"name": "lora1_pattern0", "scale": 0.6}]`。如果某个LoRA适配器未在列表中指定，其scale将默认为`0.0`。

- `tie_word_embedding`：为当前请求启用tie word embedding模式（0或1）。设为1时无需单独指定embed文件。如未指定，则使用服务端默认值。

- `checkpoint_start_pos`：开始保存检查点的 token 位置（≥ 0）；会自动对齐到 128。如未指定，则保留 slot 已有的 checkpoint 设置

- `checkpoint_interval`：保存检查点的间隔 token 数（0 = 禁用）；会自动对齐到 128。如未指定，则保留 slot 已有的 checkpoint 设置

- `max_checkpoint_count`：最多保存的检查点数量。如未指定，则保留 slot 已有的 checkpoint 设置

- `checkpoint_tail_overwrite`：尾部覆盖模式（`0`/`1`）。若为 `1`，最后一个检查点 checkpoint 将作为循环覆盖位。如未指定，则保留 slot 已有的 checkpoint 设置

### 视频输入 (qwen3_vl deepstack 模式)

对于视频理解任务，使用 `video_frames` 内容类型，通过 `frames` 数组传入 base64 编码的视频帧。服务端会按帧对进行处理，使用 deepstack 视觉管线。

启动server示例：

```shell
# 视频模式 (qwen3_vl deepstack)
./rkllm3-server -m Qwen3-VL-2B-llm.rknn --model2 Qwen3-VL-2B-vision.rknn --vocab Qwen3-VL-2B-llm.tokenizer.gguf --embed Qwen3-VL-2B-llm.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --img-width 384 --img-height 384 --video-start "<|vision_start|>" --video-end "<|vision_end|>" --video-content "<|video_pad|>"
```

用户调用示例：

```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key = "sk-no-key-required"
)

def encode_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 编码视频帧（例如按固定帧率从视频中提取）
frame_paths = ["frame_0001.jpg", "frame_0002.jpg", "frame_0003.jpg", "frame_0004.jpg"]
base64_frames = [encode_base64(fp) for fp in frame_paths]

completion = client.chat.completions.create(
    model="Qwen3-VL-2B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "video_frames",
                    "video_frames": {
                        "frames": [
                            f"data:image/jpeg;base64,{b64}"
                            for b64 in base64_frames
                        ]
                    },
                },
                { "type": "text", "text": "请描述一下视频中的内容?" },
            ],
        }
    ],
    stream=True,
    extra_body={
        "n_predict": 512,
        "id_slot": 0,
    }
)

for chunk in completion:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end='', flush=True)
    if chunk.choices[0].finish_reason == "stop":
        break
print('')
```

### POST `/v1/embeddings`: OpenAI兼容的词嵌入API

参见 [OpenAI Embeddings API documentation](https://platform.openai.com/docs/api-reference/embeddings).

示例:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key = "sk-no-key-required"
)

response = client.embeddings.create(
  model="Qwen3-Embedding-4B",
  input="The food was delicious and the waiter...",
  encoding_format="float"
)

print(response.data[0].embedding)
```

## API 错误

`rkllm3-server` 返回的错误格式与 OAI 相同：https://github.com/openai/openai-openapi

错误示例:

```json
{
    "error": {
        "code": 401,
        "message": "Invalid API Key",
        "type": "authentication_error"
    }
}
```

除了 OAI 支持的错误类型之外，我们还有特定于 rkllm3-server 功能的自定义类型：

```json
{
    "error": {
        "code": 501,
        "message": "This server does not support metrics endpoint.",
        "type": "not_supported_error"
    }
}
```

**当通过 /v1/chat/completions 端点收到无效语法时**

```json
{
    "error": {
        "code": 400,
        "message": "Failed to parse grammar",
        "type": "invalid_request_error"
    }
}
```

## Tie Word Embedding

Tie Word Embedding 是一种模型优化技术，将输入 embedding 权重与输出层共享，从而无需单独的 embed 文件（`--embed`），节省磁盘空间和内存。

> **注意**：此功能要求 RKNN 模型在转换时已开启 tie-word 支持。标准模型仍需提供 `--embed` 文件。

### CLI 参数

| 参数                   | 类型     | 默认值 | 说明                                                         |
| ---------------------- | -------- | ------ | ------------------------------------------------------------ |
| `--tie-word-embedding` | `[0\|1]` | `0`    | 启用 tie word embedding 模式，模型必须支持此功能才能启用。设为 `1` 时无需 `--embed` 文件 |

### 每次请求覆盖

若初始化时启用了tie word embedding 模式，并且还提供了embed文件，则可以通过 `extra_body` 在每个 API 请求中覆盖 tie-word 设置：

```json
{
  "messages": [{"role": "user", "content": "你好！"}],
  "tie_word_embedding": 0
}
```

若请求中未指定 `tie_word_embedding`，则使用服务端默认值（来自 CLI 或 `params.json`）。

### 示例：启动服务

```bash
# 标准模型（需要 embed 文件）
./rkllm3-server -m qwen2.5-3b.rknn --vocab qwen2.5-3b.tokenizer.gguf --embed qwen2.5-3b.embed.bin --host 0.0.0.0 --port 8080

# Tie-word 模型（不需要 embed 文件）
./rkllm3-server -m Qwen3-VL-2B-llm-tie-word.rknn --weight Qwen3-VL-2B-llm-tie-word.weight \
  --vocab Qwen3-VL-2B-llm_quant.tokenizer.gguf --tie-word-embedding 1 --host 0.0.0.0 --port 8080 -c 4096
```

### 示例：Python 每次请求覆盖

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required"
)

# 为本次请求覆盖 tie_word_embedding
completion = client.chat.completions.create(
    model="Qwen3-VL-2B",
    messages=[{"role": "user", "content": "你好！"}],
    extra_body={"tie_word_embedding": 0}
)
print(completion.choices[0].message.content)
```

