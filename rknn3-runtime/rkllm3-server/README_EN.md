[简体中文](README_CN.md) | [English](README.md)

# RKLLM3 HTTP Server

rkllm3-server is a basic LLM Server implementation based on RKNPU3.

This project is based on [llama.cpp](https://github.com/ggml-org/llama.cpp).

**Features:**
 * LLM inference based on RKNPU3
 * OpenAI API compatible chat template

## Usage

**Common params**

| Argument | Explanation |
| -------- | ----------- |
| `-a, --alias STRING` | Set model alias (for REST API) |
| `-c, --ctx-size N` | Prompt context size (default: 4096, 0 = load from model) |
| `-n, --predict, --n-predict N` | Number of tokens to predict (default: -1, -1 = context size) |
| `--stage-count N` | Number of model stages (required for multi-card inference) |
| `-m, --model FNAME` | LLM model path (for multi-card inference, set to the first-stage model path, named as `xxx_seg0.rknn`) |
| `--weight FNAME` | LLM weight path (optional; if set, for multi-card inference it must be the first-stage weight path, named as `xxx_seg0.weight`) |
| `--model2 FNAME` | Vision model path for multimodal models |
| `--weight2 FNAME` | Vision weight path for multimodal models (optional) |
| `--model3 FNAME` | Audio model path for multimodal models |
| `--weight3 FNAME` | Audio weight path for multimodal models (optional) |
| `--core-mask MASK` | NPU core mask for LLM model execution, in hex or decimal. Each bit controls one NPU core (e.g., `0x01` = core 0 only, `0x03` = core 0+1, `0xff` = all cores). Default: auto (use all available cores) |
| `--core-mask2 MASK` | NPU core mask for vision model (model2) execution. Default: auto (use all available cores) |
| `--core-mask3 MASK` | NPU core mask for audio model (model3) execution. Default: auto (use all available cores) |
| `--vocab FNAME` | Vocabulary path |
| `--embed FNAME` | Embed.bin path |
| `--embed-ple FNAME` | Embed.bin path for per_layer_inputs for gemma 4 model |
| `--rope-tensor FNAME` | rope.safetensors path for rope cache for gemma 4 model |
| `--mel-filter FNAME` | Mel filters path (for audio models) |
| `--lora-weight FNAME` | LLM lora weight path. A single file may contain multiple adapters, which are shared across all sessions; use `/lora-adapters` API to set per-session scales |
| `--decrypt-key, --key FNAME` | Path to RSA-encrypted user key envelope for LLM model (required for encrypted models) |
| `--decrypt-key2, --key2 FNAME` | Path to RSA-encrypted user key envelope for vision model (required for encrypted models) |
| `--decrypt-key3, --key3 FNAME` | Path to RSA-encrypted user key envelope for audio model (required for encrypted models) |
| `--img-start STRING` | Image input prefix for multimodal models |
| `--img-end STRING` | Image input suffix for multimodal models |
| `--img-content STRING` | Image input pad for multimodal models |
| `--audio-start STRING` | Audio input prefix for multimodal models |
| `--audio-end STRING` | Audio input suffix for multimodal models |
| `--audio-content STRING` | Audio input pad for multimodal models |
| `--video-start STRING` | Video input prefix for qwen3_vl deepstack video mode |
| `--video-end STRING` | Video input suffix for qwen3_vl deepstack video mode |
| `--video-content STRING` | Video input pad for qwen3_vl deepstack video mode |
| `--img-width N` | Input image width for multimodal models（Some pruned models require configuration, such as qwen3_vl）|
| `--img-height N` | Input image height for multimodal models（Some pruned models require configuration, such as qwen3_vl）|
| `--chat-template-file JINJA_TEMPLATE_FILE` | Set custom Jinja chat template (default: use template from model metadata) |
| `--embedding` | Is it a embedding model |
| `--reasoning-format FORMAT` | controls whether thought tags are allowed and/or extracted from the response, and in which format they're returned; one of:<br/>- none: leaves thoughts unparsed in `message.content`<br/>- deepseek: puts thoughts in `message.reasoning_content`<br/>(default: none) |
| `--reasoning [on\|off\|auto]` | Use reasoning/thinking in the chat ('on', 'off', or 'auto', default: 'auto' (detect from template)) |
| `--record-path STRING` | Set the directory to save user requests (default is empty, no saving) |
| `--n-session` | number of session, for multi-session scenarios (default: 1) |
| `--bucket-size` | Token bucket size processed per inference pass in multi-card inference, typically 128 (default: 128) |

**Sampling params**

| Argument | Explanation |
| -------- | ----------- |
| `--temp N` | Temperature (default: 0.8) |
| `--top-k N` | Top-k sampling (default: 40, 0 = disabled) |
| `--top-p N` | Top-p sampling (default: 0.9, 1.0 = disabled) |
| `--repeat-penalty N` | Last n tokens to consider for penalize (default: 1.0, 1.0 = disabled) |
| `--presence-penalty N` | Repeat alpha presence penalty (default: 0.0, 0.0 = disabled) |
| `--frequency-penalty N` | Repeat alpha frequency penalty (default: 0.0, 0.0 = disabled) |

**Example-specific params**

| Argument | Explanation |
| -------- | ----------- |
| `-h, --help, --usage` | Print usage and exit |
| `--host HOST` | Listening IP address (default: 127.0.0.1) |
| `--port PORT` | Listening port (default: 8080) |
| `-to, --timeout N` | Server read/write timeout in seconds (default: 600) |
| `--device-id STRING` | Device ID (optional; for multi-card inference, separate multiple device IDs with `#`) |
| `--log-level N` | Log level (default: 0) |

## Quick Start

Run the following commands to start the server process:

```bash
# LLM Model (standard, with embed file)
./rkllm3-server -m qwen2.5-3b.rknn --vocab qwen2.5-3b.tokenizer.gguf --embed qwen2.5-3b.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8

# LLM Model (tie-word, no embed file needed)
./rkllm3-server -m Qwen3-VL-2B-llm-tie-word.rknn --weight Qwen3-VL-2B-llm-tie-word.weight --vocab Qwen3-VL-2B-llm_quant.tokenizer.gguf --tie-word-embedding 1 --host 0.0.0.0 --port 8080 -c 4096 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8

./rkllm3-server -m gemma-4.rknn --vocab gemma-4.tokenizer.gguf --embed gemma-4.embed.bin --embed-ple gemma-4_per_layer_inputs.embed.bin --rope-tensor gemma-4.safetensors --host 0.0.0.0 --port 8080 -c -1 --n_predict 2048  --repeat-penalty 1.0 --presence-penalty 0 --frequency-penalty 0 --top-k 1 --top-p 0.8 --temp 0.8

# LLM Model (multi-card cascade inference, using dual-card cascade as an example)
./rkllm3-server --stage-count 2 -m Qwen3.5-9B-llm_seg0.rknn --weight Qwen3.5-9B-llm_seg0.weight --vocab Qwen3.5-9B-llm_seg0.gguf --embed Qwen3.5-9B-llm_seg0.embed.bin --host 0.0.0.0 --port 8080 -c 4096 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8

# Multimodal Model (LLM+VISION)
./rkllm3-server -m MiniCPM-3o-llm.rknn --model2 MiniCPM-3o-vision.rknn --vocab MiniCPM-3o-llm.tokenizer.gguf --embed MiniCPM-3o-llm.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<image>" --img-end "</image>" --img-content "<unk>"

./rkllm3-server -m Qwen2.5-VL-3B-llm.rknn --model2 Qwen2.5-VL-3B-vision.rknn --vocab Qwen2.5-VL-3B-llm.tokenizer.gguf --embed Qwen2.5-VL-3B-llm.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --img-width 392 --img-height 392

./rkllm3-server -m Qwen3-VL-2B-llm.rknn --model2 Qwen3-VL-2B-vision.rknn --vocab Qwen3-VL-2B-llm.tokenizer.gguf --embed Qwen3-VL-2B-llm.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --img-width 384 --img-height 384

# Video mode (qwen3_vl deepstack)
./rkllm3-server -m Qwen3-VL-2B-llm.rknn --model2 Qwen3-VL-2B-vision.rknn --vocab Qwen3-VL-2B-llm.tokenizer.gguf --embed Qwen3-VL-2B-llm.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --img-width 384 --img-height 384 --video-start "<|vision_start|>" --video-end "<|vision_end|>" --video-content "<|video_pad|>"

./rkllm3-server -m Qwen3.5-4B.rknn --model2 Qwen3.5-4B-vision.rknn --vocab Qwen3.5-4B.tokenizer.gguf --embed Qwen3.5-4B.embed.bin --host 0.0.0.0 --port 8080 -c 0 --n_predict 2048 --repeat-penalty 1.0 --presence-penalty 1.5 --frequency-penalty 0 --top-k 1 --top-p 0.9 --temp 1.0 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --img-width 384 --img-height 384

# Multimodal Model (LLM+AUDIO)
./rkllm3-server -m gemma-4.rknn --model3 gemma-4-audio.rknn --vocab gemma-4.tokenizer.gguf --embed gemma-4.embed.bin --embed-ple gemma-4_per_layer_inputs.embed.bin --rope-tensor gemma-4.safetensors --host 0.0.0.0 --port 8080 -c -1 --n_predict 2048  --repeat-penalty 1.0 --presence-penalty 0 --frequency-penalty 0 --top-k 1 --top-p 0.8 --temp 0.8 --audio-start "<|audio>" --audio-end "<audio|>" --audio-content "<|audio|>"

# Multimodal Model (LLM+VISION+AUDIO)
./rkllm3-server -m Qwen2.5-Omni-3B-llm.rknn --model2 Qwen2.5-Omni-3B-vision.rknn --model3 Qwen2.5-Omni-3B-audio.rknn --vocab Qwen2.5-Omni-3B-llm.tokenizer.gguf --embed Qwen2.5-Omni-3B-llm.embed.bin --mel-filter mel_128_filters.txt --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_bos|>" --img-end "<|vision_eos|>" --img-content "<|IMAGE|>" --audio-start "<|audio_bos|>" --audio-end "<|audio_eos|>" --audio-content "<|AUDIO|>"

# Embedding Model
./rkllm3-server -m Qwen3-Embedding-4B.rknn --vocab Qwen3-Embedding-4B.tokenizer.gguf --embed Qwen3-Embedding-4B.embed.bin --embedding

# Encrypted model (LLM text-only)
./rkllm3-server -m Qwen3-4B.rknn.enc --decrypt-key Qwen3-4B.key --vocab Qwen3-4B.tokenizer.gguf --embed Qwen3-4B.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512 --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8

# Encrypted model (LLM+VISION multimodal)
./rkllm3-server -m Qwen3-VL-4B-llm.rknn.enc --model2 Qwen3-VL-4B-vision.rknn.enc --decrypt-key Qwen3-VL-4B-llm.key --decrypt-key2 Qwen3-VL-4B-vision.key --vocab Qwen3-VL-4B-llm.tokenizer.gguf --embed Qwen3-VL-4B-llm.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512 --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --img-width 384 --img-height 384 --video-start "<|vision_start|>" --video-end "<|vision_end|>" --video-content "<|video_pad|>"

# Encrypted model (LLM+VISION+AUDIO omni multimodal)
./rkllm3-server -m Qwen3-Omni-4B-llm.rknn.enc --model2 Qwen3-Omni-4B-vision.rknn.enc --model3 Qwen3-Omni-4B-audio.rknn.enc --decrypt-key Qwen3-Omni-4B-llm.key --decrypt-key2 Qwen3-Omni-4B-vision.key --decrypt-key3 Qwen3-Omni-4B-audio.key --vocab Qwen3-Omni-4B-llm.tokenizer.gguf --embed Qwen3-Omni-4B-llm.embed.bin --mel-filter mel_128_filters.txt --host 0.0.0.0 --port 8080 -c 768 --n_predict 512 --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --audio-start "<|audio_start|>" --audio-end "<|audio_end|>" --audio-content "<|audio_pad|>"

# Load multiple models simultaneously (ensure sufficient memory to load multiple models)
./rkllm3-server --params-file params.json
```

When loading multiple models, the basic format of params.json is as follows:
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

When loading multi-card inference models, the basic format of params.json is as follows:
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

Note: Using the `--params-file` parameter will ignore other command-line arguments; only the parameters within params.json are effective.

Additionally, besides providing the regular rkllm3-server binary executable, RKLLM3 Server also offers a usage method via `librkllm3-server.so`, facilitating the integration of the server into applications. The usage is as follows:
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

// Status callback function for the server sub-thread
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
    std::thread server_thread( {
        // start_server blocks until stop_server is called,
        // so it must be executed in a sub-thread.
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
The strings for "model", "weight", "model2", "weight2", "model3", "weight3", "vocab", "embedding", "mel-filter", "lora-weight" inside the `const char *json` can be either file paths or file descriptor handles (to address file permission issues on Android systems), as shown below:
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

## Testing with CURL

Use https://curl.se/.

```sh
curl --request POST \
    --url http://localhost:8080/v1/chat/completions \
    --header "Content-Type: application/json" \
    --data '{"messages": [{"role": "user", "content": "Hello!"}],"n_predict": 128}'
```

## Multi-Session

The `--n-session` parameter creates multiple independent sessions (Slots), each with its own dedicated KV Cache and conversation context, enabling concurrent handling of multiple user chat requests.

When `--n-session` is greater than 1, the server creates the corresponding number of slots. Clients specify which slot to use via the `id_slot` parameter in the request body (slot ID starts from 0, range `[0, --n-session - 1]`, default is 0).

### Starting the Server

```bash
# Create 3 sessions
./rkllm3-server -m qwen2.5-3b.rknn --vocab qwen2.5-3b.tokenizer.gguf --embed qwen2.5-3b.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512 --n-session 3
```

### curl Example

```bash
# Slot 0 conversation
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
  "messages": [{"role": "user", "content": "Hello, my name is Alice"}],
  "id_slot": 0
}'

# Slot 1 conversation (independent context, unaware of Slot 0)
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
  "messages": [{"role": "user", "content": "Hello, my name is Bob"}],
  "id_slot": 1
}'
```

### Use Cases

- **Multi-user concurrent service**: Multiple users chat simultaneously, each with independent KV Cache, no interference
- **Conversation isolation**: Use different slots for different tasks/topics to avoid context confusion

### Per-Slot Context Size

When using `--n-session N` (multi-session mode), each slot can have a **different** context size (ctx-size), meeting diverse memory and context length requirements for different scenarios. You can set per-slot values via the CLI using `#` separators, or via `params.json` using JSON arrays.

**CLI example** (three slots with different ctx-sizes):

```bash
./rkllm3-server -m qwen2.5-3b.rknn --vocab qwen2.5-3b.tokenizer.gguf --embed qwen2.5-3b.embed.bin \
  --host 0.0.0.0 --port 8080 --n-session 3 --n_predict 512 \
  -c 4096#2048#1024
```

**params.json example** (same configuration using JSON arrays):

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

With `"n-session": 3`, this configuration gives:

| Slot | ctx-size | Suitable For |
|------|----------|--------------|
| 0    | 4096     | Long documents / multi-turn conversations |
| 1    | 2048     | Regular conversations |
| 2    | 1024     | Short Q&A / resource-constrained |

If the array has fewer entries than `n-session`, the scalar default is used for remaining slots. Scalar values (e.g., `"ctx-size": 4096`) are also supported as a backward-compatible way to set the same value for all slots.

> **Note**: Each slot's ctx-size affects NPU memory allocation. Larger ctx-sizes consume more memory. Please configure appropriately based on actual memory availability.

### Per-Slot Predict Length

In multi-session mode, each slot can also have a **different** prediction token count (n-predict), meeting diverse generation length requirements. Set per-slot values via CLI `#` separators or `params.json` JSON arrays.

**CLI example** (three slots with different n-predict values):

```bash
./rkllm3-server -m qwen2.5-3b.rknn --vocab qwen2.5-3b.tokenizer.gguf --embed qwen2.5-3b.embed.bin \
  --host 0.0.0.0 --port 8080 --n-session 3 -c 4096 \
  -n 512#256#-1
```

**params.json example** (same configuration using JSON arrays):

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

With `"n-session": 3`, this configuration gives:

| Slot | n-predict | Meaning |
|------|-----------|---------|
| 0    | 512       | Generate up to 512 tokens |
| 1    | 256       | Generate up to 256 tokens |
| 2    | -1        | Set as the length of the context |

If the array has fewer entries than `n-session`, the scalar default is used for remaining slots. Scalar values (e.g., `"predict": 512`) are also supported as a backward-compatible way to set the same value for all slots.

> **Note**: Each request can further override the slot's default n-predict via the `n_predict` field in `extra_body`, providing more flexible per-request control.

## API Endpoints

### GET `/health`: Returns health check result

**Response Format**

- HTTP Status Code 503
  - Body: `{"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}`
  - Description: Model is being loaded
- HTTP Status Code 200
  - Body: `{"status": "ok" }`
  - Description: The model has been successfully loaded and the server is ready

### GET `/lora-adapters`: Get list of all LoRA adapters

This endpoint returns the loaded LoRA adapters. You can add adapters using `--lora-weight` when starting the server, for example: `--lora-weight Qwen2.5-3B-Instruct.lora_weight`

A single LoRA weight file may contain multiple adapters. All adapters loaded from this file are shared across all sessions (slots), but each session can configure its own scale for each adapter independently at runtime.

By default, all adapters will be loaded with scale set to 0.

Please note that this value will be overwritten by the `lora` field for each user request.

If an adapter is disabled, the scale will be set to 0.

**Request format**
```json
{
    "model": "Qwen2.5-3B-Instruct",
    "id_slot": 0
}
```

`id_slot` is optional and defaults to `0`. It specifies which session's LoRA configuration to query. When `--n-session` is greater than 1, use this to retrieve the configuration for a specific session.

**Response format**

```json
[
  {"name": "lora0_pattern0", "scale": 0.0},
  {"name": "lora1_pattern0", "scale": 0.0}
]
```

### POST `/lora-adapters`: Set list of LoRA adapters

This sets the scale for LoRA adapters for a specific session (slot). Please note that this value will be overwritten by the `lora` field for each request.

To disable an adapter, either remove it from the list below, or set scale to 0.

**Request format**

To know the `name` of the adapter, use GET `/lora-adapters`

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

`id_slot` is optional and defaults to `0`. It specifies which session's LoRA configuration to update. When `--n-session` is greater than 1, use this to configure each session independently. The same set of adapters (loaded from `--lora-weight`) is shared across all sessions, but each session can have its own scale values.
```

## Slot Pause/Resume

Pause/Resume allows controlling the inference state of a specific slot. Pause pauses inference without destroying its KV Cache, Resume picks up from the pause point. This is useful for resource scheduling and dynamic time-slicing between concurrent slots.

> **Note**: Pause/Resume is only available when `--n-session 2` (or higher).

### POST `/slots/{slot_id}/pause`: Pause a slot

Pause the inference of the specified slot. The slot's KV Cache is preserved.

**Request format**

```json
{
    "model": "model_alias"
}
```

The `model` field is optional and only required when multiple models are loaded.

**Success response** (HTTP 200):

```json
{
    "success": true,
    "id_slot": 0,
    "state": "paused"
}
```

**Error response** (HTTP 400):

| Error message | Reason |
|---------------|--------|
| `Slot is already paused` | The slot is already in paused state |
| `Slot is not in a pausable state` | The slot is IDLE or not in a state that can be paused |
| `Invalid slot id: N` | The slot ID is out of range |

### POST `/slots/{slot_id}/resume`: Resume a slot

Resume the inference of a paused slot.

**Request format**

```json
{
    "model": "model_alias"
}
```

The `model` field is optional and only required when multiple models are loaded.

**Success response** (HTTP 200):

```json
{
    "success": true,
    "id_slot": 0,
    "state": "resumed"
}
```

**Error response** (HTTP 400):

| Error message | Reason |
|---------------|--------|
| `Slot is not paused, cannot resume` | The slot is not currently in paused state |
| `Invalid slot id: N` | The slot ID is out of range |

## Slot Interrupt

Interrupt allows directly terminating the inference of a specified slot and resetting the slot to IDLE state. Unlike Pause, Interrupt directly terminates inference and releases the slot. Use the `save_kvcache_path` parameter to automatically save KV Cache after interrupting (one-step interrupt + save, no separate save call needed). Combined with KV Cache Save/Load, this enables advanced scheduling scenarios like priority interrupt.

### POST `/slots/{slot_id}/interrupt`: Interrupt a slot

Interrupt the inference of the specified slot, resetting the slot to IDLE state.

**Request format**

```json
{
    "model": "model_alias",
    "save_kvcache_path": "/path/to/kvcache.kv"
}
```

- The `model` field is optional and only required when multiple models are loaded
- `save_kvcache_path` is optional; when specified, KV Cache and inference state are automatically saved after interrupt (state file path is derived: `xxx.kv` -> `xxx.kv.json`)

**Success response** (HTTP 200):

```json
{
    "success": true,
    "id_slot": 0,
    "state": "idle",
    "kvcache_saved": true,
    "kvcache_path": "/path/to/kvcache.kv"
}
```

If `save_kvcache_path` is not specified, the response will not include `kvcache_saved` and `kvcache_path` fields.

**Error response** (HTTP 400):

| Error message | Reason |
|---------------|--------|
| `Invalid slot id: N` | The slot ID is out of range |
| `Slot is not processing, cannot interrupt` | The slot is not currently in inference state |

*Example:*

```bash
# Interrupt slot 0
curl -X POST http://localhost:8080/slots/0/interrupt \
  -H "Content-Type: application/json" \
  -d '{}'

# Interrupt slot 0 and save KV Cache
curl -X POST http://localhost:8080/slots/0/interrupt \
  -H "Content-Type: application/json" \
  -d '{"save_kvcache_path": "/data/kvcache/task.kv"}'
```

### Priority Interrupt

The priority interrupt feature allows interrupting a low-priority task mid-inference, executing a high-priority task, and then resuming the low-priority task from where it left off. This feature combines Slot Interrupt and KV Cache Save/Load capabilities, and is useful for multi-task scheduling scenarios.

#### Typical Workflow

1. **Start low-priority task**: Start a long-text inference task on slot 0 in streaming mode
2. **Interrupt and save**: Call `/slots/0/interrupt` with `save_kvcache_path` to interrupt + save KV Cache in one step (the state file path is automatically derived from the KV cache path: `xxx.kv` -> `xxx.kv.json`)
3. **Disconnect to clean up**: Disconnect the low-priority task's client connection; the server detects the closed connection and cleans up the stale connection
4. **Execute high-priority task**: Run a high-priority task on slot 0 and wait for completion
5. **Resume low-priority task**: Re-send the exact same prompt as step 1, specifying `load_kvcache_path` in `extra_body`; the server will load the KV cache and state (derived from the cache path) and resume inference from the interrupt point

#### Python Example

The following example demonstrates the complete priority interrupt workflow:

```python
import requests
import json

BASE_URL = "http://127.0.0.1:8080"
SLOT_ID = 0

# Step 1: Start low-priority task (long text, streaming mode)
# Use a subprocess or separate thread to send the request,
# so you can interrupt after receiving a few tokens

# Step 2: Interrupt slot 0 and save KV Cache
# One-step interrupt + save, state file is derived from the KV cache path:
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

# Step 3: Disconnect the low-priority task's client (kill subprocess / close socket)
# Server detects is_connection_closed() → cancel_tasks() → cleans up stale connection

# Step 4: Execute high-priority task
resp = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "Write a short poem about spring."}],
        "max_tokens": 256,
        "stream": False,
        "id_slot": SLOT_ID
    },
    timeout=120
)
print(resp.json()["choices"][0]["message"]["content"])

# Step 5: Resume low-priority task
# Use the exact same prompt as step 1, with load_kvcache_path in extra_body.
# The state file is automatically loaded from the derived path:
#   /data/kvcache/task_low_priority.kv -> /data/kvcache/task_low_priority.kv.json
resp = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "Write a detailed article about the history of AI..."}],
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

# Read streaming output — generation continues from the stop point
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

#### Use Cases

- **Multi-task scheduling**: When a high-priority request arrives, interrupt the current low-priority task, execute the high-priority task, and then resume
- **User interruption**: When a user sends a new question while waiting for a long response, interrupt the old task, handle the new question, and then resume
- **Resource management**: Dynamically allocate NPU compute, ensuring critical tasks complete first

## KV Cache Save/Load

KV Cache save/load allows you to persist the KV Cache of a specific slot to a file, or restore it from a file, enabling **conversation KV cache persistence**. Use cases include:

- **Continue conversations**: Save the KV Cache of the current conversation, and load it after restarting the server to continue without re-providing the history
- **Warm-up acceleration**: Pre-compute prompt KV Cache files to avoid repeated prefix computation

> **Note**: `save_kvcache_path` and `load_kvcache_path` are per-request `extra_body` parameters. They **cannot be used together in the same request**. The save operation executes after generation completes; the load operation executes before generation starts. When loading KV Cache, the `messages` in the request are still processed (useful for appending new messages).

### Usage

#### Saving KV Cache

Set the `save_kvcache_path` parameter in the request. After generation completes, the KV Cache will be automatically saved to the specified path. Ensure the **parent directory** of the target path already exists.

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required"
)

# Chat and save KV Cache
completion = client.chat.completions.create(
    model="Qwen2.5-3B",
    messages=[
        {"role": "user", "content": "Hello, my name is Bob, please remember it"}
    ],
    extra_body={"save_kvcache_path": "/data/kvcache/user_bob.kv"}
)
print(completion.choices[0].message.content)
```

#### Loading KV Cache

Set the `load_kvcache_path` parameter in the request. The KV Cache will be loaded from the specified path before generation. Ensure the **file already exists**.

```python
# Load KV Cache and continue conversation
completion = client.chat.completions.create(
    model="Qwen2.5-3B",
    messages=[
        {"role": "user", "content": "What is my name?"}
    ],
    extra_body={"load_kvcache_path": "/data/kvcache/user_bob.kv"}
)
print(completion.choices[0].message.content)
# Output: Your name is Bob
```

### Combining with Multi-Session

When used with `id_slot`, each slot can independently persist its own context:

```python
# Slot 0: Save Bob's KV Cache
client.chat.completions.create(
    model="Qwen2.5-3B",
    messages=[{"role": "user", "content": "Hi, I'm Bob"}],
    extra_body={"id_slot": 0, "save_kvcache_path": "/data/kvcache/slot0.kv"}
)

# Slot 1: Save Alice's KV Cache
client.chat.completions.create(
    model="Qwen2.5-3B",
    messages=[{"role": "user", "content": "Hi, I'm Alice"}],
    extra_body={"id_slot": 1, "save_kvcache_path": "/data/kvcache/slot1.kv"}
)

# Later, load each independently with separate KV Caches
completion = client.chat.completions.create(
    model="Qwen2.5-3B",
    messages=[{"role": "user", "content": "What is my name?"}],
    extra_body={"id_slot": 0, "load_kvcache_path": "/data/kvcache/slot0.kv"}
)
print(completion.choices[0].message.content)  # Output: Your name is Bob
```

### Error Messages

| Error message                                                | Reason                                               |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| `'save_kvcache_path' and 'load_kvcache_path' cannot both be set in the same request` | Both parameters set in the same request              |
| `directory for 'save_kvcache_path' does not exist: ...`      | The parent directory of the save path does not exist |
| `file for 'load_kvcache_path' does not exist: ...`           | The file to load does not exist                      |

### Dedicated KV Cache API Endpoints

In addition to triggering save/load via `extra_body` parameters in `/v1/chat/completions` requests, dedicated REST API endpoints are provided to directly save or load the KV Cache of a specific slot.

> **Note**: These endpoints require the target slot to be in **idle state** (IDLE), meaning no inference task is currently running on that slot. If the slot is processing, the endpoint will return an error.

#### POST `/slots/{slot_id}/kvcache/save`: Save KV Cache of a slot

Save the KV Cache of the specified slot to a file.

**Request format**

```json
{
    "model": "model_alias",
    "path": "/data/kvcache/backup.kv"
}
```

| Field | Description |
|-------|-------------|
| `model` | Optional, model alias or path. Only required when multiple models are loaded |
| `path` | **Required**, file path to save the KV Cache. Ensure the **parent directory** already exists |

**Success response** (HTTP 200):

```json
{
    "success": true,
    "id_slot": 0,
    "path": "/data/kvcache/backup.kv"
}
```

**Error response** (HTTP 400 / 500):

| Error message | Reason |
|---------------|--------|
| `'path' is required for kvcache save` | The `path` field is missing from the request |
| `Invalid slot id: N` | The slot ID is out of range |
| `Failed to save kvcache for slot` | Save failed (slot may be processing, or path is invalid) |

#### POST `/slots/{slot_id}/kvcache/load`: Load KV Cache into a slot

Load KV Cache from a file into the specified slot.

> **Note**: The load operation will **overwrite** the target slot's current KV Cache. After loading, the slot inherits the KV Cache state from the file, enabling cross-slot migration or restoration of previous conversation context.

**Request format**

```json
{
    "model": "model_alias",
    "path": "/data/kvcache/backup.kv"
}
```

| Field | Description |
|-------|-------------|
| `model` | Optional, model alias or path. Only required when multiple models are loaded |
| `path` | **Required**, file path to load the KV Cache from. Ensure the **file already exists** |

**Success response** (HTTP 200):

```json
{
    "success": true,
    "id_slot": 1,
    "path": "/data/kvcache/backup.kv"
}
```

**Error response** (HTTP 400 / 500):

| Error message | Reason |
|---------------|--------|
| `'path' is required for kvcache load` | The `path` field is missing from the request |
| `Invalid slot id: N` | The slot ID is out of range |
| `Failed to load kvcache for slot` | Load failed (slot may be processing, or file does not exist) |

#### Usage Examples

```bash
# First, complete a conversation on slot 0 (via chat/completions)
# Then save slot 0's KV Cache to a file
curl -X POST http://localhost:8080/slots/0/kvcache/save \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/kvcache/slot0_backup.kv"}'

# Load the saved KV Cache into slot 1 (cross-slot migration)
curl -X POST http://localhost:8080/slots/1/kvcache/load \
  -H "Content-Type: application/json" \
  -d '{"path": "/data/kvcache/slot0_backup.kv"}'

# Continue the conversation on slot 1 after loading
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is my name?"}],
    "id_slot": 1
  }'
```

#### Typical Use Cases

- **Conversation backup & restore**: Save KV Cache at key conversation milestones, and restore after server restart
- **Cross-slot context migration**: Migrate conversation context from slot A to slot B, freeing slot A for new tasks
- **Pre-computation acceleration**: Pre-compute prompts on one slot, save the KV Cache, and load it on other slots to avoid repeated prefix computation

## KV Cache Checkpoint

The checkpoint feature allows saving KV Cache snapshots during inference (for linear-attention or sliding-window-attention models such as Qwen3.5 and Gemma4 series). These snapshots can later be restored to resume generation from a previous position, which is useful for:

- **Multi-turn sessions**: save KV Cache at key conversation milestones to save prefill time
- **Multi-session scenarios**: each concurrent slot can have its own independent checkpoint strategy
- **Per-request checkpoint policy**: adjust checkpoint parameters per API request for dynamic control

> **Note**: The checkpoint parameters only take effect for models with linear attention or sliding-window attention (e.g., Qwen3.5, Gemma4 series); they have no effect on other models. All `checkpoint_start_pos` and `checkpoint_interval` values are auto-aligned to 128 by the NPU runtime. `max_checkpoint_count` is auto-clamped to `(ctx_size - checkpoint_start_pos) / checkpoint_interval` if the specified value exceeds this limit.

### Parameters

| Parameter | CLI | JSON (`params.json`) | Per-request (`extra_body`) | Description |
|-----------|-----|----------------------|---------------------------|-------------|
| `checkpoint_start_pos` | `--checkpoint-start-pos N` or `N#N#...` | `"checkpoint-start-pos": N` or `[N, N, ...]` | `"checkpoint_start_pos": N` | Token position at which to begin saving checkpoints (must be ≥ 0); auto-aligned to 128 |
| `checkpoint_interval` | `--checkpoint-interval N` or `N#N#...` | `"checkpoint-interval": N` or `[N, N, ...]` | `"checkpoint_interval": N` | Token interval for periodic checkpoint saving (0 = disabled); auto-aligned to 128 |
| `max_checkpoint_count` | `--max-checkpoint-count N` or `N#N#...` | `"max-checkpoint-count": N` or `[N, N, ...]` | `"max_checkpoint_count": N` | Maximum number of checkpoints to save from start_pos; should be ≤ (ctx_size - start_pos) / interval; auto-adjusted if exceeded |
| `checkpoint_tail_overwrite` | `--checkpoint-tail-overwrite 0\|1` or `0\|1#...` | `"checkpoint-tail-overwrite": 0\|1` or `[0, 1, ...]` | `"checkpoint_tail_overwrite": 0\|1` | Tail overwrite mode. When `1`, the last checkpoint (`max_checkpoint_count`) acts as a rolling overwrite position while other checkpoints remain at their fixed positions (default: `0`) |

All four parameters default to `0`, meaning checkpoint saving is disabled by default.

### Configuration Methods

Checkpoint parameters can be set at three levels, with the following priority (higher overrides lower):

1. **Per-request** (highest priority): set `checkpoint_start_pos`, `checkpoint_interval`, `max_checkpoint_count`, `checkpoint_tail_overwrite` in the `/v1/chat/completions` request body's `extra_body` field
2. **Server-wide CLI**: `--checkpoint-start-pos`, `--checkpoint-interval`, `--max-checkpoint-count`, `--checkpoint-tail-overwrite` (use `#` separator for per-slot values, e.g. `--checkpoint-start-pos 128#256#0`)
3. **Server-wide params.json**: `"checkpoint-start-pos"`, `"checkpoint-interval"`, `"max-checkpoint-count"`, `"checkpoint-tail-overwrite"` (use JSON arrays for per-slot values)

### Single-Session Example

Start the server with checkpoint parameters:

```bash
./rkllm3-server -m Qwen3.5-4B.rknn --vocab Qwen3.5-4B.tokenizer.gguf --embed Qwen3.5-4B.embed.bin \
  --host 0.0.0.0 --port 8080 -c -1 --n_predict 2048 \
  --checkpoint-start-pos 128 --checkpoint-interval 128 --max-checkpoint-count 4 --checkpoint-tail-overwrite 1
```

Or use `params.json`:

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

Override checkpoint params per API request:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "extra_body": {
      "checkpoint_start_pos": 1024,
      "checkpoint_interval": 512,
      "max_checkpoint_count": 4,
      "checkpoint_tail_overwrite": 1
    }
  }'
```

If a request does NOT specify these checkpoint keys, the slot's existing checkpoint settings are preserved — they are not reset to server defaults.

### Multi-Session with Per-Slot Checkpoint Config

When using `--n-session N` (multi-session mode), each session can have a **different** checkpoint configuration. You can set per-slot values via the CLI using `#` separators, or via `params.json` using JSON arrays.

**CLI example** (three slots, different checkpoint configs per slot):

```bash
./rkllm3-server -m Qwen3.5-4B.rknn --vocab Qwen3.5-4B.tokenizer.gguf --embed Qwen3.5-4B.embed.bin \
  --host 0.0.0.0 --port 8080 -c -1 --n-session 3 --n_predict 2048 \
  --checkpoint-start-pos 128#256#0 --checkpoint-interval 128#256#0 --max-checkpoint-count 2#3#0 --checkpoint-tail-overwrite 1#0#0
```

**params.json example** (same configuration using JSON arrays):

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

With `"n-session": 3`, this configuration gives:

| Slot | `start_pos` | `interval` | `max_count` | `tail_overwrite` | Effect |
|------|-------------|------------|-------------|------------------|--------|
| 0    | 128         | 128        | 8         | 1                | Aggressive checkpointing, tail overwrite |
| 1    | 256         | 256        | 4         | 0                | Moderate checkpointing |
| 2    | 0           | 0          | 0           | 0                | Disabled |

If the array has fewer entries than `n-session`, the scalar default (or `0`) is used for remaining slots. Scalar values (e.g., `"checkpoint-interval": 128`) are also supported as a backward-compatible way to set the same value for all slots.

## OpenAI Compatible API Endpoints

### GET `/v1/models`: OpenAI compatible model information API

Returns information about the loaded model(s). See https://platform.openai.com/docs/api-reference/models.

The returned list can have multiple elements, corresponding to multiple models.

By default, the model `id` field is the model file path specified by `-m`. You can set a custom value for the model `id` field using the `--alias` parameter. For example, `--alias Qwen2.5-3B`.

Example:

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

### POST `/v1/chat/completions`: OpenAI compatible chat completions API


Given a JSON description of CHATML form in `messages`, returns the predicted completion. Supports both synchronous and streaming modes. Although not fully implementing the OpenAI API specification, it is sufficient to support many applications. Only models with a https://github.com/ggml-org/llama.cpp/wiki/templates-supported-by-llama_chat_apply_template can be used relatively normally under this endpoint. By default, the CHATML template will be used.

*Options:*

See https://platform.openai.com/docs/api-reference/chat.


*Example:*

You can use the Python `openai` library:

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

... or a raw HTTP request:

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

Additionally, for multimodal models, it is recommended to use the openai interface. Example:
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
audio_path = "demo.wav"     # If audio input exists

# Getting the Base64 string
base64_image = encode_base64(image_path)
base64_audio = encode_base64(audio_path)    # If audio input exists

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
                },  # If audio input exists
                { "type": "text", "text": "请描述一下图片?" },
            ],
        }
    ],
    stream=True,
    extra_body={
        "n_predict": 256,
        "chat_template_kwargs": { "enable_thinking": False }     # For Thinking models (like Qwen3), thinking output can be disabled this way
        "id_slot": 0,           # When the --n-session parameter is greater than 1, you can use this parameter to switch between different sessions.

        # save_kvcache_path & load_kvcache_path does not work currently
        "save_kvcache_path": "/data/kvcache/1.kv",       # save kvcache to file
        "load_kvcache_path": "/data/kvcache/1.kv",       # load kvcache from file

        # sampling parameters
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

Note: The chunk.choices[0].finish_reason in the above code indicates the completion reason: "stop" for normal completion, "length" for truncation due to reaching the max_tokens limit, and "tool_calls" for triggering tool calls.

Options supported by `extra_body`(non-OpenAI compatible):

- `n_predict`: The number of tokens to predict for the current request

- `chat_template_kwargs`: The chat template parameters for the current request, such as `{ "enable_thinking": False }` to control thinking output

- `id_slot`: Specify the ID of the current slot (when the --n-session parameter is greater than 1, you can use this parameter to switch between different sessions. This value must be less than --n-session, default is 0)

- `save_kvcache_path`: save kvcache to file (cannot be used simultaneously with load_kvcache_path)

- `load_kvcache_path`: load kvcache from file (cannot be used simultaneously with save_kvcache_path)

- `top_k`: Top-k sampling

- `top_p`: Top-p sampling

- `temperature`: Temperature

- `repeat_penalty`: Last n tokens to consider for penalize

- `frequency_penalty`: Repeat alpha frequency penalty

- `presence_penalty`: Repeat alpha presence penalty

- `lora`: A list of LoRA adapters to be applied. Each object in the list must contain `name` and `scale` fields. For example: `[{"name": "lora0_pattern0", "scale": 0.5}, {"name": "lora1_pattern0", "scale": 0.6}]`. If a LoRA adapter is not specified in the list, its scale will default to `0.0`.

- `tie_word_embedding`: Enable tie word embedding mode for the current request (0 or 1). When set to 1, no separate embed file is required. If not specified, the server-wide default is used.

- `checkpoint_start_pos`: Token position to begin saving checkpoints (≥ 0); auto-aligned to 128. If not specified, the slot's existing checkpoint settings are preserved

- `checkpoint_interval`: Token interval for periodic checkpoint saving (0 = disabled); auto-aligned to 128. If not specified, the slot's existing checkpoint settings are preserved

- `max_checkpoint_count`: Maximum number of checkpoints to save. If not specified, the slot's existing checkpoint settings are preserved

- `checkpoint_tail_overwrite`: Tail overwrite mode (`0`/`1`). When `1`, the last checkpoint acts as a rolling overwrite position. If not specified, the slot's existing checkpoint settings are preserved

### Video Input (qwen3_vl deepstack mode)

For video understanding, use the `video_frames` content type with a `frames` array containing base64-encoded frames. The server processes frames in pairs and uses the deepstack vision pipeline.

Server launch example:

```shell
# Video mode (qwen3_vl deepstack)
./rkllm3-server -m Qwen3-VL-2B-llm.rknn --model2 Qwen3-VL-2B-vision.rknn --vocab Qwen3-VL-2B-llm.tokenizer.gguf --embed Qwen3-VL-2B-llm.embed.bin --host 0.0.0.0 --port 8080 -c 768 --n_predict 512  --repeat-penalty 1.1 --presence-penalty 1.0 --frequency-penalty 1.0 --top-k 1 --top-p 0.8 --temp 0.8 --img-start "<|vision_start|>" --img-end "<|vision_end|>" --img-content "<|image_pad|>" --img-width 384 --img-height 384 --video-start "<|vision_start|>" --video-end "<|vision_end|>" --video-content "<|video_pad|>"
```

Client example:

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

# Encode video frames (e.g., extracted from a video at a fixed FPS)
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

### POST `/v1/embeddings`: OpenAI-compatible embeddings API

See [OpenAI Embeddings API documentation](https://platform.openai.com/docs/api-reference/embeddings).

Examples:

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

## API Errors

Errors returned by `rkllm3-server` follow the same format as OAI: https://github.com/openai/openai-openapi

Error example:

```json
{
    "error": {
        "code": 401,
        "message": "Invalid API Key",
        "type": "authentication_error"
    }
}
```

In addition to the error types supported by OAI, we have custom types specific to rkllm3-server functionality:

```json
{
    "error": {
        "code": 501,
        "message": "This server does not support metrics endpoint.",
        "type": "not_supported_error"
    }
}
```

**When invalid syntax is received via the /v1/chat/completions endpoint**

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

Tie Word Embedding is a model optimization technique where the input embedding weights are tied to (shared with) the output layer. This eliminates the need for a separate embed file (`--embed`), saving disk space and memory.

> **Note**: This feature requires the RKNN model to be compiled with tie-word support. Standard models still require the `--embed` file.

### CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--tie-word-embedding` | `[0\|1]` | `0` | Enable tie word embedding mode, the model must support this feature. When set to `1`, no `--embed` file is required |

### Per-Request Override

If tie word embedding mode is enabled at initialization and an embed file is also provided, you can override the tie-word setting per API request via `extra_body`:

```json
{
  "messages": [{"role": "user", "content": "Hello!"}],
  "tie_word_embedding": 0
}
```

If `tie_word_embedding` is not specified in the request, the server-wide default (from CLI or `params.json`) is used.

### Example: Starting the Server

```bash
# Standard model (requires embed file)
./rkllm3-server -m qwen2.5-3b.rknn --vocab qwen2.5-3b.tokenizer.gguf --embed qwen2.5-3b.embed.bin --host 0.0.0.0 --port 8080

# Tie-word model (no embed file required)
./rkllm3-server -m Qwen3-VL-2B-llm-tie-word.rknn --weight Qwen3-VL-2B-llm-tie-word.weight \
  --vocab Qwen3-VL-2B-llm_quant.tokenizer.gguf --tie-word-embedding 1 --host 0.0.0.0 --port 8080 -c 4096
```

### Example: Per-Request with Python

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required"
)

# Override tie_word_embedding for this request
completion = client.chat.completions.create(
    model="Qwen3-VL-2B",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"tie_word_embedding": 0}
)
print(completion.choices[0].message.content)
```
