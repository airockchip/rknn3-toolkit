# Gemma4 LLM / VLM RKNN 推理示例说明

本目录提供 Gemma4 在 RKNN3Lite 环境下的部署验证示例，包含文本推理脚本 `test.py`、图文多模态推理脚本 `test_vlm.py` 和视频多模态推理脚本 `test_vlm_video.py`。

- `test.py`：用于验证 Gemma4 LLM RKNN 模型的纯文本推理流程。
- `test_vlm.py`：用于验证 Gemma4 Vision + LLM RKNN 模型的图文多模态推理流程。
- `test_vlm_video.py`：用于验证 Gemma4 Vision + LLM RKNN 模型的视频多模态推理流程。

脚本会加载 RKNN 模型、配套外部权重、token embedding、`per_layer_inputs` embedding，并在模型需要时从 `rope_caches.safetensors` 中读取 RoPE cache 外部输入。


## 1. 文件说明

### 1.1 LLM 文本推理文件

`test.py` 默认使用以下文件：

| 文件/目录 | 默认值 | 说明 |
|---|---|---|
| LLM RKNN 模型 | `gemma-4-e2b-it.rknn` | Gemma4 LLM RKNN 模型文件 |
| LLM 权重文件 | `gemma-4-e2b-it.weight` | 由 `--rknn_path` 自动推导得到，无需单独传参 |
| Tokenizer 目录 | `gemma-4-e2b-it` | HuggingFace tokenizer 目录或 tokenizer 名称 |
| Token embedding | `gemma-4-e2b-it.embed.bin` | token embedding 二进制文件，`float16` 格式 |
| Per-layer embedding | `gemma-4-e2b-it_per_layer_inputs.embed.bin` | `per_layer_inputs` 对应的 embedding 二进制文件，`float16` 格式 |
| RoPE cache 文件 | `rope_caches.safetensors` | 当模型配置 `rope_cache_host_storage != 0` 时必须提供 |

### 1.2 VLM 图文推理文件

`test_vlm.py` 默认使用以下文件：

| 文件/目录 | 默认值 | 说明 |
|---|---|---|
| Vision RKNN 模型 | `gemma-4-vision.rknn` | Gemma4 vision encoder RKNN 模型文件 |
| Vision 权重文件 | `gemma-4-vision.weight` | 由 `--rknn_vision_path` 自动推导得到，无需单独传参 |
| LLM RKNN 模型 | `gemma-4-e2b-it.rknn` | Gemma4 LLM RKNN 模型文件 |
| LLM 权重文件 | `gemma-4-e2b-it.weight` | 由 `--rknn_llm_path` 自动推导得到，无需单独传参 |
| Tokenizer 目录 | `gemma-4-e2b-it` | HuggingFace tokenizer 目录或 tokenizer 名称 |
| Token embedding | `gemma-4-e2b-it.embed.bin` | token embedding 二进制文件，`float16` 格式 |
| Per-layer embedding | `gemma-4-e2b-it_per_layer_inputs.embed.bin` | `per_layer_inputs` 对应的 embedding 二进制文件，`float16` 格式 |
| RoPE cache 文件 | `rope_caches.safetensors` | 当模型配置 `rope_cache_host_storage != 0` 时必须提供 |
| 默认测试图片 | `demo.jpg` | VLM 输入图片 |

### 1.3 VLM 视频推理文件

`test_vlm_video.py` 默认使用以下文件：

| 文件/目录 | 默认值 | 说明 |
|---|---|---|
| Vision RKNN 模型 | `gemma-4-vision.rknn` | Gemma4 vision encoder RKNN 模型文件 |
| Vision 权重文件 | `gemma-4-vision.weight` | 由 `--rknn_vision_path` 自动推导得到，无需单独传参 |
| LLM RKNN 模型 | `gemma-4-e2b-it.rknn` | Gemma4 LLM RKNN 模型文件 |
| LLM 权重文件 | `gemma-4-e2b-it.weight` | 由 `--rknn_llm_path` 自动推导得到，无需单独传参 |
| Tokenizer 目录 | `gemma-4-e2b-it` | HuggingFace tokenizer 目录或 tokenizer 名称 |
| Token embedding | `gemma-4-e2b-it.embed.bin` | token embedding 二进制文件，`float16` 格式 |
| Per-layer embedding | `gemma-4-e2b-it_per_layer_inputs.embed.bin` | `per_layer_inputs` 对应的 embedding 二进制文件，`float16` 格式 |
| RoPE cache 文件 | `rope_caches.safetensors` | 当模型配置 `rope_cache_host_storage != 0` 时必须提供 |
| 默认测试视频 | `demo.mp4` | VLM 输入视频 |

建议目录结构如下：

```text
.
├── README.md
├── test.py
├── test_vlm.py
├── test_vlm_video.py
├── gemma-4-e2b-it.rknn
├── gemma-4-e2b-it.weight
├── gemma-4-vision.rknn
├── gemma-4-vision.weight
├── gemma-4-e2b-it.embed.bin
├── gemma-4-e2b-it_per_layer_inputs.embed.bin
├── rope_caches.safetensors
├── demo.jpg
├── demo.mp4
└── gemma-4-e2b-it/
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```

> 注意：`.rknn`、`.weight`、`embed.bin`、`per_layer_inputs.embed.bin`、`rope_caches.safetensors` 和 tokenizer 必须来自同一套模型配置。若文件不匹配，可能出现 vocab size 不一致、embedding 维度不匹配、RoPE cache 读取异常或推理结果错误。

## 2. 运行环境

请先确认运行环境已经安装以下依赖：

- Python 3.x
- `numpy`
- `transformers >= 5.5.0`
- `rknn3lite`
- 目标设备对应的 RKNN3 运行库
- `opencv-python`，`test_vlm.py` 和 `test_vlm_video.py` 需要

脚本中会检查 `transformers>=5.5.0`：

```python
require_version(
    "transformers>=5.5.0",
    "The remote code requires transformers>=5.5.0, please upgrade: pip install -U transformers"
)
```

如版本过低，可执行：

```bash
pip install -U transformers
```

VLM 示例需要读取和预处理图片，如环境中没有 OpenCV，可执行：

```bash
pip install opencv-python
```

## 3. LLM 文本推理

### 3.1 使用默认文件运行

当模型文件、embedding 文件、RoPE cache 文件和 tokenizer 均放在当前目录，并且文件名与默认配置一致时，可直接运行：

```bash
python test.py
```

### 3.2 显式指定路径运行

```bash
python test.py \
  --rknn_path ./gemma-4-e2b-it.rknn \
  --tokenizer_path ./gemma-4-e2b-it \
  --embed_path ./gemma-4-e2b-it.embed.bin \
  --per_layer_embed_path ./gemma-4-e2b-it_per_layer_inputs.embed.bin \
  --safetensors_path ./rope_caches.safetensors \
  --max_context_len 1024 \
  --max_new_tokens 1024 \
  --core_mask 0xff
```

当前 `test.py` 不需要、也不支持传入 `--weight_path`。脚本加载模型时会自动使用：

```python
rknn.load_rknn(args.rknn_path, args.rknn_path.replace(".rknn", ".weight"))
```

因此，如果指定：

```bash
--rknn_path ./gemma-4-e2b-it.rknn
```

则权重文件必须为：

```text
./gemma-4-e2b-it.weight
```

## 4. VLM 图文推理

### 4.1 使用默认文件运行

当 vision 模型、LLM 模型、embedding 文件、RoPE cache 文件、tokenizer 和测试图片均放在当前目录，并且文件名与默认配置一致时，可直接运行：

```bash
python test_vlm.py
```

默认输入图片为：

```text
demo.jpg
```

默认 prompt 为：

```text
<image>请描述这张图片。
```

### 4.2 显式指定路径运行

```bash
python test_vlm.py \
  --rknn_vision_path ./gemma-4-vision.rknn \
  --rknn_llm_path ./gemma-4-e2b-it.rknn \
  --tokenizer_path ./gemma-4-e2b-it \
  --embed_path ./gemma-4-e2b-it.embed.bin \
  --per_layer_embed_path ./gemma-4-e2b-it_per_layer_inputs.embed.bin \
  --safetensors_path ./rope_caches.safetensors \
  --image_path ./demo.jpg \
  --prompt '<image>请描述这张图片。' \
  --max_context_len 1024 \
  --max_new_tokens 1024 \
  --vision_core_mask 0xff \
  --llm_core_mask 0xff
```

如需保存 vision encoder 的输出，可使用：

```bash
python test_vlm.py \
  --rknn_vision_path ./gemma-4-vision.rknn \
  --rknn_llm_path ./gemma-4-e2b-it.rknn \
  --image_path ./demo.jpg \
  --dump_vision_output ./vision_output.npy
```

`test_vlm.py` 同样不需要、也不支持传入独立的 vision/LLM weight 参数。脚本会分别自动使用：

```python
rknn_vision.load_rknn(args.rknn_vision_path, args.rknn_vision_path.replace(".rknn", ".weight"))
rknn_llm.load_rknn(args.rknn_llm_path, args.rknn_llm_path.replace(".rknn", ".weight"))
```

因此，指定以下模型时：

```text
gemma-4-vision.rknn
gemma-4-e2b-it.rknn
```

对应权重文件必须为：

```text
gemma-4-vision.weight
gemma-4-e2b-it.weight
```

## 5. VLM 视频推理

### 5.1 视频抽帧机制

`test_vlm_video.py` 会从输入视频中自动抽取帧：

1. 读取视频的总帧数和 FPS。
2. 根据视频时长（秒），每秒抽取 1 帧，即帧数 = `ceil(duration)`。
3. 抽帧数受 `MAX_FRAMES`（默认 8）限制，超过的视频将被截断。
4. 按时间戳依次抽取帧，并在 prompt 中为每帧添加对应的时间戳前缀。

例如一个 10 秒的视频，最终会抽取 8 帧（达到 `MAX_FRAMES` 上限），prompt 中将包含：

```text
00:00 <image> 00:01 <image> ... 00:07 <image>
```

### 5.2 使用默认文件运行

当 vision 模型、LLM 模型、embedding 文件、RoPE cache 文件、tokenizer 和测试视频均放在当前目录，并且文件名与默认配置一致时，可直接运行：

```bash
python test_vlm_video.py
```

默认输入视频为：

```text
demo.mp4
```

默认 prompt 为：

```text
<video>请用中文描述这个视频。
```

### 5.3 显式指定路径运行

```bash
python test_vlm_video.py \
  --rknn_vision_path ./gemma-4-vision.rknn \
  --rknn_llm_path ./gemma-4-e2b-it.rknn \
  --tokenizer_path ./gemma-4-e2b-it \
  --embed_path ./gemma-4-e2b-it.embed.bin \
  --per_layer_embed_path ./gemma-4-e2b-it_per_layer_inputs.embed.bin \
  --safetensors_path ./rope_caches.safetensors \
  --video_path ./demo.mp4 \
  --prompt '<video>请用中文描述这个视频。' \
  --max_new_tokens 1024 \
  --vision_core_mask 0xff \
  --llm_core_mask 0xff
```

如需保存 vision encoder 的输出，可使用：

```bash
python test_vlm_video.py \
  --rknn_vision_path ./gemma-4-vision.rknn \
  --rknn_llm_path ./gemma-4-e2b-it.rknn \
  --video_path ./demo.mp4 \
  --dump_vision_output ./vision_output.npy
```

`test_vlm_video.py` 同样不需要、也不支持传入独立的 vision/LLM weight 参数。脚本会分别自动使用：

```python
rknn_vision.load_rknn(args.rknn_vision_path, args.rknn_vision_path.replace(".rknn", ".weight"))
rknn_llm.load_rknn(args.rknn_llm_path, args.rknn_llm_path.replace(".rknn", ".weight"))
```

因此，指定以下模型时：

```text
gemma-4-vision.rknn
gemma-4-e2b-it.rknn
```

对应权重文件必须为：

```text
gemma-4-vision.weight
gemma-4-e2b-it.weight
```

### 5.4 Prompt 中 `<video>` 占位符

视频推理使用 `<video>` 作为占位符，脚本会自动将其展开为带时间戳的逐帧 `<image>` 序列。格式为：

```text
MM:SS <image> MM:SS <image> ...
```

用户只需在 prompt 中使用 `<video>`，无需手动拼接帧序列。例如：

```bash
python test_vlm_video.py \
  --video_path ./demo.mp4 \
  --prompt '<video>请总结这个视频的内容。'
```

### 6.1 `test.py` 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--rknn_path` | `gemma-4-e2b-it.rknn` | LLM RKNN 模型路径。权重文件由该路径自动推导 |
| `--tokenizer_path` | `gemma-4-e2b-it` | HuggingFace tokenizer 目录或名称 |
| `--embed_path` | `gemma-4-e2b-it.embed.bin` | token embedding 文件路径 |
| `--per_layer_embed_path` | `gemma-4-e2b-it_per_layer_inputs.embed.bin` | `per_layer_inputs` embedding 文件路径 |
| `--safetensors_path` | `rope_caches.safetensors` | RoPE cache 文件路径 |
| `--max_context_len` | `1024` | 最大上下文长度检查值，仅用于和模型配置对比 |
| `--max_new_tokens` | `1024` | 单轮最大生成 token 数 |
| `--core_mask` | `0xff` | LLM 推理使用的 NPU core mask |

### 6.2 `test_vlm.py` 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--rknn_vision_path` | `gemma-4-vision.rknn` | Vision RKNN 模型路径。权重文件由该路径自动推导 |
| `--rknn_llm_path` | `gemma-4-e2b-it.rknn` | LLM RKNN 模型路径。权重文件由该路径自动推导 |
| `--tokenizer_path` | `gemma-4-e2b-it` | HuggingFace tokenizer 目录或名称 |
| `--embed_path` | `gemma-4-e2b-it.embed.bin` | token embedding 文件路径 |
| `--per_layer_embed_path` | `gemma-4-e2b-it_per_layer_inputs.embed.bin` | `per_layer_inputs` embedding 文件路径 |
| `--safetensors_path` | `rope_caches.safetensors` | RoPE cache 文件路径 |
| `--image_path` | `demo.jpg` | 输入图片路径 |
| `--prompt` | `<image>\n请描述这张图片。` | 用户输入内容，chat template 会由脚本自动添加 |
| `--max_context_len` | `1024` | 最大上下文长度检查值，仅用于和模型配置对比 |
| `--max_new_tokens` | `1024` | 单轮最大生成 token 数 |
| `--llm_core_mask` | `0xff` | LLM 推理使用的 NPU core mask |
| `--vision_core_mask` | `0xff` | Vision 推理使用的 NPU core mask |
| `--dump_vision_output` | 空 | 可选参数，用于将 vision 输出保存为 `.npy` 文件 |

### 6.3 `test_vlm_video.py` 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--rknn_vision_path` | `gemma-4-vision.rknn` | Vision RKNN 模型路径。权重文件由该路径自动推导 |
| `--rknn_llm_path` | `gemma-4-e2b-it.rknn` | LLM RKNN 模型路径。权重文件由该路径自动推导 |
| `--tokenizer_path` | `gemma-4-e2b-it` | HuggingFace tokenizer 目录或名称 |
| `--embed_path` | `gemma-4-e2b-it.embed.bin` | token embedding 文件路径 |
| `--per_layer_embed_path` | `gemma-4-e2b-it_per_layer_inputs.embed.bin` | `per_layer_inputs` embedding 文件路径 |
| `--safetensors_path` | `rope_caches.safetensors` | RoPE cache 文件路径 |
| `--video_path` | `demo.mp4` | 输入视频路径 |
| `--prompt` | `<video>请用中文描述这个视频。` | 用户输入内容，请使用 `<video>` 占位符，chat template 会由脚本自动添加 |
| `--max_new_tokens` | `1024` | 单轮最大生成 token 数 |
| `--llm_core_mask` | `0xff` | LLM 推理使用的 NPU core mask |
| `--vision_core_mask` | `0xff` | Vision 推理使用的 NPU core mask |
| `--dump_vision_output` | 空 | 可选参数，用于将 vision 输出保存为 `.npy` 文件 |

> 注意：`test_vlm_video.py` 无 `--max_context_len` 参数，脚本会直接使用 `llm_config.max_ctx_len`。视频帧数上限由脚本内 `MAX_FRAMES`（默认 8）控制。

## 7. 推理流程说明

### 7.1 LLM 推理流程

`test.py` 的主要流程如下：

1. 加载 tokenizer。
2. 创建 `RKNN3Lite(llm_mode=True, verbose=True)` 对象。
3. 通过 `load_rknn(args.rknn_path, args.rknn_path.replace(".rknn", ".weight"))` 加载 LLM 模型和权重。
4. 调用 `init_runtime(target='rk1820', core_mask=args.core_mask)` 初始化运行环境。
5. 查询 `RKNN3_QUERY_LLM_CONFIG` 和 `RKNN3_QUERY_IN_OUT_NUM`。
6. 根据 `llm_config.vocab_size` 加载并 reshape：
   - `gemma-4-e2b-it.embed.bin`
   - `gemma-4-e2b-it_per_layer_inputs.embed.bin`
7. 查询模型输入，自动收集外部输入索引：
   - `per_layer_inputs`
   - `rope_cos_cache_*`
   - `rope_sin_cache_*`
8. 当模型配置 `rope_cache_host_storage != 0` 时，加载 `rope_caches.safetensors`。
9. 设置 LLM 参数、callback 函数和 chat template。
10. 调用 `rknn.session_run()` 执行文本推理。
11. 每轮推理后调用 `clear_kvcache()` 清理 KV cache。
12. 打印 Prefill 和 Generate 阶段性能统计。

### 7.2 VLM 图文推理流程

`test_vlm.py` 的主要流程如下：

1. 分别创建 vision RKNN 对象和 LLM RKNN 对象。
2. 加载 vision RKNN 模型及其同名 `.weight` 权重文件。
3. 加载 LLM RKNN 模型及其同名 `.weight` 权重文件。
4. 初始化 LLM runtime，查询 LLM 配置和输入输出信息。
5. 加载 token embedding、`per_layer_inputs` embedding 和 RoPE cache。
6. 初始化 LLM session，并设置 chat template。
7. 初始化 vision runtime。
8. 读取输入图片，根据 vision 模型输入 shape 自动 resize，并按模型输入 dtype 转换数据。
9. 执行 vision encoder 推理，得到图像 embedding。
10. 将 vision 输出整理为 `float16` 连续内存，并通过 `RKNN3Image.image_embed` 注入 LLM。
11. 使用 `RKNN3Image` 设置图像 token 相关字段：
    - `image_start = b"<|image>"`
    - `image_content = b"<|image|>"`
    - `image_end = b"<image|>"`
12. 调用 `rknn_llm.session_run(inputs=[llm_input], prompt=args.prompt, ...)` 执行图文推理。
13. 推理结束后清理 KV cache，并打印 vision、Prefill 和 Generate 阶段性能统计。

### 7.3 VLM 视频推理流程

`test_vlm_video.py` 的主要流程如下：

1. 分别创建 vision RKNN 对象和 LLM RKNN 对象。
2. 加载 vision RKNN 模型及其同名 `.weight` 权重文件。
3. 加载 LLM RKNN 模型及其同名 `.weight` 权重文件。
4. 初始化 LLM runtime，查询 LLM 配置和输入输出信息。
5. 加载 token embedding、`per_layer_inputs` embedding 和 RoPE cache。
6. 初始化 LLM session，并设置 chat template。
7. 初始化 vision runtime。
8. 读取输入视频，按每秒 1 帧抽取，最多 `MAX_FRAMES`（默认 8）帧，记录每帧的时间戳。
9. 将每帧图片根据 vision 模型输入 shape 自动 resize，并按模型输入 dtype 转换。
10. 逐帧执行 vision encoder 推理，将所有帧的 vision 输出沿帧维拼接为 `[N, token_per_frame, dim]`。
11. 将拼接后的 vision 输出整理为 `float16` 连续内存，并通过 `RKNN3Image.image_embed` 注入 LLM。
12. 使用 `RKNN3Image` 设置图像 token 相关字段：
    - `image_start = b"<|image>"`
    - `image_content = b"<|image|>"`
    - `image_end = b"<image|>"`
13. 根据 `n_image`（帧数）自动将 `<video>` 占位符展开为带时间戳的逐帧 `<image>` 序列，例如 `00:00 <image> 00:01 <image> ...`。
14. 调用 `rknn_llm.session_run(inputs=[llm_input], prompt=prompt, ...)` 执行多帧图文推理。
15. 推理结束后清理 KV cache，并打印 vision、Prefill 和 Generate 阶段性能统计（含 per-frame FPS）。

## 8. Callback 说明

三个脚本均使用以下 LLM callback：

| Callback | 作用 |
|---|---|
| `result_callback` | 接收模型输出 token，并通过 tokenizer 增量解码打印流式结果 |
| `tokenizer_callback` | 将输入文本编码为 token id，`add_special_tokens=False` |
| `embed_callback` | 根据 token id 从 `embed.bin` 中拷贝 token embedding |
| `input_callback` | 向 `per_layer_inputs` 和 RoPE cache 外部输入填充数据 |
| `output_callback` | 当前示例中保留为空实现 |

其中 `input_callback` 是 Gemma4 推理的关键部分：

- 当输入名为 `per_layer_inputs` 时，脚本会根据当前 token id 从 `per_layer_embeds_data` 中取出对应 embedding。
- 当输入名为 `rope_cos_cache_*` 或 `rope_sin_cache_*` 时，脚本会从 `rope_caches.safetensors` 的 mmap 地址中按位置拷贝 RoPE cache。
- RoPE cache 拷贝使用 `ctypes.memmove`，避免 `mmap[...]` 切片产生额外的 Python bytes 临时拷贝。
- VLM 场景下，图像占位 token 对应的 `per_layer_inputs` 会做安全处理，真正的图像 embedding 由 `RKNN3Image.image_embed` 注入。

> 当前 `rknn3lite.api` 必须支持 `LLMInputCallback`。如果不支持，脚本会提示：`Current rknn3lite.api has no LLMInputCallback, please use the RKNN3Lite package that supports Gemma4 input callback`。

## 9. Chat template 与 prompt 说明

脚本中统一使用以下 chat template 配置：

```python
system_prompt = ""
prompt_prefix = "<bos><|turn>user\n"
prompt_postfix = "<turn|>\n<|turn>model\n"
```

调用脚本时，`--prompt` 只需要传入用户实际输入内容，不需要手动添加 `<bos>`、`<|turn>user` 或 `<|turn>model`。

LLM 文本推理中，脚本内置了以下测试 prompt：

```python
prompts = ["请解释一下相对论的基本概念？", "你是谁？", "介绍一下LLM模型的工作原理。"]
```

VLM 图文推理中，prompt 应包含 `<image>` 占位符，例如：

```bash
python test_vlm.py \
  --image_path ./demo.jpg \
  --prompt $'<image>\n请描述这张图片。'
```

VLM 视频推理中，prompt 应包含 `<video>` 占位符，脚本会自动将其展开为带时间戳的逐帧 `<image>` 序列，例如：

```bash
python test_vlm_video.py \
  --video_path ./demo.mp4 \
  --prompt '<video>请总结这个视频的内容。'
```

## 10. 常见问题

### 10.1 找不到 `.weight` 文件

当前脚本不会通过命令行单独传入 weight 文件，而是由 `.rknn` 路径自动推导。例如：

```text
模型：./model/gemma-4-e2b-it.rknn
权重：./model/gemma-4-e2b-it.weight
```

如果 `.weight` 文件名或目录不一致，请重命名或移动到与 `.rknn` 文件相同的位置。

### 10.2 tokenizer vocab size 与模型配置不一致

脚本会优先使用 `llm_config.vocab_size` 作为 embedding reshape 的词表大小。如果 tokenizer 与模型不是同一套导出配置，可能出现警告或推理异常。发布和部署时应确保 tokenizer、LLM 模型、embedding 文件来自同一版本。

### 10.3 `max_context_len` 与模型配置不一致

`--max_context_len` 仅用于和 `llm_config.max_ctx_len` 做一致性检查：

- 如果传入值小于模型配置，脚本会给出 warning，并建议使用模型配置值。
- 如果传入值大于模型配置，脚本会报错退出。

正式运行时建议将 `--max_context_len` 设置为查询到的 `llm_config.max_ctx_len`。

### 10.4 VLM 图片无法读取

如果出现 `Read image failed`，请检查：

- `--image_path` 是否正确。
- 图片文件是否存在。
- 当前环境是否安装 OpenCV。
- 图片格式是否可被 `cv2.imread()` 正常读取。

### 10.5 VLM 输出异常或图片没有生效

请重点检查：

- prompt 中是否包含 `<image>` 占位符。
- vision 模型、LLM 模型和 tokenizer 是否来自同一套 Gemma4 VLM 配置。
- vision 输出维度是否符合 LLM 侧预期。
- `image_start`、`image_content`、`image_end` 是否与当前模型导出配置一致。

### 10.6 视频无法读取

如果出现 `Open video failed`，请检查：

- `--video_path` 是否正确。
- 视频文件是否存在。
- 当前环境是否安装 OpenCV（`pip install opencv-python`）。
- 视频格式是否可被 `cv2.VideoCapture` 正常打开。

### 10.7 视频帧数异常或 prompt 中帧数与预期不符

视频抽帧规则为每秒 1 帧，最多 `MAX_FRAMES`（默认 8）帧。如果视频较长但只抽到少量帧，请检查：

- 视频的实际时长是否正常（脚本会打印 `duration` 和 `num_frames`）。
- 如需调整最大帧数，可修改脚本中的 `MAX_FRAMES` 变量。
- FPS 无法读取时，脚本默认按 30 FPS 计算。

### 10.8 视频推理输出异常或 `<video>` 没有生效

请重点检查：

- prompt 中是否包含 `<video>` 占位符（注意是小写）。
- vision 模型、LLM 模型和 tokenizer 是否来自同一套 Gemma4 VLM 配置。
- 视频帧数是否超过 `MAX_FRAMES` 导致被截断。
- 视频的 vision 输出维度与 LLM 侧预期是否匹配。
- `image_start`、`image_content`、`image_end` 是否与当前模型导出配置一致。
