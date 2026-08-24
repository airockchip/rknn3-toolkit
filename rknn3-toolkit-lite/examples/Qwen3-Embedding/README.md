# Qwen3 Embedding RKNN3 Python 示例说明

本示例是一个基于 `rknn3-toolkit-lite` 的 Qwen3 Embedding RKNN3 Python 推理脚本，用于将输入文本编码为 embedding 向量，并将模型输出保存为 `.npy` 文件。

当前 `test.py` 对应的是rknn3-model-zoo工程的`examples/Qwen3_Embedding/cpp/main.cc` 的 Python 版本，使用 rknn3-toolkit-lite 的 LLM session 流程：

```text
load_rknn -> init_runtime -> query IO/config -> init_llm_session -> session_run
```

脚本会在 `output_callback` 收到 `RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED` 状态时捕获输出 tensor，将 fp16 输出内存转换为 float32，并保存为 numpy 文件。

## 适用场景

该脚本主要用于文本向量提取，适合以下场景：

- 文本 embedding 离线验证
- RKNN3 Qwen3 Embedding 模型精度或性能测试
- 向量检索、召回、排序等流程的 embedding 生成
- 对齐 C++ demo 的 Python 侧调试

注意：当前脚本不是多模态示例，不包含 Vision 模型、图片输入、图片预处理或 `<image>` 相关逻辑。

## 模型与依赖文件

运行时需要准备以下文件：

- Qwen3 Embedding RKNN 模型：`*.rknn`
- RKNN 权重文件：`*.weight`
- embedding 表文件：`*.embed.bin`
- tokenizer 目录，例如：`Qwen3-Embedding-0.6B/`

其中：

- `--rknn_path` 必须指定 `.rknn` 文件；
- `--weight_path` 可以不指定，脚本会默认使用 `--rknn_path` 同名的 `.weight` 文件；
- `--embed_path` 必须与当前模型和 tokenizer 的 vocab 对齐；
- `embed.bin` 会按 fp16 读取，并 reshape 成 `[vocab_size, embedding_dim]`；

## 运行环境

建议准备目标板端 Python 环境，并安装以下依赖：

- `numpy`
- `transformers`
- `rknn3-toolkit-lite`

脚本还依赖：

- Python 标准库：`argparse`、`ctypes`、`os`、`sys`、`time`
- 目标设备上的 RKNN / RKLLM 运行库

脚本默认设置了 HuggingFace 镜像源：

```python
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")
```

如果你的环境不需要镜像源，可以在脚本中删除或修改该设置。

## 命令行参数

`test.py` 当前支持以下参数：

```bash
python3 test.py \
  --rknn_path /path/to/Qwen3-Embedding.rknn \
  --weight_path /path/to/Qwen3-Embedding.weight \
  --tokenizer_path Qwen3-Embedding-0.6B/ \
  --embed_path /path/to/Qwen3-Embedding.embed.bin \
  --prompt "input text to embed" \
  --output_path output.npy \
  --target rk1820 \
  --core_mask 0xff \
  --max_new_tokens 1 \
  --output_index 0 \
  --logits_name output
```

参数说明：

| 参数 | 必填 | 默认值 | 说明 |
|---|---:|---|---|
| `--rknn_path` | 是 | 无 | Qwen3 Embedding `.rknn` 模型路径 |
| `--weight_path` | 否 | 由 `--rknn_path` 替换后缀得到 | `.weight` 权重文件路径 |
| `--tokenizer_path` | 是 | 无 | tokenizer 本地目录 |
| `--embed_path` | 是 | 无 | `*.embed.bin` embedding 表路径 |
| `--prompt` | 是 | 无 | 需要编码成向量的输入文本 |
| `--output_path` | 否 | `output.npy` | 输出 embedding 保存路径 |
| `--target` | 否 | `rk1820` | `init_runtime()` 使用的目标平台 |
| `--core_mask` | 否 | `0xff` | NPU core mask，例如 `0xff` |
| `--max_new_tokens` | 否 | `1` | 对齐 C++ demo 的 `MAX_NEW_TOKENS` |
| `--output_index` | 否 | `0` | 需要捕获的输出 tensor 索引 |
| `--logits_name` | 否 | `output` | LLM session 中传入的输出名 |
| `--load_embed_to_ram` | 否 | 关闭 | 默认使用 `np.memmap`，开启后将 `embed.bin` 全量加载到内存 |
| `--local_files_only` | 否 | 关闭 | 只从本地加载 tokenizer，不联网下载 |
| `--quiet` | 否 | 关闭 | 减少 output callback 调试打印 |


## 推理流程说明

### 1. 加载 tokenizer

脚本通过 `AutoTokenizer.from_pretrained()` 加载 tokenizer：

```python
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_path,
    trust_remote_code=args.trust_remote_code,
    local_files_only=args.local_files_only,
)
```

> **安全提示**：`trust_remote_code` 默认关闭。如果 tokenizer 需要执行自定义代码，请通过 `--trust_remote_code` 显式开启。

### 2. 加载 RKNN 模型

脚本以 LLM 模式创建 rknn3-toolkit-lite 对象：

```python
from rknn3lite.api import RKNN3Lite

rknn = RKNN3Lite(llm_mode=True, verbose=not args.quiet)
```

随后调用：

```python
rknn.load_rknn(args.rknn_path, weight_path)
```

如果没有传入 `--weight_path`，默认使用：

```python
from pathlib import Path
weight_path = str(Path(args.rknn_path).with_suffix(".weight"))
```

### 3. 初始化 runtime

默认初始化参数为：

```python
rknn.init_runtime(
    target=args.target,
    core_mask=core_mask,
)
```

其中 `--target` 默认为 `rk1820`，`--core_mask` 默认为 `0xff`。

### 4. 查询模型信息

脚本会查询：

- 输入输出数量：`RKNN3_QUERY_IN_OUT_NUM`
- 输出 tensor attr：`RKNN3_QUERY_OUTPUT_ATTR`
- LLM config：`RKNN3_QUERY_LLM_CONFIG`

并打印输出 tensor 信息和 LLM config 中的 `max_ctx_len`、`max_position_embeddings`、`vocab_size`。

### 5. 加载 embedding 表

脚本会读取 `--embed_path` 指定的 fp16 二进制文件，并根据 `vocab_size` 自动计算 `embedding_dim`：

```text
embedding_dim = total_fp16_elements / vocab_size
```

默认使用 `np.memmap` 方式映射文件，减少内存占用。如果希望一次性加载到内存，可以增加：

```bash
--load_embed_to_ram
```

### 6. 初始化 LLM session

脚本会注册以下回调：

- `result_callback`：主要用于兼容和调试，embedding 模型通常不依赖生成文本；
- `tokenizer_callback`：将输入 prompt 转成 token ids；
- `embed_callback`：根据 token ids 从 `embed.bin` 中取出输入 embedding；
- `output_callback`：在 prefill 完成时捕获输出 tensor。

LLM session 中的主要参数包括：

```python
{
    "max_new_tokens": args.max_new_tokens,
    "top_k": 1,
    "top_p": 0.0,
    "temperature": 1.0,
    "repeat_penalty": 1.0,
    "vocab_size": vocab_size,
    "special_bos_id": tokenizer.bos_token_id,
    "special_eos_id": tokenizer.eos_token_id,
    "max_context_len": llm_config.max_ctx_len,
    "keep_history": 0,
    "logits_name": args.logits_name,
}
```

### 7. 执行推理并保存输出

脚本调用：

```python
rknn.session_run(prompt=args.prompt, keep_history=False, enable_thinking=False)
```

如果当前 `rknn3-toolkit-lite` 版本不支持 `enable_thinking` 或 `keep_history` 参数，脚本会自动降级尝试兼容调用。

推理完成后，脚本会从 `model_outputs[0]` 取出捕获的输出，并保存到：

```bash
output.npy
```

如果 `--output_path` 没有以 `.npy` 结尾，脚本会自动追加 `.npy` 后缀。

## 输出内容

运行成功后，终端通常会看到：

- tokenizer 加载日志
- RKNN 模型加载日志
- runtime 初始化日志
- 输入输出 tensor 信息
- LLM config 信息
- embedding 表信息，包括 `vocab_size` 和 `embedding_dim`
- output callback 捕获到的输出 tensor 信息
- Prefill / Generate 性能统计
- 保存的 `.npy` 文件路径、shape 和 dtype

示例输出片段：

```text
embedding: vocab_size=151936, embedding_dim=1024, file=/path/to/model.embed.bin
capture output tensor[0]: output
Saved embedding: output.npy, shape=(1024,), dtype=float32
done
```

实际 `shape` 以模型输出 tensor 为准。
