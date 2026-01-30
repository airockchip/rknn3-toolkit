# Qwen3 示例部署说明

本目录包含 `test.py`（LLM 推理示例）和 `test_embed.py`（使用预计算 embed 的示例）。文档说明如何准备模型文件、依赖环境，以及在开发板/本地进行推理的示例命令。

## 模型与文件
- LLM 模型（RKNN）：/xxx/Qwen3-1.7B_quant.rknn
- LLM 权重（WEIGHT）：/xxx/Qwen3-1.7B_quant.weight
- LLM embed（二进制）：/xxx/Qwen3-1.7B.embed.bin
- Tokenizer 目录：examples/Qwen3/Qwen3-1.7B-Base （或 HuggingFace 名称）

如果你使用不同路径，请在运行脚本时用命令行参数覆盖（见下文）。

## 1. 运行环境

- 推荐使用仓库根目录的 `requirements.txt` 来安装依赖。示例脚本使用 `transformers` 的 `AutoTokenizer`，以及 `numpy`、`ctypes` 等标准库。
- 请确保已经安装并可导入仓库中的 `rknn3lite` 包。

> ⚠️ 请确认目标设备（如 RK 系列开发板）上有对应运行库，并且 `.weight` 文件与 `.rknn` 匹配。

## 2. 准备模型与 tokenizer

1. 将 `Qwen3-1.7B_quant.rknn` 和配套的 `Qwen3-1.7B_quant.weight` 放到设备或工作目录中。
2. 将 `Qwen3-1.7B.embed.bin` 放到可访问路径，并确认 embedding 的尺寸和脚本中 `VOCAB_SIZE` 一致。
3. 把 tokenizer 文件夹（例如 `Qwen3-1.7B-Base`）放在 `examples/Qwen3/` 下
## 3. 运行示例脚本

两个示例脚本都有命令行参数来指定模型、tokenizer、embed 路径：

- `test.py`：使用 tokenizer 回调（脚本内部会调用 tokenizer 将文本转换为 token）。
- `test_embed.py`：预先加载 `embed.bin` 并直接传入 embedding（适用于支持 embed 模式的模型和配置）。

基本运行示例：

```bash
# 运行使用 tokenizer 的示例
python test.py \
  --rknn_path /xxx/Qwen3-1.7B_quant.rknn \
  --weight_path /xxx/Qwen3-1.7B_quant.weight \
  --tokenizer_path ./Qwen3-1.7B-Base \
  --embed_path /xxx/Qwen3-1.7B.embed.bin

# 运行使用预计算 embed 的示例
python test_embed.py \
  --rknn_llm_path /xxx/Qwen3-1.7B_quant.rknn \
  --tokenizer_path ./Qwen3-1.7B-Base \
  --embed_path /xxx/Qwen3-1.7B.embed.bin
```

脚本 `test.py` 默认会读取内置 prompts 并打印流式 token 输出；`test_embed.py` 演示如何使用外部 embedding 数据进行推理。

## 4. 输出与性能信息

demo示例输出片段：
```
好的，用户让我解释相对论的基本概念。首先，我需要回忆一下相对论的基本内容，包括狭义相对论和广义相对论。用户可能对这两个理论不太熟悉，所以需要分清楚它们的区别和联系。

先从狭义相对论开始。我记得它由爱因斯坦在1905年提出，主要涉及时间和空间的相对性。这里要提到相对速度和光速不变原理，以及同时性的相对性。可能需要举例子，比如钟慢效应和长度收缩，但要注意用简单的话解释，避免太数学化。

然后是广义相对论，1915年提出，主要关于引力和时空弯曲。需要解释引力不是力，而是时空弯曲的结果，比如地球的重力是地球质量弯曲了周围的时空。可能还要提到引力透镜和黑洞的概念，但要确保用户理解这些概念的基础。

用户可能对这些概念感到困惑，所以需要避免使用过于专业的术语，或者用比喻来解释。比如，把时空比作一张弹性网，物体的重量会弯曲它，其他物体沿着弯曲的路径运动。
...
```
注意：由于模型版本、输入图像或运行环境的差异，实际输出结果可能与上述示例略有不同。

## 5. 常见问题

- 模型加载失败：请检查路径是否正确，以及 `.weight` 文件是否与 `.rknn` 配套。
- Tokenizer 报错/编码问题：确认 `--tokenizer_path` 指向正确的 HuggingFace tokenizer 目录或有效名称。
- embed 回调失败：确认 `llm.embed.bin` 的 shape 与脚本中 `VOCAB_SIZE`、嵌入维度一致。
- 流式输出出现乱码：可能是因为 token 解码时遇到不完整 UTF-8 字符，脚本中已包含简单的安全解码处理，在出现问题时可增加缓冲或调整 tokenizer 设置。
