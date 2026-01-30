# Qwen2.5-VL 示例部署说明

本目录包含 `test.py`（单张图像 + LLM 推理示例）。文档说明如何准备模型文件、依赖环境，以及在开发板/本地进行推理的示例命令。

## 模型与文件
- LLM 模型（RKNN）：/xxx/Qwen2.5-VL-3B-llm.rknn
- Vision 模型（RKNN）：/xxx/Qwen2.5-VL-3B-vision.rknn
- LLM embed（二进制）：/xxx/Qwen2.5-VL-3B-llm.embed.bin
- Tokenizer 目录：examples/Qwen2.5VL/qwen2_5_VL_3B（或 HuggingFace 名称）

如果你使用不同路径，请在运行脚本时用命令行参数覆盖（见下文）。

## 1. 运行环境

- 推荐使用仓库根目录的 `requirements.txt` 来安装依赖。示例脚本使用 `transformers` 的 `AutoTokenizer`，以及 `numpy`、`opencv-python`、`ctypes` 等库。
- 请确保已经安装并可导入仓库中的 `rknn3lite` 包（例如使用 `pip install -e .` 在开发环境中安装）。

> ⚠️ 请确认目标设备（如 RK 系列开发板）上有对应运行库，并且 `.weight` 文件与 `.rknn` 匹配。

## 2. 准备模型与 tokenizer

1. 将 `Qwen2.5-VL-3B-llm.rknn`、`Qwen2.5-VL-3B-vision.rknn` 和配套的 `.weight` 文件放到设备或工作目录中。
2. 将 `Qwen2.5-VL-3B-llm.embed.bin` 放到可访问路径，并确认 embedding 的尺寸和脚本中 `VOCAB_SIZE` 一致。
3. 把 tokenizer 文件夹（例如 `qwen2_5_VL_3B`）放在 `examples/Qwen2.5VL/` 下或使用 HuggingFace 名称进行在线加载。

## 3. 运行示例脚本

脚本 `test.py` 支持以下命令行参数：

```bash
python test.py \
  --rknn_llm_path /xxx/Qwen2.5-VL-3B-llm.rknn \
  --rknn_vision_path /xxx/Qwen2.5-VL-3B-vision.rknn \
  --tokenizer_path ./qwen2_5_VL_3B \
  --embed_path /xxx/Qwen2.5-VL-3B-llm.embed.bin
```

脚本默认会尝试读取 `./demo.jpg` 作为输入图像。如果需要使用其他文件，请替换或通过脚本参数调整。

## 4. 输出与性能信息

单图demo示例输出片段：
```
这张图片展示了一个穿着宇航服的宇航员在月球表面。他手里拿着一瓶绿色啤酒，旁边有一个装满饮料的冷藏箱。背景是地球和星空，给人一种太空旅行的感觉。
```
注意：由于模型版本、输入图像或运行环境的差异，实际输出结果可能与上述示例略有不同。

## 5. 常见问题

- 模型加载失败：请检查路径是否正确，以及 `.weight` 文件是否与 `.rknn` 配套。
- Tokenizer 报错/编码问题：确认 `--tokenizer_path` 指向正确的 HuggingFace tokenizer 目录或有效名称。
- embed 回调失败：确认 `llm.embed.bin` 的 shape 与脚本中 `VOCAB_SIZE`、嵌入维度一致。
- 视觉模型预处理不正确：确认输入图像尺寸与模型所需尺寸一致（脚本中默认将图像 resize 到 392x392 并进行了 prune 处理，如需完整模型请移除 prune 步骤）。
