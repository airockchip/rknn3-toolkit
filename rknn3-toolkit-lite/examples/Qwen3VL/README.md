# Qwen3-VL 示例部署说明

本目录包含 `test.py`（单张图像 + LLM 推理示例）和 `test_video.py`（多帧视频 + LLM 推理示例）。文档说明如何准备模型文件、依赖环境，以及在开发板/本地进行推理的示例命令。

## 模型与文件
- LLM 模型（RKNN）：/xxx/Qwen3-VL-2B-llm.rknn
- Vision 模型（RKNN）：/xxx/Qwen3-VL-2B-vision.rknn
- LLM embed：/xxx/Qwen3-VL-2B-llm.embed.bin
- Tokenizer 目录：examples/Qwen3VL/qwen3_VL_2B

如果你使用不同路径，请在运行脚本时用命令行参数覆盖（见下文）。

## 1. 运行环境

- 推荐使用仓库根目录的 `requirements.txt`，但部分 transformers 版本可能需要指定镜像或特定分支。示例脚本使用 `transformers` 的 `AutoTokenizer`。
- 请确保已安装并更新 `rknn3-toolkit-lite`（或 rknn3-toolkit 的相应 Python 包），并能在目标设备上加载 RKNN 模型。

> ⚠️ 请确保 `rknn3-toolkit-lite` 已是最新版本，且目标开发板上有对应运行库。

## 2. 导出/准备 RKNN 模型

若你已有 RKNN 文件（`.rknn`、`.weight`、`.embed.bin`、tokenizer 目录），可跳过导出步骤。导出流程视模型源（HuggingFace / ModelScope）而定，这里给出常见分步：

1. 导出 Vision 子模块为 ONNX（见仓库相应导出脚本），再使用 RKNN 工具链转换为 `.rknn`。
2. 导出 LLM 子模块为 ONNX，并生成 `.rknn` 与 `.weight`，同时导出 `llm.embed.bin`（embedding 文件）以及 tokenizer（可为 GGUF/Tokenizer 目录）。

导出完成后，将模型文件放到设备或路径下。

## 3. 运行示例脚本

两个示例脚本都在 `examples/Qwen3VL/` 下，提供命令行参数来指定模型、tokenizer、embed 路径。

- `test.py`：单图像 + Vision -> LLM 推理流程。
- `test_video.py`：从视频中抽帧，构造多帧特征后送入 LLM 的视频接口进行推理。

基本运行示例：

```bash
# 运行单图像示例（在开发板或本地）
python test.py \
  --rknn_llm_path /xxx/Qwen3-VL-2B-llm.rknn \
  --rknn_vision_path /xxx/Qwen3-VL-2B-vision.rknn \
  --tokenizer_path ./qwen3_VL_2B \
  --embed_path /xxx/Qwen3-VL-2B-llm.embed.bin

# 运行视频示例（指定 video 文件位置）
python test_video.py \
  --rknn_llm_path /xxx/Qwen3-VL-2B-llm.rknn \
  --rknn_vision_path /xxx/Qwen3-VL-2B-vision.rknn \
  --tokenizer_path ./qwen3_VL_2B \
  --embed_path /xxx/Qwen3-VL-2B-llm.embed.bin
```

脚本默认会尝试读取 `./demo.jpg`（`test.py`）或 `test.mp4`（`test_video.py`）作为输入。如果需要使用其他文件，请在脚本内或通过替换文件名来调整。

## 4. 运行日志
单图demo示例输出片段：
```
这张图片展示了一个穿着宇航服的宇航员在月球表面。他手里拿着一瓶绿色啤酒，旁边有一个装满饮料的冷藏箱。背景是地球和星空，给人一种太空旅行的感觉。
```

视频demo示例输出片段：
```
根据您提供的视频片段，这是一段关于Rockchip（瑞芯微）公司及其相关活动的视频。以下是详细的介绍：

1. **Rockchip品牌介绍**：
   - Rockchip（瑞芯微）是一家专注于嵌入式芯片和系统解决方案的公司，其产品广泛应用于智能设备、物联网、汽车电子等领域。
   - 该公司以高性能、低功耗的芯片著称，尤其在移动计算和嵌入式系统方面有显著优势。

2. **视频内容分析**：
   - 视频中展示了一个大型的“2021”数字装置，这可能是为了庆祝或宣传Rockchip在2021年的重要里程碑。
   - 视频中还出现了“Rockchip”品牌标志，以及“2021”字样，表明这是2021年的一个重要活动或发布会。
   - 视频中还出现了“AI技术”、“计算及平台”、“影像调校”、“分论坛”等字样，这些可能是Rockchip在2021年举办的各类技术研讨会或产品发布会的主题。
...
```
注意：由于模型版本、输入图像或运行环境的差异，实际输出结果可能与上述示例略有不同。

## 5. 常见问题

- 模型加载失败：请检查路径是否正确，以及 `.weight` 文件是否与 `.rknn` 配套。
- Tokenizer 报错/编码问题：确认 `--tokenizer_path` 指向正确的 HuggingFace tokenizer 目录或可用名称。
- embed 回调失败：确认 `llm.embed.bin` 的 shape 与脚本中 `VOCAB_SIZE`、嵌入维度一致。
- 视觉模型预处理不正确：确认输入图像尺寸与模型所需尺寸一致（脚本中默认将图像 resize 到 384x384 并进行了 prune 处理，如需完整模型请移除 prune 步骤）。
