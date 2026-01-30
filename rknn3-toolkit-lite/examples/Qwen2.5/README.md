# Qwen2.5 示例部署说明

本目录包含 `test.py`（LLM 推理示例）和 `test_embed.py`（使用预计算 embed 的示例）。文档说明如何准备模型文件、依赖环境，以及在开发板/本地进行推理的示例命令。

## 模型与文件
- LLM 模型（RKNN）：/xxx/Qwen2.5-3B-Instruct.rknn 或 /xxx/Qwen2.5-VL-3B-llm.rknn（取决于你使用的模型变种）
- LLM 权重（WEIGHT）：对应 `.weight` 文件，文件名与 `.rknn` 一致但扩展名为 `.weight`
- LLM embed（二进制）：/xxx/Qwen2.5-3B-Instruct.embed.bin 或 /xxx/Qwen2.5-VL-3B-llm.embed.bin
- Tokenizer 目录：examples/Qwen2.5/Qwen2.5-Base 或 HuggingFace 名称

如果你使用不同路径，请在运行脚本时用命令行参数覆盖（见下文）。

## 1. 运行环境

- 推荐使用仓库根目录的 `requirements.txt` 来安装依赖。示例脚本使用 `transformers` 的 `AutoTokenizer`，以及 `numpy`、`ctypes` 等库。
- 请确保已经安装并可导入仓库中的 `rknn3lite` 包。

> ⚠️ 请确认目标设备（如 RK 系列开发板）上有对应运行库，并且 `.weight` 文件与 `.rknn` 匹配。

## 2. 准备模型与 tokenizer

1. 将相应的 `.rknn` 和配套的 `.weight` 放到设备或工作目录中。
2. 将 `.embed.bin` 放到可访问路径，并确认 embedding 的尺寸和脚本中 `VOCAB_SIZE` 一致。
3. 把 tokenizer 文件夹放在 `examples/Qwen2.5/` 下。

## 3. 运行示例脚本

两个示例脚本都有命令行参数来指定模型、tokenizer、embed 路径：

- `test.py`：使用 tokenizer 回调（脚本内部会调用 tokenizer 将文本转换为 token）。默认脚本已将模型路径指向 `RBNN`/`WEIGHT` 或 VL 版本（见脚本顶部的 `RKNN_MODEL` 常量）。
- `test_embed.py`：预先加载 `embed.bin` 并直接传入 embedding（适用于支持 embed 模式的模型和配置）。

基本运行示例：

```bash
# 使用 tokenizer 的示例
python test.py \
  --rknn_path /xxx/Qwen2.5-3B-Instruct.rknn \
  --weight_path /xxx/Qwen2.5-3B-Instruct.weight \
  --tokenizer_path Qwen2.5-0.5B-Instruct \
  --embed_path /xxx/Qwen2.5-3B-Instruct.embed.bin

# 使用预计算 embed 的示例
python test_embed.py \
  --rknn_llm_path /xxx/Qwen2.5-3B-Instruct.rknn \
  --tokenizer_path Qwen2.5-0.5B-Instruct \
  --embed_path /xxx/Qwen2.5-3B-Instruct.embed.bin
```

脚本 `test.py` 默认会读取内置 prompts 并打印流式 token 输出；`test_embed.py` 演示如何使用外部 embedding 数据进行推理。

## 4. 输出与性能信息

demo示例输出片段：
```
### **1. 狭义相对论（Special Relativity）**
**核心思想**：  
- **光速不变原理**：光速（约30万公里/秒）是宇宙中恒定的极限速度，无论观察者如何运动，光速始终相同。  
- **相对性原理**：物理定律在所有惯性参考系（即匀速直线运动的参考系）中形式相同。  

**关键概念**：  
- **时间膨胀**：当物体以接近光速运动时，时间会变慢（例如，高速运动的飞船中时间流逝比地球慢）。  
- **长度收缩**：物体在运动方向上的长度会缩短（如高速运动的飞船在运动方向上的长度变短）。  
- **质能等价**：能量与质量之间存在关系 $ E = mc^2 $，即质量可以转化为能量（如核反应释放的能量）。
...
```
注意：由于模型版本、输入或运行环境的差异，实际输出可能与示例不同。

## 5. 常见问题

- 模型加载失败：请检查路径是否正确，以及 `.weight` 文件是否与 `.rknn` 配套。
- Tokenizer 报错/编码问题：确认 `--tokenizer_path` 指向正确的 HuggingFace tokenizer 目录或有效名称。
- embed 回调失败：确认 `llm.embed.bin` 的 shape 与脚本中 `VOCAB_SIZE`、嵌入维度一致。
- 如果脚本中顶部的默认路径不匹配你的部署，请在命令行中覆盖这些参数。
