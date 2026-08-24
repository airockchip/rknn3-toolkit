[English](README.md)

# 简介

RKNN3 SDK 提供了将 AI 模型部署到 RK1820/RK1828/RK3572 所需的完整软件栈，包括：

- **[RKNN3-Toolkit](https://github.com/airockchip/rknn3-toolkit)**：PC 端软件开发套件，支持模型转换、推理和性能评估等。
- **RKNN3 Runtime**：板端运行时库，提供 C/C++ 编程接口，用于部署 RKNN 模型并加速 AI 应用。
- **[RKNN3 Model Zoo](https://github.com/airockchip/rknn3-model-zoo)**：模型转换与部署示例仓库，包含 CNN / LLM / VLM 等多种模型的参考实现。

**典型工作流程**：用户首先在 PC 上使用 RKNN3-Toolkit 将训练好的模型转换为 RKNN 格式，然后通过 RKNN3 Runtime API 在开发板上进行推理。



# 支持平台

  - RK1820
  - RK1828
  - RK3572

**注意**： 

-  **对于RK3588/RK3576/RK3568/RK3566/RK3562系列、RV1103/RV1106、RV1103B/RV1106B、RV1126B、RK2118，请参考：** 

    https://github.com/airockchip/rknn-toolkit2      
    
- **对于RK1808/RV1109/RV1126/RK3399Pro，请参考：** 

    https://github.com/airockchip/rknn-toolkit  
    
    https://github.com/airockchip/rknpu 

    https://github.com/airockchip/RK3399Pro_npu  


- **RKNN3 Model Zoo 提供了更多的转换及部署示例**

   https://github.com/airockchip/rknn3-model-zoo



# 支持模型

### LLM

 - [x] Qwen2.5
 - [x] Qwen3
 - [x] Qwen3.5
 - [x] Youtu-LLM
 - [x] GLM-Edge
 - [x] MiniCPM5
 - [x] Nanbeige4.2
 - [x] FunctionGemma
 - [x] LFM2.5

### VLM

 - [x] Qwen2.5-VL
 - [x] Qwen3-VL
 - [x] Qwen3.5-VL
 - [x] FastVLM
 - [x] InternVL3
 - [x] InternVL3.5
 - [x] MiMo-VL-RL
 - [x] Janus-Pro
 - [x] MiniCPM-V-4
 - [x] SmolVLM
 - [x] SmolVLM2
 - [x] UI_TARS
 - [x] gme-Qwen2-VL
 - [x] LocateAnything

### Omni

 - [x] Qwen2.5-Omni (Thinker)
 - [x] Qwen3-Omni
 - [x] Qwen3.5-Omni
 - [x] Gemma4

### ASR（语音识别）

 - [x] Qwen3-ASR
 - [x] WeNet (Conformer)
 - [x] Whisper
 - [x] SenseVoice
 - [x] Zipformer

### TTS（文本转语音）

 - [x] Qwen3_TTS
 - [x] VITS

### Embedding / Reranker

 - [x] Qwen3-Embedding
 - [x] Qwen3-Reranker

### 翻译

 - [x] HY-MT1.5

### OCR

 - [x] PaddleOCR VL

### CV（计算机视觉）

 - [x] SigLIP
 - [x] SigLIP2
 - [x] MetaCLIP2
 - [x] QA-CLIP
 - [x] DINOv2
 - [x] DINOv3
 - [x] Depth-Anything-V2-small
 - [x] Depth-Anything-V3
 - [x] Diffusion Policy
 - [x] GR00T
 - [x] MiniCPM-RobotManip
 - [x] MobileNetV1 / V2
 - [x] ResNet-50
 - [x] YOLOv5 / YOLOv6 / YOLOv8
 - [x] YOLO-World
 - [x] YOLO26 / YOLO26-Segment / YOLO26-Pose

### 预转换 RKNN 模型

用户可以从 [RKNN3_SDK 网盘](https://console.box.lenovo.com/l/H1fig1) 下载预先转换好的 RKNN 模型（提取码：`rknn`）。本次发布的模型位于 `RKNN3_SDK/rknn3_models/v1.1.0` 目录。


# 性能

性能数据请参考 [发布说明](doc/CN/00_RKNN3_SDK_发布说明_V1.1.0.pdf)

# 注意事项  

- **RKNN3-Toolkit** 与 [RKNN-Toolkit](https://github.com/airockchip/rknn-toolkit) 和 [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) **不兼容**。  



# 支持的Python版本：  

  - Python 3.10  
  - Python 3.12  

# 最新版本：V1.1.0

# 更新日志

## V1.1.0

- 新增 RK182X 多卡级联支持
- 新增 Qwen3.5 模型支持
- 新增 Qwen3.5 和 Gemma4 的 Prefix Caching 支持
- 新增自定义 CPU 算子支持
- 新增多 Session 设置不同上下文长度的支持，满足不同会话的使用需求
- 新增 Tie Word Embedding 支持
- 新增 RKNN3 Toolkit 对 macOS 平台的支持（Beta）
- 优化 KVCache 导入和导出过程中的 Host 端内存占用
- 优化 `rknn3_mem_sync` 接口性能
- 优化 GRQ 量化过程中的 GPU 显存占用
- 更新外部 GRQ 量化的使用方式
- 调整 RKNN3 Toolkit 模型导入方式：统一使用 ONNX（移除 TensorFlow/TFLite/Caffe/DarkNet 支持）

更多历史版本的更新内容，请参阅：[CHANGELOG.md](CHANGELOG.md)


# 反馈与社区支持  

- [Redmine](https://redmine.rock-chips.com) (**推荐反馈问题，请联系销售或FAE获取Redmine账号**)  
- QQ群1：1025468710（已满，请加群5）  
- QQ群2：547021958（已满，请加群5）  
- QQ群3：469385426（已满，请加群5）  
- QQ群4：958083853（已满，请加群5）  
- QQ群5：1077888690
<center class="half">  
  <img width="200" height="200"  src="res/QQGroupQRCode.png" title="QQ群二维码"/>  
  <img width="200" height="200"  src="res/QQGroup2QRCode.png" title="QQ群2二维码"/>  
  <img width="200" height="200"  src="res/QQGroup3QRCode.png" title="QQ群3二维码"/>  
  <img width="200" height="200"  src="res/QQGroup4QRCode.png" title="QQ群4二维码"/>  
  <img width="200" height="200"  src="res/QQGroup5QRCode.png" title="QQ群5二维码"/>
</center>  
