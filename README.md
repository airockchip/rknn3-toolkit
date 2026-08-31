[中文](README_CN.md)

# Introduction

RKNN3 SDK provides the complete software stack for deploying AI models on RK1820/RK1828/RK3572, including:

**Core Components**:

- **[RKNN3-Toolkit](https://github.com/airockchip/rknn3-toolkit)**: PC-side software development kit for model conversion, inference, performance evaluation, etc.
- **[RKNN3 Runtime](https://github.com/airockchip/rknn3-toolkit/tree/main/rknn3-runtime)**: On-board runtime library providing C/C++ programming interfaces for deploying RKNN models and accelerating AI applications.
- **[RKNN3-Toolkit Lite](https://github.com/airockchip/rknn3-toolkit/tree/main/rknn3-toolkit-lite)**: On-board Python inference interface that wraps the RKNN3 Runtime (C API), supporting rapid verification and upper-layer application development for CNN / LLM / VLM models.
- **[rkllm3-server](https://github.com/airockchip/rknn3-toolkit/tree/main/rknn3-runtime/rkllm3-server)**: On-board OpenAI-compatible API service supporting text, image, and audio inputs, suitable for service deployment of LLM / VLM models.

**Companion Resources**:

- **[RKNN3 Model Zoo](https://github.com/airockchip/rknn3-model-zoo)**: Model conversion and deployment example repository, including reference implementations for CNN / LLM / VLM and other models.
- **[RK1820/RK1828 Firmware](https://console.box.lenovo.com/l/PHk0PF)**: Prebuilt firmware for RK1820/RK1828 development boards; flash it to deploy and run RKNN3 models on the board.

**Typical Workflow**: The typical model deployment workflow consists of the following three stages:

1. **Model Conversion (PC side)**: Use RKNN3-Toolkit to convert trained models (e.g., ONNX) to RKNN format, configure normalization, quantization, and target platform parameters, and export `.rknn` and `.weight` model files.
2. **Model Evaluation (board-connected)**: Connect to the development board via USB or network to verify inference correctness, and perform accuracy and performance analysis.
3. **Board Deployment (on-board)**: Load and run the RKNN model in the application using the RKNN3 Runtime C API, completing business integration and accelerated inference.



# Supported Platforms

- RK1820
- RK1828
- RK3572

**Deployment Modes**:

- **Coprocessor mode (RK1820/RK1828)**: Requires a host SoC (e.g., RK3588 / RK3576) or Windows PC; communicates with the coprocessor via `rknn3_transfer_proxy` over PCIe / USB / Ethernet.
- **Non-coprocessor mode (RK3572)**: The NPU is integrated inside the SoC; models run directly on the board.

**Supported OS**: Android / Linux / Windows

**Note**:

- **For RK3588/RK3576/RK3568/RK3566/RK3562 series, RV1103/RV1106, RV1103B/RV1106B, RV1126B, RK2118, please refer to:**

  https://github.com/airockchip/rknn-toolkit2

- **For RK1808/RV1109/RV1126/RK3399Pro, please refer to:**

  https://github.com/airockchip/rknn-toolkit

  https://github.com/airockchip/rknpu

  https://github.com/airockchip/RK3399Pro_npu


- **RKNN3 Model Zoo provides more conversion and deployment examples**

  https://github.com/airockchip/rknn3-model-zoo



# Supported Models

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

### ASR (Speech Recognition)

 - [x] Qwen3-ASR
 - [x] WeNet (Conformer)
 - [x] Whisper
 - [x] SenseVoice
 - [x] Zipformer

### TTS (Text-to-Speech)

 - [x] Qwen3_TTS
 - [x] VITS

### Embedding / Reranker

 - [x] Qwen3-Embedding
 - [x] Qwen3-Reranker

### Translation

 - [x] HY-MT1.5

### OCR

 - [x] PaddleOCR VL

### CV (Computer Vision)

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

# Performance

For performance data, please refer to the [Release Notes](doc/EN/00_RKNN3_SDK_Release_Notes_V1.1.0.pdf).



# Supported Python Versions:

- Python 3.10
- Python 3.12

# Changelog

## V1.1.0
- Added RK182X multi-card cascade support
- Added Qwen3.5 model support
- Added Qwen3.5 and Gemma4 Prefix Caching support
- Added custom CPU operator support
- Added support for different context lengths across sessions
- Added Tie Word Embedding support
- Added RKNN3 Toolkit support for macOS (Beta)
- Optimized host-side memory usage during KVCache import and export
- Optimized `rknn3_mem_sync` API performance
- Optimized GPU memory usage during GRQ quantization
- Updated the usage of external GRQ quantization
- Changed the RKNN3 Toolkit model import method to use ONNX only (TensorFlow/TFLite/Caffe/DarkNet support removed)

For more information about previous releases, please refer to [CHANGELOG.md](CHANGELOG.md).



# Documentation

Detailed documentation is available in the `doc/` directory:

- [RKNN3 SDK Release Notes](doc/EN/00_RKNN3_SDK_Release_Notes_V1.1.0.pdf)
- [RKNN3 SDK Quick Start](doc/EN/01_RKNN3_SDK_QuickStart_V1.1.0.pdf)
- [RKNN3 SDK Development Guide](doc/EN/02_RKNN3_SDK_Development_Guide_V1.1.0.pdf)
- [RKNN3 Toolkit Python API Reference](doc/EN/03_RKNN3_Toolkit_Python_API_Reference_V1.1.0.pdf)
- [RKNN3 Toolkit Lite Python API Reference](doc/EN/04_RKNN3_Toolkit_Lite_Python_API_Reference_V1.1.0.pdf)
- [RKNN3 Runtime C API Reference](doc/EN/05_RKNN3_Runtime_C_API_Reference_V1.1.0.pdf)
- [RKLLM3 Server User Guide](doc/EN/06_RKLLM3_Server_User_Guide_V1.1.0.pdf)
- [RKNN3 Supported Operators and Constraints Reference](doc/EN/07_RKNN3_Supported_Operators_and_Constraints_Reference_V1.1.0.pdf)



# Cloud Drive Resources

- RK3588 EVB10 prebuilt firmware (RK3588_EVB10/RELEASE_V1.1.0):

   [https://console.box.lenovo.com/l/7oOghG](https://console.box.lenovo.com/l/7oOghG)

- RK1820/RK1828 prebuilt firmware (RK1820_RK1828/RELEASE_V1.1.0):

    [https://console.box.lenovo.com/l/PHk0PF](https://console.box.lenovo.com/l/PHk0PF)

- Pre-converted RKNN models (RKNN3_SDK/rknn3_models/v1.1.0):

    [https://console.box.lenovo.com/l/H1fig1](https://console.box.lenovo.com/l/H1fig1), access code: `rknn`



# Notes

- **RKNN3-Toolkit** is **not compatible** with [RKNN-Toolkit](https://github.com/airockchip/rknn-toolkit) and [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2).



# Feedback and Community Support

- [Redmine](https://redmine.rock-chips.com) (**Recommended for reporting issues. Please contact sales or an FAE to get a Redmine account**)
- QQ Group 1: 1025468710 (Full, please join group 5)
- QQ Group 2: 547021958 (Full, please join group 5)
- QQ Group 3: 469385426 (Full, please join group 5)
- QQ Group 4: 958083853 (Full, please join group 5)
- QQ Group 5: 1077888690
<center class="half">
  <img width="200" height="200"  src="res/QQGroupQRCode.png" title="QQ Group QR Code"/>
  <img width="200" height="200"  src="res/QQGroup2QRCode.png" title="QQ Group 2 QR Code"/>
  <img width="200" height="200"  src="res/QQGroup3QRCode.png" title="QQ Group 3 QR Code"/>
  <img width="200" height="200"  src="res/QQGroup4QRCode.png" title="QQ Group 4 QR Code"/>
  <img width="200" height="200"  src="res/QQGroup5QRCode.png" title="QQ Group 5 QR Code"/>
</center>
