# CHANGELOG

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

## V1.0.4
- Added RK3572 platform support (Beta)
- Added Windows platform support
- Added Ethernet communication type support
- Added device suspension and wake-up support
- Added server-side auto-select communication method
- Added streaming weight loading support
- Added model encryption support
- Added LoRA support
- Added session pause/resume support
- Added session input callback support
- Added multi-session parallel running support
- Added KVCache import/export interface
- Added module version compatibility check
- Added Qwen3 TTS text-to-speech model
- Added Qwen3-ASR speech recognition model
- Added VITS audio synthesis model
- Added Zipformer model
- Added MetaCLIP2 model
- Added Gemma4 multimodal model
- Added GR00T model
- Added PaddleOCR VL model
- Added SmolVLM2 vision-language model

## v1.0.0

- Significantly improved LLM/ViT performance; overall LLM decode performance improved by more than 15%.
- Expanded model support range, adding models such as Qwen3-VL / Qwen2.5-Omni(Thinker) / GLM Edge / SmolVLM.
- Added support for cross-board accuracy analysis.
- Added support for overlapping data transfer and inference.
- Added support for mRoPE.
- Added support for Function Call.
- Added support for YUV-format input.
- `rkllm3-server` now supports embedding models and audio input.
- Added support for concurrent multi-core, multi-model inference.
- Added support for custom model post-processing on the coprocessor.
- Optimized implementation of exSDPA, exMatMul, Resize, Transpose operators.
- Provides RKNN3 Toolkit Lite package to support Python API calls on development boards.


## v0.4.0b0

- Optimization of SDK functions and stability, with known bugs fixed
- Support for more models, including CNN models, ViT models, and LLM/VLM models
- Improved stability of multi-threaded inference
- Fixed precision anomalies of some operators under specific specifications, such as matmul, resize, gather, etc.
- Optimized memory usage of large models and increased the maximum supported context length
- Support for video/audio in multimodal models

## v0.3.0b0

- Optimized SDK functionality and stability; fixed known bugs from the V0.2.0 release.
- Support for more models, including CNN, LLM, and VLM models.
- Reduced server memory consumption for model conversion by over 30%.
- RKNN3 Toolkit now supports inference on connected development boards.
- RKNN3 Model Zoo supports more models, especially VLM-related examples like InternVL3-2B and Qwen2.5-VL-3B.
- Performance optimization for Yolo-series detection models, particularly for multi-batch and multi-core performance (requires modification for models, referring to RKNN3 Model Zoo for details).
- LLM inference KVCache now supports a sliding window, allowing the number of output tokens to exceed the maximum context length.
- Support for USB communication.

## v0.2.0

- Added support for the conversion and deployment of CNN, LLM, and VLM models.

## v0.1.0

- Initial version, supporting model performance evaluation.