# CHANGELOG

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