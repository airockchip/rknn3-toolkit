# Session 测试用例说明
此工程提供以下 Session 测试用例：

1. rknn3_session_test  
   通用 Session 推理示例，用于演示基于典型的 prompt 输入的推理流程。
2. rknn3_session_test_token_embed_input  
   基于 token/embed 输入接口的推理示例，可用于非 prompt 输入类型的推理场景。
3. rknn3_session_test_eval_perf  
   性能评估示例，用于评估 LLM 模型的 TPS/TTFT 等核心性能指标。
4. rknn3_session_test_function_call  
   支持 Function Calling 特性的推理示例，用于演示大模型在实际函数调用场景下的推理方式。
5. rknn3_session_test_save_checkpoint  
   保存 KVCache Checkpoint 的示例，用于演示如何在推理过程中保存 KVCache Checkpoint，适用于带有 linear attention 或者 sliding attention 的模型 (如 qwen3.5 和 gemma4 系列模型)，其他模型不会生效。
6. rknn3_session_test_multi_session  
   多 Session 示例，在同一个 context 下创建多个 session（数量由 `NUM_SESSIONS` 宏控制，默认 2），分别使用不同的 max_context_len（从模型 full-attention kvcache buffer lengths 中获取）。每个 session 依次设置 NORMAL 和 SAVE_CHECKPOINT 两种 kvcache 策略，并进行多轮推理（轮数由 `NUM_ITERATIONS` 宏控制，默认 2），每轮推理后保存 kvcache checkpoint，下一轮推理前加载该 checkpoint，用于演示多 session 推理及 kvcache checkpoint 的保存与加载。

## 运行命令与参数说明

### 运行命令

以下为各测试用例的运行命令（以 Linux/Android 平台为例）。Windows 平台需在可执行文件名后追加 `.exe`，其中 Cygwin 终端使用 `./` 前缀（如 `./rknn3_session_test.exe`），cmd/PowerShell 不需要前缀（如 `rknn3_session_test.exe`）：

```sh
# Usage: ./rknn3_session_test <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask> [keep_history] [ignore_eos_token] [key_path]
./rknn3_session_test Qwen2.5-0.5B.rknn Qwen2.5-0.5B.weight Qwen2.5-0.5B.tokenizer.gguf Qwen2.5-0.5B.embed.bin 2048 512 0xff

# Usage: ./rknn3_session_test_token_embed_input <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask> <test_type> [keep_history] [ignore_eos_token] [key_path]
./rknn3_session_test_token_embed_input Qwen2.5-0.5B.rknn Qwen2.5-0.5B.weight Qwen2.5-0.5B.tokenizer.gguf Qwen2.5-0.5B.embed.bin 2048 512 0xff 1

# Usage: ./rknn3_session_test_eval_perf <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <n_input_tokens> <max_new_tokens> <core_mask> [keep_history] [ignore_eos_token] [key_path]
./rknn3_session_test_eval_perf Qwen2.5-0.5B.rknn Qwen2.5-0.5B.weight Qwen2.5-0.5B.tokenizer.gguf Qwen2.5-0.5B.embed.bin 2048 128 128 0xff

# Usage: ./rknn3_session_test_function_call <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask> [keep_history] [ignore_eos_token] [key_path]
./rknn3_session_test_function_call Qwen2.5-0.5B.rknn Qwen2.5-0.5B.weight Qwen2.5-0.5B.tokenizer.gguf Qwen2.5-0.5B.embed.bin 2048 512 0xff

# Usage: ./rknn3_session_test_save_checkpoint <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask> <checkpoint_tail_overwrite> [keep_history] [ignore_eos_token] [key_path]
./rknn3_session_test_save_checkpoint Qwen3.5-0.8B.rknn Qwen3.5-0.8B.weight Qwen3.5-0.8B.tokenizer.gguf Qwen3.5-0.8B.embed.bin 2048 512 0xff 0

# Usage: ./rknn3_session_test_multi_session <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask> [keep_history] [ignore_eos_token] [key_path]
./rknn3_session_test_multi_session Qwen3.5-0.8B.rknn Qwen3.5-0.8B.weight Qwen3.5-0.8B.tokenizer.gguf Qwen3.5-0.8B.embed.bin 2048 512 0xff
```

### 参数说明

- `rknn_path`: rknn 文件路径
- `weight_path`: weight 文件路径
- `tokenizer.gguf`: tokenizer.gguf 文件路径
- `embedding.bin`: embedding.bin 文件路径
- `max_context_len`: 模型转换时 `rknn.config` 中配置的 max_ctx_len 值
- `n_input_tokens`: 输入的 token 数
- `max_new_tokens`: 每次会话最多生成的 token 数
- `core_mask`: 目前有 8 个核，对应 8 bit 数，使用哪一个核，就将哪一位置 1，例如使用核 0 和核 1，就将第 0 位和第 1 位置 1，得到的二进制数是 0b11，对应的十六进制数是 0x3，core_mask 设置成 0x3
- `keep_history`: 是否保留历史对话上下文（0：不保留，1：保留），默认为 0
- `ignore_eos_token`: 是否忽略结束符（0：不忽略，1：忽略），默认为 0
- `key_path`: 模型加密密钥文件路径（可选）
- `test_type`: 测试类型（0：prompt 输入，1：token 输入，2：embed 输入），仅 token_embed_input 使用
- `checkpoint_tail_overwrite`: 是否覆盖 checkpoint 尾部（0：不覆盖，1：覆盖），仅 save_checkpoint 使用

## Linux 平台使用示例

### 编译

```sh
# 请先指定编译器路径
(optional)export GCC_COMPILER=<GCC_COMPILER_PATH>

./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> [-b <build_type>]

# 例如
./build-linux.sh -t rk3588 -a aarch64 -b Release
```

### install 目录库文件说明

编译并执行 `make install` 后，`install/rknn3_session_test_RK3588_Linux/lib` 中会按平台安装以下库：

- RK3576 / RK3588：`librknn3_api.so` + `librknn3_api_rkcp.so`
- RK3572：`librknn3_api.so` + `librknn3_api_native.so`

### 推送到板端

```sh
adb push install/rknn3_session_test_<TARGET_PLATFORM>_Linux/ /data/
```

### 运行

```sh
adb shell
cd /data/rknn3_session_test_<TARGET_PLATFORM>_Linux

export LD_LIBRARY_PATH=./lib

# 运行命令与参数说明请参考 `运行命令与参数说明` 章节。例如：
# Usage: ./rknn3_session_test <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask> [keep_history] [ignore_eos_token] [key_path]
./rknn3_session_test Qwen2.5-0.5B.rknn Qwen2.5-0.5B.weight Qwen2.5-0.5B.tokenizer.gguf Qwen2.5-0.5B.embed.bin 2048 512 0xff
```


## Android 平台使用示例

### 编译

```sh
# 请先指定编译器路径
(optional)export ANDROID_NDK_PATH=<ANDROID_NDK_PATH>

./build-android.sh -t <TARGET_PLATFORM> -a <ARCH> [-b <build_type>]

# 例如
./build-android.sh -t rk3588 -a arm64-v8a -b Release
```

### install 目录库文件说明

编译并执行 `make install` 后，`install/rknn3_session_test_RK3588_Android/lib` 中会按平台安装以下库：

- RK3576 / RK3588：`librknn3_api.so` + `librknn3_api_rkcp.so`
- RK3572：`librknn3_api.so` + `librknn3_api_native.so`

### 推送到板端

```sh
adb root
adb remount
adb push install/rknn3_session_test_<TARGET_PLATFORM>_Android/ /data/
```

### 运行

```sh
adb shell
cd /data/rknn3_session_test_<TARGET_PLATFORM>_Linux

export LD_LIBRARY_PATH=./lib

# 运行命令与参数说明请参考 `运行命令与参数说明` 章节。例如：
# Usage: ./rknn3_session_test <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask> [keep_history] [ignore_eos_token] [key_path]
./rknn3_session_test Qwen2.5-0.5B.rknn Qwen2.5-0.5B.weight Qwen2.5-0.5B.tokenizer.gguf Qwen2.5-0.5B.embed.bin 2048 512 0xff
```

# Windows 平台使用示例（Cygwin）

Windows 平台的使用场景为 PC 直接连 RK182X，无需指定目标平台参数。

## 编译

在 Windows Cygwin 环境下，执行如下命令：

```sh
./build-cygwin.sh [-b <build_type>]

# 例如：
./build-cygwin.sh -b Release
```

**注意：**
- 需要在 Cygwin 终端中运行构建脚本
- 确保已安装 Cygwin 的编译工具链（gcc、g++、make 等）

## install 目录库文件说明

编译并执行 `make install` 后，`install/rknn3_session_test_Windows/lib` 中会安装以下库：

- `librknn3_api.dll` + `librknn3_api_rkcp.dll`

（Windows 平台仅有一套库，对应 PC 直连 RK182X 场景）

## 运行

在 Cygwin 终端或 Windows 命令行中运行：

```sh
# 方式 1: Cygwin 终端
cd install/rknn3_session_test_Windows/
export PATH=./lib:$PATH
# 运行命令与参数说明请参考 `运行命令与参数说明` 章节。Windows 平台需在可执行文件名后追加 `.exe`，例如：
# Usage: ./rknn3_session_test.exe <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask> [keep_history] [ignore_eos_token] [key_path]
./rknn3_session_test.exe Qwen2.5-0.5B.rknn Qwen2.5-0.5B.weight Qwen2.5-0.5B.tokenizer.gguf Qwen2.5-0.5B.embed.bin 2048 512 0xff


# 方式 2: Windows 命令行 (cmd/PowerShell)
cd install\rknn3_session_test_Windows
set PATH=.\lib;%PATH%

# 运行命令与参数说明请参考 `运行命令与参数说明` 章节。Windows 平台需在可执行文件名后追加 `.exe`，例如：
# Usage: rknn3_session_test.exe <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <max_new_tokens> <core_mask> [keep_history] [ignore_eos_token] [key_path]
rknn3_session_test.exe Qwen2.5-0.5B.rknn Qwen2.5-0.5B.weight Qwen2.5-0.5B.tokenizer.gguf Qwen2.5-0.5B.embed.bin 2048 512 0xff

```

**注：**
1. **Windows 平台下路径分隔符使用反斜杠 `\`，但在 Cygwin 终端中仍可使用正斜杠 `/`**
2. **确保 DLL 文件所在的 lib 目录已添加到系统 PATH 环境变量中**
