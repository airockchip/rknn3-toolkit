# Linux 平台使用示例

## 编译
指定编译器路径 GCC_COMPILER，例如`export GCC_COMPILER=~/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu `，执行如下命令：

```sh
# 请先指定编译器路径
(optional)export GCC_COMPILER=<GCC_COMPILER_PATH>

./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> [-b <build_type>]

# 例如: 
./build-linux.sh -t rk3588 -a aarch64 -b Release
```

## install 目录库文件说明

编译并执行 `make install` 后，`install/rknn3_model_test_Linux/lib` 中会按平台安装以下库：

- RK3576 / RK3588：`librknn3_api.so` + `librknn3_api_rkcp.so`
- RK3572：`librknn3_api.so` + `librknn3_api_native.so`

## 推送到板端

将 install/rknn3_model_test_Linux 拷贝到设备上。

- 如果使用 Rockchip 的 EVB 板 (如 RK3588 开发板)，可以使用以下命令：

连接设备并将程序和模型传输到`/userdata`

```
adb push install/rknn3_model_test_Linux /userdata/
```

- 如果你的板子有sshd服务，可以使用scp命令或者其他方式将程序和模型传输到板子上。

## 运行

```sh
adb shell
cd /userdata/rknn3_model_test_Linux/
```

```sh
export LD_LIBRARY_PATH=./lib
./rknn3_model_test resnet50v2.rknn resnet50v2.weight 0x1 in.npy out.npy
# usage:
# rknn3_model_test <model_path> <weight_path> [core_mask] [input_npy_paths] [golden_output_npy_paths] [shape_id] [loop_count] [chunk_size_mb] [load_mode] [key_path] [enable_output_checksum]
```

参数说明：
1. model_path: rknn 文件路径
2. weight_path: weight 文件路径
3. core_mask: 可选。十六进制核掩码；省略时根据设备核心数自动生成。例如使用核 0 和核 1，对应 `0x3`。
4. input_npy_paths: 可选。输入 `npy` 文件路径，多个输入用 `#` 分割（例如 `input1.npy#input2.npy`），也可以传入记录路径列表的 `.txt` 文件。
5. golden_output_npy_paths: 可选。golden 输出 `npy` 文件路径，多个输出用 `#` 分割（例如 `golden_out0.npy#golden_out1.npy`），也可以传入记录路径列表的 `.txt` 文件。
6. shape_id: 可选，默认 0。动态 shape 模型的 shape ID。
7. loop_count: 可选，默认 1。推理循环次数。
8. chunk_size_mb: 可选，默认 0。大于 0 时启用流式权重加载（MB 为单位）。
9. load_mode: 可选，默认 `path`。`path` 使用 `rknn3_load_model_from_path`；`data` 使用 `rknn3_load_model_from_data`。`chunk_size_mb > 0` 时忽略此项。
10. key_path: 可选。解密密钥文件路径；使用 `none` 显式跳过解密。
11. enable_output_checksum: 可选，默认 1。设为 `1` 时检查不同推理循环的输出 checksum 是否一致，设为 `0` 时关闭检查。

**注：**
1. **仅测试模型能否推理时，只需提供 `model_path` 和 `weight_path`；将使用随机输入，跳过 golden 对比，`core_mask` 自动生成，`loop_count` 默认为 1。**
2. **使用随机输入但需指定 `core_mask` 或 `loop_count` 时，对不需要的参数使用空字符串占位。例如：**
```sh
./rknn3_model_test resnet50v2.rknn resnet50v2.weight 0x1 "" "" "" 1000
```
3. **动态 shape 模型可通过 `shape_id` 选择 shape 组合。例如：**
```sh
./rknn3_model_test model.rknn model.weight 0x1 "" "" 1
```
4. **流式权重加载适用于大模型，可降低内存峰值。例如（shape_id=0，loop_count=1，chunk_size_mb=64）：**
```sh
./rknn3_model_test model.rknn model.weight 0x1 "" "" 0 1 64
```
5. **可选参数为位置参数，不能跳过前面的槽位；空槽请传 `""`。**
6. **关闭输出 checksum 检查时，需要保留前面的参数位置。例如：**
```sh
./rknn3_model_test model.rknn model.weight 0x1 input.npy golden.npy 0 1 0 path none 0
```

# Android 平台使用示例

## 编译
指定 ANDROID_NDK_PATH，例如`export ANDROID_NDK_PATH=~/opts/ndk/android-ndk-r17c`，然后执行如下命令：
```sh
# 请先指定编译器路径
(optional)export ANDROID_NDK_PATH=<ANDROID_NDK_PATH>

./build-android.sh -t <target> -a <arch> [-b <build_type>]

# 例如：
./build-android.sh -t rk3588 -a arm64-v8a -b Release
```

## install 目录库文件说明

编译并执行 `make install` 后，`install/rknn3_model_test_Android/lib` 中会按平台安装以下库：

- RK3576 / RK3588：`librknn3_api.so` + `librknn3_api_rkcp.so`
- RK3572：`librknn3_api.so` + `librknn3_api_native.so`

## 推送到板端

连接设备并将程序和模型传输到 `/data`

```
adb push install/rknn3_model_test_Android /data/
```

## 运行

```sh
adb shell
cd /data/rknn3_model_test_Android/
```

```sh
export LD_LIBRARY_PATH=./lib
./rknn3_model_test resnet50v2.rknn resnet50v2.weight 0x1 in.npy out.npy
# usage:
# rknn3_model_test <model_path> <weight_path> [core_mask] [input_npy_paths] [golden_output_npy_paths] [shape_id] [loop_count] [chunk_size_mb] [load_mode] [key_path] [enable_output_checksum]
```

参数说明与 Linux 平台相同（见上文 1–11 条及注意事项）。

# Windows 平台使用示例（Cygwin）

Windows 平台的使用场景为 PC 直连 RK182X，无需指定目标平台参数。

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

编译并执行 `make install` 后，`install/rknn3_model_test_Windows/lib` 中会安装以下库：

- `librknn3_api.dll` + `librknn3_api_rkcp.dll`

（Windows 平台仅有一套库，对应 PC 直连 RK182X 场景）

## 运行

在 Cygwin 终端或 Windows 命令行中运行：

```sh
# 方式 1: Cygwin 终端
cd install/rknn3_model_test_Windows/
export PATH=./lib:$PATH
./rknn3_model_test.exe resnet50v2.rknn resnet50v2.weight 0x1 in.npy out.npy

# 方式 2: Windows 命令行 (cmd/PowerShell)
cd install\rknn3_model_test_Windows
set PATH=.\lib;%PATH%
rknn3_model_test.exe resnet50v2_with_weight\resnet50v2.rknn resnet50v2_with_weight\resnet50v2.weight 0x1 resnet50v2_with_weight\in.npy resnet50v2_with_weight\out.npy

# usage:
# rknn3_model_test.exe <model_path> <weight_path> [core_mask] [input_npy_paths] [golden_output_npy_paths] [shape_id] [loop_count] [chunk_size_mb] [load_mode] [key_path] [enable_output_checksum]
```

参数说明与 Linux 平台相同（见上文 1–11 条及注意事项）。

**注：** Windows 平台下路径分隔符使用反斜杠 `\`，但在 Cygwin 终端中仍可使用正斜杠 `/`。
