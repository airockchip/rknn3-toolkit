# rknn3_model_info Demo

用于展示系统内存、RKNN 模型内存占用、KV Cache 配置等信息。该 demo **不需要 weight 文件**，仅加载 `.rknn` 模型结构并调用 `rknn3_model_init()` 进行内存评估。

## 编译

指定编译器路径 `GCC_COMPILER`，例如：

```sh
export GCC_COMPILER=~/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu

./build-linux.sh -t rk3588 -a aarch64 -b Release
```

## install 目录库文件说明

编译并执行 `make install` 后，`install/rknn3_model_info_Linux/lib` 中会按平台安装以下库：

- RK3576 / RK3588：`librknn3_api.so` + `librknn3_api_rkcp.so`
- RK3572：`librknn3_api.so` + `librknn3_api_native.so`

## 推送到板端

将 `install/rknn3_model_info_Linux` 拷贝到设备上，例如：

```sh
adb push install/rknn3_model_info_Linux /userdata/
```

## 运行

```sh
export LD_LIBRARY_PATH=./lib
./rknn3_model_info <model_path> [core_mask] [device_id] [key_path]
```

### 参数说明

1. `model_path`：RKNN 模型文件路径（必填）
2. `core_mask`：可选，十六进制 core mask；省略时根据模型 core 数自动生成
3. `device_id`：可选，目标设备 ID（可通过 `rknn3_find_devices()` 获取）；省略时使用第一个设备
4. `key_path`：可选，加密模型的 RSA 密钥信封文件路径

### 示例

```sh
# 仅查看模型内存信息
./rknn3_model_info model.rknn

# 指定 core mask
./rknn3_model_info model.rknn 0xff

# 指定 device_id
./rknn3_model_info model.rknn 0xff rk1820-xxxx

# 加密模型
./rknn3_model_info model.rknn 0xff rk1820-xxxx ./key.env
```

## 输出内容

- 设备列表
- SDK 版本
- 模型 core 数、输入输出数量
- LLM 模型的 KV Cache 配置（`max_ctx_len`、`kvcache_dtype` 等）
- 各 core 的 weight/internal 内存
- 各 core 的 command/weight/internal/kvcache 分配详情
