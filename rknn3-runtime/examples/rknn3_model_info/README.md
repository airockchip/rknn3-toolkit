# rknn3_model_info Demo

This demo displays system memory, RKNN model memory usage, KV Cache configuration, and related information. It **does not require a weight file**—only the `.rknn` model structure is loaded, and memory usage is evaluated via `rknn3_model_init()`.

## Build

Set the `GCC_COMPILER` path, for example:

```sh
export GCC_COMPILER=~/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu

./build-linux.sh -t rk3588 -a aarch64 -b Release
```

## Libraries in the install Directory

After building and running `make install`, the following libraries are installed under `install/rknn3_model_info_Linux/lib` depending on the platform:

- RK3576 / RK3588: `librknn3_api.so` + `librknn3_api_rkcp.so`
- RK3572: `librknn3_api.so` + `librknn3_api_native.so`

## Deploy to the Board

Copy `install/rknn3_model_info_Linux` to the target device, for example:

```sh
adb push install/rknn3_model_info_Linux /userdata/
```

## Run

```sh
export LD_LIBRARY_PATH=./lib
./rknn3_model_info <model_path> [core_mask] [device_id] [key_path]
```

### Arguments

1. `model_path`: Path to the RKNN model file (required)
2. `core_mask`: Optional hexadecimal core mask; if omitted, it is auto-generated from the model core count
3. `device_id`: Optional target device ID (obtainable via `rknn3_find_devices()`); if omitted, the first device is used
4. `key_path`: Optional path to the RSA key envelope file for encrypted models

### Examples

```sh
# Query model memory info only
./rknn3_model_info model.rknn

# Specify core mask
./rknn3_model_info model.rknn 0xff

# Specify device_id
./rknn3_model_info model.rknn 0xff rk1820-xxxx

# Encrypted model
./rknn3_model_info model.rknn 0xff rk1820-xxxx ./key.env
```

## Output

- Device list
- SDK version
- Model core count and input/output counts
- KV Cache configuration for LLM models (`max_ctx_len`, `kvcache_dtype`, etc.)
- Weight/internal memory per core
- Per-core command/weight/internal/kvcache allocation details
