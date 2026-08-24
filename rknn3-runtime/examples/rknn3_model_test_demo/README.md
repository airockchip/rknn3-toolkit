# RKNN3 Model Test Demo

End-to-end demo: load a model, prepare inputs, run inference, and optionally compare outputs against golden NumPy files.

## Linux

### Build

Set `GCC_COMPILER`, then run:

```sh
(optional) export GCC_COMPILER=<GCC_COMPILER_PATH>

./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> [-b <build_type>]

# Example:
./build-linux.sh -t rk3588 -a aarch64 -b Release
```

### Libraries in `install/rknn3_model_test_Linux/lib`

- RK3576 / RK3588: `librknn3_api.so` + `librknn3_api_rkcp.so`
- RK3572: `librknn3_api.so` + `librknn3_api_native.so`

### Deploy

```sh
adb push install/rknn3_model_test_Linux /userdata/
```

### Run

```sh
adb shell
cd /userdata/rknn3_model_test_Linux/
export LD_LIBRARY_PATH=./lib
./rknn3_model_test resnet50v2.rknn resnet50v2.weight 0x1 in.npy out.npy
```

## Android

### Build

```sh
(optional) export ANDROID_NDK_PATH=<ANDROID_NDK_PATH>

./build-android.sh -t <target> -a <arch> [-b <build_type>]

# Example:
./build-android.sh -t rk3588 -a arm64-v8a -b Release
```

### Deploy & run

```sh
adb push install/rknn3_model_test_Android /data/
adb shell
cd /data/rknn3_model_test_Android/
export LD_LIBRARY_PATH=./lib
./rknn3_model_test resnet50v2.rknn resnet50v2.weight 0x1 in.npy out.npy
```

## Windows (Cygwin)

On Windows, the PC connects directly to RK182X; no target platform parameter is needed.

### Build

```sh
./build-cygwin.sh [-b <build_type>]

# Example:
./build-cygwin.sh -b Release
```

### Run

```sh
cd install/rknn3_model_test_Windows/
export PATH=./lib:$PATH
./rknn3_model_test.exe resnet50v2.rknn resnet50v2.weight 0x1 in.npy out.npy
```

---

## Command-line arguments

```
rknn3_model_test <model_path> <weight_path> [core_mask] [input_npy_paths] [golden_output_npy_paths] [shape_id] [loop_count] [chunk_size_mb] [load_mode] [key_path] [enable_output_checksum]
```

| # | Argument | Required | Description |
|---|----------|----------|-------------|
| 1 | model_path | Yes | Path to the `.rknn` model file |
| 2 | weight_path | Yes | Path to the weight file |
| 3 | core_mask | No | Hex core bitmask (e.g. `0x3` for cores 0 and 1). Auto-generated if omitted |
| 4 | input_npy_paths | No | Input `.npy` paths, `#`-separated for multiple inputs, or a `.txt` path-list file. Random input if omitted |
| 5 | golden_output_npy_paths | No | Golden output `.npy` paths, `#`-separated, or a `.txt` path-list file. Cosine check skipped if omitted |
| 6 | shape_id | No | Dynamic-shape ID (default `0`) |
| 7 | loop_count | No | Inference loop count (default `1`) |
| 8 | chunk_size_mb | No | Streaming weight chunk size in MB (default `0` = load all at once) |
| 9 | load_mode | No | `path` (default) or `data`. Ignored when `chunk_size_mb > 0` |
| 10 | key_path | No | Decryption key file; use `none` to skip |
| 11 | enable_output_checksum | No | Output checksum validation across inference loops: `0` disables, `1` enables (default) |

Optional arguments are positional and must not be skipped—use `""` for an empty slot.

### Examples

**Minimal (random input, no golden check):**

```sh
./rknn3_model_test model.rknn model.weight
```

**With inputs, golden outputs, and core mask:**

```sh
./rknn3_model_test model.rknn model.weight 0x1 input.npy golden.npy
```

**Random input with custom core mask and loop count:**

```sh
./rknn3_model_test model.rknn model.weight 0x1 "" "" "" 1000
```

**Dynamic shape (shape_id = 1):**

```sh
./rknn3_model_test model.rknn model.weight 0x1 "" "" 1
```

**Streaming weight upload (64 MB chunks):**

```sh
./rknn3_model_test model.rknn model.weight 0x1 "" "" 0 1 64
```

**Load from memory (`data` mode):**

```sh
./rknn3_model_test model.rknn model.weight 0x1 input.npy golden.npy 0 1 0 data
```

**Encrypted model:**

```sh
./rknn3_model_test model.rknn model.weight 0x1 input.npy golden.npy 0 1 0 path my_key.bin
```

**Disable output checksum validation:**

```sh
./rknn3_model_test model.rknn model.weight 0x1 input.npy golden.npy 0 1 0 path none 0
```
