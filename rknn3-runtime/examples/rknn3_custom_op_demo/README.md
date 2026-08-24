# RKNN3 Custom Op Demo

验证 RKNN3 自定义算子框架，包含自定义算子 `RknnCustomOpExam` 和后处理算子 `RknnPostProcess`。

## 算子说明

### RknnCustomOpExam（CUSTOM_OP 类型）

计算：`y = clamp(x * scale + shift, min_val, max_val)`

参数覆盖所有 6 种 dtype：

| 参数名 | dtype | 值 | 含义 |
|--------|-------|-----|------|
| `min_val` | i | -3 | 最小值截断 |
| `max_val` | i | 3 | 最大值截断 |
| `scale` | f | 2.0 | 缩放因子 |
| `shift` | f | 0.5 | 偏移量 |
| `mode` | s | "linear" | 模式标记 |
| `strides` | is | [1,2,3] | 步幅数组 |
| `weights` | fs | [0.1,0.2,0.3] | 权重数组 |
| `tags` | ss | ["a","b"] | 标签数组 |

### RknnPostProcess（POSTPROCESS 类型）

直通拷贝（memcpy）。

## 目录结构

```
rknn3_custom_op_demo/
├── python/
│   ├── gen_onnx.py           # 构造 ONNX 模型
│   ├── verify_onnx.py        # 计算参考输出并保存 npy
│   └── convert2rknn.py       # 转换为 RKNN（含 reg_custom_op 注册）
├── cpp/
│   ├── main.cc               # 运行 demo（推理 + 余弦相似度对比）
│   ├── CMakeLists.txt
│   ├── build-linux.sh        # 交叉编译脚本（Linux aarch64，参照 rknn3_model_test_demo）
│   ├── build-android.sh      # 交叉编译脚本（Android arm64-v8a，参照 rknn3_model_test_demo）
│   └── plugin/
│       ├── rknn_custom_op_impl.c    # 算子实现（含 get_param 全类型示例）
│       ├── rknn_custom_op_impl.h    # 算子回调函数声明
│       ├── rknn3_custom_op.c # 插件注册入口
│       └── build.sh          # 编译插件 .so
├── install/                  # 所有编译产物
│   ├── rknn3_custom_op_demo  # 可执行文件（aarch64 / arm64-v8a）
│   ├── rknn_custom_op.rknn     # RKNN 模型
│   ├── ref_input.npy         # 参考输入
│   ├── ref_output.npy        # 参考输出
│   └── lib/                  # lib 库
└── README.md
```

## 使用流程

### 1. 生成 ONNX 模型

```bash
cd python
python3 gen_onnx.py --shape 1 3 256 256
# 产物: ../install/rknn_custom_op.onnx
```

### 2. 生成参考输出

```bash
python3 verify_onnx.py
# 产物: ../install/ref_input.npy, ../install/ref_output.npy
```

### 3. 编译插件 .so

插件 .so 必须与运行 demo 的目标架构一致：
- **RK182X（RISC-V）运行**：编 riscv 版（demo 推到板子，协处理器 RK182X 加载）
- **Android arm64 运行**：编 android 版（demo 跑在 Android，本机加载）

```bash
cd ../cpp/plugin

# Android (arm64-v8a) —— 给 Android 版 demo 用
./build.sh android

# 或 RISC-V —— 给 RK182X 用
./build.sh riscv

# 产物自动安装到 ../../install/lib/librknn_custom_op.so
# 注意：两种架构的 .so 互不兼容，切换目标时需重新编译覆盖
```

**编译要求：**

- **语言**：必须使用纯 C 语言（不支持 C++ 类、模板、命名空间等）
- **RISC-V编译器下载地址**： RK182X-GCC https://console.zbox.filez.com/l/103Dro 提取码：rknn

### 4. 转换为 RKNN

```bash
cd ../../python
python3 convert2rknn.py --onnx_model_path ../install/rknn_custom_op.onnx
# 产物: ../install/rknn_custom_op.rknn ../install/rknn_custom_op.weight
```

> 默认平台为 `rk1820`。如目标为其它平台，用 `--platform` 指定，例如：
> ```bash
> python3 convert2rknn.py --platform rk3572
> ```
> 可选值：`rk1820`（默认）/ `rk1828` / `rk3572`。

### 5. 编译 demo（交叉编译）

按目标平台二选一：

**Linux aarch64（板端运行）**

```bash
cd ../cpp
./build-linux.sh -t rk3588 -a aarch64 -b Release
# 产物自动安装到 ../install/
```

> 需指定环境变量`GCC_COMPILER` 指向 aarch64 交叉编译工具链前缀（`.../bin/aarch64-linux-gnu`），脚本据此定位 `gcc`/`g++`。

**Android arm64-v8a（Android 设备运行）**

```bash
cd ../cpp
./build-android.sh -t rk3572 -a arm64-v8a -b Release
# 产物自动安装到 ../install/
```

> 注意：Android 版 RK3572 平台 demo 需配套 Android arm64 版插件 `.so`（见第 3 步 `./build.sh android`），否则运行时加载 riscv 版会失败。

### 6. 板端运行

将 `install/` 目录整体推送到板子：

```bash
adb push install/ /data/rknn_custom_op_demo/

# 在板子上运行
adb shell
cd /data/rknn_custom_op_demo
export LD_LIBRARY_PATH=./lib
./rknn3_custom_op_demo \
    rknn_custom_op.rknn \
    rknn_custom_op.weight \
    ref_input.npy \
    ref_output.npy \
    0x01 \
    ./lib/rknn_custom_op.so
```

输出示例(RK3588 + RK182X)：
```
RK3588:
[Demo] Input shape: 1 3 256 256, total=196608 elems
[Demo] Reference output shape: 1 3 256 256, total=196608 elems
[Demo] rknn3_init...
[Demo] rknn3_load_model_from_path: rknn_custom_op.rknn rknn_custom_op.weight
[Demo] rknn3_model_init...
[Demo] rknn3_register_custom_ops_plugins: librknn_custom_op.so
[Demo] Plugin registered successfully
[Demo] n_inputs=1, n_outputs=1
[Demo] model input batch=1 channel=3 height=256 width=256
[Demo] Running inference...

========== Results ==========
Max diff:        0.001953
Cosine sim:      1.000000
>>> PASS (cosine > 0.999)
=============================

RK182X:
[Plugin] rknn3_register_custom_ops_plugin(op_index=0) called
[Plugin] rknn3_register_custom_ops_plugin(op_index=1) called
[Plugin] rknn3_register_custom_ops_plugin(op_index=2) called
[RknnPostProcess] get_output_num called
[RknnPostProcess] get_output_num called
[RknnPostProcess] get_attrs called
[RknnPostProcess] get_output_num called
[RknnCustomOpExam] compute called (n_inputs=1, n_outputs=1)
[RknnCustomOpExam] min_val (i) = -3
[RknnCustomOpExam] max_val (i) = 3
[RknnCustomOpExam] scale   (f) = 2.000000
[RknnCustomOpExam] shift   (f) = 0.500000
[RknnCustomOpExam] mode    (s) = 'linear' (len=6)
[RknnCustomOpExam] strides (is) = [1, 2, 3] (n_elems=3)
[RknnCustomOpExam] weights (fs) = [0.100000, 0.200000, 0.300000] (n_elems=3)
[RknnCustomOpExam] tags    (ss) = ['a', 'b'] (n_elems=2)
```
