# rknn3_llm_step_decode 说明文档

> English: [README_STEP_DECODE.md](README_STEP_DECODE.md)

`rknn3_llm_step_decode` 是一个**逐 token（外部控制）解码**示例：每次 `rknn3_session_run` 只执行一步（`max_new_tokens=1`），由外部循环把上一轮 `result_callback` 返回的 token 作为输入喂回模型，直到达到 `max_new_tokens` / EOS。

本示例与 `rknn3_llm_cli`（一次 `rknn3_session_run` 内部跑完 prefill + 全部 decode）的区别在于 **decode 的驱动权在调用方**：

```
一次调用完成推理:    一次 rknn3_session_run(max_new_tokens=N) 内部跑完 prefill + 全部 decode
step_decode:      run_turn() {
                      run_inference_once(prefill)        // 处理整段输入，采样产出第 1 个 token
                      for step in 1 .. max_new_tokens:
                          run_inference_once({last_token}) // 每步喂 1 个 token，产出下一个 token
                  }
```

- `run_inference_once()` 每次 `rknn3_session_run` 设 `max_new_tokens=1`、`prefill_only=true`、`disable_sampling=false`，即单步推理；输入为 prompt（prefill 阶段，仍会采样产出第 1 个 token）或单个 token（decode 阶段）。
- `run_turn()` 负责外部循环：prefill 后不断把 `result_callback` 返回的 token 喂回，直到达到 `max_new_tokens` 或命中 EOS（除非 `--ignore-eos`），并统计 TTFT / prefill / decode 耗时。decode 循环从第 2 个 token 开始（`step=1`），因为第 1 个 token 已由 prefill 阶段产出，避免重复计数。

## 1. 代码结构与复用

`rknn3_llm_cli` 与 `rknn3_llm_step_decode` **共享同一套推理引擎**（`cpp/` 目录），两个 demo 仅保留各自的前端入口：

```
cpp/
├── helpers.h      # 共享类型 / 工具（RunStats、embedding_info、日志宏、字符串助手）
├── args.h/.cc     # 命令行解析（两个 demo 的选项并集）
├── engine.h/.cc   # 共享推理引擎：会话初始化、回调、run_turn、bench、状态打印
├── rope.h/.cc     # 外置 rope 缓存（Qwen3.5 / Gemma-4）
├── lora.h/.cc     # LoRA 适配器生命周期
├── cli_main.cc    # rknn3_llm_cli 前端：REPL + 斜杠命令 + 中断监控
└── step_main.cc    # rknn3_llm_step_decode 前端：逐 token 循环入口
```

两个 target 由同一个 `cpp/CMakeLists.txt` 编译，解码策略通过 `SessionState::decode_mode`（`SESSION` / `EXTERNAL`）选择，其余逻辑全部共享。`engine.cc` 的 `run_turn()` 根据该模式分派到 `run_session_decode()`（一次调用）或 `run_external_decode()`（逐 token 循环）。

## 2. 编译

使用仓库统一编译脚本（一次编译出两个二进制）：

```bash
cd examples/rknn3_llm_cli/

# Linux（交叉编译需先 export GCC_COMPILER）
./build-linux.sh -t rk3588 -a aarch64 -b Release
# 产物位于 install/rknn3_llm_cli_rk3588_linux_aarch64/
#   rknn3_llm_cli          交互式 CLI
#   rknn3_llm_step_decode  逐 token 解码
```

## 3. 部署与运行

```bash
adb push install/rknn3_llm_cli_rk3588_linux_aarch64/ /data/
adb shell
cd /data/rknn3_llm_cli_rk3588_linux_aarch64
export LD_LIBRARY_PATH=./lib

# 启动（--model 必填，其余路径按同名自动推导）
./rknn3_llm_step_decode --model model/Qwen3.5-0.8B.rknn --core-mask 0xff

# 或显式指定全部路径
./rknn3_llm_step_decode \
    --model model/Qwen3.5-0.8B.rknn \
    --weight model/Qwen3.5-0.8B.weight \
    --vocab model/Qwen3.5-0.8B.tokenizer.gguf \
    --embed model/Qwen3.5-0.8B.embed.bin \
    -c 2048 --core-mask 0xff
```

> 说明：`--model` 必填，其余路径按 `foo.rknn` → `foo.weight` / `foo.tokenizer.gguf` / `foo.embed.bin` 自动推导（可用 `--dry-run` 验证）。模型文件由对应模型示例导出，如 `examples/Qwen3_5/python`（`export_llm.py` / `export_rknn.py`）。若模型采用外置 rope 请指定 `--rope-tensor` 参数。

### 3.1 命令行参数

本示例通过**启动参数**配置模型路径、采样参数、聊天模板、KV cache 策略、LoRA、外部 rope 与基准测试等，也支持命令行循环对话的基础交互。

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--model <path>` | RKNN 模型文件 | 必填 |
| `--weight <path>` | 权重文件（省略时自动推导） | 自动推导 |
| `--vocab <path>` | 分词器文件（省略时自动推导） | 自动推导 |
| `--embed <path>` | embedding 权重文件（省略时自动推导） | 自动推导 |
| `-c, --ctx-size <n>` | 最大上下文长度（token） | 模型配置值 |
| `-n, --predict <n>` | 每轮最大生成 token 数 | 512 |
| `--core-mask <mask>` | NPU 核心掩码，hex/dec，如 `0xff` | `0xff` |
| `--temperature <v>` | 采样温度 | 1.0 |
| `--top-k <n>` | top-k | 1 |
| `--top-p <v>` | top-p | 0.9 |
| `--repeat-penalty <v>` | 重复惩罚 | 1.2 |
| `--frequency-penalty <v>` | 频率惩罚 | 0.0 |
| `--presence-penalty <v>` | 存在惩罚 | 0.0 |
| `--keep-history <on/off>` | 多轮对话保持历史 | `on` |
| `-s, --system <prompt>` | 系统提示词 | 模型内置 |
| `--prompt-prefix <p>` | 聊天模板前缀 | 模型内置 |
| `--prompt-postfix <p>` | 聊天模板后缀 | 模型内置 |
| `-p, --prompt <text\|@file>` | 启动后先发送一条消息（`@` 开头从文件读取） | 无 |
| `--logits-name <name>` | 输出 logits 名称（模型有多个输出时） | `output` |
| `--once` | 推理 `--prompt` 指定内容后退出（只推理一次） | 关闭 |
| `--perf` | 每轮推理结束后打印该轮的性能指标 | 关闭 |
| `--verbose` | 开启详细日志 | 关闭 |
| `--ignore-eos` | 生成时忽略 EOS | 关闭 |
| `--thinking` | 开启思考模式 | 关闭 |
| `--dry-run` | 只打印解析后的模型路径并退出（验证自动推导） | 关闭 |
| `--rope-tensor <path>` | 外置 rope 缓存 | 无 |
| `--device-id <id>` | 指定 device ID（省略则用第一个设备） | 自动 |
| `--checkpoint-start-pos <n>` | kvcache checkpoint 起始位置（应 128 对齐） | 未启用 |
| `--checkpoint-interval <n>` | checkpoint 保存间隔（应 128 对齐） | 未启用 |
| `--checkpoint-max-count <n>` | 最大 checkpoint 数（超限自动裁剪） | 未启用 |
| `--checkpoint-tail-overwrite <0\|1>` | 末尾槽作为滚动覆盖槽 | 未启用 |
| `--bench [n_in] [n_out]` | 随机 token 基准测试（省略时默认 128 in / 128 out） | 无 |
| `--load <path>` | 启动时加载已保存的 kvcache | 无 |
| `--save <path>` | 退出时保存 kvcache 到文件 | 无 |
| `--lora <path>` | 启动时加载并启用 LoRA | 无 |
| `-h, --help` | 帮助 | - |

### 3.2 批处理 / 会话持久化 / 基准示例

```bash
# 发送一条消息后退出
./rknn3_llm_step_decode --model Qwen3.5-0.8B.rknn --core-mask 0xff \
    --prompt "介绍一下你自己" --once

# 会话持久化：跑完保存，下次恢复继续
./rknn3_llm_step_decode --model Qwen3.5-0.8B.rknn --core-mask 0xff \
    --prompt "你好" --once --save /data/kvcache.bin

./rknn3_llm_step_decode --model Qwen3.5-0.8B.rknn --core-mask 0xff \
    --prompt "你能做些什么？" --once --load /data/kvcache.bin

# 基准测试（随机 token）
./rknn3_llm_step_decode --model Qwen3.5-0.8B.rknn --core-mask 0xff \
    --bench               # 省略参数，默认 128 in / 128 out
./rknn3_llm_step_decode --model Qwen3.5-0.8B.rknn --core-mask 0xff \
    --bench 1024 128    # 1024 输入 token → 128 输出 token
./rknn3_llm_step_decode --model Qwen3.5-0.8B.rknn --core-mask 0xff \
    --bench 256        # 仅指定 n_in=256，n_out 默认 128
```

**`--bench` 说明**：使用固定随机种子生成 `n_in` 个输入 token，先清空 KV cache，以 `ignore_eos=1`、关闭 thinking 的配置跑一轮 prefill + 外部逐 token decode，最后打印性能表（Prefill / Generate / TTFT / tokens-per-second），结束后退出。`n_in` 超过 `max_ctx_len` 时会先裁剪到 `max_ctx_len - 1`（保证至少留 1 个 decode token 位），`n_out` 超过剩余预算时再裁剪到 `max_ctx_len - n_in`。

## 4. 基础交互说明

不带 `--once` 启动后，可循环输入文字进行对话（默认多轮对话保持历史，可用 `--keep-history off` 关闭），输入 `/exit` 退出。

```
==============================INFO==============================
  model            : Qwen3.5-0.8B.rknn

...
=============================================================
  Type your message, input /exit to quit.

> 你好
你好！有什么可以帮你的吗？
> 你是谁？
你好！我是 Qwen3.5...
> /exit
```

## 5. LoRA 使用

支持 `.lora_weight` 文件（参考 `examples/Qwen3_VL_LoRA` 导出流程）。本示例通过启动参数 `--lora <path>` 加载并启用：

```bash
./rknn3_llm_step_decode --model Qwen2.5-0.5B-Instruct.rknn --core-mask 0xff \
    --lora Qwen2.5-0.5B-Instruct.lora_weight
```

- 调用流程（`lora.cc`）：`rknn3_lora_init` → `query(LORA_NUM)` → `query(LORA_INFO)` → `rknn3_lora_load` → `rknn3_session_enable_lora`。
- LoRA 在启动时随 session 一起启用；进程退出时自动 disable + unload。
- 启动后的状态信息会显示 LoRA 状态：`<name> (enabled)` / `<name> (loaded)` / `none`。

## 6. 外置 rope

部分模型把 rope cos/sin 缓存放在 host 端，推理时通过 session 的 `input_callback` 按步填充。通过 `--rope-tensor <path>` 支持：

```bash
./rknn3_llm_step_decode --model Qwen3.5-4B.rknn --core-mask 0xff \
    --rope-tensor Qwen3.5-4B.safetensors
```

- 加载后自动查询模型 rope 输入张量（`rope_cos_cache` / `rope_sin_cache` / `per_layer_inputs`）并注册 `input_callback`。
- 自动识别两种格式：Qwen3.5（`rope_cos_cache` / `rope_sin_cache`）与 Gemma-4（`rope_cos_cache_0/1`、`rope_sin_cache_0/1`）。
- **强制校验**：当模型配置 `rope_cache_host_storage != 0` 时，未提供 `--rope-tensor` 或加载失败会**直接终止**启动；若模型不需要 host rope 而提供了 `--rope-tensor`，则忽略该参数并告警。

## 7. KV cache checkpoint 策略

针对 linear / sliding attention 模型（Qwen3.5 / Gemma4 系列）提供 checkpoint 式 KV cache 策略，通过启动参数配置：

```bash
./rknn3_llm_step_decode --model Qwen3.5-4B.rknn --core-mask 0xff \
    --checkpoint-start-pos 0 --checkpoint-interval 128 --checkpoint-max-count 8 --checkpoint-tail-overwrite 1
```

## 8. 实现说明

- 会话初始化 / 回调 / 推理流程：`rknn3_init` → `rknn3_load_model_from_path` → `rknn3_model_init` → `rknn3_session_init` → `rknn3_session_set_callback` → `rknn3_session_run`。
- 多轮对话通过 `--keep-history <on/off>`（默认 `on`）控制 `rknn3_llm_infer_param.keep_history`；`--once` 批处理跑完即退出。
- 采样参数在启动时经 `rknn3_llm_param.sampling_param` 配置，并通过 `rknn3_session_set_llm_param` 下发；checkpoint 策略通过 `rknn3_session_set_kvcache_policy(SAVE_CHECKPOINT)` 应用。
- 与 `rknn3_llm_cli` 共享的推理引擎位于 `cpp/engine.{h,cc}`；本 demo 特有逻辑只有 `step_main.cc`（逐 token 循环由 `engine.cc::run_external_decode()` 实现）与 checkpoint 参数。
