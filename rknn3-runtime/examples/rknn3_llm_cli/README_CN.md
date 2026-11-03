# rknn3_llm_cli 说明文档

`rknn3_llm_cli` 是一个运行 RKNN3 LLM 模型的交互式命令行聊天客户端（类似 llama.cpp / ollama 的 CLI）。它基于 `examples/rknn3_session_test_demo` 的会话调用方式，提供 REPL 交互输入，并内置丰富的斜杠命令，可在运行时调整采样参数、清空 KV cache、切换聊天模板、跑性能基准、保存/加载 KV cache 等。

> 本 demo 与 `rknn3_llm_step_decode`（逐 token 外部解码）**共享同一套推理引擎**（`cpp/` 目录，一次编译产出两个二进制）。逐 token 解码用法见 [`README_STEP_DECODE_CN.md`](README_STEP_DECODE_CN.md)。

## 1. 编译

### 1.1 独立编译脚本

```bash
cd examples/rknn3_llm_cli/

# Linux（x86 本机直接 -t x86 -a x86_64；交叉编译需先 export GCC_COMPILER）
./build-linux.sh -t rk3588 -a aarch64 -b Release
# 产物位于 install/rknn3_llm_cli_rk3588_linux_aarch64/
#   rknn3_llm_cli          交互式 CLI（本 demo）
#   rknn3_llm_step_decode  逐 token 解码 demo

# Android（需要先 export ANDROID_NDK_PATH）
./build-android.sh -t rk3588 -a arm64-v8a -b Release
```

## 2. 部署与运行

```bash
adb push install/rknn3_llm_cli_rk3588_linux_aarch64/ /data/

adb shell
cd /data/rknn3_llm_cli_rk3588_linux_aarch64
export LD_LIBRARY_PATH=./lib

# 交互式聊天（只需 --model，其余自动推导）
./rknn3_llm_cli --model model/Qwen2.5-0.5B-Instruct.rknn --core-mask 0xff

# 或显式指定全部四个路径
./rknn3_llm_cli \
    --model model/Qwen2.5-0.5B-Instruct.rknn \
    --weight model/Qwen2.5-0.5B-Instruct.weight \
    --vocab model/Qwen2.5-0.5B-Instruct.tokenizer.gguf \
    --embed model/Qwen2.5-0.5B-Instruct.embed.bin \
    --core-mask 0xff

# 带参数启动，并自动发送一条消息后进入 REPL
./rknn3_llm_cli \
    --model model/Qwen2.5-0.5B-Instruct.rknn \
    --ctx-size 1024 --core-mask 0xff --temperature 0.8 \
    --prompt "介绍一下你自己" --once
```

> 说明：模型文件（`.rknn` / `.weight` / `.tokenizer.gguf` / `.embed.bin`）需自行准备，参考各模型示例（如 `examples/Qwen2_5`）的导出与转换流程。只传 `--model` 时，其余路径按同名自动推导（`foo.rknn` → `foo.weight` / `foo.tokenizer.gguf` / `foo.embed.bin`）。

### 2.1 命令行参数

> 只接受命名参数（`--model` 必填），不接受位置参数。

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `-m, --model <path>` | RKNN 模型文件 | 必填 |
| `--weight <path>` | 权重文件（省略时由 `--model` 自动推导） | 自动推导 |
| `--vocab <path>` | 分词器文件（省略时自动推导） | 自动推导 |
| `--embed <path>` | embedding 权重文件（省略时自动推导） | 自动推导 |
| `-c, --ctx-size <n>` | 最大上下文长度（token） | 模型配置值 |
| `-n, --predict <n>` | 每轮最大生成 token 数 | 512 |
| `--core-mask <mask>` | NPU 核心掩码，支持 16 进制/10 进制，如 `0xff` | `0xff` |
| `--device-id <id>` | 使用的 RKNN3 设备 ID（省略时使用第一个设备） | 第一个设备 |
| `-t, --temperature <v>` | 采样温度 | 1.0 |
| `--top-k <n>` | top-k | 1 |
| `--top-p <v>` | top-p | 0.9 |
| `--repeat-penalty <v>` | 重复惩罚 | 1.2 |
| `--frequency-penalty <v>` | 频率惩罚 | 0.0 |
| `--presence-penalty <v>` | 存在惩罚 | 0.0 |
| `-s, --system <prompt>` | 系统提示词 | Qwen 风格默认值 |
| `--prompt-prefix <p>` | 聊天模板前缀 | Qwen 风格默认值 |
| `--prompt-postfix <p>` | 聊天模板后缀 | Qwen 风格默认值 |
| `-p, --prompt <text\|@file\|path>` | 启动后先发送一条消息（`@file` 强制从文件读取；传入已存在的文件路径时也会自动读取文件内容） | 无 |
| `--once` | 发送 `--prompt` 后立即退出（批处理） | 关闭 |
| `--keep-history <on/off>` | 多轮对话保持历史（默认开启） | `on` |
| `--ignore-eos` | 生成时忽略 EOS | 关闭 |
| `--thinking` | 开启思考模式 | 关闭 |
| `--logits-name <name>` | 输出 logits 名称（模型有多个输出时才需要） | `output` |
| `--dry-run` | 只打印解析后的模型路径并退出（不连接设备），用于验证自动推导 | 关闭 |
| `--load <path>` | 启动时加载已保存的 kvcache 文件（恢复会话） | 无 |
| `--save <path>` | 退出时把当前 kvcache 保存到文件 | 无 |
| `--bench [n_in] [n_out]` | 启动即跑一次随机 token 性能基准后退出（不进入 REPL）；省略参数时默认 128 in / 128 out | 无 |
| `--lora <path>` | 加载并启用 LoRA 适配器 | 无 |
| `--rope-tensor <path>` | 外置 rope 缓存文件（Qwen3.5 / Gemma-4 等需要外置 rope 的模型） | 无 |
| `--verbose` | 开启详细日志 | 关闭 |
| `-h, --help` | 帮助 | - |

> 不同模型使用不同的聊天模板（如 Gemma 使用 `<|turn|>user\n...`）。通过 `--system` / `--prompt-prefix` / `--prompt-postfix` 或运行时 `/system` / `/template` 命令切换。

### 2.2 会话保存 / 恢复 / 基准示例

```bash
# 跑一条消息后把 kvcache 存盘
./rknn3_llm_cli --model foo.rknn --prompt "你好" --once --save /data/kvcache.bin

# 加载之前保存的会话继续对话
./rknn3_llm_cli --model foo.rknn --load /data/kvcache.bin

# 启动即跑性能基准（`--bench [n_in] [n_out]`，省略时默认 128 in / 128 out）
./rknn3_llm_cli --model foo.rknn --bench
./rknn3_llm_cli --model foo.rknn --bench 32 16
```

> `--save` / `--load` 与运行时 `/save` `/load` 指令等价，只是发生在进程启动/退出时；`--bench [n_in] [n_out]` 等价于运行时 `/bench`（省略参数时均默认 128 in / 128 out）。

## 3. 交互式使用

启动后进入 REPL，直接输入文字即可对话（默认保持多轮历史，即 `keep_history=1`）：

```
  rknn3-llm-cli
  model            : model/Qwen2.5-0.5B-Instruct.rknn
=============================================================
Max Context Length      : 1024
...
  Type '/help' for commands, type your message to chat, Ctrl+D to exit.

> 你好
你好！有什么可以帮你的吗？
> 你叫什么名字
我叫 Qwen，很高兴认识你！
> /clear
KV cache cleared (system prompt kept). New conversation started.
> 
```

生成过程中输入 `/stop` 可随时中断当前生成。

> ⚠️ `/stop` 实时中断**仅在交互终端（TTY）下生效**。通过管道/脚本（如 `adb shell` 非交互）运行时 stdin 不是终端，生成期间无法接收 `/stop`（此时输入 `/stop` 会落到 REPL 作为"无生成可中断"处理）。另外，TTY 下生成期间输入的其他文本会被丢弃（这是中断监控的已知取舍）。

### 3.1 斜杠命令列表

| 命令 | 说明 |
|------|------|
| `/help` 或 `/?` | 显示帮助 |
| `/exit` / `/quit` / `/bye` / `/q` | 退出程序 |
| **会话管理** | |
| `/clear` | 清空 KV cache（保留 system prompt），开启新会话 |
| `/reset` | 清空全部 KV cache（含 system prompt） |
| `/prompt <text>` | 以原始文本单轮生成（不套聊天模板，`keep_history=0`） |
| `/stop` / `/cancel` / `/interrupt` | 中断当前生成 |
| **采样 / 生成参数** | |
| `/temp <v>` | 设置 temperature |
| `/top-k <n>` | 设置 top-k |
| `/top-p <v>` | 设置 top-p |
| `/repeat-penalty <v>` | 设置 repeat penalty |
| `/frequency-penalty <v>` | 设置 frequency penalty |
| `/presence-penalty <v>` | 设置 presence penalty |
| `/sampling <top_k> <top_p> <temp> <repeat> <freq> <presence>` | 一次设置全部采样参数 |
| `/predict <n>` 或 `/max-tokens <n>` | 设置每轮最大生成 token 数 |
| `/thinking on\|off` | 开关思考模式（重新初始化 session 以让运行时按内置模板应用 thinking） |
| `/ignore-eos on\|off` | 开关忽略 EOS |
| `/keep-history on\|off` | 开关多轮历史保持（会清空 KV cache） |
| **提示词 / 模板** | |
| `/system <prompt>` | 设置系统提示词（重新初始化 session） |
| `/template <prefix> <postfix>` | 设置聊天模板前缀 / 后缀（重新初始化 session） |
| **信息 / 性能** | |
| `/status` | 显示模型 / 采样 / 会话 / 上下文信息 |
| `/ctx` | 显示当前上下文占用 |
| `/perf` | 显示上一轮生成性能（Prefill / Generate / TTFT） |
| `/stats` | 显示累计统计 |
| `/bench [n_in] [n_out]` | 随机 token 性能基准测试（默认 128 in / 128 out） |
| `/verbose on\|off` | 开关详细日志 |
| **KV cache 持久化** | |
| `/save <path>` | 保存当前 KV cache 到文件 |
| `/load <path>` | 从文件加载 KV cache |
| **LoRA** | |
| `/lora <path>` | 把 LoRA 适配器加载到 context（init+query+load，不启用） |
| `/lora-enable` | 在 session 上启用已加载的 LoRA |
| `/lora-disable` | 在 session 上禁用 LoRA（保持已加载） |
| `/lora-unload` | 从 context 卸载 LoRA |
| `/lora-off` 或 `/disable-lora` | 一步完成 disable + unload |

## 4. LoRA 使用

LoRA 适配器支持 `.lora_weight` 文件（参考 `examples/Qwen3_VL_LoRA` 的导出流程）。`加载` 与 `启用`、`禁用` 与 `卸载` 已拆分为独立命令，可逐步控制生命周期：

```bash
# 启动时加载并启用（一步到位）
./rknn3_llm_cli --model foo.rknn --lora adapter.lora_weight

# 运行时分步控制
> /lora adapter.lora_weight     # 仅加载到 context
LoRA loaded: <adapter_name>
> /lora-enable                  # 在 session 上启用
LoRA enabled: <adapter_name> (scale=1.000)
> /lora-disable                 # 仅禁用（仍保持加载）
LoRA disabled
> /lora-unload                  # 从 context 卸载
LoRA unloaded
> /lora-off                     # 或一步 disable + unload
LoRA disabled and unloaded
```

- 调用流程（`lora.cc`）：`rknn3_lora_init` → `query(LORA_NUM)` → `query(LORA_INFO)` → `rknn3_lora_load`（`/lora`）；`rknn3_session_enable_lora`（`/lora-enable`）；`rknn3_session_disable_lora`（`/lora-disable`）；`rknn3_lora_unload`（`/lora-unload`）。
- `/thinking` `/system` `/template` 会重建 session，重建后会自动重新绑定已启用（enabled）的 LoRA；进程退出时自动 disable + unload。
- `/status` 会显示 LoRA 状态：`<name> (enabled)` / `<name> (loaded)` / `none`。

## 4.1 外置 rope

部分模型（Qwen3.5 / Gemma-4，`rope_cache_host_storage=1`）把 rope cos/sin 缓存放在 host 端，推理时通过 session 的 `input_callback` 按步填充。CLI 通过 `--rope-tensor <path>` 支持：

```bash
# Qwen3.5 / Gemma-4 等需要外置 rope 的模型
./rknn3_llm_cli --model foo.rknn --rope-tensor rope_cos_sin.bin
```

- 加载后会自动查询模型的 rope 输入张量（`rope_cos_cache` / `rope_sin_cache`）并注册 `input_callback`（参考 `examples/multicard`）。
- 自动识别两种格式：Qwen3.5（`rope_cos_cache` / `rope_sin_cache`）与 Gemma-4（`rope_cos_cache_0/1`、`rope_sin_cache_0/1`）。
- 实现位于 `rope.cc`（`rope_setup` → `load_rope_cache` + `query_ext_input_indices`；`rope_input_callback` 按 `param.pos` 拷贝对应片段到输入张量）。
- **强制校验**：当模型配置 `rope_cache_host_storage != 0` 时，未提供 `--rope-tensor` 或加载失败会**直接终止**启动（不再仅告警后继续），避免"启动成功但推理错误"。若模型不需要 host rope（`rope_cache_host_storage=0`）而提供了 `--rope-tensor`，则忽略该参数并告警。

## 5. Thinking 模式说明

不同模型的 thinking 模式由运行时通过输入 tensor 上的 `enable_thinking` 字段控制（`rknn3_llm_tensor`）。每个模型自带聊天模板，运行时以 `enable_thinking` 标志渲染该模板，无需 CLI 侧额外处理。CLI 不再维护任何 thinking 后缀逻辑。

> 注意：`/thinking off` 能可靠地抑制常规问题的思考输出；但"倾向思考"的模型对需要逐步推理的复杂问题（如数学题）即使关闭 thinking 仍可能输出推理内容，这是模型自身行为。

## 6. 实现说明

- 会话初始化 / 回调 / 推理流程与 `examples/rknn3_session_test_demo` 一致：`rknn3_init` → `rknn3_load_model_from_path` → `rknn3_model_init` → `rknn3_session_init` → `rknn3_session_set_callback` → `rknn3_session_run`。
- 多轮对话依赖 `rknn3_llm_infer_param.keep_history = 1`，此时 session 自动通过 `rknn3_session_set_chat_template` 设置的模板包裹每轮输入并累计历史；`/clear` 调用 `rknn3_session_clear_kvcache`。
- 运行时修改采样参数通过 `rknn3_session_set_llm_param` 下发。
- `/stop` 通过独立线程运行 `rknn3_session_run`，主线程监听 stdin 并调用 `rknn3_session_stop` 实现中断。
- 源码结构（`cpp/`，与 `rknn3_llm_step_decode` 共享）：`helpers.h`（共享类型 / 工具）、`args.{h,cc}`（命令行解析）、`engine.{h,cc}`（共享推理引擎，按 `SessionState::decode_mode` 分派一次调用 / 逐 token 解码）、`rope.{h,cc}`（外置 rope）、`lora.{h,cc}`（LoRA）、`cli_main.cc`（本 demo 前端）、`step_main.cc`（step_decode 前端）。
