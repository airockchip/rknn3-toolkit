# RKNN3 多卡分段推理示例

本目录提供 RKNN3 多卡分段 LLM 推理的 Python 示例脚本，对应 C++ 版本 `rknn3_model_zoo/examples/multicard/cpp/main.cc`。

脚本将一个分段模型（如 `Qwen3.5-9B-llm_seg0.rknn` / `seg1.rknn`）分配到多张 NPU 设备上，通过 pipeline 方式串联执行 prefill 与 decode，实现跨卡推理。

| 脚本 | 适用模型 | 特色 |
|---|---|---|
| `rknn_multicard_qwen35_test.py` | Qwen3.5 系列 | 支持 `--enable_thinking` 思考模式 |
| `rknn_multicard_gemma4_test.py` | Gemma4 系列 | 支持 host RoPE cache，需 transformers>=5.5.0 |

## 1. 工作原理

```
┌─────────┐     ┌─────────┐            ┌─────────┐
│ Stage 0 │────▶│ Stage 1 │────▶ ... ─▶│Stage N-1│
│ (卡 0)  │ embed│ (卡 1)  │ embed      │ (卡 N-1)│
└─────────┘     └─────────┘            └─────────┘
   prompt          embed               result_callback
   token化          传递               采样 & 输出
```

- **Stage 0**：接收文本 prompt，内部 token化 + embedding lookup，执行 prefill，通过 `output_callback` 捕获 hidden states
- **Stage 1 ~ N-2**：从上游 `StageSlot` 获取 hidden states 作为 embedding 输入，执行推理，将输出传递给下游
- **Stage N-1（最后一段）**：执行最终推理，通过 `result_callback` 进行采样并输出 token

每一段运行在独立的 NPU 设备上，通过线程间的 `StageSlot`（带条件变量的队列）传递中间结果。

---

## 2. Qwen3.5 多卡推理（rknn_multicard_qwen35_test.py）

### 2.1 文件说明

| 文件/目录 | 默认值 | 说明 |
|---|---|---|
| 分段 RKNN 模型 | `/userdata/model/Qwen3.5-9B-llm_seg0.rknn` | stage0 的 RKNN 模型路径，`_segN` 自动生成 |
| 分段权重文件 | 自动推导 | stage0 的权重路径，默认从 rknn_path 推导（`.rknn` → `.weight`） |
| Tokenizer 目录 | `./qwen3_5_9B` | HuggingFace tokenizer 目录或名称 |
| Token embedding | `/userdata/model/Qwen3.5-9B-llm.embed.bin` | token embedding 二进制文件，`float16` 格式 |

### 2.2 基本用法

```bash
python rknn_multicard_qwen35_test.py \
    --rknn_path /userdata/model/Qwen3.5-9B-llm_seg0.rknn \
    --tokenizer_path ./qwen3_5_9B \
    --embed_path /userdata/model/Qwen3.5-9B-llm.embed.bin \
    --stage_count 2 \
    --max_new_token 1024
```

### 2.3 指定设备 ID

```bash
python rknn_multicard_qwen35_test.py \
    --rknn_path /userdata/model/Qwen3.5-9B-llm_seg0.rknn \
    --tokenizer_path ./qwen3_5_9B \
    --embed_path /userdata/model/Qwen3.5-9B-llm.embed.bin \
    --stage_count 2 \
    --device_id "0:0:0:0" \
    --device_id "1:0:0:0"
```

### 2.4 启用思考模式

```bash
python rknn_multicard_qwen35_test.py \
    --rknn_path /userdata/model/Qwen3.5-9B-llm_seg0.rknn \
    --tokenizer_path ./qwen3_5_9B \
    --embed_path /userdata/model/Qwen3.5-9B-llm.embed.bin \
    --stage_count 2 \
    --enable_thinking
```

### 2.5 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--rknn_path` | str | `/userdata/model/Qwen3.5-9B-llm_seg0.rknn` | stage0 的 RKNN 模型路径，`_segN` 自动生成 |
| `--weight_path` | str | 自动推导 | stage0 的权重路径，默认从 rknn_path 推导 |
| `--tokenizer_path` | str | `./qwen3_5_9B` | HuggingFace tokenizer 路径 |
| `--embed_path` | str | `/userdata/model/Qwen3.5-9B-llm.embed.bin` | embedding bin 路径（float16） |
| `--target` | str | `rk1820` | RKNN3 目标设备类型 |
| `--core_mask` | str (hex) | `0xff` | NPU 核心掩码 |
| `--stage_count` | int | `2` | 分段数/设备数 |
| `--bucket_size` | int | `128` | output_callback 分桶 token 数 |
| `--pipeline_mode` | `sequential`\|`threaded` | `threaded` | 多卡执行模式；threaded 可重叠执行，sequential 更安全 |
| `--max_context_len` | int | `0` | 0 表示使用模型查询到的 max_ctx_len |
| `--max_new_token` | int | `1024` | decode 循环最大次数（`--max_new_tokens` 别名兼容） |
| `--prompt` | str | 默认 prompt | prompt 字符串 |
| `--ignore_eos` | flag | False | 遇到 EOS token 不停止解码 |
| `--enable_thinking` | flag | False | 启用 Qwen3.5 思考模式 |
| `--special_eos_id` | int (可重复) | 151645 | 特殊 EOS token ID，可多次指定 |
| `--special_bos_id` | int (可重复) | 151643 | 特殊 BOS token ID，可多次指定 |
| `--presence_penalty` | float | `1.5` | 采样 presence penalty |
| `--kvcache_policy` | `default`\|`normal`\|`recurrent` | `default` | KV-cache 策略 |
| `--device_id` | str (可重复) | 自动检测 | 每段的设备 ID，按 stage 顺序重复使用 |
| `--decrypt_key_path` | str | None | 加密模型解密密钥路径 |
| `--verbose` | flag | False | 打印详细调试信息 |

---

## 3. Gemma4 多卡推理（rknn_multicard_gemma4_test.py）

### 3.1 文件说明

| 文件/目录 | 默认值 | 说明 |
|---|---|---|
| 分段 RKNN 模型 | `gemma-4-12B-it_seg0.rknn` | stage0 的 RKNN 模型路径，`_segN` 自动生成 |
| 分段权重文件 | 自动推导 | stage0 的权重路径，默认从 rknn_path 推导（`.rknn` → `.weight`） |
| Tokenizer 目录 | `gemma-4-12B-it` | HuggingFace tokenizer 目录或名称 |
| Token embedding | `gemma-4-12B-it.embed.bin` | token embedding 二进制文件，`float16` 格式 |
| RoPE cache 文件 | `rope_caches.safetensors` | Gemma4 host RoPE cache，分段模型暴露 rope cache 输入时必需 |

> **依赖要求**：Gemma4 脚本需要 `transformers>=5.5.0`。

### 3.2 基本用法

```bash
python rknn_multicard_gemma4_test.py \
    --rknn_path gemma-4-12B-it_seg0.rknn \
    --tokenizer_path gemma-4-12B-it \
    --embed_path gemma-4-12B-it.embed.bin \
    --rope_path rope_caches.safetensors \
    --stage_count 2 \
    --max_new_token 1024
```

### 3.3 指定设备 ID

```bash
python rknn_multicard_gemma4_test.py \
    --rknn_path gemma-4-12B-it_seg0.rknn \
    --tokenizer_path gemma-4-12B-it \
    --embed_path gemma-4-12B-it.embed.bin \
    --rope_path rope_caches.safetensors \
    --stage_count 2 \
    --device_id "0:0:0:0" \
    --device_id "1:0:0:0"
```

### 3.4 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--rknn_path` | str | `gemma-4-12B-it_seg0.rknn` | stage0 的 RKNN 模型路径，`_segN` 自动生成 |
| `--weight_path` | str | 自动推导 | stage0 的权重路径，默认从 rknn_path 推导 |
| `--tokenizer_path` | str | `gemma-4-12B-it` | HuggingFace tokenizer 路径 |
| `--embed_path` | str | `gemma-4-12B-it.embed.bin` | token embedding bin 路径（float16） |
| `--rope_path` / `--safetensors_path` | str | `rope_caches.safetensors` | RoPE cache safetensors 路径；两个参数名互为别名 |
| `--target` | str | `rk1820` | RKNN3 目标设备类型 |
| `--core_mask` | str (hex) | `0xff` | NPU 核心掩码 |
| `--stage_count` | int | `2` | 分段数/设备数 |
| `--bucket_size` | int | `128` | output_callback 分桶 token 数 |
| `--pipeline_mode` | `sequential`\|`threaded` | `threaded` | 多卡执行模式；threaded 可重叠执行，sequential 更安全 |
| `--max_context_len` | int | `0` | 0 表示使用模型查询到的 max_ctx_len |
| `--max_new_token` | int | `1024` | decode 循环最大次数（`--max_new_tokens` 别名兼容） |
| `--prompt` | str | 默认 prompt | prompt 字符串 |
| `--ignore_eos` | flag | False | 遇到 EOS token 不停止解码 |
| `--special_eos_id` | int (可重复) | `1, 50, 106` | 特殊 EOS token ID，可多次指定 |
| `--special_bos_id` | int (可重复) | 2 | 特殊 BOS token ID，可多次指定 |
| `--presence_penalty` | float | `0.0` | 采样 presence penalty |
| `--logits_name` | str | `logits_gathered` | 最后一段 logits tensor 名称 |
| `--kvcache_policy` | `default`\|`normal`\|`recurrent` | `default` | KV-cache 策略 |
| `--device_id` | str (可重复) | 自动检测 | 每段的设备 ID，按 stage 顺序重复使用 |
| `--decrypt_key_path` | str | None | 加密模型解密密钥路径 |
| `--verbose` | flag | False | 打印详细调试信息 |

---

## 4. 两个脚本的差异对比

| 特性 | `rknn_multicard_qwen35_test.py` | `rknn_multicard_gemma4_test.py` |
|---|---|---|
| Chat template | `<\|im_start\|>` 格式 | `<\|turn\|>` 格式 |
| `--enable_thinking` | ✅ 支持 | ❌ 无 |
| `--safetensors_path` / `--rope_path` | ❌ 无 | `--rope_path`（`--safetensors_path` 别名） |
| `--logits_name` | ❌ 无（固定 `output`） | ✅ 默认 `logits_gathered` |
| `--presence_penalty` 默认值 | `1.5` | `0.0` |
| 默认 EOS ID | 151645 | 1, 50, 106 |
| 默认 BOS ID | 151643 | 2 |
| transformers 版本要求 | 无特殊要求 | ≥5.5.0 |
| `input_callback` 处理 | ❌ 无 | 仅 rope cache |
| `per_layer_inputs` | 不处理（与 C++ 一致） | 不处理（与 C++ 一致） |

## 5. 与 C++ 版本的对应关系

| C++ (`main.cc`) | Python |
|---|---|
| `StageSlot` (mutex + cv + deque) | `StageSlot` (Lock + Condition + deque) |
| `PipelineState` | `PipelineState` |
| `StageRuntime` | `StageRuntime` |
| `stage_output_callback` | `stage_output_callback` |
| `embed_callback` | `embed_callback` |
| `result_callback` | `result_callback` |
| `run_stage_worker` (std::thread) | `run_stage_worker` (threading.Thread) |
| `run_pipeline_once` | `run_pipeline_once` |
| `replace_seg_suffix` (lambda) | `replace_seg_suffix` (regex) |
| `init_tokenizer_and_embedding` | `AutoTokenizer` + `np.memmap` |
| `init_stage` | `init_stage` |

> **`per_layer_inputs` 说明**：Gemma4 脚本的 `input_callback` 只处理 rope cache，不处理 `per_layer_inputs`（直接跳过）。Qwen3.5 脚本不支持 rope 外置。

## 6. 注意事项

1. **模型分段**：需要提前将完整模型切分为 `_seg0`, `_seg1`, ... `_segN` 段
2. **设备数量**：连接的 NPU 设备数需 ≥ `--stage_count`
3. **Embedding 文件**：float16 格式，大小为 `vocab_size × embedding_dim × 2` 字节
4. **RoPE cache**：仅 Gemma4 脚本支持，当模型配置了 `rope_cache_host_storage != 0` 时需要提供；Qwen3.5 脚本不支持 rope 外置
5. **keep_history**：多卡场景下必须保持 `keep_history=True`（代码中已硬编码为 1），因为每次 `session_run` 只执行一步 prefill，需要 KV cache 跨 step 保留
6. **性能统计**：脚本输出 prefill 和 decode 阶段的 token/s 统计
7. **pipeline_mode**：`threaded` 模式使用多线程重叠执行，可能提升吞吐但增加 ctypes 稳定性风险；`sequential` 模式在调用线程上依次执行各 stage，更安全
