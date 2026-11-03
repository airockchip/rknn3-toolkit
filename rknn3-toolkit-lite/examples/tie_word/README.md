# rknn3_session_test_tie_word.py 使用说明

## 1. 测试目的

该脚本用于验证 tie-word LLM 模型的 Session 推理流程，主要包含：

- 加载 tie-word `.rknn` 与 `.weight` 模型；
- 强制设置 `RKNN_ENABLE_TIE_WORD_EMBEDDINGS=1`；
- 使用 `transformers.AutoTokenizer` 提供 tokenizer callback；
- **不加载 `embedding.bin`，但注册一个 dummy `embed_callback`（runtime 要求函数指针非 NULL）**；
- 使用 separated init 流程：`load_rknn -> init_runtime -> rknn3_query -> init_llm_session -> session_run`；
- 设置 `RKNN3_KVCACHE_POLICY_SAVE_CHECKPOINT`；
- checkpoint 参数与 C++ demo 保持一致：
  - `checkpoint_start_pos = 0`
  - `checkpoint_interval = 128`
  - `max_checkpoint_count = 100`
- 每轮推理后查询 Session state；
- 当 `n_total_tokens >= n_max_tokens - max_new_tokens` 时执行 `RKNN3_KVCACHE_CLEAR_ALL`；
- 打印 Prefill / Generate 性能统计。

## 2. 与普通 `rknn_session_test.py` 的主要区别

普通版本需要读取 `embedding.bin`，并通过 `embed_callback` 根据 token id 查 embedding。

Tie-word 模型把 word embedding 与输出权重共享，因此 Python tie-word 版本不需要：

```text
*.embed.bin
```

## 3. Tokenizer 注意事项

C++ demo 的参数是 `tokenizer.gguf`，而这个 Python 版本为了与参考 `rknn_session_test.py` 保持一致，使用：

```python
AutoTokenizer.from_pretrained(...)
```

因此 `--tokenizer_path` 应传入 Hugging Face tokenizer 名称或一个可被 `transformers.AutoTokenizer` 加载的本地 tokenizer 目录，例如：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

或：

```text
/path/to/Qwen2.5-0.5B-Instruct-tokenizer/
```

不要直接把 C++ 使用的 `*.tokenizer.gguf` 路径传给这个参数，除非你的 `transformers` 环境本身支持该路径格式。

## 4. 依赖

需要设备端已经正确安装/部署：

- `rknn3lite`
- `transformers`
- `numpy`（`transformers` 的 `return_tensors='np'` 会使用）
- 与模型匹配的 RKNN3 runtime / driver

脚本会在导入 `rknn3lite` 前设置：

```bash
export RKNN_ENABLE_TIE_WORD_EMBEDDINGS=1
```

Python 代码内部等价于：

```python
os.environ["RKNN_ENABLE_TIE_WORD_EMBEDDINGS"] = "1"
```

## 5. 模型文件

示例：

```text
model/qwen2.5_0.5b_instruct/
├── Qwen2.5-0.5B-Instruct-tie-word.rknn
└── Qwen2.5-0.5B-Instruct-tie-word.weight
```

Tie-word Python 测试不需要 `Qwen2.5-0.5B-Instruct.embed.bin`。

## 6. 执行命令

假设脚本和模型都已经放到设备上：

```bash
python3 rknn3_session_test_tie_word.py \
  --rknn_path model/qwen2.5_0.5b_instruct/Qwen2.5-0.5B-Instruct-tie-word.rknn \
  --weight_path model/qwen2.5_0.5b_instruct/Qwen2.5-0.5B-Instruct-tie-word.weight \
  --tokenizer_path Qwen/Qwen2.5-0.5B-Instruct \
  --max_context_len 2048 \
  --max_new_token 512 \
  --core_mask 0xff \
  --keep_history 0 \
  --ignore_eos_token 0
```

如果 tokenizer 已经放在设备本地，建议使用本地目录，避免运行时下载：

```bash
python3 rknn3_session_test_tie_word.py \
  --rknn_path model/qwen2.5_0.5b_instruct/Qwen2.5-0.5B-Instruct-tie-word.rknn \
  --weight_path model/qwen2.5_0.5b_instruct/Qwen2.5-0.5B-Instruct-tie-word.weight \
  --tokenizer_path model/qwen2.5_0.5b_instruct/tokenizer \
  --max_context_len 2048 \
  --max_new_token 512 \
  --core_mask 0xff
```

## 7. 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--rknn_path` | string | `Qwen2.5-0.5B-Instruct-tie-word.rknn` | tie-word RKNN 模型路径 |
| `--weight_path` | string | `Qwen2.5-0.5B-Instruct-tie-word.weight` | tie-word weight 路径 |
| `--tokenizer_path` | string | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face tokenizer 名称或本地目录 |
| `--max_context_len` | int | `2048` | Session 最大上下文长度，不可超过模型查询得到的 `max_ctx_len` |
| `--max_new_token` | int | `512` | 单轮最大生成 token 数 |
| `--core_mask` | hex string | `0xff` | NPU core mask |
| `--keep_history` | 0/1 | `0` | 是否保留历史上下文 |
| `--ignore_eos_token` | 0/1 | `0` | 是否忽略 EOS；为 1 时 Python 版将 `special_eos_id` 设置为 `-1` |
| `--decrypt_key_path` | string | `None` | 加密模型的 key/license 文件路径 |
| `--test_num` | int | `2` | 推理轮数；C++ demo 默认测试 2 轮 |

## 8. KV Cache 行为

Python 版本对应 C++ demo 的：

```text
RKNN3_KVCACHE_POLICY_SAVE_CHECKPOINT
```

固定使用：

```text
checkpoint_start_pos = 0
checkpoint_interval = 128
max_checkpoint_count = 100
```

每轮 `session_run()` 后调用 `session_query_state()`。当满足：

```text
n_total_tokens >= n_max_tokens - max_new_tokens
```

会调用：

```text
clear_kvcache(RKNN3_KVCACHE_CLEAR_ALL)
```

这样可以避免下一轮最大长度生成时超过当前 Session 的上下文容量。

## 9. `ignore_eos_token` 映射

Python Lite 的 LLM 参数使用 `special_eos_id`。

脚本按下面方式对应 C++ 的 `ignore_eos_token`：

```python
if ignore_eos_token:
    special_eos_id = -1
else:
    special_eos_id = tokenizer.eos_token_id
```

## 10. 预期结果

正常情况下可以看到类似流程：

```text
*******************************NEW TEST**********************************
RKNN_ENABLE_TIE_WORD_EMBEDDINGS=1
--> Loading tokenizer
done
--> Loading model
done
--> Init runtime environment
done
--> Query model info
...
--> Init LLM session
done
--> Set KV cache policy: SAVE_CHECKPOINT
...
--------------------Input[0]--------------------
...
--------------------Output----------------------
...
--------------------Run State-------------------
...
-----------------------------------------------------------------------------------------
 Stage      | Total Time (ms)  | Tokens   | Time per Token (ms)  | Tokens per Second
...
*******************************END TEST**********************************
```

测试通过的基本判断：

- 模型和 Session 初始化成功；
- 没有请求 `embedding.bin`；
- `SAVE_CHECKPOINT` policy 设置成功；
- 推理可以输出正常文本；
- Session state 可以正常查询；
- 程序最终正常释放 RKNN3 资源。

## 11. 与 C++ 参数的对应关系

| C++ 参数 | Python 参数 |
|---|---|
| `rknn_path` | `--rknn_path` |
| `weight_path` | `--weight_path` |
| `tokenizer.gguf` | `--tokenizer_path`，但 Python 版使用 HF tokenizer 目录/名称 |
| `max_context_len` | `--max_context_len` |
| `max_new_tokens` | `--max_new_token` |
| `core_mask` | `--core_mask` |
| `keep_history` | `--keep_history` |
| `ignore_eos_token` | `--ignore_eos_token` |
| `key_path` | `--decrypt_key_path` |
