# RKNN3 Grammar：让 LLM 稳定输出结构化结果

`grammar_demo.py` 用几个简单例子演示 RKNN3 的 **Grammar（受约束生成）** 功能。

Grammar 适合这种场景：**模型的输出需要直接交给程序处理，因此不能只是“尽量按格式输出”，而是必须满足指定格式。**

例如：

- 固定字段 JSON；
- 分类标签；
- 订单号、日期、Email 等固定格式字符串；
- Tool / Function Calling。
---

## 1. Grammar 解决什么问题

仅靠 prompt 约束输出格式并不稳定。

例如我们告诉模型：

```text
提取姓名和年龄，只返回 JSON。
```

没有 Grammar 时，模型仍可能输出：

````text
当然，结果如下：
```json
{"name":"Alice","age":28}
```
````

人可以看懂，但程序直接执行：

```python
json.loads(output)
```

会失败，因为前后还有解释文字和 Markdown。

使用 JSON Schema Grammar 后，可以把模型输出限制成：

```json
{"name":"Alice","age":28}
```

可以简单理解为：

```text
Prompt   ：告诉模型“希望回答什么”
Grammar  ：限制模型“允许生成什么格式”
```

Grammar 不是生成结束后再检查一次，而是在模型逐 token 生成时屏蔽不符合规则的候选 token。

> Grammar 保证的是**结构和语法**，不是事实一定正确。例如它能保证 `age` 是整数，但不能保证模型提取出的年龄一定真实。生产代码仍然需要做业务校验。

---

## 2. Demo 里有哪些例子

脚本一次初始化模型，然后依次运行 9 个场景：

| # | Grammar 类型 | 场景 | 输出示意 | 典型用途 |
| --- | --- | --- | --- | --- |
| 1 | JSON Schema | 提取 name / age / city | `{"name":"Alice","age":28,"city":"Shenzhen"}` | 信息提取、API 返回 |
| 2 | JSON Schema | 客户问题分类 | `{"category":"shipping","confidence":95}` | 意图识别、工单路由 |
| 3 | Builtin JSON | 任意合法 JSON | `{"demo":"grammar"}` | 通用 JSON 输出 |
| 4 | Regex | 订单号 | `ORD-123456` | 订单号、产品 ID |
| 5 | Regex | 日期 | `2026-08-27` | 日期等固定格式 |
| 6 | Regex | Email | `user@example.com` | 邮箱、编码字符串 |
| 7 | EBNF | yes / no / maybe | `yes` | 固定选项、状态值 |
| 8 | EBNF | 情感分类 | `positive` | 分类标签 |
| 9 | Structural Tag | Tool Call | `<tool_call>...<parameter=location>...</parameter>...</tool_call>` | Tool / Function Calling |

因此这个 demo 主要让客户看到五种用法：

```text
JSON Schema     -> 固定结构的业务数据
Builtin JSON    -> 任意合法 JSON
Regex           -> 固定字符串格式
EBNF            -> 固定选项 / 自定义语法
Structural Tag  -> Tool Calling 等组合结构
```

---

## 3. 前置条件

运行前请确认：

1. 设备端 `librknn3_api.so` 支持 RKNN3 Grammar 相关接口；
2. Python 环境安装了 `numpy`、`transformers`；
3. `.rknn`、`.weight`、`.embed.bin` 和 tokenizer 对应同一模型；
4. Python 可以正确 import `rknn3lite`。

如果 Toolkit Lite 没有通过 pip 安装，可先设置：

```bash
export PYTHONPATH=<toolkit-lite-python-root>:${PYTHONPATH}
```

典型部署目录：

```text
<deployment-root>/
├── grammar_demo.py
├── Qwen2.5-0.5B-Instruct.rknn
├── Qwen2.5-0.5B-Instruct.weight
├── Qwen2.5-0.5B-Instruct.embed.bin
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

Tokenizer 也可以直接使用 Hugging Face model id。

---

## 4. 最快运行方式

脚本默认配置和 `rknn_session_test.py` 类似：

```python
RKNN_MODEL     = "Qwen2.5-0.5B-Instruct.rknn"
EMBED_PATH     = "Qwen2.5-0.5B-Instruct.embed.bin"
TOKENIZER_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
```

如果模型文件在当前目录，可直接运行：

```bash
python3 grammar_demo.py
```

也可以指定路径：

```bash
python3 grammar_demo.py \
  --rknn_path ./Qwen2.5-0.5B-Instruct.rknn \
  --embed_path ./Qwen2.5-0.5B-Instruct.embed.bin \
  --tokenizer_path Qwen/Qwen2.5-0.5B-Instruct \
  --core_mask 0xff \
  --max_new_token 256
```

`.weight` 路径由 `.rknn` 路径自动推导：

```text
Qwen2.5-0.5B-Instruct.rknn
Qwen2.5-0.5B-Instruct.weight
```

所以不需要单独传 `--weight_path`。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--rknn_path` | `Qwen2.5-0.5B-Instruct.rknn` | RKNN 模型路径 |
| `--embed_path` | `Qwen2.5-0.5B-Instruct.embed.bin` | embedding 文件路径 |
| `--tokenizer_path` | `Qwen/Qwen2.5-0.5B-Instruct` | tokenizer 路径或 HF model id |
| `--core_mask` | `0xff` | NPU core mask |
| `--max_new_token` | `256` | 每个示例最大生成 token 数 |

---

## 5. Demo 的代码流程

整体流程和普通 `rknn_session_test.py` 很接近：

```text
Load tokenizer / embedding
          │
          ▼
Load RKNN model
          │
          ▼
Init runtime
          │
          ▼
Query LLM config
          │
          ▼
Init LLM session
          │
          ▼
Init Grammar       ← 只初始化一次
          │
          ▼
set_grammar(...)
          │
          ▼
session_run(...)
          │
          ▼
clear_grammar()
          │
          ▼
设置下一个 Grammar
```

对于已经会使用 RKNN3 LLM session 的客户，主要关注下面三个接口即可。

### 5.1 初始化 Grammar

```python
stop_tokens = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

token_pieces = (
    tokenizer.decode([i], skip_special_tokens=False,
                     clean_up_tokenization_spaces=False)
    for i in range(vocab_size)
)

rknn.init_grammar(
    token_pieces=token_pieces,
    stop_tokens=stop_tokens,
)
```

`init_grammar()` 会建立 Grammar 引擎需要的 token-id → token piece 映射，同一个 RKNN context 中只需要初始化一次。

### 5.2 每次请求前设置 Grammar

```python
rknn.set_grammar(grammar, grammar_type)
```

然后正常执行：

```python
rknn.session_run(
    prompt=prompt,
    keep_history=False,
    max_new_tokens=max_new_token,
)
```

### 5.3 当前请求结束后清除 Grammar

```python
rknn.clear_grammar()
rknn.clear_kvcache()
```

之后即可设置另一条 Grammar 继续运行。

---

## 6. JSON Schema：固定结构的 JSON

### 示例 1：信息提取

Demo 输入：

```text
Extract: Alice is 28 years old and lives in Shenzhen. Reply only with JSON.
```

规则：

```python
PERSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "city"],
    "additionalProperties": False,
})
```

使用：

```python
rknn.set_grammar(
    PERSON_SCHEMA,
    RKNN3GrammarType.RKNN3_GRAMMAR_JSON_SCHEMA,
)
```

输出会被限制为类似：

```json
{"name":"Alice","age":28,"city":"Shenzhen"}
```

这个 Schema 限制了：

- 必须输出 JSON object；
- 必须包含 `name`、`age`、`city`；
- `age` 必须为 `0~150` 的整数；
- `additionalProperties: false` 禁止额外字段。

### 示例 2：分类结果

Demo 还定义了：

```python
"category": {
    "type": "string",
    "enum": ["refund", "shipping", "product", "other"]
}
```

所以模型分类后，`category` 只能是四个值之一。例如：

```json
{"category":"shipping","confidence":95}
```

JSON Schema 很适合：结构化信息提取、API 返回、表单填充、分类结果和 Agent 中间数据。

---

## 7. Builtin JSON：只要求输出合法 JSON

如果不需要固定字段，只要求输出是合法 JSON，可以使用：

```python
rknn.set_grammar(
    None,
    RKNN3GrammarType.RKNN3_GRAMMAR_BUILTIN_JSON,
)
```

Demo 中会让模型生成一个小 JSON object，例如：

```json
{"name":"RKNN3 Grammar","type":"demo"}
```

两种 JSON Grammar 的选择可以简单理解为：

```text
只要求“必须是合法 JSON”      -> Builtin JSON
要求字段 / 类型 / 范围 / 枚举 -> JSON Schema
```

实际业务如果已经知道接口结构，通常优先使用 JSON Schema。

---

## 8. Regex：订单号、日期、Email 等固定格式

Regex 适合不需要 JSON、但字符串格式必须固定的场景。

Demo 中包含三个例子：

```python
# 订单号，例如 ORD-123456
r"ORD-[0-9]{6}"

# 日期，例如 2026-08-27
r"[0-9]{4}-[0-9]{2}-[0-9]{2}"

# Email，例如 user@example.com
r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
```

同样可以扩展到 UUID、产品编号、版本号、固定前后缀编码等场景。

> Regex 保证的是字符串匹配。例如 demo 的日期规则能保证输出长得像 `YYYY-MM-DD`，但不能单独判断 `2026-99-99` 是否是真实日期，业务端仍应做语义校验。

---

## 9. EBNF：固定选项和自定义语法

如果模型只能从几个结果中选择，EBNF 很直观。

例如：

```python
'root ::= "yes" | "no" | "maybe"\n'
```

最终只能生成：

```text
yes
no
maybe
```

情感分类示例：

```python
'root ::= "positive" | "negative" | "neutral"\n'
```

适合 yes/no 判断、分类标签、状态值、路由标签、固定命令，以及比简单枚举更复杂的自定义语法。

---

## 10. Structural Tag：Tool Calling

最后一个例子演示 Tool Calling 风格的结构约束。

这个 demo 固定调用 `get_weather`，并明确规定它只有一个 `location` 参数。期望输出类似：

```text
<tool_call>
<function=get_weather>
<parameter=location>Shenzhen</parameter>
</function>
</tool_call>
```

对应规则把整个 Tool Call 拆成三部分：

```python
STRUCTURAL_TAG = json.dumps({
    "type": "structural_tag",
    "format": {"type": "sequence", "elements": [
        {"type": "const_string", "value": "<tool_call>\n<function=get_weather>\n"},
        {"type": "json_schema", "json_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "maxLength": 64},
            },
            "required": ["location"],
            "additionalProperties": False,
        }, "style": "qwen_xml"},
        {"type": "const_string", "value": "\n</function>\n</tool_call>"},
    ]},
})
```

这里有两个重要点：

- function name 直接固定为 `get_weather`，方便客户看懂这个例子到底在约束什么；
- 参数 Schema 明确定义 `location`，并设置 `additionalProperties: false`，不允许模型自行生成其他参数名。

不要在这个例子里使用完全开放的 `{"type":"object","additionalProperties":true}`。开放参数对象会让约束边界过宽，模型可能持续生成参数内容，甚至重复进入新的 `<function=...>` 段，最后一直生成到 `--max_new_token` 上限。

Structural Tag 可以把固定字符串、JSON Schema 等规则组合成一个完整结构，适合 Tool Calling、Function Calling、Agent action 或自定义 tag/XML 协议。实际接入多个工具时，可以分别为每个工具定义明确的参数 Schema。

> 输出结构合法不代表工具就可以直接执行。生产环境仍应校验 function name 白名单、参数内容和用户权限。

---

## 11. 如何增加自己的例子

所有例子都集中在脚本末尾的 `examples` 中：

```python
examples = [
    (name, prompt, grammar, grammar_type),
    ...
]
```

例如新增产品 ID：

```text
PROD-AB-1234
```

只需添加：

```python
("Regex - product id",
 "Generate one product id.",
 r"PROD-[A-Z]{2}-[0-9]{4}",
 RKNN3GrammarType.RKNN3_GRAMMAR_REGEX),
```

不需要修改主流程。

如果是一个固定 JSON API，则先写 Schema，再加入 `examples`：

```python
ANSWER_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["answer", "score"],
    "additionalProperties": False,
})
```

```python
("JSON Schema - answer",
 "Answer the question and return answer and score.",
 ANSWER_SCHEMA,
 RKNN3GrammarType.RKNN3_GRAMMAR_JSON_SCHEMA),
```

这个 demo 刻意使用简单的列表和顺序式代码，就是为了让客户可以复制一个现有例子，只修改 prompt 和 grammar。

---

## 12. Grammar 的工作原理

Grammar 在 decode 阶段参与 token 选择：

```text
模型输出当前 token 的 logits
              │
              ▼
Grammar 根据当前规则状态判断哪些 token 合法
              │
              ▼
屏蔽非法 token
              │
              ▼
采样器只从合法 token 中选择
              │
              ▼
Grammar 接受该 token，状态继续前进
```

所以它和“生成完成后再用 Python validator 检查”不同。

例如 JSON 还没有正确闭合时，Grammar 会阻止模型选择会立即破坏当前 JSON 语法的 token。

---

## 13. 使用时需要注意什么

### Grammar 保证格式，不保证事实

Grammar 可以保证：

```json
{"name":"Alice","age":28}
```

中的 `age` 是整数，但不能保证 Alice 的真实年龄一定是 28。

生产环境建议组合使用：

```text
Grammar 约束
    +
JSON / 参数解析
    +
字段范围校验
    +
业务规则校验
    +
权限检查
```

### `init_grammar()` 只初始化一次

推荐流程：

```text
程序启动
  ↓
init_grammar()
  ↓
set_grammar(A) -> session_run() -> clear_grammar()
  ↓
set_grammar(B) -> session_run() -> clear_grammar()
  ↓
...
```

不同独立请求之间，demo 还会执行 `clear_kvcache()`，避免继承上一轮上下文。

---

## 14. 常见问题

### `Grammar functions not found in SO`

通常说明设备端 `librknn3_api.so` 不包含对应 Grammar API，或者 Python Toolkit Lite 与设备端 runtime library 版本不匹配。需要检查部署版本，仅修改脚本不能解决底层接口缺失。

### Builtin JSON 输出太长

Builtin JSON 只限制“必须是合法 JSON”，并不限制对象大小。如果模型持续扩展 JSON，可能达到 `--max_new_token` 上限。

实际业务已经知道字段结构时，建议使用 JSON Schema。

### JSON 格式正确，但内容不正确

这是 Grammar 的能力边界。Grammar 负责格式，模型负责具体内容，事实和业务逻辑需要应用层继续验证。

### Tool Call 可以直接执行吗？

不建议仅因为结构合法就直接执行。至少还应检查：

- function name 白名单；
- 参数 Schema；
- 参数值；
- 当前用户权限。

### Structural Tag 一直重复输出，直到 `max_new_token` 才停止

优先检查 Structural Tag 中的参数 Schema 是否过于开放。比如 `additionalProperties: true` 且没有明确 `properties` / `required` 时，模型可能在参数区域持续生成内容，导致迟迟到不了固定的结束标签。

本 demo 已将 `get_weather` 的参数限制为必填的 `location`，并使用 `additionalProperties: false`，让生成路径能够明确收敛到：

```text
</function>
</tool_call>
```

---

## 15. 总结

如果你的需求是：

```text
“我希望 LLM 不只是尽量按格式回答，
而是生成时就只能从合法格式中选择。”
```

那么就适合使用 RKNN3 Grammar。

`grammar_demo.py` 把最常见的 JSON Schema、Builtin JSON、Regex、EBNF 和 Structural Tag 放在一个顺序式 demo 中：先看完整运行流程，再从 `examples` 里复制一个最接近自己业务的例子修改即可。
