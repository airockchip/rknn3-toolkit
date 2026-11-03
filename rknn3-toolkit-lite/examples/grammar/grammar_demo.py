#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

import ctypes
import json
from argparse import ArgumentParser

import numpy as np
from transformers import AutoTokenizer

from rknn3lite.api import (
    RKNN3Lite,
    RKLLMCallback,
    LLMResultCallback,
    LLMGetEmbedCallback,
    LLMTokenizerCallback,
)
from rknn3lite.api.rknn3_types import RKNN3GrammarType, RKNN3QueryCmd


# ============================================= Default Config =============================================

RKNN_MODEL     = "Qwen2.5-0.5B-Instruct.rknn"
EMBED_PATH     = "Qwen2.5-0.5B-Instruct.embed.bin"
TOKENIZER_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

system_prompt  = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
prompt_prefix  = "<|im_start|>user\n"
prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"

tokenizer = None
embeds_data = None


# ============================================= Callbacks =============================================

def result_callback(userdata, result_ptr, state):
    if not hasattr(result_callback, "tokens"):
        result_callback.tokens = []
        result_callback.last_text = ""

    if state == 0:
        for i in range(result_ptr.contents.num_tokens):
            result_callback.tokens.append(result_ptr.contents.token_ids[i])

        text = tokenizer.decode(
            result_callback.tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).split("\ufffd", 1)[0]

        new_text = text[len(result_callback.last_text):]
        if new_text:
            print(new_text, end="", flush=True)
            result_callback.last_text = text

    elif state in (2, 3, 4, 5):
        result_callback.tokens.clear()
        result_callback.last_text = ""

    return 0


def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    raw = text_ptr[:text_len] if isinstance(text_ptr, bytes) else ctypes.string_at(text_ptr, text_len)
    text = raw.decode("utf-8", errors="ignore")
    tokens = tokenizer(text, return_tensors="np", truncation=True)["input_ids"][0][:n_tokens_max]
    for i, token_id in enumerate(tokens):
        tokens_ptr[i] = int(token_id)
    return len(tokens)


def embed_callback(userdata, tokens_ptr, num_tokens, embed, length):
    embedding_dim = embeds_data.shape[1]
    if length != num_tokens * embedding_dim * np.dtype(np.float16).itemsize:
        print("invalid embed buffer")
        return -1

    dst = np.ctypeslib.as_array(
        ctypes.cast(embed, ctypes.POINTER(ctypes.c_uint16)),
        shape=(num_tokens * embedding_dim,),
    ).view(np.float16)
    dst[:] = embeds_data[[tokens_ptr[i] for i in range(num_tokens)]].ravel()
    return 0


# ============================================= Grammar Examples =============================================

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

CLASSIFY_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["refund", "shipping", "product", "other"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["category", "confidence"],
    "additionalProperties": False,
})

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


def run_grammar(rknn, name, prompt, grammar, grammar_type, max_new_token):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Input : {prompt}")
    print("Output: ", end="", flush=True)

    ret = rknn.set_grammar(grammar, grammar_type)
    if ret != 0:
        print("\nSet grammar failed!")
        return ret

    ret, _ = rknn.session_run(prompt=prompt, keep_history=False, max_new_tokens=max_new_token)
    print()

    rknn.clear_grammar()
    rknn.clear_kvcache()
    return ret


# ============================================= Main =============================================

if __name__ == "__main__":
    parser = ArgumentParser(description="RKNN3 Grammar simple demo")
    parser.add_argument("--rknn_path",      type=str, default=RKNN_MODEL,     help="rknn model path")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH, help="huggingface tokenizer path")
    parser.add_argument("--embed_path",     type=str, default=EMBED_PATH,     help="embedding bin path")
    parser.add_argument("--max_new_token",  type=int, default=256,            help="max new tokens")
    parser.add_argument("--core_mask",      type=str, default="0xff",         help="NPU core mask in hex")
    args = parser.parse_args()

    core_mask = int(args.core_mask, 16)

    # Step 1: Load tokenizer & embedding
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    embeds_data = np.fromfile(args.embed_path, dtype=np.float16)

    # Step 2: Load model
    rknn = RKNN3Lite(llm_mode=True, verbose=True)
    ret = rknn.load_rknn(args.rknn_path, args.rknn_path.replace(".rknn", ".weight"))
    if ret != 0:
        print("Load model failed!")
        exit(ret)

    # Step 3: Init runtime
    ret = rknn.init_runtime(target="rk1820", core_mask=core_mask)
    if ret != 0:
        print("Init runtime failed!")
        exit(ret)

    # Step 4: Query vocab size and reshape embedding
    llm_config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
    vocab_size = llm_config.vocab_size
    embedding_dim = embeds_data.size // vocab_size
    embeds_data = embeds_data.reshape(vocab_size, embedding_dim)
    print(f"vocab_size={vocab_size}, embedding_dim={embedding_dim}")

    # Step 5: Build callback
    callback = RKLLMCallback()
    callback.result_callback = LLMResultCallback(result_callback)
    callback.result_userdata = None

    callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    _tok_ud = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(_tok_ud), ctypes.c_void_p)

    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    _emb_ud = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(_emb_ud), ctypes.c_void_p)

    # Step 6: Init LLM session
    llm_args = [{
        "max_new_tokens": args.max_new_token,
        "top_k": 1,
        "top_p": 0.9,
        "temperature": 1.0,
        "repeat_penalty": 1.1,
        "vocab_size": vocab_size,
        "special_eos_id": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1,
        "max_context_len": llm_config.max_ctx_len,
        "keep_history": 0,
        "logits_name": b"output",
    }]
    ret = rknn.init_llm_session(llm_args=llm_args, llm_callback=callback)
    if ret != 0:
        print("Init LLM session failed!")
        exit(ret)

    rknn.set_chat_template(system_prompt, prompt_prefix, prompt_postfix)

    # Step 7: Init Grammar once
    print("--> Init Grammar")
    stop_tokens = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    token_pieces = (
        tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for i in range(vocab_size)
    )
    ret = rknn.init_grammar(token_pieces=token_pieces, stop_tokens=stop_tokens)
    if ret != 0:
        print("Init grammar failed!")
        exit(ret)
    print("done")

    # Step 8: Run several easy-to-understand examples
    examples = [
        ("1. JSON Schema - extract person info",
         "Extract: Alice is 28 years old and lives in Shenzhen. Reply only with JSON.",
         PERSON_SCHEMA, RKNN3GrammarType.RKNN3_GRAMMAR_JSON_SCHEMA),

        ("2. JSON Schema - classify customer request",
         "Classify: My package has not arrived yet. Return category and confidence only.",
         CLASSIFY_SCHEMA, RKNN3GrammarType.RKNN3_GRAMMAR_JSON_SCHEMA),

        ("3. Builtin JSON - any valid JSON",
         "Give me a small JSON object describing this demo. Reply only with JSON.",
         None, RKNN3GrammarType.RKNN3_GRAMMAR_BUILTIN_JSON),

        ("4. Regex - order number",
         "Generate one order number like ORD-123456.",
         r"ORD-[0-9]{6}", RKNN3GrammarType.RKNN3_GRAMMAR_REGEX),

        ("5. Regex - date",
         "Reply with one date in YYYY-MM-DD format.",
         r"[0-9]{4}-[0-9]{2}-[0-9]{2}", RKNN3GrammarType.RKNN3_GRAMMAR_REGEX),

        ("6. Regex - email",
         "Reply with one email address only.",
         r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", RKNN3GrammarType.RKNN3_GRAMMAR_REGEX),

        ("7. EBNF - yes / no / maybe",
         "Is water wet? Answer only yes, no, or maybe.",
         'root ::= "yes" | "no" | "maybe"\n', RKNN3GrammarType.RKNN3_GRAMMAR_EBNF),

        ("8. EBNF - sentiment",
         "Classify: I really like this product. Reply with one label only.",
         'root ::= "positive" | "negative" | "neutral"\n', RKNN3GrammarType.RKNN3_GRAMMAR_EBNF),

        ("9. Structural Tag - tool call",
         "Call get_weather with location Shenzhen. Reply only with the tool call.",
         STRUCTURAL_TAG, RKNN3GrammarType.RKNN3_GRAMMAR_STRUCTURAL_TAG),
    ]

    for name, prompt, grammar, grammar_type in examples:
        ret = run_grammar(rknn, name, prompt, grammar, grammar_type, args.max_new_token)
        if ret != 0:
            print("RKNN llm inference failed!")
            rknn.release()
            exit(ret)

    print("\nAll grammar examples finished.")
    rknn.release()
