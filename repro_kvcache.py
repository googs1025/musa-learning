"""
repro_kvcache.py — 复现 Qwen2 在 MUSA 上使用 KV cache 时的报错

用法:
    python repro_kvcache.py
"""

import torchada
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d"

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

inputs = tokenizer("hello", return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

print("开始生成（use_cache=True，期望复现报错）...")
try:
    outputs = model.generate(**inputs, max_new_tokens=5)
    print(f"[OK] 生成成功: {outputs}")
except Exception as e:
    print(f"[FAIL] 报错: {e}")
