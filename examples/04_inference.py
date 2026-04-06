"""
04_inference.py — 用 HuggingFace 模型在 MUSA 上做推理

学习目标:
  - 加载预训练模型到 MUSA 设备
  - tokenizer + model.generate 的基本流程
  - torchada 让 HuggingFace 代码零修改

前置依赖:
    pip install transformers accelerate

用法:
    python examples/04_inference.py
"""

import torchada  # 放第一行
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("请先安装: pip install transformers accelerate")
    raise

# 用一个小模型，方便下载和测试
# 如果服务器上有更大的模型，把 MODEL_NAME 换掉即可
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

DEVICE = "cuda"   # torchada 自动映射到 musa

print(f"加载模型: {MODEL_NAME}")
print(f"目标设备: {DEVICE}  (MUSA 服务器上自动映射到 musa)\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # 省显存
    device_map=DEVICE,           # 直接加载到 GPU
)
model.eval()

print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
print(f"模型设备:  {next(model.parameters()).device}\n")

# ── 推理 ──────────────────────────────────────────────────────
prompts = [
    "介绍一下摩尔线程GPU的特点：",
    "PyTorch 和 torch_musa 的关系是：",
]

for prompt in prompts:
    print(f"Prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,         # 贪心解码，结果确定
            temperature=1.0,
            repetition_penalty=1.1,
        )

    # 只打印新生成的部分
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"回复: {response}\n")

print(f"显存峰值: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
