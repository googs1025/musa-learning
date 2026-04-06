"""
08_finetune.py — 用 LoRA 在 MUSA 上微调小模型

学习目标:
  - peft 库的 LoraConfig 和 get_peft_model 用法
  - 只训练 LoRA 参数，冻结原始权重
  - 结合 AMP 和 use_cache=False（绕开 KV cache bug）

前置依赖:
    pip install peft

用法:
    python examples/08_finetune.py
"""

import torchada  # 放第一行
import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    print("请先安装: pip install peft transformers")
    raise

DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
MAX_NEW_TOKENS = 30
TRAIN_STEPS = 5        # 演示用，只跑 5 步
LR = 2e-4

print(f"加载模型: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=DEVICE,
)

# ── 1. 配置 LoRA ───────────────────────────────────────────────
print("\n── 1. 配置 LoRA ──")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                     # LoRA 秩（rank），越大容量越大，显存越多
    lora_alpha=16,           # 缩放因子，通常设为 2*r
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],   # 只对 attention q/v 加 LoRA
    bias="none",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# 输出示例: trainable params: 589,824 || all params: 494,622,720 || trainable%: 0.12

# ── 2. 构造简单训练样本 ────────────────────────────────────────
print("\n── 2. 训练样本 ──")
PROMPTS = [
    "摩尔线程是",
    "MUSA 和 CUDA 的区别是",
    "torch_musa 的作用是",
]


def make_batch(prompt: str):
    """把 prompt 编码成 input_ids，label 和 input_ids 一致（causal LM 自回归）"""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = enc["input_ids"].to(DEVICE)
    return input_ids, input_ids.clone()   # labels = input_ids


# ── 3. 训练循环 ────────────────────────────────────────────────
print("\n── 3. 训练 ──")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

model.train()
for step in range(TRAIN_STEPS):
    prompt = PROMPTS[step % len(PROMPTS)]
    input_ids, labels = make_batch(prompt)

    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        # use_cache=False：绕开 torch_musa KV cache torch.cat dim bug
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = out.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f"  step {step+1}/{TRAIN_STEPS}  loss={loss.item():.4f}")

# ── 4. 推理（验证 LoRA 效果）─────────────────────────────────
print("\n── 4. 微调后推理 ──")
model.eval()
prompt = "摩尔线程是"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=False,   # 同样绕开 KV cache bug
    )

new_tokens = out[0][inputs["input_ids"].shape[1]:]
print(f"Prompt: {prompt}")
print(f"生成: {tokenizer.decode(new_tokens, skip_special_tokens=True)}")

print(f"\n显存峰值: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
print("\n完成! （只训了 5 步，结果不代表真实微调效果）")
