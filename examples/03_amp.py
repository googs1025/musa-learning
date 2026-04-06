"""
03_amp.py — 混合精度训练 (AMP / autocast)

学习目标:
  - torch.cuda.amp.autocast + GradScaler 的用法
  - 混合精度对速度和显存的影响
  - 在 MUSA 上和 CUDA 写法完全一致

用法:
    python examples/03_amp.py
"""

import time

import torchada  # 放第一行
import torch
import torch.nn as nn

DEVICE = "cuda"
BATCH_SIZE = 256
INPUT_DIM = 512
NUM_CLASSES = 100
STEPS = 30


class BigMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


model = BigMLP().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()


def run(use_amp: bool):
    torch.cuda.reset_peak_memory_stats()
    model.train()
    t0 = time.time()

    for _ in range(STEPS):
        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=DEVICE)
        y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)

        optimizer.zero_grad()

        if use_amp:
            # autocast 让部分运算自动降为 fp16，加速计算 + 省显存
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            # GradScaler 防止 fp16 梯度下溢
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    return elapsed, peak_mem


print(f"{'模式':<12} {'耗时 (s)':<12} {'峰值显存 (MB)'}")
print("-" * 40)

t, m = run(use_amp=False)
print(f"{'fp32':<12} {t:<12.2f} {m:.1f}")

t, m = run(use_amp=True)
print(f"{'amp (fp16)':<12} {t:<12.2f} {m:.1f}")

print("\n混合精度通常可以减少 30-50% 显存，同时加速训练。")
