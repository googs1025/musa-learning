"""
06_profiling.py — 用 torch.profiler 分析 MUSA 算子耗时

学习目标:
  - torch.profiler.profile 的基本用法
  - 如何找出训练 / 推理的耗时瓶颈
  - MUSA 上 profiler 和 CUDA 写法完全一致

用法:
    python examples/06_profiling.py
"""

import torchada  # 放第一行
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

DEVICE = "cuda"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


model = MLP().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("=== torch.profiler 示例 ===\n")

# ── 1. 基础用法：profile 一次前向 + 反向 ──────────────────────
print("── 1. 分析单次 forward + backward ──")

x = torch.randn(256, 512, device=DEVICE)
y = torch.randint(0, 10, (256,), device=DEVICE)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with record_function("forward"):
        logits = model(x)
        loss = criterion(logits, y)
    with record_function("backward"):
        loss.backward()

# 按 GPU 时间排序，只看 Top-10
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# ── 2. 带 warmup 的 schedule 用法（更准确）─────────────────────
print("\n── 2. 带 warmup 的 schedule（推荐用于训练循环）──")


def train_step():
    xb = torch.randn(256, 512, device=DEVICE)
    yb = torch.randint(0, 10, (256,), device=DEVICE)
    optimizer.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward()
    optimizer.step()


with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    with_stack=False,
) as prof:
    for step in range(5):
        train_step()
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print("\nTrace 已保存到 ./log/profiler/，可用 TensorBoard 打开:")
print("  tensorboard --logdir=./log/profiler")

# ── 3. 快速计时：不用 profiler，只用 synchronize ──────────────
print("\n── 3. 轻量计时（不需要 profiler 的场景）──")
import time

torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    train_step()
torch.cuda.synchronize()
elapsed = (time.time() - t0) * 1000
print(f"100 步训练耗时: {elapsed:.1f} ms  ({elapsed/100:.2f} ms/step)")

print(f"\n显存峰值: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
print("\n完成!")
