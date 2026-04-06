"""
07_dataloader.py — DataLoader 多进程与 pin_memory 优化

学习目标:
  - num_workers 对吞吐量的影响
  - pin_memory=True 在 MUSA 上是否有效
  - 对比不同配置下的数据加载速度

用法:
    python examples/07_dataloader.py
"""

import time

import torchada  # 放第一行
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda"
BATCH_SIZE = 256
N_SAMPLES = 10_000
N_BATCHES = 20   # 只跑前 20 个 batch，够对比

# ── 合成数据集 ─────────────────────────────────────────────────
# 故意放在 CPU，模拟真实的磁盘数据 → DataLoader → GPU 流程
X = torch.randn(N_SAMPLES, 512)   # CPU tensor
Y = torch.randint(0, 10, (N_SAMPLES,))

dataset = TensorDataset(X, Y)

model = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10)
).to(DEVICE)
criterion = nn.CrossEntropyLoss()

print("=== DataLoader 配置对比 ===\n")
print(f"{'配置':<35} {'吞吐 (samples/s)':<20} {'显存峰值 MB'}")
print("-" * 70)


def benchmark(num_workers: int, pin_memory: bool):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    n = 0

    for i, (xb, yb) in enumerate(loader):
        if i >= N_BATCHES:
            break
        xb = xb.to(DEVICE, non_blocking=pin_memory)
        yb = yb.to(DEVICE, non_blocking=pin_memory)
        with torch.no_grad():
            _ = criterion(model(xb), yb)
        n += xb.size(0)

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    throughput = n / elapsed
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    return throughput, peak_mem


configs = [
    (0, False, "num_workers=0, pin_memory=False"),
    (0, True,  "num_workers=0, pin_memory=True"),
    (2, False, "num_workers=2, pin_memory=False"),
    (2, True,  "num_workers=2, pin_memory=True"),
    (4, True,  "num_workers=4, pin_memory=True"),
]

for nw, pm, label in configs:
    tp, mem = benchmark(nw, pm)
    print(f"{label:<35} {tp:<20,.0f} {mem:.1f}")

print("""
结论提示:
  - pin_memory=True：数据先放到固定内存（锁页内存），GPU 可以用 DMA 直接搬运，
    在 CUDA 上效果明显；MUSA 上效果视驱动版本而定。
  - num_workers > 0：多进程预取数据，掩盖 I/O 延迟，GPU 不会因等数据而空转。
  - non_blocking=True：配合 pin_memory，数据搬运和计算可以异步进行。
""")

print("完成!")
