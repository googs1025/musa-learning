"""
01_basic_ops.py — MUSA 基础 Tensor 操作

学习目标:
  - 如何把 tensor 搬到 MUSA 设备
  - 常用算子在 MUSA 上跑
  - 和 CUDA 写法的对比

用法:
    python examples/01_basic_ops.py
"""

import torchada  # 放第一行，自动把 cuda 调用映射到 musa
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}  (在 MUSA 服务器上 cuda → musa)\n")

# ── 1. 创建 Tensor ──────────────────────────────────────────
print("=== 1. 创建 Tensor ===")
x = torch.randn(4, 4).to(device)
print(f"x.device = {x.device}")
print(f"x.dtype  = {x.dtype}")
print(x)

# ── 2. 基础运算 ─────────────────────────────────────────────
print("\n=== 2. 基础运算 ===")
y = torch.ones(4, 4).to(device)
print("x + y =", (x + y).mean().item())
print("x * y =", (x * y).mean().item())
print("x @ y =", (x @ y).mean().item())   # 矩阵乘法

# ── 3. 数学函数 ─────────────────────────────────────────────
print("\n=== 3. 数学函数 ===")
print("relu(x).mean() =", torch.relu(x).mean().item())
print("sigmoid(x).mean() =", torch.sigmoid(x).mean().item())
print("softmax(x, dim=1) =", torch.softmax(x, dim=1).sum(dim=1))  # 每行和为 1

# ── 4. 形状操作 ─────────────────────────────────────────────
print("\n=== 4. 形状操作 ===")
z = x.view(2, 8)
print(f"view(2, 8):  {z.shape}")
z = x.reshape(1, 16)
print(f"reshape(1, 16): {z.shape}")
z = x.permute(1, 0)
print(f"permute(1, 0): {z.shape}")

# ── 5. 索引与切片 ────────────────────────────────────────────
print("\n=== 5. 索引与切片 ===")
print("第 0 行:", x[0])
print("第 0 列:", x[:, 0])
print("前 2 行前 2 列:\n", x[:2, :2])

# ── 6. 显存管理 ─────────────────────────────────────────────
print("\n=== 6. 显存管理 ===")
print(f"已用显存: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
del x, y, z
torch.cuda.empty_cache()
print(f"清理后:  {torch.cuda.memory_allocated() / 1e6:.2f} MB")

print("\n完成!")
