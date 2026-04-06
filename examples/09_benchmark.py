"""
09_benchmark.py — 常用算子 MUSA vs CPU 速度对比

学习目标:
  - 量化 MUSA 的加速倍数
  - 不同算子（matmul、conv2d、attention）的加速差异
  - 正确的 GPU 计时方法（synchronize）

用法:
    python examples/09_benchmark.py
"""

import time

import torchada  # 放第一行
import torch
import torch.nn as nn
import torch.nn.functional as F

WARMUP = 5
REPEAT = 20


def timeit(fn, device: str) -> float:
    """运行 fn WARMUP 次预热，再重复 REPEAT 次计时，返回平均毫秒。"""
    for _ in range(WARMUP):
        fn()
    if device != "cpu":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(REPEAT):
        fn()
    if device != "cpu":
        torch.cuda.synchronize()

    return (time.time() - t0) / REPEAT * 1000   # ms


def compare(name: str, fn_cpu, fn_gpu, gpu_device: str):
    cpu_ms = timeit(fn_cpu, "cpu")
    gpu_ms = timeit(fn_gpu, gpu_device)
    speedup = cpu_ms / gpu_ms
    print(f"  {name:<28} CPU={cpu_ms:7.2f}ms  GPU={gpu_ms:7.2f}ms  加速={speedup:5.1f}x")


has_gpu = torchada.is_musa_platform() or torch.cuda.is_available()
GPU = "cuda" if has_gpu else None

print("=== MUSA vs CPU 基准测试 ===\n")

if not has_gpu:
    print("未检测到 GPU，跳过对比，仅显示 CPU 结果。")
else:
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# ── 1. 矩阵乘法 ───────────────────────────────────────────────
print("── 1. 矩阵乘法 (matmul) ──")
for N in [512, 2048, 4096]:
    a_cpu = torch.randn(N, N)
    b_cpu = torch.randn(N, N)
    if has_gpu:
        a_gpu = a_cpu.to(GPU)
        b_gpu = b_cpu.to(GPU)
        compare(f"matmul {N}x{N}",
                lambda: torch.matmul(a_cpu, b_cpu),
                lambda: torch.matmul(a_gpu, b_gpu),
                GPU)
    else:
        ms = timeit(lambda: torch.matmul(a_cpu, b_cpu), "cpu")
        print(f"  matmul {N}x{N:<18} CPU={ms:7.2f}ms")

# ── 2. 卷积 ───────────────────────────────────────────────────
print("\n── 2. 2D 卷积 (conv2d) ──")
conv_cpu = nn.Conv2d(64, 128, kernel_size=3, padding=1)
for B, H in [(8, 64), (16, 128)]:
    x_cpu = torch.randn(B, 64, H, H)
    if has_gpu:
        conv_gpu = conv_cpu.to(GPU)
        x_gpu = x_cpu.to(GPU)
        compare(f"conv2d B={B} H={H}",
                lambda: conv_cpu(x_cpu),
                lambda: conv_gpu(x_gpu),
                GPU)
    else:
        ms = timeit(lambda: conv_cpu(x_cpu), "cpu")
        print(f"  conv2d B={B} H={H:<20} CPU={ms:7.2f}ms")

# ── 3. Scaled Dot-Product Attention ───────────────────────────
print("\n── 3. Scaled Dot-Product Attention ──")
for B, S, D in [(4, 128, 64), (4, 512, 64)]:
    q_cpu = torch.randn(B, 8, S, D)
    k_cpu = torch.randn(B, 8, S, D)
    v_cpu = torch.randn(B, 8, S, D)
    if has_gpu:
        q_gpu = q_cpu.to(GPU)
        k_gpu = k_cpu.to(GPU)
        v_gpu = v_cpu.to(GPU)
        compare(
            f"attention B={B} S={S}",
            lambda: F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu),
            lambda: F.scaled_dot_product_attention(q_gpu, k_gpu, v_gpu),
            GPU,
        )
    else:
        ms = timeit(lambda: F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu), "cpu")
        print(f"  attention B={B} S={S:<18} CPU={ms:7.2f}ms")

# ── 4. 激活函数 ──────────────────────────────────────────────
print("\n── 4. 激活函数 ──")
for act_name, act_fn in [
    ("relu", torch.relu),
    ("gelu", torch.nn.functional.gelu),
    ("silu", torch.nn.functional.silu),
]:
    x_cpu = torch.randn(1024, 1024)
    if has_gpu:
        x_gpu = x_cpu.to(GPU)
        compare(act_name,
                lambda: act_fn(x_cpu),
                lambda: act_fn(x_gpu),
                GPU)
    else:
        ms = timeit(lambda: act_fn(x_cpu), "cpu")
        print(f"  {act_name:<28} CPU={ms:7.2f}ms")

if has_gpu:
    print(f"\n显存峰值: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
print("\n完成!")
