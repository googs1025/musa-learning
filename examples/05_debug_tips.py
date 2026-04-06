"""
05_debug_tips.py — 常见问题排查和调试技巧

学习目标:
  - 识别 MUSA 上的常见报错
  - 设备检查的正确写法
  - 显存泄漏排查

用法:
    python examples/05_debug_tips.py
"""

import torchada
import torch

print("=== 调试技巧合集 ===\n")

# ── 技巧 1：正确的平台检测 ────────────────────────────────────
print("── 技巧 1：平台检测 ──")

# 错误写法（在 MUSA 上永远返回 False，不要用这个做 GPU 检测）
has_cuda = torch.cuda.is_available()
print(f"torch.cuda.is_available() = {has_cuda}  ← MUSA 上返回 False，勿用于判断是否有 GPU")

# 正确写法
has_gpu = torchada.is_musa_platform() or torch.cuda.is_available()
print(f"有 GPU: {has_gpu}")
print(f"平台:   {torchada.detect_platform()}")

# ── 技巧 2：正确的 device.type 判断 ──────────────────────────
print("\n── 技巧 2：device.type 判断 ──")
device = torch.device("cuda:0") if has_gpu else torch.device("cpu")

# 错误写法（MUSA 上 device.type == "musa"，不等于 "cuda"）
print(f'device.type == "cuda"  →  {device.type == "cuda"}  ← 在 MUSA 上是 False')

# 正确写法
print(f'torchada.is_gpu_device(device)  →  {torchada.is_gpu_device(device)}  ← 推荐')
print(f'device.type in ("cuda", "musa")  →  {device.type in ("cuda", "musa")}  ← 也可以')

# ── 技巧 3：显存使用追踪 ──────────────────────────────────────
print("\n── 技巧 3：显存追踪 ──")

def print_mem(label):
    alloc = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    print(f"  {label:<20} 已分配={alloc:.1f}MB  已保留={reserved:.1f}MB")

print_mem("初始状态")

if has_gpu:
    x = torch.randn(1000, 1000, device="cuda")
    print_mem("创建 1000x1000 tensor")

    y = torch.randn(1000, 1000, device="cuda")
    print_mem("再创建一个")

    del x
    print_mem("del x 后")

    torch.cuda.empty_cache()
    print_mem("empty_cache() 后")
    del y

# ── 技巧 4：同步与计时 ────────────────────────────────────────
print("\n── 技巧 4：正确计时 GPU 操作 ──")
import time

if has_gpu:
    x = torch.randn(2000, 2000, device="cuda")

    # 错误写法：GPU 操作是异步的，time.time() 拿到的不是真实耗时
    t0 = time.time()
    _ = torch.matmul(x, x)
    t_wrong = (time.time() - t0) * 1000
    print(f"  不同步直接计时: {t_wrong:.2f}ms  ← 可能不准")

    # 正确写法：先 synchronize，等 GPU 真正跑完
    torch.cuda.synchronize()
    t0 = time.time()
    _ = torch.matmul(x, x)
    torch.cuda.synchronize()
    t_correct = (time.time() - t0) * 1000
    print(f"  synchronize 后计时: {t_correct:.2f}ms  ← 准确")

# ── 技巧 5：常见报错速查 ──────────────────────────────────────
print("\n── 技巧 5：常见报错速查 ──")
tips = [
    ("device type mismatch",       "两个 tensor 不在同一个设备，用 .to(device) 统一"),
    ("CUDA out of memory",         "显存不足：减小 batch_size，或加 amp autocast"),
    ("Expected all tensors on cuda", "模型或数据还在 CPU，检查 .to(device) 是否漏掉"),
    ("NotImplementedError (算子)",  "torch_musa 暂不支持该算子，去 torch_musa issues 反馈"),
    ("is_available() 返回 False",   "这是正常的！MUSA 不映射 cuda.is_available，用 musa.is_available()"),
]
for err, fix in tips:
    print(f"  ⚠  {err}")
    print(f"     → {fix}\n")

print("完成!")
