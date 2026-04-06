"""
verify_env.py — 上服务器第一件事，跑这个脚本验证环境是否正常。

用法:
    python verify_env.py
"""

import sys


def check(label, fn):
    try:
        result = fn()
        print(f"  [OK] {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


print("=" * 55)
print("  MUSA 环境验证")
print("=" * 55)

# 1. Python
print(f"\n[1] Python {sys.version.split()[0]}")

# 2. torch
print("\n[2] PyTorch")
import torch

check("torch 版本", lambda: torch.__version__)
check("torch.cuda.is_available (NVIDIA)", lambda: torch.cuda.is_available())

# 3. torch_musa
print("\n[3] torch_musa")
try:
    import torch_musa

    check("torch_musa 版本", lambda: torch_musa.__version__)
    check("musa 设备数量", lambda: torch.musa.device_count())
    check("musa 是否可用", lambda: torch.musa.is_available())
    check("当前设备名", lambda: torch.musa.get_device_name(0))
except ImportError:
    print("  [SKIP] torch_musa 未安装（在摩尔线程 GPU 服务器上才需要）")

# 4. torchada
print("\n[4] torchada")
try:
    import torchada

    check("torchada 版本", lambda: torchada.__version__)
    check("检测到的平台", lambda: torchada.detect_platform())
    check("是否 MUSA 平台", lambda: torchada.is_musa_platform())
except ImportError:
    print("  [SKIP] torchada 未安装，运行: pip install torchada")

# 5. 基础算子
print("\n[5] 基础算子")


def run_matmul():
    device = "musa" if torch.musa.is_available() else "cpu"
    x = torch.randn(512, 512, device=device)
    y = torch.matmul(x, x.T)
    return f"shape={tuple(y.shape)}, device={y.device}"


check("矩阵乘法", run_matmul)

# 6. 显存
print("\n[6] 显存")
try:
    if torch.musa.is_available():
        check("已用显存 (MB)", lambda: round(torch.cuda.memory_allocated() / 1e6, 2))
        check("显存总量 (GB)", lambda: round(torch.musa.get_device_properties(0).total_memory / 1e9, 1))
    else:
        print("  [SKIP] 无 MUSA 设备")
except Exception as e:
    print(f"  [SKIP] {e}")

print("\n" + "=" * 55)
print("  验证完成")
print("=" * 55)
