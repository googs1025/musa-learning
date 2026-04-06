"""
repro_cat_dim.py — 复现 torch.cat 负数 dim 在 MUSA 上的 bug

用法:
    python repro_cat_dim.py
"""

import torchada
import torch

device = "cuda"  # torchada 映射到 musa

a = torch.randn(1, 8, 4, 64, device=device)
b = torch.randn(1, 8, 1, 64, device=device)

print(f"a.shape = {a.shape}  device={a.device}")
print(f"b.shape = {b.shape}  device={b.device}")
print()

# 正数 dim（应该成功）
try:
    out = torch.cat([a, b], dim=2)
    print(f"[OK]   torch.cat(dim=2)  → shape={tuple(out.shape)}")
except Exception as e:
    print(f"[FAIL] torch.cat(dim=2)  → {e}")

# 负数 dim（预期在 MUSA 上失败）
try:
    out = torch.cat([a, b], dim=-2)
    print(f"[OK]   torch.cat(dim=-2) → shape={tuple(out.shape)}")
except Exception as e:
    print(f"[FAIL] torch.cat(dim=-2) → {e}")
