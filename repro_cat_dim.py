"""
repro_cat_dim.py — 复现推理时 KV cache 更新报错

原始报错来自:
    transformers/cache_utils.py:119
    self.keys = torch.cat([self.keys, key_states], dim=-2)
    RuntimeError: Wrong Cat dim: 2

用法:
    python repro_cat_dim.py
"""

import torchada
import torch

device = "cuda"
print(f"device: {torch.tensor(1).to(device).device}\n")

# ── 测试 1：fp32（基础验证）──────────────────────────────────
print("=== 测试 1：fp32 ===")
a = torch.randn(1, 8, 4, 64, device=device)
b = torch.randn(1, 8, 1, 64, device=device)
for dim in [2, -2]:
    try:
        out = torch.cat([a, b], dim=dim)
        print(f"  [OK]   torch.cat(dim={dim:2d})  → {tuple(out.shape)}")
    except Exception as e:
        print(f"  [FAIL] torch.cat(dim={dim:2d})  → {e}")

# ── 测试 2：fp16（注意力层实际使用的精度）───────────────────
print("\n=== 测试 2：fp16 ===")
a = torch.randn(1, 8, 4, 64, device=device, dtype=torch.float16)
b = torch.randn(1, 8, 1, 64, device=device, dtype=torch.float16)
for dim in [2, -2]:
    try:
        out = torch.cat([a, b], dim=dim)
        print(f"  [OK]   torch.cat(dim={dim:2d})  → {tuple(out.shape)}")
    except Exception as e:
        print(f"  [FAIL] torch.cat(dim={dim:2d})  → {e}")

# ── 测试 3：模拟真实 KV cache 更新（Qwen2-0.5B 的注意力维度）
print("\n=== 测试 3：模拟 KV cache 更新（Qwen2-0.5B, fp16）===")
# Qwen2-0.5B: num_kv_heads=2, head_dim=64
batch, num_kv_heads, head_dim = 1, 2, 64
try:
    # 模拟已有 4 个 token 的 cache
    cache_k = torch.randn(batch, num_kv_heads, 4, head_dim, device=device, dtype=torch.float16)
    # 新来 1 个 token 的 key
    new_k   = torch.randn(batch, num_kv_heads, 1, head_dim, device=device, dtype=torch.float16)
    # KV cache 更新操作
    updated = torch.cat([cache_k, new_k], dim=-2)
    print(f"  [OK]   cache update → {tuple(updated.shape)}")
except Exception as e:
    print(f"  [FAIL] cache update → {e}")

# ── 测试 4：3D tensor 的 dim=-2 ─────────────────────────────
print("\n=== 测试 4：3D tensor ===")
a = torch.randn(1, 4, 64, device=device, dtype=torch.float16)
b = torch.randn(1, 1, 64, device=device, dtype=torch.float16)
for dim in [1, -2]:
    try:
        out = torch.cat([a, b], dim=dim)
        print(f"  [OK]   torch.cat(dim={dim:2d})  → {tuple(out.shape)}")
    except Exception as e:
        print(f"  [FAIL] torch.cat(dim={dim:2d})  → {e}")
