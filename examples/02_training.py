"""
02_training.py — 在 MUSA 上跑一个完整的训练循环

学习目标:
  - 定义模型 / 数据 / optimizer
  - 完整的 forward → loss → backward → step 循环
  - 用 torchada 做到和 CUDA 代码一字不差

用法:
    python examples/02_training.py
"""

import torchada  # 放第一行
import torch
import torch.nn as nn

# ── 超参数 ────────────────────────────────────────────────────
DEVICE = "cuda"   # torchada 自动映射到 musa
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
INPUT_DIM = 128
NUM_CLASSES = 10


# ── 模型：简单的多层感知机 ─────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


# ── 合成数据（不依赖真实数据集）────────────────────────────────
def make_batch():
    x = torch.randn(BATCH_SIZE, INPUT_DIM, device=DEVICE)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)
    return x, y


# ── 训练 ──────────────────────────────────────────────────────
model = MLP().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"设备: {DEVICE}  (MUSA 服务器上自动映射)\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct = 0

    for step in range(20):  # 每 epoch 20 个 batch
        x, y = make_batch()

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()

    avg_loss = total_loss / 20
    acc = correct / (20 * BATCH_SIZE) * 100
    print(f"Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  acc={acc:.1f}%")

print("\n训练完成!")
print(f"显存峰值: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
