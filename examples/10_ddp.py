"""
10_ddp.py — 多卡分布式训练 (DistributedDataParallel)

学习目标:
  - torch.distributed 初始化流程
  - DDP 包裹模型的写法
  - DistributedSampler 确保每卡看不同数据
  - torchada 让 CUDA DDP 代码在 MUSA 上零修改运行

前置条件:
  - 机器上有 ≥ 2 块 MUSA GPU（或 CUDA GPU）
  - 单卡机器可用 nproc_per_node=1 运行，退化为单卡训练

用法（双卡）:
    torchrun --nproc_per_node=2 examples/10_ddp.py

用法（单卡验证流程）:
    torchrun --nproc_per_node=1 examples/10_ddp.py
"""

import os

import torchada  # 放第一行，必须在 torch.distributed 之前
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

# ── 超参数 ────────────────────────────────────────────────────
EPOCHS = 3
BATCH_SIZE = 128      # 每卡的 batch size
INPUT_DIM = 256
NUM_CLASSES = 10
LR = 1e-3
N_SAMPLES = 2048


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def main():
    # ── 初始化进程组 ──────────────────────────────────────────
    # torchrun 会自动设置 RANK / LOCAL_RANK / WORLD_SIZE 环境变量
    dist.init_process_group(backend="nccl")   # torchada 把 nccl 映射到 mccl
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"             # torchada 映射到 musa:N

    if rank == 0:
        print(f"world_size={world_size}，每卡 batch_size={BATCH_SIZE}，"
              f"等效全局 batch={BATCH_SIZE * world_size}\n")

    # ── 数据 ─────────────────────────────────────────────────
    X = torch.randn(N_SAMPLES, INPUT_DIM)
    Y = torch.randint(0, NUM_CLASSES, (N_SAMPLES,))
    dataset = TensorDataset(X, Y)

    # DistributedSampler 把数据均分到每张卡，避免重复
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

    # ── 模型 ─────────────────────────────────────────────────
    model = MLP().to(device)
    # DDP：把模型包一层，梯度自动 all-reduce
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ── 训练 ─────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        sampler.set_epoch(epoch)   # 每 epoch shuffle 不同，防止数据重复
        model.train()
        total_loss = 0.0
        correct = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()          # DDP 自动在这里做梯度 all-reduce
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()

        # 只在 rank 0 打印，避免多进程重复输出
        if rank == 0:
            avg_loss = total_loss / len(loader)
            acc = correct / (len(loader) * BATCH_SIZE) * 100
            mem_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  acc={acc:.1f}%  "
                  f"显存峰值={mem_mb:.0f}MB")

    if rank == 0:
        print("\n训练完成!")
        print("DDP 要点:")
        print("  - dist.init_process_group() 初始化进程组")
        print("  - DistributedSampler 确保各卡数据不重叠")
        print("  - DDP(model) 包裹后梯度自动 all-reduce")
        print("  - sampler.set_epoch(epoch) 保证每轮 shuffle 不同")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
