#!/usr/bin/env bash
# 一键运行全部非 DDP 示例（DDP 需要 torchrun，单独运行）
set -e

SCRIPTS=(
    examples/01_basic_ops.py
    examples/02_training.py
    examples/03_amp.py
    examples/04_inference.py
    examples/05_debug_tips.py
    examples/06_profiling.py
    examples/07_dataloader.py
    examples/08_finetune.py
    examples/09_benchmark.py
)

for script in "${SCRIPTS[@]}"; do
    echo "================================================"
    echo "▶ $script"
    echo "================================================"
    python "$script"
    echo ""
done

echo "全部示例运行完毕！"
echo "DDP 示例请单独运行:"
echo "  torchrun --nproc_per_node=2 examples/10_ddp.py"
