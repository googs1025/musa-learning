# musa-learning

在摩尔线程 GPU（MUSA）上学习 PyTorch 的实战示例项目。

## 验证过的环境

| 项目 | 版本 |
|---|---|
| GPU | MTT S4000（显存 51.4 GB） |
| Python | 3.10.8 |
| torch | 2.2.0a0+git8ac9b20 |
| torch_musa | 1.3.0 |
| torchada | 0.1.48 |

## 安装

```bash
git clone https://github.com/googs1025/musa-learning.git
cd musa-learning
pip install -r requirements.txt
```

> **注意**：`torch` 和 `torch_musa` 在 AutoDL 等摩尔线程镜像中已预装，无需手动安装。
> `transformers` 需要 < 5.0 版本，因为 5.x 要求 PyTorch >= 2.4，与 torch_musa 2.2 不兼容。

> **可选依赖**：`08_finetune.py` 需要 `pip install peft`。

## 快速验证

```bash
python verify_env.py
```

成功输出示例：

```
[3] torch_musa
  [OK] torch_musa 版本: 1.3.0+81caf0a
  [OK] musa 设备数量: 1
  [OK] musa 是否可用: True
  [OK] 当前设备名: MTT S4000

[4] torchada
  [OK] 检测到的平台: Platform.MUSA
  [OK] 是否 MUSA 平台: True

[5] 基础算子
  [OK] 矩阵乘法: shape=(512, 512), device=musa:0

[6] 显存
  [OK] 显存总量 (GB): 51.4
```

## 示例列表

按顺序学习：

```bash
python examples/01_basic_ops.py    # Tensor 基础操作
python examples/02_training.py     # 完整训练循环
python examples/03_amp.py          # 混合精度（AMP）
python examples/04_inference.py    # HuggingFace 模型推理
python examples/05_debug_tips.py   # 常见问题与调试技巧
python examples/06_profiling.py    # torch.profiler 性能分析
python examples/07_dataloader.py   # DataLoader 多进程与 pin_memory
python examples/08_finetune.py     # LoRA 微调（需 pip install peft）
python examples/09_benchmark.py    # MUSA vs CPU 算子基准测试
torchrun --nproc_per_node=2 examples/10_ddp.py  # 多卡 DDP（单卡用 nproc_per_node=1）
```

一键运行全部示例（DDP 除外）：

```bash
bash run_all.sh
```

## 核心用法

只需在代码第一行加 `import torchada`，之后的 `torch.cuda.*` 代码**零修改**运行在 MUSA 上：

```python
import torchada          # 放第一行
import torch

x = torch.randn(1000, 1000).cuda()   # 自动跑在 MUSA 上
print(x.device)                       # musa:0
```

## 相关项目

- [torchada](https://github.com/MooreThreads/torchada) — CUDA→MUSA 适配层
- [torch_musa](https://github.com/MooreThreads/torch_musa) — PyTorch MUSA 后端
