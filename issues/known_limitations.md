# torch_musa 已知兼容问题汇总

本文档汇总在 torch_musa 1.3.x / torch 2.2 环境中遇到的兼容问题，
包含复现方法、根因分析和绕过方案。

---

## 1. KV Cache torch.cat dim 报错

**错误:** `RuntimeError: Wrong Cat dim: 2`

**触发场景:** transformers `DynamicCache.update()` 内部调用 `torch.cat([...], dim=-2)`，
在 MUSA 上 forward 计算图中处理负数 dim 时出错。

**独立复现:**

```python
import torchada, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
inputs = tokenizer("hello", return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
model.generate(**inputs, max_new_tokens=5)   # RuntimeError: Wrong Cat dim: 2
```

**注意:** 单独运行 `torch.cat([a, b], dim=-2)` 没有问题，问题出在模型 forward 计算图内。

**绕过方案:** 推理和微调时传 `use_cache=False`：

```python
model.generate(**inputs, max_new_tokens=50, use_cache=False)
```

**影响:** 推理速度略慢（无 KV cache 加速），但结果正确。

**详细报告:** [torch_musa_kvcache_cat_dim.md](torch_musa_kvcache_cat_dim.md)

---

## 2. torch.cuda.is_available() 返回 False

**现象:** `torch.cuda.is_available()` 在 MUSA 上返回 `False`，即使 GPU 存在。

**原因:** MUSA 是独立后端，不映射 CUDA 设备枚举。

**绕过方案:**

```python
import torchada
has_gpu = torchada.is_musa_platform() or torch.cuda.is_available()
```

---

## 3. device.type 不等于 "cuda"

**现象:** 把 tensor 放到 MUSA 后，`tensor.device.type == "cuda"` 返回 `False`，
因为实际类型是 `"musa"`。

**绕过方案:**

```python
import torchada
# 推荐
torchada.is_gpu_device(tensor.device)
# 或
tensor.device.type in ("cuda", "musa")
```

---

## 4. transformers >= 5.0 不兼容

**现象:** 安装 transformers 5.x 后报错，要求 `torch >= 2.4`，但 torch_musa 目前支持到 2.2。

**绕过方案:** 固定 transformers 版本：

```bash
pip install "transformers<5.0"
```

---

## 更新日志

| 日期 | 内容 |
|---|---|
| 2026-04-06 | 初始版本，汇总 4 个已知问题 |
