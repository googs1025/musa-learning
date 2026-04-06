# RuntimeError: Wrong Cat dim: 2 when running Qwen2 inference with KV cache on MUSA

**Target repo:** MooreThreads/torch_musa

## Environment

| Item | Version |
|---|---|
| GPU | MTT S4000 |
| torch | 2.2.0a0+git8ac9b20 |
| torch_musa | 1.3.0+81caf0a |
| torchada | 0.1.48 |
| transformers | 4.x |
| Python | 3.10.8 |

## Problem

Running Qwen2 inference with KV cache enabled raises:

```
RuntimeError: Wrong Cat dim: 2
```

The error originates from `DynamicCache.update()` in transformers:

```python
# transformers/cache_utils.py
self.keys = torch.cat([self.keys, key_states], dim=-2)
```

## Minimal Reproduction

```python
import torchada
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    dtype=torch.float16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
inputs = tokenizer("hello", return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

model.generate(**inputs, max_new_tokens=5)
# RuntimeError: Wrong Cat dim: 2
```

Reproduction script: `repro_kvcache.py`

## Investigation

Standalone `torch.cat` with `dim=-2` works fine on MUSA:

```python
a = torch.randn(1, 2, 4, 64, device="cuda", dtype=torch.float16)
b = torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.float16)
torch.cat([a, b], dim=-2)  # OK
```

The bug is triggered only inside the model forward pass with KV cache enabled,
suggesting an issue in how torch_musa handles `torch.cat` within the attention
computation graph.

## Workaround

Pass `use_cache=False` to `model.generate()`:

```python
model.generate(**inputs, max_new_tokens=5, use_cache=False)
```
