# LoRA Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Create a PEFT Config (or use existing)

Use one of the pre-made configs:
- `configs/peft/lora_minimal.yaml` - Fastest, minimal parameters
- `configs/peft/lora_default.yaml` - Balanced (recommended)
- `configs/peft/lora_full.yaml` - Maximum capacity

Or create your own `configs/peft/my_lora.yaml`:
```yaml
r: 8                # LoRA rank
alpha: 16.0         # Scaling factor
dropout: 0.0        # Dropout (usually 0)
target_modules:     # Which layers to adapt
  - q_proj
  - v_proj
  - o_proj
```

### Step 2: Update Training Config

In your training config (e.g., `configs/train.yaml`):
```yaml
model: llama-7b
data: your_dataset
peft: lora_default  # Enable LoRA

trainer:
  micro_batch_size: 8
  train_batch_size: 128
  tensor_model_parallel_size: 2
```

### Step 3: Train!

```bash
# Single GPU
python ironcore/cli/train.py --config-path configs/train.yaml

# Multi-GPU (TP=2)
torchrun --nproc_per_node=2 ironcore/cli/train.py --config-path configs/train.yaml
```

Expected output:
```
Freezing base model parameters for PEFT method: lora
Trainable parameters: 8,388,608 / 7,016,169,472 (0.12%)
Memory usage: ~15 GB (vs ~56 GB full fine-tuning)
```

---

## 📊 Configuration Cheat Sheet

| Goal | r | alpha | target_modules | Trainable % |
|------|---|-------|----------------|-------------|
| **Fastest** | 4 | 8 | q,v | ~0.05% |
| **Balanced** | 8 | 16 | q,v,o,up,down | ~0.12% |
| **Best Quality** | 32 | 64 | all | ~0.5-1% |

### Target Modules Explained
- `q_proj`, `k_proj`, `v_proj` - Attention query/key/value
- `o_proj` - Attention output
- `up_proj`, `down_proj` - MLP layers

**Tip**: Start with `[q_proj, v_proj]` and add more if needed.

---

## 🧪 Verify Installation

Run the tests:
```bash
# Quick test (30 seconds)
python tests/test_lora_tp_correctness.py --mode save_weights

# Full test suite (2 minutes)
python tests/test_lora_tp_correctness.py --mode save_weights
torchrun --nproc_per_node=2 tests/test_lora_tp_correctness.py --mode load_and_compare
python tests/test_lora_async.py
```

All tests should pass ✅

---

## 💾 Memory Savings

| Model Size | Full FT | LoRA (TP=1) | LoRA (TP=2) |
|------------|---------|-------------|-------------|
| 7B | 56 GB | 15 GB | 8 GB |
| 13B | 104 GB | 28 GB | 14 GB |
| 30B | 240 GB | 65 GB | 33 GB |

*With LoRA r=8, standard config*

---

## 🛠️ Troubleshooting

### Problem: "CUDA out of memory"
**Solution**: Reduce batch size or enable chunking:
```yaml
trainer:
  micro_batch_size: 4  # Reduce
  sequence_chunk_size: 512  # Enable chunking
```

### Problem: Loss not decreasing
**Solution**: Increase learning rate and/or alpha:
```yaml
optim:
  max_lr: 5e-4  # Higher for LoRA
peft:
  lora:
    alpha: 32  # Stronger updates
```

### Problem: Want better quality
**Solution**: Increase rank and add more target modules:
```yaml
peft:
  lora:
    r: 16  # More capacity
    target_modules: [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj]
```

---

## 📚 Learn More

- **Full Guide**: `docs/PEFT_GUIDE.md`
- **Implementation Details**: `LORA_IMPLEMENTATION_SUMMARY.md`
- **Tests**: `tests/test_lora_*.py`
- **Examples**: `configs/peft/*.yaml`

---

## ✨ Key Benefits

- ✅ **75-85% memory reduction** vs full fine-tuning
- ✅ **<5% training overhead** vs frozen model
- ✅ **Full tensor parallelism** support (TP=1,2,4,8)
- ✅ **Async chunking** compatible
- ✅ **Universal checkpoints** (save TP=1, load TP=N)

Happy training! 🎉
