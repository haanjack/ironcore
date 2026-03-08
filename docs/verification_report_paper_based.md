# Muon Optimizer Verification Report - Paper-Based Settings

**Date:** 2026-03-08
**Config:** Paper-based learning rate (Moonlight paper approach)
**Model:** GPT-2 Medium (354M parameters)

---

## Executive Summary

This verification compares the Muon optimizer against AdamW using **fair, paper-based learning rate settings** from the Moonlight paper (arxiv:2502.16982v1). Both optimizers used identical hyperparameters (max_lr=0.00042, min_lr=0.000042) to ensure a fair comparison.

**Key Result:** Muon achieves **9.9% better convergence** (loss 4.82 vs 5.35 at step 500) with **25% less optimizer memory**.

---

## Experimental Setup

### Hardware
- 2x GPUs with DistributedOptimizer
- Gradient Accumulation Steps: 32
- Micro Batch Size: 2
- Global Batch Size: 128

### Hyperparameters (Both Optimizers)
| Parameter | Value |
|-----------|-------|
| max_lr | 0.00042 |
| min_lr | 0.000042 |
| warmup_steps | 50 |
| annealing_steps | 500 |
| weight_decay | 0.1 |
| clip_grad | 1.0 |

### Muon-Specific Settings
| Parameter | Value |
|-----------|-------|
| muon_momentum | 0.95 |
| muon_newton_schulz_steps | 5 |
| muon_lr_scale | 1.0 |
| adamw_lr_scale | 1.0 |

### Model Architecture
| Parameter | Value |
|-----------|-------|
| num_layers | 24 |
| d_model | 1024 |
| d_ffn | 4096 |
| num_attention_heads | 16 |
| max_seq_len | 1024 |
| precision | bf16 |
| flash_attn | true |

---

## Results

### Final Loss Comparison

| Optimizer | Loss @ Step 500 | Training Time |
|-----------|-----------------|---------------|
| **Muon** | **4.82** | 1.34h |
| AdamW | 5.35 | 1.33h |
| **Improvement** | **-9.9%** | - |

### Loss Progression

| Step | Muon Loss | AdamW Loss | Delta | % Improvement |
|------|-----------|------------|-------|---------------|
| 20 | 8.15 | 7.96 | +0.19 | -2.4% |
| 40 | 7.38 | 7.14 | +0.25 | -3.5% |
| 60 | 6.80 | 6.76 | +0.04 | -0.6% |
| 80 | 6.44 | 6.58 | -0.15 | +2.3% |
| 100 | 6.16 | 6.35 | -0.19 | +3.1% |
| 120 | 6.04 | 6.28 | -0.25 | +4.0% |
| 140 | 5.86 | 6.13 | -0.27 | +4.5% |
| 160 | 5.87 | 6.20 | -0.33 | +5.4% |
| 180 | 5.74 | 6.08 | -0.35 | +5.7% |
| 200 | 5.60 | 5.97 | -0.37 | +6.2% |
| 220 | 5.46 | 5.85 | -0.40 | +6.8% |
| 240 | 5.44 | 5.88 | -0.44 | +7.5% |
| 260 | 5.17 | 5.69 | -0.52 | +9.1% |
| 280 | 5.23 | 5.70 | -0.47 | +8.2% |
| 300 | 5.12 | 5.63 | -0.51 | +9.0% |
| 320 | 5.07 | 5.60 | -0.53 | +9.5% |
| 340 | 5.05 | 5.57 | -0.52 | +9.4% |
| 360 | 4.91 | 5.44 | -0.52 | +9.6% |
| 380 | 4.91 | 5.40 | -0.49 | +9.1% |
| 400 | 4.87 | 5.38 | -0.51 | +9.5% |
| 420 | 4.88 | 5.39 | -0.51 | +9.5% |
| 440 | 4.77 | 5.32 | -0.55 | +10.3% |
| 460 | 4.88 | 5.38 | -0.50 | +9.4% |
| 480 | 4.90 | 5.42 | -0.52 | +9.6% |
| **500** | **4.82** | **5.35** | **-0.53** | **+9.9%** |

### Memory Usage

| Metric | Muon | AdamW | Difference |
|--------|------|-------|------------|
| Optimizer States | 1162 MiB | 1546 MiB | **-25%** |
| Model Parameters | 676 MiB | 676 MiB | 0% |
| Activations (est.) | 693 MiB | 693 MiB | 0% |
| Peak Allocated | 13501 MiB | 13885 MiB | -2.8% |

### Parameter Distribution (Muon)

| Category | Parameters | Percentage |
|----------|------------|------------|
| Muon (2D hidden weights) | 201,326,592 | 56.7% |
| AdamW (embeddings/biases/norms) | 153,544,704 | 43.3% |

---

## Analysis

### Convergence Behavior

1. **Early Training (Steps 0-60):** AdamW shows slightly faster initial convergence, which is expected as the learning rate ramps up during warmup.

2. **Mid Training (Steps 60-200):** Muon begins to outperform AdamW, with the gap widening steadily. This aligns with the theoretical advantage of Newton-Schulz orthogonalization for 2D weight matrices.

3. **Late Training (Steps 200-500):** Muon maintains a consistent 9-10% advantage, demonstrating stable convergence even with the cosine learning rate decay.

### Memory Efficiency

Muon's 25% reduction in optimizer state memory comes from:
- Muon only stores momentum (single tensor per parameter)
- AdamW stores both first and second moment estimates (2 tensors per parameter)

This is particularly beneficial for large-scale distributed training where optimizer states are partitioned across data-parallel ranks.

---

## Configuration Files

### Muon Config (`configs/benchmark/verify_muon.yaml`)
```yaml
optim:
  optimizer: muon
  lr_scheduler: cosine
  max_lr: 0.00042
  min_lr: 0.000042
  warmup_steps: 50
  annealing_steps: 500
  weight_decay: 0.1
  muon_momentum: 0.95
  muon_newton_schulz_steps: 5
  muon_lr_scale: 1.0
  adamw_lr_scale: 1.0
```

### AdamW Config (`configs/benchmark/verify_adamw.yaml`)
```yaml
optim:
  optimizer: adam
  lr_scheduler: cosine
  max_lr: 0.00042
  min_lr: 0.000042
  warmup_steps: 50
  annealing_steps: 500
  weight_decay: 0.1
```

---

## Reproduce

```bash
# Muon verification
torchrun --nproc_per_node=2 -m ironcore train --config configs/benchmark/verify_muon.yaml

# AdamW verification
torchrun --nproc_per_node=2 -m ironcore train --config configs/benchmark/verify_adamw.yaml

# View results in TensorBoard
tensorboard --logdir logs/tensorboard
```

---

## Conclusion

The Muon optimizer demonstrates **9.9% better convergence** compared to AdamW when using fair, paper-based learning rate settings. This confirms that Muon's advantage is genuine and not solely due to aggressive learning rate scaling.

Key benefits of Muon:
1. **Better convergence** - Lower final loss with same hyperparameters
2. **Memory efficiency** - 25% less optimizer state memory
3. **Stability** - Consistent performance throughout training

---

## References

- Moonlight paper: arxiv:2502.16982v1
- Muon blog post: https://kellerjordan.github.io/posts/muon/
