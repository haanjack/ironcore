# Convergence Validation: GPT-2 small vs nanoGPT

**Status:** In progress  
**Config:** `configs/experiments/nanogpt_convergence.yaml` + `configs/model/nanogpt-small.yaml`  
**Reference:** [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — `configs/train_gpt2.py`

## Goal

Verify that IronCore's pretraining loss curve matches the nanoGPT reference implementation
on identical hyperparameters. A close match confirms that the training loop, optimizer, and
data pipeline produce numerically equivalent results.

## Setup

| | IronCore | nanoGPT |
|---|---|---|
| Model | GPT-2 small (124M) | GPT-2 small (124M) |
| Dataset | OpenWebText (99/1 train/val split) | OpenWebText (99/1 train/val split) |
| Tokens/step | 491,520 (480 seq × 1024) | 491,520 |
| Peak LR | 6e-4 | 6e-4 |
| Min LR | 6e-5 | 6e-5 |
| LR schedule | Cosine | Cosine |
| Warmup | 2,000 steps | 2,000 steps |
| Weight decay | 0.1 | 0.1 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) | AdamW (β₁=0.9, β₂=0.95) |
| Grad clip | 1.0 | 1.0 |
| Seed | 1337 | 1337 |
| Bias | **None** (bias=False) | None (bias=False) |
| LayerNorm bias | **None** | None |
| GELU variant | **`gelu_new`** (tanh approx.) | `gelu_new` |

## Reference values (nanoGPT)

| Tokens processed | Steps | nanoGPT val loss | Est. time (RTX 3090) |
|---|---|---|---|
| ~500M | 1,000 | ~3.30 | ~45 min |
| ~2.5B | 5,000 | ~3.10 | ~3.5 h |
| **~5B** | **10,000** | **~3.00** | **~7 h** |
| ~10B | 20,000 | ~2.97 | ~14 h |
| ~25B | 50,000 | ~2.90 | ~35 h |
| convergence | 200,000+ | ~2.85 | — |

**Recommended run:** 10k steps — enough to validate the loss curve against multiple reference points in ~7 hours.

> **LR schedule note:** `annealing_steps` is set to 600,000 (matching nanoGPT's `lr_decay_iters`)
> regardless of `train_steps`. This ensures the LR at step N is identical to nanoGPT's LR at step N.

## IronCore results

<!-- Fill in after training runs -->

| Tokens processed | Steps | IronCore val loss | nanoGPT val loss | Δ |
|---|---|---|---|---|
| ~500M | 1,000 | — | ~3.30 | — |
| ~5B | 10,000 | — | ~3.00 | — |
| ~25B | 50,000 | — | ~2.90 | — |
| ~49B | 100,000 | — | ~2.87 | — |

## How to run

```bash
# Preprocess OpenWebText (one-time)
ironcore preprocess --config configs/data/nanogpt_owt.yaml

# Train (single GPU, ~14h on RTX 3090 for 100k steps)
ironcore train --config configs/experiments/nanogpt_convergence.yaml

# Resume if interrupted
ironcore train --config configs/experiments/nanogpt_convergence.yaml
# (auto-resumes from latest_step.txt in outputs/nanogpt_convergence/)
```

Logs go to the run's WandB project (or stdout if WandB is not configured).
Track `val_loss` vs `tokens_processed` for direct comparison with nanoGPT.

## Notes on differences

- **FlashAttention:** IronCore uses FlashAttention by default; nanoGPT has an optional FA path. Both give numerically equivalent outputs.
- **Vocab padding:** `vocab_padding_unit: 128` pads the vocabulary from 50,257 to 50,304 for CUDA efficiency. The extra embeddings are zero-initialized and not reachable from the tokenizer — no effect on loss.
- **Dropout:** nanoGPT supports configurable dropout; this run uses 0.0 (no dropout) matching the nanoGPT default training config.
- **Model config:** `nanogpt-small.yaml` differs from the general-purpose `gpt2-small.yaml` in three ways: `bias=False` for all linear projections, `layernorm_bias=False`, and `activation_type=gelu_new`. These match nanoGPT's `train_gpt2.py` exactly.
