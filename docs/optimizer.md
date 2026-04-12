# Optimizer

## Overview

`get_optimizer()` in `ironcore/optimizer/__init__.py` builds the optimizer from config:
- `optim.optimizer == "muon"` → calls `get_muon_optimizer()`, which creates a `MuonOptimizer` with 4 hybrid param groups.
- Any other value → creates `AdamWOptimizer`.

When PEFT is active, base model parameters are frozen before optimizer creation; only adapter parameters are trainable.

## Muon

`MuonOptimizer` in `ironcore/optimizer/muon.py` implements MomentUm Orthogonalized by Newton-Schulz.

### Which parameters get Muon

`is_muon_param()` returns `True` for a parameter when **all** of the following hold:
1. It is a 2D weight tensor (`param.dim() == 2`, `name.endswith(".weight")`)
2. It belongs to an attention or MLP projection (matches one of `linear_q`, `linear_kv`, `attn_output`, `mlp.up_proj`, `mlp.down_proj`, `mlp.gate_proj`)
3. It does **not** match any of: `embedding`, `output_layer`, `lm_head`, `position_embedding`, `pos_embedding`

All other parameters (1D biases, norms, embeddings, output projections) use AdamW.

### Param groups

`get_muon_optimizer()` builds 4 groups:

| Group | Optimizer | Weight decay |
|---|---|---|
| muon + decay | Muon | `weight_decay` |
| muon + no_decay | Muon | 0 |
| adamw + decay | AdamW | `weight_decay` |
| adamw + no_decay | AdamW | 0 |

LRs:
- Muon groups: `max_lr * muon_lr_scale`
- AdamW groups: `max_lr * adamw_lr_scale`

### Algorithm (per step, Muon group)

1. **Momentum buffer update:** `buf = momentum * buf + (1 - momentum) * grad`
2. **Nesterov look-ahead:** `g = momentum * buf + (1 - momentum) * grad`
3. **Newton-Schulz orthogonalization:** `g = zeropower_via_newtonschulz5(g, steps=newton_schulz_steps)`
   - Coefficients `a=3.4445, b=-4.7750, c=2.0315` tuned for 5th-order convergence
   - Works in bfloat16; transposes tall matrices before iteration
4. **RMS scaling:** `g *= 0.2 * sqrt(max(rows, cols))`
5. **Decoupled weight decay:** `p *= (1 - lr * weight_decay)`
6. **Parameter update:** `p -= lr * g`

Non-2D parameters that land in a Muon group fall back to standard SGD with momentum.

## AdamW

`AdamWOptimizer` in `ironcore/optimizer/adamw.py` is a standard AdamW with decoupled weight decay, bias correction, and optional AMSGrad. State tensors (`exp_avg`, `exp_avg_sq`) are stored in fp32.

## DistributedOptimizer (ZeRO-1)

`DistributedOptimizer` in `ironcore/optimizer/distributed_optimizer.py` wraps any optimizer and partitions **optimizer states** across DP ranks:

- Parameters and gradients remain fully replicated (DDP all-reduces gradients as usual).
- Round-robin assignment: parameter `i` is owned by rank `i % dp_size`.
- After each inner optimizer step, updated parameters are broadcast from owner ranks to all others using bucketed async broadcasts (`bucket_cap_mb` controls bucket size).

**Incompatible with FSDP** — `parallel.use_distributed_optimizer` and `parallel.use_fsdp` must not both be true.

Memory savings at DP size N: optimizer states drop from `2P` to `2P/N` bytes; parameters and gradients unchanged at `2P` total.

## LR scheduler

`get_lr_scheduler()` in `ironcore/optimizer/lr_scheduler.py`:

- `"cosine"` (default): `CosineAnnealingLR` — linear warmup for `warmup_steps`, then cosine decay from `max_lr` to `min_lr` over `annealing_steps`, flat at `min_lr` thereafter.
- `"linear"`: `LinearDecayLRScheduler` — linear warmup then linear decay to zero.

If `annealing_steps == 0`, it defaults to `operation.train_steps`.

## Configuration reference

### OptimConfig (`ironcore/config/config_optim.py`)

| Field | Default | Description |
|---|---|---|
| `optimizer` | `"adam"` | `"adam"` or `"muon"` |
| `lr_scheduler` | `"cosine"` | `"cosine"` or `"linear"` |
| `max_lr` | `5e-4` | Peak learning rate |
| `min_lr` | `0.0` | Minimum LR (cosine floor) |
| `warmup_steps` | `0` | Linear warmup steps |
| `annealing_steps` | `0` | Cosine decay steps (0 = train_steps) |
| `weight_decay` | `0.01` | Decoupled weight decay |
| `adam_beta1` | `0.9` | AdamW β₁ |
| `adam_beta2` | `0.95` | AdamW β₂ |
| `adam_eps` | `1e-8` | AdamW ε |
| `clip_grad` | `1.0` | Gradient clipping threshold |
| `muon_momentum` | `0.95` | Muon Nesterov momentum β |
| `muon_newton_schulz_steps` | `5` | Newton-Schulz iterations |
| `muon_lr_scale` | `1.0` | Muon LR = `max_lr * muon_lr_scale` |
| `adamw_lr_scale` | `1.0` | AdamW LR = `max_lr * adamw_lr_scale` |
| `load_checkpoint_optim_state` | `true` | Load optimizer state from checkpoint |
| `load_checkpoint_lr_scheduler` | `true` | Load LR scheduler state from checkpoint |

### ParallelConfig (distributed optimizer fields)

| Field | Default | Description |
|---|---|---|
| `use_distributed_optimizer` | `false` | Enable ZeRO-1 optimizer state partitioning |
| `dist_opt_bucket_cap_mb` | `25.0` | Broadcast bucket size in MB |
| `use_fsdp` | `false` | Enable FSDP (mutually exclusive with distributed optimizer) |
