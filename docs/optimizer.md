# Optimizer

> This guide covers optimizer selection and configuration. For the Muon algorithm
> (Newton-Schulz orthogonalization), ZeRO-1 partition mechanics, and LR scheduler
> internals, see the [Optimizer system design](design/optimizer.md).

## Choosing an optimizer

| | AdamW (`adam`) | Muon (`muon`) |
|---|---|---|
| Applies to | All parameters | 2D weights (attention + MLP projections) — AdamW for the rest |
| Memory | `2P` optimizer states | Same (Muon buffers are on-device) |
| When to use | General purpose, safe default | Transformer training; often converges faster |

Muon applies only to 2D weight matrices in attention and MLP projections. All other
parameters (embeddings, biases, norms, output layer) always use AdamW regardless of which
optimizer is selected.

```yaml
optim:
  optimizer: muon   # adam | muon
```

## DistributedOptimizer (ZeRO-1)

Wraps any optimizer to shard optimizer states across DP ranks. Parameters and gradients
remain fully replicated; only moment tensors are partitioned, saving `(N-1)/N` of optimizer
state memory at DP size N.

```yaml
parallel:
  use_distributed_optimizer: true
  dist_opt_bucket_cap_mb: 25.0   # parameter broadcast bucket size
```

**Incompatible with FSDP** — use one or the other for optimizer state sharding.

**Note:** Only AdamW states can be offloaded to CPU RAM when combined with the offload
subsystem. Muon's orthogonalization requires on-device computation.

## LR scheduler

```yaml
optim:
  lr_scheduler: cosine   # cosine | linear
  max_lr: 5e-4
  min_lr: 5e-5
  warmup_steps: 100
  annealing_steps: 0     # 0 = use operation.train_steps
```

- **`cosine`** (default): linear warmup, then cosine decay from `max_lr` to `min_lr`, flat at `min_lr` thereafter.
- **`linear`**: linear warmup, then linear decay to zero.

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `optimizer` | `"adam"` | `"adam"` or `"muon"` |
| `lr_scheduler` | `"cosine"` | `"cosine"` or `"linear"` |
| `max_lr` | `5e-4` | Peak learning rate |
| `min_lr` | `0.0` | Minimum LR (cosine floor) |
| `warmup_steps` | `0` | Linear warmup steps |
| `annealing_steps` | `0` | Cosine decay steps (0 = `train_steps`) |
| `weight_decay` | `0.01` | Decoupled weight decay |
| `adam_beta1` | `0.9` | AdamW β₁ |
| `adam_beta2` | `0.95` | AdamW β₂ |
| `adam_eps` | `1e-8` | AdamW ε |
| `clip_grad` | `1.0` | Gradient clipping threshold |
| `muon_momentum` | `0.95` | Muon Nesterov momentum |
| `muon_newton_schulz_steps` | `5` | Newton-Schulz iterations |
| `muon_lr_scale` | `1.0` | Muon LR = `max_lr × muon_lr_scale` |
| `adamw_lr_scale` | `1.0` | AdamW LR = `max_lr × adamw_lr_scale` |
| `load_checkpoint_optim_state` | `true` | Restore optimizer state from checkpoint |
| `load_checkpoint_lr_scheduler` | `true` | Restore LR scheduler state from checkpoint |
| `use_distributed_optimizer` | `false` | ZeRO-1 optimizer state partitioning |
| `dist_opt_bucket_cap_mb` | `25.0` | Broadcast bucket size in MB |
