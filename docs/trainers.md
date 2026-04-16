# Trainers

## Trainer selection

`ironcore/cli/train.py` picks the trainer based on `data.task_type`:

| `task_type` | Trainer |
|---|---|
| `pretrain`, `sft` | `LanguageModelTrainer` |
| `dpo` | `DPOTrainer` |
| `grpo` | `GRPOTrainer` |

## BaseTrainer lifecycle

`BaseTrainer` in `ironcore/trainers/base_trainer.py` is the abstract base for all trainers.

### Construction (`__init__`)

Lightweight — only stores config and sets up logging/control objects. No GPU resources are acquired.

### Initialization (`_initialize()`)

Called by `__enter__` or automatically at the start of `train()`. Idempotent. Acquires all heavy resources in this fixed order:

1. `initialize_process(config)` — sets up the distributed environment
2. `initialize_model_parallel(tensor_model_parallel_size)` — creates TP/DP process groups
3. `initialize_expert_parallel(...)` — only when `model.moe.use_moe` and `expert_model_parallel_size > 1`
4. `get_data_iterator(config)` — builds train/eval data iterators
5. Build model (`LanguageModel`) and optimizer (`get_optimizer`)
6. Optionally wrap with `DistributedOptimizer`
7. **`torch.compile(model)`** — must happen before parallelism wrapping
8. `initialize_parallelism(config, model)` — wraps with DDP or FSDP

**Critical:** `torch.compile` must precede `initialize_parallelism()`. Compiling after DDP/FSDP wrapping produces incorrect results.

### Context manager pattern

```python
with LanguageModelTrainer(config, forward_step, loss_fn) as trainer:
    trainer.train()
```

`__exit__` calls `dist.destroy_process_group()` and closes loggers.

### Training loop (`train()`)

Template method — subclasses do not override it directly:

1. Calls `_pre_train_setup()` (checkpoint load + subclass setup)
2. Loops `train_step(step)` until `operation.train_steps`
3. Checkpoints at `trainer.save_checkpoint_steps`, evaluates at `operation.eval_interval`
4. Saves a final checkpoint if the loop ended without hitting a checkpoint boundary

### Extension hooks

Override these in subclasses — do **not** override `train()`:

| Hook | When called | Default behaviour |
|---|---|---|
| `_pre_train_setup()` | Before training loop | Loads checkpoint, calls `_post_checkpoint_load()` |
| `_post_checkpoint_load(last_step)` | After checkpoint load | No-op |
| `_on_checkpoint_save(step)` | Before each checkpoint | No-op |

### Gradient accumulation

`_run_gradient_accumulation()` loops `gradient_accumulation_steps` micro-batches. DDP/FSDP gradient synchronization is suppressed on all but the final micro-batch using the `model.no_sync()` context manager.

### Mixed precision

`torch.autocast` is active for the entire forward pass. `GradScaler` is enabled only when dtype is `float16`. Gradients are unscaled (via `scaler.unscale_()`) before gradient clipping and norm computation.

## LanguageModelTrainer

`ironcore/trainers/language_model_trainer.py` — used for `pretrain` and `sft`.

Each `train_step` runs gradient accumulation, computes grad/param norms, and steps the optimizer. Loss is next-token cross-entropy. An additional per-token accuracy metric is computed and logged.

### SFT response-only loss masking

When `task_type: sft`, the SFT collator produces `labels` with prompt tokens set to `-100`. `LanguageModel.get_masks_and_position_ids()` derives `loss_mask` from these labels: positions where `labels == -100` get `loss_mask = 0` (ignored in loss), all others get `loss_mask = 1`. This matches TRL/axolotl behavior where only response tokens contribute to the loss.

For `task_type: pretrain`, labels are never masked and `loss_mask` is all ones.

## DPOTrainer

`ironcore/trainers/dpo_trainer.py`

`_post_checkpoint_load()` creates a frozen reference model (deep copy of the policy model). Under FSDP, the reference model is built from a `state_dict` copy rather than `deepcopy` to avoid internal FSDP state entanglement.

Each `train_step` runs forward passes on chosen and rejected sequences (optionally concatenated when `alignment.concat_forward_passes: true`) and calls `dpo_loss()`. See `docs/alignment.md`.

## GRPOTrainer

`ironcore/trainers/grpo_trainer.py`

Each `train_step` first generates rollouts via `generate_rollouts_batched()` (in chunks of `grpo_rollout_micro_group_size`), scores them through `RewardManager`, computes advantages, and then runs `grpo_num_epochs` gradient update epochs over the rollout buffer. See `docs/alignment.md`.

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `trainer.micro_batch_size` | `2` | Per-GPU batch size for a single forward pass |
| `trainer.gradient_accumulation_steps` | `null` | Micro-batches before a parameter update (derived from `train_batch_size` when unset) |
| `trainer.compile_model` | `false` | Enable `torch.compile` |
| `trainer.compile_mode` | `"default"` | Compilation mode: `default` \| `reduce-overhead` \| `max-autotune` |
| `trainer.compile_backend` | `"inductor"` | Compiler backend |
| `trainer.compile_dynamic` | `false` | Allow dynamic shapes |
| `operation.activation_recompute` | `false` | Recompute activations instead of storing |
| `trainer.tensor_model_parallel_size` | `1` | Number of TP ranks |
