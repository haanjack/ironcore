# Trainers

## Trainer selection

`ironcore/cli/train.py` picks the trainer from `data.task_type`:

| `task_type` | Trainer | Use case |
|---|---|---|
| `pretrain` | `LanguageModelTrainer` | Next-token prediction on raw text |
| `sft` | `LanguageModelTrainer` | Supervised fine-tuning with response-only loss masking |
| `dpo` | `DPOTrainer` | Direct Preference Optimization |
| `grpo` | `GRPOTrainer` | Group Relative Policy Optimization (online rollout) |

## BaseTrainer

All trainers extend `BaseTrainer`. The initialization order is fixed — do not override `train()`:

1. Distributed process setup
2. TP/DP process groups
3. EP process groups (MoE only)
4. Build model + optimizer
5. `torch.compile(model)` — must precede DDP/FSDP wrapping
6. DDP or FSDP wrapping

Training resumes automatically if a checkpoint exists at `trainer.model_path`. Checkpoints
are saved every `save_checkpoint_steps` steps and at the end of training.

### Extension hooks

Override these in subclasses instead of `train()`:

| Hook | When called |
|---|---|
| `_pre_train_setup()` | Before training loop — loads checkpoint, calls `_post_checkpoint_load()` |
| `_post_checkpoint_load(last_step)` | After checkpoint load |
| `_on_checkpoint_save(step)` | Before each checkpoint save |

## LanguageModelTrainer

Used for `pretrain` and `sft`. Loss is next-token cross-entropy. Per-token accuracy is logged alongside loss.

**SFT response-only masking:** prompt tokens are set to `-100` in `labels` by the SFT collator.
Only response tokens contribute to the loss. For `pretrain`, all tokens are unmasked.

## DPOTrainer

Requires a preference dataset with chosen/rejected pairs. A frozen reference model is created
from the loaded checkpoint at the start of training. Under FSDP, the reference model is
built from a state dict copy (not `deepcopy`) to avoid FSDP internal state entanglement.

See [alignment.md](alignment.md) for DPO config options.

## GRPOTrainer

Each training step:

1. **Rollout phase** — generates completions in chunks of `grpo_rollout_micro_group_size`
2. **Reward scoring** — `RewardManager` scores completions (parallelized via thread pool)
3. **Update phase** — runs `grpo_num_epochs` gradient steps over the rollout buffer

See [alignment.md](alignment.md) and [reward_manager.md](reward_manager.md) for config.

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `trainer.micro_batch_size` | `2` | Per-GPU batch size for a single forward pass |
| `trainer.gradient_accumulation_steps` | `null` | Micro-batches before a parameter update (derived from `train_batch_size` when unset) |
| `trainer.compile_model` | `false` | Enable `torch.compile` |
| `trainer.compile_mode` | `"default"` | `default` \| `reduce-overhead` \| `max-autotune` |
| `trainer.compile_backend` | `"inductor"` | Compiler backend |
| `trainer.compile_dynamic` | `false` | Allow dynamic shapes |
| `operation.activation_recompute` | `false` | Recompute activations instead of storing |
| `operation.train_steps` | — | Total training steps |
| `operation.eval_interval` | `100` | Evaluate every N steps |
| `trainer.tensor_model_parallel_size` | `1` | TP degree |
