# Checkpointing

> This guide covers how to configure and use checkpointing. For the internal architecture,
> TP weight gather/split mechanics, and DistributedOptimizer state handling, see the
> [Checkpointing system design](design/checkpointing.md).

## Overview

IronCore saves checkpoints in two formats:

| Format | Path | Best for |
|---|---|---|
| **Universal** | `step_N/pytorch_model.bin` | Portability, changing TP degree |
| **Distributed** | `step_N/tp{r}/pytorch_model.bin` | Fast parallel saves, fixed TP |

Both formats store model weights, optimizer states, LR scheduler state, and step counter in a single `torch.save` dict per rank. `latest_step.txt` tracks the most recently written step for automatic resume.

## Basic setup

Set `trainer.model_path` to enable saving and loading:

```yaml
trainer:
  model_path: checkpoints/my-run
  save_checkpoint_steps: 500   # save every 500 steps
```

Training automatically resumes from the latest checkpoint on restart. To disable saving (e.g., evaluation runs):

```yaml
operation:
  no_save: true
```

## Resuming training

On `train()`, IronCore reads `latest_step.txt` and loads the matching checkpoint before the first step. Training continues from `step = last_saved_step + 1`. No config change needed — just re-run the same command.

To resume from a specific step rather than the latest:

```python
from ironcore.checkpointing import load_checkpoint

load_checkpoint(config, model, optimizer, lr_scheduler, step=1000)
```

To resume weights only (skip optimizer state — e.g., after architecture change or fine-tuning from a pretrained base):

```yaml
optim:
  load_checkpoint_optim_state: false
  load_checkpoint_lr_scheduler: false
```

## Universal checkpointing (changing TP degree)

The default format (`save_dist_ckpt: false`) is TP-degree agnostic. You can save at TP=4 and
resume at TP=1, or scale up from TP=2 to TP=8, without any conversion step.

```yaml
operation:
  save_dist_ckpt: false   # default — universal format
```

On save, IronCore gathers TP-sharded weights and optimizer moment tensors from all TP ranks and writes a single file from rank 0. On load, the full tensors are split for the new TP degree automatically.

This applies to LoRA adapters too: `lora_B` is gathered/split with column-parallel layers; `lora_A` with row-parallel layers.

## Distributed checkpointing (parallel I/O)

When saving speed matters more than portability, enable distributed format. Each TP rank writes its own shard concurrently:

```yaml
operation:
  save_dist_ckpt: true
```

This produces `step_N/tp{r}/pytorch_model.bin` per rank. Saves complete `TP_size` times faster at scale, but the checkpoint is tied to the saved TP degree — loading at a different TP will fail.

## HuggingFace interop

### Loading from a HuggingFace checkpoint

```yaml
trainer:
  load_from_hf: meta-llama/Llama-3.1-8B   # HF model id or local path
```

`detect_checkpoint_format()` handles safetensors, PyTorch, single-file, and sharded formats automatically. Before building the model, run `detect_bias_from_hf_state_dict()` to infer which projections have biases so `BiasConfig` is set correctly.

### Exporting to HuggingFace format

```python
from ironcore.checkpointing.hf_interop import export_to_huggingface

export_to_huggingface(
    config=config,
    model=model,
    save_directory="exports/llama-finetuned",
    format="safetensors",  # or "pytorch"
    max_shard_size="5GB",
)
```

Output is the standard HF layout:

```
exports/llama-finetuned/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-00003.safetensors
└── ...
```

### HF config.json generation

To write a HF-compatible `config.json` alongside native checkpoints (required for direct `AutoModel.from_pretrained` use):

```yaml
model:
  hf_model_type: llama
  hf_architecture: LlamaForCausalLM
```

Both fields must be set. When set, `config.json` is written to `{model_path}/` on every save.

## LoRA checkpoints

LoRA adapter weights are saved **together** with base model weights — no separate adapter file:

```
step_N/pytorch_model.bin
  model.layers.0.linear_q.weight    ← base weight
  model.layers.0.linear_q.lora_A    ← LoRA A
  model.layers.0.linear_q.lora_B    ← LoRA B
  ...
```

Loading works identically to a full checkpoint — PEFT config must match at load time (same `r`, `alpha`, `target_modules`).

## Inspecting checkpoints

```bash
# Summary: format, parameter count, size, dtype breakdown, step
ironcore inspect-checkpoint --path checkpoints/my-run

# Per-layer shapes and statistics
ironcore inspect-checkpoint --path checkpoints/my-run --verbose

# Diff two checkpoints (max_abs_diff, mean_abs_diff per layer)
ironcore inspect-checkpoint --path checkpoints/run-a --compare checkpoints/run-b
```

## Configuration reference

| Field | Group | Default | Description |
|---|---|---|---|
| `model_path` | `trainer` | `""` | Checkpoint directory (empty = no save/load) |
| `load_from_hf` | `trainer` | `null` | HF model id or local path to load from |
| `save_checkpoint_steps` | `trainer` | — | Save every N steps |
| `no_save` | `operation` | `false` | Disable checkpoint saving |
| `save_dist_ckpt` | `operation` | `false` | `true` = distributed per-rank, `false` = universal |
| `load_checkpoint_optim_state` | `optim` | `true` | Restore optimizer states on resume |
| `load_checkpoint_lr_scheduler` | `optim` | `true` | Restore LR scheduler state on resume |
| `hf_model_type` | `model` | `null` | HF model type string; enables `config.json` generation |
| `hf_architecture` | `model` | `null` | HF architecture class name |
