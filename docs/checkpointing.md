# Checkpointing

## Disk layout

Two on-disk formats are supported, selected by `operation.save_dist_ckpt`:

| Format | Path | When |
|---|---|---|
| Universal | `step_X/pytorch_model.bin` | `save_dist_ckpt: false` (default) |
| Distributed | `step_X/tpN/pytorch_model.bin` | `save_dist_ckpt: true` |

`latest_step.txt` in the model directory records the most recently saved step and is read by `load_checkpoint()` when no explicit step is given, enabling automatic resume.

`config.json` (HuggingFace-compatible) is written alongside every checkpoint by `HFConfigManager.save_hf_config()`.

## Saving

`save_checkpoint()` in `ironcore/checkpointing/native.py` writes a single `torch.save` dict per rank containing:
- `model_state_dict`
- `optimizer_state_dict` (keyed by **parameter name**, not integer index — stable across formats)
- `lr_scheduler`
- `step`
- `config`, `hf_config`

**Universal format (TP > 1, `save_dist_ckpt: false`):** TP rank 0 gathers sharded parameters from all TP workers via `comm.gather_from_model_parallel_workers()` before writing. Only DP rank 0 and TP rank 0 write to disk.

**Distributed format (`save_dist_ckpt: true`):** each TP rank writes its own shard to `step_X/tp{rank}/pytorch_model.bin`. No gather required.

### LoRA-aware gather

When saving a universal checkpoint from a TP-parallel model with LoRA adapters:
- Column-parallel layers: gather `weight`, `bias`, **and `lora_B`**
- Row-parallel layers: gather `weight` **and `lora_A`**

### DistributedOptimizer checkpoint

When `parallel.use_distributed_optimizer: true` and saving a universal checkpoint, `_gather_distributed_optimizer_states()` all-gathers optimizer states (partitioned across DP ranks) to rank 0, then merges into a single keyed-by-name dict.

For distributed checkpoints the local partition is saved directly.

## Loading

`load_checkpoint()` auto-detects format by checking whether `step_X/tpN/` exists.

**Loading universal into TP > 1:** for each parameter, calls `comm.split_to_model_parallel_workers()` to shard the gathered weight across ranks. The same LoRA-aware logic applies in reverse:
- Column-parallel: split `weight`, `bias`, `lora_B`
- Row-parallel: split `weight`, `lora_A`

**Loading optimizer state:** optimizer state is keyed by parameter name, then re-mapped to parameter objects. AdamW `exp_avg`/`exp_avg_sq` tensors are reshaped to match the current parameter shape (handles TP topology changes).

When `parallel.use_distributed_optimizer: true`, `_partition_optimizer_states_for_load()` filters the full state dict to only the parameters owned by this DP rank before calling `optimizer.optimizer.load_state_dict()`.

**DDP prefix normalization:** the loader strips or adds `module.` prefix automatically if the checkpoint was saved from a DDP-wrapped model but is being loaded into a non-DDP model (or vice versa).

## HuggingFace interop

`load_from_huggingface()` and `export_to_huggingface()` in `ironcore/checkpointing/hf_interop.py`:

- `detect_checkpoint_format()` probes for `model.safetensors`, `model.safetensors.index.json`, `pytorch_model.bin`, or `pytorch_model.bin.index.json` (sharded variants supported).
- `WeightMapper` in `ironcore/checkpointing/weight_mapping.py` handles bidirectional key and tensor translation for `Architecture.GPT2` and `Architecture.LLAMA` (aliases: Mistral, Qwen2/3, Gemma, Gemma2, Mixtral).

**LLaMA weight transforms:**
- HF stores linear weights as `[out, in]`; IronCore uses `[in, out]` — transpose applied in `WeightMapper`.
- HF has separate `k_proj` / `v_proj`; IronCore uses fused `linear_kv` — concatenated along the output dim.
- HF has separate `gate_proj` / `up_proj` (SwiGLU); IronCore fuses them into `mlp.up_proj` — concatenated then transposed.

`get_architecture()` resolves model type strings (including `"qwen2"`, `"gemma"`, etc.) to `Architecture.LLAMA`.

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `trainer.model_path` | `""` | Checkpoint directory (empty = no save/load) |
| `trainer.load_from_hf` | `null` | HuggingFace model name or local path to load weights from |
| `operation.save_dist_ckpt` | `false` | Save one file per TP rank instead of universal |
| `optim.load_checkpoint_optim_state` | `true` | Restore optimizer states from checkpoint |
| `optim.load_checkpoint_lr_scheduler` | `true` | Restore LR scheduler state from checkpoint |
