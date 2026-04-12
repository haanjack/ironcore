# Getting Started

## Installation

```bash
git clone <repo>
cd ironcore-dev
pip install -e .
```

## Training

### Single-node, single-GPU

```bash
ironcore train --config configs/example.yaml
```

### Distributed (torchrun)

```bash
torchrun --nproc_per_node 4 -m ironcore train --config configs/<name>.yaml
```

`RANK`, `LOCAL_RANK`, and `WORLD_SIZE` are set automatically by `torchrun` and read by the trainer.

## Configuration structure

Configs are YAML files parsed into `MainConfig`. All sections are optional and fall back to defaults.

```yaml
# Trainer settings
trainer:
  micro_batch_size: 4
  gradient_accumulation_steps: 8
  tensor_model_parallel_size: 1   # TP degree
  model_path: models/my_run       # checkpoint directory
  save_checkpoint_steps: 1000
  use_flash_attn: true

# Training operation
operation:
  train_steps: 10000
  eval_interval: 1000
  activation_recompute: false

# Model (name refers to a built-in config, or specify inline)
model: gpt2-small

# Data
data:
  task_type: pretrain             # pretrain | sft | dpo | grpo
  seq_length: 1024
  datasets:
    - source: openwebtext
      task_type: pretrain
      ratio: 1.0

# Optimizer
optim:
  optimizer: adam                 # adam | muon
  lr_scheduler: cosine
  max_lr: 6e-4
  min_lr: 6e-5
  warmup_steps: 100
  weight_decay: 0.1
  clip_grad: 1.0

# Initialization
init:
  seed: 1337
```

See `configs/example.yaml` for a complete pretrain example and `configs/train_lora_example.yaml` for a LoRA fine-tuning example.

## Loading pretrained HuggingFace weights

```yaml
trainer:
  load_from_hf: "Qwen/Qwen2.5-0.5B-Instruct"
```

Weights are downloaded via `huggingface_hub.snapshot_download` and loaded by `load_from_huggingface()`. Architecture is auto-detected from `config.json`. See `docs/checkpointing.md` for weight mapping details.

## Data preprocessing

```bash
ironcore preprocess --config configs/data/my_data.yaml
```

Tokenizes raw text files and writes `.bin`/`.idx` files for `StreamingBinaryDataset`. For FIM preprocessing, set `data.fim_rate > 0` in the data config.

## Subsystem documentation

- **Checkpointing** (native + HF interop): `docs/checkpointing.md`
- **Optimizer** (Muon + AdamW, ZeRO-1): `docs/optimizer.md`
- **Trainers** (BaseTrainer lifecycle): `docs/trainers.md`
- **Alignment** (DPO + GRPO): `docs/alignment.md`
- **Dataloader** (streaming datasets, bin-packing): `docs/dataloader.md`
- **Inference & KV cache**: `docs/inference.md`
- **Evaluation** (HellaSwag + perplexity): `docs/eval.md`
- **Reward system** (GRPO rewards): `docs/reward_manager.md`
