# IronCore

**Personal LLM Training Framework for Learning & Experimentation**

A personal project for practicing AI development and testing training algorithms. Built from scratch to understand LLM training internals — distributed training, parallelism, alignment, and optimization.

Inspired by NVIDIA Megatron-LM, HuggingFace Transformers, and from my own experiences.

## Features

- **Training modes**: Pretraining, SFT, DPO, and GRPO (Group Relative Policy Optimization)
- **Data preprocessing**: FIM (Fill-in-the-Middle) with PSM format and configurable split strategies
- **Parallelism**: Tensor Parallelism (TP), Expert Parallelism (EP), Data Parallelism (DP), multi-node, and FSDP
- **Model architectures**: GPT-2/3, LLaMA, Gemma, Qwen, Phi via unified `TransformerModel`
- **Mixture of Experts (MoE)**: Expert routing with load-balance and Z-loss, expert parallelism
- **PEFT / LoRA**: Parameter-efficient fine-tuning with TP-correct implementations
- **GRPO / RL alignment**: Online rollout generation, group-relative advantage normalization, KL penalty, multi-epoch replay with IS ratio clipping, multi-backend rewards (math, code, keyword, API, local model)
- **Optimizer**: Muon (Newton-Schulz orthogonalization) + AdamW hybrid with 4 param groups; ZeRO-1 `DistributedOptimizer`
- **Checkpointing**: Native (universal + distributed TP formats) and HuggingFace-interop save/load
- **KV cache**: Stateful `KVCacheManager` with prefix caching for efficient rollout generation
- **MFU tracking**: Model FLOP utilization monitoring during training via `MFUCalculator`
- **Logging**: TensorBoard, WandB, and MLflow via pluggable logger classes (`TensorboardLogger`, `WandbLogger`, `MLFlowLogger`)
- Runs on my precious dual RTX 3090 (with NVLink)

<details>
<summary>My test machine</summary>
<image src="docs/assets/my_rig.png" alt="My Rig" width="600"/>
</details>

## Installation

IronCore requires the NGC PyTorch container for full functionality — flash attention ships with the base image and cannot be installed via pip on the host.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete container-first setup guide.

**Quick start inside the container:**

```bash
git clone <repo-url>
cd ironcore
pip install -e .[dev]
```

## Quick Start

### Container Setup

```bash
# Copy and fill in DATASET_DIR and MODEL_DIR
cp .env.example .env

# Build the container
./scripts/docker/build.sh          # CUDA (default)
ARCH=rocm ./scripts/docker/build.sh  # ROCm

# Launch an interactive shell
./scripts/docker/launch.sh bash
```

The repo is mounted at `/workspace` inside the container.

### Data Preprocessing

```bash
ironcore preprocess --config configs/data/pretrain_example.yaml
ironcore preprocess --config configs/data/pretrain_example.yaml --inspect
```

### Training

The training mode is determined by `data.task_type` in your config (`pretrain`, `sft`, `dpo`, `grpo`).

**Single GPU:**
```bash
ironcore train --config configs/example.yaml
```

**Tensor Parallel (2 GPUs) — set TP degree in config:**
```yaml
# In your config YAML
trainer:
  tensor_model_parallel_size: 2
```
```bash
torchrun --nproc_per_node 2 -m ironcore train --config configs/example.yaml
```

**Data Parallel (2 GPUs) — ensure TP degree is 1 in config:**
```yaml
# In your config YAML
trainer:
  tensor_model_parallel_size: 1
```
```bash
torchrun --nproc_per_node 2 -m ironcore train --config configs/example.yaml
```

**Multi-node:**
```bash
torchrun --nproc_per_node 8 --nnodes 2 --node_rank 0 \
    --master_addr <MASTER_IP> --master_port 29500 \
    -m ironcore train --config configs/example.yaml
```

**DPO Training:**
```bash
ironcore train --config configs/alignment/dpo_default.yaml
```

**GRPO Training:**
```bash
ironcore train --config configs/alignment/<grpo_config>.yaml
```

> A consolidated GRPO example config is not yet in `configs/alignment/`. Set `data.task_type: grpo` and `alignment.method: grpo` in your own config; see [docs/alignment.md](docs/alignment.md) for the full field reference.

**LoRA Fine-tuning:**
```bash
ironcore train --config configs/train_lora_example.yaml
```

## Configuration

| Group | Description |
|-------|-------------|
| `model` | Model architecture (`gpt2-small`, `llama`, etc.) |
| `data` | Dataset config, task type (`pretrain` \| `sft` \| `dpo` \| `grpo`), and tokenizer |
| `trainer` | Batch sizes, parallelism, checkpointing |
| `optim` | Optimizer, learning rate, scheduler |
| `operation` | Training steps, eval intervals |
| `peft` | LoRA rank, alpha, target modules |
| `alignment` | DPO/GRPO method, beta, group size |
| `alignment.generation` | Rollout generation (temperature, top-p, chat template) |
| `alignment.reward_manager` | Reward backend (math, code, keyword, API, local model) |
| `init` | Random seed |

### Supported Model Architectures

| Family | Models |
|--------|--------|
| GPT | `gpt2-small`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `gpt3` |
| LLaMA | `llama-7b`, `llama-13b`, `llama-70b` |
| Gemma | `gemma-1-2b`, `gemma-1-7b` (Gemma 1 only) |
| Qwen | `qwen-*` |
| Phi | `phi-1`, `phi-2` |

**Architecture features:** Pre/post-norm, RMSNorm, GQA/MQA, RoPE, GELU/SiLU/SwiGLU/GeGLU

**Limitations:** No sliding window attention, no multimodal support, no encoder-decoder.

### Example Config

```yaml
trainer:
  micro_batch_size: 4
  train_batch_size: 480
  gradient_accumulation_steps: 60
  tensor_model_parallel_size: 1

operation:
  train_steps: 2000
  eval_interval: 1000

model: gpt2-small

data:
  task_type: pretrain  # pretrain | sft | dpo | grpo
  config_path: configs/data/full_owt_pretrain.yaml

optim:
  optimizer: adam
  max_lr: 6.0e-4
  warmup_steps: 100
```

## Documentation

| Topic | Doc |
|-------|-----|
| Contributing (setup, coding standards, PR workflow) | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Getting started | [docs/getting_started.md](docs/getting_started.md) |
| Checkpointing (native + HF interop) | [docs/checkpointing.md](docs/checkpointing.md) |
| Optimizer (Muon + AdamW, ZeRO-1) | [docs/optimizer.md](docs/optimizer.md) |
| Trainers (BaseTrainer lifecycle) | [docs/trainers.md](docs/trainers.md) |
| Alignment (DPO + GRPO) | [docs/alignment.md](docs/alignment.md) |
| Dataloader (streaming, bin-packing, FIM) | [docs/dataloader.md](docs/dataloader.md) |
| Inference & KV cache | [docs/inference.md](docs/inference.md) |
| Evaluation (HellaSwag + perplexity) | [docs/eval.md](docs/eval.md) |
| Reward system (GRPO rewards) | [docs/reward_manager.md](docs/reward_manager.md) |
| LoRA / PEFT guide | [docs/peft_guide.md](docs/peft_guide.md) |
| CI/CD setup | [docs/ci_cd_guide.md](docs/ci_cd_guide.md) |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
