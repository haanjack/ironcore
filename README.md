# IronCore

**Personal LLM Training Framework for Learning & Experimentation**

A personal project for practicing AI development and testing training algorithms. Built from
scratch to understand LLM training internals — distributed training, parallelism, alignment,
and optimization.

Inspired by NVIDIA Megatron-LM, HuggingFace Transformers, and from my own experiences.

## Features

- **Training modes** — Pretraining, SFT, DPO, and GRPO (Group Relative Policy Optimization)
- **Parallelism** — Tensor Parallelism (TP), Expert Parallelism (EP), Data Parallelism (DP),
  multi-node, and FSDP; `DistributedOptimizer` for ZeRO-1 state sharding
- **Model architectures** — GPT-2/3, LLaMA/LLaMA-2/3, Gemma/Gemma-2, Qwen/Qwen2/Qwen3,
  Mistral, Mixtral, Phi-1/2 via a single `TransformerModel`; GQA/MQA, RoPE, SwiGLU/GeGLU
- **HF weight loading** — load any LLaMA-family HF checkpoint directly (Qwen2.5, Llama-3,
  Gemma-2, Mistral, Mixtral, …) via `trainer.pretrained_model_name_or_path`
- **Mixture of Experts (MoE)** — expert routing with load-balance loss and Z-loss, expert parallelism
- **PEFT / LoRA** — TP-correct, replicated adapters; `offloadable=False` keeps adapters on GPU
  while base weights stream to host
- **GRPO / RL alignment** — online rollout generation, group-relative advantage normalization,
  KL penalty, multi-epoch replay with IS-ratio clipping, multi-backend rewards (math, code,
  keyword, API, local model)
- **Optimizer** — Muon (Newton-Schulz orthogonalization) + AdamW hybrid, ZeRO-1
  `DistributedOptimizer`
- **Offload** — RAM-first staircase scaling: optimizer-state offload, weight streaming with GPU
  staging pool, activation spilling; combine all three for maximum VRAM reduction
- **Checkpointing** — native (universal + distributed TP formats) and HuggingFace-interop save/load
- **KV cache** — stateful `KVCacheManager` for inference; block-based `BlockKVCacheManager`
  with prefix sharing and reference-counted CoW for efficient GRPO rollout
- **Data preprocessing** — FIM (Fill-in-the-Middle) with PSM format, configurable split
  strategies, bin-packing SFT collator
- **MFU tracking** — Model FLOP utilization monitoring via `MFUCalculator`
- **Logging** — TensorBoard, WandB, and MLflow via pluggable logger classes

Runs on my precious dual RTX 3090 (with NVLink).

<details>
<summary>My test machine</summary>
<image src="docs/assets/my_rig.png" alt="My Rig" width="600"/>
</details>

## Installation

IronCore requires the NGC PyTorch container for full functionality — flash attention ships with
the base image and cannot be installed via pip on the host.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete container-first setup guide.

**Quick start inside the container:**

```bash
git clone https://github.com/haanjack/ironcore
cd ironcore
pip install -e ".[dev]"
```

## Quick Start

### Container Setup

```bash
# Copy and fill in DATASET_DIR and MODEL_DIR
cp .env.example .env

# Build the container
./scripts/docker/build.sh                # CUDA (default)
ARCH=rocm-wsl ./scripts/docker/build.sh  # ROCm (WSL2)

# Launch an interactive shell
./scripts/docker/launch.sh bash
```

The repo is mounted at `/workspace` inside the container.

Running on an AMD APU under WSL2 (e.g. Strix Halo / gfx1151) needs the ROCDXG bridge, which
`ARCH=rocm-wsl` bakes into the image — see [docs/rocm_wsl_setup.md](docs/rocm_wsl_setup.md).

### Data Preprocessing

```bash
ironcore preprocess --config configs/data/pretrain_example.yaml
ironcore preprocess --config configs/data/pretrain_example.yaml --inspect
```

### Training

The training mode is set by `data.task_type` in your config (`pretrain`, `sft`, `dpo`, `grpo`).

**Single GPU — smoke test with random data:**
```bash
ironcore train --config configs/example.yaml
```

**Tensor Parallel (2 GPUs):**
```yaml
# configs/example.yaml
trainer:
  tensor_model_parallel_size: 2
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

**DPO:**
```bash
ironcore train --config configs/alignment/dpo_default.yaml
```

**GRPO** — set `data.task_type: grpo` and `alignment.method: grpo`; see
[docs/alignment.md](docs/alignment.md) for the full field reference.

**LoRA fine-tuning:**
```bash
ironcore train --config configs/train_lora_example.yaml
```

**Train beyond your VRAM (offload):**
```yaml
# Add to any config to offload optimizer states + stream weights
offload:
  enabled: true
  optimizer_offload: true
  optimizer_state_precision: "bf16"
  weight_offload: true
  activation_spill: true
```

### Load a HuggingFace checkpoint

```yaml
trainer:
  pretrained_model_name_or_path: "Qwen/Qwen2.5-0.5B-Instruct"
  # or a local path to any LLaMA-family HF checkpoint
```

## Configuration

| Group | Description |
|-------|-------------|
| `model` | Model architecture (`gpt2-small`, `llama`, etc.) |
| `data` | Dataset config, task type (`pretrain` \| `sft` \| `dpo` \| `grpo`), tokenizer |
| `trainer` | Batch sizes, parallelism, checkpointing, flash attention |
| `optim` | Optimizer (`adam` \| `muon`), learning rate, scheduler |
| `operation` | Training steps, eval intervals, activation recompute |
| `peft` | LoRA rank, alpha, target modules |
| `alignment` | DPO/GRPO method, beta, group size, rollout, reward manager |
| `offload` | Optimizer offload, weight streaming, activation spilling |
| `init` | Random seed, init std |

### Supported Model Architectures

| Family | Notes |
|--------|-------|
| GPT | `gpt2-small` through `gpt3` |
| LLaMA | LLaMA, LLaMA-2, LLaMA-3; HF weight loading supported |
| Gemma | Gemma 1 + Gemma 2; HF weight loading supported |
| Qwen | Qwen, Qwen2, Qwen2.5, Qwen3; HF weight loading supported |
| Mistral / Mixtral | LLaMA-architecture mapping; HF weight loading supported |
| Phi | Phi-1, Phi-2 |

**Architecture features:** Pre/post-norm, RMSNorm, GQA/MQA, RoPE, GELU/SiLU/SwiGLU/GeGLU, MoE

**Limitations:** No sliding window attention, no multimodal support, no encoder-decoder.

### Example Config

```yaml
trainer:
  micro_batch_size: 4
  train_batch_size: 8
  gradient_accumulation_steps: 2
  tensor_model_parallel_size: 1
  save_checkpoint_steps: 1000
  log_interval: 10
  model_path: outputs/example

operation:
  train_steps: 2000
  eval_interval: 1000

model: gpt2-small

data:
  task_type: pretrain       # pretrain | sft | dpo | grpo
  use_mock_data: true       # use random data (no preprocessing needed)

optim:
  optimizer: adam           # adam | muon
  lr_scheduler: cosine
  max_lr: 6.0e-4
  min_lr: 6.0e-5
  warmup_steps: 100
  weight_decay: 0.1
  clip_grad: 1.0

init:
  seed: 42
```

## Documentation

| Topic | Doc |
|-------|-----|
| Contributing (setup, coding standards, PR workflow) | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Getting started | [docs/getting_started.md](docs/getting_started.md) |
| CLI guide & reference | [docs/cli_guide.md](docs/cli_guide.md) |
| Parallelism (TP/EP/DP/FSDP) | [docs/parallelism.md](docs/parallelism.md) |
| Trainers (BaseTrainer lifecycle) | [docs/trainers.md](docs/trainers.md) |
| Optimizer (Muon + AdamW, ZeRO-1) | [docs/optimizer.md](docs/optimizer.md) |
| Offload (RAM-first staircase scaling) | [docs/offload.md](docs/offload.md) |
| Alignment (DPO + GRPO) | [docs/alignment.md](docs/alignment.md) |
| Reward system (GRPO rewards) | [docs/reward_manager.md](docs/reward_manager.md) |
| Dataloader (streaming, bin-packing, FIM) | [docs/dataloader.md](docs/dataloader.md) |
| Inference & KV cache | [docs/inference.md](docs/inference.md) |
| Evaluation (HellaSwag + perplexity) | [docs/eval.md](docs/eval.md) |
| LoRA / PEFT guide | [docs/peft_guide.md](docs/peft_guide.md) |
| Checkpointing (native + HF interop) | [docs/checkpointing.md](docs/checkpointing.md) |
| CI/CD setup | [docs/ci_cd_guide.md](docs/ci_cd_guide.md) |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
