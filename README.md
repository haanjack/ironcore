# IronCore

**Personal LLM Training Framework for Learning & Experimentation**

A personal project for practicing AI development and testing training algorithms. Built from scratch to understand LLM training internals — distributed training, parallelism, alignment, and optimization.

Inspired by NVIDIA Megatron-LM, HuggingFace Transformers, and from my own experiences.

## Features

- **Training modes**: Pretraining, SFT, FIM, DPO, and GRPO (Group Relative Policy Optimization)
- **Parallelism**: Tensor Parallelism (TP), Expert Parallelism (EP) and Data Parallelism (DP), including multi-node and FSDP
- **Model architectures**: GPT-2/3, LLaMA, Gemma, Qwen, Phi via unified `TransformerModel`
- **Mixture of Experts (MoE)**: Expert routing with load-balance loss and expert parallelism
- **PEFT / LoRA**: Parameter-efficient fine-tuning with TP-correct implementations
- **GRPO (WIP) / RL alignment**: Online rollout generation, group-relative advantage, KL penalty, multi-backend rewards (math, code, API, local model)
- **Checkpointing**: Native and HuggingFace-interop checkpoint save/load
- **KV cache**: Paged KV cache with prefix caching for efficient rollout generation
- **MFU tracking**: Model FLOP utilization monitoring during training
- **Logging**: WandB integration via `WandbLogger`
- Runs on my precious dual RTX 3090 (with NVLink)
<details>
<summary>My test Machine</summary>
<!-- <image src="https://raw.githubusercontent.com/hanjack/ironcore/main/docs/assets/my_rig.png" alt="My Rig" width="600"/> -->
<image src="docs/assets/my_rig.png" alt="My Rig" width="600"/>
</details>

## Installation

```bash
git clone <repo-url>
cd ironcore
pip install -e .
```

## Quick Start

### Docker Setup

```bash
cp .env.example .env
./scripts/docker/build.sh
docker run -it --gpus all -v $(pwd):/workspace ironcore:dev
```

### Data Preprocessing

```bash
ironcore preprocess --config configs/data/pretrain_example.yaml
ironcore preprocess --config configs/data/pretrain_example.yaml --inspect
ironcore preprocess --config configs/data/pretrain_example.yaml --only-inspect
```

### Training

The training mode is determined by `task_type` in your data config (`pretrain`, `sft`, `fim`, `dpo`).

**Single GPU:**
```bash
ironcore train --config configs/example.yaml
```

**Tensor Parallel (2 GPUs):**
```bash
torchrun --nproc_per_node 2 -m ironcore train --config configs/example.yaml --tensor-model-parallel-size 2
```

**Data Parallel (2 GPUs):**
```bash
torchrun --nproc_per_node 2 -m ironcore train --config configs/example.yaml --tensor-model-parallel-size 1
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

**GRPO Training (GSM8K math reasoning):**
```bash
ironcore train --config configs/grpo_gsm8k.yaml
```

**GRPO Training (toy/smoke test):**
```bash
ironcore train --config configs/grpo_toy.yaml
```

**LoRA Fine-tuning:**
```bash
ironcore train --config configs/train_lora_example.yaml
```

## Configuration

| Group | Description |
|-------|-------------|
| `model` | Model architecture (`gpt2-small`, `llama`, etc.) |
| `data` | Dataset config and preprocessing |
| `trainer` | Batch sizes, parallelism, checkpointing |
| `optim` | Optimizer, learning rate, scheduler |
| `operation` | Training steps, eval intervals |
| `peft` | LoRA rank, alpha, target modules |
| `alignment` | DPO/GRPO method, beta, group size |
| `alignment.generation` | Rollout generation (temperature, top-p, chat template) |
| `alignment.reward` | Reward backend (math, code, API, local model) |

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
  config_path: configs/data/full_owt_pretrain.yaml

optim:
  optimizer: adam
  max_lr: 6.0e-4
  warmup_steps: 100
```

## Project Structure

```
ironcore/
├── configs/
│   ├── model/              # Model architecture configs
│   ├── data/               # Data preprocessing configs
│   ├── peft/               # LoRA configs
│   ├── alignment/          # DPO/GRPO configs
│   └── *.yaml              # Training configs (incl. grpo_gsm8k, grpo_toy)
├── ironcore/
│   ├── cli/                # CLI entrypoints (train, preprocess, inspect)
│   ├── config/             # Dataclass configs for all subsystems
│   ├── models/             # TransformerModel implementation
│   ├── layers/             # Attention, MLP, embedding, KV cache
│   ├── trainers/           # BaseTrainer, LMTrainer, DPOTrainer, GRPOTrainer
│   ├── alignment/          # Rollout, rewards, GRPO/DPO loss, buffer, dataset
│   ├── dataloader/         # Dataset, collator, data config
│   ├── optimizer/          # Optimizer, LR scheduler, distributed optimizer
│   ├── parallel/           # TP/DP/EP process groups and utilities
│   ├── peft/               # LoRA implementation and utilities
│   ├── checkpointing/      # Native and HF-interop checkpointing
│   ├── eval/               # Evaluator and scoring utilities
│   ├── preprocessing/      # Tokenized data serialization
│   ├── tokenizer/          # Tokenizer wrapper
│   ├── mfu.py              # MFU (Model FLOP Utilization) calculator
│   ├── logger.py           # WandB logger
│   └── trainer.py          # Trainer entrypoint
├── examples/               # Standalone usage examples
├── scripts/
│   ├── docker/             # Docker build and launch scripts
│   └── *.py                # Data preparation and debug scripts
├── tests/                  # Unit, integration, performance, and property tests
└── docs/                   # Guides and reports
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
