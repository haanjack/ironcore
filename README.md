# IronCore

**Personal LLM Training Framework for Learning & Experimentation**

A personal project for practicing AI development and testing training algorithms. Built from scratch to understand LLM training internals — distributed training, parallelism, alignment, and optimization.

Inspired by NVIDIA Megatron-LM and HuggingFace Transformers.

## Features

- **Training modes**: Pretraining, SFT, FIM, and DPO (Direct Preference Optimization)
- **Parallelism**: Tensor Parallelism (TP) and Data Parallelism (DP), including multi-node
- **Model architectures**: GPT-2/3, LLaMA, Gemma, Qwen, Phi via unified `TransformerModel`
- **Mixture of Experts (MoE)**: Expert routing with load-balance loss and expert parallelism
- **PEFT / LoRA**: Parameter-efficient fine-tuning with async and TP-correct implementations
- **Checkpointing**: Native and HuggingFace-interop checkpoint save/load
- **MFU tracking**: Model FLOP utilization monitoring during training
- **Logging**: WandB integration via `WandbLogger`
- Runs on dual RTX 3090 (with NVLink)

**Planned:**
- RL integration (RLHF / RLAIF / PPO)
- Inference engine for rollout and evaluation
- Triton kernels and low-level optimizations

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
| `alignment` | DPO beta and reference model settings |

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
│   ├── alignment/          # DPO configs
│   └── *.yaml              # Training configs
├── ironcore/
│   ├── cli/                # CLI entrypoints (train, preprocess, inspect)
│   ├── config/             # Dataclass configs for all subsystems
│   ├── models/             # TransformerModel implementation
│   ├── layers/             # Attention, MLP, embedding, parallel MLP
│   ├── trainers/           # BaseTrainer, LMTrainer, DPOTrainer
│   ├── dataloader/         # Dataset, collator, data config
│   ├── optimizer/          # Optimizer, LR scheduler, distributed optimizer
│   ├── parallel/           # TP/DP process groups and utilities
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
│   └── *.py                # Data preparation scripts
├── tests/                  # Unit, integration, performance, and property tests
└── docs/                   # Guides and reports
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
