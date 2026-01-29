# IronCore

**Personal LLM Training Framework for Learning & Experimentation**

IronCore is a personal project for practicing AI development and testing training algorithms. Built from scratch to understand the internals of LLM training:
- **Learning by Building**: Hands-on implementation of distributed training, parallelism strategies, and optimization techniques
- **Algorithm Testing**: Playground for experimenting with training methods and model architectures
- **Unified Data Pipeline**: Single preprocessing system for pretrain, SFT, and DPO tasks
- **Consumer Hardware**: Developed on Dual RTX 3090 (with NVLINK!!)
- **RL Integration**: Bridge training and inference for RLHF/RLAIF experiments - not yet
- **Inference Engine**: Integrate rollout and evaluation capabilities - not yet
- **DSL Experimentation**: Testing Triton kernels and low-level optimizations (with or without agent) - not yet

This project inspired by NVIDIA Megatron-LM and HuggingFace's Transformers.


## Installation

```bash
# Clone and install in editable mode
git clone <repo-url>
cd ironcore
pip install -e .
```

## Quick Start

### Docker Setup

1. Create a `.env` file for build arguments (optional):
   ```bash
   cp .env.example .env
   # Edit .env if you need to provide a github_access_token or other vars
   ```

2. Build the Docker image:
   ```bash
   ./scripts/docker/build.sh
   ```

3. Run container with GPU support:
   ```bash
   docker run -it --gpus all -v $(pwd):/workspace ironcore:dev
   ```

### Data Preprocessing

Preprocess your dataset before training:

```bash
# Preprocess dataset
ironcore preprocess --config configs/data/pretrain_example.yaml

# Preprocess with inspection
ironcore preprocess --config configs/data/pretrain_example.yaml --inspect

# Inspect existing preprocessed data
ironcore preprocess --config configs/data/pretrain_example.yaml --only-inspect
```

### Training

IronCore supports both **Pretraining** and **Supervised Fine-Tuning (SFT)** using the same training command. The mode is determined by the `task_type` field in your dataset configuration file (under `configs/data/`).

- **Pretraining**: Set `task_type: pretrain` in the data config.
- **SFT**: Set `task_type: sft` in the data config.

**Single GPU:**
```bash
ironcore train --config configs/example.yaml
```

**Distributed Training (Data Parallel):**
```bash
# DP with 2 GPUs
torchrun --nproc_per_node 2 -m ironcore train --config configs/example.yaml --tensor-model-parallel-size 1
```

**Distributed Training (Tensor Parallel):**
```bash
# TP with 2 GPUs
torchrun --nproc_per_node 2 -m ironcore train --config configs/example.yaml --tensor-model-parallel-size 2
```

**Multi-node Training:**
```bash
# On each node (adjust --node_rank for each)
torchrun --nproc_per_node 8 --nnodes 2 --node_rank 0 \
    --master_addr <MASTER_IP> --master_port 29500 \
    -m ironcore train --config configs/example.yaml
```

## Configuration

IronCore uses a hierarchical YAML configuration system with the following groups:

| Group | Description |
|-------|-------------|
| `model` | Model architecture (e.g., `gpt2-small`, `llama-7b`) |
| `data` | Dataset configuration and preprocessing settings |
| `trainer` | Batch sizes, parallelism, checkpointing |
| `optim` | Optimizer, learning rate, scheduler |
| `operation` | Training steps, evaluation intervals |
| `init` | Random seed, initialization |
| `utils` | Logging and utilities |

### Supported Model Architectures

IronCore uses a unified `TransformerModel` that supports multiple architectures through configuration. Model-specific components (normalization, activation functions, attention patterns) are configured via YAML files.

**Supported Models:**
- **GPT**: `gpt2-small`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `gpt3`
- **LLaMA**: `llama-7b`, `llama-13b`, `llama-70b` (and derivatives)
- **Gemma**: `gemma-1-2b`, `gemma-1-7b` (Gemma 1 only - Gemma 2/3 require sliding window attention)
- **Qwen**: `qwen-*` models
- **Phi**: `phi-1`, `phi-2` (Phi-3 is multimodal, not supported)

**Architecture Features:**
- Pre-norm and post-norm layer normalization
- RMSNorm and LayerNorm support
- Grouped Query Attention (GQA) and Multi-Query Attention (MQA)
- RoPE (Rotary Position Embeddings)
- Activation functions: GELU, SiLU, SwiGLU, GeGLU

**Current Limitations:**
- No sliding window attention (required for Mistral, Gemma 2/3)
- No multimodal support (required for Phi-3, LLaVA, etc.)
- No encoder-decoder architectures (T5, BART, etc.)

### Example Config Structure

```yaml
# Main training config
trainer:
  micro_batch_size: 4
  train_batch_size: 480
  gradient_accumulation_steps: 60
  tensor_model_parallel_size: 1

operation:
  train_steps: 2000
  eval_interval: 1000

model: gpt2-small  # References configs/model/gpt2-small.yaml

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
│   ├── model/          # Model architecture configs
│   ├── data/           # Data preprocessing configs
│   └── *.yaml          # Training configs
├── ironcore/
│   ├── cli/            # Command-line interface
│   ├── config/         # Configuration classes
│   ├── dataloader/     # Data loading utilities
│   ├── models/         # Model implementations (TransformerModel)
│   ├── layers/         # Attention, MLP, normalization layers
│   ├── parallel/       # Parallelism utilities (TP, DP)
│   └── trainer.py      # Main trainer
├── data/
│   ├── preprocessed/   # Preprocessed binary data
│   └── cache/          # HuggingFace cache
└── scripts/
    └── docker/         # Docker build scripts
```
