#!/bin/bash
# Memory profiling script for Muon vs AdamW comparison using Docker
# Requires: Docker with NVIDIA Container Toolkit

set -e

IMAGE_NAME="ironcore-muon:latest"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================"
echo "Building Docker image..."
echo "========================================"
docker build -t $IMAGE_NAME -f Dockerfile .

echo ""
echo "========================================"
echo "Memory Profiling: Muon vs AdamW"
echo "========================================"
echo "Running 3 steps per configuration"
echo ""

# Test configurations
MICRO_BATCH_SIZES=(2 4 8)

for MBS in "${MICRO_BATCH_SIZES[@]}"; do
    GAS=$((128 / MBS / 2))

    echo "========================================"
    echo "Micro-batch size: $MBS, Gradient accumulation: $GAS"
    echo "========================================"

    # Muon test
    echo ""
    echo "--- MUON (mbs=$MBS) ---"
    docker run --rm --gpus all \
        -v $PROJECT_ROOT:/workspace \
        -w /workspace \
        $IMAGE_NAME \
        bash -c "
            cat > /tmp/mem_test.yaml << 'EOFCONFIG'
trainer:
  micro_batch_size: $MBS
  train_batch_size: 128
  gradient_accumulation_steps: $GAS
  tensor_model_parallel_size: 1
  save_checkpoint_steps: 10000
  log_interval: 1
  model_path: /tmp/mem_test
  use_flash_attn: true
  vocab_padding_unit: 128

operation:
  train_steps: 3
  eval_interval: 10000
  eval_samples: 100
  activation_recompute: false
  no_save: true

model:
  num_attention_heads: 16
  num_attention_groups: 16
  head_dim: 64
  max_seq_len: 1024
  max_position_embeddings: 1024
  num_layers: 24
  d_model: 1024
  d_ffn: 4096
  untie_embed: false
  reset_position_ids: false
  reset_attention_mask: false
  eod_mask_loss: false
  precision: bf16
  vocab_name_or_path: gpt2
  tokenizer_type: bbpe
  hf_model_type: gpt2
  hf_architecture: GPT2LMHeadModel

data:
  config_path: configs/data/openwebtext_test.yaml
  task_type: pretrain

optim:
  optimizer: muon
  lr_scheduler: cosine
  max_lr: 0.00042
  min_lr: 0.000042
  warmup_steps: 500
  annealing_steps: 2000
  weight_decay: 0.1
  muon_momentum: 0.95
  muon_newton_schulz_steps: 5
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1.0e-8
  clip_grad: 1.0

init:
  seed: 42
  init_std: 0.02

parallel:
  use_distributed_optimizer: true

utils:
  tensorboard_dir: /tmp/tensorboard/mem_test
EOFCONFIG
            torchrun --nproc_per_node=2 -m ironcore train --config /tmp/mem_test.yaml
        " 2>&1 | grep -E "(step:|Memory|Peak|loss:)" | head -10

    # AdamW test
    echo ""
    echo "--- ADAMW (mbs=$MBS) ---"
    docker run --rm --gpus all \
        -v $PROJECT_ROOT:/workspace \
        -w /workspace \
        $IMAGE_NAME \
        bash -c "
            cat > /tmp/mem_test.yaml << 'EOFCONFIG'
trainer:
  micro_batch_size: $MBS
  train_batch_size: 128
  gradient_accumulation_steps: $GAS
  tensor_model_parallel_size: 1
  save_checkpoint_steps: 10000
  log_interval: 1
  model_path: /tmp/mem_test
  use_flash_attn: true
  vocab_padding_unit: 128

operation:
  train_steps: 3
  eval_interval: 10000
  eval_samples: 100
  activation_recompute: false
  no_save: true

model:
  num_attention_heads: 16
  num_attention_groups: 16
  head_dim: 64
  max_seq_len: 1024
  max_position_embeddings: 1024
  num_layers: 24
  d_model: 1024
  d_ffn: 4096
  untie_embed: false
  reset_position_ids: false
  reset_attention_mask: false
  eod_mask_loss: false
  precision: bf16
  vocab_name_or_path: gpt2
  tokenizer_type: bbpe
  hf_model_type: gpt2
  hf_architecture: GPT2LMHeadModel

data:
  config_path: configs/data/openwebtext_test.yaml
  task_type: pretrain

optim:
  optimizer: adam
  lr_scheduler: cosine
  max_lr: 0.00042
  min_lr: 0.000042
  warmup_steps: 500
  annealing_steps: 2000
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1.0e-8
  clip_grad: 1.0

init:
  seed: 42
  init_std: 0.02

parallel:
  use_distributed_optimizer: true

utils:
  tensorboard_dir: /tmp/tensorboard/mem_test
EOFCONFIG
            torchrun --nproc_per_node=2 -m ironcore train --config /tmp/mem_test.yaml
        " 2>&1 | grep -E "(step:|Memory|Peak|loss:)" | head -10

    echo ""
done

echo "========================================"
echo "Memory profiling complete"
echo "========================================"
