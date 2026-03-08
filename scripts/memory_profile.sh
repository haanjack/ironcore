#!/bin/bash
# Memory profiling script for Muon vs AdamW comparison
# Runs 3 steps for each micro-batch size configuration

set -e

MICRO_BATCH_SIZES=(2 4 8 16)
STEPS=3

echo "========================================"
echo "Memory Profiling: Muon vs AdamW"
echo "========================================"
echo "Micro-batch sizes to test: ${MICRO_BATCH_SIZES[@]}"
echo "Steps per run: $STEPS"
echo ""

for MBS in "${MICRO_BATCH_SIZES[@]}"; do
    # Calculate gradient accumulation steps to maintain batch size of 128
    GAS=$((128 / MBS / 2))  # Divide by 2 for 2 GPUs

    echo "========================================"
    echo "Testing micro_batch_size=$MBS, gradient_accumulation_steps=$GAS"
    echo "========================================"

    # Create temporary config for this test
    cat > /tmp/mem_test_muon.yaml << EOF
trainer:
  micro_batch_size: $MBS
  train_batch_size: 128
  gradient_accumulation_steps: $GAS
  tensor_model_parallel_size: 1
  save_checkpoint_steps: 10000
  log_interval: 1
  model_path: /tmp/mem_test_muon
  use_flash_attn: true
  vocab_padding_unit: 128

operation:
  train_steps: $STEPS
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
  tensorboard_dir: /tmp/tensorboard/mem_test_muon
EOF

    cat > /tmp/mem_test_adamw.yaml << EOF
trainer:
  micro_batch_size: $MBS
  train_batch_size: 128
  gradient_accumulation_steps: $GAS
  tensor_model_parallel_size: 1
  save_checkpoint_steps: 10000
  log_interval: 1
  model_path: /tmp/mem_test_adamw
  use_flash_attn: true
  vocab_padding_unit: 128

operation:
  train_steps: $STEPS
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
  tensorboard_dir: /tmp/tensorboard/mem_test_adamw
EOF

    echo ""
    echo "--- MUON (mbs=$MBS) ---"
    torchrun --nproc_per_node=2 -m ironcore train --config /tmp/mem_test_muon.yaml 2>&1 | grep -E "(memory|Memory|GPU|MiB|GiB|Peak|Allocated|Reserved)" || echo "No memory stats captured"

    echo ""
    echo "--- ADAMW (mbs=$MBS) ---"
    torchrun --nproc_per_node=2 -m ironcore train --config /tmp/mem_test_adamw.yaml 2>&1 | grep -E "(memory|Memory|GPU|MiB|GiB|Peak|Allocated|Reserved)" || echo "No memory stats captured"

    echo ""
done

echo "========================================"
echo "Memory profiling complete"
echo "========================================"
