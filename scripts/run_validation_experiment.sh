#!/usr/bin/env bash
# Run full SFT -> DPO validation experiment on HH-RLHF with GPT-2 Small
# Usage: bash scripts/run_validation_experiment.sh
# Expects to be run inside the Docker container from /workspace

set -euo pipefail

WORKSPACE="/workspace"
cd "$WORKSPACE"

echo "======================================================================"
echo "  DPO Validation Experiment: GPT-2 Small on HH-RLHF"
echo "======================================================================"
echo ""

# ── Phase 1: SFT ─────────────────────────────────────────────────────────
echo "[Phase 1] SFT Training (3000 steps, GPT-2 Small)"
echo "  Config: configs/val_sft_gpt2_small.yaml"
echo "  Output: models/val_sft_gpt2_small"
echo ""

torchrun --nproc_per_node=1 -m ironcore train \
    --config configs/val_sft_gpt2_small.yaml \
    2>&1 | tee /tmp/sft_training.log

echo ""
echo "[Phase 1] SFT training complete!"
echo ""

# ── Phase 2: Prepare DPO Init ─────────────────────────────────────────────
echo "[Phase 2] Preparing DPO init checkpoint from SFT weights"

python scripts/prepare_dpo_init.py \
    --sft-path models/val_sft_gpt2_small \
    --dpo-path models/val_dpo_gpt2_small

echo "[Phase 2] DPO init checkpoint ready!"
echo ""

# ── Phase 3: DPO ─────────────────────────────────────────────────────────
echo "[Phase 3] DPO Training (2000 steps, beta=0.1)"
echo "  Config: configs/val_dpo_gpt2_small.yaml"
echo "  Output: models/val_dpo_gpt2_small"
echo ""

torchrun --nproc_per_node=1 -m ironcore train \
    --config configs/val_dpo_gpt2_small.yaml \
    2>&1 | tee /tmp/dpo_training.log

echo ""
echo "[Phase 3] DPO training complete!"
echo ""

# ── Phase 4: Evaluation ───────────────────────────────────────────────────
echo "[Phase 4] Evaluating SFT preference accuracy"

torchrun --nproc_per_node=1 scripts/eval_preference_accuracy.py \
    --config configs/val_sft_gpt2_small.yaml \
    --test-data data/local/hh_rlhf/dpo_test.jsonl \
    --max-samples 500 \
    --output /tmp/eval_sft.json \
    2>&1 | tee /tmp/eval_sft.log

echo ""
echo "[Phase 4] Evaluating DPO preference accuracy"

torchrun --nproc_per_node=1 scripts/eval_preference_accuracy.py \
    --config configs/val_dpo_gpt2_small.yaml \
    --test-data data/local/hh_rlhf/dpo_test.jsonl \
    --max-samples 500 \
    --output /tmp/eval_dpo.json \
    2>&1 | tee /tmp/eval_dpo.log

echo ""
echo "======================================================================"
echo "  EXPERIMENT COMPLETE"
echo "======================================================================"
echo ""
echo "Results:"
echo "  SFT:  $(cat /tmp/eval_sft.json 2>/dev/null || echo 'N/A')"
echo "  DPO:  $(cat /tmp/eval_dpo.json 2>/dev/null || echo 'N/A')"
