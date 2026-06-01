#!/bin/bash
# Sanity check: 1.1B model, 200 steps, 3 configurations.
# Run inside NGC container:
#   bash scripts/sanity_offload.sh
#
# Note: DP=2 + weight_offload is not supported (DDP all_reduce fails on CPU
# tensors). Only TP=2 is tested for multi-GPU weight streaming.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Single-GPU runs need env vars that torchrun normally provides
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1

CONFIGS=("1_baseline" "2_offload_1gpu" "4_offload_2gpu_tp")
LOGDIR="outputs/sanity_offload/logs"
mkdir -p "$LOGDIR"

echo "=== Sanity Offload Test Suite ==="
echo "Model: ~1.1B (22 layers, d_model=1792, 14 heads)"
echo "Steps: 200 each"
echo ""

# Run 1: Baseline — 1 GPU, no offload
echo "[1/3] Baseline (no offload, 1 GPU)..."
python -m ironcore train --config configs/sanity_offload/1_baseline.yaml 2>&1 | tee "$LOGDIR/1_baseline.log"
echo "[1/3] Done."
echo ""

# Run 2: Offload — 1 GPU
echo "[2/3] Offload (weight streaming + optimizer offload + activation spill, 1 GPU)..."
python -m ironcore train --config configs/sanity_offload/2_offload_1gpu.yaml 2>&1 | tee "$LOGDIR/2_offload_1gpu.log"
echo "[2/3] Done."
echo ""

# Run 3: Offload — 2 GPU TP=2 (the new feature)
echo "[3/3] Offload + TP=2 (weight streaming + tensor parallel, 2 GPUs)..."
torchrun --nproc_per_node 2 -m ironcore train --config configs/sanity_offload/4_offload_2gpu_tp.yaml 2>&1 | tee "$LOGDIR/4_offload_2gpu_tp.log"
echo "[3/3] Done."
echo ""

# Summary: extract final loss from each log
echo "=== Summary ==="
for name in "${CONFIGS[@]}"; do
    log="$LOGDIR/${name}.log"
    if [ -f "$log" ]; then
        final_loss=$(grep -oP 'step: \d+, loss: [0-9.]+' "$log" | tail -1)
        echo "  $name: $final_loss"
    else
        echo "  $name: LOG NOT FOUND"
    fi
done
echo ""
echo "Logs saved to $LOGDIR/"
