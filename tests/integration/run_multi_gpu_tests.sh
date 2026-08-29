#!/bin/bash
# Run multi-GPU tests for KV cache with Tensor Parallelism
#
# Usage:
#   ./run_multi_gpu_tests.sh              # Run with 2 GPUs (default)
#   ./run_multi_gpu_tests.sh 4            # Run with 4 GPUs
#   ./run_multi_gpu_tests.sh 2 --verbose  # Run with extra flags
#
# Requirements:
#   - torchrun (part of PyTorch)
#   - Multiple GPUs available
#   - NCCL or Gloo backend configured

set -e

# Default number of GPUs
NUM_GPUS=${1:-2}
shift || true

# Pass remaining arguments to pytest
EXTRA_ARGS="$@"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=============================================="
echo "Running Multi-GPU KV Cache Tests"
echo "=============================================="
echo "Number of GPUs: ${NUM_GPUS}"
echo "Extra args: ${EXTRA_ARGS}"
echo "Project root: ${PROJECT_ROOT}"
echo "=============================================="

# Check if torchrun is available
if ! command -v torchrun &> /dev/null; then
    echo "Error: torchrun not found. Please install PyTorch."
    exit 1
fi

# Check if enough GPUs are available
AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo "Warning: Requested ${NUM_GPUS} GPUs but only ${AVAILABLE_GPUS} available."
    echo "Tests may fail or be skipped."
fi

# Run TP2 tests
echo ""
echo ">>> Running TP=${NUM_GPUS} KV Cache Tests <<<"
echo ""

torchrun --nproc_per_node=${NUM_GPUS} \
    -m pytest \
    "${SCRIPT_DIR}/../multi_gpu/kvcache/test_kv_cache_tp.py" \
    -v \
    --tb=short \
    ${EXTRA_ARGS}

echo ""
echo "=============================================="
echo "Multi-GPU tests completed!"
echo "=============================================="
