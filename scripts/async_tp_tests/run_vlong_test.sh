#!/bin/bash
set -e

echo "Starting Very Long Context Test (SeqLen=8192): Async TP vs Sync TP"
echo "Experiment Name: vlong-context-test"
echo "Target Steps: 20" # Reduced steps for very long context
echo "Logs: logs/vlong_context_test/"

mkdir -p logs/vlong_context_test

export MASTER_ADDR=localhost
export MASTER_PORT=12357 # Changed port to avoid conflict
export LOCAL_RANK=0
export PYTHONPATH=$(pwd):$PYTHONPATH

# Prepare Dummy Data (for 8192 seq len)
echo "Preparing Dummy Data for 8192 seq len..."
# The dummy data script creates 1M tokens in 1000 docs of 1000 tokens.
# This is enough for 8192 seq len, as dataloader slices as needed.
python scripts/prepare_dummy_data.py

# Run Sync Experiment
echo "----------------------------------------------------------------"
echo "Running Synchronous TP Experiment (8k context)..."
python examples/train.py --config-path scripts/async_tp_tests/configs/gpt2_vlong_sync.yaml --max_seq_len 8192
echo "Sync Experiment Completed."

# Run Async Experiment
echo "----------------------------------------------------------------"
echo "Running Asynchronous TP Experiment (8k context, chunk=1024)..."
python examples/train.py --config-path scripts/async_tp_tests/configs/gpt2_vlong_async.yaml --max_seq_len 8192
echo "Async Experiment Completed."
