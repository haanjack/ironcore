#!/bin/bash
set -e

echo "Starting Long Context Test (SeqLen=4096): Async TP vs Sync TP"
echo "Experiment Name: long-context-test"
echo "Target Steps: 50"
echo "Logs: logs/long_context_test/"

mkdir -p logs/long_context_test

export MASTER_ADDR=localhost
export MASTER_PORT=12356
export LOCAL_RANK=0
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run Sync Experiment
echo "----------------------------------------------------------------"
echo "Running Synchronous TP Experiment (4k context)..."
python examples/train.py --config-path scripts/async_tp_tests/configs/gpt2_long_sync.yaml
echo "Sync Experiment Completed."

# Run Async Experiment
echo "----------------------------------------------------------------"
echo "Running Asynchronous TP Experiment (4k context, chunk=1024)..."
python examples/train.py --config-path scripts/async_tp_tests/configs/gpt2_long_async.yaml
echo "Async Experiment Completed."
