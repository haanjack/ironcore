#!/bin/bash
set -e

echo "Starting A/B Test: Async TP (Chunk Size=512) vs Sync TP"
echo "Experiment Name: async-tp test"
echo "Target Steps: 200"
echo "Logs: logs/async_tp_test/"

# Ensure logs dir exists
mkdir -p logs/async_tp_test

# Prepare Dummy Data
echo "Preparing Dummy Data..."
python scripts/prepare_dummy_data.py

export MASTER_ADDR=localhost
export MASTER_PORT=12355
export LOCAL_RANK=0
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run Sync Experiment
echo "----------------------------------------------------------------"
echo "Running Synchronous TP Experiment..."
echo "Config: scripts/async_tp_tests/configs/gpt2_small_sync.yaml"
python examples/train.py --config-path scripts/async_tp_tests/configs/gpt2_small_sync.yaml
echo "Sync Experiment Completed."

# Run Async Experiment
echo "----------------------------------------------------------------"
echo "Running Asynchronous TP Experiment..."
echo "Config: scripts/async_tp_tests/configs/gpt2_small_async.yaml"
python examples/train.py --config-path scripts/async_tp_tests/configs/gpt2_small_async.yaml
echo "Async Experiment Completed."

echo "----------------------------------------------------------------"
echo "A/B Test Completed. Inspect logs with Tensorboard:"
echo "tensorboard --logdir logs/async_tp_test"
