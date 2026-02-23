# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Training Script Example

This script demonstrates how to launch training for both Pretraining and Supervised Fine-Tuning (SFT).
The distinction between Pretraining and SFT is determined solely by the `task_type` field
in your dataset configuration YAML file.

Usage:
    torchrun -m ironcore --config configs/example.yaml

    OR directly with this script:
    torchrun examples/train.py --config configs/example.yaml

Configuration:
    To switch between Pretraining and SFT, modify the `task_type` in your data config:

    For Pretraining:
        train_datasets:
          - name: openwebtext
            task_type: pretrain  <-- Set to 'pretrain'
            ...

    For SFT:
        train_datasets:
          - name: ultrachat
            task_type: sft       <-- Set to 'sft'
            ...
"""

from ironcore.config import load_trainer_config
from ironcore.trainer import Trainer
from ironcore.training_utils import forward_step, loss_func

if __name__ == "__main__":
    config = load_trainer_config()

    with Trainer(config, forward_step_func=forward_step, loss_fn=loss_func) as trainer:
        trainer.train()
