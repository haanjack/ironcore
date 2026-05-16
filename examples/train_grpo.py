# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO Training Example

Demonstrates Group Relative Policy Optimization training with optional
paged KV cache rollout for efficient generation.

Usage:
    torchrun --nproc_per_node=2 examples/train_grpo.py --config tests/fixtures/configs/grpo_paged_smoke.yaml
    torchrun --nproc_per_node=2 examples/train_grpo.py --config tests/fixtures/configs/grpo_baseline_smoke.yaml
"""

import argparse

from ironcore.config import load_trainer_config
from ironcore.trainers.grpo_trainer import GRPOTrainer
from ironcore.training_utils import forward_step, get_loss_func


def main():
    import sys

    parser = argparse.ArgumentParser(description="GRPO training example")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    args = parser.parse_args()

    sys.argv = ["train", "--config-path", args.config]
    config = load_trainer_config()

    loss_fn = get_loss_func(config.data.task_type)
    with GRPOTrainer(config, forward_step_func=forward_step, loss_fn=loss_fn) as trainer:
        trainer.train()


if __name__ == "__main__":
    main()
