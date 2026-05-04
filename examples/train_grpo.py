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


def main():
    parser = argparse.ArgumentParser(description="GRPO training example")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    args = parser.parse_args()

    config = load_trainer_config(args.config)

    with GRPOTrainer(config) as trainer:
        trainer.train()


if __name__ == "__main__":
    main()
