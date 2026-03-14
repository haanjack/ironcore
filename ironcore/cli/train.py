# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Training CLI command."""

import os
import sys
from argparse import Namespace
from pathlib import Path

from ironcore.config import (
    AlignmentConfig,
    DataConfig,
    InitConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PEFTConfig,
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
    _config_validation,
    _load_config_from_yaml,
)
from ironcore.trainers import DPOTrainer, GRPOTrainer, LanguageModelTrainer
from ironcore.training_utils import forward_step, get_loss_func

# Get rank early for conditional printing
_RANK = int(os.getenv("RANK", "0"))


def _print(msg: str = "") -> None:
    """Print only from rank 0."""
    if _RANK == 0:
        print(msg)


def run_train(args):
    """Run training command.

    Args:
        args: Command-line arguments from argparse
            - config: Path to training configuration YAML file
    """
    config_path = Path(args.config)

    if not config_path.exists():
        _print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    _print(f"Loading training configuration from: {config_path}")

    # Create a namespace with the config path for _load_config_from_yaml
    config_args = Namespace(config_path=str(config_path))

    # Initialize default config
    config = MainConfig(
        model=ModelConfig(),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        peft=PEFTConfig(),
        profiler=ProfilerConfig(),
        alignment=AlignmentConfig(),
    )

    # Load config from YAML using the proper loader
    _load_config_from_yaml(config, config_args)

    # Set rank/world_size from environment
    config.parallel.rank = _RANK
    config.parallel.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    config.parallel.world_size = int(os.getenv("WORLD_SIZE", "1"))

    # Validate config
    _config_validation(config)

    # Select loss function based on task type (now a declared field in DataConfig)
    task_type = config.data.task_type
    loss_fn = get_loss_func(task_type)
    _print(f"Task type: {task_type}, using loss function: {loss_fn.__name__}")

    # Initialize trainer based on task type
    _print("\nInitializing trainer...")
    if task_type in ("pretrain", "sft"):
        _print("Using LanguageModelTrainer for language modeling")
        trainer = LanguageModelTrainer(config, forward_step_func=forward_step, loss_fn=loss_fn)
    elif task_type == "dpo":
        # Validate alignment config for DPO
        if config.alignment is None or config.alignment == AlignmentConfig():
            _print(
                "Error: DPO requires 'alignment' configuration section in config file. "
                "Please define alignment hyperparameters (e.g., beta, label_smoothing)."
            )
            sys.exit(1)

        # Validate DPO-specific parameters
        if config.alignment.dpo_beta <= 0:
            _print(f"Error: alignment.dpo_beta must be positive, got {config.alignment.dpo_beta}")
            sys.exit(1)

        _print("Using DPOTrainer for preference optimization")
        trainer = DPOTrainer(config, forward_step_func=forward_step, loss_fn=loss_fn)
    elif task_type == "grpo":
        # Validate alignment config for GRPO
        if config.alignment is None or config.alignment == AlignmentConfig():
            _print(
                "Error: GRPO requires 'alignment' configuration section in config file. "
                "Please define alignment hyperparameters (e.g., grpo_beta, grpo_group_size)."
            )
            sys.exit(1)

        # Validate GRPO-specific parameters
        if config.alignment.grpo_beta <= 0:
            _print(
                f"Error: alignment.grpo_beta should be positive, got {config.alignment.grpo_beta}"
            )
            sys.exit(1)
        if config.alignment.grpo_group_size <= 0:
            _print(
                f"Error: alignment.grpo_group_size should be positive, got {config.alignment.grpo_group_size}"
            )
            sys.exit(1)

        _print("Using GRPOTrainer for Group Relative Policy Optimization")
        trainer = GRPOTrainer(config, forward_step_func=forward_step, loss_fn=loss_fn)
    else:
        _print(f"Error: Unknown task type: {task_type}")
        sys.exit(1)

    _print("\nStarting training...")
    trainer.train()
