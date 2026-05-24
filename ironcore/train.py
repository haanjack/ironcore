# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Training entrypoint — usable as Python API or via torchrun."""

import logging
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

logger = logging.getLogger(__name__)


def load_full_config(
    config_path: str | Path,
    overrides: dict | None = None,
) -> MainConfig:
    """Load a fully resolved MainConfig from a YAML file.

    Args:
        config_path: Path to training config YAML.
        overrides: Optional dict of config overrides, e.g. {"model.num_layers": 12}.

    Returns:
        Fully resolved and validated MainConfig.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

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

    args = Namespace(config_path=str(config_path))
    _load_config_from_yaml(config, args)

    if overrides:
        for key, value in overrides.items():
            try:
                _set_nested_attr(config, key, value)
            except AttributeError:
                raise ValueError(f"Invalid config override: '{key}' is not a valid config path") from None

    config.parallel.rank = int(os.getenv("RANK", "0"))
    config.parallel.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    config.parallel.world_size = int(os.getenv("WORLD_SIZE", "1"))

    _config_validation(config)
    return config


def _set_nested_attr(obj, path: str, value) -> None:
    """Set a nested attribute using dot notation, e.g. 'model.num_layers'."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    attr_name = parts[-1]
    current = getattr(obj, attr_name)
    if isinstance(current, bool) and isinstance(value, str):
        value = value.lower() in ("true", "yes", "1")
    elif not isinstance(value, type(current)) and current is not None:
        try:
            value = type(current)(value)
        except (ValueError, TypeError):
            pass
    setattr(obj, attr_name, value)


def train(config: MainConfig) -> None:
    """Run training with the given configuration.

    Selects the appropriate trainer based on config.data.task_type.
    """
    task_type = config.data.task_type
    loss_fn = get_loss_func(task_type)
    logger.info("Task type: %s, loss function: %s", task_type, loss_fn.__name__)

    if task_type in ("pretrain", "sft"):
        trainer = LanguageModelTrainer(config, forward_step_func=forward_step, loss_fn=loss_fn)
    elif task_type == "dpo":
        if config.alignment is None or config.alignment == AlignmentConfig():
            raise ValueError(
                "DPO requires 'alignment' configuration section. "
                "Define alignment hyperparameters (e.g., beta, label_smoothing)."
            )
        if config.alignment.dpo_beta <= 0:
            raise ValueError(
                f"alignment.dpo_beta must be positive, got {config.alignment.dpo_beta}"
            )
        trainer = DPOTrainer(config, forward_step_func=forward_step, loss_fn=loss_fn)
    elif task_type == "grpo":
        if config.alignment is None or config.alignment == AlignmentConfig():
            raise ValueError(
                "GRPO requires 'alignment' configuration section. "
                "Define alignment hyperparameters (e.g., grpo_beta, grpo_group_size)."
            )
        if config.alignment.grpo_beta <= 0:
            raise ValueError(
                f"alignment.grpo_beta must be positive, got {config.alignment.grpo_beta}"
            )
        if config.alignment.grpo_group_size <= 0:
            raise ValueError(
                f"alignment.grpo_group_size must be positive, got {config.alignment.grpo_group_size}"
            )
        trainer = GRPOTrainer(config, forward_step_func=forward_step, loss_fn=loss_fn)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    trainer.train()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(prog="ironcore.train", description="IronCore Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")
    cli_args = parser.parse_args()

    override_dict = {}
    for item in cli_args.overrides:
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            override_dict[k] = v

    try:
        config = load_full_config(cli_args.config, overrides=override_dict or None)
        train(config)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
