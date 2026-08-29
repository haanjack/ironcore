# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path
from typing import Union

from ironcore import get_logger, get_tokenizer
from ironcore.eval.tasks.base_task import Task


def get_evaluators(
    dataset_configs: list[dict[str, Union[str, float, int]]],
    batch_size: int = 1,
    num_samples: int = None,
    cache_dir: Union[str, Path] = None,
) -> list[Task]:
    """
    build evaluator according to config
    """
    if dataset_configs is None:
        return []
    # dataset_configs: [{dataset_path: ..., ratio: ...}, ]

    logger = get_logger()
    tokenizer = get_tokenizer()

    logger.info("Loading evaluation tasks")
    for dataset_config in dataset_configs:
        task_name = dataset_config["name"]
        logger.info(f" - {task_name}")

    if cache_dir is None:
        cache_dir = Path("data/evaluation_cache")

    evaluators = []
    for dataset_config in dataset_configs:
        task_name = dataset_config["name"].lower()
        num_samples = dataset_config.get("samples", None)

        module_name = f"ironcore.eval.tasks.{task_name}"
        try:
            module = importlib.import_module(module_name)
            evaluator_class = next(
                cls
                for cls in vars(module).values()
                if isinstance(cls, type) and issubclass(cls, Task) and cls != Task
            )
        except (ImportError, StopIteration) as e:
            raise ValueError(
                f"Evaluator for task '{task_name}' could not be loaded from "
                f"'{module_name}' (a Task subclass must be defined there). "
                f"Check for a typo in the task name."
            ) from e
        evaluators.append(evaluator_class(tokenizer, batch_size, num_samples, cache_dir=cache_dir))

    return evaluators
