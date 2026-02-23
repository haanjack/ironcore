# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from pathlib import Path
from typing import Union

from ironcore import get_logger, get_tokenizer
from ironcore.eval.tasks.base_task import Task
from ironcore.utils import get_dataset_base_dir


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
        cache_dir = get_dataset_base_dir() / "evaluation_cache"

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
            evaluators.append(
                evaluator_class(tokenizer, batch_size, num_samples, cache_dir=cache_dir)
            )
        except (ImportError, StopIteration):
            print(f"Evaluator for task '{task_name}' could not be loaded.")

    return evaluators
