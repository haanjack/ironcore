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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from datasets import load_dataset
from torch.utils.data import DataLoader

from ironcore import get_logger


class Task(ABC):
    def __init__(
        self,
        task_name: str,
        split_name: str,
        tokenizer,
        batch_size: int = 1,
        num_samples: int = None,
        cache_dir: Union[str, Path] = None,
    ):
        """
        initilize evaluator
        """
        self._task_name = task_name
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.cache_dir = cache_dir

        self.logger = get_logger()

        self._get_dataloader()

    @property
    def task_name(self):
        return self._task_name

    @abstractmethod
    def process(self, model):
        raise NotImplementedError("")

    @abstractmethod
    def _preprocess(self):
        raise NotImplementedError("")

    def _get_dataloader(self):
        split = f"{self.split_name}[:{self.num_samples}]" if self.num_samples else self.split_name
        dataset = load_dataset(
            self.task_name,
            split=split,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        if self.num_samples is None:
            self.num_samples = len(dataset)
        preprocessed_dataset = dataset.map(
            self._preprocess,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Preprocessing evaluation task: {self.task_name}",
        )
        self.data_loader = DataLoader(preprocessed_dataset, batch_size=self.batch_size)

    @abstractmethod
    def _get_batch(self):
        raise NotImplementedError("")

    def _do_predict(self):
        raise NotImplementedError("")

    @abstractmethod
    def _get_score(self):
        raise NotImplementedError("")
