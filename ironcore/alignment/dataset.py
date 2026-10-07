# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO Dataset for alignment training.

Supports multiple formats:
- Verifiable tasks (math, code) with ground truth
- Reward model tasks without ground truth
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader

from ironcore import get_tokenizer

if TYPE_CHECKING:
    pass


@dataclass
class GRPOSample:
    """Single sample for GRPO training."""

    prompt: str
    input_ids: torch.Tensor  # Tokenized prompt
    attention_mask: torch.Tensor
    metadata: dict  # Contains answer/test_cases/type/etc.


class GRPODataset(IterableDataset):
    """Dataset for GRPO training supporting multiple formats.

    Handles:
    - Verifiable tasks (math, code) with ground truth
    - Reward model tasks without ground truth

    Expected JSON/JSONL formats:

    Math:
        {"prompt": "Solve: 2x + 3 = 7", "answer": "x = 2", "type": "math"}

    Code:
        {"prompt": "def fibonacci(n):\\n    ", "test_cases": ["assert fib(5)==5"], "type": "code"}

    Reward model (no ground truth):
        {"prompt": "Write a haiku about coding"}

    Arbitrary metadata:
        {"prompt": "...", "difficulty": "hard", "category": "algebra", ...}
    """

    def __init__(
        self,
        data_path: str | Path,
        max_prompt_length: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        tokenizer=None,
        prompt_column: str = "prompt",
        answer_column: str = "answer",
    ):
        self.data_path = Path(data_path)
        self.max_prompt_length = max_prompt_length
        self.shuffle = shuffle
        self.seed = seed
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
        self.prompt_column = prompt_column
        self.answer_column = answer_column

        self.samples = self._load_data()
        if not self.samples:
            raise ValueError("GRPO dataset contains no samples")
        self.signature = hashlib.sha256(self.data_path.read_bytes()).hexdigest()
        self.cursor = {"epoch": 0, "offset": 0}
        self.rank, self.world_size = 0, 1
        if torch.distributed.is_initialized():
            from ironcore.parallel import parallel_states

            self.rank = parallel_states.get_data_parallel_group_rank()
            self.world_size = parallel_states.get_data_parallel_world_size()

    def state_dict(self):
        return {
            "signature": self.signature,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "max_prompt_length": self.max_prompt_length,
            "rank": self.rank,
            "world_size": self.world_size,
            "cursor": self.cursor.copy(),
        }

    def load_state_dict(self, state):
        for key in ("signature", "seed", "shuffle", "max_prompt_length", "rank", "world_size"):
            if state[key] != getattr(self, key):
                raise ValueError(
                    "GRPO data resume requires the same data and sampling configuration"
                )
        self.cursor = state["cursor"].copy()

    def _load_data(self) -> list[dict]:
        """Load data from file."""
        samples = []

        if self.data_path.suffix == ".jsonl":
            with open(self.data_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        elif self.data_path.suffix == ".json":
            with open(self.data_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = [data]
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        return samples

    def __iter__(self) -> Iterator[GRPOSample]:
        """Iterate over samples infinitely (cycles through dataset)."""
        import math

        world_size = self.world_size
        rank = self.rank

        worker = get_worker_info()
        if worker is not None:
            rank = rank * worker.num_workers + worker.id
            world_size *= worker.num_workers
        indices = list(range(len(self.samples)))

        # Cycle infinitely through samples
        while True:
            epoch = self.cursor["epoch"]
            if self.shuffle:
                # Re-shuffle each epoch
                rng = random.Random(self.seed + epoch)
                shuffled_indices = indices.copy()
                rng.shuffle(shuffled_indices)
            else:
                shuffled_indices = indices.copy()

            # Shard indices for distributed training
            # We pad to make sure all ranks get the exact same number of samples per epoch
            # This ensures that distributed batches remain aligned across ranks.
            if world_size > 1:
                num_samples = len(shuffled_indices)
                num_samples_per_rank = math.ceil(num_samples / world_size)
                total_size = num_samples_per_rank * world_size
                padding = [
                    shuffled_indices[i % num_samples] for i in range(total_size - num_samples)
                ]
                shuffled_indices += padding
                sharded_indices = shuffled_indices[rank:total_size:world_size]
            else:
                sharded_indices = shuffled_indices

            for offset in range(self.cursor["offset"], len(sharded_indices)):
                idx = sharded_indices[offset]
                sample = self.samples[idx]

                # Get prompt using configurable column name with fallback
                prompt = sample.get(self.prompt_column, sample.get("prompt", ""))
                encoded = self.tokenizer(
                    prompt,
                    max_length=self.max_prompt_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )

                # Build metadata (include all fields from sample)
                metadata = {
                    "type": sample.get("type", "unknown"),
                    "original_prompt": prompt,
                }
                # Copy all other fields except the prompt column itself
                for key, value in sample.items():
                    if key not in (self.prompt_column, "prompt", "input_ids", "attention_mask"):
                        metadata[key] = value

                self.cursor = {"epoch": epoch, "offset": offset + 1}
                yield GRPOSample(
                    prompt=prompt,
                    input_ids=encoded["input_ids"].squeeze(0),
                    attention_mask=encoded["attention_mask"].squeeze(0),
                    metadata=metadata,
                )
            self.cursor = {"epoch": epoch + 1, "offset": 0}

    def __len__(self) -> int:
        return len(self.samples)


def collate_grpo_samples(samples: list[GRPOSample]) -> dict:
    """Collate function for GRPO samples with dynamic padding.

    Args:
        samples: List of GRPOSample objects

    Returns:
        Dictionary with:
        - input_ids: [B, max_len] padded tensors
        - attention_mask: [B, max_len] padded masks
        - prompts: list[str] raw prompt texts
        - metadata: list[dict]
    """
    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence(
        [s.input_ids for s in samples],
        batch_first=True,
        padding_value=0,  # Standard pad value, tokenizer specific check recommended
    )
    attention_mask = pad_sequence(
        [s.attention_mask for s in samples],
        batch_first=True,
        padding_value=0,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": [s.prompt for s in samples],
        "metadata": [s.metadata for s in samples],
    }


def get_grpo_dataloader(
    data_path: str | Path,
    batch_size: int,
    max_prompt_length: int = 1024,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    prompt_column: str = "prompt",
    answer_column: str = "answer",
) -> StatefulDataLoader:
    """Create DataLoader for GRPO training.

    Args:
        data_path: Path to data file (JSON or JSONL)
        batch_size: Batch size
        max_prompt_length: Maximum prompt length
        shuffle: Whether to shuffle
        seed: Random seed
        num_workers: Number of data loader workers
        prompt_column: Column name for prompts in data
        answer_column: Column name for answers in data

    Returns:
        DataLoader
    """
    dataset = GRPODataset(
        data_path=data_path,
        max_prompt_length=max_prompt_length,
        shuffle=shuffle,
        seed=seed,
        prompt_column=prompt_column,
        answer_column=answer_column,
    )

    return StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_grpo_samples,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers else None,
        snapshot_every_n_steps=1,
        generator=torch.Generator().manual_seed(seed),
    )


def get_grpo_data_iterator(
    config,
    split: str = "train",
) -> Iterator[dict]:
    """Create data iterator for GRPO training from config.

    Args:
        config: MainConfig object
        split: "train" or "eval"

    Returns:
        Iterator over batches
    """
    from pathlib import Path

    from ironcore.config import _validate_path_within_dir
    from ironcore.dataloader.data_config import DataConfig
    from ironcore.parallel import parallel_states

    # Allowed base directory for data config files
    _DATA_CONFIG_BASE_DIR = Path("configs/data").resolve()

    # Load data configuration (same logic as get_data_iterator)
    if hasattr(config.data, "config_path") and config.data.config_path:
        config_path = Path(config.data.config_path)
        if not config_path.is_absolute():
            config_path = config_path.resolve()
        if not config_path.exists():
            fallback = _DATA_CONFIG_BASE_DIR / f"{config_path.name}.yaml"
            if _validate_path_within_dir(fallback, _DATA_CONFIG_BASE_DIR) and fallback.exists():
                config_path = fallback
        if not _validate_path_within_dir(config_path, _DATA_CONFIG_BASE_DIR):
            raise ValueError(
                f"Config path '{config.data.config_path}' is outside allowed directory 'configs/data/'"
            )
        data_config = DataConfig.from_yaml(config_path)
    elif hasattr(config.data, "datasets") and len(config.data.datasets) > 0:
        data_config = config.data
    elif isinstance(config.data, str):
        data_config = DataConfig.from_yaml(_DATA_CONFIG_BASE_DIR / f"{config.data}.yaml")
    elif hasattr(config.data, "seq_length"):
        data_config = config.data
    else:
        raise ValueError(f"Cannot load data config from: {config.data}")

    # Get data path from loaded data_config
    if split == "train":
        datasets = data_config.datasets
    elif split == "eval":
        datasets = data_config.eval_datasets if data_config.eval_datasets else data_config.datasets
    elif split == "test":
        datasets = data_config.test_datasets if data_config.test_datasets else data_config.datasets
    else:
        raise ValueError(f"Invalid split: {split}")

    if not datasets:
        raise ValueError(f"No {split} datasets found in config")

    # Get source from first dataset
    data_path = datasets[0].source

    # Get field mapping from config
    prompt_column = getattr(datasets[0], "prompt_column", "prompt")
    answer_column = getattr(datasets[0], "answer_column", "answer")

    dataloader = get_grpo_dataloader(
        data_path=data_path,
        batch_size=(
            config.trainer.train_batch_size // parallel_states.get_data_parallel_world_size()
            if split == "train"
            else config.trainer.eval_batch_size
        ),
        max_prompt_length=(
            config.model.max_position_embeddings - config.alignment.generation.max_new_tokens
        ),
        shuffle=(split == "train"),
        seed=config.init.seed,
        num_workers=config.trainer.num_workers,
        prompt_column=prompt_column,
        answer_column=answer_column,
    )

    from ironcore.dataloader.stateful import CheckpointableIterator

    return CheckpointableIterator(dataloader)
