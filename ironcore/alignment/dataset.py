# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO Dataset for alignment training.

Supports multiple formats:
- Verifiable tasks (math, code) with ground truth
- Reward model tasks without ground truth
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, IterableDataset

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
    ):
        self.data_path = Path(data_path)
        self.max_prompt_length = max_prompt_length
        self.shuffle = shuffle
        self.seed = seed
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()

        self.samples = self._load_data()

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
        """Iterate over samples."""
        indices = list(range(len(self.samples)))

        if self.shuffle:
            random.Random(self.seed).shuffle(indices)

        for idx in indices:
            sample = self.samples[idx]

            # Tokenize prompt
            prompt = sample["prompt"]
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
            # Copy all other fields
            for key, value in sample.items():
                if key not in ("prompt", "input_ids", "attention_mask"):
                    metadata[key] = value

            yield GRPOSample(
                prompt=prompt,
                input_ids=encoded["input_ids"].squeeze(0),
                attention_mask=encoded["attention_mask"].squeeze(0),
                metadata=metadata,
            )

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
        "metadata": [s.metadata for s in samples],
    }


def get_grpo_dataloader(
    data_path: str | Path,
    batch_size: int,
    max_prompt_length: int = 1024,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader for GRPO training.

    Args:
        data_path: Path to data file (JSON or JSONL)
        batch_size: Batch size
        max_prompt_length: Maximum prompt length
        shuffle: Whether to shuffle
        seed: Random seed
        num_workers: Number of data loader workers

    Returns:
        DataLoader
    """
    dataset = GRPODataset(
        data_path=data_path,
        max_prompt_length=max_prompt_length,
        shuffle=shuffle,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_grpo_samples,
        num_workers=num_workers,
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
    data_config = config.data

    data_path = data_config.train_file if split == "train" else data_config.eval_file

    dataloader = get_grpo_dataloader(
        data_path=data_path,
        batch_size=data_config.train_batch_size,
        max_prompt_length=config.model.max_position_embeddings,
        shuffle=(split == "train"),
        seed=config.init.seed,
        num_workers=getattr(data_config, "num_workers", 0),
    )

    return iter(dataloader)
