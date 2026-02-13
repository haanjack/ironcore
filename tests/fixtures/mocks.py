# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Mock classes for testing."""

from __future__ import annotations

import random
from typing import Any

import torch
from torch.utils.data import Dataset


class MockTokenizer:
    """Mock tokenizer for testing without loading real tokenizers."""

    def __init__(
        self,
        vocab_size: int = 1000,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ):
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self._token_to_id = {f"token_{i}": i for i in range(vocab_size)}
        self._id_to_token = {i: f"token_{i}" for i in range(vocab_size)}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Mock encode - returns deterministic tokens based on text hash."""
        tokens = [hash(text) % (self.vocab_size - 3) + 3 for _ in range(len(text) // 4 + 1)]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens[: min(len(tokens), 512)]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Mock decode - returns placeholder text."""
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in {self.bos_token_id, self.eos_token_id}]
        return " ".join(self._id_to_token.get(t, "<unk>") for t in token_ids)

    def __call__(self, text: str, **kwargs) -> dict[str, Any]:
        """Make tokenizer callable like HuggingFace tokenizers."""
        ids = self.encode(text, add_special_tokens=kwargs.get("add_special_tokens", True))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class MockDataset(Dataset):
    """Mock dataset for testing without loading real data."""

    def __init__(
        self,
        num_samples: int = 100,
        seq_len: int = 128,
        vocab_size: int = 1000,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed
        random.seed(seed)
        self._data = [
            {
                "input_ids": [random.randint(0, vocab_size - 1) for _ in range(seq_len)],
                "labels": [random.randint(0, vocab_size - 1) for _ in range(seq_len)],
            }
            for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


class MockRandom:
    """Mock random for reproducible testing."""

    def __init__(self, return_value: float = 0.5):
        self.return_value = return_value
        self.calls = []

    def random(self) -> float:
        self.calls.append("random")
        return self.return_value

    def randint(self, a: int, b: int) -> int:
        self.calls.append(("randint", a, b))
        return a

    def choice(self, seq: list) -> Any:
        self.calls.append(("choice", seq))
        return seq[0] if seq else None

    def uniform(self, a: float, b: float) -> float:
        self.calls.append(("uniform", a, b))
        return (a + b) / 2


class MockDistributed:
    """Mock distributed environment for single-GPU testing."""

    def __init__(self, world_size: int = 1, rank: int = 0):
        self.world_size = world_size
        self.rank = rank

    def get_rank(self) -> int:
        return self.rank

    def get_world_size(self) -> int:
        return self.world_size

    def is_initialized(self) -> bool:
        return True

    def all_reduce(self, tensor: torch.Tensor, op=None) -> torch.Tensor:
        """Mock all_reduce - just returns the tensor unchanged."""
        return tensor

    def all_gather(self, tensor_list: list, tensor: torch.Tensor) -> None:
        """Mock all_gather - fills all positions with the same tensor."""
        for i in range(len(tensor_list)):
            tensor_list[i] = tensor.clone()

    def barrier(self) -> None:
        """Mock barrier - no-op."""
        pass
