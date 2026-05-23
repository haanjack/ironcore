# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Random token dataset for testing — no preprocessing required."""

from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset


class RandomTokenDataset(IterableDataset):
    """Generates random token IDs on the fly.

    Produces batches in the same format as the streaming dataloader:
    ``{"input_ids": Tensor[B, S], "labels": Tensor[B, S]}``.

    Args:
        seq_length: Sequence length (max_seq_len from model config).
        vocab_size: Vocabulary size.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        seq_length: int = 1024,
        vocab_size: int = 50304,
        seed: int = 42,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        rank = 0
        world_size = 1
        if torch.distributed.is_initialized():
            try:
                from ironcore.parallel import parallel_states

                rank = parallel_states.get_data_parallel_group_rank()
                world_size = parallel_states.get_data_parallel_world_size()
            except (AssertionError, AttributeError):
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()

        if worker_info is not None:
            rank = rank * worker_info.num_workers + worker_info.id
            world_size = world_size * worker_info.num_workers

        rng = torch.Generator()
        rng.manual_seed(self.seed + rank)

        while True:
            input_ids = torch.randint(0, self.vocab_size, (self.seq_length,), generator=rng)
            labels = input_ids.clone()
            yield {"input_ids": input_ids, "labels": labels}


def get_random_data_iterator(
    seq_length: int = 1024,
    vocab_size: int = 50304,
    batch_size: int = 4,
    seed: int = 42,
) -> dict[str, Iterator[dict[str, torch.Tensor]]]:
    """Create random data iterators for train/eval/test splits.

    Returns the same dict format as ``get_data_iterator``: ``{"train": ..., "eval": ..., "test": ...}``.
    """
    iterators = {}
    for split in ("train", "eval", "test"):
        dataset = RandomTokenDataset(
            seq_length=seq_length,
            vocab_size=vocab_size,
            seed=seed + hash(split) % (2**31),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
        )
        iterators[split] = iter(dataloader)
    return iterators
