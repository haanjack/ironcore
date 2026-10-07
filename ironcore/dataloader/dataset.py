# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
True streaming dataset implementation for IronCore.

Key improvements over universal_dataset.py:
1. O(1) memory usage - no loading of full indices/positions
2. Lazy metadata loading using memory-mapped .idx files
3. Infinite iteration support for pretraining
4. Deterministic shuffling with block-based approach
"""

import copy
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import IterableDataset

from ironcore.dataloader.data_config import DataConfig
from ironcore.parallel import parallel_states


class StreamingBinaryDataset:
    """
    Memory-efficient wrapper for .bin/.idx datasets.

    Uses memory-mapped files for both data and metadata.
    """

    def __init__(self, bin_path: Path, idx_path: Path):
        """
        Initialize dataset with memory-mapped files.

        Args:
            bin_path: Path to .bin file (token data)
            idx_path: Path to .idx file (metadata)
        """
        self.bin_path = bin_path
        self.idx_path = idx_path

        # Memory-map metadata (don't load into RAM)
        self.metadata = np.load(str(idx_path), allow_pickle=False, mmap_mode="r")

        # Determine dtype from file size
        file_size = bin_path.stat().st_size
        total_tokens = int(self.metadata["offset"][-1]) + int(self.metadata["length"][-1])

        if file_size // total_tokens == 2:
            dtype = np.uint16
        elif file_size // total_tokens == 4:
            dtype = np.uint32
        else:
            raise ValueError(
                f"Unsupported bytes per token: {file_size // total_tokens}. Expected 2 or 4."
            )

        # Memory-map token data
        self.data = np.memmap(str(bin_path), dtype=dtype, mode="r")

    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            Dict with keys:
                - token_ids: np.ndarray of token IDs
                - metadata: Dict of metadata fields
        """
        meta = self.metadata[idx]

        offset = int(meta["offset"])
        length = int(meta["length"])

        token_ids = self.data[offset : offset + length]

        return {
            "token_ids": token_ids,
            "metadata": {
                "type": str(meta["type"]),
                "group_id": int(meta["group_id"]),
                "mask_ranges": json.loads(str(meta["mask_ranges"])) if meta["mask_ranges"] else [],
            },
        }

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in dataset."""
        return len(self.data)

    def __getstate__(self):
        return {"bin_path": self.bin_path, "idx_path": self.idx_path}

    def __setstate__(self, state):
        self.__init__(state["bin_path"], state["idx_path"])


class StreamingDataset(IterableDataset):
    """
    True streaming dataset with O(1) memory usage.

    Supports two modes:
        - pretrain: Infinite streaming with block-based shuffling
        - sft: Epoch-based streaming with lazy sampling
    """

    def __init__(
        self,
        data_config: DataConfig,
        mode: Literal["pretrain", "sft", "dpo"] = "pretrain",
        seed: int = 1337,
        split: str = "train",
    ):
        """
        Initialize streaming dataset.

        Shuffle buffer size is automatically tuned based on dataset size:
        - Uses 1% of dataset size for good shuffle quality
        - Capped at [1K, 100K] to balance randomness and memory

        Args:
            data_config: Data configuration
            mode: Training mode (pretrain/sft/dpo)
            seed: Random seed for reproducibility
            split: Data split (train/eval/test)
        """
        super().__init__()

        self.config = data_config
        self.mode = mode
        self.seed = seed
        self.split = split
        self.seq_length = data_config.seq_length

        # Select datasets based on split
        source_datasets = []
        self.is_separate_split = False

        if split == "train":
            source_datasets = data_config.datasets
        elif split == "eval":
            if data_config.eval_datasets:
                source_datasets = data_config.eval_datasets
                self.is_separate_split = True
            else:
                source_datasets = data_config.datasets
        elif split == "test":
            if data_config.test_datasets:
                source_datasets = data_config.test_datasets
                self.is_separate_split = True
            else:
                source_datasets = data_config.datasets
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load datasets with memory-mapped files
        self.datasets: list[StreamingBinaryDataset] = []
        self.weights: list[float] = []

        for ds_config in source_datasets:
            # Filter by task type
            if mode == "pretrain" and ds_config.task_type != "pretrain":
                continue
            if mode == "sft" and ds_config.task_type != "sft":
                continue
            if mode == "dpo" and ds_config.task_type != "dpo":
                continue

            # Load dataset
            output_path = data_config.get_dataset_output_path(ds_config)
            bin_path = output_path / "data.bin"
            idx_path = output_path / "data.idx.npy"

            if not bin_path.exists() or not idx_path.exists():
                raise FileNotFoundError(
                    f"Dataset {ds_config.name} not preprocessed. Run: python -m ironcore prepare"
                )

            dataset = StreamingBinaryDataset(bin_path, idx_path)
            self.datasets.append(dataset)
            self.weights.append(ds_config.ratio)

        if not self.datasets:
            raise ValueError(f"No datasets found for mode={mode}, split={split}")

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        # Compute split ranges
        self._compute_split_ranges()

        # Auto-tune shuffle buffer size based on dataset size
        self._auto_tune_shuffle_buffer()

        # Multi-GPU support: deterministic sharding
        if dist.is_initialized():
            try:
                self.rank = parallel_states.get_data_parallel_group_rank()
                self.world_size = parallel_states.get_data_parallel_world_size()
            except (AssertionError, AttributeError):
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def _compute_split_ranges(self):
        """Compute start/end indices for train/eval/test splits."""
        split_ratios = {
            "train": self.config.splits[0],
            "eval": self.config.splits[1],
            "test": self.config.splits[2],
        }

        self.split_ranges = {}

        if self.mode == "pretrain":
            # For pretrain, split based on total tokens
            for dataset in self.datasets:
                total_tokens = dataset.total_tokens

                if self.is_separate_split:
                    start, end = 0, total_tokens
                else:
                    train_end = int(total_tokens * split_ratios["train"])
                    eval_end = train_end + int(total_tokens * split_ratios["eval"])

                    if self.split == "train":
                        start, end = 0, train_end
                    elif self.split == "eval":
                        start, end = train_end, eval_end
                    elif self.split == "test":
                        start, end = eval_end, total_tokens
                    else:
                        raise ValueError(f"Invalid split: {self.split}")

                self.split_ranges[id(dataset)] = (start, end)
        else:
            # For SFT/DPO, split based on number of samples
            for dataset in self.datasets:
                total_samples = len(dataset)

                if self.is_separate_split:
                    start, end = 0, total_samples
                else:
                    train_end = int(total_samples * split_ratios["train"])
                    eval_end = train_end + int(total_samples * split_ratios["eval"])

                    if self.split == "train":
                        start, end = 0, train_end
                    elif self.split == "eval":
                        start, end = train_end, eval_end
                    elif self.split == "test":
                        start, end = eval_end, total_samples
                    else:
                        raise ValueError(f"Invalid split: {self.split}")

                self.split_ranges[id(dataset)] = (start, end)

    def _auto_tune_shuffle_buffer(self):
        """
        Auto-tune shuffle buffer size based on dataset size.

        Strategy:
        - Use 1% of dataset size for good shuffle quality
        - Cap between 1K (minimum) and 100K (maximum for memory efficiency)
        - Smaller datasets get near-perfect shuffle
        - Larger datasets get good-enough shuffle without memory issues

        This eliminates the need for manual tuning while providing:
        - 10K samples → buffer=1K (10% shuffle quality)
        - 100K samples → buffer=1K (1% shuffle quality, still excellent)
        - 1M samples → buffer=10K (1% shuffle quality)
        - 10M+ samples → buffer=100K (capped at 1% or 100K, whichever is smaller)
        """
        if self.mode == "pretrain":
            # Calculate total positions
            total_tokens = sum(
                end - start
                for dataset in self.datasets
                for start, end in [self.split_ranges[id(dataset)]]
            )
            num_positions = total_tokens // self.seq_length

            # Auto-tune: 1% of dataset, capped at [1K, 100K]
            self.shuffle_buffer_size = max(1000, min(100000, num_positions // 100))
        else:
            # For SFT/DPO: based on number of samples
            total_samples = sum(
                end - start
                for dataset in self.datasets
                for start, end in [self.split_ranges[id(dataset)]]
            )

            # Auto-tune: 1% of dataset, capped at [1K, 50K]
            # (SFT samples are more memory-intensive, so lower cap)
            self.shuffle_buffer_size = max(1000, min(50000, total_samples // 100))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_serialized_ranges"] = [self.split_ranges[id(d)] for d in self.datasets]
        state.pop("split_ranges")
        return state

    def __setstate__(self, state):
        ranges = state.pop("_serialized_ranges")
        self.__dict__.update(state)
        self.split_ranges = {id(d): r for d, r in zip(self.datasets, ranges, strict=True)}

    def _signature(self):
        return {
            "mode": self.mode,
            "split": self.split,
            "seed": self.seed,
            "seq_length": self.seq_length,
            "rank": self.rank,
            "world_size": self.world_size,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "weights": self.weights,
            "files": [
                (
                    str(d.bin_path.resolve()),
                    d.bin_path.stat().st_size,
                    d.bin_path.stat().st_mtime_ns,
                    str(d.idx_path.resolve()),
                    d.idx_path.stat().st_size,
                    d.idx_path.stat().st_mtime_ns,
                )
                for d in self.datasets
            ],
        }

    def state_dict(self):
        return {
            "version": 1,
            "signature": self._signature(),
            "cursor": copy.deepcopy(getattr(self, "_stream_state", None)),
        }

    def load_state_dict(self, state):
        if state["version"] != 1 or state["signature"] != self._signature():
            raise ValueError("Dataset cursor requires the same files, seed, split and DP topology")
        self._restore_state = copy.deepcopy(state["cursor"])

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        rank = self.rank * workers + worker_id
        world = self.world_size * workers
        state = getattr(self, "_restore_state", None)
        self._restore_state = None
        if self.mode == "pretrain":
            return self._iter_pretrain_streaming(state, rank, world)
        return self._iter_sft_streaming(state, rank, world)

    def _iter_pretrain_streaming(self, state=None, rank=None, world=None):
        rank = self.rank if rank is None else rank
        world = self.world_size if world is None else world
        ranges = [(i, *self.split_ranges[id(d)]) for i, d in enumerate(self.datasets)]
        positions = sum(end - start for _, start, end in ranges) // self.seq_length
        if positions == 0:
            raise ValueError("Pretraining split has no complete context window")
        cursor = (
            copy.deepcopy(state) if state is not None else {"epoch": 0, "block": 0, "offset": 0}
        )
        while True:
            epoch, block, offset = cursor["epoch"], cursor["block"], cursor["offset"]
            # Block-local randomness permits direct seeking without replaying I/O.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, block]))
            permutation = np.arange(block, min(block + self.shuffle_buffer_size, positions))
            rng.shuffle(permutation)
            local = [
                int(pos) * self.seq_length
                for i, pos in enumerate(permutation)
                if (block + i) % world == rank
            ]
            for i in range(offset, len(local)):
                cursor = {"epoch": epoch, "block": block, "offset": i + 1}
                self._stream_state = cursor
                global_pos = local[i]
                base = 0
                for ds_idx, start, end in ranges:
                    length = end - start
                    if global_pos < base + length:
                        local_pos = start + global_pos - base
                        data = self.datasets[ds_idx].data
                        tokens = data[local_pos : min(local_pos + self.seq_length + 1, end)]
                        if len(tokens) < self.seq_length + 1:
                            # repeat from this dataset's split without train/eval leakage
                            needed = self.seq_length + 1 - len(tokens)
                            tokens = np.concatenate([tokens, np.resize(data[start:end], needed)])
                        yield torch.from_numpy(tokens.astype(np.int64))
                        break
                    base += length
            block += self.shuffle_buffer_size
            cursor = {
                "epoch": epoch + int(block >= positions),
                "block": 0 if block >= positions else block,
                "offset": 0,
            }
            self._stream_state = cursor

    def _iter_sft_streaming(self, state=None, rank=None, world=None):
        rank = self.rank if rank is None else rank
        world = self.world_size if world is None else world
        info = [(i, *self.split_ranges[id(d)]) for i, d in enumerate(self.datasets)]
        counts = [end - start for _, start, end in info]
        total = sum(counts)
        rng = np.random.default_rng(self.seed)
        weighted = np.array(
            [n * w for n, w in zip(counts, self.weights, strict=True)], dtype=np.float64
        )
        indices = [0] * len(info)
        buffers = [[] for _ in info]  # indices only, no token/metadata materialization
        begin = 0
        if state is not None:
            rng.bit_generator.state = state["rng"]
            weighted = np.array(state["weighted"], dtype=np.float64)
            indices, buffers, begin = (
                copy.deepcopy(state["indices"]),
                copy.deepcopy(state["buffers"]),
                state["next"],
            )
        size = max(1, self.shuffle_buffer_size // len(info))
        for global_idx in range(begin, total):
            if weighted.sum() <= 0:
                break
            selected = int(rng.choice(len(info), p=weighted / weighted.sum()))
            buffer = buffers[selected]
            while len(buffer) < size and indices[selected] < counts[selected]:
                buffer.append(indices[selected])
                indices[selected] += 1
            pick = int(rng.integers(len(buffer)))
            sample_idx = buffer[pick]
            if indices[selected] < counts[selected]:
                buffer[pick] = indices[selected]
                indices[selected] += 1
            else:
                buffer[pick] = buffer[-1]
                buffer.pop()
            if indices[selected] >= counts[selected] and not buffer:
                weighted[selected] = 0
            self._stream_state = {
                "next": global_idx + 1,
                "rng": rng.bit_generator.state,
                "weighted": weighted.tolist(),
                "indices": indices,
                "buffers": buffers,
            }
            # All ranks advance the same global permutation before sharding.
            # Previously sharding before RNG advancement duplicated DP samples.
            if global_idx % world != rank:
                continue
            ds_idx, start, _ = info[selected]
            sample = self.datasets[ds_idx][start + sample_idx]
            yield {
                "token_ids": torch.from_numpy(sample["token_ids"].astype(np.int64)),
                "metadata": sample["metadata"],
            }
