"""
Universal Dataset for Pretrain and SFT modes.

Implements pure streaming for pretrain and weighted mixing for all modes.
"""

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import IterableDataset

from ironcore.dataloader.data_config import DataConfig, DatasetConfig
from ironcore.parallel import parallel_states


class BinaryDataset:
    """
    Wrapper for a single .bin/.idx dataset.

    Provides access to tokens via memory-mapped file.
    """

    def __init__(self, bin_path: Path, idx_path: Path):
        """
        Initialize dataset.

        Args:
            bin_path: Path to .bin file (token data)
            idx_path: Path to .idx file (metadata)
        """
        self.bin_path = bin_path
        self.idx_path = idx_path

        # Load metadata
        self.metadata = np.load(idx_path, allow_pickle=False)

        # Memory-map token data
        # Determine dtype from file size
        file_size = bin_path.stat().st_size
        total_tokens = self.metadata['offset'][-1] + self.metadata['length'][-1]

        if file_size // total_tokens == 2:
            dtype = np.uint16
        elif file_size // total_tokens == 4:
            dtype = np.uint32
        else:
            # Fallback: try both
            dtype = np.uint16

        self.data = np.memmap(str(bin_path), dtype=dtype, mode='r')

    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            Dict with keys:
                - token_ids: np.ndarray of token IDs
                - metadata: Dict of metadata fields
        """
        meta = self.metadata[idx]

        offset = int(meta['offset'])
        length = int(meta['length'])

        token_ids = self.data[offset:offset + length]

        return {
            'token_ids': token_ids,
            'metadata': {
                'type': str(meta['type']),
                'group_id': int(meta['group_id']),
                'mask_ranges': json.loads(str(meta['mask_ranges'])) if meta['mask_ranges'] else []
            }
        }

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in dataset."""
        return len(self.data)


class WeightedMixingDataset(IterableDataset):
    """
    Universal dataset supporting weighted mixing of multiple datasets.

    Supports two modes:
        - pretrain: Pure streaming (slicing by max_seq_len)
        - sft: Per-sample loading for bin-packing
    """

    def __init__(
        self,
        data_config: DataConfig,
        mode: Literal["pretrain", "sft", "dpo"] = "pretrain",
        seed: int = 1337,
        split: str = "train",
    ):
        """
        Initialize dataset.

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
        self.max_seq_len = data_config.max_seq_len

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
                # Fallback to splitting main dataset
                source_datasets = data_config.datasets
        elif split == "test":
            if data_config.test_datasets:
                source_datasets = data_config.test_datasets
                self.is_separate_split = True
            else:
                # Fallback to splitting main dataset
                source_datasets = data_config.datasets
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load datasets
        self.datasets: List[BinaryDataset] = []
        self.weights: List[float] = []

        for ds_config in source_datasets:
            # Filter by task type (for mode compatibility)
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
                    f"Dataset {ds_config.name} not preprocessed. "
                    f"Run: python -m ironcore prepare"
                )

            dataset = BinaryDataset(bin_path, idx_path)
            self.datasets.append(dataset)
            self.weights.append(ds_config.ratio)

        if not self.datasets:
            raise ValueError(f"No datasets found for mode={mode}, split={split}")

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        # Compute split ranges for train/eval/test
        self._compute_split_ranges()

        # Multi-GPU support: deterministic sharding
        # IMPORTANT: Use data parallel ranks, not global ranks!
        # For tensor parallel training, all TP ranks must see the SAME data.
        # Only shard data across data parallel ranks.
        if dist.is_initialized():
            try:
                # Use data parallel ranks if model parallelism is initialized
                self.rank = parallel_states.get_data_parallel_group_rank()
                self.world_size = parallel_states.get_data_parallel_world_size()
            except (AssertionError, AttributeError):
                # Fallback: model parallelism not initialized, use global ranks
                # This happens in non-parallel training or pure data parallel training
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
            "test": self.config.splits[2]
        }

        self.split_ranges = {}

        if self.mode == "pretrain":
            # For pretrain, split based on total tokens
            for dataset in self.datasets:
                total_tokens = dataset.total_tokens

                if self.is_separate_split:
                    # Use full dataset for this split
                    start, end = 0, total_tokens
                else:
                    # Compute split boundaries based on ratios
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
                    # Use full dataset for this split
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

    def __iter__(self):
        """
        Iterate over dataset based on mode.

        For pretrain: Yields token slices of max_seq_len
        For SFT/DPO: Yields individual samples
        """
        if self.mode == "pretrain":
            return self._iter_pretrain()
        else:
            return self._iter_sft()

    def _iter_pretrain(self):
        """
        Pretrain mode: Pure streaming with deterministic sharding.

        Strategy:
            1. Concatenate all datasets (virtually)
            2. Generate global sequence of (dataset_idx, position) pairs
            3. Shard across ranks: index % world_size == rank
            4. Yield slices of max_seq_len + 1 tokens
        """
        rng = np.random.default_rng(seed=self.seed)

        # Build global token stream metadata
        # (dataset_idx, start_token, end_token)
        token_ranges = []
        for ds_idx, dataset in enumerate(self.datasets):
            start, end = self.split_ranges[id(dataset)]
            token_ranges.append((ds_idx, start, end))

        # Compute total tokens
        total_tokens = sum(end - start for _, start, end in token_ranges)

        # Generate positions to sample
        # Use stride = max_seq_len for non-overlapping slices
        positions = list(range(0, total_tokens, self.max_seq_len))

        # Shuffle positions with seed for reproducibility
        rng.shuffle(positions)

        # Shard positions across ranks (deterministic)
        # Only shard if we have multiple data parallel ranks (world_size > 1)
        # For pure TP training (world_size == 1), all ranks get all data
        if self.world_size > 1:
            rank_positions = [pos for i, pos in enumerate(positions) if i % self.world_size == self.rank]
        else:
            rank_positions = positions

        # Yield slices
        for global_pos in rank_positions:
            # Determine which dataset this position falls into
            current_offset = 0
            for ds_idx, start, end in token_ranges:
                ds_length = end - start
                if global_pos < current_offset + ds_length:
                    # This position is in this dataset
                    local_pos = global_pos - current_offset + start
                    dataset = self.datasets[ds_idx]

                    # Extract slice (max_seq_len + 1 for label shifting)
                    slice_end = min(local_pos + self.max_seq_len + 1, end)
                    token_ids = dataset.data[local_pos:slice_end]

                    # Handle wrap-around if slice is too short
                    if len(token_ids) < self.max_seq_len + 1:
                        # Wrap to beginning of dataset
                        needed = (self.max_seq_len + 1) - len(token_ids)
                        wrap_tokens = dataset.data[start:start + needed]
                        token_ids = np.concatenate([token_ids, wrap_tokens])

                    yield torch.from_numpy(token_ids.astype(np.int64))
                    break

                current_offset += ds_length

    def _iter_sft(self):
        """
        SFT/DPO mode: Weighted random sampling of individual samples.

        Strategy:
            1. Create pool of (dataset_idx, sample_idx) pairs
            2. Shuffle with seed
            3. Shard across ranks: index % world_size == rank
            4. Yield individual samples
        """
        rng = np.random.default_rng(seed=self.seed)

        # Build sample pool
        sample_pool = []
        for ds_idx, dataset in enumerate(self.datasets):
            start, end = self.split_ranges[id(dataset)]
            weight = self.weights[ds_idx]

            # Add samples with their weights
            for sample_idx in range(start, end):
                sample_pool.append((ds_idx, sample_idx, weight))

        # Weighted shuffle
        # Create sampling probabilities
        weights_array = np.array([w for _, _, w in sample_pool])
        weights_array /= weights_array.sum()

        # Sample indices
        num_samples = len(sample_pool)
        sampled_indices = rng.choice(
            num_samples,
            size=num_samples,
            replace=False,  # No replacement for one epoch
            p=weights_array
        )

        # Shard across ranks (deterministic)
        # Only shard if we have multiple data parallel ranks (world_size > 1)
        # For pure TP training (world_size == 1), all ranks get all data
        if self.world_size > 1:
            rank_indices = [idx for i, idx in enumerate(sampled_indices) if i % self.world_size == self.rank]
        else:
            rank_indices = sampled_indices

        # Yield samples
        for idx in rank_indices:
            ds_idx, sample_idx, _ = sample_pool[idx]
            dataset = self.datasets[ds_idx]

            sample = dataset[sample_idx]

            # Return sample with metadata
            yield {
                'token_ids': torch.from_numpy(sample['token_ids'].astype(np.int64)),
                'metadata': sample['metadata']
            }
