"""
True streaming dataset implementation for IronCore.

Key improvements over universal_dataset.py:
1. O(1) memory usage - no loading of full indices/positions
2. Lazy metadata loading using memory-mapped .idx files
3. Infinite iteration support for pretraining
4. Deterministic shuffling with block-based approach
"""

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
            raise ValueError(f"Unsupported bytes per token: {file_size // total_tokens}. Expected 2 or 4.")

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
            num_positions = total_tokens // self.max_seq_len

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

    def __iter__(self):
        """
        Iterate over dataset based on mode.

        For pretrain: Yields token slices with block-based shuffling
        For SFT/DPO: Yields individual samples with lazy generation
        """
        if self.mode == "pretrain":
            return self._iter_pretrain_streaming()
        else:
            return self._iter_sft_streaming()

    def _iter_pretrain_streaming(self):
        """
        Pretrain mode: True streaming with block-based shuffling.

        Strategy:
            1. Compute total positions (don't store them)
            2. Process positions in blocks of shuffle_buffer_size
            3. Shuffle each block and yield
            4. Infinite iteration by cycling epochs

        Memory: O(shuffle_buffer_size) instead of O(total_positions)
        """
        # Build token ranges metadata (lightweight)
        token_ranges = []
        for ds_idx, dataset in enumerate(self.datasets):
            start, end = self.split_ranges[id(dataset)]
            token_ranges.append((ds_idx, start, end))

        # Compute total tokens and number of positions
        total_tokens = sum(end - start for _, start, end in token_ranges)
        num_positions = total_tokens // self.max_seq_len

        # Infinite loop for continuous pretraining
        epoch = 0
        while True:
            # Create RNG with epoch-specific seed for reproducibility
            rng = np.random.default_rng(seed=self.seed + epoch)

            # Generate position permutation block-by-block
            for block_start in range(0, num_positions, self.shuffle_buffer_size):
                block_end = min(block_start + self.shuffle_buffer_size, num_positions)

                # Generate positions for this block only
                block_positions = np.arange(block_start, block_end) * self.max_seq_len

                # Shuffle block
                rng.shuffle(block_positions)

                # Shard across data parallel ranks
                if self.world_size > 1:
                    rank_positions = [
                        pos
                        for i, pos in enumerate(block_positions)
                        if (block_start + i) % self.world_size == self.rank
                    ]
                else:
                    rank_positions = block_positions

                # Yield slices from this block
                for global_pos in rank_positions:
                    # Find which dataset this position belongs to
                    current_offset = 0
                    for ds_idx, start, end in token_ranges:
                        ds_length = end - start
                        if global_pos < current_offset + ds_length:
                            # Extract slice
                            local_pos = global_pos - current_offset + start
                            dataset = self.datasets[ds_idx]

                            slice_end = min(local_pos + self.max_seq_len + 1, end)
                            token_ids = dataset.data[local_pos:slice_end]

                            # Handle wrap-around
                            if len(token_ids) < self.max_seq_len + 1:
                                needed = (self.max_seq_len + 1) - len(token_ids)
                                wrap_tokens = dataset.data[start : start + needed]
                                token_ids = np.concatenate([token_ids, wrap_tokens])

                            yield torch.from_numpy(token_ids.astype(np.int64))
                            break

                        current_offset += ds_length

            epoch += 1

    def _iter_sft_streaming(self):
        """
        SFT/DPO mode: True streaming with lazy index generation.

        Strategy:
            1. Compute total samples (don't store indices)
            2. Generate shuffled indices on-the-fly with weighted sampling
            3. Only sample from non-exhausted datasets (prevents division by zero)
            4. Yield samples lazily

        Memory: O(1) instead of O(num_samples)
        """
        # Compute sample counts per dataset
        dataset_info = []
        total_samples = 0

        for ds_idx, dataset in enumerate(self.datasets):
            start, end = self.split_ranges[id(dataset)]
            num_samples = end - start
            dataset_info.append(
                {
                    "ds_idx": ds_idx,
                    "start": start,
                    "end": end,
                    "num_samples": num_samples,
                    "weight": self.weights[ds_idx],
                }
            )
            total_samples += num_samples

        # Create RNG
        rng = np.random.default_rng(seed=self.seed)

        # Generate weighted sampling probabilities
        # Effective probability = (num_samples * weight) / total_weighted_samples
        weighted_counts = np.array(
            [info["num_samples"] * info["weight"] for info in dataset_info], dtype=np.float64
        )

        # Use reservoir sampling for memory-efficient weighted shuffle
        # For each position, decide which dataset it comes from
        indices_per_dataset = [0] * len(dataset_info)

        for global_idx in range(total_samples):
            # Shard check: only process if this rank owns this index
            if self.world_size > 1 and global_idx % self.world_size != self.rank:
                continue

            # Check if all datasets are exhausted
            current_weighted_counts_sum = weighted_counts.sum()
            if current_weighted_counts_sum <= 0:
                # All available samples have been yielded.
                break

            # Weighted random selection of dataset (only from non-exhausted datasets)
            dataset_probs = weighted_counts / current_weighted_counts_sum
            selected_ds_idx = rng.choice(len(dataset_info), p=dataset_probs)

            info = dataset_info[selected_ds_idx]

            # The selected dataset is guaranteed to have samples because its weight was > 0.
            sample_idx = info["start"] + indices_per_dataset[selected_ds_idx]
            indices_per_dataset[selected_ds_idx] += 1

            # Fetch and yield sample
            dataset = self.datasets[info["ds_idx"]]
            sample = dataset[sample_idx]

            yield {
                "token_ids": torch.from_numpy(sample["token_ids"].astype(np.int64)),
                "metadata": sample["metadata"],
            }

            # If the dataset is now exhausted, set its weight to 0 for future selections.
            if indices_per_dataset[selected_ds_idx] >= info["num_samples"]:
                weighted_counts[selected_ds_idx] = 0


def get_streaming_data_iterator(config):
    """
    Create streaming data iterators for train/eval/test splits.

    Drop-in replacement for get_data_iterator() with true streaming support.

    Args:
        config: MainConfig object

    Returns:
        dict: Dictionary with 'train', 'eval', 'test' iterators
    """
    from torch.utils.data import DataLoader

    from ironcore.dataloader.collator import UniversalCollator

    # Load data configuration
    if hasattr(config.data, "config_path"):
        data_config = DataConfig.from_yaml(config.data.config_path)
    else:
        data_config = DataConfig.from_yaml(Path("configs/data") / f"{config.data}.yaml")

    # Determine task type
    task_type = getattr(config.data, "task_type", "pretrain")

    iterators = {}

    for split in ["train", "eval", "test"]:
        # Create streaming dataset
        dataset = StreamingDataset(
            data_config=data_config,
            mode=task_type,  # type: ignore
            split=split,
            seed=1337,
        )

        # Create collator
        collator = UniversalCollator(
            mode=task_type,  # type: ignore
            max_seq_len=data_config.seq_length,
            pad_token_id=0,
            use_flash_attention=getattr(config.trainer, "use_flash_attn", False),
            return_full_attention_mask=True,
        )

        # Create dataloader
        batch_size = (
            config.trainer.micro_batch_size if split == "train" else config.trainer.eval_batch_size
        )

        # For IterableDataset, num_workers > 0 can cause issues with seeding
        # Set num_workers=0 for deterministic behavior
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=0,  # Important for reproducibility with IterableDataset
        )

        # For train: already infinite, so just create iterator
        # For eval/test: single pass
        iterators[split] = iter(dataloader)

    return iterators
