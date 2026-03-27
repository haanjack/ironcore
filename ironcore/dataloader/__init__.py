# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""IronCore dataloader module."""

from pathlib import Path

from torch.utils.data import DataLoader

from ironcore.config import _validate_path_within_dir
from ironcore.dataloader.collator import UniversalCollator
from ironcore.dataloader.data_config import DataConfig
from ironcore.dataloader.dataset import (
    StreamingBinaryDataset as BinaryDataset,
)
from ironcore.dataloader.dataset import (
    StreamingDataset as WeightedMixingDataset,
)

__all__ = ["UniversalCollator", "WeightedMixingDataset", "BinaryDataset", "get_data_iterator"]

# Allowed base directory for data config files
_DATA_CONFIG_BASE_DIR = Path("configs/data").resolve()


def get_data_iterator(config):
    """
    Create data iterators for train/eval/test splits.

    Args:
        config: MainConfig object

    Returns:
        dict: Dictionary with 'train', 'eval', 'test' iterators
    """
    # Load data configuration with path validation
    if hasattr(config.data, "config_path") and config.data.config_path:
        config_path = Path(config.data.config_path)
        if not _validate_path_within_dir(config_path, _DATA_CONFIG_BASE_DIR):
            raise ValueError(
                f"Config path '{config_path}' is outside allowed directory 'configs/data/'"
            )
        data_config = DataConfig.from_yaml(config_path)
    elif hasattr(config.data, "datasets") and len(config.data.datasets) > 0:
        # Data config is already populated from inline config
        data_config = config.data
    elif isinstance(config.data, str):
        # config.data is a string name referencing a data config file
        data_config = DataConfig.from_yaml(_DATA_CONFIG_BASE_DIR / f"{config.data}.yaml")
    elif hasattr(config.data, "seq_length"):
        # config.data is already a DataConfig object from inline config
        data_config = config.data
    else:
        raise ValueError(f"Cannot load data config from: {config.data}")

    # Determine task type
    task_type = getattr(config.data, "task_type", "pretrain")

    # Prepare data iterators for each split
    iterators = {}

    for split in ["train", "eval", "test"]:
        # Create dataset with streaming implementation
        # shuffle_buffer_size is auto-tuned: 1% of dataset, capped at [1K, 100K]
        dataset = WeightedMixingDataset(
            data_config=data_config,
            mode=task_type,  # type: ignore
            split=split,
            seed=1337,
        )

        # Create collator
        collator = UniversalCollator(
            mode=task_type,  # type: ignore
            max_seq_len=data_config.seq_length,
            pad_token_id=0,  # GPT-2 uses 0 as pad token
            use_flash_attention=getattr(config.trainer, "use_flash_attn", False),
            return_full_attention_mask=True,
        )

        # Create dataloader
        batch_size = (
            config.trainer.micro_batch_size if split == "train" else config.trainer.eval_batch_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=0,  # Streaming datasets handle their own prefetching
        )

        # Store iterator
        # Streaming dataset already provides infinite iteration for pretrain mode
        iterators[split] = iter(dataloader)

    return iterators
