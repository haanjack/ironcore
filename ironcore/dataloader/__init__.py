"""IronCore dataloader module."""

import itertools
from pathlib import Path
from torch.utils.data import DataLoader

from ironcore.dataloader.collator import UniversalCollator
from ironcore.dataloader.universal_dataset import WeightedMixingDataset
from ironcore.dataloader.data_config import DataConfig

__all__ = ["UniversalCollator", "WeightedMixingDataset", "get_data_iterator"]


def get_data_iterator(config):
    """
    Create data iterators for train/eval/test splits.

    Args:
        config: MainConfig object

    Returns:
        dict: Dictionary with 'train', 'eval', 'test' iterators
    """
    # Load data configuration
    if hasattr(config.data, 'config_path'):
        data_config = DataConfig.from_yaml(config.data.config_path)
    else:
        # Fallback: try to load from configs/data/{name}.yaml
        data_config = DataConfig.from_yaml(Path("configs/data") / f"{config.data}.yaml")

    # Determine task type
    task_type = getattr(config.data, 'task_type', 'pretrain')

    # Prepare data iterators for each split
    iterators = {}

    for split in ['train', 'eval', 'test']:
        # Create dataset
        dataset = WeightedMixingDataset(
            data_config=data_config,
            mode=task_type, # type: ignore
            split=split,
            seed=1337,
        )

        # Create collator
        collator = UniversalCollator(
            mode=task_type, # type: ignore
            max_seq_len=data_config.seq_length,
            pad_token_id=0,  # GPT-2 uses 0 as pad token
            use_flash_attention=getattr(config.trainer, 'use_flash_attn', False),
            return_full_attention_mask=True,
        )

        # Create dataloader
        batch_size = config.trainer.micro_batch_size if split == 'train' else config.trainer.eval_batch_size
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=0,
        )

        # Store iterator
        # For train split, use cycling iterator to repeat data for multiple epochs
        # For eval/test, use single-pass iterator
        if split == 'train':
            iterators[split] = itertools.cycle(iter(dataloader))
        else:
            iterators[split] = iter(dataloader)

    return iterators
