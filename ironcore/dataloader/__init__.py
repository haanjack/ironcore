"""IronCore dataloader module."""

from pathlib import Path

from torch.utils.data import DataLoader

from ironcore.dataloader.collator import UniversalCollator
from ironcore.dataloader.data_config import DataConfig
from ironcore.dataloader.dataset import (
    StreamingBinaryDataset as BinaryDataset,
)
from ironcore.dataloader.dataset import (
    StreamingDataset as WeightedMixingDataset,
)

__all__ = [
    "UniversalCollator",
    "WeightedMixingDataset",
    "BinaryDataset",
    "get_data_iterator",
    "get_vla_data_iterator",
]


def get_data_iterator(config):
    """
    Create data iterators for train/eval/test splits.

    Args:
        config: MainConfig object

    Returns:
        dict: Dictionary with 'train', 'eval', 'test' iterators
    """
    # Load data configuration
    if hasattr(config.data, "config_path"):
        data_config = DataConfig.from_yaml(config.data.config_path)
    else:
        # Fallback: try to load from configs/data/{name}.yaml
        data_config = DataConfig.from_yaml(Path("configs/data") / f"{config.data}.yaml")

    # Determine task type
    task_type = getattr(config.data, "task_type", "pretrain")

    # Check if VLA task type
    if task_type == "vla":
        return get_vla_data_iterator(config)

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


def get_vla_data_iterator(config):
    """Create data iterators for VLA training.

    Args:
        config: MainConfig object with VLA configuration

    Returns:
        dict: Dictionary with 'train', 'eval', 'test' iterators
    """
    from ironcore.dataloader.vla_collator import VLACollator
    from ironcore.dataloader.vla_dataset import VLADataset, VLADatasetConfig

    # Get VLA-specific configuration
    vla_config = config.vla
    vision_config = vla_config.vision

    # Get data configuration from YAML or use defaults
    data_config_path = getattr(config.data, "config_path", None)
    if data_config_path:
        import yaml

        with open(data_config_path) as f:
            data_yaml = yaml.safe_load(f)
        dataset_configs = data_yaml.get("datasets", [])
    else:
        # Use default config
        dataset_configs = [
            {
                "name": "vla_default",
                "source": getattr(config.data, "source", "data/vla"),
                "task_type": "vla",
            }
        ]

    # Create iterators for each split
    iterators = {}

    for split in ["train", "eval", "test"]:
        datasets = []

        for ds_config in dataset_configs:
            # Create dataset configuration
            vla_ds_config = VLADatasetConfig(
                name=ds_config.get("name", "vla"),
                source=ds_config.get("source", "data/vla"),
                task_type="vla",
                image_column=ds_config.get("image_column", "image"),
                instruction_column=ds_config.get("instruction_column", "instruction"),
                action_column=ds_config.get("action_column", "action"),
                action_dim=vla_config.action.action_dim,
                prediction_horizon=vla_config.action.prediction_horizon,
            )

            # Create dataset
            dataset = VLADataset(
                config=vla_ds_config,
                vision_config=vision_config,
                split=split,
                image_token_id=vla_config.image_token_id,
                action_token_id=vla_config.action_token_id,
            )

            datasets.append(dataset)

        # For now, use first dataset (can be extended for mixing)
        dataset = datasets[0] if datasets else None

        if dataset is None:
            continue

        # Create collator
        collator = VLACollator(
            vision_config=vision_config,
            pad_token_id=0,
            max_seq_len=config.model.max_seq_len,
            image_token_id=vla_config.image_token_id,
            action_token_id=vla_config.action_token_id,
        )

        # Create dataloader
        batch_size = (
            config.trainer.micro_batch_size if split == "train" else config.trainer.eval_batch_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=0,
        )

        iterators[split] = iter(dataloader)

    return iterators
