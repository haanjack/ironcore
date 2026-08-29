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
from ironcore.dataloader.random_dataset import (
    RandomTokenDataset,
    get_random_data_iterator,
)

__all__ = [
    "UniversalCollator",
    "WeightedMixingDataset",
    "BinaryDataset",
    "RandomTokenDataset",
    "get_data_iterator",
    "get_random_data_iterator",
]

# Allowed base directory for data config files
_DATA_CONFIG_BASE_DIR = Path("configs/data").resolve()


def _resolve_pad_token_id(data_config) -> int:
    """Padding id for the collator: the configured one, else the tokenizer's EOS.

    This used to be the literal 0 with the comment "GPT-2 uses 0 as pad token",
    which ignored `data.pad_token_id` entirely. For any tokenizer where 0 is real
    vocabulary rather than a pad slot — LLaMA and Qwen both ship here — every
    padded position was silently filled with a real token instead.
    """
    configured = getattr(data_config, "pad_token_id", None)
    if configured is not None:
        return configured

    try:
        from ironcore.global_vars import get_tokenizer

        eos = get_tokenizer().eos_token_id
    except Exception:  # tokenizer not built yet (unit tests, preprocessing paths)
        eos = None
    return eos if eos is not None else 0


def get_data_iterator(config):
    """
    Create data iterators for train/eval/test splits.

    Args:
        config: MainConfig object

    Returns:
        dict: Dictionary with 'train', 'eval', 'test' iterators
    """
    # Mock data mode — skip dataset preparation entirely
    if getattr(config.data, "use_mock_data", False):
        # Prefer data.seq_length when set; fall back to model.max_seq_len so
        # smoke tests on long-context models (e.g. Qwen2.5-0.5B with max_seq_len
        # 32768) don't allocate 32K-token mock batches when the user asked for
        # a smaller seq_length. (Fable issue #71.)
        seq_length = getattr(config.data, "seq_length", None) or config.model.max_seq_len
        return get_random_data_iterator(
            seq_length=seq_length,
            vocab_size=getattr(config.model, "padded_vocab_size", 50304),
            batch_size=config.trainer.micro_batch_size,
        )

    # Load data configuration with path validation
    if hasattr(config.data, "config_path") and config.data.config_path:
        config_path = Path(config.data.config_path)
        if not config_path.is_absolute():
            config_path = config_path.resolve()
        # If not found at resolved path, try configs/data/<basename>.yaml
        # (inline YAML dicts store short names like 'data/grpo_gsm8k'
        # which resolve to <cwd>/data/grpo_gsm8k, not configs/data/)
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
        # Data config is already populated from inline config
        data_config = config.data
    elif isinstance(config.data, str):
        # config.data is a string name referencing a data config file
        data_config = DataConfig.from_yaml(_DATA_CONFIG_BASE_DIR / f"{config.data}.yaml")
    elif hasattr(config.data, "seq_length"):
        # config.data is already a DataConfig object from inline config
        data_config = config.data
    else:
        raise ValueError(
            "No dataset configuration found. "
            "Either set data.use_mock_data: true for testing, "
            "or provide a dataset config via data.config_path or data.datasets."
        )

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
            pad_token_id=_resolve_pad_token_id(data_config),
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
