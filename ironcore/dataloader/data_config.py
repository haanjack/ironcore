"""
Data configuration parser for the universal data pipeline.

Parses YAML configurations that define datasets, task types, mixing ratios,
and preprocessing parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import yaml


@dataclass
class DatasetConfig:
    """Configuration for a single dataset source."""

    # Dataset identifier
    name: str

    # Source: HuggingFace dataset name or local path
    source: str

    # Task type: determines how data is processed
    task_type: Literal["pretrain", "sft", "dpo"]

    # Mixing ratio for weighted sampling
    ratio: float = 1.0

    # Optional: Subset/split name (e.g., "train", "validation")
    split: Optional[str] = "train"

    # Optional: Dataset subset (e.g., for HF datasets with configs)
    subset: Optional[str] = None

    # Optional: Column names for different task types
    text_column: str = "text"  # For pretrain
    messages_column: str = "messages"  # For SFT (chat format)
    chosen_column: str = "chosen"  # For DPO
    rejected_column: str = "rejected"  # For DPO

    # Optional: Chat template for SFT/DPO
    # If not specified, uses tokenizer's default
    chat_template: Optional[str] = None

    # Optional: Maximum number of samples to use (for debugging)
    max_samples: Optional[int] = None

    # Preprocessed output path (auto-generated if not specified)
    output_path: Optional[Path] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.ratio <= 0:
            raise ValueError(f"Dataset {self.name}: ratio must be positive, got {self.ratio}")

        if self.task_type not in ["pretrain", "sft", "dpo"]:
            raise ValueError(f"Dataset {self.name}: invalid task_type {self.task_type}")


@dataclass
class UniversalDataConfig:
    """Top-level data configuration."""

    # List of datasets to use
    datasets: List[DatasetConfig]

    # Optional: Separate evaluation and test datasets
    eval_datasets: List[DatasetConfig] = field(default_factory=list)
    test_datasets: List[DatasetConfig] = field(default_factory=list)

    # Tokenizer configuration
    vocab_name_or_path: str = "gpt2"
    tokenizer_type: str = "bbpe"  # bbpe, tiktoken, sentencepiece

    # Sequence parameters
    seq_length: int = 1024
    max_seq_len: int = 1024  # Alias for compatibility

    # Training data splits
    splits: List[float] = field(default_factory=lambda: [0.99, 0.01, 0.0])  # train/eval/test

    # Padding token ID (None = use EOS)
    pad_token_id: Optional[int] = None

    # Output directory for preprocessed data
    preprocessed_dir: Path = Path("./data/preprocessed")

    # Cache directory for downloads
    cache_dir: Path = Path("./data/cache")

    # Number of workers for preprocessing
    num_workers: int = 4

    # Preprocessing-specific settings
    preprocessing: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure paths are Path objects
        self.preprocessed_dir = Path(self.preprocessed_dir)
        self.cache_dir = Path(self.cache_dir)

        # Validate splits sum to 1.0
        if abs(sum(self.splits) - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {sum(self.splits)}")

        # Ensure at least one dataset
        if not self.datasets:
            raise ValueError("At least one dataset must be specified")

        # Set max_seq_len from seq_length if not specified
        if hasattr(self, 'seq_length') and not hasattr(self, 'max_seq_len'):
            self.max_seq_len = self.seq_length

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "UniversalDataConfig":
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            UniversalDataConfig instance
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "UniversalDataConfig":
        """
        Create UniversalDataConfig from a dictionary.

        Handles various YAML structures used in the ironcore project.
        """
        def _parse_datasets(ds_list: List[Dict]) -> List[DatasetConfig]:
            parsed = []
            for ds in ds_list:
                parsed.append(DatasetConfig(
                    name=ds.get("name", ds.get("dataset_path", "unknown")),
                    source=ds.get("dataset_path", ds.get("source", ds.get("name"))),
                    task_type=ds.get("task_type", "pretrain"),
                    ratio=ds.get("ratio", 1.0),
                    subset=ds.get("subset"),
                    split=ds.get("split", "train"),
                    text_column=ds.get("text_column", "text"),
                    messages_column=ds.get("messages_column", "messages"),
                    chosen_column=ds.get("chosen_column", "chosen"),
                    rejected_column=ds.get("rejected_column", "rejected"),
                    chat_template=ds.get("chat_template"),
                    max_samples=ds.get("max_samples"),
                ))
            return parsed

        # Extract datasets
        datasets = []
        eval_datasets = []
        test_datasets = []

        # Handle different YAML structures
        if "train_datasets" in config_dict:
            # Structure: train_datasets: [{ name: ..., dataset_path: ..., ratio: ... }]
            datasets = _parse_datasets(config_dict["train_datasets"])

            if "eval_datasets" in config_dict:
                eval_datasets = _parse_datasets(config_dict["eval_datasets"])

            if "test_datasets" in config_dict:
                test_datasets = _parse_datasets(config_dict["test_datasets"])

        elif "datasets" in config_dict:
            # Structure: datasets: [{ name: ..., source: ..., ... }]
            # This is likely the older format or a simplified one.
            # Assuming 'datasets' are train datasets.
            for ds in config_dict["datasets"]:
                datasets.append(DatasetConfig(**ds))

        else:
            raise ValueError("Config must contain 'train_datasets' or 'datasets' key")

        # Extract other config parameters
        return cls(
            datasets=datasets,
            eval_datasets=eval_datasets,
            test_datasets=test_datasets,
            vocab_name_or_path=config_dict.get("vocab_name_or_path", "gpt2"),
            tokenizer_type=config_dict.get("tokenizer_type", "bbpe"),
            seq_length=config_dict.get("seq_length", config_dict.get("max_seq_len", 1024)),
            splits=config_dict.get("splits", [0.99, 0.01, 0.0]),
            pad_token_id=config_dict.get("pad_token_id"),
            preprocessed_dir=Path(config_dict.get("preprocessed_dir", "./data/preprocessed")),
            cache_dir=Path(config_dict.get("cache_dir", "./data/cache")),
            num_workers=config_dict.get("num_workers", 4),
            preprocessing=config_dict.get("preprocessing", {}),
        )

    def get_dataset_output_path(self, dataset: DatasetConfig) -> Path:
        """Get the output path for a preprocessed dataset."""
        if dataset.output_path:
            return Path(dataset.output_path)

        # Auto-generate path: preprocessed_dir / dataset_name / task_type
        output_path = self.preprocessed_dir / dataset.name / dataset.task_type
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


def load_data_config(config_path: Union[str, Path]) -> UniversalDataConfig:
    """
    Convenience function to load data configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        UniversalDataConfig instance
    """
    return UniversalDataConfig.from_yaml(config_path)


# Alias for backwards compatibility
DataConfig = UniversalDataConfig
