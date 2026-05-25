# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Unified data configuration for the ironcore training framework."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

import yaml

from .config import BaseConfig

# Default FIM tokens
FIM_DEFAULTS = {
    "fim_prefix_token": "<fim_prefix>",
    "fim_suffix_token": "<fim_suffix>",
    "fim_middle_token": "<fim_middle>",
}


@dataclass
class DatasetConfig(BaseConfig):
    """Configuration for a single dataset source."""

    name: str = field(default="", metadata={"help": "Dataset identifier name"})
    source: str = field(default="", metadata={"help": "HuggingFace dataset name or local path"})
    task_type: Literal["pretrain", "sft", "dpo", "grpo"] = field(
        default="pretrain", metadata={"help": "Task type: pretrain, sft, dpo, or grpo"}
    )
    ratio: float = field(default=1.0, metadata={"help": "Mixing ratio for weighted sampling"})
    split: str | None = field(default="train", metadata={"help": "Dataset split name"})
    subset: str | None = field(default=None, metadata={"help": "Dataset subset for HF datasets"})

    # Column names for different task types
    text_column: str = field(default="text", metadata={"help": "Text column name for pretrain"})
    messages_column: str = field(default="messages", metadata={"help": "Messages column for SFT"})
    chosen_column: str = field(default="chosen", metadata={"help": "Chosen column for DPO"})
    rejected_column: str = field(default="rejected", metadata={"help": "Rejected column for DPO"})

    # GRPO-specific column names
    prompt_column: str = field(default="prompt", metadata={"help": "Prompt column for GRPO"})
    answer_column: str = field(
        default="answer", metadata={"help": "Answer column for GRPO (ground truth)"}
    )

    chat_template: str | None = field(
        default=None, metadata={"help": "Chat template (uses tokenizer default if None)"}
    )
    max_samples: int | None = field(default=None, metadata={"help": "Max samples (for debugging)"})
    output_path: Path | None = field(default=None, metadata={"help": "Preprocessed output path"})
    dataset_path: str | None = field(
        default=None, metadata={"help": "Path to raw dataset (alternative to source)"}
    )

    def __post_init__(self):
        if self.ratio <= 0:
            raise ValueError(f"Dataset {self.name}: ratio must be positive, got {self.ratio}")
        if self.task_type not in ["pretrain", "sft", "dpo", "grpo"]:
            raise ValueError(f"Dataset {self.name}: invalid task_type {self.task_type}")


@dataclass
class DataConfig(BaseConfig):
    """Unified data configuration for training."""

    # Task type - determines training mode and data handling
    task_type: Literal["pretrain", "sft", "dpo", "grpo"] = field(
        default="pretrain", metadata={"help": "Task type: pretrain, sft, dpo, or grpo"}
    )

    # Mock data mode — skip preprocessing, generate random token IDs on the fly
    use_mock_data: bool = field(
        default=False,
        metadata={
            "help": "Use mock data (random token IDs) instead of real data (no preprocessing required)"
        },
    )

    # Dataset Configuration
    datasets: list[DatasetConfig] = field(
        default_factory=list, metadata={"help": "List of training datasets"}
    )
    eval_datasets: list[DatasetConfig] = field(
        default_factory=list, metadata={"help": "List of evaluation datasets"}
    )
    test_datasets: list[DatasetConfig] = field(
        default_factory=list, metadata={"help": "List of test datasets"}
    )
    data_path: list[str] | None = field(
        default=None, metadata={"help": "Legacy data path (backward compatibility)"}
    )

    # Tokenizer Configuration
    vocab_name_or_path: str = field(
        default="gpt2", metadata={"help": "Tokenizer vocab name or path"}
    )
    tokenizer_type: str = field(
        default="bbpe", metadata={"help": "Tokenizer type: bbpe, tiktoken, sentencepiece"}
    )
    vocab_size: int = field(default=51200, metadata={"help": "Vocabulary size"})
    num_token_types: int = field(default=2, metadata={"help": "Number of token types"})

    # Sequence & Splits
    seq_length: int = field(default=1024, metadata={"help": "Sequence length for training"})
    splits: list[float] = field(
        default_factory=lambda: [0.99, 0.01, 0.0],
        metadata={"help": "Train/eval/test split ratios"},
    )

    # Padding
    pad_token_id: int | None = field(
        default=None, metadata={"help": "Padding token ID (None = use EOS)"}
    )
    pad_to_max_length: bool = field(default=False, metadata={"help": "Pad sequences to max length"})

    # Preprocessing
    preprocessed_dir: Path = field(
        default=Path("./data/preprocessed"),
        metadata={"help": "Output directory for preprocessed data"},
    )
    cache_dir: Path = field(default=Path("./data/cache"), metadata={"help": "Cache directory"})
    num_workers: int = field(default=4, metadata={"help": "Number of preprocessing workers"})
    preprocessing: dict = field(
        default_factory=dict, metadata={"help": "Preprocessing-specific settings"}
    )

    # FIM Configuration
    fim_rate: float = field(
        default=0.0, metadata={"help": "FIM transformation rate (0.0 = disabled)"}
    )
    fim_split_type: Literal["random", "line_aware"] = field(
        default="random", metadata={"help": "FIM split type"}
    )
    fim_prefix_token: str = field(
        default=FIM_DEFAULTS["fim_prefix_token"], metadata={"help": "FIM prefix token"}
    )
    fim_suffix_token: str = field(
        default=FIM_DEFAULTS["fim_suffix_token"], metadata={"help": "FIM suffix token"}
    )
    fim_middle_token: str = field(
        default=FIM_DEFAULTS["fim_middle_token"], metadata={"help": "FIM middle token"}
    )

    def __post_init__(self):
        if isinstance(self.preprocessed_dir, str):
            self.preprocessed_dir = Path(self.preprocessed_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if self.splits and abs(sum(self.splits) - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {sum(self.splits)}")
        if self.task_type not in ["pretrain", "sft", "dpo", "grpo"]:
            raise ValueError(
                f"Invalid task_type '{self.task_type}'. Must be one of: pretrain, sft, dpo, grpo"
            )

    @classmethod
    def from_yaml(cls, filename: Union[str, Path]) -> "DataConfig":
        """Load configuration from a YAML file."""
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"Config file not found: {filename}")
        with open(filename) as f:
            return cls._parse_config_dict(yaml.safe_load(f))

    @classmethod
    def _parse_config_dict(cls, d: dict) -> "DataConfig":
        """Parse a dictionary into DataConfig."""
        import logging

        def parse_dataset(ds: dict) -> DatasetConfig:
            return DatasetConfig(
                name=ds.get("name", ds.get("dataset_path", "unknown")),
                source=ds.get("dataset_path", ds.get("source", ds.get("name", ""))),
                task_type=ds.get("task_type", "pretrain"),
                ratio=ds.get("ratio", 1.0),
                subset=ds.get("subset"),
                split=ds.get("split", "train"),
                text_column=ds.get("text_column", "text"),
                messages_column=ds.get("messages_column", "messages"),
                chosen_column=ds.get("chosen_column", "chosen"),
                rejected_column=ds.get("rejected_column", "rejected"),
                prompt_column=ds.get("prompt_column", "prompt"),
                answer_column=ds.get("answer_column", "answer"),
                chat_template=ds.get("chat_template"),
                max_samples=ds.get("max_samples"),
            )

        # Extract datasets based on YAML structure
        datasets, eval_datasets, test_datasets = [], [], []
        if "train_datasets" in d:
            datasets = [parse_dataset(ds) for ds in d["train_datasets"]]
            eval_datasets = [parse_dataset(ds) for ds in d.get("eval_datasets", [])]
            test_datasets = [parse_dataset(ds) for ds in d.get("test_datasets", [])]
        elif "datasets" in d:
            datasets = [parse_dataset(ds) for ds in d["datasets"] if isinstance(ds, dict)]
        else:
            logging.warning("No datasets specified in config")

        return DataConfig(
            task_type=d.get("task_type", "pretrain"),
            datasets=datasets,
            eval_datasets=eval_datasets,
            test_datasets=test_datasets,
            data_path=d.get("data_path"),
            vocab_name_or_path=d.get("vocab_name_or_path", "gpt2"),
            tokenizer_type=d.get("tokenizer_type", "bbpe"),
            vocab_size=d.get("vocab_size", 51200),
            num_token_types=d.get("num_token_types", 2),
            seq_length=d.get("seq_length", 1024),
            splits=d.get("splits", [0.99, 0.01, 0.0]),
            pad_token_id=d.get("pad_token_id"),
            pad_to_max_length=d.get("pad_to_max_length", False),
            preprocessed_dir=Path(d.get("preprocessed_dir", "./data/preprocessed")),
            cache_dir=Path(d.get("cache_dir", "./data/cache")),
            num_workers=d.get("num_workers", 4),
            preprocessing=d.get("preprocessing", {}),
            fim_rate=d.get("fim_rate", 0.0),
            fim_split_type=d.get("fim_split_type", "random"),
            fim_prefix_token=d.get("fim_prefix_token", FIM_DEFAULTS["fim_prefix_token"]),
            fim_suffix_token=d.get("fim_suffix_token", FIM_DEFAULTS["fim_suffix_token"]),
            fim_middle_token=d.get("fim_middle_token", FIM_DEFAULTS["fim_middle_token"]),
        )

    def get_dataset_output_path(self, dataset: DatasetConfig) -> Path:
        """Get the output path for a preprocessed dataset."""
        if dataset.output_path:
            return Path(dataset.output_path)
        output_path = self.preprocessed_dir / dataset.name / dataset.task_type
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


def load_data_config(config_path: Union[str, Path]) -> DataConfig:
    """Load data configuration from a YAML file."""
    return DataConfig.from_yaml(config_path)


# Backward compatibility
UniversalDataConfig = DataConfig
