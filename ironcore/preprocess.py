# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Data preprocessing entrypoint -- usable as Python API or via CLI."""

import logging
import sys
from pathlib import Path

from ironcore.config.config_data import DataConfig
from ironcore.preprocessing.serializer import DataSerializer

logger = logging.getLogger(__name__)


def _resolve_tokenizer(data_config: DataConfig):
    """Load and return the tokenizer specified in data_config.

    Args:
        data_config: Data configuration with tokenizer_type and vocab_name_or_path.

    Returns:
        Tokenizer instance (HuggingFace or tiktoken).

    Raises:
        ValueError: If tokenizer_type is not recognized.
    """
    if data_config.tokenizer_type == "bbpe":
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(data_config.vocab_name_or_path)
    elif data_config.tokenizer_type == "tiktoken":
        import tiktoken

        return tiktoken.get_encoding(data_config.vocab_name_or_path)
    else:
        raise ValueError(
            f"Unknown tokenizer type: {data_config.tokenizer_type!r}. "
            "Expected 'bbpe' or 'tiktoken'."
        )


def preprocess(
    config: str | Path | DataConfig,
    *,
    verbose: bool = True,
) -> DataConfig:
    """Run data preprocessing (serialization) for all configured datasets.

    Accepts either a DataConfig instance or a path to a data-configuration YAML
    file.  Tokenizer loading and serialization are performed automatically.

    Args:
        config: A DataConfig instance, or a path to a YAML file that can be
            parsed by DataConfig.from_yaml().
        verbose: Whether to print progress information to stdout.

    Returns:
        The resolved DataConfig used during preprocessing.

    Raises:
        FileNotFoundError: If *config* is a path that does not exist.
        ValueError: If the tokenizer type is unsupported or serialization fails.

    Example::

        from ironcore.preprocess import preprocess

        # From YAML
        data_config = preprocess("configs/data/pretrain.yaml")

        # From DataConfig instance
        from ironcore.config.config_data import DataConfig
        cfg = DataConfig.from_yaml("configs/data/pretrain.yaml")
        preprocess(cfg, verbose=False)
    """
    # Resolve config --------------------------------------------------------
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        logger.info("Loading data config from: %s", config_path)
        data_config = DataConfig.from_yaml(config_path)
    elif isinstance(config, DataConfig):
        data_config = config
    else:
        raise TypeError(
            f"config must be a path string, Path, or DataConfig, got {type(config).__name__}"
        )

    # Summarise configuration -----------------------------------------------
    if verbose:
        print("Datasets to process:")
        for ds in data_config.datasets:
            print(f"  - {ds.name} ({ds.task_type}): ratio={ds.ratio}")
        print(f"\nTokenizer: {data_config.vocab_name_or_path}")
        print(f"Sequence length: {data_config.seq_length}")
        print(
            f"Splits: train={data_config.splits[0]:.1%}, "
            f"eval={data_config.splits[1]:.1%}, "
            f"test={data_config.splits[2]:.1%}"
        )

    # Load tokenizer ---------------------------------------------------------
    if verbose:
        print("\nLoading tokenizer...")
    tokenizer = _resolve_tokenizer(data_config)

    # Serialize --------------------------------------------------------------
    serializer = DataSerializer(data_config=data_config, tokenizer=tokenizer, verbose=verbose)
    serializer.serialize_all()

    if verbose:
        print("\nSerialization completed successfully.")

    return data_config


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="ironcore.preprocess",
        description="IronCore Data Preprocessing",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to data configuration YAML file"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    cli_args = parser.parse_args()

    try:
        preprocess(cli_args.config, verbose=not cli_args.quiet)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
