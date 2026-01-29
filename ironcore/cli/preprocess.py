"""Data preprocessing (serialization) CLI command."""

import sys
from pathlib import Path

from ironcore.dataloader.data_config import DataConfig
from ironcore.preprocessing.serializer import DataSerializer


def run_preprocess(args):
    """Run data preprocessing command.

    Args:
        args: Command-line arguments from argparse
            - config: Path to data configuration YAML file
            - inspect: Whether to inspect output files after preprocessing
            - only_inspect: Whether to skip preprocessing and only run inspection
            - preview: Number of samples to preview (implies inspection)
    """
    # Check if we should run inspection
    should_inspect = args.inspect or args.only_inspect or (hasattr(args, 'preview') and args.preview > 0)

    if not args.only_inspect:
        # Run Preprocessing (Serialization)
        config_path = Path(args.config)

        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        print(f"Loading configuration from: {config_path}")
        data_config = DataConfig.from_yaml(config_path)

        print(f"\nDatasets to process:")
        for ds in data_config.datasets:
            print(f"  - {ds.name} ({ds.task_type}): ratio={ds.ratio}")

        print(f"\nTokenizer: {data_config.vocab_name_or_path}")
        print(f"Sequence length: {data_config.seq_length}")
        print(f"Splits: train={data_config.splits[0]:.1%}, eval={data_config.splits[1]:.1%}, test={data_config.splits[2]:.1%}")

        # Load tokenizer
        print(f"\nLoading tokenizer...")
        if data_config.tokenizer_type == "bbpe":
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(data_config.vocab_name_or_path)
        elif data_config.tokenizer_type == "tiktoken":
            import tiktoken
            tokenizer = tiktoken.get_encoding(data_config.vocab_name_or_path)
        else:
            raise ValueError(f"Unknown tokenizer type: {data_config.tokenizer_type}")

        # Initialize serializer
        serializer = DataSerializer(
            data_config=data_config,
            tokenizer=tokenizer,
            verbose=True
        )

        # Serialize all datasets
        print("\nStarting serialization...")
        try:
            serializer.serialize_all()
            print("\n✓ Serialization completed successfully!")
        except Exception as e:
            print(f"\n✗ Error during serialization: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Run Inspection if requested
    if should_inspect:
        print("\nInspecting datasets...")
        from ironcore.cli.inspect import run_inspect
        run_inspect(args)
