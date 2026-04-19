# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Export IronCore checkpoints to HuggingFace format."""

import sys
from argparse import Namespace
from pathlib import Path

from .utils import load_full_config


def run_export(args: Namespace) -> None:
    """Export an IronCore checkpoint to HuggingFace format.

    Args:
        args: Command-line arguments.
    """
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config, force single-GPU
    config = load_full_config(config_path)
    config.parallel.rank = 0
    config.parallel.local_rank = 0
    config.parallel.world_size = 1
    config.trainer.tensor_model_parallel_size = 1

    # Set checkpoint path
    if args.checkpoint:
        config.trainer.model_path = args.checkpoint

    if not config.trainer.model_path:
        print(
            "Error: no checkpoint path specified. Use --checkpoint or set trainer.model_path in config."
        )
        sys.exit(1)

    # Initialize parallel state for model construction
    from ironcore import global_vars
    from ironcore.parallel import parallel_states

    if global_vars.GLOBAL_STATES is None:
        global_vars.set_global_states(config)

    if parallel_states._TENSOR_MODEL_PARALLEL_GROUP is None:
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=1, timeout_in_minutes=10.0
        )

    # Build model
    from ironcore.language_model import LanguageModel

    model = LanguageModel(config)
    model.eval()

    # Load checkpoint
    from ironcore.checkpointing.native import load_checkpoint

    step = load_checkpoint(config, model)
    print(f"Loaded checkpoint from step {step}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.model.name} ({num_params:,} params)")

    # Determine architecture
    architecture = args.architecture
    if architecture is None:
        architecture = config.model.hf_model_type or "llama"

    # Convert shard size from MB to bytes
    shard_size = None
    if args.shard_size:
        shard_size = args.shard_size * 1024 * 1024

    # Export
    from ironcore.checkpointing.hf_interop import export_to_huggingface

    result = export_to_huggingface(
        model,
        output_dir,
        architecture=architecture,
        use_safetensors=(args.format == "safetensors"),
        shard_size=shard_size,
    )

    print(f"\nExported to: {output_dir}")
    for f in result["files"]:
        size_mb = f.stat().st_size / (1024 * 1024) if f.exists() else 0
        print(f"  {f.name} ({size_mb:.1f} MB)")
    print(f"Config: {result['config_file']}")
