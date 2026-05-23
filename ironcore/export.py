# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Programmatic entrypoint for exporting IronCore checkpoints to HuggingFace format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ironcore.config import MainConfig


def export(
    config: MainConfig,
    output_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    architecture: str | None = None,
    use_safetensors: bool = True,
    shard_size: int | None = None,
) -> dict:
    """Export an IronCore checkpoint to HuggingFace format.

    Args:
        config: Fully resolved MainConfig (from ``load_full_config``).
        output_dir: Directory to write the HF checkpoint into.
        checkpoint: Override checkpoint path (sets ``config.trainer.model_path``).
        architecture: Target HF architecture name.  Auto-detected from
            ``config.model.hf_model_type`` when *None*.
        use_safetensors: Write safetensors files instead of PyTorch binaries.
        shard_size: Maximum shard size in **bytes**.  No sharding when *None*.

    Returns:
        Dict with export metadata (``files``, ``config_file``).
    """
    # --- lazy imports for heavy modules ------------------------------------
    from ironcore import global_vars
    from ironcore.checkpointing.native import load_checkpoint
    from ironcore.language_model import LanguageModel
    from ironcore.parallel import parallel_states

    # --- resolve paths -----------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Force single-GPU for export
    config.parallel.rank = 0
    config.parallel.local_rank = 0
    config.parallel.world_size = 1
    config.trainer.tensor_model_parallel_size = 1

    if checkpoint is not None:
        config.trainer.model_path = str(checkpoint)

    if not config.trainer.model_path:
        raise ValueError(
            "No checkpoint path specified.  Pass checkpoint= or set trainer.model_path in config."
        )

    # --- initialise parallel state (single-rank) ---------------------------
    if global_vars.GLOBAL_STATES is None:
        global_vars.set_global_states(config)

    if parallel_states._TENSOR_MODEL_PARALLEL_GROUP is None:
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=1, timeout_in_minutes=10.0
        )

    # --- build & load model ------------------------------------------------
    model = LanguageModel(config)
    model.eval()

    step = load_checkpoint(config, model)

    num_params = sum(p.numel() for p in model.parameters())

    # --- resolve architecture ----------------------------------------------
    arch = architecture if architecture is not None else (config.model.hf_model_type or "llama")

    # --- export ------------------------------------------------------------
    from ironcore.checkpointing.hf_interop import export_to_huggingface

    result = export_to_huggingface(
        model,
        output_dir,
        architecture=arch,
        use_safetensors=use_safetensors,
        shard_size=shard_size,
    )

    # Annotate with metadata useful for callers
    result["step"] = step
    result["num_params"] = num_params
    result["output_dir"] = output_dir

    return result


# ---------------------------------------------------------------------------
# CLI: ``python -m ironcore.export --config ... --output-dir ...``
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ironcore.export",
        description="Export an IronCore checkpoint to HuggingFace format",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for HF checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (overrides trainer.model_path)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="safetensors",
        choices=["safetensors", "pytorch"],
        help="Output format (default: safetensors)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Shard size in MB (no sharding if omitted)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="Target architecture (auto-detect if omitted)",
    )
    return parser


def main() -> None:
    """CLI entry-point invoked via ``python -m ironcore.export``."""
    parser = _build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    from ironcore.train import load_full_config

    config = load_full_config(config_path)

    # Convert shard size from MB to bytes
    shard_size: int | None = None
    if args.shard_size is not None:
        shard_size = args.shard_size * 1024 * 1024

    result = export(
        config,
        args.output_dir,
        checkpoint=args.checkpoint,
        architecture=args.architecture,
        use_safetensors=(args.format == "safetensors"),
        shard_size=shard_size,
    )

    output_dir = result["output_dir"]
    print(f"\nExported to: {output_dir}")
    for f in result["files"]:
        size_mb = f.stat().st_size / (1024 * 1024) if f.exists() else 0
        print(f"  {f.name} ({size_mb:.1f} MB)")
    print(f"Config: {result['config_file']}")


if __name__ == "__main__":
    main()
