# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Shared CLI utilities for experiment tools."""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from ironcore.utils import load_yaml_config

# Re-export from the canonical location (used by other cli modules)
from ironcore.utils.subprocess import (  # noqa: F401
    launch_training,
    parse_losses_from_stdout,
    parse_metrics_from_stdout,
    write_temp_config,
)


def print_results_table(results: list[dict], columns: list[str], title: str = "") -> None:
    """Pretty-print a table of results."""
    if not results:
        print("No results to display.")
        return

    if title:
        print(f"\n{title}")
        print("=" * len(title))

    col_widths = {}
    for col in columns:
        col_widths[col] = max(
            len(col),
            *(len(str(row.get(col, ""))) for row in results),
        )

    header = " | ".join(col.upper().ljust(col_widths[col]) for col in columns)
    sep = "-+-".join("-" * col_widths[col] for col in columns)
    print(header)
    print(sep)

    for row in results:
        line = " | ".join(str(row.get(col, "")).ljust(col_widths[col]) for col in columns)
        print(line)


def gather_metadata(config_path: str | Path | None = None) -> dict[str, Any]:
    """Gather experiment metadata: git hash, date, config info."""
    metadata: dict[str, Any] = {
        "commit": _get_git_hash(),
        "date": datetime.now().isoformat(),
        "config_path": str(config_path) if config_path else None,
    }

    if config_path:
        config = load_yaml_config(config_path)
        metadata.update(_extract_config_metadata(config))

    return metadata


def _get_git_hash() -> str:
    """Get current git commit hash (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _extract_config_metadata(config: dict) -> dict[str, Any]:
    """Extract model, parallelism, and hyperparameter info from config dict."""
    model = config.get("model", {})
    if isinstance(model, str):
        model_name = model
    else:
        layers = model.get("num_layers", "?")
        d_model = model.get("d_model", "?")
        model_name = f"{layers}L-{d_model}d"

    trainer = config.get("trainer", {})
    parallel = config.get("parallel", {})
    optim = config.get("optim", {})
    operation = config.get("operation", {})

    return {
        "model": model_name,
        "parallelism": {
            "tp": trainer.get("tensor_model_parallel_size", 1),
            "dp": None,
            "fsdp": parallel.get("use_fsdp", False),
        },
        "hyperparams": {
            "optimizer": optim.get("optimizer", "?"),
            "max_lr": optim.get("max_lr", "?"),
            "warmup_steps": optim.get("warmup_steps", "?"),
            "micro_batch_size": trainer.get("micro_batch_size", "?"),
            "train_batch_size": trainer.get("train_batch_size", "?"),
            "gradient_accumulation_steps": trainer.get("gradient_accumulation_steps", "?"),
            "train_steps": operation.get("train_steps", "?"),
        },
    }
