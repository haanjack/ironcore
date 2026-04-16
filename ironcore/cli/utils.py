# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Shared CLI utilities for experiment tools."""

import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def launch_training(
    config_path: str, num_gpus: int = 1, timeout: int = 3600
) -> subprocess.CompletedProcess:
    """Launch a training run as a subprocess.

    Args:
        config_path: Path to training config YAML.
        num_gpus: Number of GPUs to use (via torchrun).
        timeout: Maximum wall-clock time in seconds.

    Returns:
        CompletedProcess with stdout/stderr captured.
    """
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "-m",
        "ironcore",
        "train",
        "--config",
        str(config_path),
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def parse_losses_from_stdout(stdout: str) -> list[float]:
    """Extract loss values from training log output.

    Looks for lines matching the BaseTrainer.log_training format:
        'step: N, loss: X.XXXX, ...'

    Args:
        stdout: Training log output as a string.

    Returns:
        List of loss values in order of occurrence.
    """
    pattern = r"loss:\s*([\d.]+)"
    matches = re.findall(pattern, stdout)
    return [float(m) for m in matches]


def parse_metrics_from_stdout(stdout: str) -> dict[str, Any]:
    """Extract training metrics from a single-step training log output.

    Args:
        stdout: Training log output as a string.

    Returns:
        Dict with keys: loss, grad_norm, param_norm, iter_time, tokens_per_second,
        tflops_per_gpu. Values are None if not found.
    """
    metrics: dict[str, float | None] = {}

    loss_matches = parse_losses_from_stdout(stdout)
    if loss_matches:
        metrics["loss"] = loss_matches[-1]

    grad_match = re.search(r"grad_norm:\s*([\d.]+)", stdout)
    if grad_match:
        metrics["grad_norm"] = float(grad_match.group(1))

    param_match = re.search(r"param_norm:\s*([\d.]+)", stdout)
    if param_match:
        metrics["param_norm"] = float(param_match.group(1))

    time_match = re.search(r"iter_time:\s*([\d.]+)s", stdout)
    if time_match:
        metrics["iter_time"] = float(time_match.group(1))

    tps_match = re.search(r"tok/s:\s*([\d.]+)", stdout)
    if tps_match:
        metrics["tokens_per_second"] = float(tps_match.group(1))

    tflops_match = re.search(r"TFLOPS/s/GPU:\s*([\d.]+)", stdout)
    if tflops_match:
        metrics["tflops_per_gpu"] = float(tflops_match.group(1))

    return metrics


def deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge two dicts. Override values take precedence.

    Args:
        base: Base dictionary.
        override: Dictionary with values to override.

    Returns:
        New merged dictionary (does not mutate inputs).
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def write_temp_config(
    base_config_dict: dict,
    overrides: dict | None = None,
    output_path: str | Path | None = None,
    original_config_path: str | Path | None = None,
) -> Path:
    """Write a (optionally modified) config dict to a YAML file.

    Temp configs are written into the same directory as the original config
    so that relative config references (e.g. ``model: micro``) resolve
    correctly. IronCore resolves these as
    ``<config_parent>/<group>/<name>.yaml``, so the temp file must share the
    same parent directory as the original.

    Args:
        base_config_dict: Base config as a dict.
        overrides: Optional dict of fields to override (deep-merged).
        output_path: Optional explicit output path. If None, writes a temp
            file in the same directory as the original config.
        original_config_path: Path to the original config file, used to
            place the temp file in the correct directory.

    Returns:
        Path to the written config file.
    """
    if overrides:
        config = deep_merge(base_config_dict, overrides)
    else:
        config = deep_merge(base_config_dict, {})  # shallow copy

    if output_path is not None:
        output_path = Path(output_path)
    elif original_config_path:
        # Place temp file in same directory so relative refs resolve correctly
        config_dir = Path(original_config_path).resolve().parent
        tmp_name = f".ironcore_tmp_{id(config) % 100000}.yaml"
        output_path = config_dir / tmp_name
    else:
        tmp_dir = tempfile.mkdtemp(prefix="ironcore_")
        output_path = Path(tmp_dir) / "config.yaml"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def load_yaml_config(config_path: str | Path) -> dict:
    """Load a YAML config file as a plain dict.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Config as a dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_results_table(results: list[dict], columns: list[str], title: str = "") -> None:
    """Pretty-print a table of results.

    Args:
        results: List of dicts, each representing a row.
        columns: List of column keys to display.
        title: Optional table title.
    """
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
    """Gather experiment metadata: git hash, date, config info.

    Args:
        config_path: Optional path to training config for extracting details.

    Returns:
        Dict with keys: commit, date, config_path, model, parallelism, hardware, hyperparams.
    """
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
    """Extract model, parallelism, and hyperparameter info from config dict.

    Args:
        config: Config as a dict.

    Returns:
        Dict with extracted metadata fields.
    """
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
            "dp": None,  # Determined at runtime
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
