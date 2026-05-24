# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Programmatic evaluation entrypoint — run benchmarks against trained checkpoints."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ironcore.utils import deep_merge, load_yaml_config


def evaluate(
    config_path: str | Path,
    *,
    task: str = "hellaswag",
    checkpoint: str | Path | None = None,
    num_samples: int | None = None,
    batch_size: int | None = None,
    num_gpus: int | None = None,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Run evaluation benchmarks against a trained checkpoint.

    Launches a training subprocess with ``train_steps=0`` and evaluation
    enabled.  The trainer runs configured eval tasks and this function
    parses the results from stdout.

    Args:
        config_path: Path to training config YAML.
        task: Evaluation task name (e.g. ``"hellaswag"``).
        checkpoint: Override checkpoint path.  Sets ``trainer.model_path``.
        num_samples: Optional limit on number of evaluation samples.
        batch_size: Optional eval batch size override.
        num_gpus: Number of GPUs (defaults to tensor_model_parallel_size from config).
        timeout: Maximum wall-clock time in seconds.

    Returns:
        Dict with keys: ``task``, ``config``, ``checkpoint``, ``results``,
        ``status``.
    """
    from ironcore.utils.subprocess import launch_training, write_temp_config

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_yaml_config(config_path)

    overrides: dict[str, Any] = {
        "operation": {
            "train_steps": 0,
            "no_save": True,
        },
        "profiler": {
            "gpu_profiler": False,
            "torch_profiler": False,
            "comm_profiler": False,
            "layer_timing": False,
        },
    }

    if checkpoint is not None:
        overrides.setdefault("trainer", {})["model_path"] = str(checkpoint)

    if num_samples is not None:
        overrides.setdefault("operation", {})["eval_samples"] = num_samples

    if batch_size is not None:
        overrides.setdefault("trainer", {})["eval_batch_size"] = batch_size

    # Ensure eval datasets are configured for the requested task
    data_config = config.get("data", {})
    if isinstance(data_config, dict) and not data_config.get("eval_datasets"):
        data_config["eval_datasets"] = [{"name": task}]

    patched = deep_merge(config, overrides)

    tp_size = config.get("trainer", {}).get("tensor_model_parallel_size", 1)
    gpus = num_gpus if num_gpus is not None else max(1, tp_size)

    temp_path = write_temp_config(patched, original_config_path=config_path)

    try:
        result = launch_training(str(temp_path), num_gpus=gpus, timeout=timeout)
    except Exception as e:
        return {
            "task": task,
            "config": str(config_path),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "results": {},
            "status": "error",
            "error": str(e),
        }
    finally:
        temp_path.unlink(missing_ok=True)

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        return {
            "task": task,
            "config": str(config_path),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "results": {},
            "status": "failed",
            "error": stderr[-500:] if stderr else stdout[-500:],
        }

    # Parse eval metrics from stdout
    metrics: dict[str, float] = {}

    for name, value in re.findall(r"(eval/\S+):\s*([\d.]+)", stdout):
        metrics[name] = float(value)

    acc_matches = re.findall(r"accuracy:\s*([\d.]+)", stdout)
    if acc_matches:
        metrics["accuracy"] = float(acc_matches[-1])

    return {
        "task": task,
        "config": str(config_path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "results": metrics,
        "status": "ok",
    }
