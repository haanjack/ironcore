# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Training subprocess helpers — launching runs and parsing output."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from ironcore.utils import deep_merge


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
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_lines = []
    assert proc.stdout is not None  # guaranteed by stdout=PIPE
    deadline = time.monotonic() + timeout if timeout else None
    for line in proc.stdout:
        if deadline and time.monotonic() > deadline:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise subprocess.TimeoutExpired(cmd, timeout, output="".join(output_lines))
        print(line, end="")
        output_lines.append(line)
    proc.wait()
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="".join(output_lines),
        stderr="",
    )


def write_temp_config(
    base_config_dict: dict,
    overrides: dict | None = None,
    output_path: str | Path | None = None,
    original_config_path: str | Path | None = None,
) -> Path:
    """Write a (optionally modified) config dict to a YAML file.

    Temp configs are written into the same directory as the original config
    so that relative config references (e.g. ``model: micro``) resolve
    correctly.

    Args:
        base_config_dict: Base config as a dict.
        overrides: Optional dict of fields to override (deep-merged).
        output_path: Optional explicit output path.
        original_config_path: Path to the original config file, used to
            place the temp file in the correct directory.

    Returns:
        Path to the written config file.
    """
    if overrides:
        config = deep_merge(base_config_dict, overrides)
    else:
        config = deep_merge(base_config_dict, {})

    if output_path is not None:
        output_path = Path(output_path)
    elif original_config_path:
        config_dir = Path(original_config_path).resolve().parent
        tmp_name = f".ironcore_tmp_{os.getpid()}_{id(config) % 100000}.yaml"
        output_path = config_dir / tmp_name
    else:
        tmp_dir = tempfile.mkdtemp(prefix="ironcore_")
        output_path = Path(tmp_dir) / "config.yaml"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def parse_losses_from_stdout(stdout: str) -> list[float]:
    """Extract loss values from training log output."""
    pattern = r"\bloss:\s*([\d.]+)"
    matches = re.findall(pattern, stdout)
    return [float(m) for m in matches]


def parse_metrics_from_stdout(stdout: str) -> dict[str, Any]:
    """Extract training metrics from a single-step training log output."""
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
