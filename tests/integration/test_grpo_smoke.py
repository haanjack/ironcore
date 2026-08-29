# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GRPO training smoke tests.

Runs actual GRPO training via subprocess (torchrun) to verify the full
pipeline — rollout, reward, advantage computation, policy update — works
with both baseline (functional KV cache) and paged rollout variants.

Requires: 2 GPUs, flash attention, HuggingFace model cache.
Marked with @pytest.mark.grpo (excluded from default pytest runs).
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]  # tests/integration → repo root
GRPO_BASELINE_CONFIG = str(
    REPO_ROOT / "tests" / "fixtures" / "configs" / "grpo_baseline_smoke.yaml"
)
GRPO_PAGED_CONFIG = str(REPO_ROOT / "tests" / "fixtures" / "configs" / "grpo_paged_smoke.yaml")
# Single-GPU variant command — exercises the full GRPO pipeline
# (rollout → reward → advantage → policy update) on a one-GPU machine so the
# pipeline is not uncovered when only 1 GPU is available. (Fable issue #79.)
TORCHRUN_CMD = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2"]
TORCHRUN_CMD_SINGLE = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1"]


def _free_port() -> int:
    """Port for the nested torchrun.

    Without one it takes torchrun's default 29500, which on a shared runner is
    the port most likely to be busy — and the collision reports as
    "Training failed with exit code 1" rather than as a busy host.
    """
    import socket

    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _resolve_config_paths(config_path: str, single_gpu: bool = False) -> str:
    """Resolve relative configs/... paths to absolute paths in YAML.

    With single_gpu, also rebalance the batch block. The fixture configs were
    written for the 2-GPU class, and `_config_validation` enforces
    `micro_batch_size * gradient_accumulation_steps * dp_world_size ==
    train_batch_size`. dp_world_size comes from torchrun's WORLD_SIZE, so under
    --nproc_per_node=1 the same block fails validation before training starts.
    Trading the lost data-parallel replicas for accumulation steps keeps the
    global batch size — and therefore the optimizer trajectory — identical.
    """
    import yaml

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file) as f:
        config = yaml.safe_load(f)
    if not config:
        return config_path

    def resolve_paths(obj):
        modified = False
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value.startswith("configs/"):
                    abs_path = str((REPO_ROOT / value).resolve())
                    obj[key] = abs_path
                    modified = True
                elif isinstance(value, (dict, list)):
                    if resolve_paths(value):
                        modified = True
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str) and item.startswith("configs/"):
                    abs_path = str((REPO_ROOT / item).resolve())
                    obj[i] = abs_path
                    modified = True
                elif isinstance(item, (dict, list)):
                    if resolve_paths(item):
                        modified = True
        return modified

    modified = resolve_paths(config)

    if single_gpu:
        trainer = config.get("trainer") or {}
        micro_bs = trainer.get("micro_batch_size")
        train_bs = trainer.get("train_batch_size")
        if micro_bs and train_bs and train_bs % micro_bs == 0:
            trainer["gradient_accumulation_steps"] = train_bs // micro_bs
            modified = True

    if not modified:
        return config_path

    config_dir = config_file.parent
    suffix = "1gpu" if single_gpu else "2gpu"
    temp_file = config_dir / f"resolved_{config_file.stem}_{suffix}.yaml"
    with open(temp_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(temp_file)


def _run_training(config: str, single_gpu: bool = False) -> subprocess.CompletedProcess:
    """Run torchrun training job, return CompletedProcess."""
    resolved_config = _resolve_config_paths(config, single_gpu=single_gpu)
    cmd = (
        (TORCHRUN_CMD_SINGLE if single_gpu else TORCHRUN_CMD) + [f"--master_port={_free_port()}"]
    ) + [
        "-m",
        "ironcore",
        "train",
        "--config",
        resolved_config,
    ]
    try:
        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    finally:
        if resolved_config != config:
            Path(resolved_config).unlink(missing_ok=True)


def _assert_training_success(result: subprocess.CompletedProcess, label: str) -> None:
    tail = 3000
    assert result.returncode == 0, (
        f"[{label}] Training failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout[-tail:]}\nSTDERR:\n{result.stderr[-tail:]}"
    )
    combined = result.stdout + result.stderr
    assert "grpo_loss" in combined, f"[{label}] No grpo_loss logged — training may not have run"
    assert "Finishing training" in combined, f"[{label}] Training did not complete cleanly"


class TestGRPOTraining:
    @pytest.mark.grpo
    @pytest.mark.cuda
    @pytest.mark.mp
    def test_grpo_baseline_training(self):
        result = _run_training(GRPO_BASELINE_CONFIG)
        _assert_training_success(result, "baseline")

    @pytest.mark.grpo
    @pytest.mark.cuda
    @pytest.mark.mp
    def test_grpo_paged_rollout_training(self):
        result = _run_training(GRPO_PAGED_CONFIG)
        _assert_training_success(result, "paged")


class TestGRPOTrainingSingleGPU:
    """Single-GPU end-to-end GRPO coverage.

    The multi-GPU class above is marked ``@mp`` and skips when fewer than 2
    GPUs are visible, leaving the GRPO pipeline with no end-to-end coverage
    on a single-GPU machine. This class runs the same configs with
    ``--nproc_per_node=1`` so rollout → reward → advantage → policy update
    is exercised without requiring multiple GPUs. (Fable issue #79.)
    """

    @pytest.mark.grpo
    @pytest.mark.cuda
    def test_grpo_baseline_training_single_gpu(self):
        result = _run_training(GRPO_BASELINE_CONFIG, single_gpu=True)
        _assert_training_success(result, "baseline-single-gpu")

    @pytest.mark.grpo
    @pytest.mark.cuda
    def test_grpo_paged_rollout_training_single_gpu(self):
        result = _run_training(GRPO_PAGED_CONFIG, single_gpu=True)
        _assert_training_success(result, "paged-single-gpu")
