# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GRPO training smoke tests.

Runs actual GRPO training via subprocess (torchrun) to verify the full
pipeline — rollout, reward, advantage computation, policy update — works
with both baseline (functional KV cache) and paged rollout variants.

Requires: 2 GPUs, flash attention, HuggingFace model cache.
Marked with @pytest.mark.rlvr (excluded from default pytest runs).
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]  # tests/integration → repo root
GRPO_BASELINE_CONFIG = str(REPO_ROOT / "tests" / "fixtures" / "configs" / "grpo_baseline_smoke.yaml")
GRPO_PAGED_CONFIG = str(REPO_ROOT / "tests" / "fixtures" / "configs" / "grpo_paged_smoke.yaml")
TORCHRUN_CMD = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2"]


def _resolve_config_paths(config_path: str) -> str:
    """Resolve relative configs/... paths to absolute paths in YAML."""
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

    if not resolve_paths(config):
        return config_path

    config_dir = config_file.parent
    temp_file = config_dir / f"resolved_{config_file.stem}.yaml"
    with open(temp_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(temp_file)


def _run_training(config: str) -> subprocess.CompletedProcess:
    """Run torchrun training job, return CompletedProcess."""
    resolved_config = _resolve_config_paths(config)
    cmd = TORCHRUN_CMD + ["-m", "ironcore", "train", "--config", resolved_config]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


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
    @pytest.mark.rlvr
    def test_grpo_baseline_training(self):
        result = _run_training(GRPO_BASELINE_CONFIG)
        _assert_training_success(result, "baseline")

    @pytest.mark.rlvr
    def test_grpo_paged_rollout_training(self):
        result = _run_training(GRPO_PAGED_CONFIG)
        _assert_training_success(result, "paged")
