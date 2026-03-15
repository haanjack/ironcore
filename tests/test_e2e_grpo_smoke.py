# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""E2E smoke tests for GRPO training (tests 65-67).

These tests run actual torchrun training jobs and are marked @pytest.mark.e2e.
They are excluded from the default test run:

    pytest -m "not e2e"   # skip E2E (default CI)
    pytest -m e2e         # run only E2E (manual)

Each test takes ~5-10 minutes on dual RTX 3090.

Test 65: New-style reward_manager config trains without error
Test 66: Legacy reward config trains without error (regression)
Test 67: RewardManager math reward is within 5% tolerance of legacy math reward
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
LEGACY_CONFIG = "configs/grpo_gsm8k_smoke_fsdp.yaml"
RM_CONFIG = "configs/grpo_gsm8k_smoke_rm.yaml"
# composite_math configs (derived from grpo_gsm8k.yaml, the production config).
# Used by test 67: apples-to-apples comparison of legacy vs RewardManager code paths
# using the same composite_math reward (correctness + format). Results from this
# pair constitute Run 7 in grpo_test_1303.md.
LEGACY_COMPOSITE_CONFIG = "configs/grpo_gsm8k_smoke_composite.yaml"
RM_COMPOSITE_CONFIG = "configs/grpo_gsm8k_smoke_rm_composite.yaml"
TORCHRUN_CMD = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2"]


def _run_training(config: str, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run torchrun training job, return CompletedProcess."""
    cmd = TORCHRUN_CMD + ["-m", "ironcore", "train", "--config", config]
    if extra_args:
        cmd += extra_args
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=900,  # 15 min hard ceiling
    )


def _extract_mean_rewards(output: str) -> list[float]:
    """Parse mean_reward values from training log output."""
    # Matches log lines like: "mean_reward=0.123" or "mean_reward: 0.123"
    pattern = r"mean_reward[=:\s]+([0-9]+\.[0-9]+)"
    return [float(m) for m in re.findall(pattern, output)]


def _extract_reward_stats(output: str) -> dict[str, float]:
    """Return mean and std of logged mean_reward values."""
    values = _extract_mean_rewards(output)
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    std = variance**0.5
    return {"mean": mean, "std": std, "n": n}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_65_new_style_config_trains_without_error():
    """Test 65: 10-step GRPO training with new-style reward_manager config runs cleanly."""
    result = _run_training(RM_CONFIG)

    assert result.returncode == 0, (
        f"Training with reward_manager config failed (exit {result.returncode}).\n"
        f"STDOUT:\n{result.stdout[-3000:]}\n"
        f"STDERR:\n{result.stderr[-3000:]}"
    )

    combined = result.stdout + result.stderr
    assert "mean_reward" in combined, (
        "No mean_reward logged — training may not have computed rewards.\n"
        f"STDOUT tail:\n{result.stdout[-2000:]}"
    )

    rewards = _extract_mean_rewards(combined)
    assert len(rewards) > 0, "Could not parse any mean_reward values from output"
    assert all(isinstance(r, float) for r in rewards)


@pytest.mark.e2e
def test_66_legacy_config_trains_without_error():
    """Test 66: 10-step GRPO training with legacy reward config runs cleanly (regression)."""
    result = _run_training(LEGACY_CONFIG)

    assert result.returncode == 0, (
        f"Training with legacy config failed (exit {result.returncode}).\n"
        f"STDOUT:\n{result.stdout[-3000:]}\n"
        f"STDERR:\n{result.stderr[-3000:]}"
    )

    combined = result.stdout + result.stderr
    assert "mean_reward" in combined, "No mean_reward logged"

    rewards = _extract_mean_rewards(combined)
    assert len(rewards) > 0, "Could not parse any mean_reward values from output"


@pytest.mark.e2e
def test_67_reward_distribution_within_tolerance():
    """Test 67: RewardManager composite_math reward is within 5% of legacy composite_math.

    Both configs derive from grpo_gsm8k.yaml (the production config) with 20 steps
    for smoke testing. LEGACY_COMPOSITE_CONFIG uses legacy reward: composite_math;
    RM_COMPOSITE_CONFIG uses reward_manager: with a single composite_math function
    (weight 1.0) — semantically identical but via the RewardManager code path.

    Results from this pair are recorded as Run 7 in grpo_test_1303.md.
    5% tolerance accounts for floating-point and worker-ordering differences.
    """
    legacy_result = _run_training(LEGACY_COMPOSITE_CONFIG)
    rm_result = _run_training(RM_COMPOSITE_CONFIG)

    assert legacy_result.returncode == 0, "Legacy composite config failed — cannot compare distributions"
    assert rm_result.returncode == 0, "RewardManager composite config failed — cannot compare distributions"

    legacy_stats = _extract_reward_stats(legacy_result.stdout + legacy_result.stderr)
    rm_stats = _extract_reward_stats(rm_result.stdout + rm_result.stderr)

    assert legacy_stats["n"] > 0, "No reward values parsed from legacy composite run"
    assert rm_stats["n"] > 0, "No reward values parsed from RM composite run"

    legacy_mean = legacy_stats["mean"]
    rm_mean = rm_stats["mean"]

    # Both should produce non-degenerate rewards (composite_math gives partial credit
    # even when the final answer is wrong, so mean_reward > 0 from step 1)
    assert legacy_mean > 0.0, f"Legacy composite mean_reward degenerate: {legacy_mean:.4f}"
    assert rm_mean > 0.0, f"RewardManager composite mean_reward degenerate: {rm_mean:.4f}"

    # Within 5% relative tolerance of the larger value
    larger = max(abs(legacy_mean), abs(rm_mean))
    rel_diff = abs(legacy_mean - rm_mean) / (larger + 1e-9)
    print(
        f"\n[Test 67 Run 7]\n"
        f"  legacy   mean_reward={legacy_mean:.4f} (n={legacy_stats['n']})\n"
        f"  rm       mean_reward={rm_mean:.4f}   (n={rm_stats['n']})\n"
        f"  rel_diff={rel_diff:.2%}"
    )
    assert rel_diff <= 0.05, (
        f"Reward distribution diverged beyond 5% tolerance (Run 7 baseline).\n"
        f"  legacy   mean_reward={legacy_mean:.4f} (n={legacy_stats['n']})\n"
        f"  rm       mean_reward={rm_mean:.4f}   (n={rm_stats['n']})\n"
        f"  rel_diff={rel_diff:.2%}"
    )
