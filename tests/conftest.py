# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Root pytest configuration for ironcore tests."""

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """
    Smart test skipping based on GPU availability.

    Skips tests that explicitly require CUDA or MP=2 if resources unavailable.
    Use @pytest.mark.cuda or @pytest.mark.mp on test functions/classes.

    CI/CD strategy:
    - GitHub Actions: Runs with "not cuda and not mp" filter (CPU-only)
    - Local development: All tests can run (cpu + gpu + distributed)
    """
    skip_cuda = pytest.mark.skip(reason="GPU not available (CUDA required)")
    skip_mp = pytest.mark.skip(reason="MP requires 2+ GPUs")

    for item in items:
        # Skip cuda tests if no GPU
        if "cuda" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_cuda)

        # Skip mp tests if insufficient GPUs
        if "mp" in item.keywords and torch.cuda.device_count() < 2:
            item.add_marker(skip_mp)
