# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Root pytest configuration for ironcore tests."""

import pytest


def pytest_collection_modifyitems(config, items):
    """
    Smart test skipping based on GPU availability.

    Skips tests that explicitly require CUDA or MP=2 if resources unavailable.
    Use @pytest.mark.cuda or @pytest.mark.mp on test functions/classes.

    CI/CD strategy:
    - GitHub Actions: Runs with "not cuda and not mp" filter (CPU-only)
    - Local development: All tests can run (cpu + gpu + distributed)
    """
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
    except ImportError:
        cuda_available = False
        gpu_count = 0

    skip_cuda = pytest.mark.skip(reason="GPU not available (CUDA required)")
    skip_mp = pytest.mark.skip(reason="MP requires 2+ GPUs")

    for item in items:
        if "cuda" in item.keywords and not cuda_available:
            item.add_marker(skip_cuda)

        if "mp" in item.keywords and gpu_count < 2:
            item.add_marker(skip_mp)
