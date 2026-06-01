# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
System information utilities.

Provides host memory information using psutil. This is used for
auto-detecting safe default values for the pinned memory pool used
by the offload system.
"""

import psutil


def available_host_memory_gb() -> float:
    """
    Get available host RAM in GB.

    Returns:
        Available host memory in gigabytes (float).
    """
    return psutil.virtual_memory().available / 1024**3


def total_host_memory_gb() -> float:
    """
    Get total host RAM in GB.

    Returns:
        Total host memory in gigabytes (float).
    """
    return psutil.virtual_memory().total / 1024**3


def recommend_pinned_pool_gb(model_params_billion: float) -> float:
    """
    Recommend a safe pinned memory pool size based on model and available RAM.

    The recommendation follows these rules:
    - Target 40% of available RAM, capped at 32 GB
    - Floor at max(8 GB, 4x model params in billions)
    - Ensures pool size is reasonable for both small and large models

    Args:
        model_params_billion: Model size in billions of parameters (e.g., 7.0 for 7B)

    Returns:
        Recommended pool size in GB (float).
    """
    avail = available_host_memory_gb()
    target = min(avail * 0.40, 32.0)
    floor = max(8.0, model_params_billion * 4.0)
    return max(floor, target)
