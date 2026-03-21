# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for parallel unit tests."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from ironcore.parallel import parallel_states


@pytest.fixture(autouse=True)
def initialize_parallel_states_for_single_gpu():
    """
    Initialize parallel_states for single-GPU test mode.

    This fixture:
    1. Sets all worldsizes to 1 (single GPU mode)
    2. Creates mock distributed process groups
    3. Cleans up after the test

    This is needed because smoke tests run in non-distributed mode,
    but clip_grad_norm() expects parallel_states to be initialized
    (as it's only called from trainer context where init has occurred).
    """
    # Store original values
    original_tp_size = parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE
    original_dp_size = parallel_states._DATA_PARALLEL_WORLD_SIZE
    original_tp_group = parallel_states._TENSOR_MODEL_PARALLEL_GROUP
    original_dp_group = parallel_states._DATA_PARALLEL_GROUP

    # Initialize for single GPU mode (world size = 1)
    parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
    parallel_states._DATA_PARALLEL_WORLD_SIZE = 1
    # Use None for groups in single GPU mode (no communication needed)
    parallel_states._TENSOR_MODEL_PARALLEL_GROUP = None
    parallel_states._DATA_PARALLEL_GROUP = None

    yield

    # Restore original values after test
    parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE = original_tp_size
    parallel_states._DATA_PARALLEL_WORLD_SIZE = original_dp_size
    parallel_states._TENSOR_MODEL_PARALLEL_GROUP = original_tp_group
    parallel_states._DATA_PARALLEL_GROUP = original_dp_group
