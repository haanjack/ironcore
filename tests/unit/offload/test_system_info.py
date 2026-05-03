# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for system_info utilities.
"""

import pytest

from ironcore.utils import (
    available_host_memory_gb,
    recommend_pinned_pool_gb,
    total_host_memory_gb,
)


class TestSystemInfo:
    """Test host memory information functions."""

    def test_available_host_memory_positive(self):
        """Available host memory should be positive."""
        avail = available_host_memory_gb()
        assert avail > 0, "Available host memory must be positive"

    def test_total_host_memory_positive(self):
        """Total host memory should be positive."""
        total = total_host_memory_gb()
        assert total > 0, "Total host memory must be positive"

    def test_available_less_than_total(self):
        """Available memory should be less than or equal to total."""
        avail = available_host_memory_gb()
        total = total_host_memory_gb()
        assert avail <= total, "Available memory cannot exceed total"

    def test_recommend_pinned_pool_returns_positive(self):
        """Recommendation should return positive value for typical model."""
        result = recommend_pinned_pool_gb(7.0)  # 7B model
        assert result > 0, "Recommended pool size must be positive"

    def test_recommend_pinned_pool_respects_floor(self):
        """Small model should still get minimum floor (8GB)."""
        result = recommend_pinned_pool_gb(0.5)  # 0.5B model (tiny)
        assert result >= 8.0, "Small model should get at least 8GB floor"

    def test_recommend_pinned_pool_respects_ceiling(self):
        """Auto-detect should be capped at 32GB based on available RAM (40% rule)."""
        # For small/medium models, the 40% of available RAM is the limiting factor
        # The 32GB cap applies to the RAM-based target, not the model-size floor
        small_result = recommend_pinned_pool_gb(1.0)  # 1B model
        # Should be capped at 32GB (from avail * 0.40, which would be higher on this system)
        assert small_result <= 32.0, "RAM-based target should be capped at 32GB"

    def test_recommend_pinned_pool_increases_with_model_size(self):
        """Larger model should get larger recommendation (within limits)."""
        small = recommend_pinned_pool_gb(1.0)
        large = recommend_pinned_pool_gb(10.0)
        # Within the [8GB, 32GB] range, larger model should get larger pool
        if small < 32.0 and large < 32.0:
            assert large > small, "Larger model should need larger pool"
