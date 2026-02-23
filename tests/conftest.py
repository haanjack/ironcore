# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Root pytest configuration for ironcore tests.

Provides:
- Custom markers for test categorization
- Fixture discovery from tests/fixtures/
- Common test utilities
"""

import pytest

# Register custom markers
markers = [
    "unit: Fast, isolated unit tests",
    "integration: Multi-component integration tests",
    "performance: Performance and benchmark tests",
    "property: Property-based tests",
    "regression: Regression tests for specific bugs",
    "slow: Tests that take longer to run",
    "distributed: Tests requiring multiple GPUs or distributed setup",
    "cuda: Tests requiring CUDA/GPU",
    "flash_attn: Tests requiring flash-attn package",
]


def pytest_configure(config):
    """Register custom markers."""
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Add marker for distributed tests based on file name patterns."""
    for item in items:
        # Auto-mark distributed tests
        if "tp_" in item.nodeid or "distributed" in item.nodeid:
            item.add_marker(pytest.mark.distributed)

        # Auto-mark cuda tests
        if any(x in item.nodeid for x in ["cuda", "attention", "transformer", "tp_"]):
            item.add_marker(pytest.mark.cuda)
