# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""
Root pytest configuration for ironcore tests.

Provides:
- Custom markers for test categorization
- Fixture discovery from tests/fixtures/
- Common test utilities
"""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
    parser.addoption(
        "--run-tp",
        action="store_true",
        default=False,
        help="Run tensor parallel tests (requires multi-GPU)",
    )


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
    "tp: Tests requiring tensor parallelism",
    "gpu: Tests requiring GPU",
]


def pytest_configure(config):
    """Register custom markers."""
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Add markers and skip tests based on options."""
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_tp = pytest.mark.skip(reason="need --run-tp option to run")

    for item in items:
        # Auto-mark distributed tests
        if "tp_" in item.nodeid or "distributed" in item.nodeid:
            item.add_marker(pytest.mark.distributed)

        # Auto-mark cuda tests
        if any(x in item.nodeid for x in ["cuda", "attention", "transformer", "tp_"]):
            item.add_marker(pytest.mark.cuda)

        # Skip based on options
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "tp" in item.keywords and not config.getoption("--run-tp"):
            item.add_marker(skip_tp)
