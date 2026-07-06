# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Root pytest configuration for ironcore tests."""

import os

import pytest

# Set by torch.distributed.run (torchrun)'s elastic agent; never set by any test
# in this suite, so it's a reliable sentinel for "actually launched under torchrun"
# as opposed to just "RANK happens to be in os.environ" (which import-time
# os.environ.setdefault(...) calls elsewhere in the suite can cause even in a
# plain single-process pytest run).
_TORCHRUN_SENTINEL = "TORCHELASTIC_RUN_ID"

_ENV_KEYS_TO_ISOLATE = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


def pytest_collection_modifyitems(config, items):
    """
    Smart test skipping based on GPU availability.

    Skips tests that explicitly require CUDA or MP=2 if resources unavailable.
    Use @pytest.mark.cuda or @pytest.mark.mp on test functions/classes.

    CI/CD strategy:
    - GitHub Actions: Runs with "not cuda and not mp" filter (CPU-only)
    - Local development: All tests can run (cpu + gpu + distributed)

    `mp` additionally requires a real torchrun launch (see `_TORCHRUN_SENTINEL`
    above) — with 2+ GPUs present but no torchrun, a test marked `mp` needs a
    second rank that will never show up, so running it single-process just
    hangs instead of skipping.
    """
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
    except ImportError:
        cuda_available = False
        gpu_count = 0

    under_torchrun = _TORCHRUN_SENTINEL in os.environ

    skip_cuda = pytest.mark.skip(reason="GPU not available (CUDA required)")
    skip_mp = pytest.mark.skip(reason="MP requires 2+ GPUs launched via torchrun")

    for item in items:
        if "cuda" in item.keywords and not cuda_available:
            item.add_marker(skip_cuda)

        if "mp" in item.keywords and (gpu_count < 2 or not under_torchrun):
            item.add_marker(skip_mp)


@pytest.fixture(autouse=True)
def _isolate_distributed_env():
    """
    Snapshot and restore RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
    around every test.

    Several tests set these directly (e.g. a single-process world_size=1
    `dist.init_process_group` helper) and never clean up, which then leaks
    into later tests/modules for the rest of the pytest process. This only
    guards against *runtime* leaks (state set while a test runs); it does not
    protect against *import-time* pollution (module-level `os.environ` writes
    executed during collection, before any fixture runs) — those need fixing
    at the source, see `tests/fixtures/utils.py:single_gpu_env`.
    """
    originals = {k: os.environ.get(k) for k in _ENV_KEYS_TO_ISOLATE}
    yield
    for k, orig in originals.items():
        if orig is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = orig
