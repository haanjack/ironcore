# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def init_tp():
    """Initialize tensor parallel state for tests that use model forward pass."""
    from ironcore.parallel import parallel_states

    already_initialized = parallel_states.is_model_parallel_initialized()
    if not already_initialized:
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=1, timeout_in_minutes=10.0
        )
    yield
    if not already_initialized:
        parallel_states.destroy_model_parallel()
