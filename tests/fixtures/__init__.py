# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Test fixtures for ironcore tests."""

# Re-export fixtures for easy import
from tests.fixtures.config_fixtures import (  # noqa: F401
    SMALL_MODEL_CONFIG,
    STANDARD_MODEL_CONFIG,
    create_gqa_config,
    create_mqa_config,
    create_small_test_config,
    create_standard_test_config,
    create_test_config,
    create_tp_config,
    mock_config,
    vla_config,
)
from tests.fixtures.mocks import (  # noqa: F401
    MockDataset,
    MockDistributed,
    MockRandom,
    MockTokenizer,
)
from tests.fixtures.model_fixtures import (  # noqa: F401
    attention_layer,
    causal_attention_mask,
    cuda_device,
    device,
    gqa_attention_layer,
    mqa_attention_layer,
    random_hidden_states,
    random_input_ids,
    small_attention_layer,
    transformer_model,
    transformer_model_bf16,
    init_parallel_states,
)
from tests.fixtures.utils import (  # noqa: F401
    TensorComparator,
    assert_finite,
    assert_shape,
    assert_tensors_close,
    compute_gradient_norm,
    compute_parameter_norm,
    count_parameters,
    create_causal_mask,
    get_memory_usage,
    reset_memory_stats,
    set_seed,
)

__all__ = [
    # Config factories
    "SMALL_MODEL_CONFIG",
    "STANDARD_MODEL_CONFIG",
    "create_gqa_config",
    "create_mqa_config",
    "create_small_test_config",
    "create_standard_test_config",
    "create_test_config",
    "create_tp_config",
    "mock_config",
    "vla_config",
    # Mocks
    "MockDataset",
    "MockDistributed",
    "MockRandom",
    "MockTokenizer",
    # Model fixtures
    "attention_layer",
    "causal_attention_mask",
    "cuda_device",
    "device",
    "gqa_attention_layer",
    "mqa_attention_layer",
    "random_hidden_states",
    "random_input_ids",
    "small_attention_layer",
    "transformer_model",
    "transformer_model_bf16",
    "init_parallel_states",
    # Utils
    "TensorComparator",
    "assert_finite",
    "assert_shape",
    "assert_tensors_close",
    "compute_gradient_norm",
    "compute_parameter_norm",
    "count_parameters",
    "create_causal_mask",
    "get_memory_usage",
    "reset_memory_stats",
    "set_seed",
]
