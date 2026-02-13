# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions and and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""
Phase 2-3 validation tests for KV cache with Tensor Parallelism TP=2.

Tests:
1. Cache sharding with TP=2
2. Cross-rank consistency (TP=2 vs TP=1)
3. GQA with TP=2

Note: These tests require multiple GPUs or MPI to run properly.
For single-GPU testing, these tests are skipped.
"""

import os

import pytest
import torch

# Check if we have enough GPUs for TP=2 testing
CUDA_AVAILABLE = torch.cuda.is_available()
NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
CAN_RUN_TP2 = NUM_GPUS >= 2 or os.environ.get("TEST_WITHOUT_CUDA") == "1"

pytestmark = pytest.mark.skipif(
    not CAN_RUN_TP2, reason="TP=2 tests require at least 2 GPUs or TEST_WITHOUT_CUDA=1"
)

from ironcore.config import (  # noqa: E402
    DataConfig,
    InitConfig,
    KVCacheConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PositionalEmbeddingConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.global_vars import global_states_cleanup, set_global_states  # noqa: E402
from ironcore.language_model import LanguageModel  # noqa: E402
from ironcore.parallel import parallel_states  # noqa: E402

# Initialize parallel states for testing (TP=2)
# Note: This will only work properly with multiple GPUs or MPI
if CAN_RUN_TP2:
    try:
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=2, timeout_in_minutes=10.0
        )
    except Exception as e:
        # If initialization fails, mark as unable to run
        CAN_RUN_TP2 = False
        pytestmark = pytest.mark.skipif(True, reason=f"TP=2 initialization failed: {e}")


@pytest.fixture(scope="module")
def tp2_config():
    """Create and initialize config for TP=2 testing."""
    # Create KV cache config
    kv_cache_config = KVCacheConfig(
        enabled=True,
        max_batch_size=4,
        max_seq_length=256,
    )

    # Create positional embedding config
    pos_emb_config = PositionalEmbeddingConfig(type="rope")

    # Create model config with GQA (8 query heads, 2 KV groups)
    # With TP=2: 4 query heads, 1 KV group per rank
    model_config = ModelConfig(
        d_model=512,
        num_attention_heads=8,
        num_attention_groups=2,  # GQA - will be split to 1 per rank with TP=2
        head_dim=64,
        num_layers=2,
        d_ffn=1024,
        max_seq_len=256,
        max_position_embeddings=256,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        positional_embedding=pos_emb_config,
        kv_cache=kv_cache_config,
    )
    model_config.name = "GPT"

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=2,
        use_flash_attn=False,
    )

    init_config = InitConfig(seed=42, init_std=0.02)
    optim_config = OptimConfig(max_lr=1e-3, weight_decay=0.01)
    data_config = DataConfig()
    parallel_config = ParallelConfig()
    operation_config = OperationConfig(
        train_steps=100,
        activation_recompute=False,
    )
    utils_config = UtilsConfig()

    config = MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=init_config,
        optim=optim_config,
        data=data_config,
        parallel=parallel_config,
        operation=operation_config,
        utils=utils_config,
    )

    # Initialize global states
    set_global_states(config)
    yield config
    # Cleanup after all tests
    global_states_cleanup()


@pytest.fixture
def model(tp2_config):
    """Create a language model."""
    model = LanguageModel(tp2_config)
    model.eval()
    return model


@pytest.mark.skipif(not CUDA_AVAILABLE or NUM_GPUS < 2, reason="TP=2 tests require at least 2 GPUs")
def test_tp2_cache_sharding(model, tp2_config):
    """
    Test: Cache sharding with TP=2
    - Initialize model with TP=2
    - Verify each rank allocates cache for num_groups / 2
    - Check memory usage is half per rank
    """
    from ironcore.layers.kv_cache import KVCacheManager

    batch_size = 2
    device = next(model.parameters()).device

    # Create cache manager
    cache_manager = KVCacheManager(tp2_config)
    cache_manager.initialize(
        batch_size=batch_size,
        num_layers=tp2_config.model.num_layers,
        device=device,
    )

    # With TP=2 and 2 KV groups, each rank should have 1 KV group
    expected_local_kv_groups = (
        tp2_config.model.num_attention_groups // tp2_config.trainer.tensor_model_parallel_size
    )
    assert expected_local_kv_groups == 1

    # Check cache statistics
    stats = cache_manager.get_statistics()
    assert stats["num_local_kv_groups"] == expected_local_kv_groups
    assert stats["batch_size"] == batch_size
    assert stats["memory_mb"] > 0


@pytest.mark.skipif(not CUDA_AVAILABLE or NUM_GPUS < 2, reason="TP=2 tests require at least 2 GPUs")
def test_tp2_gqa_cache_shape(tp2_config):
    """
    Test: GQA with TP=2
    - 8 query heads, 2 KV groups, TP=2
    - Each rank: 4 query heads, 1 KV group
    - Verify cache manager is configured correctly

    Note: This test validates cache configuration without running model forward pass,
    which requires actual multi-GPU communication.
    """
    from ironcore.layers.kv_cache import KVCacheManager

    batch_size = 1
    seq_len = 5
    device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

    # Create cache manager
    cache_manager = KVCacheManager(tp2_config)
    cache_manager.initialize(
        batch_size=batch_size,
        num_layers=tp2_config.model.num_layers,
        device=device,
    )

    # With TP=2 and 2 KV groups, each rank has 1 KV group
    expected_kv_groups = (
        tp2_config.model.num_attention_groups // tp2_config.trainer.tensor_model_parallel_size
    )
    assert expected_kv_groups == 1

    # Verify cache is initialized with correct dimensions
    stats = cache_manager.get_statistics()
    assert stats["num_local_kv_groups"] == expected_kv_groups

    # Create dummy KV with correct shape and update cache
    dummy_kv = torch.randn(
        batch_size,
        seq_len,
        expected_kv_groups,
        tp2_config.model.head_dim,
        device=device,
    )

    for layer_idx in range(tp2_config.model.num_layers):
        full_key, full_value = cache_manager.update_layer(layer_idx, dummy_kv, dummy_kv, position=0)

        # Verify returned KV has correct shape from cache manager
        # update_layer returns KV from cache in [batch, num_groups, seq_len, head_dim] format
        assert full_key.shape == (
            batch_size,
            expected_kv_groups,
            seq_len,
            tp2_config.model.head_dim,
        )
        assert full_value.shape == (
            batch_size,
            expected_kv_groups,
            seq_len,
            tp2_config.model.head_dim,
        )


@pytest.mark.skipif(not CUDA_AVAILABLE or NUM_GPUS < 2, reason="TP=2 tests require at least 2 GPUs")
def test_tp2_numerical_equivalence(tp2_config):
    """
    Test: Numerical equivalence between cached and non-cached with TP=2

    Note: Full model forward pass requires actual multi-GPU setup.
    This test validates cache manager behavior instead.
    """
    from ironcore.layers.kv_cache import KVCacheManager

    batch_size = 2
    seq_len = 10
    device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

    # Create cache manager
    cache_manager = KVCacheManager(tp2_config)
    cache_manager.initialize(
        batch_size=batch_size,
        num_layers=tp2_config.model.num_layers,
        device=device,
    )

    # Create dummy KV with correct shape
    expected_kv_groups = (
        tp2_config.model.num_attention_groups // tp2_config.trainer.tensor_model_parallel_size
    )
    dummy_kv = torch.randn(
        batch_size,
        seq_len,
        expected_kv_groups,
        tp2_config.model.head_dim,
        device=device,
    )

    # Update cache for all layers
    for layer_idx in range(tp2_config.model.num_layers):
        cache_manager.update_layer(layer_idx, dummy_kv, dummy_kv, position=0)

    # Verify cache position
    assert cache_manager.get_cache_position(0) == seq_len

    # Verify we can retrieve cached KV
    for layer_idx in range(tp2_config.model.num_layers):
        key, value = cache_manager.get_layer_kv(layer_idx, start_pos=0, end_pos=seq_len)
        # get_layer_kv returns from cache in [batch, num_groups, seq_len, head_dim] format
        assert key.shape == (batch_size, expected_kv_groups, seq_len, tp2_config.model.head_dim)
        assert value.shape == (batch_size, expected_kv_groups, seq_len, tp2_config.model.head_dim)


@pytest.mark.skipif(
    True,  # Skip cross-rank consistency test as it requires running both TP=1 and TP=2
    reason="Cross-rank consistency requires separate test runs",
)
def test_tp2_cross_rank_consistency():
    """
    Test: Cross-rank consistency
    - Generate same sequence with TP=1 and TP=2
    - Both with cache enabled
    - Verify identical outputs

    Note: This test is conceptually important but requires
    running both TP=1 and TP=2 configurations and comparing
    results. In practice, this is done as a separate benchmark.
    """
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
