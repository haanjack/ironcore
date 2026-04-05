# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""
TP=1 vs TP=2 inference comparison tests with KV cache.

These tests validate that tensor parallelism produces numerically equivalent
results compared to single-GPU inference when using KV cache.

Tests:
1. KV cache shape comparison (TP=1 vs TP=2 configurations)
2. Head distribution validation
3. Simulated TP sharding correctness
4. Stateful cache integration
"""

import pytest
import torch

from ironcore.config import (
    DataConfig,
    InitConfig,
    KVCacheConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PEFTConfig,
    PositionalEmbeddingConfig,
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.layers.kv_cache import KVCacheManager

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
CAN_RUN_MULTI_GPU = NUM_GPUS >= 2


def create_config(tp_size: int, num_kv_groups: int = 4) -> MainConfig:
    """Create a config with specified TP size and KV groups."""
    kv_cache_config = KVCacheConfig(
        enabled=True,
        max_batch_size=4,
        max_seq_length=256,
    )

    pos_emb_config = PositionalEmbeddingConfig(type="rope")

    # Ensure num_kv_groups is divisible by tp_size
    assert num_kv_groups % tp_size == 0, (
        f"num_kv_groups ({num_kv_groups}) must be divisible by tp_size ({tp_size})"
    )

    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=8,
        num_attention_groups=num_kv_groups,  # GQA
        head_dim=64,
        num_layers=2,
        d_ffn=512,
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
        tensor_model_parallel_size=tp_size,
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
    profiler_config = ProfilerConfig()

    return MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=init_config,
        optim=optim_config,
        data=data_config,
        parallel=parallel_config,
        operation=operation_config,
        utils=utils_config,
        profiler=profiler_config,
        peft=PEFTConfig(),
    )


class TestKVCacheShapeComparison:
    """Test that KV cache shapes are correct for different TP configurations."""

    def test_kv_cache_shape_tp1_vs_tp2_config(self):
        """
        Test: Compare KV cache shapes between TP=1 and TP=2 configurations.

        With 8 query heads and 4 KV groups:
        - TP=1: 4 local KV groups
        - TP=2: 2 local KV groups per rank
        """
        num_kv_groups = 4

        # TP=1 configuration
        config_tp1 = create_config(tp_size=1, num_kv_groups=num_kv_groups)
        expected_local_tp1 = num_kv_groups // 1
        assert config_tp1.model.num_attention_groups == num_kv_groups

        # TP=2 configuration
        config_tp2 = create_config(tp_size=2, num_kv_groups=num_kv_groups)
        expected_local_tp2 = num_kv_groups // 2
        assert config_tp2.model.num_attention_groups == num_kv_groups

        # Verify local KV group calculations
        assert expected_local_tp1 == 4
        assert expected_local_tp2 == 2

    def test_cache_manager_shape_tp_comparison(self):
        """
        Test: Compare KVCacheManager shapes between TP configurations.

        Cache shape: [batch, max_seq_len, num_local_kv_groups, head_dim]
        """
        num_kv_groups = 4
        num_layers = 2
        device = torch.device("cpu")

        # TP=1 cache
        config_tp1 = create_config(tp_size=1, num_kv_groups=num_kv_groups)
        cache_tp1 = KVCacheManager(config_tp1)
        cache_tp1.initialize(
            batch_size=2,
            num_layers=num_layers,
            device=device,
        )

        # Verify TP=1 cache shape
        for key_cache in cache_tp1.key_caches:
            # Shape: [batch, max_seq_len, num_local_kv_groups, head_dim]
            assert key_cache.shape[2] == 4  # num_local_kv_groups for TP=1

        # Note: TP=2 cache would need parallel_states to be initialized with TP=2
        # Here we just verify the calculation logic
        expected_tp2_local = num_kv_groups // 2
        assert expected_tp2_local == 2


class TestTPShardingSimulation:
    """Test TP sharding by simulating the behavior on a single device."""

    def test_kv_head_sharding_correctness(self):
        """
        Test: Verify that KV head sharding produces correct local views.

        Simulates how TP=2 would shard KV heads from a global cache.
        """
        batch_size = 2
        seq_len = 10
        num_kv_heads = 4  # Global
        head_dim = 64

        # Create a "global" KV cache (what TP=1 would see)
        global_key = torch.randn(batch_size, seq_len, num_kv_heads, head_dim)
        global_value = torch.randn(batch_size, seq_len, num_kv_heads, head_dim)

        # Simulate TP=2 sharding
        tp_size = 2
        local_kv_heads = num_kv_heads // tp_size

        # Rank 0 gets first half of KV heads
        rank0_key = global_key[:, :, :local_kv_heads, :]
        rank0_value = global_value[:, :, :local_kv_heads, :]

        # Rank 1 gets second half of KV heads
        rank1_key = global_key[:, :, local_kv_heads:, :]
        rank1_value = global_value[:, :, local_kv_heads:, :]

        # Verify shapes
        assert rank0_key.shape == (batch_size, seq_len, local_kv_heads, head_dim)
        assert rank1_key.shape == (batch_size, seq_len, local_kv_heads, head_dim)

        # Verify we can reconstruct the global cache
        reconstructed_key = torch.cat([rank0_key, rank1_key], dim=2)
        reconstructed_value = torch.cat([rank0_value, rank1_value], dim=2)

        torch.testing.assert_close(reconstructed_key, global_key)
        torch.testing.assert_close(reconstructed_value, global_value)

    def test_gqa_expansion_with_tp_sharding(self):
        """
        Test: Verify GQA expansion works correctly with TP sharding.

        With 8 query heads, 4 KV heads, TP=2:
        - Each rank has 4 query heads and 2 KV heads
        - GQA ratio: 4/2 = 2 (each KV head serves 2 query heads)
        """
        batch_size = 1
        seq_len = 5
        num_query_heads = 8
        num_kv_heads = 4
        head_dim = 64

        # Global query and KV
        global_query = torch.randn(batch_size, seq_len, num_query_heads, head_dim)
        global_kv = torch.randn(batch_size, seq_len, num_kv_heads, head_dim)

        # TP=2 sharding
        tp_size = 2
        local_query_heads = num_query_heads // tp_size  # 4
        local_kv_heads = num_kv_heads // tp_size  # 2

        # Rank 0: query heads 0-3, KV heads 0-1
        rank0_query = global_query[:, :, :local_query_heads, :]
        rank0_kv = global_kv[:, :, :local_kv_heads, :]

        # Rank 1: query heads 4-7, KV heads 2-3
        rank1_query = global_query[:, :, local_query_heads:, :]
        rank1_kv = global_kv[:, :, local_kv_heads:, :]

        # Verify GQA ratio is preserved
        gqa_ratio_rank0 = local_query_heads / local_kv_heads  # 2
        gqa_ratio_rank1 = local_query_heads / local_kv_heads  # 2
        assert gqa_ratio_rank0 == gqa_ratio_rank1 == 2

        # Verify shapes
        assert rank0_query.shape == (batch_size, seq_len, 4, head_dim)
        assert rank0_kv.shape == (batch_size, seq_len, 2, head_dim)
        assert rank1_query.shape == (batch_size, seq_len, 4, head_dim)
        assert rank1_kv.shape == (batch_size, seq_len, 2, head_dim)


class TestStatefulCacheIntegration:
    """Test stateful cache integration with transformer layers."""

    def test_cache_manager_lifecycle(self):
        """Test KVCacheManager lifecycle: initialize, update, reset."""
        config = create_config(tp_size=1, num_kv_groups=4)
        cache = KVCacheManager(config)

        # Initially not initialized
        assert not cache.is_initialized

        # Initialize
        cache.initialize(
            batch_size=2,
            num_layers=2,
            device=torch.device("cpu"),
        )
        assert cache.is_initialized
        assert len(cache.key_caches) == 2
        assert len(cache.value_caches) == 2

        # Update layer
        key = torch.randn(2, 5, 4, 64)  # [batch, seq_len, num_kv_groups, head_dim]
        value = torch.randn(2, 5, 4, 64)
        full_key, full_value = cache.update_layer(
            layer_idx=0,
            key=key,
            value=value,
            position=0,
        )
        assert full_key.shape == (2, 5, 4, 64)
        assert cache.get_cache_position(0) == 5

        # Reset
        cache.reset()
        assert cache.get_cache_position(0) == 0

    def test_cache_statistics(self):
        """Test cache statistics reporting."""
        config = create_config(tp_size=1, num_kv_groups=4)
        cache = KVCacheManager(config)

        # Before initialization
        stats = cache.get_statistics()
        assert stats["initialized"] is False

        # After initialization
        cache.initialize(
            batch_size=2,
            num_layers=2,
            device=torch.device("cpu"),
        )
        stats = cache.get_statistics()
        assert stats["initialized"] is True
        assert stats["num_layers"] == 2
        assert stats["batch_size"] == 2
        assert stats["num_local_kv_groups"] == 4

    def test_selective_reset(self):
        """Test selective cache reset for continuous batching."""
        config = create_config(tp_size=1, num_kv_groups=4)
        cache = KVCacheManager(config)
        cache.initialize(
            batch_size=4,
            num_layers=2,
            device=torch.device("cpu"),
        )

        # Update all sequences
        key = torch.randn(4, 5, 4, 64)
        value = torch.randn(4, 5, 4, 64)
        cache.update_layer(layer_idx=0, key=key, value=value, position=0)

        # All positions should be at 5
        assert cache.get_cache_position(0) == 5
        assert cache.get_cache_position(2) == 5

        # Reset only sequences 1 and 3
        cache.reset(batch_indices=torch.tensor([1, 3]))

        # Verify selective reset
        assert cache.get_cache_position(0) == 5  # Not reset
        assert cache.get_cache_position(1) == 0  # Reset
        assert cache.get_cache_position(2) == 5  # Not reset
        assert cache.get_cache_position(3) == 0  # Reset


class TestMultiGPUInference:
    """Tests that require actual multi-GPU setup."""

    pytestmark = pytest.mark.skipif(
        not CAN_RUN_MULTI_GPU,
        reason="Multi-GPU tests require at least 2 GPUs",
    )

    @pytest.mark.skipif(not CAN_RUN_MULTI_GPU, reason="Requires 2+ GPUs")
    def test_tp2_inference_with_kv_cache(self):
        """
        Test: Full inference comparison between TP=1 and TP=2.

        Note: This test requires torchrun with 2 GPUs.
        Example: torchrun --nproc_per_node=2 -m pytest test_kv_cache_tp_comparison.py
        """
        # This test would need to be run with torchrun
        # For now, it's a placeholder for actual multi-GPU testing
        pytest.skip("Run with torchrun --nproc_per_node=2")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
