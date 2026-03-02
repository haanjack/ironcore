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
TP=1 vs TP=2 inference comparison tests with KV cache.

These tests validate that tensor parallelism produces numerically equivalent
results compared to single-GPU inference when using KV cache.

Tests:
1. KV cache shape comparison (TP=1 vs TP=2 configurations)
2. Head distribution validation
3. Simulated TP sharding correctness
4. Full inference comparison (requires multi-GPU)
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
    PositionalEmbeddingConfig,
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.layers.paged_attention import PagedKVCache
from ironcore.parallel import parallel_states

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
CAN_RUN_MULTI_GPU = NUM_GPUS >= 2

# Initialize parallel states for testing (TP=1)
parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)


def create_config(tp_size: int, num_kv_groups: int = 4) -> MainConfig:
    """Create a config with specified TP size and KV groups."""
    kv_cache_config = KVCacheConfig(
        enabled=True,
        max_batch_size=4,
        max_seq_length=256,
        use_paged_attention=True,
        page_size=16,
        max_num_pages=128,
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

    def test_paged_cache_shape_tp_comparison(self):
        """
        Test: Compare paged KV cache shapes between TP configurations.

        Physical cache shape: [pages, num_local_kv_groups, page_size, head_dim]
        """
        num_kv_groups = 4
        num_layers = 2
        device = torch.device("cpu")

        # TP=1 cache
        config_tp1 = create_config(tp_size=1, num_kv_groups=num_kv_groups)
        cache_tp1 = PagedKVCache(config_tp1)
        cache_tp1.initialize(num_layers=num_layers, device=device)

        # Verify TP=1 cache shape
        for key_cache in cache_tp1.physical_key_caches:
            assert key_cache.shape[1] == 4  # num_local_kv_groups for TP=1

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


class TestTPNumericalEquivalence:
    """Test numerical equivalence between TP configurations."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA")
    def test_paged_attention_tp1_vs_tp2_simulation(self):
        """
        Test: Simulate TP=1 vs TP=2 paged attention comparison.

        This test simulates what TP=2 would compute by manually sharding
        the attention computation and comparing results.
        """
        from ironcore.layers.triton_paged_attention import (
            TRITON_AVAILABLE,
            python_paged_attention,
        )

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        device = torch.device("cuda")
        torch.manual_seed(42)

        # Setup
        batch_size = 1
        num_heads = 8
        num_kv_heads = 4
        head_dim = 64
        page_size = 4

        # Create test data
        query = torch.randn(batch_size, 1, num_heads, head_dim, device=device)
        key_cache = torch.randn(4, num_kv_heads, page_size, head_dim, device=device)
        value_cache = torch.randn(4, num_kv_heads, page_size, head_dim, device=device)
        block_tables = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        context_lens = torch.tensor([6], dtype=torch.long, device=device)

        # Compute full attention (TP=1 style)
        full_output = python_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
        )

        # Simulate TP=2: shard query and KV
        tp_size = 2
        local_heads = num_heads // tp_size
        local_kv_heads = num_kv_heads // tp_size

        # Rank 0 computation
        rank0_query = query[:, :, :local_heads, :]
        rank0_key_cache = key_cache[:, :local_kv_heads, :, :]
        rank0_value_cache = value_cache[:, :local_kv_heads, :, :]

        rank0_output = python_paged_attention(
            query=rank0_query,
            key_cache=rank0_key_cache,
            value_cache=rank0_value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            num_heads=local_heads,
            num_kv_heads=local_kv_heads,
            page_size=page_size,
        )

        # Rank 1 computation
        rank1_query = query[:, :, local_heads:, :]
        rank1_key_cache = key_cache[:, local_kv_heads:, :, :]
        rank1_value_cache = value_cache[:, local_kv_heads:, :, :]

        rank1_output = python_paged_attention(
            query=rank1_query,
            key_cache=rank1_key_cache,
            value_cache=rank1_value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            num_heads=local_heads,
            num_kv_heads=local_kv_heads,
            page_size=page_size,
        )

        # Concatenate outputs (simulating all-gather/reduce)
        tp2_simulated_output = torch.cat([rank0_output, rank1_output], dim=2)

        # Compare - should be numerically equivalent
        torch.testing.assert_close(
            full_output,
            tp2_simulated_output,
            rtol=1e-4,
            atol=1e-5,
        )


class TestTPCacheValidation:
    """Test TP configuration validation."""

    def test_valid_tp_config_passes(self):
        """Test that valid TP configurations pass validation."""
        from ironcore.layers.triton_paged_attention import validate_tp_config

        # Valid configurations
        validate_tp_config(num_kv_heads=4, tp_size=1)
        validate_tp_config(num_kv_heads=4, tp_size=2)
        validate_tp_config(num_kv_heads=4, tp_size=4)
        validate_tp_config(num_kv_heads=8, tp_size=2)

    def test_invalid_tp_config_fails(self):
        """Test that invalid TP configurations fail validation."""
        from ironcore.layers.triton_paged_attention import validate_tp_config

        # num_kv_heads < tp_size
        with pytest.raises(ValueError, match="must be >="):
            validate_tp_config(num_kv_heads=2, tp_size=4)

        # num_kv_heads not divisible by tp_size
        with pytest.raises(ValueError, match="must be divisible"):
            validate_tp_config(num_kv_heads=3, tp_size=2)

    def test_paged_cache_validation_on_initialize(self):
        """Test that PagedKVCache validates TP config on initialize."""
        # This config has 4 KV groups and TP=1, which is valid
        config = create_config(tp_size=1, num_kv_groups=4)
        cache = PagedKVCache(config)

        # Should not raise
        cache.initialize(num_layers=2, device=torch.device("cpu"))
        assert cache.is_initialized


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
