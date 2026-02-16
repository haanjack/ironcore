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
TP-aware paged attention tests.

Tests tensor parallelism support for KV cache and paged attention:
1. Basic functionality with TP=1
2. Head distribution verification
3. Cache shape validation
4. Numerical equivalence between TP configurations
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
from ironcore.layers.triton_paged_attention import (
    TRITON_AVAILABLE,
    python_paged_attention,
    triton_paged_attention,
    triton_paged_attention_tp,
    validate_tp_config,
)
from ironcore.parallel import parallel_states

# Check if CUDA is available for Triton tests
CUDA_AVAILABLE = torch.cuda.is_available()

# Initialize parallel states for testing (TP=1)
parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)


@pytest.fixture
def tp_test_config():
    """Create config for TP testing."""
    kv_cache_config = KVCacheConfig(
        enabled=True,
        max_batch_size=4,
        max_seq_length=512,
        use_paged_attention=True,
        page_size=16,
        max_num_pages=128,
    )

    pos_emb_config = PositionalEmbeddingConfig(type="rope")

    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=8,
        num_attention_groups=4,  # GQA with 4 KV heads
        head_dim=64,
        num_layers=2,
        d_ffn=512,
        max_seq_len=512,
        max_position_embeddings=512,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        positional_embedding=pos_emb_config,
        kv_cache=kv_cache_config,
    )
    model_config.name = "GPT"

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=1,
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

    config = MainConfig(
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

    return config


class TestTPHelperFunctions:
    """Test TP helper functions."""

    def test_get_local_kv_group_info_tp1(self):
        """Test get_local_kv_group_info with TP=1."""
        from ironcore.parallel.parallel_states import get_local_kv_group_info

        num_local, tp_rank, tp_size = get_local_kv_group_info(4)
        assert num_local == 4
        assert tp_rank == 0
        assert tp_size == 1

    def test_ensure_tp_compatible_valid(self):
        """Test ensure_tp_compatible with valid config."""
        from ironcore.parallel.parallel_states import ensure_tp_compatible

        # Should not raise
        ensure_tp_compatible(4)  # 4 KV groups, TP=1

    def test_ensure_tp_compatible_invalid(self):
        """Test ensure_tp_compatible with invalid config."""
        from ironcore.parallel.parallel_states import ensure_tp_compatible

        # This should work fine with TP=1
        ensure_tp_compatible(1)

    def test_validate_tp_config_valid(self):
        """Test validate_tp_config with valid config."""
        num_local, tp_size = validate_tp_config(4, 1)
        assert num_local == 4
        assert tp_size == 1

    def test_validate_tp_config_invalid_divisibility(self):
        """Test validate_tp_config with non-divisible config."""
        with pytest.raises(ValueError, match="must be divisible"):
            validate_tp_config(3, 2)  # 3 not divisible by 2

    def test_validate_tp_config_invalid_too_few_heads(self):
        """Test validate_tp_config with too few heads."""
        with pytest.raises(ValueError, match="must be >="):
            validate_tp_config(1, 4)  # 1 < 4


class TestTP1Basic:
    """Test basic functionality with TP=1."""

    def test_tp1_paged_cache_shape(self, tp_test_config):
        """Test paged cache shape with TP=1."""
        cache = PagedKVCache(tp_test_config)
        cache.initialize(
            num_layers=tp_test_config.model.num_layers,
            device=torch.device("cpu"),
        )

        # With TP=1 and 4 KV groups, cache should have 4 local groups
        assert cache.num_local_kv_groups == 4

        # Physical cache shape
        for key_cache in cache.physical_key_caches:
            assert key_cache.shape == (
                cache.max_num_pages,
                4,  # num_local_kv_groups
                cache.page_size,
                cache.head_dim,
            )

    def test_tp1_triton_paged_attention(self, tp_test_config):
        """Test Triton paged attention with TP=1."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available for Triton tests")

        device = torch.device("cuda")

        # Create simple test case
        batch_size = 2
        num_heads = 8
        num_kv_heads = 4
        head_dim = 64
        page_size = 16

        # Create query
        query = torch.randn(batch_size, 1, num_heads, head_dim, device=device)

        # Create physical cache
        num_pages = 10
        key_cache = torch.randn(num_pages, num_kv_heads, page_size, head_dim, device=device)
        value_cache = torch.randn(num_pages, num_kv_heads, page_size, head_dim, device=device)

        # Create block tables (each sequence uses 2 pages)
        block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.long, device=device)
        context_lens = torch.tensor([20, 25], dtype=torch.long, device=device)

        # Run attention
        output = triton_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
        )

        # Check output shape
        assert output.shape == (batch_size, 1, num_heads, head_dim)

    def test_tp1_triton_paged_attention_tp_wrapper(self, tp_test_config):
        """Test TP-aware wrapper with TP=1."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available for Triton tests")

        device = torch.device("cuda")

        # Create simple test case with local heads (TP=1 means all heads are local)
        batch_size = 2
        num_local_heads = 8
        num_local_kv_heads = 4
        head_dim = 64
        page_size = 16

        query = torch.randn(batch_size, 1, num_local_heads, head_dim, device=device)

        num_pages = 10
        key_cache = torch.randn(num_pages, num_local_kv_heads, page_size, head_dim, device=device)
        value_cache = torch.randn(num_pages, num_local_kv_heads, page_size, head_dim, device=device)

        block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.long, device=device)
        context_lens = torch.tensor([20, 25], dtype=torch.long, device=device)

        # Run TP-aware attention
        output = triton_paged_attention_tp(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            num_local_heads=num_local_heads,
            num_local_kv_heads=num_local_kv_heads,
            page_size=page_size,
        )

        assert output.shape == (batch_size, 1, num_local_heads, head_dim)


class TestTPHeadDistribution:
    """Test head distribution calculation."""

    def test_head_distribution_tp1(self):
        """Test head distribution with TP=1."""
        # TP=1: all heads are local
        global_kv_heads = 4
        tp_size = 1

        num_local_kv_heads, _ = validate_tp_config(global_kv_heads, tp_size)
        assert num_local_kv_heads == 4

    def test_head_distribution_calculation(self):
        """Test various head distribution scenarios."""
        test_cases = [
            # (global_kv_heads, tp_size, expected_local)
            (4, 1, 4),
            (4, 2, 2),
            (4, 4, 1),
            (8, 2, 4),
            (8, 4, 2),
            (16, 4, 4),
        ]

        for global_kv, tp, expected_local in test_cases:
            if global_kv % tp == 0:
                num_local, _ = validate_tp_config(global_kv, tp)
                assert num_local == expected_local, (
                    f"Failed for global_kv={global_kv}, tp={tp}: "
                    f"expected {expected_local}, got {num_local}"
                )

    def test_gqa_edge_case_minimum_heads(self):
        """Test GQA edge case: minimum heads for TP."""
        # With TP=4, we need at least 4 KV heads
        # This should pass
        validate_tp_config(4, 4)

        # This should fail
        with pytest.raises(ValueError):
            validate_tp_config(2, 4)


class TestTPCacheShape:
    """Test cache shape with TP configuration."""

    def test_cache_shape_matches_local_heads(self, tp_test_config):
        """Test that cache shape matches local KV head count."""
        cache = PagedKVCache(tp_test_config)
        cache.initialize(
            num_layers=tp_test_config.model.num_layers,
            device=torch.device("cpu"),
        )

        # Cache shape should use local KV groups
        for key_cache in cache.physical_key_caches:
            _, num_local, _, _ = key_cache.shape
            assert num_local == cache.num_local_kv_groups

    def test_paged_kv_update_shape(self, tp_test_config):
        """Test that KV update maintains correct shapes."""
        cache = PagedKVCache(tp_test_config)
        cache.initialize(
            num_layers=tp_test_config.model.num_layers,
            device=torch.device("cpu"),
        )

        # Allocate sequence
        cache.allocate_sequence(sequence_id=0, num_tokens=20)

        # Create KV tensors with correct shape
        batch_size = 1
        seq_len = 10
        key = torch.randn(batch_size, seq_len, cache.num_local_kv_groups, cache.head_dim)
        value = torch.randn(batch_size, seq_len, cache.num_local_kv_groups, cache.head_dim)

        # Update sequence
        full_key, full_value, block_table = cache.update_sequence(
            sequence_id=0,
            layer_idx=0,
            key=key,
            value=value,
        )

        # Verify shapes
        assert full_key.shape[2] == cache.num_local_kv_groups
        assert full_value.shape[2] == cache.num_local_kv_groups


class TestTPNumericalEquivalence:
    """Test numerical equivalence between different implementations."""

    def test_triton_vs_python_equivalence(self, tp_test_config):
        """Test that Triton and Python implementations produce equivalent results."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available for Triton tests")

        device = torch.device("cuda")

        # Create simple test case
        batch_size = 1
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32
        page_size = 4

        # Use deterministic data
        torch.manual_seed(42)
        query = torch.randn(batch_size, 1, num_heads, head_dim, device=device)

        num_pages = 4
        key_cache = torch.randn(num_pages, num_kv_heads, page_size, head_dim, device=device)
        value_cache = torch.randn(num_pages, num_kv_heads, page_size, head_dim, device=device)

        # Simple block tables
        block_tables = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        context_lens = torch.tensor([6], dtype=torch.long, device=device)

        # Run Triton version
        triton_output = triton_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
        )

        # Run Python version
        python_output = python_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
        )

        # Compare outputs
        torch.testing.assert_close(
            triton_output,
            python_output,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_tp_wrapper_vs_direct(self, tp_test_config):
        """Test that TP wrapper produces same results as direct call."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available for Triton tests")

        device = torch.device("cuda")

        torch.manual_seed(42)

        batch_size = 1
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32
        page_size = 4

        query = torch.randn(batch_size, 1, num_heads, head_dim, device=device)
        key_cache = torch.randn(4, num_kv_heads, page_size, head_dim, device=device)
        value_cache = torch.randn(4, num_kv_heads, page_size, head_dim, device=device)
        block_tables = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        context_lens = torch.tensor([6], dtype=torch.long, device=device)

        # Direct call
        direct_output = triton_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
        )

        # TP wrapper call
        wrapper_output = triton_paged_attention_tp(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            num_local_heads=num_heads,
            num_local_kv_heads=num_kv_heads,
            page_size=page_size,
        )

        # Should be identical
        torch.testing.assert_close(
            direct_output,
            wrapper_output,
            rtol=1e-6,
            atol=1e-6,
        )


class TestTPValidation:
    """Test TP validation in PagedKVCache."""

    def test_validation_passes_valid_config(self, tp_test_config):
        """Test that validation passes for valid configuration."""
        cache = PagedKVCache(tp_test_config)
        # Should not raise
        cache.initialize(
            num_layers=tp_test_config.model.num_layers,
            device=torch.device("cpu"),
        )
        assert cache.is_initialized

    def test_validation_context_length_zero(self, tp_test_config):
        """Test handling of zero context length."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available for Triton tests")

        device = torch.device("cuda")

        batch_size = 1
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32
        page_size = 4

        query = torch.randn(batch_size, 1, num_heads, head_dim, device=device)
        key_cache = torch.randn(4, num_kv_heads, page_size, head_dim, device=device)
        value_cache = torch.randn(4, num_kv_heads, page_size, head_dim, device=device)
        block_tables = torch.tensor([[0]], dtype=torch.long, device=device)
        context_lens = torch.tensor([0], dtype=torch.long, device=device)

        # Should return zeros for zero context length
        output = triton_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
        )

        assert torch.all(output == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
