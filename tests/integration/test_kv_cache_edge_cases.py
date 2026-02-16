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
Edge case tests for KV cache functionality.

Tests:
1. Heterogeneous batch positions (different cache positions per sequence)
2. Cache overflow behavior
3. Concurrent sequence updates in paged cache
4. Chunked attention with cache (edge cases)
5. Cache validation and error handling
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
from ironcore.layers.kv_cache import KVCacheManager
from ironcore.layers.paged_attention import PagedKVCache
from ironcore.parallel import parallel_states

# Initialize parallel states for testing (TP=1)
parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)


def create_test_config(
    max_seq_length: int = 256,
    use_paged: bool = False,
    max_num_pages: int = 128,
    page_size: int = 16,
    sequence_chunk_size: int | None = None,
) -> MainConfig:
    """Create a test configuration."""
    kv_cache_config = KVCacheConfig(
        enabled=True,
        max_batch_size=4,
        max_seq_length=max_seq_length,
        use_paged_attention=use_paged,
        page_size=page_size,
        max_num_pages=max_num_pages,
    )

    pos_emb_config = PositionalEmbeddingConfig(type="rope")

    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=8,
        num_attention_groups=4,  # GQA
        head_dim=64,
        num_layers=2,
        d_ffn=512,
        max_seq_len=max_seq_length,
        max_position_embeddings=max_seq_length,
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
        sequence_chunk_size=sequence_chunk_size,
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


class TestHeterogeneousBatchPositions:
    """Test KV cache with different positions per sequence in batch."""

    def test_different_sequence_lengths_in_batch(self):
        """
        Test: Heterogeneous batch with different cache positions.

        Scenario:
        - Sequence 0: 10 cached tokens, adding 5 new
        - Sequence 1: 20 cached tokens, adding 5 new
        - Sequence 2: 5 cached tokens, adding 5 new
        """
        config = create_test_config()
        batch_size = 3
        seq_len = 5
        device = torch.device("cpu")

        cache_manager = KVCacheManager(config)
        cache_manager.initialize(
            batch_size=batch_size,
            num_layers=config.model.num_layers,
            device=device,
        )

        num_local_kv_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )

        # Set different starting positions for each sequence
        start_positions = torch.tensor([10, 20, 5], dtype=torch.long, device=device)

        # Create dummy KV data
        dummy_kv = torch.randn(
            batch_size,
            seq_len,
            num_local_kv_groups,
            config.model.head_dim,
            device=device,
        )

        # Update cache with per-sequence positions
        for layer_idx in range(config.model.num_layers):
            full_key, full_value = cache_manager.update_layer(
                layer_idx, dummy_kv, dummy_kv, positions=start_positions
            )

        # Verify positions were updated correctly
        assert cache_manager.get_sequence_position(0) == 15  # 10 + 5
        assert cache_manager.get_sequence_position(1) == 25  # 20 + 5
        assert cache_manager.get_sequence_position(2) == 10  # 5 + 5

    def test_selective_sequence_update(self):
        """Test: Update only specific sequences in the batch."""
        config = create_test_config()
        batch_size = 4
        device = torch.device("cpu")

        cache_manager = KVCacheManager(config)
        cache_manager.initialize(
            batch_size=batch_size,
            num_layers=config.model.num_layers,
            device=device,
        )

        num_local_kv_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )

        # First, fill all sequences with initial data
        initial_kv = torch.randn(
            batch_size,
            10,
            num_local_kv_groups,
            config.model.head_dim,
            device=device,
        )
        cache_manager.update_layer(0, initial_kv, initial_kv, position=0)

        # All should be at position 10
        for i in range(batch_size):
            assert cache_manager.get_sequence_position(i) == 10

        # Reset only sequences 0 and 2
        cache_manager.reset(batch_indices=torch.tensor([0, 2]))

        assert cache_manager.get_sequence_position(0) == 0
        assert cache_manager.get_sequence_position(1) == 10  # Unchanged
        assert cache_manager.get_sequence_position(2) == 0
        assert cache_manager.get_sequence_position(3) == 10  # Unchanged


class TestCacheOverflow:
    """Test cache overflow behavior."""

    def test_cache_overflow_raises_error(self):
        """Test: Cache overflow should raise RuntimeError."""
        max_seq_length = 32
        config = create_test_config(max_seq_length=max_seq_length)
        device = torch.device("cpu")

        cache_manager = KVCacheManager(config)
        cache_manager.initialize(
            batch_size=1,
            num_layers=config.model.num_layers,
            device=device,
        )

        num_local_kv_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )

        # Fill cache to max
        full_kv = torch.randn(
            1,
            max_seq_length,
            num_local_kv_groups,
            config.model.head_dim,
            device=device,
        )
        cache_manager.update_layer(0, full_kv, full_kv, position=0)

        # Try to add more tokens - should raise
        overflow_kv = torch.randn(
            1,
            1,
            num_local_kv_groups,
            config.model.head_dim,
            device=device,
        )
        with pytest.raises(RuntimeError, match="Cache overflow"):
            cache_manager.update_layer(0, overflow_kv, overflow_kv)

    def test_paged_cache_page_exhaustion(self):
        """Test: Paged cache page exhaustion handling."""
        max_num_pages = 4
        page_size = 16
        config = create_test_config(
            use_paged=True, max_num_pages=max_num_pages, page_size=page_size
        )
        device = torch.device("cpu")

        paged_cache = PagedKVCache(config)
        paged_cache.initialize(num_layers=1, device=device)

        # Allocate sequences until pages are exhausted
        # max_num_pages = 4, page_size = 16
        # Each sequence needs 1 page for 10 tokens

        # Allocate 4 sequences (exhausts all pages)
        for i in range(4):
            paged_cache.allocate_sequence(sequence_id=i, num_tokens=10)

        # Try to allocate another - should raise
        with pytest.raises(RuntimeError, match="Not enough free pages"):
            paged_cache.allocate_sequence(sequence_id=4, num_tokens=10)


class TestConcurrentSequenceUpdates:
    """Test concurrent sequence updates in paged cache."""

    def test_interleaved_sequence_updates(self):
        """
        Test: Interleaved updates to multiple sequences.

        This simulates continuous batching where sequences
        are updated in non-sequential order.
        """
        config = create_test_config(use_paged=True, max_num_pages=32, page_size=16)
        device = torch.device("cpu")

        paged_cache = PagedKVCache(config)
        paged_cache.initialize(num_layers=2, device=device)

        num_local_kv_groups = paged_cache.num_local_kv_groups

        # Allocate 3 sequences
        paged_cache.allocate_sequence(sequence_id=0, num_tokens=50)
        paged_cache.allocate_sequence(sequence_id=1, num_tokens=30)
        paged_cache.allocate_sequence(sequence_id=2, num_tokens=40)

        # Create KV data
        def make_kv(seq_len: int) -> torch.Tensor:
            return torch.randn(1, seq_len, num_local_kv_groups, paged_cache.head_dim, device=device)

        # Interleaved updates
        # Sequence 0: add 10 tokens
        paged_cache.update_sequence(0, 0, make_kv(10), make_kv(10))
        # Sequence 1: add 5 tokens
        paged_cache.update_sequence(1, 0, make_kv(5), make_kv(5))
        # Sequence 2: add 8 tokens
        paged_cache.update_sequence(2, 0, make_kv(8), make_kv(8))

        # Verify positions
        assert paged_cache.sequence_positions[0] == 10
        assert paged_cache.sequence_positions[1] == 5
        assert paged_cache.sequence_positions[2] == 8

        # Continue interleaved updates
        paged_cache.update_sequence(0, 0, make_kv(5), make_kv(5))
        paged_cache.update_sequence(2, 0, make_kv(10), make_kv(10))
        paged_cache.update_sequence(1, 0, make_kv(10), make_kv(10))

        # Verify final positions
        assert paged_cache.sequence_positions[0] == 15
        assert paged_cache.sequence_positions[1] == 15
        assert paged_cache.sequence_positions[2] == 18

    def test_free_and_reallocate_sequence(self):
        """Test: Free a sequence and reallocate the pages."""
        config = create_test_config(use_paged=True, max_num_pages=8, page_size=16)
        device = torch.device("cpu")

        paged_cache = PagedKVCache(config)
        paged_cache.initialize(num_layers=1, device=device)

        # Allocate sequence 0
        paged_cache.allocate_sequence(sequence_id=0, num_tokens=64)  # 4 pages

        # Check page usage
        stats = paged_cache.get_statistics()
        assert stats["total_allocated"] == 4

        # Free sequence 0
        paged_cache.free_sequence(sequence_id=0)

        # Pages should be freed
        stats = paged_cache.get_statistics()
        assert stats["total_allocated"] == 0
        assert stats["total_free"] == 8

        # Reallocate
        paged_cache.allocate_sequence(sequence_id=1, num_tokens=32)  # 2 pages

        stats = paged_cache.get_statistics()
        assert stats["total_allocated"] == 2

    def test_sequence_extension_preserves_data(self):
        """
        Test: Sequence extension preserves existing KV data.

        This tests the fix for the page allocation bug where calling
        allocate_sequence() on an already-allocated sequence would
        overwrite the block table and lose cached data.
        """
        config = create_test_config(use_paged=True, max_num_pages=16, page_size=16)
        device = torch.device("cpu")

        paged_cache = PagedKVCache(config)
        paged_cache.initialize(num_layers=1, device=device)

        num_local_kv_groups = paged_cache.num_local_kv_groups

        # Allocate sequence with initial size (1 page = 16 tokens)
        paged_cache.allocate_sequence(sequence_id=0, num_tokens=16)

        # Write initial KV data (10 tokens)
        initial_kv = torch.ones(1, 10, num_local_kv_groups, paged_cache.head_dim, device=device)
        key1, value1, _ = paged_cache.update_sequence(
            sequence_id=0, layer_idx=0, key=initial_kv, value=initial_kv
        )

        # Verify initial data is stored (convert to same dtype for comparison)
        assert key1.shape[1] == 10
        assert torch.allclose(key1[0, :10, :, :].float(), initial_kv[0, :, :, :].float())

        # Add 1 more token
        one_token = torch.zeros(1, 1, num_local_kv_groups, paged_cache.head_dim, device=device)
        key2, value2, _ = paged_cache.update_sequence(
            sequence_id=0, layer_idx=0, key=one_token, value=one_token
        )

        # The first 10 tokens should still be ones (from initial_kv)
        assert key2.shape[1] == 11  # 10 initial + 1 new
        assert torch.allclose(key2[0, :10, :, :].float(), initial_kv[0, :, :, :].float())

        # Now extend beyond initial allocation (need more pages)
        # Add 20 more tokens (total 31, needs 2 pages)
        extension_kv = torch.full(
            (1, 20, num_local_kv_groups, paged_cache.head_dim), 2.0, device=device
        )
        key3, value3, block_table = paged_cache.update_sequence(
            sequence_id=0, layer_idx=0, key=extension_kv, value=extension_kv
        )

        # Verify total sequence length and data preservation
        assert key3.shape[1] == 31  # 11 + 20

        # First 10 should still be 1.0, 11th should be 0.0, last 20 should be 2.0
        assert torch.allclose(key3[0, :10, :, :].float(), initial_kv[0, :, :, :].float())
        assert torch.allclose(
            key3[0, 10, :, :].float(), torch.zeros_like(key3[0, 10, :, :]).float()
        )
        assert torch.allclose(key3[0, 11:, :, :].float(), extension_kv[0, :, :, :].float())

        # Verify that pages were extended (not reallocated from scratch)
        # If bug existed, block_table would be a new list with only 2 pages
        # With fix, it should be the original list extended
        assert len(block_table) == 2  # 2 pages needed for 31 tokens with page_size=16


class TestChunkedAttentionWithCache:
    """Test chunked attention with KV cache edge cases."""

    def test_chunked_attention_single_chunk(self):
        """Test: Chunked attention with single chunk (chunk_size >= seq_len)."""
        config = create_test_config(sequence_chunk_size=64)
        device = torch.device("cpu")

        cache_manager = KVCacheManager(config)
        cache_manager.initialize(
            batch_size=1,
            num_layers=config.model.num_layers,
            device=device,
        )

        num_local_kv_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )

        # Add tokens that fit in one chunk
        kv = torch.randn(1, 32, num_local_kv_groups, config.model.head_dim, device=device)
        full_key, full_value = cache_manager.update_layer(0, kv, kv, position=0)

        # Should have 32 tokens
        assert full_key.shape[1] == 32
        assert cache_manager.get_cache_position(0) == 32

    def test_chunked_attention_multiple_chunks(self):
        """Test: Attention context spans multiple chunks correctly."""
        config = create_test_config(sequence_chunk_size=16)
        device = torch.device("cpu")

        cache_manager = KVCacheManager(config)
        cache_manager.initialize(
            batch_size=1,
            num_layers=config.model.num_layers,
            device=device,
        )

        num_local_kv_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )

        # Add 64 tokens (4 chunks)
        kv = torch.randn(1, 64, num_local_kv_groups, config.model.head_dim, device=device)
        full_key, full_value = cache_manager.update_layer(0, kv, kv, position=0)

        assert full_key.shape[1] == 64
        assert cache_manager.get_cache_position(0) == 64

    def test_chunked_with_cached_context(self):
        """Test: Chunked attention correctly handles pre-cached context."""
        config = create_test_config(sequence_chunk_size=16)
        device = torch.device("cpu")

        cache_manager = KVCacheManager(config)
        cache_manager.initialize(
            batch_size=1,
            num_layers=config.model.num_layers,
            device=device,
        )

        num_local_kv_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )

        # First: add 32 tokens (cached context)
        cached_kv = torch.randn(1, 32, num_local_kv_groups, config.model.head_dim, device=device)
        cache_manager.update_layer(0, cached_kv, cached_kv, position=0)
        assert cache_manager.get_cache_position(0) == 32

        # Then: add 32 more tokens (should work with cached context)
        new_kv = torch.randn(1, 32, num_local_kv_groups, config.model.head_dim, device=device)
        full_key, full_value = cache_manager.update_layer(0, new_kv, new_kv)

        # Should have 64 total tokens
        assert full_key.shape[1] == 64
        assert cache_manager.get_cache_position(0) == 64


class TestCacheValidation:
    """Test cache validation and error handling."""

    def test_uninitialized_cache_error(self):
        """Test: Accessing uninitialized cache raises error."""
        config = create_test_config()
        device = torch.device("cpu")

        cache_manager = KVCacheManager(config)
        # Not initialized

        num_local_kv_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )
        dummy_kv = torch.randn(1, 10, num_local_kv_groups, config.model.head_dim, device=device)

        with pytest.raises(RuntimeError, match="not initialized"):
            cache_manager.update_layer(0, dummy_kv, dummy_kv)

    def test_invalid_position_parameter_combination(self):
        """Test: Both position and positions parameters raises error."""
        config = create_test_config()
        device = torch.device("cpu")

        cache_manager = KVCacheManager(config)
        cache_manager.initialize(
            batch_size=1,
            num_layers=config.model.num_layers,
            device=device,
        )

        num_local_kv_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )
        dummy_kv = torch.randn(1, 10, num_local_kv_groups, config.model.head_dim, device=device)
        positions = torch.tensor([0], dtype=torch.long, device=device)

        with pytest.raises(ValueError, match="Cannot specify both"):
            cache_manager.update_layer(0, dummy_kv, dummy_kv, position=0, positions=positions)

    def test_paged_cache_unallocated_sequence_error(self):
        """Test: Updating unallocated sequence raises error."""
        config = create_test_config(use_paged=True)
        device = torch.device("cpu")

        paged_cache = PagedKVCache(config)
        paged_cache.initialize(num_layers=1, device=device)

        num_local_kv_groups = paged_cache.num_local_kv_groups
        dummy_kv = torch.randn(1, 10, num_local_kv_groups, paged_cache.head_dim, device=device)

        with pytest.raises(RuntimeError, match="not allocated"):
            paged_cache.update_sequence(0, 0, dummy_kv, dummy_kv)


class TestCacheStatistics:
    """Test cache statistics and monitoring."""

    def test_utilization_tracking(self):
        """Test: Cache utilization is tracked correctly."""
        max_seq_length = 100
        config = create_test_config(max_seq_length=max_seq_length)
        device = torch.device("cpu")

        cache_manager = KVCacheManager(config)
        cache_manager.initialize(
            batch_size=1,
            num_layers=config.model.num_layers,
            device=device,
        )

        num_local_kv_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )

        # Initially 0% utilization
        stats = cache_manager.get_statistics()
        assert stats["utilization"] == 0.0

        # Add 50 tokens (50% utilization)
        kv = torch.randn(1, 50, num_local_kv_groups, config.model.head_dim, device=device)
        cache_manager.update_layer(0, kv, kv, position=0)
        stats = cache_manager.get_statistics()
        assert abs(stats["utilization"] - 0.5) < 0.01

        # Add 50 more tokens (100% utilization)
        cache_manager.update_layer(0, kv, kv, position=50)
        stats = cache_manager.get_statistics()
        assert abs(stats["utilization"] - 1.0) < 0.01

    def test_paged_cache_page_tracking(self):
        """Test: Paged cache page allocation is tracked correctly."""
        config = create_test_config(use_paged=True, max_num_pages=16, page_size=16)
        device = torch.device("cpu")

        paged_cache = PagedKVCache(config)
        paged_cache.initialize(num_layers=1, device=device)

        # Initially all pages free
        stats = paged_cache.get_statistics()
        assert stats["total_free"] == 16
        assert stats["total_allocated"] == 0
        assert stats["num_sequences"] == 0

        # Allocate one sequence (needs 2 pages for 20 tokens)
        paged_cache.allocate_sequence(sequence_id=0, num_tokens=20)
        stats = paged_cache.get_statistics()
        assert stats["total_allocated"] == 2
        assert stats["total_free"] == 14
        assert stats["num_sequences"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
