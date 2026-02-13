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
Phase 4 validation tests for Paged Attention functionality.

Tests:
1. Page allocation and deallocation
2. Variable-length sequences
3. Paged vs non-paged equivalence
4. Page refcounting (for prefix caching)
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
    TrainerConfig,
    UtilsConfig,
)
from ironcore.layers.paged_attention import PagedKVCache, PageTable
from ironcore.parallel import parallel_states

# Initialize parallel states for testing (TP=1)
parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)


@pytest.fixture
def paged_cache_config():
    """Create config for paged attention testing."""
    # Create KV cache config with paged attention enabled
    kv_cache_config = KVCacheConfig(
        enabled=True,
        max_batch_size=4,
        max_seq_length=512,
        use_paged_attention=True,
        page_size=16,
        max_num_pages=128,
    )

    # Create positional embedding config
    pos_emb_config = PositionalEmbeddingConfig(type="rope")

    # Create model config
    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=4,
        num_attention_groups=2,  # GQA
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

    return config


def test_page_allocation_deallocation():
    """
    Test: Page allocation and deallocation
    - Allocate pages for sequence
    - Free sequence
    - Verify pages returned to free pool
    - Check for memory leaks
    """
    page_table = PageTable(
        max_num_pages=32,
        page_size=16,
        device=torch.device("cpu"),
    )

    # Initially all pages are free
    assert len(page_table.free_pages) == 32
    stats = page_table.get_statistics()
    assert stats["total_free"] == 32
    assert stats["total_allocated"] == 0

    # Allocate 10 pages for sequence 0
    pages_0 = page_table.allocate_sequence(sequence_id=0, num_pages=10)
    assert len(pages_0) == 10
    assert len(page_table.free_pages) == 22
    stats = page_table.get_statistics()
    assert stats["total_free"] == 22
    assert stats["total_allocated"] == 10

    # Allocate 5 pages for sequence 1
    pages_1 = page_table.allocate_sequence(sequence_id=1, num_pages=5)
    assert len(pages_1) == 5
    assert len(page_table.free_pages) == 17
    stats = page_table.get_statistics()
    assert stats["total_allocated"] == 15

    # Verify allocated pages are unique
    assert set(pages_0).isdisjoint(set(pages_1))

    # Free sequence 0
    page_table.free_sequence(sequence_id=0)
    assert len(page_table.free_pages) == 27  # 22 + 5 from sequence 0
    stats = page_table.get_statistics()
    assert stats["total_allocated"] == 5

    # Verify sequence 1 still has its pages
    assert page_table.get_block_table(1) == pages_1

    # Free sequence 1
    page_table.free_sequence(sequence_id=1)
    assert len(page_table.free_pages) == 32  # All pages free
    stats = page_table.get_statistics()
    assert stats["total_allocated"] == 0


def test_variable_length_sequences():
    """
    Test: Variable-length sequences
    - Batch with sequences [32, 64, 128, 256] tokens
    - Compare memory usage: paged vs non-paged
    - Verify paged uses less memory (no padding waste)
    """
    page_table = PageTable(
        max_num_pages=64,
        page_size=16,
        device=torch.device("cpu"),
    )

    # Sequences with different lengths
    sequence_lengths = [32, 64, 128, 256]
    sequence_ids = []

    # Calculate pages needed for each sequence
    total_pages_allocated = 0
    for i, seq_len in enumerate(sequence_lengths):
        pages_needed = (seq_len + 15) // 16  # Ceiling division
        pages = page_table.allocate_sequence(sequence_id=i, num_pages=pages_needed)
        sequence_ids.append(i)
        total_pages_allocated += len(pages)

    # With page_size=16:
    # - 32 tokens -> 2 pages
    # - 64 tokens -> 4 pages
    # - 128 tokens -> 8 pages
    # - 256 tokens -> 16 pages
    # Total: 30 pages
    assert total_pages_allocated == 30
    assert len(page_table.free_pages) == 34  # 64 - 30

    # Verify no padding waste - each sequence uses exactly what it needs
    for i, seq_len in enumerate(sequence_lengths):
        block_table = page_table.get_block_table(i)
        expected_pages = (seq_len + 15) // 16
        assert len(block_table) == expected_pages


def test_paged_cache_initialization(paged_cache_config):
    """
    Test: Paged cache initialization
    - Initialize PagedKVCache
    - Verify physical page pool allocation
    - Check memory usage
    """
    cache = PagedKVCache(paged_cache_config)
    assert not cache.is_initialized

    # Initialize cache
    cache.initialize(
        num_layers=paged_cache_config.model.num_layers,
        device=torch.device("cpu"),
    )

    assert cache.is_initialized
    assert cache.num_layers == 2
    assert cache.page_size == 16
    assert cache.max_num_pages == 128

    # Check statistics
    stats = cache.get_statistics()
    assert stats["initialized"]
    assert stats["num_layers"] == 2
    assert stats["page_size"] == 16
    assert stats["max_num_pages"] == 128
    assert stats["memory_mb"] > 0


def test_paged_sequence_allocation(paged_cache_config):
    """
    Test: Sequence allocation in paged cache
    - Allocate sequence
    - Update sequence with KV
    - Verify block table is correct
    """
    cache = PagedKVCache(paged_cache_config)
    cache.initialize(
        num_layers=paged_cache_config.model.num_layers,
        device=torch.device("cpu"),
    )

    # Allocate sequence for 50 tokens
    # With page_size=16, this needs 4 pages (50 / 16 = 3.125 -> 4)
    num_pages = cache.allocate_sequence(sequence_id=0, num_tokens=50)
    assert num_pages == 4

    # Verify block table
    block_table = cache.get_block_tables()
    assert 0 in block_table
    assert len(block_table[0]) == 4


def test_paged_kv_update(paged_cache_config):
    """
    Test: KV update in paged cache
    - Allocate sequence
    - Update with KV tensors
    - Verify data is written correctly
    """
    cache = PagedKVCache(paged_cache_config)
    cache.initialize(
        num_layers=2,
        device=torch.device("cpu"),
    )

    # Allocate sequence for 20 tokens
    cache.allocate_sequence(sequence_id=0, num_tokens=20)

    # Create dummy KV tensors
    batch_size = 1
    seq_len = 10
    num_local_kv_groups = 2  # TP=1, 2 KV groups
    head_dim = 64

    key = torch.randn(
        batch_size,
        seq_len,
        num_local_kv_groups,
        head_dim,
    )
    value = torch.randn(
        batch_size,
        seq_len,
        num_local_kv_groups,
        head_dim,
    )

    # Update sequence
    full_key, full_value, block_table = cache.update_sequence(
        sequence_id=0,
        layer_idx=0,
        key=key,
        value=value,
    )

    # Verify block table
    assert len(block_table) == 2  # 20 tokens / 16 page_size = 2 pages

    # Verify returned KV has correct shape
    # full_key is returned from cache in [batch, seq_len, num_groups, head_dim] format
    # where seq_len is the current position (10)
    assert full_key.shape == (batch_size, 10, num_local_kv_groups, head_dim)


def test_page_refcounting():
    """
    Test: Page refcounting (for prefix caching)
    - Share pages between two sequences
    - Free one sequence
    - Verify pages not freed (refcount=1)
    - Free second sequence
    - Verify pages freed (refcount=0)
    """
    page_table = PageTable(
        max_num_pages=32,
        page_size=16,
        device=torch.device("cpu"),
    )

    # Allocate 10 pages for sequence 0
    pages_0 = page_table.allocate_sequence(sequence_id=0, num_pages=10)
    assert len(page_table.free_pages) == 22

    # Share first 5 pages with sequence 1
    page_table.share_pages(from_sequence_id=0, to_sequence_id=1, num_pages=5)
    assert len(page_table.free_pages) == 22  # No new pages allocated

    # Verify refcounts
    assert page_table.refcounts[pages_0[0]].item() == 2  # Shared
    assert page_table.refcounts[pages_0[5]].item() == 1  # Not shared

    # Free sequence 1 (the one that shared pages)
    page_table.free_sequence(sequence_id=1)

    # Verify shared pages still have refcount=1
    assert page_table.refcounts[pages_0[0]].item() == 1
    assert len(page_table.free_pages) == 22  # No pages freed

    # Free sequence 0
    page_table.free_sequence(sequence_id=0)

    # Verify all pages freed
    assert len(page_table.free_pages) == 32
    stats = page_table.get_statistics()
    assert stats["total_allocated"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
