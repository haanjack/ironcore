# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for BlockKVCacheManager and paged attention utilities.

"""
Tests for block-based paged KV cache:

1. Block allocation and freeing
2. Prefill write and read-back
3. Incremental decode writes
4. Prefix sharing via reference counting
5. Pool exhaustion
6. Block boundary handling
7. Gather correctness

Note: In real usage, write_prefill/write_decode are called from TransformerLayer.forward()
once per layer. Position is managed externally:
- write_prefill does NOT advance token_positions (all layers write to same logical positions)
- write_decode advances token_positions on layer_idx==0 (decode happens layer-by-layer
  in the transformer, but position only needs to advance once per token)
- Tests manually advance positions or use write_decode for decode steps.
"""

import pytest
import torch

from ironcore.layers.block_kv_cache import BlockKVCacheManager
from ironcore.layers.paged_attention import gather_kv_blocks, gather_kv_blocks_batched


def _make_paged_config(
    block_size: int = 4,
    max_batch_size: int = 8,
    max_seq_length: int = 32,
    num_layers: int = 2,
    num_kv_groups: int = 2,
    head_dim: int = 8,
) -> tuple[BlockKVCacheManager, int]:
    """Create a minimal BlockKVCacheManager for testing.

    Returns (manager, num_layers).
    """
    config = type("Config", (), {
        "model": type("Model", (), {
            "kv_cache": type("KV", (), {
                "block_size": block_size,
                "max_batch_size": max_batch_size,
                "max_seq_length": max_seq_length,
                "gpu_memory_utilization": 0.9,
            })(),
            "num_attention_groups": num_kv_groups,
            "head_dim": head_dim,
        })(),
        "trainer": type("Trainer", (), {"tensor_model_parallel_size": 1})(),
    })()

    original_compute = BlockKVCacheManager._compute_pool_size
    BlockKVCacheManager._compute_pool_size = lambda self, *a, **kw: 64

    mgr = BlockKVCacheManager(config)
    mgr.initialize(
        batch_size=max_batch_size,
        num_layers=num_layers,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    BlockKVCacheManager._compute_pool_size = original_compute
    return mgr, num_layers


class TestBlockAllocation:
    def test_allocate_single_block(self):
        mgr, _ = _make_paged_config()
        blocks = mgr.allocate_blocks(seq_id=0, count=1)
        assert len(blocks) == 1
        assert blocks[0] >= 0
        assert mgr.num_valid_blocks[0].item() == 1
        assert mgr.block_tables[0, 0].item() == blocks[0]

    def test_allocate_multiple_blocks(self):
        mgr, _ = _make_paged_config()
        blocks = mgr.allocate_blocks(seq_id=0, count=3)
        assert len(blocks) == 3
        assert mgr.num_valid_blocks[0].item() == 3
        for i, b in enumerate(blocks):
            assert mgr.block_tables[0, i].item() == b

    def test_allocate_separate_sequences(self):
        mgr, _ = _make_paged_config()
        blocks_0 = mgr.allocate_blocks(seq_id=0, count=2)
        blocks_1 = mgr.allocate_blocks(seq_id=1, count=2)
        assert set(blocks_0).isdisjoint(set(blocks_1))

    def test_free_returns_blocks_to_pool(self):
        mgr, _ = _make_paged_config()
        initial_free = len(mgr.free_blocks)
        mgr.allocate_blocks(seq_id=0, count=5)
        assert len(mgr.free_blocks) == initial_free - 5
        mgr.free_sequence(seq_id=0)
        assert len(mgr.free_blocks) == initial_free

    def test_free_clears_block_table(self):
        mgr, _ = _make_paged_config()
        mgr.allocate_blocks(seq_id=0, count=3)
        mgr.free_sequence(seq_id=0)
        assert mgr.num_valid_blocks[0].item() == 0
        assert (mgr.block_tables[0, :3] == -1).all()

    def test_allocate_zero_blocks(self):
        mgr, _ = _make_paged_config()
        blocks = mgr.allocate_blocks(seq_id=0, count=0)
        assert blocks == []

    def test_not_initialized_raises(self):
        mgr, _ = _make_paged_config()
        mgr.is_initialized = False
        with pytest.raises(RuntimeError, match="not initialized"):
            mgr.allocate_blocks(seq_id=0, count=1)


class TestPrefillWrite:
    def test_write_and_gather(self):
        block_size = 4
        mgr, num_layers = _make_paged_config(block_size=block_size)
        seq_len = 8  # 2 full blocks
        ng = mgr.num_local_kv_groups
        hd = mgr.head_dim

        key = torch.arange(seq_len, dtype=torch.float32).view(seq_len, 1, 1).expand(seq_len, ng, hd)
        value = key * 10.0

        mgr.allocate_blocks(seq_id=0, count=seq_len // block_size)
        for layer_idx in range(num_layers):
            mgr.write_prefill(layer_idx, seq_id=0, key=key, value=value)
        mgr.advance_position(seq_id=0, tokens=seq_len)

        for layer_idx in range(num_layers):
            gathered_key, gathered_value = mgr.get_layer_kv_gathered(layer_idx, seq_id=0)
            assert gathered_key.shape == (1, seq_len, ng, hd)
            torch.testing.assert_close(gathered_key.squeeze(0), key)
            torch.testing.assert_close(gathered_value.squeeze(0), value)

    def test_write_partial_block(self):
        block_size = 4
        mgr, num_layers = _make_paged_config(block_size=block_size)
        seq_len = 6  # 1 full block + 2 tokens

        key = torch.randn(seq_len, mgr.num_local_kv_groups, mgr.head_dim)
        value = torch.randn_like(key)

        blocks_needed = (seq_len + block_size - 1) // block_size
        mgr.allocate_blocks(seq_id=0, count=blocks_needed)
        for layer_idx in range(num_layers):
            mgr.write_prefill(layer_idx, seq_id=0, key=key, value=value)
        mgr.advance_position(seq_id=0, tokens=seq_len)

        for layer_idx in range(num_layers):
            gathered_key, _ = mgr.get_layer_kv_gathered(layer_idx, seq_id=0)
            assert gathered_key.shape == (1, seq_len, mgr.num_local_kv_groups, mgr.head_dim)
            torch.testing.assert_close(gathered_key.squeeze(0), key)

    def test_write_4d_input(self):
        mgr, num_layers = _make_paged_config(block_size=4)
        seq_len = 4
        ng = mgr.num_local_kv_groups
        hd = mgr.head_dim

        key = torch.randn(1, seq_len, ng, hd)

        mgr.allocate_blocks(seq_id=0, count=1)
        for layer_idx in range(num_layers):
            mgr.write_prefill(layer_idx, seq_id=0, key=key, value=key)
        mgr.advance_position(seq_id=0, tokens=seq_len)

        gathered_key, _ = mgr.get_layer_kv_gathered(0, seq_id=0)
        torch.testing.assert_close(gathered_key.squeeze(0), key.squeeze(0))


class TestDecodeWrite:
    def test_incremental_decode(self):
        block_size = 4
        mgr, num_layers = _make_paged_config(block_size=block_size)
        ng = mgr.num_local_kv_groups
        hd = mgr.head_dim

        # Prefill: 3 tokens (1 block)
        mgr.allocate_blocks(seq_id=0, count=1)
        prefill_key = torch.randn(3, ng, hd)
        for layer_idx in range(num_layers):
            mgr.write_prefill(layer_idx, seq_id=0, key=prefill_key, value=prefill_key)
        mgr.advance_position(seq_id=0, tokens=3)

        # Decode 5 more tokens one at a time
        decode_keys = []
        for t in range(5):
            k = torch.randn(1, ng, hd)
            decode_keys.append(k)
            for layer_idx in range(num_layers):
                mgr.write_decode(layer_idx, seq_id=0, key=k, value=k)
            mgr.advance_position(seq_id=0, tokens=1)

        # Total: 3 prefill + 5 decode = 8 tokens
        total_tokens = 8
        assert mgr.get_cache_position(0) == total_tokens

        for layer_idx in range(num_layers):
            gathered_key, _ = mgr.get_layer_kv_gathered(layer_idx, seq_id=0)
            assert gathered_key.shape == (1, total_tokens, ng, hd)
            torch.testing.assert_close(gathered_key[0, :3], prefill_key)
            for t in range(5):
                torch.testing.assert_close(gathered_key[0, 3 + t], decode_keys[t].squeeze(0))

    def test_decode_auto_allocates_on_boundary(self):
        block_size = 4
        mgr, num_layers = _make_paged_config(block_size=block_size)

        # Fill 1 block with 4 tokens
        mgr.allocate_blocks(seq_id=0, count=1)
        key = torch.randn(4, mgr.num_local_kv_groups, mgr.head_dim)
        for layer_idx in range(num_layers):
            mgr.write_prefill(layer_idx, seq_id=0, key=key, value=key)
        mgr.advance_position(seq_id=0, tokens=4)

        assert mgr.num_valid_blocks[0].item() == 1

        # Decode one more token — should auto-allocate a new block (layer 0 triggers it)
        new_key = torch.randn(1, mgr.num_local_kv_groups, mgr.head_dim)
        for layer_idx in range(num_layers):
            mgr.write_decode(layer_idx, seq_id=0, key=new_key, value=new_key)
        mgr.advance_position(seq_id=0, tokens=1)

        assert mgr.num_valid_blocks[0].item() == 2
        assert mgr.get_cache_position(0) == 5

    def test_decode_raises_for_multi_token(self):
        mgr, num_layers = _make_paged_config()
        mgr.allocate_blocks(seq_id=0, count=1)
        key = torch.randn(2, mgr.num_local_kv_groups, mgr.head_dim)
        with pytest.raises(AssertionError, match="single token"):
            mgr.write_decode(0, seq_id=0, key=key, value=key)


class TestPrefixSharing:
    def test_share_prefix_refcount(self):
        block_size = 4
        mgr, num_layers = _make_paged_config(block_size=block_size)

        key = torch.randn(8, mgr.num_local_kv_groups, mgr.head_dim)
        mgr.allocate_blocks(seq_id=0, count=2)
        for layer_idx in range(num_layers):
            mgr.write_prefill(layer_idx, seq_id=0, key=key, value=key)
        mgr.advance_position(seq_id=0, tokens=8)

        src_blocks = mgr.block_tables[0, :2].clone()
        mgr.share_prefix(src_seq_id=0, dst_seq_ids=[1, 2])

        assert (mgr.block_tables[1, :2] == src_blocks).all()
        assert (mgr.block_tables[2, :2] == src_blocks).all()

        # Ref count: src(1) + index_add for each dst share call (1 each) = 1 + 1 + 1 = 3
        for b in src_blocks.tolist():
            assert mgr.ref_counts[b].item() == 3

    def test_shared_blocks_persist_after_src_free(self):
        block_size = 4
        mgr, num_layers = _make_paged_config(block_size=block_size)

        key = torch.randn(8, mgr.num_local_kv_groups, mgr.head_dim)
        mgr.allocate_blocks(seq_id=0, count=2)
        for layer_idx in range(num_layers):
            mgr.write_prefill(layer_idx, seq_id=0, key=key, value=key)
        mgr.advance_position(seq_id=0, tokens=8)

        src_blocks = mgr.block_tables[0, :2].clone()
        mgr.share_prefix(src_seq_id=0, dst_seq_ids=[1])

        mgr.free_sequence(seq_id=0)
        for b in src_blocks.tolist():
            assert mgr.ref_counts[b].item() == 1

        gathered_key, _ = mgr.get_layer_kv_gathered(0, seq_id=1)
        torch.testing.assert_close(gathered_key.squeeze(0), key)

    def test_shared_blocks_freed_when_all_seqs_free(self):
        block_size = 4
        mgr, num_layers = _make_paged_config(block_size=block_size)

        key = torch.randn(8, mgr.num_local_kv_groups, mgr.head_dim)
        mgr.allocate_blocks(seq_id=0, count=2)
        for layer_idx in range(num_layers):
            mgr.write_prefill(layer_idx, seq_id=0, key=key, value=key)
        mgr.advance_position(seq_id=0, tokens=8)

        initial_free = len(mgr.free_blocks)
        mgr.share_prefix(src_seq_id=0, dst_seq_ids=[1])

        mgr.free_sequence(seq_id=0)
        assert len(mgr.free_blocks) == initial_free

        mgr.free_sequence(seq_id=1)
        assert len(mgr.free_blocks) == initial_free + 2

    def test_dst_can_decode_after_share(self):
        block_size = 4
        mgr, num_layers = _make_paged_config(block_size=block_size)

        prefill = torch.randn(8, mgr.num_local_kv_groups, mgr.head_dim)
        mgr.allocate_blocks(seq_id=0, count=2)
        for layer_idx in range(num_layers):
            mgr.write_prefill(layer_idx, seq_id=0, key=prefill, value=prefill)
        mgr.advance_position(seq_id=0, tokens=8)

        mgr.share_prefix(src_seq_id=0, dst_seq_ids=[1])

        # Decode on seq 1 — prefix blocks are fully shared, decode writes to new blocks
        decode_key = torch.randn(1, mgr.num_local_kv_groups, mgr.head_dim)
        for layer_idx in range(num_layers):
            mgr.write_decode(layer_idx, seq_id=1, key=decode_key, value=decode_key)
        mgr.advance_position(seq_id=1, tokens=1)

        assert mgr.num_valid_blocks[1].item() == 3
        assert mgr.get_cache_position(1) == 9

        gathered_key, _ = mgr.get_layer_kv_gathered(0, seq_id=1)
        assert gathered_key.shape == (1, 9, mgr.num_local_kv_groups, mgr.head_dim)
        torch.testing.assert_close(gathered_key[0, :8], prefill)
        torch.testing.assert_close(gathered_key[0, 8], decode_key.squeeze(0))


class TestPoolExhaustion:
    def test_exhaustion_raises_error(self):
        config = type("Config", (), {
            "model": type("Model", (), {
                "kv_cache": type("KV", (), {
                    "block_size": 4,
                    "max_batch_size": 4,
                    "max_seq_length": 32,
                    "gpu_memory_utilization": 0.9,
                })(),
                "num_attention_groups": 2,
                "head_dim": 8,
            })(),
            "trainer": type("Trainer", (), {"tensor_model_parallel_size": 1})(),
        })()

        original_compute = BlockKVCacheManager._compute_pool_size
        BlockKVCacheManager._compute_pool_size = lambda self, *a, **kw: 8
        mgr = BlockKVCacheManager(config)
        mgr.initialize(batch_size=4, num_layers=1, device=torch.device("cpu"), dtype=torch.float32)
        BlockKVCacheManager._compute_pool_size = original_compute

        mgr.allocate_blocks(seq_id=0, count=8)

        with pytest.raises(RuntimeError, match="exhausted"):
            mgr.allocate_blocks(seq_id=1, count=1)


class TestStatistics:
    def test_stats_after_init(self):
        mgr, _ = _make_paged_config()
        stats = mgr.get_statistics()
        assert stats["initialized"]
        assert stats["type"] == "paged"
        assert stats["block_size"] == 4
        assert stats["num_physical_blocks"] == 64
        assert stats["num_free_blocks"] == 64
        assert stats["num_used_blocks"] == 0
        assert stats["utilization"] == 0.0
        assert stats["memory_mb"] > 0

    def test_stats_after_allocation(self):
        mgr, _ = _make_paged_config()
        mgr.allocate_blocks(seq_id=0, count=5)
        stats = mgr.get_statistics()
        assert stats["num_used_blocks"] == 5
        assert stats["num_free_blocks"] == 59
        assert abs(stats["utilization"] - 5 / 64) < 1e-6


class TestGatherKVBlocks:
    def test_gather_full_blocks(self):
        block_size = 4
        pool_size = 8
        cache = torch.randn(pool_size, block_size, 2, 8)
        block_table = torch.tensor([3, 1, 5, -1, -1, -1, -1, -1])
        num_valid = 3
        total_tokens = 12

        result = gather_kv_blocks(cache, block_table, num_valid, total_tokens, block_size)
        assert result.shape == (1, 12, 2, 8)
        expected = torch.cat([cache[3], cache[1], cache[5]], dim=0).unsqueeze(0)
        torch.testing.assert_close(result, expected)

    def test_gather_partial_last_block(self):
        block_size = 4
        pool_size = 8
        cache = torch.randn(pool_size, block_size, 2, 8)
        block_table = torch.tensor([3, 1, -1, -1, -1, -1, -1, -1])
        num_valid = 2
        total_tokens = 6

        result = gather_kv_blocks(cache, block_table, num_valid, total_tokens, block_size)
        assert result.shape == (1, 6, 2, 8)
        expected = torch.cat([cache[3], cache[1, :2]], dim=0).unsqueeze(0)
        torch.testing.assert_close(result, expected)

    def test_gather_single_partial_block(self):
        block_size = 4
        pool_size = 4
        cache = torch.randn(pool_size, block_size, 2, 8)
        block_table = torch.tensor([2, -1, -1, -1])
        num_valid = 1
        total_tokens = 3

        result = gather_kv_blocks(cache, block_table, num_valid, total_tokens, block_size)
        assert result.shape == (1, 3, 2, 8)
        torch.testing.assert_close(result.squeeze(0), cache[2, :3])

    def test_gather_zero_tokens(self):
        pool_size = 4
        cache = torch.randn(pool_size, 4, 2, 8)
        block_table = torch.tensor([-1, -1, -1, -1])
        result = gather_kv_blocks(cache, block_table, 0, 0, 4)
        assert result.shape == (1, 0, 2, 8)

    def test_gather_non_sequential_physical_blocks(self):
        """Verify correct ordering when physical blocks are non-sequential."""
        block_size = 4
        pool_size = 16
        cache = torch.zeros(pool_size, block_size, 1, 1)
        cache[7] = 1.0
        cache[0] = 2.0
        cache[15] = 3.0

        block_table = torch.tensor([7, 0, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        num_valid = 3
        total_tokens = 12

        result = gather_kv_blocks(cache, block_table, num_valid, total_tokens, block_size)
        expected = torch.tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]],
                                  [[2.0]], [[2.0]], [[2.0]], [[2.0]],
                                  [[3.0]], [[3.0]], [[3.0]], [[3.0]]]])
        torch.testing.assert_close(result, expected)


class TestGatherKVBlocksBatched:
    def test_batched_full_blocks(self):
        block_size = 4
        pool_size = 16
        ng, hd = 2, 8
        cache = torch.randn(pool_size, block_size, ng, hd)
        block_tables = torch.full((8, 8), -1, dtype=torch.long)
        block_tables[0, :2] = torch.tensor([3, 1])
        block_tables[1, :3] = torch.tensor([5, 0, 7])

        result = gather_kv_blocks_batched(
            cache, block_tables, seq_ids=[0, 1],
            num_valid_blocks=[2, 3], token_positions=[8, 12], block_size=block_size,
        )
        assert result.shape == (2, 12, ng, hd)
        expected_0 = torch.cat([cache[3], cache[1]], dim=0)
        torch.testing.assert_close(result[0, :8], expected_0)
        expected_1 = torch.cat([cache[5], cache[0], cache[7]], dim=0)
        torch.testing.assert_close(result[1], expected_1)

    def test_batched_partial_last_block(self):
        block_size = 4
        pool_size = 16
        ng, hd = 2, 8
        cache = torch.randn(pool_size, block_size, ng, hd)
        block_tables = torch.full((8, 8), -1, dtype=torch.long)
        block_tables[0, :2] = torch.tensor([3, 1])
        block_tables[1, :1] = torch.tensor([5])

        result = gather_kv_blocks_batched(
            cache, block_tables, seq_ids=[0, 1],
            num_valid_blocks=[2, 1], token_positions=[6, 3], block_size=block_size,
        )
        assert result.shape == (2, 6, ng, hd)
        expected_0 = torch.cat([cache[3], cache[1, :2]], dim=0)
        torch.testing.assert_close(result[0, :6], expected_0)
        expected_1 = cache[5, :3]
        torch.testing.assert_close(result[1, :3], expected_1)
        assert (result[1, 3:] == 0).all()

    def test_batched_empty_seqs(self):
        pool_size = 8
        cache = torch.randn(pool_size, 4, 2, 8)
        block_tables = torch.full((4, 4), -1, dtype=torch.long)

        result = gather_kv_blocks_batched(
            cache, block_tables, seq_ids=[], num_valid_blocks=[], token_positions=[], block_size=4,
        )
        assert result.shape == (0, 0, 2, 8)

    def test_batched_matches_single_gather(self):
        """Verify batched gather produces identical results to per-sequence gather_kv_blocks."""
        block_size = 4
        pool_size = 16
        ng, hd = 2, 8
        cache = torch.randn(pool_size, block_size, ng, hd)
        max_blocks = 8
        block_tables = torch.full((8, max_blocks), -1, dtype=torch.long)

        seq_ids = [0, 2, 5]
        num_valid = [3, 1, 2]
        token_positions = [10, 3, 7]
        for i, sid in enumerate(seq_ids):
            for j in range(num_valid[i]):
                block_tables[sid, j] = (i * 5 + j) % pool_size

        result = gather_kv_blocks_batched(
            cache, block_tables, seq_ids=seq_ids,
            num_valid_blocks=num_valid, token_positions=token_positions, block_size=block_size,
        )

        max_len = max(token_positions)
        assert result.shape == (3, max_len, ng, hd)

        for idx, sid in enumerate(seq_ids):
            single = gather_kv_blocks(
                cache, block_tables[sid], num_valid[idx], token_positions[idx], block_size,
            )
            torch.testing.assert_close(
                result[idx, :token_positions[idx]], single.squeeze(0)
            )


class TestBatchedDecodeWrite:
    def test_batched_decode_write(self):
        block_size = 4
        mgr, num_layers = _make_paged_config(block_size=block_size)
        ng = mgr.num_local_kv_groups
        hd = mgr.head_dim

        key0 = torch.randn(4, ng, hd)
        key1 = torch.randn(4, ng, hd)

        mgr.allocate_blocks(seq_id=0, count=1)
        mgr.allocate_blocks(seq_id=1, count=1)
        for layer_idx in range(num_layers):
            mgr.write_prefill(layer_idx, seq_id=0, key=key0, value=key0)
            mgr.write_prefill(layer_idx, seq_id=1, key=key1, value=key1)
        mgr.advance_position(seq_id=0, tokens=4)
        mgr.advance_position(seq_id=1, tokens=4)

        # Batched decode: write 1 token to both seqs
        dk0 = torch.randn(1, ng, hd)
        dk1 = torch.randn(1, ng, hd)
        batched_key = torch.cat([dk0, dk1], dim=0).unsqueeze(1)  # [2, 1, ng, hd]

        for layer_idx in range(num_layers):
            mgr.write_decode_batched(layer_idx, seq_ids=[0, 1], key=batched_key, value=batched_key)
        mgr.advance_positions_batched([0, 1], 1)

        assert mgr.get_cache_position(0) == 5
        assert mgr.get_cache_position(1) == 5

        # Verify gathered KV includes decode tokens
        for sid, prefill_key, decode_key in [(0, key0, dk0), (1, key1, dk1)]:
            gathered, _ = mgr.get_layer_kv_gathered(0, sid)
            assert gathered.shape == (1, 5, ng, hd)
            torch.testing.assert_close(gathered[0, :4], prefill_key)
            torch.testing.assert_close(gathered[0, 4], decode_key.squeeze(0))

    def test_batched_gather_matches_single(self):
        """get_layer_kv_gathered_batched must match per-sequence gather."""
        block_size = 4
        mgr, num_layers = _make_paged_config(block_size=block_size)
        ng = mgr.num_local_kv_groups
        hd = mgr.head_dim

        # Set up 3 sequences with different lengths
        seq_lengths = [6, 4, 10]
        for i, slen in enumerate(seq_lengths):
            key = torch.randn(slen, ng, hd)
            blocks_needed = (slen + block_size - 1) // block_size
            mgr.allocate_blocks(seq_id=i, count=blocks_needed)
            for layer_idx in range(num_layers):
                mgr.write_prefill(layer_idx, seq_id=i, key=key, value=key)
            mgr.advance_position(seq_id=i, tokens=slen)

        seq_ids = [0, 1, 2]
        for layer_idx in range(num_layers):
            batched_key, batched_value = mgr.get_layer_kv_gathered_batched(layer_idx, seq_ids)
            assert batched_key.shape[0] == 3
            assert batched_key.shape[1] == max(seq_lengths)

            for idx, sid in enumerate(seq_ids):
                single_key, single_value = mgr.get_layer_kv_gathered(layer_idx, sid)
                torch.testing.assert_close(batched_key[idx, :seq_lengths[idx]], single_key.squeeze(0))
                torch.testing.assert_close(batched_value[idx, :seq_lengths[idx]], single_value.squeeze(0))

    def test_advance_positions_batched(self):
        mgr, _ = _make_paged_config()
        mgr.allocate_blocks(seq_id=0, count=1)
        mgr.allocate_blocks(seq_id=1, count=1)
        mgr.allocate_blocks(seq_id=2, count=1)

        mgr.advance_positions_batched([0, 2], 5)

        assert mgr.get_cache_position(0) == 5
        assert mgr.get_cache_position(1) == 0  # untouched
        assert mgr.get_cache_position(2) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
