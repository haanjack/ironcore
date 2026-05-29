# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Block-based paged KV cache for efficient GRPO rollouts.
# Inspired by vLLM's PagedAttention (arxiv 2309.06180).

"""
Block-based paged KV cache manager.

Key concepts:
- Physical blocks: Fixed-size KV storage blocks in a pool (analogous to physical memory frames)
- Block tables: Per-sequence mapping from logical block index to physical block index (page table)
- Reference counting: Shared prefix blocks are referenced by multiple sequences; freed only when ref==0
- On-demand allocation: Blocks allocated as tokens arrive, not pre-allocated per max_seq_len

Cache layout per layer:
    physical_key_caches:   [num_physical_blocks, block_size, num_local_kv_groups, head_dim]
    physical_value_caches: [num_physical_blocks, block_size, num_local_kv_groups, head_dim]
    block_tables:          [max_batch_size, max_num_blocks_per_seq]   (int64, -1 = unallocated)
"""

from __future__ import annotations

import torch

from ironcore.config import MainConfig
from ironcore.layers.kv_cache_utils import compute_memory_mb, compute_utilization
from ironcore.utils import get_model_dtype

_CPU_FALLBACK_CACHE_BYTES = 4 * 1024**3  # 4 GB — generous budget for CPU-only testing


class BlockKVCacheManager:
    """Manages block-based paged KV cache for autoregressive generation.

    Memory is organized as a pool of fixed-size physical blocks.
    Each sequence owns a block table mapping logical positions to physical blocks.
    Prefix sharing between sequences is achieved by copying block table entries
    and incrementing reference counts — no KV tensor duplication.
    """

    def __init__(self, config: MainConfig):
        self.config = config
        self.model_config = config.model
        self.cache_config = config.model.kv_cache
        self.block_size = self.cache_config.block_size

        # TP-aware: each rank stores only its local KV groups
        self.num_local_kv_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )
        self.head_dim = config.model.head_dim

        # Storage (allocated in initialize())
        self.physical_key_caches: list[torch.Tensor] = []
        self.physical_value_caches: list[torch.Tensor] = []
        self.block_tables: torch.Tensor | None = None  # [max_batch_size, max_num_blocks_per_seq]
        self.num_valid_blocks: torch.Tensor | None = None  # [max_batch_size]
        self.token_positions: torch.Tensor | None = (
            None  # [max_batch_size] — externally managed positions
        )
        self.tokens_written: torch.Tensor | None = (
            None  # [max_batch_size] — actual tokens in cache (updated by _write_kv_to_blocks)
        )

        # Pool management
        self.free_blocks: list[int] = []
        self.ref_counts: torch.Tensor | None = None  # [num_physical_blocks]

        self.num_physical_blocks = 0
        self.max_num_blocks_per_seq = 0
        self.is_initialized = False
        self.device = None
        self.dtype = None

    def initialize(
        self,
        batch_size: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ):
        """Allocate the physical block pool and per-sequence block tables.

        Args:
            batch_size: Maximum number of concurrent sequences.
            num_layers: Number of transformer layers.
            device: Device to allocate on.
            dtype: Data type for cache (defaults to model dtype).
        """
        if self.is_initialized:
            # Re-initialize if batch_size changed (e.g., different prompt count per step).
            if batch_size != self.block_tables.shape[0]:
                self._deallocate()
            else:
                # Same size — just reset state.
                for sid in range(self.block_tables.shape[0]):
                    self.free_sequence(sid)
                self.block_tables.fill_(-1)
                self.num_valid_blocks.zero_()
                self.token_positions.zero_()
                self.tokens_written.zero_()
                return
        if dtype is None:
            dtype = get_model_dtype(self.config)

        self.device = device
        self.dtype = dtype

        max_seq_len = self.cache_config.max_seq_length
        self.max_num_blocks_per_seq = (max_seq_len + self.block_size - 1) // self.block_size

        # Calculate pool size from GPU memory
        gpu_util = self.cache_config.gpu_memory_utilization
        self.num_physical_blocks = self._compute_pool_size(num_layers, batch_size, gpu_util, device)

        bs = self.block_size
        ng = self.num_local_kv_groups
        hd = self.head_dim

        # Allocate physical block pool per layer
        self.physical_key_caches = []
        self.physical_value_caches = []
        for _ in range(num_layers):
            self.physical_key_caches.append(
                torch.zeros(self.num_physical_blocks, bs, ng, hd, device=device, dtype=dtype)
            )
            self.physical_value_caches.append(
                torch.zeros(self.num_physical_blocks, bs, ng, hd, device=device, dtype=dtype)
            )

        # Block tables: -1 means unallocated
        self.block_tables = torch.full(
            (batch_size, self.max_num_blocks_per_seq),
            -1,
            dtype=torch.long,
            device=device,
        )
        # Number of valid (allocated) logical blocks per sequence
        self.num_valid_blocks = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Actual number of tokens written per sequence (may be < num_valid_blocks * block_size)
        self.token_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

        # True count of tokens written to cache (updated atomically by _write_kv_to_blocks).
        # This tracks writes even when token_positions hasn't been externally advanced yet.
        self.tokens_written = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Reference counts for shared blocks
        self.ref_counts = torch.zeros(self.num_physical_blocks, dtype=torch.long, device=device)

        # Free block stack (all blocks initially free)
        self.free_blocks = list(range(self.num_physical_blocks - 1, -1, -1))

        self.is_initialized = True

    def _deallocate(self):
        """Free all tensors and reset to uninitialized state."""
        if not self.is_initialized:
            return
        del self.physical_key_caches
        del self.physical_value_caches
        del self.block_tables
        del self.num_valid_blocks
        del self.token_positions
        del self.tokens_written
        del self.ref_counts
        self.free_blocks = []
        self.is_initialized = False

    def _compute_pool_size(
        self,
        num_layers: int,
        batch_size: int,
        gpu_util: float,
        device: torch.device,
    ) -> int:
        """Calculate number of physical blocks from available GPU memory.

        Accounts for the model's own memory usage by querying current allocation.
        """
        # Bytes per block per layer (key + value)
        bytes_per_block_per_layer = (
            self.block_size * self.num_local_kv_groups * self.head_dim * 2  # K + V
        )
        if self.dtype is not None:
            bytes_per_block_per_layer *= self.dtype.itemsize
        else:
            bytes_per_block_per_layer *= 2  # default bf16

        total_per_block = bytes_per_block_per_layer * num_layers

        # Available memory for cache
        if device.type == "cuda":
            total_mem = torch.cuda.get_device_properties(device).total_memory
            # Subtract both allocated and reserved memory, plus a 10% safety
            # margin for fragmentation and other process overhead.
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            available = (total_mem - max(allocated, reserved)) * gpu_util * 0.9
        else:
            available = _CPU_FALLBACK_CACHE_BYTES

        # Reserve minimum blocks: each sequence needs at least a few blocks
        min_blocks = batch_size * 2

        num_blocks = max(min_blocks, int(available // total_per_block))

        # Cap at max_num_blocks_per_seq * batch_size as a sanity upper bound
        max_blocks = self.max_num_blocks_per_seq * batch_size
        return min(num_blocks, max_blocks)

    def _allocate_single_block(self, seq_id: int) -> int:
        """Allocate one block and return its physical index."""
        blocks = self.allocate_blocks(seq_id, 1)
        return blocks[0]

    def allocate_blocks(self, seq_id: int, count: int) -> list[int]:
        """Allocate `count` physical blocks for a sequence.

        Args:
            seq_id: Sequence identifier (row index in block_tables).
            count: Number of blocks to allocate.

        Returns:
            List of allocated physical block indices.

        Raises:
            RuntimeError: If not enough free blocks available.
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        if count <= 0:
            return []

        if len(self.free_blocks) < count:
            raise RuntimeError(
                f"Block pool exhausted: need {count} blocks, "
                f"but only {len(self.free_blocks)} free (total={self.num_physical_blocks}). "
                f"Try reducing batch_size, increasing gpu_memory_utilization, or using CPU offloading."
            )

        current_valid = self.num_valid_blocks[seq_id]
        if current_valid + count > self.max_num_blocks_per_seq:
            raise RuntimeError(
                f"Sequence {seq_id} exceeds max blocks: "
                f"{current_valid + count} > {self.max_num_blocks_per_seq}"
            )

        allocated = []
        for _ in range(count):
            block_idx = self.free_blocks.pop()
            self.block_tables[seq_id, current_valid + len(allocated)] = block_idx
            self.ref_counts[block_idx] = 1
            allocated.append(block_idx)

        self.num_valid_blocks[seq_id] = current_valid + count
        return allocated

    def write_prefill(
        self,
        layer_idx: int,
        seq_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """Write prefill KV into allocated blocks for a single layer.

        The caller must have already allocated sufficient blocks via `allocate_blocks()`.
        tokens_written is updated on layer_idx==0. token_positions is NOT advanced here;
        the caller must explicitly call advance_position() after prefill completes.

        Args:
            layer_idx: Transformer layer index.
            seq_id: Sequence identifier.
            key: [1, seq_len, num_local_kv_groups, head_dim] (or [seq_len, ng, hd])
            value: Same shape as key.
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized")

        if key.dim() == 4:
            key = key.squeeze(0)
        if value.dim() == 4:
            value = value.squeeze(0)

        seq_len = key.shape[0]
        num_blocks_needed = (seq_len + self.block_size - 1) // self.block_size
        current_blocks = self.num_valid_blocks[seq_id]

        if current_blocks < num_blocks_needed:
            raise RuntimeError(
                f"Not enough blocks allocated for seq {seq_id} layer {layer_idx}: "
                f"need {num_blocks_needed}, have {current_blocks}. Call allocate_blocks() first."
            )

        self._write_kv_to_blocks(layer_idx, seq_id, key, value)

    def write_decode(
        self,
        layer_idx: int,
        seq_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """Write a single decode token's KV into the cache.

        Automatically allocates a new block when the current block is full.

        Args:
            layer_idx: Transformer layer index.
            seq_id: Sequence identifier.
            key: [1, 1, num_local_kv_groups, head_dim] (or [1, ng, hd])
            value: Same shape as key.
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized")

        if key.dim() == 4:
            key = key.squeeze(0)
        if value.dim() == 4:
            value = value.squeeze(0)

        seq_len = key.shape[0]
        assert seq_len == 1, f"write_decode expects single token, got seq_len={seq_len}"

        total_tokens = self.token_positions[seq_id]
        logical_block_idx = total_tokens // self.block_size

        # Allocate new block if current position exceeds allocated blocks
        if layer_idx == 0 and logical_block_idx >= self.num_valid_blocks[seq_id]:
            self.allocate_blocks(seq_id, 1)

        self._write_kv_to_blocks(layer_idx, seq_id, key, value)

    def advance_position(self, seq_id: int, tokens: int):
        if not self.is_initialized:
            return
        self.token_positions[seq_id] += tokens

    def advance_positions_batched(self, seq_ids: list[int] | torch.Tensor, tokens: int):
        """Advance token positions for multiple sequences."""
        if not self.is_initialized:
            return
        self.token_positions[seq_ids] += tokens

    def write_decode_batched(
        self,
        layer_idx: int,
        seq_ids: list[int] | torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """Batched single-token decode write for multiple sequences.

        Args:
            layer_idx: Transformer layer index.
            seq_ids: Sequence IDs (list or 1D tensor).
            key: [batch, 1, num_local_kv_groups, head_dim]
            value: Same shape as key.
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized")

        if key.dim() == 4:
            assert key.shape[1] == 1, (
                f"write_decode_batched expects single token, got seq_len={key.shape[1]}"
            )
            key = key.squeeze(1)
        if value.dim() == 4:
            value = value.squeeze(1)

        if isinstance(seq_ids, torch.Tensor):
            seq_ids = seq_ids.tolist()

        # Vectorized block allocation check
        positions = self.token_positions[seq_ids]
        logical_block_idxs = positions // self.block_size
        num_valid = self.num_valid_blocks[seq_ids]
        needs_alloc = logical_block_idxs >= num_valid

        if layer_idx == 0:
            for i, sid in enumerate(seq_ids):
                if needs_alloc[i]:
                    self.allocate_blocks(sid, 1)

        # Vectorized write: scatter each seq's token into its block
        for i, sid in enumerate(seq_ids):
            self._write_kv_to_blocks(layer_idx, sid, key[i : i + 1], value[i : i + 1])

    def get_layer_kv_gathered_batched(
        self,
        layer_idx: int,
        seq_ids: list[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather KV for multiple sequences into padded batched tensor.

        Uses vectorized gather_kv_blocks_batched to avoid per-sequence Python loops.

        Args:
            layer_idx: Transformer layer index.
            seq_ids: Sequence IDs.

        Returns:
            (key, value) with shape [batch, max_seq_len, num_local_kv_groups, head_dim]
        """
        from ironcore.layers.paged_attention import gather_kv_blocks_batched

        if not self.is_initialized:
            raise RuntimeError("Cache not initialized")

        if isinstance(seq_ids, torch.Tensor):
            seq_ids = seq_ids.tolist()

        seq_id_t = torch.tensor(seq_ids, dtype=torch.long, device=self.device)
        num_valid_list = self.num_valid_blocks[seq_id_t].tolist()
        tokens_written_list = self.tokens_written[seq_id_t].tolist()

        for i, tw in enumerate(tokens_written_list):
            assert tw > 0, (
                f"tokens_written[{seq_ids[i]}] == 0 in get_layer_kv_gathered_batched — "
                f"gather called before any write completed for this sequence"
            )

        token_pos_list = [int(tw) for tw in tokens_written_list]

        key = gather_kv_blocks_batched(
            self.physical_key_caches[layer_idx],
            self.block_tables,
            seq_ids,
            num_valid_list,
            token_pos_list,
            self.block_size,
        )
        value = gather_kv_blocks_batched(
            self.physical_value_caches[layer_idx],
            self.block_tables,
            seq_ids,
            num_valid_list,
            token_pos_list,
            self.block_size,
        )

        return key, value

    def _write_kv_to_blocks(
        self,
        layer_idx: int,
        seq_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """Write KV tensors into the physical block pool.

        Writes to positions tracked by token_positions. Does NOT advance
        token_positions — callers (write_decode) handle that explicitly.
        """
        key_cache = self.physical_key_caches[layer_idx]
        value_cache = self.physical_value_caches[layer_idx]

        start_token = self.token_positions[seq_id]
        total_new_tokens = key.shape[0]
        num_valid = self.num_valid_blocks[seq_id]

        token_offset = 0
        while token_offset < total_new_tokens:
            logical_block_idx = (start_token + token_offset) // self.block_size
            pos_in_block = (start_token + token_offset) % self.block_size

            if logical_block_idx >= num_valid:
                raise RuntimeError(
                    f"Block underflow: logical_block_idx={logical_block_idx} "
                    f"but only {num_valid} blocks allocated for seq {seq_id}"
                )

            physical_block_idx = self.block_tables[seq_id, logical_block_idx]

            remaining_in_block = self.block_size - pos_in_block
            tokens_to_write = min(remaining_in_block, total_new_tokens - token_offset)

            key_end = token_offset + tokens_to_write
            key_cache[physical_block_idx, pos_in_block : pos_in_block + tokens_to_write] = key[
                token_offset:key_end
            ]
            value_cache[physical_block_idx, pos_in_block : pos_in_block + tokens_to_write] = value[
                token_offset:key_end
            ]

            token_offset = key_end

        # Track actual tokens written (only on layer 0 to avoid double-counting)
        if layer_idx == 0:
            self.tokens_written[seq_id] = start_token + total_new_tokens

    def share_prefix(self, src_seq_id: int, dst_seq_ids: list[int]):
        """Share prefix blocks from source to destinations via reference counting.

        Copies the block table entries and increments reference counts.
        No KV tensor data is copied — this is O(num_prefix_blocks) metadata ops.

        Args:
            src_seq_id: Source sequence whose prefix blocks to share.
            dst_seq_ids: Destination sequence IDs to share with.
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized")

        src_num_blocks = self.num_valid_blocks[src_seq_id].item()

        for dst_id in dst_seq_ids:
            if self.num_valid_blocks[dst_id].item() > 0:
                raise RuntimeError(
                    f"Cannot share prefix to seq_id={dst_id}: already has "
                    f"{self.num_valid_blocks[dst_id].item()} blocks allocated. "
                    f"Free the sequence first."
                )

        # Check if last prefix block is partial — if so, it needs COW.
        # When decode writes to a partial block, it would corrupt the shared
        # physical block for all sequences sharing that prefix.
        src_tokens = self.tokens_written[src_seq_id].item()
        last_block_has_room = (src_tokens % self.block_size) != 0

        for dst_id in dst_seq_ids:
            if last_block_has_room:
                # Allocate a fresh block and copy the partial block's KV data
                new_block = self._allocate_single_block(dst_id)
                last_idx = src_num_blocks - 1
                old_block = self.block_tables[src_seq_id, last_idx].item()
                # Copy first: block_tables, then KV data
                self.block_tables[dst_id, : src_num_blocks - 1] = self.block_tables[
                    src_seq_id, : src_num_blocks - 1
                ]
                self.block_tables[dst_id, last_idx] = new_block
                # Deep-copy KV from old block to new block
                for layer_k, layer_v in zip(
                    self.physical_key_caches, self.physical_value_caches, strict=True
                ):
                    layer_k[new_block] = layer_k[old_block].clone()
                    layer_v[new_block] = layer_v[old_block].clone()
            else:
                # All blocks full — safe to share directly
                self.block_tables[dst_id, :src_num_blocks] = self.block_tables[
                    src_seq_id, :src_num_blocks
                ]

            self.num_valid_blocks[dst_id] = src_num_blocks
            self.token_positions[dst_id] = self.token_positions[src_seq_id]
            self.tokens_written[dst_id] = self.tokens_written[src_seq_id]

        # Increment reference counts for shared blocks.
        # With COW, only the full blocks are shared; the partial last block
        # was deep-copied and is individually owned by each dst.
        if last_block_has_room:
            shared_count = src_num_blocks - 1
        else:
            shared_count = src_num_blocks

        if shared_count > 0:
            src_block_indices = self.block_tables[src_seq_id, :shared_count]
            increment = torch.full_like(src_block_indices, len(dst_seq_ids))
            self.ref_counts.index_add_(0, src_block_indices, increment)

    def free_sequence(self, seq_id: int):
        """Free all blocks owned by a sequence.

        Decrements reference counts; blocks with ref==0 are returned to the free pool.

        Args:
            seq_id: Sequence identifier to free.
        """
        if not self.is_initialized:
            return

        num_valid = self.num_valid_blocks[seq_id].item()
        if num_valid == 0:
            return

        block_indices = self.block_tables[seq_id, :num_valid].clone()

        # Decrement reference counts
        self.ref_counts.index_add_(0, block_indices, -torch.ones_like(block_indices))

        # Guard: ref counts must never go negative (indicates double-free or ref bug)
        neg_mask = self.ref_counts[block_indices] < 0
        if neg_mask.any():
            bad = block_indices[neg_mask].tolist()
            self.ref_counts.clamp_min_(0)
            raise RuntimeError(
                f"ref_counts went negative for blocks {bad} on free_sequence({seq_id}). "
                f"This indicates a reference counting bug."
            )

        # Collect blocks whose ref count dropped to 0
        zero_mask = self.ref_counts[block_indices] == 0
        freed_indices = block_indices[zero_mask]

        # Return freed blocks to pool
        for idx in freed_indices.tolist():
            self.free_blocks.append(idx)

        # Clear block table entries
        self.block_tables[seq_id, :num_valid] = -1
        self.num_valid_blocks[seq_id] = 0
        self.token_positions[seq_id] = 0
        self.tokens_written[seq_id] = 0

    def get_layer_kv_gathered(
        self, layer_idx: int, seq_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather a sequence's KV blocks into contiguous tensors.

        Compatible with existing Attention methods that expect
        [batch, seq_len, num_groups, head_dim] layout.

        Args:
            layer_idx: Transformer layer index.
            seq_id: Sequence identifier.

        Returns:
            (key, value) with shape [1, total_tokens, num_local_kv_groups, head_dim]
        """
        if not self.is_initialized:
            raise RuntimeError("Cache not initialized")

        total_tokens = self.tokens_written[seq_id].item()

        # Fall back to token_positions if tokens_written not yet set
        if total_tokens == 0:
            total_tokens = self.token_positions[seq_id].item()

        if total_tokens == 0:
            ng, hd = self.num_local_kv_groups, self.head_dim
            return (
                torch.zeros(1, 0, ng, hd, device=self.device, dtype=self.dtype),
                torch.zeros(1, 0, ng, hd, device=self.device, dtype=self.dtype),
            )

        # Number of fully filled blocks and remainder
        num_full_blocks = total_tokens // self.block_size
        remainder = total_tokens % self.block_size

        if num_full_blocks == 0:
            # Only a partial first block
            block_idx = self.block_tables[seq_id, 0].item()
            key = self.physical_key_caches[layer_idx][block_idx, :remainder].unsqueeze(0)
            value = self.physical_value_caches[layer_idx][block_idx, :remainder].unsqueeze(0)
        elif remainder == 0:
            block_indices = self.block_tables[seq_id, :num_full_blocks].long()
            key = self.physical_key_caches[layer_idx][block_indices].reshape(
                1, total_tokens, -1, self.head_dim
            )
            value = self.physical_value_caches[layer_idx][block_indices].reshape(
                1, total_tokens, -1, self.head_dim
            )
        else:
            # Full blocks + partial last block
            full_indices = self.block_tables[seq_id, :num_full_blocks].long()
            last_block_idx = self.block_tables[seq_id, num_full_blocks].item()

            full_key = self.physical_key_caches[layer_idx][full_indices].reshape(
                -1, self.num_local_kv_groups, self.head_dim
            )
            last_key = self.physical_key_caches[layer_idx][last_block_idx, :remainder]

            full_value = self.physical_value_caches[layer_idx][full_indices].reshape(
                -1, self.num_local_kv_groups, self.head_dim
            )
            last_value = self.physical_value_caches[layer_idx][last_block_idx, :remainder]

            key = torch.cat([full_key, last_key], dim=0).unsqueeze(0)
            value = torch.cat([full_value, last_value], dim=0).unsqueeze(0)

        return key, value

    def _get_seq_position(self, seq_id: int) -> int:
        """Get total token count for a sequence."""
        if not self.is_initialized:
            return 0
        return self.token_positions[seq_id].item()

    def get_cache_position(self, seq_id: int = 0) -> int:
        """Public API: get total cached tokens for a sequence."""
        return self._get_seq_position(seq_id)

    def get_sequence_position(self, batch_idx: int) -> int:
        return self.get_cache_position(batch_idx)

    def get_statistics(self) -> dict:
        """Get cache statistics for monitoring."""
        if not self.is_initialized:
            return {"initialized": False, "memory_mb": 0, "utilization": 0.0}

        all_caches = self.physical_key_caches + self.physical_value_caches
        memory_mb = compute_memory_mb(all_caches, self.dtype)

        max_capacity = self.num_physical_blocks
        used_blocks = max_capacity - len(self.free_blocks)
        utilization = compute_utilization(used_blocks, max_capacity)

        return {
            "initialized": True,
            "type": "paged",
            "block_size": self.block_size,
            "num_physical_blocks": max_capacity,
            "num_free_blocks": len(self.free_blocks),
            "num_used_blocks": used_blocks,
            "max_num_blocks_per_seq": self.max_num_blocks_per_seq,
            "batch_size": self.block_tables.shape[0] if self.block_tables is not None else 0,
            "num_local_kv_groups": self.num_local_kv_groups,
            "memory_mb": memory_mb,
            "utilization": utilization,
        }
