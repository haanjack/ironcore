# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Communication primitives for Expert Parallelism.

This module provides two approaches for EP communication:

1. All-Reduce Approach (Simple):
   - Each rank processes tokens routed to its local experts
   - All-reduce combines partial outputs
   - Simpler but no overlap opportunity

2. All-to-All Approach (Async-Optimized):
   - Dispatch: Sort tokens by expert, all-to-all to destination ranks
   - Process: Each rank only processes its received tokens
   - Gather: All-to-all results back to original ranks
   - Can overlap shared expert computation with communication

Dispatch Phase (All-to-All):
    Input: [batch, seq, hidden] -> sorted by expert -> all-to-all
    Output: [tokens_for_local_experts, hidden] per rank

Combine Phase (All-to-All):
    Input: [expert_outputs, hidden] per rank -> all-to-all -> unsort
    Output: [batch, seq, hidden] combined with weights
"""

from dataclasses import dataclass

import torch
from torch import distributed as dist

from .parallel_states import (
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
)

# =============================================================================
# All-Reduce Approach (Simple, Current Implementation)
# =============================================================================


def all_reduce_ep(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce across expert parallel ranks.

    Used to combine partial outputs from each EP rank after expert processing.

    Args:
        tensor: Tensor to all-reduce

    Returns:
        Reduced tensor (in-place modification)
    """
    ep_size = get_expert_model_parallel_world_size()

    if ep_size == 1:
        return tensor

    ep_group = get_expert_model_parallel_group()
    if ep_group is None:
        return tensor

    from ironcore.profiler import timed_comm
    with timed_comm("ep_all_reduce"):
        dist.all_reduce(tensor, group=ep_group)
    return tensor


class _AllReduceEP(torch.autograd.Function):
    """Autograd function for all-reduce across EP ranks.

    Uses in-place all_reduce to avoid unnecessary tensor copies.
    """

    @staticmethod
    def forward(ctx, input_tensor, ep_size):
        ctx.ep_size = ep_size
        if ep_size == 1:
            return input_tensor

        ep_group = get_expert_model_parallel_group()
        if ep_group is None:
            return input_tensor

        from ironcore.profiler import timed_comm
        with timed_comm("ep_all_reduce_fwd"):
            dist.all_reduce(input_tensor, group=ep_group)
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        ep_size = ctx.ep_size
        if ep_size == 1:
            return grad_output, None

        ep_group = get_expert_model_parallel_group()
        if ep_group is None:
            return grad_output, None

        from ironcore.profiler import timed_comm
        with timed_comm("ep_all_reduce_bwd"):
            dist.all_reduce(grad_output, group=ep_group)
        return grad_output, None


def all_reduce_ep_with_grad(input_tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce with gradient support for backward pass."""
    ep_size = get_expert_model_parallel_world_size()
    return _AllReduceEP.apply(input_tensor, ep_size)


# =============================================================================
# All-to-All Data Structures
# =============================================================================


@dataclass
class DispatchOutput:
    """Output from dispatch operation.

    Attributes:
        tokens: [num_local_tokens, hidden] tokens for local experts
        token_indices: Original token indices for gather
        expert_indices: Expert index for each token
        weights: Routing weight for each token
        num_tokens_per_expert: Number of tokens assigned to each local expert
    """

    tokens: torch.Tensor
    token_indices: torch.Tensor
    expert_indices: torch.Tensor
    weights: torch.Tensor
    num_tokens_per_expert: list[int]


@dataclass
class DispatchMetadata:
    """Metadata needed for gather operation.

    Saved during dispatch, used during gather.
    """

    original_shape: tuple[int, ...]
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    send_counts: list[int]  # Tokens sent to each rank
    recv_counts: list[int]  # Tokens received from each rank
    sort_indices: torch.Tensor  # For unsorting during gather


# =============================================================================
# All-to-All Async Communication
# =============================================================================


class AllToAllDispatcher:
    """Manages all-to-all token dispatch and gather for EP.

    This class provides async communication support for MoE layers,
    allowing overlap of communication with computation.
    """

    # Maximum buffer size to prevent OOM errors (100M elements)
    MAX_BUFFER_SIZE = 100_000_000

    def __init__(self, num_experts: int, ep_size: int):
        """Initialize dispatcher.

        Args:
            num_experts: Total number of routed experts
            ep_size: Expert parallel world size
        """
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.experts_per_rank = num_experts // ep_size

        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
            )

        # Cache EP info
        self.ep_rank = get_expert_model_parallel_rank()
        self.ep_group = get_expert_model_parallel_group()

        # Expert range for this rank
        self.local_expert_start = self.ep_rank * self.experts_per_rank
        self.local_expert_end = self.local_expert_start + self.experts_per_rank

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[DispatchOutput, DispatchMetadata]:
        """Dispatch tokens to expert ranks using batched all-to-all.

        Packs all 4 tensors into a single buffer for one all_to_all_single call,
        reducing communication overhead from 4 operations to 1.

        Args:
            hidden_states: [batch, seq, hidden] input tokens
            topk_indices: [batch, seq, top_k] expert indices
            topk_weights: [batch, seq, top_k] routing weights

        Returns:
            Tuple of (DispatchOutput, DispatchMetadata)
        """
        # 1. Prepare and flatten inputs
        prepared = self._prepare_dispatch_inputs(hidden_states, topk_indices, topk_weights)

        if self.ep_size == 1:
            return self._dispatch_single_rank(
                prepared["x_expanded"],
                prepared["expert_indices"],
                prepared["token_weights"],
                prepared["token_indices"],
                original_shape=hidden_states.shape,
                topk_indices=topk_indices,
                topk_weights=topk_weights,
            )

        # 2. Sort tokens by destination rank
        sorted_data = self._sort_by_destination(
            prepared["x_expanded"],
            prepared["expert_indices"],
            prepared["token_weights"],
            prepared["token_indices"],
        )

        # 3. Exchange tokens across EP ranks
        recv_data = self._exchange_tokens(sorted_data, prepared["hidden_size"])

        # 4. Unpack received data
        return self._unpack_dispatch_output(
            recv_data,
            prepared["hidden_size"],
            original_shape=hidden_states.shape,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
        )

    def _prepare_dispatch_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> dict:
        """Flatten and prepare inputs for dispatch."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len
        top_k = topk_indices.shape[-1]
        device = hidden_states.device

        # Flatten
        x_flat = hidden_states.view(num_tokens, hidden_size)
        indices_flat = topk_indices.view(num_tokens, top_k)
        weights_flat = topk_weights.view(num_tokens, top_k)

        # Expand for top-k routing
        x_expanded = x_flat.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_size)
        expert_indices = indices_flat.reshape(-1)
        token_weights = weights_flat.reshape(-1)
        token_indices = (
            torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        )

        return {
            "x_expanded": x_expanded,
            "expert_indices": expert_indices,
            "token_weights": token_weights,
            "token_indices": token_indices,
            "hidden_size": hidden_size,
            "num_tokens": num_tokens,
        }

    def _sort_by_destination(
        self,
        x_expanded: torch.Tensor,
        expert_indices: torch.Tensor,
        token_weights: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> dict:
        """Sort tokens by destination rank for all-to-all."""
        dest_ranks = expert_indices // self.experts_per_rank
        send_counts = torch.bincount(dest_ranks, minlength=self.ep_size).tolist()
        sort_indices = torch.argsort(dest_ranks, stable=True)

        return {
            "sorted_x": x_expanded[sort_indices],
            "sorted_expert_indices": expert_indices[sort_indices],
            "sorted_weights": token_weights[sort_indices],
            "sorted_token_indices": token_indices[sort_indices],
            "send_counts": send_counts,
            "sort_indices": sort_indices,
        }

    def _exchange_tokens(
        self,
        sorted_data: dict,
        hidden_size: int,
    ) -> dict:
        """Exchange tokens across EP ranks using all-to-all.

        Note on packed buffer format (performance consideration):
        - Indices are converted to float for packing into contiguous buffer
        - This wastes some bandwidth but reduces all_to_all calls from 4 to 1
        - For typical hidden sizes (1024-4096), overhead is <1%
        - Future optimization: Use separate int32/float16 buffers with multiple all_to_all

        Note on buffer allocation (performance consideration):
        - New tensors are allocated each forward pass
        - PyTorch's caching allocator handles most fragmentation
        - Future optimization: Pool buffers with max size tracking for training
        """
        device = sorted_data["sorted_x"].device
        dtype = sorted_data["sorted_x"].dtype

        # Efficient counts exchange using all_to_all_single (P-3)
        send_counts = sorted_data["send_counts"]
        send_counts_tensor = torch.tensor(send_counts, device=device, dtype=torch.long)
        recv_counts_tensor = torch.zeros(self.ep_size, device=device, dtype=torch.long)
        from ironcore.profiler import timed_comm
        with timed_comm("ep_all_to_all_counts"):
            dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=self.ep_group)
        recv_counts = recv_counts_tensor.tolist()

        total_send = sum(send_counts)
        total_recv = sum(recv_counts)

        # Check buffer size limits to prevent OOM
        total_elements = total_send * (hidden_size + 3)
        if total_elements > self.MAX_BUFFER_SIZE:
            raise RuntimeError(
                f"Buffer allocation ({total_elements:,} elements) exceeds max ({self.MAX_BUFFER_SIZE:,}). "
                f"Consider reducing batch size or sequence length."
            )

        # Pack all data into single buffer (kept for compatibility)
        packed_size_per_token = hidden_size + 3
        send_packed = torch.zeros(total_send, packed_size_per_token, device=device, dtype=dtype)
        send_packed[:, :hidden_size] = sorted_data["sorted_x"][:total_send]
        send_packed[:, hidden_size] = sorted_data["sorted_expert_indices"][:total_send].to(dtype)
        send_packed[:, hidden_size + 1] = sorted_data["sorted_weights"][:total_send]
        send_packed[:, hidden_size + 2] = sorted_data["sorted_token_indices"][:total_send].to(dtype)

        recv_packed = torch.zeros(total_recv, packed_size_per_token, device=device, dtype=dtype)

        # Single batched all_to_all_single call
        if self.ep_group is not None and total_send > 0:
            send_split_sizes = [s * packed_size_per_token for s in send_counts]
            recv_split_sizes = [s * packed_size_per_token for s in recv_counts]
            with timed_comm("ep_all_to_all_tokens"):
                dist.all_to_all_single(
                    recv_packed.view(-1),
                    send_packed.view(-1),
                    output_split_sizes=recv_split_sizes,
                    input_split_sizes=send_split_sizes,
                    group=self.ep_group,
                )

        return {
            "recv_packed": recv_packed,
            "recv_counts": recv_counts,
            "send_counts": send_counts,
            "sort_indices": sorted_data["sort_indices"],
            "hidden_size": hidden_size,
        }

    def _unpack_dispatch_output(
        self,
        recv_data: dict,
        hidden_size: int,
        original_shape: tuple[int, ...],
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[DispatchOutput, DispatchMetadata]:
        """Unpack received data into dispatch output."""
        recv_packed = recv_data["recv_packed"]

        # Unpack received data
        recv_x_flat = recv_packed[:, :hidden_size]
        recv_idx_flat = recv_packed[:, hidden_size].to(torch.long)
        recv_weight_flat = recv_packed[:, hidden_size + 1]
        recv_token_idx_flat = recv_packed[:, hidden_size + 2].to(torch.long)

        # Convert to local expert indices
        local_expert_indices_local = recv_idx_flat - self.local_expert_start

        # Count tokens per local expert (vectorized)
        num_tokens_per_expert = [0] * self.experts_per_rank
        if local_expert_indices_local.numel() > 0:
            valid_mask = (local_expert_indices_local >= 0) & (
                local_expert_indices_local < self.experts_per_rank
            )
            valid_indices = local_expert_indices_local[valid_mask]
            counts = torch.bincount(valid_indices, minlength=self.experts_per_rank)
            num_tokens_per_expert = counts.tolist()

        dispatch_output = DispatchOutput(
            tokens=recv_x_flat,
            token_indices=recv_token_idx_flat,
            expert_indices=local_expert_indices_local,
            weights=recv_weight_flat,
            num_tokens_per_expert=num_tokens_per_expert,
        )

        dispatch_metadata = DispatchMetadata(
            original_shape=original_shape,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            send_counts=recv_data["send_counts"],
            recv_counts=recv_data["recv_counts"],
            sort_indices=recv_data["sort_indices"],
        )

        return dispatch_output, dispatch_metadata

    def _dispatch_single_rank(
        self,
        x_expanded: torch.Tensor,
        expert_indices: torch.Tensor,
        token_weights: torch.Tensor,
        token_indices: torch.Tensor,
        original_shape: tuple[int, ...],
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[DispatchOutput, DispatchMetadata]:
        """Handle dispatch for single-rank EP (no communication)."""
        num_tokens_total = x_expanded.shape[0]

        # Sort by expert index for efficient processing
        sort_indices = torch.argsort(expert_indices)
        sorted_x = x_expanded[sort_indices]
        sorted_expert_indices = expert_indices[sort_indices]
        sorted_weights = token_weights[sort_indices]
        sorted_token_indices = token_indices[sort_indices]

        # Count tokens per expert
        num_tokens_per_expert = torch.bincount(
            sorted_expert_indices, minlength=self.num_experts
        ).tolist()

        dispatch_output = DispatchOutput(
            tokens=sorted_x,
            token_indices=sorted_token_indices,
            expert_indices=sorted_expert_indices,
            weights=sorted_weights,
            num_tokens_per_expert=num_tokens_per_expert,
        )

        dispatch_metadata = DispatchMetadata(
            original_shape=original_shape,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            send_counts=[num_tokens_total],
            recv_counts=[num_tokens_total],
            sort_indices=sort_indices,
        )

        return dispatch_output, dispatch_metadata

    def gather(
        self,
        expert_outputs: torch.Tensor,
        metadata: DispatchMetadata,
        async_op: bool = False,
    ) -> tuple[torch.Tensor, dist.Work | None]:
        """Gather expert outputs and combine with routing weights.

        Args:
            expert_outputs: [num_local_outputs, hidden] outputs from local experts
            metadata: Metadata saved from dispatch
            async_op: Whether to perform async communication

        Returns:
            Tuple of (combined_output, async_handle)
        """
        original_shape = metadata.original_shape
        batch_size, seq_len, hidden_size = original_shape
        batch_size * seq_len
        metadata.topk_indices.shape[-1]
        device = expert_outputs.device
        dtype = expert_outputs.dtype

        if self.ep_size == 1:
            # No communication needed
            output = self._gather_single_rank(expert_outputs, metadata, original_shape)
            return output, None

        # Prepare for all-to-all gather (reverse of dispatch)
        # recv_counts from dispatch becomes send_counts for gather
        # send_counts from dispatch becomes recv_counts for gather
        send_counts = metadata.recv_counts  # What we received during dispatch
        recv_counts = metadata.send_counts  # What we sent during dispatch

        total_send = sum(send_counts)
        total_recv = sum(recv_counts)

        # Create send/recv tensors
        send_outputs_flat = (
            expert_outputs[:total_send]
            if expert_outputs.shape[0] >= total_send
            else torch.zeros(total_send, hidden_size, device=device, dtype=dtype)
        )
        recv_outputs_flat = torch.zeros(total_recv, hidden_size, device=device, dtype=dtype)

        # Convert to lists for the API
        send_split_sizes = send_counts
        recv_split_sizes = recv_counts

        # Perform all_to_all_single
        if self.ep_group is not None and total_send > 0:
            # For 2D tensor, reshape to 1D and adjust split sizes
            send_outputs_1d = send_outputs_flat.view(-1)
            recv_outputs_1d = recv_outputs_flat.view(-1)

            send_split_sizes_out = [s * hidden_size for s in send_split_sizes]
            recv_split_sizes_out = [s * hidden_size for s in recv_split_sizes]

            from ironcore.profiler import timed_comm
            with timed_comm("ep_all_to_all_gather"):
                dist.all_to_all_single(
                    recv_outputs_1d,
                    send_outputs_1d,
                    output_split_sizes=recv_split_sizes_out,
                    input_split_sizes=send_split_sizes_out,
                    group=self.ep_group,
                )

        # Combine outputs using routing weights
        output = self._combine_outputs(recv_outputs_flat, metadata, original_shape)

        return output, None

    def _gather_single_rank(
        self,
        expert_outputs: torch.Tensor,
        metadata: DispatchMetadata,
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Handle gather for single-rank EP."""
        return self._combine_outputs(expert_outputs, metadata, original_shape)

    def _combine_outputs(
        self,
        expert_outputs: torch.Tensor,
        metadata: DispatchMetadata,
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Combine expert outputs using routing weights.

        Uses vectorized operations instead of Python loops for efficiency.

        Args:
            expert_outputs: [num_outputs, hidden] expert outputs
            metadata: Dispatch metadata with routing info
            original_shape: Original [batch, seq, hidden] shape

        Returns:
            Combined output tensor
        """
        batch_size, seq_len, hidden_size = original_shape
        num_tokens = batch_size * seq_len
        top_k = metadata.topk_indices.shape[-1]
        device = expert_outputs.device
        dtype = expert_outputs.dtype

        # Initialize output
        output = torch.zeros(num_tokens, hidden_size, device=device, dtype=dtype)

        # Flatten routing info
        topk_indices_flat = metadata.topk_indices.view(num_tokens, top_k)
        topk_weights_flat = metadata.topk_weights.view(num_tokens, top_k)

        # Build reverse mapping: for each (token, k) pair, find position in expert_outputs
        # This matches the dispatch sort order
        flat_indices = topk_indices_flat.view(-1)  # [num_tokens * top_k]
        flat_weights = topk_weights_flat.view(-1)

        # Sort by expert index (same as dispatch)
        sort_order = torch.argsort(flat_indices)

        # Map sorted position back to (token, k) for weight application
        token_k_indices = torch.arange(num_tokens * top_k, device=device)
        sorted_token_k = token_k_indices[sort_order]

        # Vectorized combination using index_add_
        valid_count = min(expert_outputs.shape[0], sorted_token_k.shape[0])
        if valid_count > 0:
            # Get token indices for valid outputs
            token_indices = sorted_token_k[:valid_count] // top_k

            # Get weights for valid outputs
            weights = flat_weights[sorted_token_k[:valid_count]]

            # Apply weights to expert outputs
            weighted_outputs = weights.unsqueeze(-1) * expert_outputs[:valid_count]

            # Use index_add_ for efficient accumulation
            output.index_add_(0, token_indices, weighted_outputs)

        return output.view(batch_size, seq_len, hidden_size)


# =============================================================================
# High-Level API for MoE Layer
# =============================================================================


def dispatch_tokens(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> tuple[DispatchOutput, DispatchMetadata]:
    """Dispatch tokens to expert ranks.

    Args:
        hidden_states: [batch, seq, hidden] input tokens
        topk_indices: [batch, seq, top_k] expert indices
        topk_weights: [batch, seq, top_k] routing weights
        num_experts: Total number of experts

    Returns:
        Tuple of (DispatchOutput, DispatchMetadata)
    """
    ep_size = get_expert_model_parallel_world_size()
    dispatcher = AllToAllDispatcher(num_experts, ep_size)
    return dispatcher.dispatch(hidden_states, topk_indices, topk_weights)


def gather_tokens(
    expert_outputs: torch.Tensor,
    metadata: DispatchMetadata,
    async_op: bool = False,
) -> tuple[torch.Tensor, dist.Work | None]:
    """Gather expert outputs and combine.

    Args:
        expert_outputs: [num_outputs, hidden] outputs from local experts
        metadata: Metadata from dispatch
        async_op: Whether to use async communication

    Returns:
        Tuple of (combined_output, async_handle)
    """
    ep_size = get_expert_model_parallel_world_size()
    num_experts = metadata.topk_indices.max().item() + 1
    dispatcher = AllToAllDispatcher(num_experts, ep_size)
    return dispatcher.gather(expert_outputs, metadata, async_op=async_op)
