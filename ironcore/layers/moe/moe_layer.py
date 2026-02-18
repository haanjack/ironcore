# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Mixture of Experts MLP Layer.

This module implements the DeepSeek-MoE style MoE layer with:
- Shared experts: Always-active experts that process all tokens
- Routed experts: Top-k routing with fine-grained segmentation
- Load balancing loss: Auxiliary loss for even expert utilization

Two communication approaches are supported:
1. All-Reduce: Simpler, each rank processes tokens for its experts
2. All-to-All: Async-optimized with communication overlap opportunity

Architecture:
    Input: [batch, seq_len, hidden_size]
    |
    +---> Shared Experts (process ALL tokens)
    |     Expert 0, Expert 1, ... Expert N-1
    |     |
    |     +---> Sum -> shared_output
    |
    +---> Router -> Top-k Selection
          |
          Routed Experts (selected by router)
          Expert i, Expert j, Expert k, ...
          |
          +---> Weighted Combine -> routed_output
    |
    Output = shared_output + routed_output
"""

import warnings
from enum import Enum

import torch
from torch import nn

from ironcore.config import MainConfig
from ironcore.layers.module import BaseModule
from ironcore.parallel.expert_parallel import (
    AllToAllDispatcher,
    DispatchOutput,
    all_reduce_ep_with_grad,
)
from ironcore.parallel.expert_parallel.parallel_states import get_expert_model_parallel_rank

from .expert import ExpertMLP
from .load_balance_loss import LoadBalanceLoss, get_expert_utilization
from .router import RouterOutput, TopKRouter
from .utils import flatten_moe_inputs, validate_moe_input


class CommunicationMode(Enum):
    """Communication mode for expert parallelism."""

    ALL_REDUCE = "all_reduce"  # Simple, each rank processes local tokens
    ALL_TO_ALL = "all_to_all"  # Async-optimized with communication overlap


class MoEMLP(BaseModule):
    """Mixture of Experts MLP layer.

    Combines shared experts (process all tokens) with routed experts
    (selected via top-k routing).

    Supports two communication modes:
    - ALL_REDUCE: Each rank processes tokens for its local experts,
                  then all-reduce combines partial outputs
    - ALL_TO_ALL: Tokens are dispatched to expert ranks via all-to-all,
                  processed, then gathered back. Allows overlap with
                  shared expert computation.

    Args:
        config: Main configuration containing model and MoE settings
        communication_mode: Communication mode for EP (default: ALL_REDUCE)
    """

    def __init__(
        self,
        config: MainConfig,
        communication_mode: CommunicationMode = CommunicationMode.ALL_REDUCE,
    ):
        super().__init__(config)

        model_config = config.model
        moe_config = model_config.moe

        # MoE parameters
        self.hidden_size = model_config.d_model
        self.num_shared_experts = moe_config.num_shared_experts
        self.num_routed_experts = moe_config.num_routed_experts
        self.top_k = moe_config.num_experts_per_token
        self.aux_loss_alpha = moe_config.aux_loss_alpha

        # Communication mode
        self.communication_mode = communication_mode

        # Expert parallelism parameters
        self.ep_size = moe_config.expert_model_parallel_size
        self.tp_size = config.trainer.tensor_model_parallel_size

        # Calculate expert intermediate size
        self.expert_intermediate_size = (
            moe_config.expert_intermediate_size
            if moe_config.expert_intermediate_size is not None
            else model_config.d_ffn
        )

        # Shared experts - process ALL tokens
        # These are not affected by expert parallelism (each rank has full shared experts)
        self.shared_experts = nn.ModuleList(
            [
                ExpertMLP(
                    config=config,
                    hidden_size=self.hidden_size,
                    intermediate_size=self.expert_intermediate_size,
                    expert_id=i,
                )
                for i in range(self.num_shared_experts)
            ]
        )

        # Calculate local expert range for expert parallelism
        if self.ep_size > 1:
            if self.num_routed_experts % self.ep_size != 0:
                raise ValueError(
                    f"num_routed_experts ({self.num_routed_experts}) must be divisible by "
                    f"expert_model_parallel_size ({self.ep_size})"
                )
            self.local_num_experts = self.num_routed_experts // self.ep_size
            self.expert_start_idx = get_expert_model_parallel_rank() * self.local_num_experts
            self.expert_end_idx = self.expert_start_idx + self.local_num_experts
        else:
            self.local_num_experts = self.num_routed_experts
            self.expert_start_idx = 0
            self.expert_end_idx = self.num_routed_experts

        # Routed experts - each rank holds a subset if EP > 1
        self.routed_experts = nn.ModuleList(
            [
                ExpertMLP(
                    config=config,
                    hidden_size=self.hidden_size,
                    intermediate_size=self.expert_intermediate_size,
                    expert_id=self.expert_start_idx + i,
                )
                for i in range(self.local_num_experts)
            ]
        )

        # Router
        self.router = TopKRouter(
            config=config,
            hidden_size=self.hidden_size,
            num_experts=self.num_routed_experts,
            top_k=self.top_k,
        )

        # Load balance loss
        self.load_balance_loss = LoadBalanceLoss(
            num_experts=self.num_routed_experts,
            aux_loss_alpha=self.aux_loss_alpha,
        )

        # Dispatcher for all-to-all mode
        if self.communication_mode == CommunicationMode.ALL_TO_ALL:
            self.dispatcher = AllToAllDispatcher(self.num_routed_experts, self.ep_size)
        else:
            self.dispatcher = None

        # Store auxiliary loss for accumulation
        self._aux_loss = None

        # Track expert selection counts for analysis
        self._expert_selection_counts = torch.zeros(self.num_routed_experts, dtype=torch.long)
        self._total_selections = 0

        # Store router info for logging (detached, on CPU)
        self._router_probs = None  # [batch*seq, num_experts] average probs per token
        self._selected_indices = None  # [batch*seq*top_k] flattened selected indices

        # Reusable CUDA stream for shared expert overlap (lazy init)
        self._shared_stream = None

    def forward(
        self,
        x: torch.Tensor,
        async_communication: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.distributed.Work | None]:
        """Forward pass through MoE layer.

        Args:
            x: [batch, seq, hidden] input tokens
            async_communication: Whether to use async communication

        Returns:
            [batch, seq, hidden] output with shared + routed expert contributions
            If async_communication=True, returns (output, handle) tuple

        Raises:
            ValueError: If input tensor is invalid
        """
        # Input validation
        validate_moe_input(x, self.hidden_size, "MoEMLP")

        if self.communication_mode == CommunicationMode.ALL_TO_ALL:
            return self._forward_alltoall(x, async_communication)
        else:
            return self._forward_allreduce(x, async_communication)

    def _forward_allreduce(
        self,
        x: torch.Tensor,
        async_communication: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.distributed.Work | None]:
        """Forward pass using all-reduce approach.

        Simple approach: each rank processes tokens for its local experts,
        then all-reduce combines partial outputs.
        """
        batch_size, seq_len, hidden_size = x.shape

        # 1. Shared experts - process ALL tokens
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += expert(x)

        # 2. Routed experts - top-k routing
        router_output: RouterOutput = self.router(x, self.training)

        # Track expert selections for analysis (vectorized)
        if self.training:
            with torch.no_grad():
                flat_indices = router_output.topk_indices.flatten()
                valid_mask = (flat_indices >= 0) & (flat_indices < self.num_routed_experts)
                valid_indices = flat_indices[valid_mask]
                if valid_indices.numel() > 0:
                    # Use bincount for fast histogram
                    counts = torch.bincount(valid_indices.long(), minlength=self.num_routed_experts)
                    # Move tracking tensor to same device as counts if needed
                    if self._expert_selection_counts.device != counts.device:
                        self._expert_selection_counts = self._expert_selection_counts.to(
                            counts.device
                        )
                    self._expert_selection_counts += counts.long()
                self._total_selections += flat_indices.numel()

                # Store router info for TensorBoard logging (detach and move to CPU)
                # Router probs: softmax over all experts
                router_probs = torch.softmax(
                    router_output.router_logits, dim=-1
                )  # [batch, seq, num_experts]
                self._router_probs = router_probs.detach().cpu()  # Store for logging

                # Selected indices: flatten the top-k indices
                self._selected_indices = (
                    router_output.topk_indices.flatten().detach().cpu()
                )  # [batch*seq*top_k]

        # Compute and store auxiliary loss for training
        if self.training:
            self._aux_loss = self.load_balance_loss(
                router_logits=router_output.router_logits,
                topk_indices=router_output.topk_indices,
            )

        # Route tokens to experts
        routed_output = self._route_and_combine_allreduce(
            x=x,
            topk_weights=router_output.topk_weights,
            topk_indices=router_output.topk_indices,
        )

        # 3. Combine shared + routed outputs
        output = shared_output + routed_output

        if async_communication:
            return output, None

        return output

    def _forward_alltoall(
        self,
        x: torch.Tensor,
        async_communication: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.distributed.Work | None]:
        """Forward pass using all-to-all approach with overlap.

        Async-optimized approach with shared expert / dispatch overlap:
        1. Get routing decisions
        2. Start shared experts FIRST (in separate stream for overlap)
        3. Dispatch tokens to expert ranks (overlaps with shared experts)
        4. Synchronize to ensure shared experts complete
        5. Process local experts
        6. Gather results (async if requested)
        7. Combine with shared output
        """
        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        dtype = x.dtype

        # 1. Get routing decisions first
        router_output: RouterOutput = self.router(x, self.training)

        # Compute and store auxiliary loss for training
        if self.training:
            self._aux_loss = self.load_balance_loss(
                router_logits=router_output.router_logits,
                topk_indices=router_output.topk_indices,
            )

        # 2. Start shared experts FIRST (can overlap with dispatch)
        # Use a separate CUDA stream for shared experts to enable overlap
        shared_output = torch.zeros(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
        shared_done = None  # Event to signal completion

        if x.is_cuda and torch.cuda.is_available():
            # Lazy init of reusable stream
            if self._shared_stream is None:
                self._shared_stream = torch.cuda.Stream()

            # Run shared experts in separate stream
            with torch.cuda.stream(self._shared_stream):
                for expert in self.shared_experts:
                    shared_output += expert(x)

            # Record completion event for later synchronization
            shared_done = torch.cuda.Event()
            shared_done.record(self._shared_stream)
        else:
            # CPU fallback - no stream support
            for expert in self.shared_experts:
                shared_output += expert(x)

        # 3. Dispatch tokens to expert ranks (overlaps with shared experts on GPU)
        dispatch_output, metadata = self.dispatcher.dispatch(
            hidden_states=x,
            topk_indices=router_output.topk_indices,
            topk_weights=router_output.topk_weights,
        )

        # 4. Synchronize to ensure shared experts complete before combining
        if shared_done is not None:
            shared_done.synchronize()

        # 5. Process local experts with dispatched tokens
        expert_outputs = self._process_dispatched_tokens(dispatch_output)

        # 6. Gather results
        routed_output, handle = self.dispatcher.gather(
            expert_outputs=expert_outputs,
            metadata=metadata,
            async_op=async_communication,
        )

        # 7. Combine shared + routed outputs
        output = shared_output + routed_output

        if async_communication:
            return output, handle

        return output

    def _route_and_combine_allreduce(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Route tokens to experts and combine outputs (all-reduce approach).

        Each rank processes tokens routed to its local experts,
        then all-reduce combines partial outputs.
        """
        batch_size, seq_len, hidden_size = x.shape

        # Flatten for processing using helper
        x_flat, topk_weights_flat, topk_indices_flat, num_tokens, _ = flatten_moe_inputs(
            x, topk_weights, topk_indices
        )

        # Initialize output
        output = torch.zeros(num_tokens, hidden_size, device=x.device, dtype=x.dtype)

        # Process each local expert
        for local_expert_idx, expert in enumerate(self.routed_experts):
            global_expert_idx = self.expert_start_idx + local_expert_idx

            # Find tokens routed to this expert
            expert_mask = topk_indices_flat == global_expert_idx

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            token_indices, k_indices = torch.where(expert_mask)
            expert_input = x_flat[token_indices]

            # Process through expert
            expert_output = expert(expert_input)

            # Get weights for these positions
            weights = topk_weights_flat[token_indices, k_indices]

            # Weighted contribution
            output[token_indices] += weights.unsqueeze(-1) * expert_output

        # Reshape back
        output = output.view(batch_size, seq_len, hidden_size)

        # For EP > 1, all-reduce across EP ranks to combine partial outputs
        if self.ep_size > 1:
            output = all_reduce_ep_with_grad(output)

        return output

    def _process_dispatched_tokens(
        self,
        dispatch_output: DispatchOutput,
    ) -> torch.Tensor:
        """Process dispatched tokens through local experts.

        Args:
            dispatch_output: Output from dispatcher containing tokens and metadata

        Returns:
            [num_outputs, hidden] outputs from local experts
        """
        tokens = dispatch_output.tokens
        num_tokens_per_expert = dispatch_output.num_tokens_per_expert

        # Allocate output tensor
        tokens.shape[0]
        tokens.shape[1]
        outputs = torch.zeros_like(tokens)

        # Check for edge case: no tokens dispatched to any expert
        total_tokens = sum(num_tokens_per_expert)
        if total_tokens == 0:
            warnings.warn(
                "No tokens were dispatched to any local expert. "
                "This may indicate a routing issue or extreme load imbalance.",
                UserWarning,
                stacklevel=2,
            )
            return outputs

        # Track position for each expert
        expert_position = 0

        # Process each local expert
        for local_expert_idx, expert in enumerate(self.routed_experts):
            num_tokens = num_tokens_per_expert[local_expert_idx]

            if num_tokens == 0:
                continue

            # Get tokens for this expert
            start_pos = expert_position
            end_pos = start_pos + num_tokens

            expert_input = tokens[start_pos:end_pos]

            # Process through expert
            expert_output = expert(expert_input)

            # Store output
            outputs[start_pos:end_pos] = expert_output

            expert_position = end_pos

        return outputs

    def get_aux_loss(self) -> torch.Tensor | None:
        """Get accumulated auxiliary loss.

        Returns:
            Auxiliary loss tensor or None if not computed
        """
        return self._aux_loss

    def clear_aux_loss(self):
        """Clear stored auxiliary loss."""
        self._aux_loss = None

    def get_expert_utilization(self, topk_indices: torch.Tensor) -> torch.Tensor:
        """Get expert utilization statistics.

        Args:
            topk_indices: [batch, seq, top_k] expert indices

        Returns:
            [num_experts] utilization fractions
        """
        return get_expert_utilization(topk_indices, self.num_routed_experts)

    def get_expert_stats(self) -> dict:
        """Get expert selection statistics for analysis.

        Returns:
            Dict with:
                - expert_counts: List of selection counts per expert
                - total_selections: Total number of selections
        """
        return {
            "expert_counts": self._expert_selection_counts.tolist(),
            "total_selections": self._total_selections,
        }

    def get_router_probs(self) -> torch.Tensor | None:
        """Get router probabilities for all experts.

        Returns:
            [batch*seq, num_experts] tensor of average probabilities per token, or None
        """
        return self._router_probs

    def get_selected_indices(self) -> torch.Tensor | None:
        """Get flattened selected expert indices.

        Returns:
            [batch*seq*top_k] tensor of selected expert indices, or None
        """
        return self._selected_indices

    def reset_expert_stats(self):
        """Reset expert selection statistics."""
        self._expert_selection_counts.zero_()
        self._total_selections = 0
        self._router_probs = None
        self._selected_indices = None

    def finalize(
        self,
        x: torch.Tensor,
        handle: torch.distributed.Work | None,
    ) -> torch.Tensor:
        """Finalize async forward pass.

        Args:
            x: Output from forward()
            handle: Async handle to wait on

        Returns:
            Output tensor after waiting for async operations
        """
        if handle is not None:
            handle.wait()
        return x
