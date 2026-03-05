# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
DistributedOptimizer - Optimizer State Partitioning.

Partitions optimizer states across data-parallel ranks while keeping
parameters and gradients fully replicated.

Memory savings (dp_size = N):
  Parameters:       P  bytes  (replicated)
  Gradients:        P  bytes  (all-reduced via DDP)
  Optimizer states: 2P/N bytes (partitioned across DP ranks)

Total per rank: 2P + 2P/N (vs 4P without partitioning)

Communication pattern:
  1. Forward pass: Normal (no communication)
  2. Backward pass: DDP all-reduces gradients
  3. Optimizer step:
     a. Each rank updates only its partition of parameters
     b. Broadcast updated parameters from owner ranks to all other ranks

Critical distinction from FSDP:
  FSDP shards parameters during forward - all-gather on every layer access.
  DistributedOptimizer keeps parameters FULLY REPLICATED - no forward-pass overhead.

When to use:
  - Use DistributedOptimizer when optimizer states are the memory bottleneck
  - Use FSDP with shard_grad_op when both optimizer states and gradients are bottlenecks
  - Use FSDP with full_shard when even parameters don't fit
"""

import logging
from typing import Any, Dict, List, Optional, Set

import torch
from torch import distributed as dist
from torch.optim import Optimizer

try:
    from ironcore.global_vars import get_logger
    logger = get_logger()
except (ImportError, AssertionError):
    logger = logging.getLogger(__name__)


class DistributedOptimizer(Optimizer):
    """Distributed optimizer that partitions optimizer states across DP ranks.

    Wraps an existing optimizer and partitions optimizer states across
    data-parallel ranks, while keeping parameters and gradients fully replicated.

    This provides memory savings for optimizer states (which are typically 2x the
    parameter count in fp32 for Adam) without the communication overhead of
    parameter sharding during forward/backward passes.

    Args:
        optimizer: The inner optimizer to wrap (e.g., AdamWOptimizer)
        process_group: Data-parallel process group. Defaults to get_data_parallel_group().
        bucket_cap_mb: Maximum bucket size in megabytes for parameter broadcasting.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        process_group: Optional[dist.ProcessGroup] = None,
        bucket_cap_mb: float = 25.0,
    ):
        # We don't call super().__init__ because we want to delegate everything
        # to the inner optimizer, but we must inherit from Optimizer to pass
        # isinstance checks.
        self.optimizer = optimizer
        self.process_group = process_group
        self.bucket_cap_mb = bucket_cap_mb

        if process_group is None:
            try:
                from ironcore.parallel.parallel_states import get_data_parallel_group

                self.process_group = get_data_parallel_group()
            except RuntimeError:
                self.process_group = None

        if dist.is_available() and dist.is_initialized() and self.process_group is not None:
            self.dp_size = dist.get_world_size(group=self.process_group)
            self.dp_rank = dist.get_rank(group=self.process_group)
        else:
            self.dp_size = 1
            self.dp_rank = 0

        # Collect all parameters in a deterministic flat list across all param groups
        self.all_params: List[torch.nn.Parameter] = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                self.all_params.append(p)

        # Round-robin assignment: param i owned by rank (i % dp_size)
        self.local_param_indices: Set[int] = {
            i for i in range(len(self.all_params)) if i % self.dp_size == self.dp_rank
        }

        # Create buckets for efficient broadcasting
        self._buckets = self._create_buckets()

        # Log memory savings
        total_params = sum(p.numel() for p in self.all_params)
        local_params = sum(
            self.all_params[i].numel() for i in range(len(self.all_params))
            if i in self.local_param_indices
        )

        # Optimizer states (Adam moments) are typically always float32
        bytes_per_element = 4
        total_opt_bytes = total_params * 2 * bytes_per_element
        local_opt_bytes = local_params * 2 * bytes_per_element

        if self.dp_rank == 0:
            logger.info(
                f"[DistributedOptimizer] Optimizer state partitioning enabled | "
                f"dp_size={self.dp_size}, "
                f"total_params={total_params:,}, "
                f"local_params={local_params:,} ({100.0 * local_params / max(total_params, 1):.1f}%), "
                f"total_opt_state={total_opt_bytes / 1024**2:.1f} MiB, "
                f"local_opt_state={local_opt_bytes / 1024**2:.1f} MiB, "
                f"buckets={len(self._buckets)}"
            )

    def _create_buckets(self) -> List[Dict[str, Any]]:
        """Group parameters into buckets by owner rank for efficient broadcasting."""
        if self.dp_size <= 1:
            return []

        # Group parameters by their owner rank
        rank_to_params = {r: [] for r in range(self.dp_size)}
        for i, p in enumerate(self.all_params):
            owner_rank = i % self.dp_size
            rank_to_params[owner_rank].append(p)

        buckets = []
        bucket_cap_bytes = self.bucket_cap_mb * 1024 * 1024

        for rank, params in rank_to_params.items():
            current_bucket = []
            current_size = 0
            
            for p in params:
                param_size = p.numel() * p.element_size()
                if current_bucket and (current_size + param_size > bucket_cap_bytes):
                    buckets.append({"rank": rank, "params": current_bucket})
                    current_bucket = []
                    current_size = 0
                
                current_bucket.append(p)
                current_size += param_size
            
            if current_bucket:
                buckets.append({"rank": rank, "params": current_bucket})
                
        return buckets

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    @property
    def state(self):
        return self.optimizer.state

    def __getattr__(self, name):
        """Delegate unknown attribute access to inner optimizer."""
        if name in (
            "optimizer",
            "zero_stage",
            "process_group",
            "dp_size",
            "dp_rank",
            "all_params",
            "local_param_indices",
            "_buckets",
            "bucket_cap_mb",
        ):
            raise AttributeError(name)
        return getattr(self.optimizer, name)

    @torch.no_grad()
    def step(self, closure=None):
        """Optimizer step: update local parameter partition, then broadcast all params.

        DDP has already all-reduced gradients. After updating the local partition,
        broadcasts each parameter from its owner rank to all other ranks to restore
        the fully-replicated state.
        """
        # DDP has already all-reduced gradients (happens in last backward step).
        # Temporarily null out non-local param grads so the inner optimizer skips them.
        saved_grads: Dict[int, torch.Tensor] = {}
        if self.dp_size > 1:
            for i, p in enumerate(self.all_params):
                if i not in self.local_param_indices and p.grad is not None:
                    saved_grads[i] = p.grad
                    p.grad = None

        # Step 1: inner optimizer updates ONLY parameters owned by this rank
        loss = self.optimizer.step(closure)

        # Restore non-local gradients
        for i, g in saved_grads.items():
            self.all_params[i].grad = g

        if self.dp_size <= 1:
            return loss

        # Step 2: broadcast updated parameters from owner ranks to all others
        # Using bucketing to minimize communication overhead
        for bucket in self._buckets:
            owner_rank = bucket["rank"]
            params = bucket["params"]

            # For each parameter in the bucket, perform broadcast
            # Note: broadcast is non-destructive on non-owner ranks
            handles = [
                dist.broadcast(p.data, src=owner_rank, group=self.process_group, async_op=True)
                for p in params
            ]

            for h in handles:
                h.wait()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Delegate zero_grad to inner optimizer."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return inner optimizer state dict (represents local partition)."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict into inner optimizer."""
        self.optimizer.load_state_dict(state_dict)

    def __repr__(self):
        return (
            f"DistributedOptimizer("
            f"dp_size={self.dp_size}, "
            f"dp_rank={self.dp_rank}, "
            f"local_params={len(self.local_param_indices)}/{len(self.all_params)}, "
            f"buckets={len(self._buckets)}, "
            f"inner={self.optimizer!r})"
        )
