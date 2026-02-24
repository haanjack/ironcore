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
     b. All-gather updated parameters from all ranks

Critical distinction from FSDP:
  FSDP shards parameters during forward - all-gather on every layer access.
  DistributedOptimizer keeps parameters FULLY REPLICATED - no forward-pass overhead.

When to use:
  - Use DistributedOptimizer when optimizer states are the memory bottleneck
  - Use FSDP with shard_grad_op when both optimizer states and gradients are bottlenecks
  - Use FSDP with full_shard when even parameters don't fit

Compatibility:
  - Wraps an existing AdamWOptimizer (or any Optimizer with .param_groups)
  - Requires model wrapped in DDP (use_fsdp=False)
  - Compatible with Tensor Parallelism and Expert Parallelism
  - Incompatible with FSDP (use FSDP's built-in sharding instead)
"""

import torch
from torch import distributed as dist


class DistributedOptimizer:
    """Distributed optimizer that partitions optimizer states across DP ranks.

    Wraps an existing optimizer and partitions optimizer states across
    data-parallel ranks, while keeping parameters and gradients fully replicated.

    This provides memory savings for optimizer states (which are typically 2x the
    parameter count in fp32 for Adam) without the communication overhead of
    parameter sharding during forward/backward passes.

    Args:
        optimizer: The inner optimizer to wrap (e.g., AdamWOptimizer)
        process_group: Data-parallel process group. Defaults to get_data_parallel_group().

    Example:
        >>> optimizer = AdamWOptimizer(model.parameters(), lr=1e-4)
        >>> optimizer = DistributedOptimizer(optimizer)
        >>> # Use normally with DDP-wrapped model
    """

    def __init__(self, optimizer, process_group=None):
        self.optimizer = optimizer
        self.process_group = process_group

        if process_group is None:
            # Try to get the DP group, but don't fail if not initialized
            try:
                from ironcore.parallel.parallel_states import get_data_parallel_group

                self.process_group = get_data_parallel_group()
            except RuntimeError:
                # Parallel states not initialized - use default (single process)
                self.process_group = None

        if dist.is_available() and dist.is_initialized() and self.process_group is not None:
            self.dp_size = dist.get_world_size(group=self.process_group)
            self.dp_rank = dist.get_rank(group=self.process_group)
        else:
            self.dp_size = 1
            self.dp_rank = 0

        # Collect all parameters in a deterministic flat list across all param groups
        self.all_params: list[torch.nn.Parameter] = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                self.all_params.append(p)

        # Assign parameters to ranks by round-robin on parameter index
        self.local_param_indices: set[int] = {
            i for i in range(len(self.all_params)) if i % self.dp_size == self.dp_rank
        }

        self._log_memory_breakdown()

    def _log_memory_breakdown(self):
        """Log optimizer state memory breakdown across DP ranks."""
        try:
            from ironcore.global_vars import get_logger

            logger = get_logger()
        except Exception:
            return

        total_params = sum(p.numel() for p in self.all_params)
        local_params = sum(self.all_params[i].numel() for i in self.local_param_indices)

        # Optimizer state memory: 2 moments (fp32) per parameter = 8 bytes each
        bytes_per_element = 4  # float32
        total_opt_bytes = total_params * 2 * bytes_per_element
        local_opt_bytes = local_params * 2 * bytes_per_element

        if self.dp_rank == 0:
            logger.info(
                f"[DistributedOptimizer] Optimizer state partitioning enabled | "
                f"dp_size={self.dp_size}, "
                f"total_params={total_params:,}, "
                f"local_params={local_params:,} ({100.0 * local_params / max(total_params, 1):.1f}%), "
                f"total_opt_state={total_opt_bytes / 1024**2:.1f} MiB, "
                f"local_opt_state={local_opt_bytes / 1024**2:.1f} MiB "
                f"(~{100.0 * local_opt_bytes / max(total_opt_bytes, 1):.1f}% of baseline)"
            )

    @property
    def param_groups(self):
        """Delegate to inner optimizer for GradScaler compatibility."""
        return self.optimizer.param_groups

    def __getattr__(self, name):
        """Delegate unknown attribute access to inner optimizer."""
        # Avoid infinite recursion for attributes that exist on this object
        if name in (
            "optimizer",
            "zero_stage",
            "process_group",
            "dp_size",
            "dp_rank",
            "all_params",
            "local_param_indices",
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
        saved_grads: dict[int, torch.Tensor] = {}
        if self.dp_size > 1:
            for i, p in enumerate(self.all_params):
                if i not in self.local_param_indices and p.grad is not None:
                    saved_grads[i] = p.grad
                    p.grad = None

        # Run inner optimizer on local partition only
        result = self.optimizer.step(closure)

        # Restore saved gradients (needed for grad_norm logging, if any)
        for i, grad in saved_grads.items():
            self.all_params[i].grad = grad

        # Broadcast updated parameters from each owner rank to all ranks
        self._all_gather_params()

        return result

    def _all_gather_params(self):
        """Broadcast updated parameter data from each owner rank to all ranks.

        Uses zero-out + all_reduce(SUM) to avoid needing global rank lookups:
        Non-owner ranks zero their copy; owner rank keeps the new value.
        After all_reduce SUM, all ranks hold the owner's updated value.
        """
        if self.dp_size <= 1:
            return

        # Phase 1: non-owner ranks zero their copy
        for i, p in enumerate(self.all_params):
            if i % self.dp_size != self.dp_rank:
                p.data.zero_()

        # Phase 2: all_reduce SUM — owner contributes new value, others contribute 0
        handles = []
        for p in self.all_params:
            handles.append(
                dist.all_reduce(
                    p.data, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True
                )
            )
        for h in handles:
            h.wait()

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for all parameters."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return optimizer state dict (local partition only)."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)

    def __repr__(self):
        return (
            f"DistributedOptimizer("
            f"dp_size={self.dp_size}, "
            f"dp_rank={self.dp_rank}, "
            f"local_params={len(self.local_param_indices)}/{len(self.all_params)}, "
            f"inner={self.optimizer!r})"
        )
