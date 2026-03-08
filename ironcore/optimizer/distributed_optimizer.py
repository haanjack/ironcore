# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
ZeRO-1/2 DistributedOptimizer.

ZeRO-1: Partition optimizer states across DP ranks.
         Grad all-reduce (DDP) → each rank updates its partition → broadcast params.

ZeRO-2: Partition gradients and optimizer states across DP ranks.
         Reduce-scatter grads → each rank updates its partition → broadcast params.

Critical distinction from FSDP (ZeRO-3):
  FSDP shards parameters during forward — all-gather on every layer access.
  ZeRO-1/2 keeps parameters FULLY REPLICATED — no forward-pass overhead.

ZeRO-1 memory per rank (dp_size = N):
  Parameters:       P  bytes  (replicated)
  Gradients:        P  bytes  (all-reduced)
  Optimizer states: 2P/N bytes ← savings here

ZeRO-2 additionally:
  Gradients:        P/N bytes (reduce-scattered)
  Total: P + P/N + 2P/N

Compatibility notes:
  - Wraps an existing AdamWOptimizer (or any Optimizer with .param_groups)
  - Requires model wrapped in DDP (initialized via initialize_parallelism with use_fsdp=False)
  - For ZeRO-2: trainer must wrap backward with model.no_sync() for all accumulation steps,
    then call _sync_gradients_zero2() before optimizer.step()
  - Incompatible with FSDP (enforced by config validation)
"""

import torch
from torch import distributed as dist


class DistributedOptimizer:
    """ZeRO-1/2 optimizer wrapper for Megatron-style distributed optimization.

    Wraps an existing optimizer and partitions optimizer states (ZeRO-1) and
    optionally gradients (ZeRO-2) across data-parallel ranks, while keeping
    parameters fully replicated.

    Args:
        optimizer: The inner optimizer to wrap (e.g., AdamWOptimizer)
        zero_stage: ZeRO stage (1 or 2)
        process_group: Data-parallel process group. Defaults to get_data_parallel_group().
    """

    def __init__(self, optimizer, zero_stage: int, process_group=None):
        assert zero_stage in (1, 2), f"zero_stage must be 1 or 2, got {zero_stage}"

        self.optimizer = optimizer
        self.zero_stage = zero_stage

        if process_group is None:
            from ironcore.parallel.parallel_states import get_data_parallel_group

            process_group = get_data_parallel_group()

        self.process_group = process_group

        if dist.is_available() and dist.is_initialized():
            self.dp_size = dist.get_world_size(group=process_group)
            self.dp_rank = dist.get_rank(group=process_group)
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
                f"[ZeRO-{self.zero_stage}] DistributedOptimizer initialized: "
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

    def _sync_gradients_zero2(self):
        """Reduce-scatter gradients for ZeRO-2.

        All-reduces each parameter's gradient across DP ranks (same communication
        volume as reduce-scatter + all-gather, simpler to implement). Then zeros
        out gradients for parameters this rank does not own, freeing memory.

        Call this BEFORE scaler.unscale_() and clip_grad_norm() in the training step.
        The trainer must have used model.no_sync() for all backward passes when
        zero_stage=2 to prevent DDP's automatic gradient all-reduce.
        """
        if self.dp_size <= 1:
            return

        for i, p in enumerate(self.all_params):
            if p.grad is None:
                continue
            # All-reduce to get averaged gradient on all ranks
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=self.process_group)
            # Free gradient memory for parameters owned by other ranks
            if i not in self.local_param_indices:
                p.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        """Optimizer step: update local parameter partition, then broadcast all params.

        For ZeRO-1: expects gradients already all-reduced by DDP.
        For ZeRO-2: expects _sync_gradients_zero2() already called.

        After updating the local partition, broadcasts each parameter from its
        owner rank to all other ranks to restore the fully-replicated state.
        """
        # ZeRO-1: DDP has already all-reduced gradients (happens in last backward step).
        # Temporarily null out non-local param grads so the inner optimizer skips them.
        # ZeRO-2: non-local grads already None from _sync_gradients_zero2().
        saved_grads: dict[int, torch.Tensor] = {}
        if self.zero_stage == 1 and self.dp_size > 1:
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
            f"zero_stage={self.zero_stage}, "
            f"dp_size={self.dp_size}, "
            f"dp_rank={self.dp_rank}, "
            f"local_params={len(self.local_param_indices)}/{len(self.all_params)}, "
            f"inner={self.optimizer!r})"
        )
