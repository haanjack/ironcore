# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Activation spilling manager for M3: forward D2H spill + backward H2D prefetch.

During the forward pass, intermediate activations (hidden_states, post-attention
residual) are asynchronously copied to pinned host memory via D2H transfers.
During the backward pass, those activations are prefetched back to GPU via H2D
transfers before each sub-layer's gradient computation.

This replaces activation checkpointing: instead of recomputing activations during
backward, they are fetched from host memory. Peak host memory is bounded by
free-after-consume: once backward consumes an activation, its pinned memory is
returned to the pool.

Thread safety: not thread-safe. All calls from the training loop's main thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ironcore.offload.config import OffloadConfig
    from ironcore.offload.memory_pool import PinnedMemoryPool
    from ironcore.offload.transfer_engine import MemoryTransferEngine

_log = logging.getLogger(__name__)


class _SpillCheckpointFn(torch.autograd.Function):
    """
    Custom autograd Function for M3 activation offloading.

    Replaces activation checkpointing (recomputation) with host-based activation
    storage. The input activation is spilled to host memory during forward, and
    restored from host during backward.

    Forward: spills the input activation to host via D2H, computes the sub-block
    with torch.no_grad() (no intermediate activations retained on GPU).

    Backward: restores the input from host via H2D, recomputes the sub-block
    with torch.enable_grad() to rebuild the autograd graph, then computes gradients.

    GPU memory savings: all intermediate activations within the sub-block (layernorm
    output, QKV projections, attention scores, MLP intermediates) are freed. Only
    the spilled input is retained (on host, not GPU).
    """

    @staticmethod
    def forward(ctx, block_fn, scheduler, layer_idx, sub_layer, activation, *aux_args):
        ctx.block_fn = block_fn
        ctx.scheduler = scheduler
        ctx.layer_idx = layer_idx
        ctx.sub_layer = sub_layer

        # Save activation metadata (not the tensor itself)
        ctx.activation_shape = activation.shape
        ctx.activation_dtype = activation.dtype
        ctx.activation_device = activation.device

        # Save auxiliary args - move tensors to CPU to free GPU memory
        ctx.aux_args = tuple(
            a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in aux_args
        )

        # Save RNG state for consistent dropout during recomputation
        if activation.is_cuda:
            ctx.had_cuda_rng = True
            ctx.fwd_rng_state = torch.cuda.get_rng_state(activation.device)
        else:
            ctx.had_cuda_rng = False

        # Spill primary activation to host (async D2H)
        scheduler.on_sublayer_forward(layer_idx, sub_layer, activation)

        # Compute forward without building autograd graph
        with torch.no_grad():
            output = block_fn(activation, *aux_args)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # M2: Ensure weights are on GPU for recomputation
        if ctx.scheduler is not None:
            ctx.scheduler.on_backward_layer_start(ctx.layer_idx)

        # Restore activation from host (H2D)
        activation = torch.empty(
            ctx.activation_shape,
            dtype=ctx.activation_dtype,
            device=ctx.activation_device,
        )
        ctx.scheduler.on_sublayer_backward(ctx.layer_idx, ctx.sub_layer, activation)
        activation.requires_grad_(True)

        # Restore auxiliary args to original device
        aux_args = tuple(
            a.to(ctx.activation_device) if isinstance(a, torch.Tensor) else a for a in ctx.aux_args
        )
        del ctx.aux_args

        # Recompute forward with grad enabled, using saved RNG state for
        # consistent dropout masks
        if ctx.had_cuda_rng:
            with torch.random.fork_rng(devices=[ctx.activation_device.index]):
                torch.cuda.set_rng_state(ctx.fwd_rng_state, device=ctx.activation_device)
                with torch.enable_grad():
                    output = ctx.block_fn(activation, *aux_args)
        else:
            with torch.enable_grad():
                output = ctx.block_fn(activation, *aux_args)

        # Compute gradients
        torch.autograd.backward(output, grad_output)

        # M2: Evict weights after backward recomputation.
        # Backward runs in reverse: MLP(sub=1) first, then attention(sub=0).
        # Only evict after sub_layer=0 (the last sub-block in backward order)
        # to avoid moving param.grad to CPU between sub-blocks, which causes
        # device mismatch when the next sub-block's backward accumulates.
        # For full_layer granularity (single sub_layer=0), this always evicts.
        if ctx.scheduler is not None and ctx.sub_layer == 0:
            ctx.scheduler.on_backward_layer_end(ctx.layer_idx)

        # None for block_fn, scheduler, layer_idx, sub_layer
        # activation.grad for the spilled input
        # None for each aux arg
        return (None, None, None, None, activation.grad) + (None,) * len(aux_args)


def _get_logger():
    """Get the ironcore logger if initialized, else fall back to stdlib logging."""
    try:
        from ironcore.global_vars import get_logger

        return get_logger()
    except (ImportError, AssertionError):
        return _log


@dataclass
class SpilledActivation:
    """Tracks a single spilled activation tensor."""

    # Pinned host buffer (owned by PinnedMemoryPool, returned on free)
    host_tensor: torch.Tensor
    # GPU tensor reference (for shape/dtype, not kept alive)
    shape: tuple[int, ...]
    dtype: torch.dtype
    # Transfer handle (set during D2H, consumed during backward)
    transfer_handle: object | None = None
    # Whether this activation has been consumed by backward
    consumed: bool = False


class ActivationSpillManager:
    """
    Manages activation spilling (D2H) during forward and prefetching (H2D)
    during backward.

    Two sub-layers per TransformerLayer:
      - sub_layer=0: layer input (hidden_states), needed for attention gradient
      - sub_layer=1: post-attention residual (norm_input), needed for MLP gradient

    Lifecycle per micro-batch:
      1. on_microbatch_forward_start(idx) — begin tracking
      2. on_sublayer_forward(layer, sub_layer, tensor) — D2H spill
      3. on_microbatch_forward_end() — all forward spills submitted
      4. on_microbatch_backward_start(idx) — begin backward, prefetch in reverse
      5. on_sublayer_backward(layer, sub_layer, gpu_dst) — H2D prefetch + free
      6. on_microbatch_backward_end() — all activations freed

    Args:
        pool: PinnedMemoryPool for host allocations
        engine: MemoryTransferEngine for async transfers
        num_layers: Total number of TransformerLayers
        gradient_accumulation_steps: Number of micro-batches per training step
    """

    def __init__(
        self,
        pool: PinnedMemoryPool,
        engine: MemoryTransferEngine,
        num_layers: int,
        gradient_accumulation_steps: int = 1,
    ):
        self._pool = pool
        self._engine = engine
        self._num_layers = num_layers
        self._gradient_accumulation_steps = gradient_accumulation_steps
        # Key: (microbatch_idx, layer_idx, sub_layer) -> SpilledActivation
        self._activations: dict[tuple[int, int, int], SpilledActivation] = {}
        self._current_microbatch = 0
        self._backward_microbatch = 0
        self._total_spilled_bytes = 0
        self._total_prefetched_bytes = 0

    @classmethod
    def from_config(
        cls,
        config: OffloadConfig,
        pool: PinnedMemoryPool,
        engine: MemoryTransferEngine,
        num_layers: int,
        gradient_accumulation_steps: int = 1,
    ) -> ActivationSpillManager:
        """Create an ActivationSpillManager from OffloadConfig."""
        return cls(
            pool=pool,
            engine=engine,
            num_layers=num_layers,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    # --- Forward pass ---

    def on_microbatch_forward_start(self, microbatch_idx: int) -> None:
        """Begin tracking activations for a micro-batch forward pass."""
        self._current_microbatch = microbatch_idx

    def on_sublayer_forward(
        self,
        layer_idx: int,
        sub_layer: int,
        tensor: torch.Tensor,
    ) -> None:
        """
        Spill an activation tensor to host memory (D2H).

        Submits an async D2H transfer. The tensor remains on GPU and is usable
        until the next sub-layer overwrites it. The host copy is available for
        backward prefetch.

        Args:
            layer_idx: Layer index (0 to num_layers-1)
            sub_layer: 0 for layer input, 1 for post-attention residual
            tensor: The activation tensor on GPU to spill
        """
        assert tensor.device.type == "cuda", f"Activation must be on CUDA, got {tensor.device}"
        key = (self._current_microbatch, layer_idx, sub_layer)

        # Allocate pinned host buffer matching tensor shape and dtype
        numel = tensor.numel()
        host_tensor = self._pool.allocate(numel, tensor.dtype)

        # Submit async D2H transfer
        # Use reshape(-1) instead of flatten() to avoid silent contiguous copies
        # on non-contiguous tensors (e.g. from TP slicing)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        handle = self._engine.submit_d2h(
            src=tensor.reshape(-1),
            dst=host_tensor,
        )

        self._activations[key] = SpilledActivation(
            host_tensor=host_tensor,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            transfer_handle=handle,
        )
        self._total_spilled_bytes += numel * tensor.element_size()

    def on_microbatch_forward_end(self) -> None:
        """All forward spills for current micro-batch have been submitted."""
        pass

    # --- Backward pass ---

    def on_microbatch_backward_start(self, microbatch_idx: int) -> None:
        """Begin backward pass. Activations will be prefetched in reverse order."""
        self._backward_microbatch = microbatch_idx

    def on_sublayer_backward(
        self,
        layer_idx: int,
        sub_layer: int,
        gpu_dst: torch.Tensor,
    ) -> None:
        """
        Prefetch a spilled activation back to GPU (H2D), then free host memory.

        Waits for the D2H transfer to complete (should be done by now), then
        submits an H2D transfer to copy the activation back to the provided
        GPU buffer. After the transfer completes, the pinned host memory is
        returned to the pool.

        Args:
            layer_idx: Layer index
            sub_layer: 0 for layer input, 1 for post-attention residual
            gpu_dst: GPU tensor to copy the activation into

        Returns:
            The GPU tensor with the restored activation data.
        """
        key = (self._backward_microbatch, layer_idx, sub_layer)
        activation = self._activations.get(key)

        if activation is None:
            _get_logger().warning(
                f"No spilled activation for key {key}. "
                f"Was on_sublayer_forward called during the forward pass?"
            )
            return

        # Wait for D2H to complete (should be done already)
        if activation.transfer_handle is not None:
            self._engine.wait(activation.transfer_handle)
            activation.transfer_handle = None

        # Submit H2D prefetch
        # Caller must provide a contiguous GPU buffer; non-contiguous tensors
        # would require a separate allocation, breaking the caller's contract.
        if not gpu_dst.is_contiguous():
            raise ValueError(
                f"on_sublayer_backward expects a contiguous gpu_dst, got strides={gpu_dst.stride()}"
            )
        h2d_handle = self._engine.submit_h2d(
            src=activation.host_tensor,
            dst=gpu_dst.reshape(-1),
        )
        self._engine.wait(h2d_handle)
        self._engine.synchronize_with_default_stream()

        self._total_prefetched_bytes += (
            activation.host_tensor.numel() * activation.host_tensor.element_size()
        )

        # Free-after-consume: return pinned memory to pool
        self._pool.free(activation.host_tensor)
        activation.consumed = True
        del self._activations[key]

    def on_microbatch_backward_end(self) -> None:
        """All activations for current micro-batch have been consumed and freed."""
        pass

    # --- Lifecycle ---

    def on_training_step_end(self) -> None:
        """
        Called after all micro-batches' forward and backward passes complete.

        Ensures all pending transfers are synchronized. Should be no-op if
        free-after-consume worked correctly.
        """
        self._engine.synchronize()

        # Safety: free any activations that weren't consumed (shouldn't happen)
        if self._activations:
            _get_logger().warning(
                f"ActivationSpillManager: {len(self._activations)} activations "
                f"not consumed during backward. Freeing."
            )
            for _key, activation in self._activations.items():
                if not activation.consumed:
                    if activation.transfer_handle is not None:
                        self._engine.wait(activation.transfer_handle)
                    self._pool.free(activation.host_tensor)
            self._activations.clear()

    def shutdown(self) -> None:
        """Release all resources held by the spill manager."""
        self.on_training_step_end()

    # --- Properties ---

    @property
    def pending_count(self) -> int:
        """Number of spilled activations awaiting backward consumption."""
        return len(self._activations)

    @property
    def total_spilled_bytes(self) -> int:
        return self._total_spilled_bytes

    @property
    def total_prefetched_bytes(self) -> int:
        return self._total_prefetched_bytes

    def __repr__(self) -> str:
        return (
            f"ActivationSpillManager("
            f"layers={self._num_layers}, "
            f"pending={self.pending_count}, "
            f"spilled={self._total_spilled_bytes / 1024**3:.1f}GB, "
            f"prefetched={self._total_prefetched_bytes / 1024**3:.1f}GB)"
        )
