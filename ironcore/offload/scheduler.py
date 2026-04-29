# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Execution scheduler for weight streaming during training.

Orchestrates the lifecycle of weight transfers around the forward/backward pass.
Prefetches weights N layers ahead of compute, evicts weights after use.

M2 scope: weight streaming. The scheduler is called from:
  - BaseTrainer: on_step_start / on_step_end lifecycle
  - TransformerModel.forward(): per-layer on_layer_start / on_layer_end hooks

Integration contract:
  1. Trainer calls scheduler.on_training_step_start() before gradient accumulation
  2. TransformerModel.forward() calls scheduler.on_layer_start(i) before layer i
  3. TransformerModel.forward() calls scheduler.on_layer_end(i) after layer i
  4. Trainer calls scheduler.on_training_step_end() after optimizer step

Thread safety: not thread-safe. All calls from the training loop's main thread.
"""

from __future__ import annotations

import logging
import time

import torch
from torch import nn

from ironcore.offload.gpu_staging_pool import GPUStagingPool
from ironcore.offload.hooks import ActivationSpillManager
from ironcore.offload.memory_pool import PinnedMemoryPool
from ironcore.offload.tile_manager import TileManager, WeightGroup
from ironcore.offload.transfer_engine import MemoryTransferEngine

_log = logging.getLogger(__name__)


def _get_logger():
    """Get the ironcore logger if initialized, else fall back to stdlib logging."""
    try:
        from ironcore.global_vars import get_logger

        return get_logger()
    except (ImportError, AssertionError):
        return _log


def _is_offloadable_param(param: nn.Parameter) -> bool:
    """Check if a parameter should be streamed. LoRA params are excluded."""
    return getattr(param, "offloadable", True)


def _collect_layer_params(layer: nn.Module) -> list[nn.Parameter]:
    """
    Collect all offloadable parameters from a TransformerLayer.

    Walks the module tree to find all nn.Parameter objects that should be
    streamed. Excludes:
      - Parameters with offloadable=False (LoRA adapters)
      - Buffers (not parameters)
      - Parameters already on CPU (shouldn't happen, but defensive)
    """
    params = []
    for module in layer.modules():
        for param in module.parameters(recurse=False):
            if _is_offloadable_param(param) and param.device.type == "cuda":
                params.append(param)
    return params


class ExecutionScheduler:
    """
    Weight streaming scheduler for training.

    Manages the lifecycle of weight transfers around the training loop.
    Prefetches layer weights asynchronously before they are needed, and
    evicts them from GPU after use to free VRAM for the next layer.

    Args:
        model: The TransformerModel
        pool: PinnedMemoryPool for host allocations
        engine: MemoryTransferEngine for async transfers
        tile_manager: TileManager for weight tiling
        prefetch_layers: Number of layers to prefetch ahead (default 2)
        device: Target CUDA device
    """

    def __init__(
        self,
        model: nn.Module,
        pool: PinnedMemoryPool,
        engine: MemoryTransferEngine,
        tile_manager: TileManager,
        prefetch_layers: int = 2,
        device: torch.device | None = None,
        spill_manager: ActivationSpillManager | None = None,
        gpu_pool: GPUStagingPool | None = None,
    ):
        self._model = model
        self._pool = pool
        self._engine = engine
        self._tile_manager = tile_manager
        self._prefetch_layers = prefetch_layers
        self._device = device or torch.device("cuda")
        self._spill_manager = spill_manager
        self._gpu_pool = gpu_pool

        # Populated during init
        self._num_layers = 0
        self._weight_groups: dict[int, WeightGroup] = {}
        self._layer_on_gpu: set[int] = set()

        # State tracking
        self._step_count = 0
        self._current_forward_layer = -1
        self._current_microbatch = 0

        # Per-step timing accumulators (reset each step)
        self._step_start: float = 0.0
        self._h2d_time: float = 0.0  # total H2D transfer wait time
        self._snapshot_time: float = 0.0  # D2H snapshot time
        self._prefetch_wait_time: float = 0.0  # time waiting for prefetch completes

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        config: object,
        device: torch.device | None = None,
    ) -> ExecutionScheduler | None:
        """
        Create and initialize a scheduler from a model and OffloadConfig.

        Returns None if weight streaming is not applicable (e.g. FSDP enabled,
        no CUDA, model too small to benefit).

        Args:
            model: The TransformerModel (after parallelism wrapping)
            config: OffloadConfig
            device: Target CUDA device
        """
        from ironcore.offload.config import OffloadConfig

        assert isinstance(config, OffloadConfig)

        if not config.enabled:
            return None

        # Need at least one offload feature enabled
        if not config.weight_offload and not config.activation_spill:
            return None

        if not torch.cuda.is_available():
            return None

        # Skip weight streaming if FSDP is enabled (FSDP manages its own parameter movement)
        # Note: activation spilling can still work with FSDP (spills activations, not weights)
        if config.weight_offload and _is_fsdp_enabled(model, config):
            _get_logger().info(
                "Weight streaming skipped: FSDP is enabled. "
                "FSDP manages its own parameter sharding/unsharding."
            )
            return None

        # Skip weight streaming if activation checkpointing is enabled.
        # Checkpointing replays the forward pass during backward, and the
        # scheduler's per-layer hooks only fire in TransformerModel.forward(),
        # not during recomputation. Weights must stay resident for correctness.
        # Note: activation spilling itself disables checkpointing via config validation.
        if config.weight_offload and _is_checkpointing_enabled(model):
            _get_logger().info(
                "Weight streaming skipped: activation checkpointing is enabled. "
                "Weights must stay on GPU for backward recomputation."
            )
            return None

        if device is None:
            device = torch.device("cuda")

        pool = PinnedMemoryPool.from_config(config)
        engine = MemoryTransferEngine.from_config(config, device)

        gpu_pool = None
        if config.weight_offload:
            gpu_pool = GPUStagingPool.from_config(config, device)

        tile_manager = TileManager.from_config(config, pool, device, gpu_pool=gpu_pool)

        # M3: Create activation spill manager if enabled
        spill_manager = None
        if config.activation_spill:
            from ironcore.models.transformer import TransformerModel

            num_layers = len(model.layers) if isinstance(model, TransformerModel) else 0
            spill_manager = ActivationSpillManager.from_config(
                config=config,
                pool=pool,
                engine=engine,
                num_layers=num_layers,
                gradient_accumulation_steps=1,  # Updated by trainer after init
            )

        scheduler = cls(
            model=model,
            pool=pool,
            engine=engine,
            tile_manager=tile_manager,
            prefetch_layers=config.weight_prefetch_layers,
            device=device,
            spill_manager=spill_manager,
            gpu_pool=gpu_pool,
        )
        scheduler._register_all_layers()
        scheduler._propagate_to_layers()

        # Auto-size GPU pool based on registered layer sizes
        if gpu_pool is not None:
            layer_byte_sizes = [g.total_gpu_bytes for g in scheduler._weight_groups.values()]
            gpu_pool.auto_size(layer_byte_sizes, config.weight_prefetch_layers)

        return scheduler

    def _register_all_layers(self) -> None:
        """Register all TransformerLayers for weight streaming."""
        from ironcore.models.transformer import TransformerModel

        if not isinstance(self._model, TransformerModel):
            _get_logger().warning(
                f"Weight streaming: model is {type(self._model).__name__}, "
                f"not TransformerModel. Skipping."
            )
            return

        layers = self._model.layers
        self._num_layers = len(layers)

        logger = _get_logger()
        total_params = 0

        for i, layer in enumerate(layers):
            params = _collect_layer_params(layer)
            if params:
                group = self._tile_manager.register_layer(i, params)
                self._weight_groups[i] = group
                total_params += sum(p.numel() for p in params)

        if not self._weight_groups:
            _get_logger().warning("Weight streaming: no offloadable parameters found.")
            return

        param_bytes = total_params * 4  # assume fp32 for estimate
        logger.info(
            f"Weight streaming initialized: "
            f"{len(self._weight_groups)}/{self._num_layers} layers registered, "
            f"{total_params:,} parameters ({param_bytes / 1024**3:.1f}GB fp32), "
            f"prefetch_ahead={self._prefetch_layers}, "
            f"pool={self._pool}"
        )

    def _propagate_to_layers(self) -> None:
        """Attach scheduler reference to each TransformerLayer for in-forward hooks."""
        from ironcore.models.transformer import TransformerModel

        if isinstance(self._model, TransformerModel):
            for layer in self._model.layers:
                layer._offload_scheduler = self

    # --- Training loop lifecycle hooks ---

    def on_training_step_start(self) -> None:
        """
        Called at the start of each training step (before gradient accumulation).

        Issues prefetch for the first N layers so they are ready when the
        forward pass starts.
        """
        self._step_count += 1
        self._current_forward_layer = -1
        self._layer_on_gpu.clear()

        # Reset per-step timing
        self._step_start = time.monotonic()
        self._h2d_time = 0.0
        self._snapshot_time = 0.0
        self._prefetch_wait_time = 0.0

        # Prefetch first N layers
        t0 = time.monotonic()
        for i in range(min(self._prefetch_layers, self._num_layers)):
            self._prefetch_layer(i)
        self._h2d_time += time.monotonic() - t0

    def on_training_step_end(self) -> None:
        """
        Called after optimizer step completes.

        Snapshots updated parameter values back to host memory, then
        synchronizes all pending transfers. This ensures the host-side
        copies reflect the optimizer's updates before the next step
        prefetches from host.
        """
        # Snapshot updated params (after optimizer step) back to host
        t0 = time.monotonic()
        for group in self._weight_groups.values():
            self._tile_manager.snapshot_params_to_host(group)

        self._engine.synchronize()
        self._snapshot_time = time.monotonic() - t0

        # M3: Clean up any remaining spilled activations
        if self._spill_manager is not None:
            self._spill_manager.on_training_step_end()

    # --- Forward pass per-layer hooks ---

    def on_layer_start(self, layer_idx: int) -> None:
        """
        Called before layer `layer_idx` executes in the forward pass.

        Waits for this layer's weight transfer to complete, applies weights
        to parameters via in-place .data copy, then prefetches the next
        layers.
        """
        group = self._weight_groups.get(layer_idx)
        if group is None:
            return

        # Skip if already loaded from a prior micro-batch
        if layer_idx in self._layer_on_gpu:
            return

        # Wait for this layer's transfer
        t0 = time.monotonic()
        for tile in group.tiles:
            if tile.transfer_handle is not None:
                self._engine.wait(tile.transfer_handle)
                tile.transfer_handle = None
        self._prefetch_wait_time += time.monotonic() - t0

        # Ensure default stream sees the transferred data
        self._engine.synchronize_with_default_stream()

        # Apply weights to parameters
        self._tile_manager.apply_tiles_to_params(group)
        self._layer_on_gpu.add(layer_idx)

        # Return staging buffers to pool -- content is now in param.data
        if self._gpu_pool is not None:
            self._tile_manager.return_gpu_buffers(group)

        # Prefetch next layers
        for ahead in range(1, self._prefetch_layers + 1):
            next_idx = layer_idx + ahead
            if next_idx < self._num_layers and next_idx not in self._layer_on_gpu:
                self._prefetch_layer(next_idx)

        self._current_forward_layer = layer_idx

    def on_layer_end(self, layer_idx: int) -> None:
        """
        Called after layer `layer_idx` completes in the forward pass.

        For training: weights stay on GPU for the backward pass.
        Eviction happens after the backward pass completes.
        """
        pass

    def on_backward_layer_start(self, layer_idx: int) -> None:
        """
        Called before layer `layer_idx` during backward pass.

        For activation checkpointing: weights may have been evicted and need
        to be reloaded for recomputation. For standard backward: weights
        should already be on GPU.
        """
        group = self._weight_groups.get(layer_idx)
        if group is None:
            return

        if layer_idx not in self._layer_on_gpu:
            # Weights were evicted, need to reload for backward
            self._prefetch_layer(layer_idx)
            for tile in group.tiles:
                if tile.transfer_handle is not None:
                    self._engine.wait(tile.transfer_handle)
                    tile.transfer_handle = None
            self._engine.synchronize_with_default_stream()
            self._tile_manager.apply_tiles_to_params(group)
            self._layer_on_gpu.add(layer_idx)

    def on_backward_layer_end(self, layer_idx: int) -> None:
        """
        Called after layer `layer_idx` during backward pass.

        Weights are no longer needed for this step. Evict from GPU by
        clearing the GPU staging buffers (they'll be refilled next step).
        """
        group = self._weight_groups.get(layer_idx)
        if group is None:
            return

        # Mark as evicted
        self._layer_on_gpu.discard(layer_idx)

    def on_backward_pass_end(self) -> None:
        """
        Called after the entire backward pass completes (all layers).
        Staging buffers already returned to pool in on_layer_start.
        """
        self._engine.synchronize()
        self._layer_on_gpu.clear()

    # --- M3: Activation spilling lifecycle ---

    def set_gradient_accumulation_steps(self, steps: int) -> None:
        """Update gradient accumulation steps (called by trainer after init)."""
        self._current_microbatch = 0
        if self._spill_manager is not None:
            self._spill_manager._gradient_accumulation_steps = steps

    def on_microbatch_forward_start(self, microbatch_idx: int) -> None:
        """Called before a micro-batch forward pass."""
        self._current_microbatch = microbatch_idx
        if self._spill_manager is not None:
            self._spill_manager.on_microbatch_forward_start(microbatch_idx)

    def on_sublayer_forward(self, layer_idx: int, sub_layer: int, tensor: torch.Tensor) -> None:
        """
        Spill an activation to host during forward.

        Called from TransformerLayer.custom_forward at sub-layer boundaries:
          - sub_layer=0: layer input (hidden_states)
          - sub_layer=1: post-attention residual (norm_input)
        """
        if self._spill_manager is not None:
            self._spill_manager.on_sublayer_forward(layer_idx, sub_layer, tensor)

    def on_microbatch_forward_end(self) -> None:
        """Called after a micro-batch forward pass completes."""
        if self._spill_manager is not None:
            self._spill_manager.on_microbatch_forward_end()

    def on_microbatch_backward_start(self, microbatch_idx: int) -> None:
        """Called before a micro-batch backward pass."""
        if self._spill_manager is not None:
            self._spill_manager.on_microbatch_backward_start(microbatch_idx)

    def on_sublayer_backward(self, layer_idx: int, sub_layer: int, gpu_dst: torch.Tensor) -> None:
        """
        Prefetch a spilled activation from host during backward.

        Called from TransformerLayer backward hooks at sub-layer boundaries.
        """
        if self._spill_manager is not None:
            self._spill_manager.on_sublayer_backward(layer_idx, sub_layer, gpu_dst)

    def on_microbatch_backward_end(self) -> None:
        """Called after a micro-batch backward pass completes."""
        if self._spill_manager is not None:
            self._spill_manager.on_microbatch_backward_end()

    @property
    def spill_manager(self) -> ActivationSpillManager | None:
        """Access the activation spill manager (None if not enabled)."""
        return self._spill_manager

    # --- Internal methods ---

    def _prefetch_layer(self, layer_idx: int) -> None:
        """Issue async H2D transfer for a layer's weights."""
        group = self._weight_groups.get(layer_idx)
        if group is None:
            return

        # Skip if already on GPU or already in-flight (pending transfer)
        if layer_idx in self._layer_on_gpu:
            return
        if any(tile.transfer_handle is not None for tile in group.tiles):
            return

        # Borrow GPU staging buffers from the pool
        if self._gpu_pool is not None:
            self._tile_manager.borrow_gpu_buffers(group)

        for tile in group.tiles:
            handle = self._engine.submit_h2d(
                src=tile.host_tensor,
                dst=tile.gpu_tensor,
            )
            tile.transfer_handle = handle

    @property
    def num_registered_layers(self) -> int:
        return len(self._weight_groups)

    @property
    def is_active(self) -> bool:
        return len(self._weight_groups) > 0

    def get_group(self, layer_idx: int) -> WeightGroup | None:
        """Get the WeightGroup for a layer, or None if not registered."""
        return self._weight_groups.get(layer_idx)

    def get_metrics(self) -> dict[str, float]:
        """Return per-step offload timing and memory metrics.

        Returns dict with:
          - offload_overhead_ms: total wall time spent in offload operations
          - h2d_ms: time issuing H2D transfers (prefetch)
          - d2h_snapshot_ms: time snapshotting updated params back to host
          - prefetch_wait_ms: time waiting for prefetch transfers to complete
          - host_pool_used_mb: pinned pool memory in use
          - host_pool_total_mb: pinned pool total capacity
        """
        step_elapsed = time.monotonic() - self._step_start if self._step_start > 0 else 0.0
        return {
            "offload_overhead_ms": step_elapsed * 1000,
            "h2d_ms": self._h2d_time * 1000,
            "d2h_snapshot_ms": self._snapshot_time * 1000,
            "prefetch_wait_ms": self._prefetch_wait_time * 1000,
            "host_pool_used_mb": self._pool.total_used_bytes / (1024 * 1024),
            "host_pool_total_mb": self._pool.total_allocated_bytes / (1024 * 1024),
            "gpu_staging_used_mb": self._gpu_pool.total_used_bytes / (1024 * 1024)
            if self._gpu_pool
            else 0,
            "gpu_staging_total_mb": self._gpu_pool.total_allocated_bytes / (1024 * 1024)
            if self._gpu_pool
            else 0,
        }

    def shutdown(self) -> None:
        """
        Release all resources held by the scheduler.

        Frees GPU staging buffers and pinned host memory. Call this when
        training is complete or the scheduler is no longer needed.
        """
        self._engine.synchronize()

        # Free GPU staging buffers
        for group in self._weight_groups.values():
            for tile in group.tiles:
                if tile.gpu_tensor is not None and self._gpu_pool is not None:
                    self._gpu_pool.free(tile.gpu_tensor)
                tile.gpu_tensor = None  # type: ignore[assignment]
                tile.host_tensor = None  # type: ignore[assignment]
                tile.transfer_handle = None

        self._weight_groups.clear()
        self._layer_on_gpu.clear()
        self._gpu_pool = None
        if self._spill_manager is not None:
            self._spill_manager.shutdown()
            self._spill_manager = None

        # Release model reference
        self._model = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"ExecutionScheduler("
            f"layers={self.num_registered_layers}/{self._num_layers}, "
            f"prefetch_ahead={self._prefetch_layers}, "
            f"step={self._step_count})"
        )


def _is_fsdp_enabled(model: nn.Module, config: object) -> bool:
    """Check if FSDP is enabled in the config."""
    # Walk the config to find use_fsdp
    # The OffloadConfig doesn't have this, but MainConfig does via parallel
    # We check the model for FSDP wrapping instead
    from torch.distributed.fsdp import FullyShardedDataParallel

    for module in model.modules():
        if isinstance(module, FullyShardedDataParallel):
            return True
    return False


def _is_checkpointing_enabled(model: nn.Module) -> bool:
    """Check if activation checkpointing is enabled on the model."""
    if hasattr(model, "activation_recompute"):
        return bool(model.activation_recompute)
    return False
