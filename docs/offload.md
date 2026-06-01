# Offload (RAM-First Staircase Scaling)

## Overview

The offload subsystem moves tensors between GPU VRAM and host RAM to train models that don't fit entirely in VRAM. Configured via `offload:` section in YAML config.

Three independent features, each gated on `offload.enabled: true`:

| Feature | What moves | Direction |
|---|---|---|
| Optimizer state offload | AdamW/Muon momentum, variance states | GPU → CPU (after step), CPU → GPU (before step) |
| Weight streaming | Layer weights (attention, MLP projections) | CPU → GPU (prefetch before compute), GPU → CPU (snapshot after step) |
| Activation spilling | Intermediate activations at sub-layer boundaries | GPU → CPU (forward), CPU → GPU (backward) |

Each can be enabled independently. All three can run simultaneously.

## Quick start

```yaml
offload:
  enabled: true

  # Optimizer state offload
  optimizer_offload: true

  # Weight streaming
  weight_offload: true
  weight_prefetch_layers: 2
  weight_storage_precision: fp32

  # Activation spilling
  activation_spill: false
```

## Optimizer state offload

After the optimizer step, momentum and variance states are moved to CPU RAM. Before the next step, they are moved back to GPU. Parameters with fewer than `optimizer_min_param_elements` (default 65536) elements stay on GPU (not worth the transfer overhead).

Optimizer state offload supports fp32, fp16, and bf16 precision for optimizer states via `optimizer_state_precision` (default: fp32). Lower precision reduces RAM usage and PCIe bandwidth but may affect training stability for AdamW.

## Weight streaming

### How it works

Layer weights live in pinned host memory. During the forward pass, weights are prefetched to GPU asynchronously `weight_prefetch_layers` layers ahead of compute. After a layer executes, its staging buffer is returned to the pool for reuse. After the backward pass completes, updated weights are snapshotted back to host memory.

Lifecycle per training step:

1. **`on_training_step_start()`**: Prefetch first N layers
2. **`on_layer_start(i)`**: Wait for layer i's transfer, apply weights to params (swap CPU param.data for GPU staging buffer), prefetch next layers
3. **`on_layer_end(i)`**: With activation spilling active (auto-enabled): evict weights from GPU — replace param.data with host tile values, return GPU staging buffer to pool. Without activation spilling: no-op (weights stay for backward).
4. **`on_backward_layer_start(i)`**: Reload evicted weights if needed for backward recomputation
5. **`on_backward_layer_end(i)`**: Evict weights again after backward computation
6. **`on_training_step_end()`**: Snapshot updated params (after optimizer step) back to host tiles

### GPU staging buffer pool

Weight streaming uses a **pooled allocation** strategy. Instead of allocating permanent GPU staging buffers for every layer at init, a fixed-size pool of CUDA memory is shared across layers. Buffers are borrowed before H2D transfer and returned immediately after weights are copied into `param.data`.

This means only `(weight_prefetch_layers + 1)` layers' worth of GPU staging memory is needed at any point, regardless of total layer count. For a 36-layer model with `weight_prefetch_layers: 2`, only 3 layers' worth of staging buffers exist.

### Pool sizing

The pool capacity is auto-sized by default. After all layers register their weights, the scheduler computes:

```
budget = max(sum of consecutive layer_bytes[i:i+prefetch_layers+1])
```

This sliding-window approach finds the largest sum of `weight_prefetch_layers + 1` consecutive layers, which accounts for varying layer sizes (e.g., MoE layers with larger expert parameters). Chunk size is also auto-sized to at least the largest single layer's bytes, so each layer fits in a single chunk.

**Manual override** via config:

```yaml
offload:
  gpu_staging_pool_mb: 512.0   # hard cap on pool VRAM (0 = auto-size)
  gpu_staging_chunk_mb: 256.0  # CUDA chunk size for pool allocations
```

When manually set, the pool raises `RuntimeError: GPUStagingPool budget exceeded` if allocations would exceed the cap. This can happen if layers vary significantly in size or if prefetch timing causes more than `prefetch_layers + 1` layers to be in-flight.

**Recommendation**: leave `gpu_staging_pool_mb: 0` (auto-size) unless you see budget errors, then set it to `1.1x` the auto-computed value for headroom.

### Storage precision

Weights on host can be stored at lower precision to reduce RAM usage and PCIe bandwidth:

```yaml
offload:
  weight_storage_precision: bf16  # fp32, fp16, or bf16
```

Weights are dequantized back to the original dtype when copied to GPU. The tradeoff is memory savings vs. potential precision loss.

### Incompatibilities

Weight streaming is **incompatible** with:
- **FSDP** (`parallel.use_fsdp: true`): FSDP manages its own parameter sharding/unsharding.
- **Activation checkpointing** (`model.activation_recompute: true`): Checkpointing replays the forward pass during backward, but scheduler hooks only fire in the main forward pass. Weights must stay resident for correctness.

Activation spilling is a replacement for checkpointing that is compatible with weight streaming.

### CPU-resident parameters and optimizer

When `weight_offload: true`, the model is kept on CPU. Layer weights are temporarily swapped to GPU staging buffers during forward/backward via `param.data` replacement (preserving `nn.Parameter` identity for optimizer references). After each layer executes, `param.data` is restored to a CPU tensor backed by host tile values.

Because parameters and optimizer states (exp_avg, exp_avg_sq) are CPU-resident, the AdamW optimizer step runs entirely on CPU. This is **20-40x slower** than GPU AdamW but is inherent to the design — the GPU does not have enough VRAM to hold both weights and optimizer states. The tradeoff is acceptable because:

1. The optimizer step is a small fraction of total step time (forward + backward dominate).
2. The alternative (no offload) is OOM — there is no GPU-only path for models exceeding VRAM.
3. CPU AdamW correctness is numerically identical to GPU AdamW (same float32 accumulation).

### Eviction and snapshot optimization

During forward and backward, each layer's weights are evicted from GPU after execution. The eviction:

1. Replaces `param.data` with the host tile values (not uninitialized memory).
2. Returns the GPU staging buffer to the pool for reuse by other layers.
3. **Does not** D2H-snapshot the GPU values back to host tiles — weights are read-only during forward/backward, so the host tiles already contain the correct values.

The D2H snapshot to host tiles happens only once per step, in `on_training_step_end()`, after the optimizer updates `param.data` on CPU. This saves ~2 redundant D2H copies per layer per step (~5% step time improvement at 13B scale).

## Activation spilling

When enabled, intermediate activations are spilled to CPU at sub-layer boundaries during the forward pass (layer input, post-attention residual) and prefetched back during the backward pass. Freed after consumption to limit host memory.

Replaces activation checkpointing. Enabling `activation_spill: true` automatically disables `activation_recompute` with a warning.

Granularity: `sub_layer` (default and only option). Spills at attention/MLP boundaries for lower host memory usage.

## Memory pools

### PinnedMemoryPool (host)

Pre-allocated pinned (page-locked) host memory for DMA transfers. Enables async PCIe transfers via CUDA streams. Fixed-size chunks with free-list allocator and coalescing.

Config:
```yaml
offload:
  pinned_memory_pool_gb: 100.0  # total host memory budget
  pinned_chunk_gb: 4.0          # chunk size
```

### GPUStagingPool (device)

Pre-allocated CUDA memory for weight staging buffers. Same chunk + free-list pattern as PinnedMemoryPool, but on GPU. Thread-safe via `threading.Lock`.

Config:
```yaml
offload:
  gpu_staging_pool_mb: 0.0      # 0 = auto-size
  gpu_staging_chunk_mb: 256.0   # chunk size
```

## Implementation

Key files:

| File | Purpose |
|---|---|
| `ironcore/offload/config.py` | `OffloadConfig` dataclass |
| `ironcore/offload/memory_pool.py` | `PinnedMemoryPool`, `_PinnedChunk` |
| `ironcore/offload/gpu_staging_pool.py` | `GPUStagingPool`, `_GPUChunk` |
| `ironcore/offload/tile_manager.py` | `WeightTile`, `WeightGroup`, `TileManager` |
| `ironcore/offload/transfer_engine.py` | `MemoryTransferEngine` (async H2D/D2H) |
| `ironcore/offload/scheduler.py` | `ExecutionScheduler` (lifecycle orchestration) |
| `ironcore/offload/hooks.py` | `ActivationSpillManager` (activation_spill) |
