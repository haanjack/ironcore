# Offload (RAM-First Staircase Scaling)

> This is the user-facing guide (how to configure and run offload). For the architecture,
> memory analysis, and how offload relates to FSDP, see the
> [Offload system design](design/offload.md).

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

Layer recomputation supports weight streaming with either CPU or GPU input boundaries. `weight_offload` enables safe graph-free layer replay but does not enable `activation_spill`. Choose the boundary storage explicitly.

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

`activation_spill: true` selects CPU storage for recompute inputs; false keeps them on GPU. This setting is independent of `operation.recompute_interval`. It does not disable `operation.activation_recompute`.

Granularity: `sub_layer` (default) or `full_layer`. Grouped intervals greater than one require `full_layer`.

### Recompute interval

```yaml
operation:
  activation_recompute: true
  recompute_interval: 2
offload:
  enabled: true
  weight_offload: true
  activation_spill: true
  activation_spill_granularity: full_layer
```

`recompute_interval` is the number of Transformer layers in one recompute segment. Default `1` preserves the existing layer boundaries; `2` stores one input for each pair of layers; `0` disables whole-layer recomputation. With GPU-resident weights, enable `operation.activation_recompute` explicitly. Weight streaming requires a positive interval and enables replay automatically. CPU activation storage is never enabled automatically.

### Omit a specified number of input boundaries

`operation.recompute_skip_count` specifies how many **layer input checkpoint boundaries** to omit. The default `null` selects interval mode. Count mode requires `recompute_interval: 1` (the default) and `0 <= recompute_skip_count < num_layers`. Other intervals conflict with count mode. The first input is always retained for replay.

```yaml
operation:
  activation_recompute: true
  recompute_skip_count: 8
offload:
  activation_spill: false  # true selects CPU storage instead
```

For 32 layers, count 8 retains 24 input boundaries: eight pairs and sixteen single-layer segments, evenly distributed. Segment lengths differ by at most one; larger counts necessarily create longer segments. Single-layer segments use only one backward replay, without the extra segment reconstruction. Backward weight prefetch resumes at the actual previous segment's start, including nonuniform segment lengths.

This does not mean only eight layer interiors recompute. All interiors still recompute; count mode controls additional input reconstruction. It works with GPU checkpoints independently of CPU spilling. Omitted CPU boundaries require `full_layer` granularity. Grouped FSDP remains unsupported. Manual index selection is not implemented. The public example's `--recompute-skip-count` and host budget estimate use the same segment plan.

The interval also works with all weights and inputs on GPU (`offload.enabled: false`). GPU checkpointing groups layers into a uniform segment, including a shorter final segment. With streamed weights, backward first rebuilds that segment's inputs using GPU layer checkpoints, then reloads and evicts weights one layer at a time. This reduces original host boundaries while bounding resident weights and internal activation graphs. GPU boundary memory and replay work still increase with segment size.

Grouped FSDP intervals are not yet supported. Gemma offload still requires no PLE or shared KV; GPU-resident Gemma recompute supports both. MLP/MoE internal blocking and recomputation remain separate from this whole-layer interval.

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
