# Offload System Design

## Overview

The offload subsystem enables training models that exceed GPU VRAM by moving optimizer states, model weights, and activations between GPU and host RAM. Three independent mechanisms — optimizer offload, weight streaming, activation spilling — can be combined orthogonally.

## Target Hardware

Consumer desktop with single GPU (8–24 GB VRAM), 32–128 GB host RAM. Single-node only. No cross-node communication.

## Architecture

```
ExecutionScheduler                     # Central coordinator
  |
  +-- PinnedMemoryPool                 # Host page-locked memory allocator
  +-- GPUStagingPool                   # GPU pre-allocated memory allocator
  +-- MemoryTransferEngine             # Async H2D/D2H DMA on CUDA streams
  +-- TileManager                      # Weight tiling, precision conversion, reassembly
  +-- ActivationSpillManager           # Forward D2H spill + backward H2D prefetch
```

### ExecutionScheduler

Created via `ExecutionScheduler.from_model(model, config, device)`. Owns all sub-components and orchestrates data movement across the training loop. The trainer calls hooks at each phase:

```
on_training_step_start()              # Prefetch first N layers
  for each micro-batch:
    on_microbatch_forward_start(i)     # Begin tracking activations
    [forward pass — per-layer hooks]
    on_microbatch_forward_end(i)
    on_microbatch_backward_start(i)
    [backward pass — per-layer hooks]
    on_microbatch_backward_end(i)
  on_backward_pass_end()               # Move grads to CPU, evict weights
  [optimizer step — CPU]
  on_training_step_end()               # Snapshot updated params to host
```

### PinnedMemoryPool

Host-side allocator using `cudaMallocHost` (page-locked memory for async DMA). Fixed-size chunks with free-list coalescing. Shared by weight tiles and spilled activations. Budget enforced — allocation fails hard if exceeded. Thread-safe via `threading.Lock`.

### GPUStagingPool

GPU-side mirror of PinnedMemoryPool. Auto-sizes to hold `prefetch_layers + 1` consecutive layer weights using a sliding-window max over layer sizes. Staging buffers are borrowed by TileManager, used as temporary `param.data`, then returned.

### MemoryTransferEngine

Manages async DMA transfers on dedicated CUDA streams. Each transfer creates a `TransferHandle` with a `torch.cuda.Event` for synchronization. Stream barriers ensure DMA does not race with compute on the default stream.

### TileManager

Manages per-layer `WeightGroup` objects. Each parameter in a layer becomes one `WeightTile`. Allocates pinned host buffers in `weight_storage_precision` (e.g., bf16) and copies initial weights with precision conversion. During forward/backward, borrows GPU staging buffers and swaps `param.data` to GPU views while preserving `nn.Parameter` identity.

### ActivationSpillManager

Tracks spilled activations keyed by `(microbatch_idx, layer_idx, sub_layer)`. Forward: allocates pinned buffer, submits async D2H. Backward: waits for D2H, submits H2D to restore, frees buffer immediately (free-after-consume). Peak host memory bounded because activations are freed as backward consumes them.

---

## Optimizer State Offload

Moves optimizer states (momentum, variance) to CPU RAM. Parameters remain on GPU.

### Data flow

```
GPU                              CPU
┌──────────┐                 ┌──────────────────┐
│ params   │ ──grad D2H──→  │ exp_avg (bf16)   │
│          │                 │ exp_avg_sq (bf16) │
│          │ ←──delta H2D── │ AdamW compute     │
└──────────┘                 └──────────────────┘
```

### Implementation

File: `ironcore/offload/optimizer_helpers.py`

Two compute paths in `_adamw_offloaded_step()`:

1. **GPU params (optimizer offload only):** Grad transferred GPU→CPU, AdamW math on CPU (SIMD/AVX-512), delta transferred CPU→GPU. States never leave CPU.

2. **CPU params (optimizer + weight offload combined mode):** Both params and states on CPU. `.to()` is a no-op. Math runs natively on CPU. After update, states written back to host in `state_dtype`.

Shared by both `AdamWOptimizer.step()` and `MuonOptimizer._step_adamw()`.

### Config

| Field | Default | Description |
|-------|---------|-------------|
| `optimizer_offload` | `false` | Enable optimizer offload |
| `optimizer_state_precision` | `"fp32"` | Storage dtype: `"fp32"` or `"bf16"` |
| `optimizer_min_param_elements` | `65536` | Skip offload for params smaller than this |

### VRAM savings

Removes optimizer states from GPU: ~2× model size in fp32, ~1× model size in bf16.

### Measured performance

| Model | Mode | Step Time | Throughput |
|-------|------|-----------|------------|
| 3B | optimizer offload | 11.2s | 5.4/min |

---

## Weight Streaming

Streams model weights to GPU layer-by-layer during forward/backward, keeping them on CPU between uses. Only the current layer + prefetch window reside on GPU simultaneously.

### Data flow

```
Forward pass (layer i):
                    Prefetch window
                    ┌─────────────┐
CPU                 │ GPU          │
┌───────┐          │ ┌─────────┐  │
│ L(i-1)│          │ │ L(i)    │  │ ← executing
│ L(i)  │ ──H2D──→│ │ L(i+1)  │  │ ← prefetching
│ L(i+1)│          │ │ L(i+2)  │  │ ← prefetching
│ L(i+2)│          │ └─────────┘  │
│ ...   │          │              │
└───────┘          └─────────────┘

After layer i completes:
  - L(i) evicted from GPU (buffer returned to staging pool)
  - L(i+3) prefetched from CPU
```

### Implementation

Files: `scheduler.py`, `tile_manager.py`, `gpu_staging_pool.py`

1. **Init:** Model stays on CPU. TileManager registers all layers, allocates pinned host buffers in `weight_storage_precision`, copies initial weights.

2. **Forward:** `on_layer_start(i)` waits for prefetch transfer, swaps `param.data` to GPU staging buffer view, prefetches next N layers. `on_layer_end(i)` optionally evicts weights.

3. **Backward:** Layers traversed in reverse. `on_backward_layer_start(i)` reloads weights. `on_backward_layer_end(i)` evicts and prefetches previous layers.

4. **Step end:** `snapshot_params_to_host()` copies updated params back to host tiles.

### Config

| Field | Default | Description |
|-------|---------|-------------|
| `weight_offload` | `false` | Enable weight streaming |
| `weight_prefetch_layers` | `2` | Forward lookahead depth |
| `backward_weight_prefetch_layers` | `1` | Backward prefetch depth |
| `weight_storage_precision` | `"bf16"` | Host storage dtype |
| `gpu_staging_pool_mb` | `0` | Max GPU staging (0 = auto) |
| `gpu_staging_chunk_mb` | `256` | GPU staging chunk size |

### VRAM savings

Removes model weights from GPU: ~1× model size. GPU only needs `prefetch_layers + 1` layers at a time.

### Measured performance

| Model | Mode | Step Time | Throughput |
|-------|------|-----------|------------|
| 7B | Full (full offload) | 23.8s | 2.5/min |
| 13B | Full (full offload, bf16 optim) | 61.7s | 1.0/min |

---

## Activation Spilling

Saves forward-pass activations to CPU during forward, restores them during backward. Replaces activation checkpointing (which recomputes instead of saving).

### Data flow

```
Forward (sub_layer within layer i):
  GPU                                 CPU
  ┌──────────────┐                   ┌──────────────┐
  │ activation_i │ ──async D2H──→    │ activation_i │ (pinned)
  │ [freed]      │                   │              │
  └──────────────┘                   └──────────────┘

Backward (same sub_layer):
  CPU                                 GPU
  ┌──────────────┐                   ┌──────────────┐
  │ activation_i │ ──async H2D──→    │ activation_i │
  │ [freed]      │                   │ [consumed]   │
  └──────────────┘                   └──────────────┘
```

### Implementation

File: `ironcore/offload/hooks.py`

`_SpillCheckpointFn` is a `torch.autograd.Function`:

- **Forward:** Saves activation metadata (shape, dtype). Submits async D2H via ActivationSpillManager. Computes sub-block under `torch.no_grad()`. Saves RNG state for dropout consistency.
- **Backward:** Ensures weights are on GPU. Restores activation from host (H2D prefetch). Recomputes sub-block with `torch.enable_grad()` using saved RNG state. Calls `torch.autograd.backward()` for gradients. Evicts weights after last sub-block.

Two granularity modes:
- `"sub_layer"`: Two spills per layer (hidden_states before attention, norm_input before MLP).
- `"full_layer"`: One spill per layer (attention + MLP together).

### Config

| Field | Default | Description |
|-------|---------|-------------|
| `activation_spill` | `false` | Enable activation spilling |
| `activation_spill_granularity` | `"sub_layer"` | Spill granularity |

### VRAM savings

Removes activation memory from GPU. Peak activation memory is the largest single sub-layer, not the full forward pass.

### Effect on context length

With activation spilling, steady-state VRAM is constant regardless of seq_len because activations are spilled to CPU. Measured on 13B:

| Seq Len | Steady-state VRAM | Status |
|---------|-------------------|--------|
| 512 | 1.59 GB | OK |
| 1024 | 1.59 GB | OK |
| 2048 | 1.59 GB | OK |

---

## Component Interactions

### optimizer offload (optimizer offload)

Simplest mode. No scheduler needed. Optimizer states allocated on CPU. Parameters stay on GPU. AdamW runs on CPU.

```
┌────────────────────┐
│ GPU                │
│ ┌────────────────┐ │
│ │ model weights  │ │
│ │ activations    │ │
│ └────────────────┘ │
└────────────────────┘

┌────────────────────┐
│ CPU (pinned)       │
│ ┌────────────────┐ │
│ │ exp_avg        │ │
│ │ exp_avg_sq     │ │
│ └────────────────┘ │
└────────────────────┘
```

Best for: models where weights fit in GPU but optimizer states don't.

### Optimizer + weight offload (optimizer + weight offload)

Weights stream to GPU layer-by-layer. Optimizer states on CPU. When weight streaming is active, params are on CPU during optimizer step, so the GPU→CPU grad transfer becomes a no-op.

```
┌────────────────────┐
│ GPU                │
│ ┌────────────────┐ │
│ │ staging window │ │ ← prefetch_layers + 1 layers
│ │ activations    │ │
│ └────────────────┘ │
└────────────────────┘

┌────────────────────┐
│ CPU (pinned)       │
│ ┌────────────────┐ │
│ │ ALL weights    │ │
│ │ exp_avg        │ │
│ │ exp_avg_sq     │ │
│ └────────────────┘ │
└────────────────────┘
```

Best for: models where weights alone exceed GPU VRAM.

### Full offload (optimizer + weight + activation)

All three mechanisms active. Weights stream in/out, activations spill to CPU, optimizer states on CPU. Maximum memory savings.

```
┌────────────────────┐
│ GPU                │
│ ┌────────────────┐ │
│ │ staging window │ │ ← current layer weights
│ └────────────────┘ │
└────────────────────┘

┌────────────────────────────┐
│ CPU (pinned)               │
│ ┌────────────────────────┐ │
│ │ ALL weights            │ │
│ │ optimizer states       │ │
│ │ spilled activations    │ │ ← freed as backward consumes
│ └────────────────────────┘ │
└────────────────────────────┘
```

Best for: maximum model size on minimum GPU VRAM.

---

## Configuration Reference

Full config example for 13B on 24 GB GPU:

```yaml
offload:
  enabled: true
  optimizer_offload: true
  optimizer_state_precision: bf16
  weight_offload: true
  weight_prefetch_layers: 2
  weight_storage_precision: bf16
  activation_spill: true
  activation_spill_granularity: sub_layer
  pinned_memory_pool_gb: 80.0
  gpu_staging_pool_mb: 0.0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Master switch |
| `optimizer_offload` | bool | `false` | offload optimizer states to CPU |
| `optimizer_state_precision` | str | `"fp32"` | `"fp32"` or `"bf16"` |
| `optimizer_min_param_elements` | int | `65536` | Min param size for offload |
| `weight_offload` | bool | `false` | per-layer weight streaming |
| `weight_prefetch_layers` | int | `2` | Forward lookahead depth |
| `backward_weight_prefetch_layers` | int | `1` | Backward prefetch depth |
| `weight_storage_precision` | str | `"bf16"` | Host weight dtype |
| `gpu_staging_pool_mb` | float | `0` | GPU staging budget (0 = auto) |
| `gpu_staging_chunk_mb` | float | `256` | GPU staging chunk size |
| `activation_spill` | bool | `false` | activation D2H/H2D |
| `activation_spill_granularity` | str | `"sub_layer"` | `"sub_layer"` or `"full_layer"` |
| `pinned_memory_pool_gb` | float | `-1` | Pinned host memory budget (-1 = auto) |
| `pinned_chunk_gb` | float | `4` | Pinned chunk size |
| `prefetch_streams` | int | `1` | CUDA streams for async transfers |

---

## Trainer Integration

The offload system integrates with `BaseTrainer` via lifecycle hooks:

```python
# base_trainer.py — initialization
if config.offload.enabled and (weight_offload or activation_spill):
    scheduler = ExecutionScheduler.from_model(model, config, device)
    model._offload_scheduler = scheduler

# base_trainer.py — training loop
scheduler.on_training_step_start()
for mb in range(grad_accum_steps):
    scheduler.on_microbatch_forward_start(mb)
    loss = forward_step(model, data_iter)
    scheduler.on_microbatch_forward_end(mb)
    scheduler.on_microbatch_backward_start(mb)
    scaled_loss.backward()
    scheduler.on_microbatch_backward_end(mb)
scheduler.on_backward_pass_end()
optimizer.step()
scheduler.on_training_step_end()
```

The model (`TransformerModel`) calls per-layer hooks during forward:

```python
# transformer.py — forward pass
for i, layer in enumerate(self.layers):
    if scheduler:
        scheduler.on_layer_start(i)
    hidden = layer(hidden, ...)
    if scheduler:
        scheduler.on_layer_end(i)
```

---

## Multi-GPU Compatibility

| Mode | DDP | FSDP shard_grad_op | FSDP full_shard | TP |
|------|-----|---------------------|-----------------|-----|
| optimizer offload | Supported | Supported (use_orig_params=True) | Blocked (duplicated states) | Supported |
| weight streaming | Supported | Supported | Supported | Supported |
| activation spilling | Supported | Supported | Supported | Supported |
| full offload | Supported | Supported | activation spilling | Supported |

Config validation guards prevent invalid combinations:
- optimizer offload + FSDP full_shard: raises ValueError (duplicated optimizer states on host)
- optimizer offload + FSDP CPUOffload: raises ValueError (redundant)
- optimizer offload + FSDP without use_orig_params: raises ValueError (FlatParameter breaks optimizer refs)

---

## Loss Parity

All offload modes produce numerically identical training results:

| Mode | Steps | Final Loss | Delta vs Baseline |
|------|-------|------------|-------------------|
| optimizer offload | 1000 | 5.832 | +0.005 |
| weight streaming | 1000 | 5.838 | +0.011 |
| activation spilling | 1000 | 5.835 | +0.008 |
| Full (full offload) | 1000 | 5.842 | +0.015 |

Source: `tests/unit/offload/test_pairwise_*.py`, `tests/unit/offload/test_m3_convergence.py`

---

## Known Bottleneck

CPU AdamW is the dominant throughput bottleneck for full offload on consumer hardware. GPU utilization is ~10% (idle ~90% of step time). Longer context does not help — GPU compute is negligible relative to CPU optimizer.

Root cause: DDR5 dual-channel (~96 GB/s) memory bandwidth limits the ~104 GB data movement per AdamW step. Server hardware with 8-channel DDR5 ECC (~200+ GB/s) is needed for production throughput.

For prototype-scale validation on consumer GPUs, the current throughput (1 step/min for 13B) is sufficient.

---

## File Index

| File | Responsibility |
|------|---------------|
| `ironcore/offload/scheduler.py` | ExecutionScheduler — training loop orchestration |
| `ironcore/offload/memory_pool.py` | PinnedMemoryPool — host page-locked allocator |
| `ironcore/offload/gpu_staging_pool.py` | GPUStagingPool — GPU pre-allocated allocator |
| `ironcore/offload/transfer_engine.py` | MemoryTransferEngine — async H2D/D2H |
| `ironcore/offload/tile_manager.py` | TileManager — weight tiling and reassembly |
| `ironcore/offload/hooks.py` | ActivationSpillManager — activation D2H/H2D |
| `ironcore/offload/optimizer_helpers.py` | optimizer — CPU-side AdamW with offloaded states |
| `ironcore/offload/config.py` | Deprecated re-export of OffloadConfig |
| `ironcore/config/config_offload.py` | OffloadConfig dataclass |
| `ironcore/utils/offload_visualizer.py` | Live training metrics display |
| `ironcore/utils/offload_metrics.py` | Metrics collection utilities |
