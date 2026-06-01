# Offload + Distributed Training Architecture Design

> **Scope**: Optimizer state offload, weight streaming, and activation spilling interactions with DDP, FSDP, and DistributedOptimizer
> **Audience**: Engineers evaluating configuration options for multi-GPU training

> **Quick Start**: See [§7.2 Configuration Selection Criteria](#72-configuration-selection-criteria) for common scenarios, or [§7.3 Complete Configuration Reference](#73-complete-configuration-reference) for all options.

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Components](#2-system-components)
3. [Capability Comparison Matrix](#3-capability-comparison-matrix)
4. [Configuration Catalog](#4-configuration-catalog)
5. [Memory Analysis](#5-memory-analysis)
6. [Prefetch and Overlap Analysis](#6-prefetch-and-overlap-analysis)
7. [User Decision Guide](#7-user-decision-guide)
8. [Implementation Recommendations](#8-implementation-recommendations)
9. [Additional Considerations](#9-additional-considerations)
10. [Verification Results and Gap Summary](#10-verification-results-and-gap-summary)

---

## 1. Executive Summary

Ironcore's offload system provides three independent modes for reducing GPU VRAM usage:

- **Optimizer state offload** (`optimizer_offload`): Keeps AdamW/Muon optimizer states on CPU with configurable precision
- **Weight streaming** (`weight_offload`): Streams layer weights from CPU to GPU with async prefetch and a GPU staging pool
- **Activation spilling** (`activation_spill`): Spills intermediate activations to CPU during forward, restores during backward

For multi-GPU training, PyTorch provides FSDP (FullyShardedDataParallel), which shards parameters, gradients, and optimizer states across ranks. FSDP and the offload system have overlapping capabilities (both can manage optimizer state placement) and complementary capabilities (FSDP does not handle activation spilling).

This document analyzes every combination, identifies where they conflict vs. complement, and provides implementation direction for making them work together without duplicating host memory.

### Key findings

1. **Activation spilling is universally compatible** — works with all parallelism strategies (DDP, FSDP, single-GPU) because it operates on intermediate tensors, not parameters. No other system (FSDP, DeepSpeed, Megatron) provides CPU-based activation spilling.

2. **Weight streaming is DDP/single-GPU only** — conflicts with FSDP's parameter sharding/unsharding. Already blocked by config validation.

3. **Optimizer state offload complements FSDP SHARD_GRAD_OP** — FSDP SHARD_GRAD_OP shards params+grads but not optimizer states. Optimizer offload fills this gap with CPU offload + configurable precision. But optimizer offload must NOT be used with FSDP FULL_SHARD (duplicates optimizer states in host memory).

4. **Optimizer offload + DistributedOptimizer stack multiplicatively** — DistributedOptimizer gives ZeRO-1 (shard optimizer states across DP ranks), optimizer offload moves the per-shard states to CPU with bf16 precision. Each rank's host memory = `1/N * 2 * params * 2B` (bf16).

5. **Backward prefetch is a gap** — FSDP has BACKWARD_PRE (overlaps all-gather with backward compute). Our offload system has forward weight prefetch but lacks backward activation prefetch. Adding it could improve throughput 5-10% for large models.

---

## 2. System Components

### 2.1 Offload Modes

#### Optimizer State Offload (`optimizer_offload`)

**What it does**: Stores AdamW exp_avg and exp_avg_sq (and optionally max_exp_avg_sq for AMSGrad) on CPU instead of GPU.

**Two compute paths**:

| Path | When | How | PCIe traffic |
|------|------|-----|-------------|
| CPU-compute | Params on GPU (Optimizer state offload-only, no Weight streaming) | AdamW math runs on CPU (SIMD/AVX-512 via MKL). Transfers grad (D2H) and delta (H2D). States never leave CPU. | `2N * dtype_size` per param per step |
| GPU-compute | Params on CPU (Weight streaming active) | States already on CPU. `.to()` is a no-op. Math runs on CPU natively. | 0 (everything on CPU) |

**Key features beyond FSDP's optimizer management**:

- Configurable state precision (`optimizer_state_precision`: fp32/bf16/fp16). BF16 halves host memory and PCIe traffic vs FSDP's fp32-only.
- Per-parameter offload threshold (`optimizer_min_param_elements`: default 65536). Skips tiny params where transfer overhead exceeds savings.
- LoRA param exclusion via `offloadable=False` attribute. Keeps LoRA adapter states on GPU for low-latency fine-tuning.
- CPU-compute AdamW path runs SIMD-optimized math, avoiding the need to stage full optimizer states on GPU.

**Configuration**:
```yaml
offload:
  enabled: true
  optimizer_offload: true
  optimizer_state_precision: "bf16"    # fp32, bf16, or fp16
  optimizer_min_param_elements: 65536   # skip params smaller than this
```

#### Weight Streaming (`weight_offload`)

**What it does**: Stores layer weights in pinned host memory. Streams weights to GPU via async H2D transfers with a pooled staging buffer. After forward/backward, evicts weights from GPU and snapshots updated values back to host.

**Incompatible with**: FSDP (manages its own parameter sharding), activation checkpointing (replays forward without scheduler hooks).

**Forward prefetch**: N layers ahead (configurable via `weight_prefetch_layers`, default 2). H2D transfers overlap with current layer's compute on a dedicated CUDA stream.

**Backward prefetch**: Single-layer lookahead when combined with Activation spilling. After layer N's backward completes, submits async H2D for layer N-1's weights while autograd traverses between layers.

**Configuration**:
```yaml
offload:
  enabled: true
  weight_offload: true
  weight_prefetch_layers: 2
  weight_storage_precision: "bf16"
  gpu_staging_pool_mb: 0.0       # 0 = auto-size
  gpu_staging_chunk_mb: 256.0
```

> **Auto-activation note**: Setting `weight_offload=true` automatically enables `activation_spill` via [config/__init__.py:211-219](../ironcore/config/__init__.py#L211-L219) (weight eviction requires `no_autograd_graph` boundaries which activation_spill's sub-layer hooks provide). If you see `activation_spill` enabled without explicitly configuring it, this auto-guard is the cause.

#### Activation Spilling (`activation_spill`)

**What it does**: Replaces activation checkpointing. During forward, spills intermediate activations (layer input, post-attention residual) to pinned host memory via async D2H. During backward, restores activations via H2D and recomputes the sub-block under `torch.enable_grad()` to rebuild the autograd graph.

**Granularity**: `sub_layer` (default) spills at attention/MLP boundaries (2 spills per layer). `full_layer` spills at layer boundaries (1 spill per layer, but retains more GPU intermediates).

**Free-after-consume**: Each activation's pinned host buffer is returned to the pool immediately after backward consumes it. Peak host memory is bounded to the set of activations awaiting backward.

**Forward D2H**: Async on dedicated CUDA stream. Transfer overlaps with next sub-layer's compute.

**Backward H2D**: Currently synchronous (blocking wait). This is a known gap — see [Section 6](#6-prefetch-and-overlap-analysis).

**Configuration**:
```yaml
offload:
  enabled: true
  activation_spill: true
  activation_spill_granularity: "sub_layer"  # "sub_layer" or "full_layer"
```

### 2.2 Parallelism Strategies

#### DDP (DistributedDataParallel)

Replicates all parameters across ranks. Synchronizes gradients via all-reduce after backward. Each rank holds a full copy of params, grads, and optimizer states.

**VRAM per rank**: `params + grads + optimizer_states + activations`

#### FSDP (FullyShardedDataParallel)

Shards parameters, gradients, and optimizer states across ranks. Unshards (all-gather) before forward/backward, reshard after. The sharding level depends on strategy:

| Strategy | Params | Grads | Optimizer States | Shorthand |
|----------|--------|-------|------------------|-----------|
| `FULL_SHARD` | Sharded (1/N) | Sharded (1/N) | Sharded (1/N) | ZeRO-3 |
| `SHARD_GRAD_OP` | Sharded (1/N) | Sharded (1/N) | **Replicated (full)** | ZeRO-2 |
| `NO_SHARD` | Replicated (full) | Replicated (full) | Replicated (full) | DDP equivalent |
| `HYBRID_SHARD` | FULL_SHARD intra-node, replicated inter-node | | | ZeRO-3 + DP |

**FSDP CPUOffload** (`cpu_offload=CPUOffload(offload_params=True)`): Offloads parameters to CPU when not computing. Optimizer step runs on CPU. No configurable precision. No activation offloading. **Limitation**: Does not support gradient accumulation outside `no_sync()`.

**FSDP BACKWARD_PRE**: Prefetches the next layer's all-gather while the current layer's backward computes. Overlaps NCCL communication with CUDA compute.

**FSDP use_orig_params**: Preserves original parameter objects (required for `torch.compile`). Optimizer references remain valid after FSDP wrapping. Required for Optimizer state offload+FSDP compatibility.

#### DistributedOptimizer (ZeRO-1)

Wraps an existing optimizer (AdamWOptimizer or MuonOptimizer) and partitions optimizer states across DP ranks via round-robin. Each rank only updates and stores optimizer states for its local partition (1/N of total). After `optimizer.step()`, updated parameters are broadcast from owner rank to all others.

**Orthogonal to**: Optimizer state offload (placement), Activation spilling (activations), Weight streaming (weight streaming). DistributedOptimizer decides *which* params each rank owns; Optimizer state offload decides *where* those states live (CPU vs GPU).

**Incompatible with**: FSDP (FSDP has its own optimizer state sharding).

---

## 3. Capability Comparison Matrix

### 3.1 Feature matrix: FSDP vs Offload

| Capability | FSDP FULL_SHARD | FSDP CPUOffload | FSDP SHARD_GRAD_OP | Optimizer state offload (Optimizer Offload) | Weight streaming (Weight Streaming) | Activation spilling (Activation Spill) |
|---|---|---|---|---|---|---|
| Parameter sharding | Yes (ZeRO-3) | No | Yes (ZeRO-2) | No | No | No |
| Gradient sharding | Yes (ZeRO-3) | No | Yes (ZeRO-2) | No | No | No |
| Optimizer state sharding | Yes (ZeRO-3) | No | **No** (replicated) | No | No | No |
| Optimizer state CPU offload | No (on GPU) | Yes (on CPU) | No (on GPU) | **Yes (on CPU)** | N/A | N/A |
| Configurable state precision | No (fp32 only) | No (fp32 only) | No (fp32 only) | **Yes (fp32/bf16/fp16)** | N/A | N/A |
| Per-param offload threshold | No | No | No | **Yes** (min_param_elements) | No | No |
| LoRA param exclusion | No | No | No | **Yes** (offloadable attr) | No | No |
| CPU-compute AdamW (SIMD) | No | No | No | **Yes** (AVX-512/MKL) | N/A | N/A |
| Weight streaming with GPU staging | No | No | No | No | **Yes** (tiled pool) | No |
| Activation D2H/H2D spill | No | No | No | No | No | **Yes** |
| Backward prefetch (params) | **Yes** (BACKWARD_PRE) | **Yes** (BACKWARD_PRE) | **Yes** (BACKWARD_PRE) | No | Partial (1-layer) | No |
| Forward prefetch (params) | **Yes** (forward_prefetch) | **Yes** (forward_prefetch) | **Yes** (forward_prefetch) | No | **Yes** (N-layer) | No |
| Multi-GPU required | Yes | Yes | Yes | No | No | No |
| Gradient accumulation | Yes | **No** (broken with CPUOffload) | Yes | Yes | Yes | Yes |

### 3.2 Overlap analysis

| Pair | Overlap? | Risk | Current status |
|------|----------|------|----------------|
| Weight streaming vs FSDP param sharding | **Full** | Both manage parameter placement | **Blocked** (config validation + runtime) |
| Optimizer state offload vs FSDP FULL_SHARD optim states | **Full** | Both manage optimizer state placement — Optimizer state offload creates second copy on CPU | **NOT blocked** (host OOM risk) |
| Optimizer state offload vs FSDP CPUOffload | **Partial** | Both run optimizer step on CPU | **NOT blocked** (redundant, wastes PCIe bandwidth) |
| Optimizer state offload vs FSDP SHARD_GRAD_OP optim states | **None** | SHARD_GRAD_OP doesn't touch optimizer states | **Compatible** |
| Activation spilling vs FSDP | **None** | Orthogonal concerns (activations vs params) | **Compatible** |
| Optimizer state offload vs DistributedOptimizer | **None** | Orthogonal (placement vs partitioning) | **Compatible** |
| Activation spilling vs DistributedOptimizer | **None** | Orthogonal | **Compatible** |

### 3.3 Support Matrix: Ironcore Offload vs FSDP vs DeepSpeed

| Feature | Ironcore Offload | FSDP | DeepSpeed (ZeRO) |
|---------|-----------------|------|------------------|
| **Parameter Sharding** | ❌ No | ✅ Yes (ZeRO-3) | ✅ Yes (ZeRO-3) |
| **Gradient Sharding** | ❌ No | ✅ Yes (ZeRO-2/3) | ✅ Yes (ZeRO-2/3) |
| **Optimizer State Sharding** | ❌ No (use DistributedOptimizer) | ✅ Yes (ZeRO-3) | ✅ Yes (ZeRO-3) |
| **Optimizer State CPU Offload** | ✅ Yes (Optimizer state offload, configurable precision) | ⚠️ Yes (CPUOffload, fp32 only) | ✅ Yes (ZeRO-Infinity, fp32 only) |
| **Optimizer State Precision** | ✅ fp32/bf16/fp16 configurable | ❌ fp32 only | ⚠️ fp32 only (some variants fp16) |
| **Weight Streaming** | ✅ Yes (Weight streaming, async H2D with staging) | ❌ No | ⚠️ Yes (ZeRO-Infinity only) |
| **Activation CPU Spill** | ✅ Yes (Activation spilling, unique feature) | ❌ No | ❌ No |
| **Checkpoint Offload** | ❌ No | ❌ No | ✅ Yes (ZeRO-Infinity) |
| **Per-Parameter Offload Control** | ✅ Yes (offloadable attr, threshold) | ❌ No | ⚠️ Partial (parameter groups) |
| **LoRA Adapter Support** | ✅ Yes (offloadable=False) | ⚠️ Partial (requires care) | ⚠️ Partial |
| **Mixed Precision (bf16) Training** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Gradient Accumulation** | ✅ Yes | ✅ Yes | ✅ Yes (except CPUOffload) |
| **Tensor Parallelism** | ✅ Yes (compatible) | ✅ Yes | ✅ Yes |
| **Pipeline Parallelism** | ❌ No (out of scope) | ❌ No | ✅ Yes |
| **MoE / Expert Parallelism** | ✅ Yes (compatible) | ✅ Yes | ✅ Yes |
| **Single-Node Optimization** | ✅ Primary target | ✅ Supported | ✅ Supported |
| **Multi-Node Scaling** | ⚠️ Not optimized | ✅ Yes | ✅ Yes |
| **NVLink Optimization** | ✅ Yes (PCIe overlap) | ✅ Yes | ✅ Yes |
| **Telemetry / Monitoring** | ✅ Built-in (H2D/D2H tracking) | ❌ Limited | ⚠️ Through profiling tools |
| **Live Metrics Visualizer** | ✅ Yes (terminal-based) | ❌ No | ❌ No |

**Key Differentiators:**

1. **Activation CPU Spill (Activation spilling)**: Unique to Ironcore. Neither FSDP nor DeepSpeed provides CPU-based activation spilling. FSDP has activation checkpointing (recomputation) but not CPU spill.

2. **Configurable Optimizer Precision**: Ironcore supports fp32/bf16/fp16 for optimizer states. FSDP and DeepSpeed are primarily fp32-only for states.

3. **Weight Streaming with GPU Staging**: Ironcore's weight streaming provides true async weight streaming with a GPU staging pool. DeepSpeed ZeRO-Infinity has similar functionality but is more complex to configure.

4. **Per-Parameter Control**: Ironcore allows fine-grained control via `offloadable` attribute and `optimizer_min_param_elements` threshold.

5. **Built-in Telemetry**: Ironcore includes H2D/D2H transfer tracking, bandwidth measurement, and live visualization.

---

## 4. Configuration Catalog

Each configuration below shows a memory layout diagram and explains what lives where.

### 4.1 Single-GPU (no parallelism)

```
GPU Memory                              Host Memory
+----------------------------------+    +----------------------------------+
| Model parameters (full)          |    | Optimizer states (Optimizer state offload, bf16)      |
| Gradients (full)                 |    | Spilled activations (Activation spilling)         |
| Activations (if no Activation spilling)           |    | Weight tiles (Weight streaming, bf16)          |
| GPU staging pool (Weight streaming only)       |    +----------------------------------+
+----------------------------------+
```

**Available modes**: Optimizer state offload + Weight streaming + Activation spilling (all)

**VRAM breakdown** (13B model, bf16 params, bf16 optimizer states):

| Component | Without offload | With Optimizer state offload+Weight streaming+Activation spilling |
|-----------|----------------|----------------|
| Parameters | 26 GB | ~0.5 GB (staging pool for 3 layers) |
| Optimizer states | 52 GB (fp32) | 0 GB (on CPU, bf16 = 26 GB host) |
| Activations | 8-20 GB | ~0 GB (spilled to host) |
| Gradients | 26 GB | 26 GB (must stay on GPU) |
| **Total GPU** | **112-132 GB** | **~27 GB** |
| **Total Host** | ~0 GB | ~30-50 GB |

**When to use**: Single-GPU training when the model doesn't fit in VRAM. This is the highest-savings configuration.

### 4.2 DDP

```
Each rank's GPU Memory                   Each rank's Host Memory
+----------------------------------+    +----------------------------------+
| Model parameters (full replica)  |    | Optimizer states (Optimizer state offload, bf16)      |
| Gradients (full, pre-sync)      |    | Spilled activations (Activation spilling)         |
| Activations (if no Activation spilling)          |    | Weight tiles (Weight streaming, bf16)          |
| GPU staging pool (Weight streaming only)      |    +----------------------------------+
+----------------------------------+
         |                                      |
         +------ all-reduce gradients ----------+
```

**Available modes**: Optimizer state offload + Weight streaming + Activation spilling (all)

**Key difference from single-GPU**: Each rank holds a full model replica. Gradient all-reduce after backward. No parameter sharding.

**VRAM per rank**: Same as single-GPU. No savings from parallelism itself.

**When to use**: Multiple GPUs but FSDP is not desired (e.g., TP-only, or model fits per-GPU with offload).

### 4.3 DDP + DistributedOptimizer (ZeRO-1)

```
Each rank's GPU Memory                   Each rank's Host Memory
+----------------------------------+    +----------------------------------+
| Model parameters (full replica)  |    | Optimizer states (Optimizer state offload, bf16)      |
| Gradients (full, pre-sync)      |    | for local partition only (1/N)   |
| Activations (if no Activation spilling)          |    | Spilled activations (Activation spilling)         |
| GPU staging pool (Weight streaming only)      |    | Weight tiles (Weight streaming, bf16)          |
+----------------------------------+    +----------------------------------+
         |                                      |
         +------ all-reduce gradients ----------+
         +------ broadcast updated params ------+
                 (from owner rank)
```

**Available modes**: Optimizer state offload + Weight streaming + Activation spilling (all)

**How DistributedOptimizer + Optimizer state offload compose**:

```
DistributedOptimizer.step()
  1. Null out non-local param grads (params not owned by this rank)
  2. Call inner AdamWOptimizer.step()
     -> For each local param:
        -> offload_enabled? -> _adamw_offloaded_step()
           -> CPU-compute path (params on GPU): grad D2H, AdamW on CPU, delta H2D
           -> States stay on CPU in configurable dtype
  3. Broadcast updated param.data from owner rank to others
```

DistributedOptimizer decides *which* params (1/N round-robin). Optimizer state offload decides *where* states live (CPU with bf16). No overlap in responsibility.

**Host memory per rank** (13B model, 4 GPUs, bf16 optimizer states):

| Component | DDP + Optimizer state offload only | DDP + DistributedOptimizer + Optimizer state offload |
|-----------|---------------|---------------------------------|
| Optimizer states on CPU | 26 GB (full, bf16) | **6.5 GB** (1/4, bf16) |
| Weight tiles (Weight streaming) | 26 GB (full, bf16) | 26 GB (full, bf16) |
| Spilled activations (Activation spilling) | 4-10 GB | 4-10 GB |
| **Total host** | **56-62 GB** | **37-43 GB** |

**When to use**: Multi-GPU training where optimizer states dominate VRAM. DistributedOptimizer shards states across ranks, Optimizer state offload moves the per-shard states to CPU. Best DDP configuration for optimizer-heavy workloads.

### 4.4 FSDP FULL_SHARD (ZeRO-3)

```
Each rank's GPU Memory                   Each rank's Host Memory
+----------------------------------+    +----------------------------------+
| Parameter shard (1/N)            |    | (nothing from FSDP)              |
| Gradient shard (1/N)            |    |                                  |
| Optimizer state shard (1/N)     |    | (FSDP optim states on GPU)       |
| Unsharded params (transient)    |    |                                  |
| Activations (if no Activation spilling)          |    +----------------------------------+
+----------------------------------+
         |                                      |
         +------ all-gather params (fwd/bwd) --+
         +------ reduce-scatter grads ----------+
```

**Available offload modes**: Activation spilling only

**Why Optimizer state offload is forbidden**: FSDP FULL_SHARD already shards optimizer states across ranks. Optimizer state offload would create a second copy of optimizer states on CPU — duplicating host memory. For a 13B model with 4 GPUs: FSDP holds 13 GB optimizer state shard per rank on GPU; Optimizer state offload would add 52 GB (fp32) or 26 GB (bf16) per rank on CPU. The host OOM risk is real.

**Why Weight streaming is forbidden**: FSDP manages parameter sharding/unsharding. Weight streaming would conflict with FSDP's all-gather mechanism.

**Why Activation spilling is compatible**: Activation spilling operates on intermediate activations (tensors between layers), not parameters. FSDP manages when parameters are on GPU; Activation spilling manages when activations are on CPU. Completely orthogonal.

**VRAM per rank** (13B model, 4 GPUs):

| Component | FSDP alone | FSDP + Activation spilling |
|-----------|------------|-----------|
| Parameter shard | 6.5 GB | 6.5 GB |
| Optimizer state shard | 13 GB (fp32) | 13 GB (fp32) |
| Gradient shard | 6.5 GB | 6.5 GB |
| Unsharded params (transient) | 26 GB (2 consecutive layers) | 26 GB |
| Activations | 8-20 GB | ~0 GB (spilled to host) |
| **Total GPU** | **60-72 GB** | **~52 GB** |

**Host memory per rank**: Spilled activations only = 4-10 GB (bounded by free-after-consume).

**When to use**: Multi-GPU training where FSDP provides sufficient parameter/optimizer sharding. Activation spilling adds activation spilling for further VRAM reduction. This is the recommended multi-GPU configuration for most cases.

### 4.5 FSDP SHARD_GRAD_OP (ZeRO-2)

```
Each rank's GPU Memory                   Each rank's Host Memory
+----------------------------------+    +----------------------------------+
| Parameter shard (1/N)            |    | Optimizer states (Optimizer state offload, bf16)      |
| Gradient shard (1/N)            |    | for FULL model (not sharded)     |
| Optimizer states (if no Optimizer state offload)     |    | Spilled activations (Activation spilling)         |
| Unsharded params (transient)    |    +----------------------------------+
| Activations (if no Activation spilling)          |
+----------------------------------+
         |                                      |
         +------ all-gather params (fwd/bwd) --+
         +------ reduce-scatter grads ----------+
```

**Available offload modes**: Optimizer state offload + Activation spilling

**Why Optimizer state offload adds value here**: SHARD_GRAD_OP shards params and grads but **does not shard optimizer states**. Each rank holds full optimizer states (52 GB fp32 for 13B). Optimizer state offload offloads those states to CPU with configurable precision (26 GB bf16). This is the gap Optimizer state offload fills.

**Why `use_orig_params=True` is required**: Optimizer state offload's AdamWOptimizer holds references to original parameter objects. FSDP with `use_orig_params=True` preserves these references as views into sharded FlatParameters. Without it, FSDP replaces parameters with FlatParameters, breaking optimizer references.

**VRAM per rank** (13B model, 4 GPUs):

| Component | FSDP SHARD_GRAD_OP | FSDP SHARD_GRAD_OP + Optimizer state offload + Activation spilling |
|-----------|--------------------|-------------------------------|
| Parameter shard | 6.5 GB | 6.5 GB |
| Optimizer states | 52 GB (fp32, full!) | 0 GB (on CPU, bf16) |
| Gradient shard | 6.5 GB | 6.5 GB |
| Unsharded params (transient) | 26 GB | 26 GB |
| Activations | 8-20 GB | ~0 GB (spilled to host) |
| **Total GPU** | **99-111 GB** | **~39 GB** |

**Host memory per rank**: Optimizer states (26 GB bf16) + spilled activations (4-10 GB) = 30-36 GB.

**When to use**: Multi-GPU training where optimizer states are the dominant VRAM consumer. FSDP handles param/grad sharding, Optimizer state offload offloads the unsharded optimizer states. Particularly useful for models where the optimizer state ratio is high (e.g., AdamW with fp32 states = 2x model size).

### 4.6 FSDP FULL_SHARD + CPUOffload

```
Each rank's GPU Memory                   Each rank's Host Memory
+----------------------------------+    +----------------------------------+
| (transient during compute)       |    | Parameters (offloaded)           |
| Unsharded params (transient)    |    | Gradients (offloaded)            |
| Activations (if no Activation spilling)          |    | Optimizer states (fp32, full)    |
+----------------------------------+    +----------------------------------+
         |                                      |
         +------ all-gather params (fwd/bwd) --+
         +------ reduce-scatter grads ----------+
```

**Available offload modes**: Activation spilling only

**Why Optimizer state offload is redundant here**: FSDP CPUOffload already runs the optimizer step on CPU with fp32 states. Optimizer state offload would duplicate optimizer states in host memory (same data, different management). No benefit.

**Why Activation spilling still adds value**: FSDP CPUOffload does not handle activation spilling. Activation spilling reduces activation VRAM to near-zero.

**Limitation**: FSDP CPUOffload **does not support gradient accumulation** outside `no_sync()`. If you need gradient accumulation with micro-batching, use FSDP FULL_SHARD + Activation spilling instead (without CPUOffload).

**When to use**: Extreme memory-constrained scenarios where even parameter shards must be offloaded. Activation spilling adds activation spilling. Avoid if using gradient accumulation.

### 4.7 Checkpoint Compatibility (All Configurations)

In all configurations in §4.1-§4.6, **resume-from-checkpoint is compatible with offload states**. [checkpointing/native.py:380-461](../ironcore/checkpointing/native.py#L380-L461) automatically handles:

- **On save**: Checks `optimizer.offload_enabled`, param's `offloadable` attribute, and `optimizer_min_param_elements` threshold to serialize offloaded states while preserving CPU location
- **On restore**: Re-distributes states to CPU or GPU based on the same criteria. Also handles TP-shard splitting (line 414-446)
- HuggingFace interop checkpoints apply the same logic

**Verified tests**: [tests/unit/offload/test_checkpoint_offload.py](../tests/unit/offload/test_checkpoint_offload.py), [tests/integration/offload/test_checkpoint_offload.py](../tests/integration/offload/test_checkpoint_offload.py).

See §9.5 for detailed behavior.

---

## 5. Memory Analysis

### 5.1 Per-component memory formulas

For a model with P parameters, N GPUs, and D dtype bytes (2 for bf16, 4 for fp32):

| Component | Formula | Notes |
|-----------|---------|-------|
| Parameters (full) | `P * D` | bf16 model: 2P |
| Parameters (sharded) | `P * D / N` | Per-rank with FSDP |
| Gradients (full) | `P * D` | Same size as params |
| Gradients (sharded) | `P * D / N` | Per-rank with FSDP |
| Optimizer states (fp32, AdamW) | `2 * P * 4` | exp_avg + exp_avg_sq |
| Optimizer states (bf16, AdamW) | `2 * P * 2` | Optimizer state offload with bf16 precision |
| Optimizer states (AMSGrad fp32) | `3 * P * 4` | + max_exp_avg_sq tracked persistently |
| Optimizer states (AMSGrad bf16) | `3 * P * 2` | Optimizer state offload + AMSGrad with bf16 |
| Optimizer states (sharded fp32) | `2 * P * 4 / N` | Per-rank with FSDP FULL_SHARD |
| Optimizer states (ZeRO-1 + Optimizer state offload bf16) | `2 * P * 2 / N` | Per-rank with DistributedOptimizer + Optimizer state offload |
| GPU staging pool (Weight streaming) | `(prefetch_layers + 1) * layer_bytes` | Sliding window of consecutive layers |
| Activations (varies) | `batch * seq_len * hidden * layers * k` | k = 1-5 depending on checkpointing |
| Spilled activations on host | Similar to GPU activations | Bounded by free-after-consume |

### 5.2 Per-configuration comparison (13B model, 4 GPUs, bf16 params)

| Configuration | GPU per rank | Host per rank | Notes |
|---|---|---|---|
| No offload, DDP | 112-132 GB | ~0 GB | Requires A100-80GB or H100 |
| Optimizer state offload+Weight streaming+Activation spilling, DDP | ~27 GB | 30-50 GB | Fits in consumer GPU (RTX 4090) |
| Optimizer state offload+Weight streaming+Activation spilling+DistOpt, DDP | ~27 GB | 37-43 GB | ZeRO-1 reduces host optim states |
| FSDP FULL_SHARD | 60-72 GB | ~0 GB | Requires A100-80GB |
| FSDP FULL_SHARD + Activation spilling | ~52 GB | 4-10 GB | Fits in A100-80GB comfortably |
| FSDP SHARD_GRAD_OP | 99-111 GB | ~0 GB | Doesn't fit in any single GPU |
| FSDP SHARD_GRAD_OP + Optimizer state offload + Activation spilling | ~39 GB | 30-36 GB | Fits in consumer GPU |
| FSDP FULL_SHARD + CPUOffload + Activation spilling | ~30 GB | 36-46 GB | Maximum host offload, but no grad accum |

### 5.3 Host memory avoidance rules

**The critical constraint**: Avoid duplicating optimizer states in host memory.

| Configuration | Host optimizer memory per rank | Duplication? |
|---|---|---|
| FSDP FULL_SHARD (GPU optimizer) | 0 GB | No duplication |
| FSDP FULL_SHARD + CPUOffload | `2 * P * 4 / N` | FSDP-managed, no duplication |
| FSDP FULL_SHARD + Optimizer state offload (WRONG) | `2 * P * 4 / N + 2 * P * 2` | **Duplicated!** FSDP shard + Optimizer state offload full copy |
| FSDP SHARD_GRAD_OP + Optimizer state offload (bf16) | `2 * P * 2` | No duplication (SHARD_GRAD_OP doesn't shard optimizer) |
| DDP + Optimizer state offload (bf16) | `2 * P * 2` | No duplication |
| DDP + DistOpt + Optimizer state offload (bf16) | `2 * P * 2 / N` | No duplication (DistOpt shards, Optimizer state offload offloads per-shard) |

---

## 6. Prefetch and Overlap Analysis

### 6.1 FSDP's BACKWARD_PRE

FSDP uses `backward_prefetch=BackwardPrefetch.BACKWARD_PRE` (hardcoded in `parallel.py:153`). This prefetches the next FSDP unit's all-gather **before** the current unit's gradient computation begins.

```
Timeline (FSDP backward):
Layer 3 backward: |--all-gather L3--|--compute grad L3--|
Layer 2 backward:                  |--all-gather L2 (prefetched)--|--compute grad L2--|
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                      Overlapped with L3 grad compute
```

**What it overlaps**: NCCL all-gather (inter-GPU communication) with CUDA gradient computation.

**Limitation**: Only overlaps communication/compute for multi-GPU. Does not help with CPU/GPU transfers for offloaded data.

### 6.2 Our current prefetch capabilities

| Component | Forward prefetch | Backward prefetch | Mechanism |
|-----------|-----------------|-------------------|-----------|
| Weight streaming: Weight streaming | **Yes** (N layers ahead, async H2D on dedicated CUDA stream) | **Partial** (1 layer lookahead with Activation spilling active) | `weight_prefetch_layers` config, `MemoryTransferEngine` async streams |
| Activation spilling: Activation spill forward D2H | **Yes** (async D2H on dedicated stream, fire-and-forget) | N/A | `submit_d2h()` returns handle, waited on only during backward |
| Activation spilling: Activation spill backward H2D | N/A | **No** (synchronous blocking wait) | `on_sublayer_backward()` submits H2D then immediately waits |
| Optimizer state offload: Optimizer offload | N/A | **No** (all transfers synchronous, `non_blocking=False`) | `_adamw_offloaded_step_cpu_compute()` uses blocking copies |

### 6.3 Gap analysis

**Gap 1: Activation backward H2D is fully blocking**

Current code in `hooks.py:on_sublayer_backward()`:
```
1. Wait for forward D2H to complete (should be done by now)
2. Submit H2D transfer for activation
3. Immediately wait for H2D to complete      <-- BLOCKING
4. Synchronize with default stream            <-- BLOCKING
5. Recompute forward, run backward
```

The H2D transfer runs on a dedicated CUDA stream but is immediately waited on. No overlap with computation.

**Impact**: For a 24-layer model with sub_layer granularity:
- 48 activation H2D transfers per backward pass
- Each transfer: `batch * seq_len * hidden * dtype_size / PCIe_bandwidth`
- Example: 2 * 2048 * 1024 * 2 / 32 GB/s = ~0.13 ms per transfer
- Total blocking time: ~6.2 ms per step
- If step takes 200 ms, that's ~3% overhead

For a 48-layer 13B model:
- 96 activation H2D transfers
- Each: 2 * 2048 * 5120 * 2 / 32 GB/s = ~1.3 ms
- Total: ~125 ms
- If step takes 2000 ms, that's ~6% overhead

**Gap 2: Backward weight prefetch is limited to 1 layer**

`on_backward_layer_end()` only prefetches `layer_idx - 1` (hardcoded). The `weight_prefetch_layers` config is ignored during backward. Forward prefetches N layers; backward prefetches only 1.

**Gap 3: Single transfer stream**

`MemoryTransferEngine` defaults to `prefetch_streams=1` and this is not configurable via `OffloadConfig`. With one stream, weight and activation transfers serialize. Two streams could overlap weight H2D with activation H2D.

**Gap 4: `drain_completed()` is dead code**

The `MemoryTransferEngine` has a `drain_completed()` method (line 196) that polls for completed transfers without blocking. It is defined but never called. This could enable opportunistic checking: "is the prefetch done yet?" without a hard wait.

### 6.4 Proposed backward prefetch for activations

**Idea**: When backward starts for layer N, immediately submit async H2D for layer N-1's activations. The transfer runs on the dedicated stream while layer N's backward computes. When backward reaches layer N-1, the activation is already on GPU (or nearly so).

```
Current (blocking):
Layer 2 backward: |--H2D act L2 (block)--|--recompute L2--|--backward L2--|
Layer 1 backward:                                          |--H2D act L1 (block)--|--recompute L1--|--backward L1--|

Proposed (prefetched):
Layer 2 backward: |--H2D act L2 (block)--|--recompute L2--|--backward L2--|
Layer 1 backward: [H2D act L1 (async, started during L2 backward)]
                                                        (wait, likely done)--|--recompute L1--|--backward L1--|
```

**Implementation sketch**:

In `scheduler.py:on_backward_layer_end()`, after prefetching weights for the previous layer, also prefetch activations:

```python
def on_backward_layer_end(self, layer_idx: int):
    # ... existing weight eviction + weight prefetch for layer_idx - 1 ...

    # NEW: Prefetch activation for layer_idx - 1
    if self._spill_manager and layer_idx > 0:
        for sub_layer in reversed(range(self._spill_manager.num_sub_layers)):
            key = (self._current_microbatch, layer_idx - 1, sub_layer)
            self._spill_manager.prefetch_activation(key)  # async H2D
```

In `hooks.py:on_sublayer_backward()`, check if activation is already prefetched:

```python
def on_sublayer_backward(self, microbatch_idx, layer_idx, sub_layer, gpu_dst):
    key = (microbatch_idx, layer_idx, sub_layer)
    activation = self._spilled[key]

    if activation.is_prefetched:
        # Already on GPU from earlier prefetch — just wait for completion
        self._engine.wait(activation.prefetch_handle)
        self._engine.synchronize_with_default_stream()
    else:
        # Fallback: synchronous path (current behavior)
        self._engine.wait(activation.transfer_handle)  # wait for D2H
        handle = self._engine.submit_h2d(...)
        self._engine.wait(handle)
        self._engine.synchronize_with_default_stream()
```

**Estimated improvement**: 3-6% per step for large models. The H2D latency is hidden behind the previous layer's backward compute.

**Risk**: Correctness depends on the async H2D completing before the activation is consumed. If compute is faster than transfer (small models, fast GPUs), we still wait. No correctness risk — just no benefit.

### 6.5 Proposed multi-layer backward weight prefetch

**Idea**: Extend `on_backward_layer_end()` to prefetch `weight_prefetch_layers` layers backward (not just 1).

```python
def on_backward_layer_end(self, layer_idx: int):
    # ... existing eviction ...

    # Extended: prefetch N layers backward
    for ahead in range(1, self._prefetch_layers + 1):
        prev_idx = layer_idx - ahead
        if prev_idx >= 0 and prev_idx not in self._layer_on_gpu:
            self._prefetch_layer(prev_idx)
```

**Estimated improvement**: Marginal for most cases. Backward traversal between consecutive layers is already short (autograd overhead). The current 1-layer lookahead covers most of the benefit. Extending to 2-3 layers helps only if backward compute per layer is very fast (< 1ms).

**Recommendation**: Low priority. Implement activation backward prefetch first (higher impact).

### 6.6 Should we implement FSDP-style BACKWARD_PRE for DDP?

**No.** FSDP's BACKWARD_PRE overlaps NCCL all-gather (inter-GPU communication) with compute. With DDP, there is no all-gather — gradients are synchronized via all-reduce after the entire backward pass completes. There is nothing to prefetch for DDP.

For DDP + Weight streaming + Activation spilling, the equivalent optimization is our weight and activation backward prefetch (Section 6.4, 6.5). This overlaps PCIe DMA (H2D) with CUDA compute, which is the single-node analog of FSDP's NCCL/compute overlap.

---

## 7. User Decision Guide

### 7.1 Decision tree

```
How many GPUs?
|
+-- 1 GPU
|   |
|   +-- Model fits in VRAM?
|       +-- Yes: No offload needed
|       +-- No: Enable offload.enabled=true
|           |
|           +-- What doesn't fit?
|               +-- Optimizer states too large: Optimizer state offload (optimizer_offload=true)
|               +-- Activations too large: Activation spilling (activation_spill=true)
|               +-- Parameters too large: Weight streaming (weight_offload=true)
|               +-- Multiple: Optimizer state offload+Weight streaming+Activation spilling (all)
|
+-- Multiple GPUs
    |
    +-- Using FSDP?
        |
        +-- Yes
        |   |
        |   +-- FSDP FULL_SHARD + model fits per-GPU?
        |   |   +-- Yes: FSDP alone. Add Activation spilling if activations don't fit.
        |   |   +-- No (optimizer OOM): Consider SHARD_GRAD_OP + Optimizer state offload + Activation spilling
        |   |
        |   +-- FSDP SHARD_GRAD_OP?
        |   |   +-- Optimizer states don't fit: Add Optimizer state offload + Activation spilling
        |   |   +-- Only activations don't fit: Add Activation spilling
        |   |
        |   +-- Need CPUOffload (extreme constraint)?
        |       +-- Yes: FSDP FULL_SHARD + CPUOffload + Activation spilling
        |       +-- Note: CPUOffload breaks gradient accumulation
        |
        +-- No (using DDP)
            |
            +-- Optimizer states too large per-rank?
                +-- Yes: DDP + DistributedOptimizer + Optimizer state offload + Activation spilling
                +-- No: DDP + Optimizer state offload + Activation spilling (or Optimizer state offload+Weight streaming+Activation spilling for max savings)
```

### 7.2 Configuration selection criteria

| Scenario | Recommended config | YAML |
|----------|-------------------|------|
| Single GPU, 7B model, 24GB VRAM | Optimizer state offload+Weight streaming+Activation spilling | `offload.enabled=true, optimizer_offload=true, weight_offload=true, activation_spill=true` |
| Single GPU, 7B model, 48GB VRAM | Optimizer state offload+Activation spilling | `offload.enabled=true, optimizer_offload=true, activation_spill=true` |
| Single GPU, model fits but want headroom | Optimizer state offload | `offload.enabled=true, optimizer_offload=true` |
| 4x GPU, FSDP, 13B model, 80GB each | FSDP FULL_SHARD + Activation spilling | `parallel.use_fsdp=true, offload.enabled=true, activation_spill=true` |
| 4x GPU, FSDP, 13B model, 48GB each | FSDP SHARD_GRAD_OP + Optimizer state offload + Activation spilling | `parallel.use_fsdp=true, parallel.fsdp_sharding_strategy="shard_grad_op", parallel.fsdp_use_orig_params=true, offload.enabled=true, optimizer_offload=true, activation_spill=true` |
| 4x GPU, DDP, 13B model, 48GB each | DDP + DistOpt + Optimizer state offload + Activation spilling | `parallel.use_distributed_optimizer=true, offload.enabled=true, optimizer_offload=true, activation_spill=true` |
| 8x GPU, FSDP, 70B model | FSDP FULL_SHARD + Activation spilling | `parallel.use_fsdp=true, offload.enabled=true, activation_spill=true` |

#### 2x RTX 3090 Workstation (NVLink bridge, 128GB+ RAM) — Qwen Family Reference

Based on this codebase's primary development environment (2x RTX 3090, NVLink, 128GB+ host RAM), here are the recommended configurations for Qwen2.5 / Qwen3 family:

| Model | Recommended config | Key rationale |
|---|---|---|
| Qwen2.5-1.5B / Qwen3-1.7B | DDP, offload not needed | Fits in single 3090. Throughput-focused. Optimizer state offload optional |
| Qwen2.5-3B / Qwen3-4B | DDP + Optimizer state offload (Activation spilling optional) | 24GB sufficient. host bf16 ~6GB only |
| Qwen2.5-7B / Qwen3-8B | DDP + DistOpt + Optimizer state offload + Activation spilling | GPU ~12-15GB. ZeRO-1 reduces host optim (~7GB/rank) |
| Qwen2.5-7B + TP=2 | TP=2 + Optimizer state offload + Activation spilling | TP shard halves per-GPU params. NVLink accelerates TP all-reduce |
| Qwen2.5-14B / Qwen3-14B | TP=2 + Optimizer state offload + Activation spilling, **or** FSDP SHARD_GRAD_OP + Optimizer state offload + Activation spilling | param/grad split → GPU ~14-16GB. TP-aware offload pending verification (Phase D) |
| Qwen2.5-32B / Qwen3-32B | FSDP FULL_SHARD + CPUOffload + Activation spilling (no grad_accum) | params also CPU. host ~80GB. NVLink accelerates all-gather. **grad accum not supported** |
| **Qwen3-30B-A3B (MoE)** | FSDP FULL_SHARD + CPUOffload + Activation spilling + EP=2 (experimental) | Active 3B so compute light. But full 30B params on host (~60GB bf16 + 30GB bf16 optim ≈ 90GB host). **MoE × offload verification case** |
| Qwen2.5-32B + LoRA fine-tuning | DDP + LoRA (`offloadable=False` adapters) + Activation spilling | base weight freeze. Optimizer state offload disabled (adapters stay on GPU) |
| Qwen3-235B-A22B (MoE) | **Out of scope** (exceeds single-node limit — host 200GB+ required) | Reference only |

YAML example (Qwen2.5-7B + DDP + DistOpt + Optimizer state offload + Activation spilling):
```yaml
parallel:
  use_distributed_optimizer: true
offload:
  enabled: true
  optimizer_offload: true
  optimizer_state_precision: "bf16"
  activation_spill: true
  activation_spill_granularity: "sub_layer"
  pinned_memory_pool_gb: -1.0   # auto: psutil-based auto-detect (post-Phase B-1)
```

YAML example (Qwen2.5-14B + FSDP SHARD_GRAD_OP + Optimizer state offload + Activation spilling):
```yaml
parallel:
  use_fsdp: true
  fsdp_sharding_strategy: "shard_grad_op"
  fsdp_use_orig_params: true   # Required for Optimizer state offload+FSDP
offload:
  enabled: true
  optimizer_offload: true
  optimizer_state_precision: "bf16"
  activation_spill: true
```

---

## 7.3 Complete Configuration Reference

### Quick Start: Choosing Your Configuration

This section helps you quickly determine which offload options to enable based on your hardware and model.

#### Step 1: Identify your constraint

Answer these questions:
1. **What's your per-GPU VRAM?** (e.g., 24GB RTX 3090, 48GB RTX 6000, 80GB A100)
2. **What's your host RAM?** (e.g., 64GB, 128GB, 256GB)
3. **Which model are you training?** (e.g., 7B, 13B, 70B parameters)
4. **How many GPUs?** (1, 2, 4, 8)

#### Step 2: Find your scenario

| VRAM | Host RAM | GPUs | Model Size | Recommended Config |
|------|----------|------|------------|-------------------|
| 24GB | 64GB+ | 1 | 1-3B | No offload needed |
| 24GB | 128GB+ | 1 | 7B | `enabled=true, optimizer_offload=true, activation_spill=true` |
| 24GB | 128GB+ | 2 | 7B | `enabled=true, optimizer_offload=true, activation_spill=true, use_distributed_optimizer=true` |
| 48GB | 128GB+ | 1-2 | 7-14B | `enabled=true, optimizer_offload=true, activation_spill=true` |
| 48GB | 256GB+ | 4 | 13-32B | FSDP `shard_grad_op` + `optimizer_offload=true, activation_spill=true` |
| 80GB | 256GB+ | 4-8 | 30B+ | FSDP `full` + `activation_spill=true` |

#### Step 3: Configure and validate

**Example YAML** (single GPU, 7B model, 24GB VRAM):
```yaml
offload:
  enabled: true
  optimizer_offload: true           # Optimizer state offload: Save ~50% VRAM (optimizer states → CPU)
  optimizer_state_precision: "bf16"  # Halves host memory vs fp32
  activation_spill: true             # Activation spilling: Save ~30% VRAM (activations → CPU)
  activation_spill_granularity: "sub_layer"
  pinned_memory_pool_gb: -1.0        # Auto-detect from available RAM
```

**Example YAML** (4x GPU, 13B model, FSDP):
```yaml
parallel:
  use_fsdp: true
  fsdp_sharding_strategy: "shard_grad_op"  # ZeRO-2
  fsdp_use_orig_params: true               # REQUIRED for Optimizer state offload+FSDP
offload:
  enabled: true
  optimizer_offload: true           # Optimizer state offload: Offload unsharded optimizer states
  optimizer_state_precision: "bf16"
  activation_spill: true             # Activation spilling: Activation spilling
```

**Validate your config**:
```bash
ironcore config-check --config configs/your_config.yaml
```

### OffloadConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Master switch for all offload features |
| `optimizer_offload` | bool | false | Enable Optimizer state offload (optimizer state offload to CPU) |
| `optimizer_state_precision` | str | "fp32" | Precision for optimizer states: "fp32", "bf16", "fp16" |
| `optimizer_min_param_elements` | int | 65536 | Skip offload for params smaller than this |
| `weight_offload` | bool | false | Enable Weight streaming (weight streaming) |
| `weight_prefetch_layers` | int | 2 | Forward prefetch depth for weight streaming |
| `backward_weight_prefetch_layers` | int | 1 | Backward prefetch depth for weight streaming |
| `weight_storage_precision` | str | "bf16" | Precision for streamed weights on host |
| `gpu_staging_pool_mb` | float | 0.0 | GPU staging pool size (0 = auto) |
| `gpu_staging_chunk_mb` | float | 256.0 | Chunk size for GPU staging pool |
| `activation_spill` | bool | false | Enable Activation spilling (activation spilling) |
| `activation_spill_granularity` | str | "sub_layer" | "sub_layer" or "full_layer" |
| `pinned_memory_pool_gb` | float | -1.0 | Pinned memory pool size (-1 = auto-detect) |
| `pinned_chunk_gb` | float | 4.0 | Chunk size for pinned pool allocation |
| `prefetch_streams` | int | 1 | Number of CUDA streams for async transfers |

### Detailed Argument Explanations

#### Master Switch

| Argument | Details |
|----------|---------|
| `enabled` | **Required**: Set to `true` to enable all offload features. All other offload options are ignored unless this is `true`. |

#### Optimizer State Offload (`optimizer_offload`)

| Argument | Details |
|----------|---------|
| `optimizer_offload` | **What it does**: Moves AdamW optimizer states (exp_avg, exp_avg_sq) from GPU to CPU. Saves ~50% GPU VRAM (optimizer states = 2× model size for AdamW).<br>**When to use**: Model parameters fit but optimizer states don't. Example: 7B bf16 model = 14GB params + 52GB fp32 optimizer states = 66GB total. Optimizer state offload reduces to ~27GB.<br>**Precision options**: See `optimizer_state_precision` below. |
| `optimizer_state_precision` | **Options**: `"fp32"` (default, safest), `"bf16"` (recommended, halves host memory), `"fp16"` (risky, may overflow).<br>**Trade-off**: bf16 saves 50% host memory with minimal convergence impact. fp16 can overflow AdamW's exp_avg_sq.<br>**Host memory**: fp32 = 2× model size, bf16 = 1× model size. |
| `optimizer_min_param_elements` | **What it does**: Skips CPU offload for parameters smaller than this threshold. Default: 65536 elements.<br>**Why**: Tiny params (LayerNorm, biases) have high PCIe transfer overhead relative to memory savings.<br>**When to adjust**: Increase if you have many small params and want more GPU optimizer states. |

#### Weight Streaming (`weight_offload`)

| Argument | Details |
|----------|---------|
| `weight_offload` | **What it does**: Stores layer weights on CPU, streams to GPU during forward pass. Saves ~30-50% GPU VRAM.<br>**⚠️ Auto-enables Activation spilling**: Setting this to `true` automatically enables `activation_spill` for safety.<br>**Incompatible with**: FSDP (blocked by config validation). |
| `weight_prefetch_layers` | **What it does**: Number of layers ahead to prefetch weights. Default: 2.<br>**How it works**: While computing layer N, H2D transfers weights for layers N+1, N+2 asynchronously.<br>**When to adjust**: Increase (3-4) for faster GPUs or larger batch sizes. Decrease (1) for low-memory systems. |
| `backward_weight_prefetch_layers` | **What it does**: Backward pass prefetch depth. Default: 1.<br>**How it works**: During backward, prefetches weights for previous layers to overlap with gradient computation.<br>**When to adjust**: Usually leave at 1. Higher values have diminishing returns. |
| `weight_storage_precision` | **What it does**: Precision for storing weights on host. Default: `"bf16"`.<br>**Options**: `"bf16"` (recommended), `"fp32"` (doubles host memory), `"fp16"` (risky). |
| `gpu_staging_pool_mb` | **What it does**: GPU buffer size for prefetched weights. Default: 0 (auto-size).<br>**How it works**: Weights are staged here before layer computation. Pool size = (prefetch_layers + 1) × layer_size.<br>**When to set manually**: Only if auto-size fails (rare). |
| `gpu_staging_chunk_mb` | **What it does**: Chunk size for GPU staging pool allocation. Default: 256 MB.<br>**When to adjust**: Increase (512-1024) for very large layers. Decrease (128) for fragmented GPU memory. |

#### Activation Spilling (`activation_spill`)

| Argument | Details |
|----------|---------|
| `activation_spill` | **What it does**: Spills intermediate activations (layer inputs, attention outputs) to CPU during forward, restores during backward. Saves ~20-30% GPU VRAM.<br>**How it works**: Replaces activation checkpointing with CPU-backed spill.<br>**Granularity**: See `activation_spill_granularity`. |
| `activation_spill_granularity` | **Options**: `"sub_layer"` (default, 2 spills/layer), `"full_layer"` (1 spill/layer).<br>**Trade-off**: `sub_layer` spills more (better VRAM savings) but has higher PCIe overhead. `full_layer` retains more GPU activations. |

#### Memory Pool Configuration

| Argument | Details |
|----------|---------|
| `pinned_memory_pool_gb` | **What it does**: Total pinned host memory budget. Default: -1 (auto-detect from available RAM).<br>**Why pinned**: Required for async CUDA transfers. Non-pinned memory forces synchronous transfers.<br>**Auto-detect**: When -1, uses 50% of available RAM (via `psutil`).<br>**When to set manually**: Override auto-detect for specific constraints. Example: `16.0` for 16GB pool. |
| `pinned_chunk_gb` | **What it does**: Chunk size for pinned memory allocations. Default: 4.0 GB.<br>**When to adjust**: Increase (8-16) for large models to reduce allocation overhead. Decrease (1-2) for fragmented host memory. |
| `prefetch_streams` | **What it does**: Number of CUDA streams for async transfers. Default: 1.<br>**When to increase**: 2-4 streams can overlap weight H2D with activation H2D. Most useful with Weight streaming+Activation spilling combined.<br>**Diminishing returns**: >4 streams rarely helps (PCIe bandwidth is the bottleneck). |

### Usage Examples by Scenario

#### Example 1: Single GPU, model doesn't fit (Optimizer state offload+Weight streaming+Activation spilling)
```yaml
offload:
  enabled: true
  optimizer_offload: true
  optimizer_state_precision: "bf16"
  weight_offload: true
  weight_prefetch_layers: 2
  activation_spill: true
  activation_spill_granularity: "sub_layer"
  pinned_memory_pool_gb: -1.0  # Auto-detect from available RAM
```

#### Example 2: Single GPU, only optimizer states are large (Optimizer state offload only)
```yaml
offload:
  enabled: true
  optimizer_offload: true
  optimizer_state_precision: "bf16"
  pinned_memory_pool_gb: 8.0  # 8GB pinned pool sufficient for bf16 states
```

#### Example 3: DDP + ZeRO-1 with Optimizer state offload+Activation spilling (multi-GPU optimizer savings)
```yaml
parallel:
  use_distributed_optimizer: true
  dist_opt_bucket_cap_mb: 25.0
offload:
  enabled: true
  optimizer_offload: true
  optimizer_state_precision: "bf16"
  activation_spill: true
  pinned_memory_pool_gb: -1.0
```

#### Example 4: FSDP SHARD_GRAD_OP + Optimizer state offload + Activation spilling (best for most multi-GPU)
```yaml
parallel:
  use_fsdp: true
  fsdp_sharding_strategy: "shard_grad_op"
  fsdp_use_orig_params: true  # REQUIRED for Optimizer state offload+FSDP compatibility
offload:
  enabled: true
  optimizer_offload: true
  optimizer_state_precision: "bf16"
  activation_spill: true
```

#### Example 5: FSDP FULL_SHARD + Activation spilling (Optimizer state offload is blocked to avoid duplication)
```yaml
parallel:
  use_fsdp: true
  fsdp_sharding_strategy: "full"
offload:
  enabled: true
  activation_spill: true  # Activation spilling works, Optimizer state offload must be disabled
```

### Validation Rules (Automatically Enforced)

The following configurations are automatically blocked with `ValueError`:

1. **Optimizer state offload + FSDP FULL_SHARD**: Duplicates optimizer states in host memory
   - Use `fsdp_sharding_strategy: "shard_grad_op"` or disable Optimizer state offload

2. **Optimizer state offload + FSDP CPUOffload**: Redundant optimizer offloading
   - Use only one: either Optimizer state offload or FSDP CPUOffload

3. **Optimizer state offload + FSDP without use_orig_params**: Breaks optimizer parameter references
   - Set `fsdp_use_orig_params: true` when using Optimizer state offload with FSDP

4. **Weight streaming + FSDP**: Weight streaming conflicts with FSDP parameter sharding
   - Already blocked; weight streaming is DDP/single-GPU only

### Config Validation Warnings

The following produce warnings (not errors):

1. **Optimizer state offload + FSDP SHARD_GRAD_OP without use_orig_params**: Required for correctness
2. **Weight streaming enabled without activation_spill**: Auto-enables activation_spill (Activation spilling)
3. **pinned_memory_pool_gb exceeds 80% of total RAM**: Host OOM risk
4. **TP > 1 with offload enabled**: Experimental, each rank streams independently

---

### 7.3 Precision guidance

| Setting | Host memory | Training stability | When to use |
|---------|------------|-------------------|-------------|
| `optimizer_state_precision: "fp32"` | 2x (baseline) | Best | Default. Always safe. |
| `optimizer_state_precision: "bf16"` | 1x (halved) | Good for most models | When host memory is tight. Monitor loss for divergence. |
| `optimizer_state_precision: "fp16"` | 1x (halved) | Risky (overflow) | Not recommended for AdamW (exp_avg_sq can overflow fp16 range). |

> **RTX 3090 (Ampere) note**: bf16 on RTX 3090 runs via FP32 ALU emulation, so it may be slightly slower than H100/A100, but is still safer than fp16 for AdamW (exp_avg_sq can exceed fp16 dynamic range). bf16 is recommended over fp16 on Ampere.

### 7.4 Activation spilling activation_spill_granularity guidance

| Granularity | Host memory per layer | GPU memory saved | When to use |
|-------------|----------------------|-----------------|-------------|
| `sub_layer` | 2 activations (hidden + norm_input) | Maximum | Default. Best for most cases. |
| `full_layer` | 1 activation (layer input only) | Less (retains attention/MLP intermediates) | When host memory is very tight and some GPU intermediates are acceptable. |

---

### 7.5 Telemetry and Monitoring

The offload system includes built-in telemetry for monitoring H2D/D2H transfers, bandwidth, and stall events during training.

#### Enabling Telemetry

Set the environment variable before training:
```bash
export IRONCORE_OFFLOAD_TELEMETRY=1
ironcore train --config configs/offload.yaml
```

#### Live Monitoring (Terminal Visualizer)

Add the visualizer to your training script for real-time metrics:

```python
from ironcore.utils.offload_visualizer import start_offload_visualizer

# Start visualizer (updates every 10 steps)
viz = start_offload_visualizer(update_interval=10)

try:
    trainer.train()
finally:
    viz.stop()  # Prints final summary
```

**Output example**:
```
============================================================
[Offload Telemetry] Step 100
────────────────────────────────────────────────────────────
H2D: 45.23 GB | 28.50 GB/s | 450 transfers
D2H: 12.87 GB | 24.30 GB/s | 225 transfers
Stalls: 3 events | 125.3 ms total
Queue: 2/8 depth
============================================================
```

#### Metrics Tracked

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `total_h2d_bytes` | Cumulative host→-device data | Monitor trend |
| `total_d2h_bytes` | Cumulative device→host data | Monitor trend |
| `h2d_bandwidth_gb_s` | Effective H2D bandwidth | >20 GB/s (PCIe 4.0) |
| `d2h_bandwidth_gb_s` | Effective D2H bandwidth | >20 GB/s (PCIe 4.0) |
| `stall_events` | Transfer queue full events | =0 or very low |
| `max_queue_depth` | Maximum observed queue depth | Should be ≤ `prefetch_streams` |

#### Hardware Benchmark

Before training, benchmark your system's PCIe/NVLink bandwidth:

```bash
# Basic benchmark (10-500 MB transfers)
python scripts/benchmark_offload_pcie.py --sizes 10 50 100 500

# Compare NVLink on vs off
NCCL_P2P_DISABLE=1 python scripts/benchmark_offload_pcie.py --output no_nvlink.json
NCCL_P2P_DISABLE=0 python scripts/benchmark_offload_pcie.py --output with_nvlink.json

# Custom sizes and output
python scripts/benchmark_offload_pcie.py --sizes 100 500 1000 --output bandwidth_results.json
```

**Output example**:
```json
{
  "device": "NVIDIA GeForce RTX 3090",
  "nvlink_enabled": false,
  "dtype": "bfloat16",
  "benchmarks": {
    "h2d": [
      {"size_mb": 100, "bandwidth_gb_s": 28.5},
      {"size_mb": 500, "bandwidth_gb_s": 29.2}
    ],
    "d2h": [
      {"size_mb": 100, "bandwidth_gb_s": 24.8},
      {"size_mb": 500, "bandwidth_gb_s": 25.1}
    ]
  }
}
```

#### Troubleshooting via Telemetry

| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| Low bandwidth (<10 GB/s) | Non-pinned memory, slow storage, or PCIe 3.0 | Check `pinned_memory_pool_gb` is set |
| High stall events | Transfer queue backing up | Increase `prefetch_streams` or reduce `weight_prefetch_layers` |
| D2H much slower than H2D | GPU compute blocking transfers | Check for unnecessary `torch.cuda.synchronize()` calls |
| Bandwidth drops over time | Thermal throttling or memory fragmentation | Monitor GPU temp, check `nvidia-smi` |

---

## 8. Historical Implementation Notes

**File**: `ironcore/config/__init__.py`

**Change 1**: Block Optimizer state offload + FSDP FULL_SHARD (host OOM risk from duplicating optimizer states)

```python
# After the existing weight_offload + FSDP block (line ~210)
if config.offload.optimizer_offload and config.parallel.use_fsdp:
    if config.parallel.fsdp_sharding_strategy == "full":
        raise ValueError(
            "offload.optimizer_offload with FSDP full_shard duplicates "
            "optimizer states in host memory. Use fsdp_sharding_strategy: "
            "shard_grad_op instead, or disable optimizer_offload to let "
            "FSDP handle optimizer state sharding."
        )
```

**Change 2**: Block Optimizer state offload + FSDP CPUOffload (redundant, both run optimizer on CPU)

```python
if config.offload.optimizer_offload and config.parallel.use_fsdp:
    if config.parallel.fsdp_offload_params:
        raise ValueError(
            "offload.optimizer_offload is redundant with FSDP CPUOffload. "
            "Both run the optimizer step on CPU. Disable optimizer_offload "
            "and let FSDP handle CPU offloading, or disable fsdp_offload_params."
        )
```

**Change 3**: Warn Optimizer state offload + FSDP without use_orig_params

```python
if config.offload.optimizer_offload and config.parallel.use_fsdp:
    if not config.parallel.fsdp_use_orig_params:
        import warnings
        warnings.warn(
            "optimizer_offload with FSDP requires use_orig_params=True. "
            "Without it, FSDP replaces parameters with FlatParameters, "
            "breaking the optimizer's parameter references.",
            stacklevel=2,
        )
```

### 8.2 Optimizer creation order fix

**File**: `ironcore/trainers/base_trainer.py`

**Problem**: Optimizer is created at line 314 BEFORE FSDP wrapping at line 358. With `use_orig_params=True`, this works because FSDP preserves original parameter objects. Without it, FSDP replaces parameters with FlatParameters, breaking optimizer references.

**Fix**: When FSDP is active, require `use_orig_params=True`. The current code already defaults to `False`.

```python
# In _build_model_and_optimizer(), before get_optimizer():
if self.config.parallel.use_fsdp and not self.config.parallel.fsdp_use_orig_params:
    if self.config.offload.optimizer_offload:
        raise ValueError(
            "FSDP with optimizer_offload requires fsdp_use_orig_params=True. "
            "Set parallel.fsdp_use_orig_params=true in your config."
        )
```

### 8.3 Activation backward prefetch implementation

**Priority**: Medium. 3-6% throughput improvement for large models with Activation spilling.

**Files to modify**:

1. `ironcore/offload/hooks.py` — Add `prefetch_activation()` method to `ActivationSpillManager`. Submits async H2D without blocking. Stores the handle in `SpilledActivation`.

2. `ironcore/offload/scheduler.py` — In `on_backward_layer_end()`, call `self._spill_manager.prefetch_activation()` for the previous layer's activations.

3. `ironcore/offload/hooks.py` — In `on_sublayer_backward()`, check `is_prefetched` flag. If true, only wait for the prefetch handle (not a new H2D submission).

**Testing**: Add a test that verifies backward correctness with prefetch enabled. Compare loss trajectory with and without prefetch over 50 steps.

### 8.4 Configurable transfer streams

**Priority**: Low. Only helps when both weight and activation transfers need to overlap simultaneously.

**File**: `ironcore/offload/config.py`

Add to `OffloadConfig`:
```python
prefetch_streams: int = 1  # Number of dedicated CUDA streams for async transfers
```

**File**: `ironcore/offload/transfer_engine.py`

Change `from_config()` to read the config value instead of hardcoding `prefetch_streams=1`.

### 8.5 Testing requirements

| Test | What it verifies |
|------|-----------------|
| `test_optimizer_offload_fsdp_full_shard_blocked` | Config validation blocks Optimizer state offload + FULL_SHARD |
| `test_optimizer_offload_fsdp_cpuoffload_blocked` | Config validation blocks Optimizer state offload + CPUOffload |
| `test_activation_spill_fsdp_integration` | Activation spilling activation spill works with FSDP FULL_SHARD. Loss parity over 50 steps. |
| `test_optimizer_offload_shard_grad_op_fsdp` | Optimizer state offload + SHARD_GRAD_OP + Activation spilling with FSDP. Optimizer states on CPU, params sharded. Loss parity. |
| `test_optimizer_offload_distributed_optimizer` | Optimizer state offload + DistributedOptimizer. Each rank holds 1/N optimizer states on CPU. |
| `test_backward_prefetch_correctness` | Backward produces identical gradients with and without activation prefetch. |
| `test_backward_prefetch_throughput` | Measure step time with and without prefetch. Verify improvement. |

### 8.6 Summary of recommended changes (ordered by priority)

1. **Config validation** (8.1) — Prevent misconfiguration that causes host OOM. Low risk, high impact.
2. **Optimizer creation order guard** (8.2) — Require `use_orig_params=True` when Optimizer state offload+FSDP. Low risk.
3. **Activation backward prefetch** (8.3) — 3-6% throughput improvement for Activation spilling users. Medium effort.
4. **Configurable transfer streams** (8.4) — Minor config change. Low priority.
5. **Integration tests** (8.5) — Verify all configurations work end-to-end.

---

## 9. Additional Considerations

Topics not covered in the English original but encountered during operations and implementation:

### 9.1 TP (Tensor Parallel) × Offload Interaction

Current `ironcore/offload/` code is not TP-aware (grep for `tp_size`, `expert_parallel` yields zero results).

- **Optimizer state offload + TP**: Optimizer states are created in TP-sharded weight form. Optimizer state offload offloads per-parameter, so correctness is preserved, but the `optimizer_min_param_elements` threshold applies to sharded shapes. After sharding, parameters may fall below the threshold and unintentionally remain on GPU.
- **Weight streaming + TP**: Streaming TP-sharded weights reduces PCIe traffic, but verification is needed to ensure all-gather/all-reduce doesn't conflict with weight rematerialization timing.
- **Activation spilling + TP**: Activations are TP-split along sequence/hidden dimensions, so spill size is also 1/TP. High compatibility likelihood.

**Recommendation**: Add TP integration tests.

### 9.2 MoE / Expert Parallelism + Offload

Undocumented in this document. EP (Expert Parallelism) shards experts across ranks. The offload layer-by-layer scheduler doesn't operate at the expert level. Clarify whether MoE model training works.

### 9.3 Pipeline Parallelism

**Out of scope**. This codebase targets single-node workstation optimization. PP is currently unimplemented and there are no plans to introduce it (low value for single-node/2-GPU environments). If multi-node track is introduced in the future, this will be addressed in a separate integration phase.

### 9.4 Pinned Memory Default 100 GB Risk

The `pinned_memory_pool_gb: 100.0` default poses host OOM or OOM-killer risk on consumer systems (64-128 GB RAM). Recommend RAM auto-detection (e.g., within 50% of available RAM) + conservative default (e.g., 32 GB).

### 9.5 Checkpoint Compatibility

Offload state save/restore is automatically handled at [checkpointing/native.py:380-461](../ironcore/checkpointing/native.py#L380-L461) (not mentioned in English original).

**Save (line 384-394)**:
- Checks `optimizer.offload_enabled`, param's `offloadable` attribute, and `offload_min_param_elements` threshold
- Offloaded states (`exp_avg`, `exp_avg_sq`, AMSGrad's `max_exp_avg_sq`) are serialized while preserving CPU location — no GPU staging

**Restore (line 415-461)**:
- Handles TP-shard splitting (line 414-446) — maintains state consistency across TP ranks
- Re-distributes states to CPU or GPU based on the same offload criteria (line 449-460)

**Verified tests**:
- [tests/unit/offload/test_checkpoint_offload.py](../tests/unit/offload/test_checkpoint_offload.py)
- [tests/integration/offload/test_checkpoint_offload.py](../tests/integration/offload/test_checkpoint_offload.py)

**HuggingFace interop**: Same logic applies. Automatic dtype conversion when exporting/importing fp32 states.

**Caution**: Offload config must match between save and restore (`optimizer_state_precision`, `optimizer_min_param_elements`). If configs differ, some states may restore to incorrect device locations.

### 9.6 Convergence Regression

User memory reports an Weight streaming+Activation spilling+grad_accum>1 device mismatch bug (commit `4a1597f`) and 1000-step loss divergence issues. Document should specify:
- Regression test location
- Validation step count (e.g., 1000-step baseline)
- Known residual caveats

### 9.7 CPU Compute Thread Contention

**When critical**: Optimizer state offload-only (CPU compute) mode where params are on GPU and Weight streaming is inactive. AdamW math runs on CPU via SIMD/AVX-512 (MKL), competing with dataloader workers, gradient all-reduce background threads, NCCL helpers, and OMP/MKL thread pools.

**Recommended formula**:
```bash
export OMP_NUM_THREADS=$(python -c "import os; print(max(1, os.cpu_count() - dataloader_workers - 2))")
export MKL_NUM_THREADS=$OMP_NUM_THREADS
```

Where `dataloader_workers` is `TrainerConfig.dataloader_num_workers`.

**Example** (16-core CPU + dataloader_workers=4):
```bash
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
```

**Verification**: Use `top -H` or `htop` to check CPU thread distribution during training. If AdamW step time exceeds GPU compute time, thread starvation is likely.

**When Weight streaming/Activation spilling active**: AdamW takes GPU-compute path (params on CPU), reducing OMP impact. However, dataloader always uses CPU, so base recommendation remains.

### 9.8 NUMA Locality

On multi-socket systems, if pinned memory doesn't match the GPU's NUMA node, cross-socket UPI traffic increases PCIe latency. **Ignorable on single-socket workstations** (typical 2x RTX 3090 environment).

**Verification** (check if multi-socket):
```bash
lscpu | grep "NUMA node(s)"   # > 1 means multi-socket
nvidia-smi topo -m            # GPU↔NUMA node mapping
```

**Recommendation** (multi-socket systems only):
```bash
# Bind each GPU rank to nearest NUMA node
numactl --cpunodebind=0 --membind=0 \
  torchrun --nproc_per_node=2 -m ironcore train --config configs/...

# Or per-rank binding (script wrapper needed)
```

**Programmatic**: PyTorch lacks NUMA-aware pinned alloc API. After `torch.cuda.set_device()`, use `os.sched_setaffinity()` to bind CPU affinity to GPU's NUMA node as a second-best approach.

**Phase B-1 (`system_info.py`) connection**: `psutil.virtual_memory()` doesn't provide NUMA-aware info. If NUMA detection is needed in the future, consider `numa` package or parsing `/sys/devices/system/node/`.

### 9.9 PCIe Bandwidth Contention

On systems without NVLink (e.g., PCIe-only 4×GPU), Optimizer state offload grad D2H/delta H2D contends with DDP all-reduce / FSDP all-gather for PCIe. Quantitative analysis:
- PCIe Gen4 x16: ~32 GB/s unidirectional
- 13B model grad: 26 GB → requires 0.8s/step transfer bandwidth

Assume half effective bandwidth when bidirectional + concurrent communication.

### 9.10 AMSGrad Memory Correction

Section 5.1 table only covers standard AdamW (`exp_avg + exp_avg_sq`). AMSGrad persistently tracks `max_exp_avg_sq`:

| Optimizer | Formula | 13B host (Optimizer state offload bf16) | 13B host (fp32) |
|---|---|---|---|
| AdamW | `2 × P × D` | 26 GB | 52 GB |
| AdamW + AMSGrad | `3 × P × D` | **39 GB** | **78 GB** |

**Applies when**: AMSGrad enabled in [optimizer/](../ironcore/optimizer/) (config `optimizer.amsgrad: true`). Optimizer state offload offload state also stores `max_exp_avg_sq` on CPU at the same precision.

**Host memory impact**: 1.5×. All "Host per rank" values in §5.2 table should be multiplied by 1.5 for re-estimation. Phase B-1 auto-recommendation (§9.4) should also reflect AMSGrad enablement.

**English sync**: This change adds the AMSGrad row to the §5.1 table in [docs/offload_fsdp_architecture.md](offload_fsdp_architecture.md#51-per-component-memory-formulas).

### 9.11 CUDA Graph Incompatibility

**Current state**: Codebase doesn't use CUDA Graphs (grep for `torch.cuda.graph`, `make_graphed_callables`, `CUDAGraph` yields 0 results). So no actual conflicts exist.

**Fundamental incompatibility**: Optimizer state offload/Weight streaming/Activation spilling async prefetch produces different tensor pointers/shapes each step — dynamic behavior. CUDA Graphs replay captured kernel sequences as frozen kernels + frozen pointers, incompatible with this dynamic pattern.

**If introduced in future**:
- Don't use `torch.cuda.graph` / `make_graphed_callables` for training with offload enabled
- CUDA Graphs only viable for inference path (offload disabled)
- No need to explicitly block this in config validation (until codebase introduces graphs)

### 9.12 Host OOM Recovery

When pinned pool is exhausted:
- Current: May raise exceptions
- Recommended: Fallback to synchronous path or clear error message (guidance to increase `pinned_memory_pool_gb`)

### 9.13 Operational Telemetry / Metrics

Recommended metrics for production monitoring:
- Transfer queue depth (submitted vs completed difference)
- Stall count (cumulative `wait()` blocking time)
- Host memory headroom (`MemAvailable` vs pool usage)
- PCIe utilization (nvidia-smi `dmon -s p`)

### 9.14 `weight_offload` Auto-Activation Side Effect

When `weight_offload=true` is set, [config/__init__.py:211-219](../ironcore/config/__init__.py#L211-L219) **automatically enables** `activation_spill` + emits `warnings.warn()`.

**Root cause**: Weight eviction during backward requires autograd graph to be separated at layer boundaries for safety (otherwise backward referencing evicted weights causes segfault). `activation_spill`'s sub-layer boundaries provide this separation.

**Observable symptoms**:
- Setting only `weight_offload: true` creates spilled activation regions in host memory
- Training log outputs: `UserWarning: offload.weight_offload requires activation spilling for weight eviction (no_autograd_graph). Enabling offload.activation_spill automatically.`

**Response**:
- Intentional behavior — leave as-is
- To explicitly disable, set `weight_offload: false` or explicitly set `activation_spill: true` together (to avoid silent enable)

**No change intended**: This auto-activation is a safety guard by design. Don't remove. Just documented in Appendix B table notes (in this change).

---

## 10. Verification Results and Gap Summary

Cross-referencing English document claims against actual codebase:

| Item | English doc claim | Actual code | Location |
|------|---------------|-----------|------|
| Optimizer state offload + FSDP FULL_SHARD blocked | "NOT blocked" | ✅ Matches — no blocking | [config/__init__.py:206-210](../ironcore/config/__init__.py#L206-L210) |
| Optimizer state offload + FSDP CPUOffload blocked | "NOT blocked" | ✅ Matches — no blocking | Same |
| Backward activation H2D | "Synchronous blocking (Gap 1)" | ✅ Matches — `wait()` + `synchronize_with_default_stream()` | [hooks.py:333-334](../ironcore/offload/hooks.py#L333-L334) |
| Backward weight prefetch | "1-layer hardcoded (Gap 2)" | ✅ Matches — only `layer_idx - 1` | [scheduler.py:463-464](../ironcore/offload/scheduler.py#L463-L464) |
| `prefetch_streams=1` (Gap 3) | "Not configurable" | ✅ Matches — hardcoded in `from_config` | [transfer_engine.py:77](../ironcore/offload/transfer_engine.py#L77) |
| `drain_completed()` (Gap 4) | "Dead code" | ✅ Matches — not called outside tests | [transfer_engine.py:196](../ironcore/offload/transfer_engine.py#L196) |
| FSDP `BACKWARD_PRE` | "Hardcoded" | ✅ Matches | [parallel/parallel.py:153](../ironcore/parallel/parallel.py#L153) |
| Optimizer creation order | "Optimizer created before FSDP wrap" | ✅ Matches — line 314 vs 358 | [trainers/base_trainer.py](../ironcore/trainers/base_trainer.py) |

**Test coverage gaps**: No integration tests for Optimizer state offload+FSDP, Activation spilling+FSDP, Optimizer state offload+DistOpt ([tests/](../tests/) search results).

**Recent related commits**:
- `932779a` — Optimizer state offload-only mode CPU AdamW (40%+ VRAM savings)
- `4a1597f` — Activation spilling grad_accum>1 device mismatch fix
- `78adb10` — backward prefetch overlap attempt (though activation H2D still blocking)

---

## Appendix A: Current enforcement status

| Combination | Config validation | Runtime check | Status |
|---|---|---|---|
| DistributedOptimizer + FSDP | **Blocked** (error) | N/A | Correct |
| weight_offload + FSDP | **Blocked** (error) | **Blocked** (scheduler) | Correct |
| optimizer_offload + FSDP FULL_SHARD | **NOT blocked** | N/A | **Gap (host OOM risk)** |
| optimizer_offload + FSDP CPUOffload | **NOT blocked** | N/A | **Gap (redundant)** |
| activation_spill + FSDP | Not blocked (compatible) | Not blocked (compatible) | Correct |
| optimizer_offload + DistributedOptimizer | Not blocked (compatible) | Not blocked (compatible) | Correct |

## Appendix B: OffloadConfig field reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | False | Master switch for all offload features |
| `optimizer_offload` | bool | False | Optimizer state offload: offload optimizer states to CPU |
| `optimizer_state_precision` | str | "fp32" | Precision for CPU optimizer states (fp32/bf16/fp16) |
| `optimizer_min_param_elements` | int | 65536 | Skip offload for params smaller than this |
| `weight_offload` | bool | False | Weight streaming: enable weight streaming. **⚠️ Setting to `true` automatically enables `activation_spill`** (see [config/__init__.py:211-219](../ironcore/config/__init__.py#L211-L219) and §9.14) |
| `weight_prefetch_layers` | int | 2 | Number of layers to prefetch ahead |
| `weight_storage_precision` | str | "bf16" | Precision for host weight tiles |
| `gpu_staging_pool_mb` | float | 0.0 | GPU staging pool size (0 = auto) |
| `gpu_staging_chunk_mb` | float | 256.0 | GPU staging chunk size |
| `activation_spill` | bool | False | Activation spilling: enable activation spilling |
| `activation_spill_granularity` | str | "sub_layer" | "sub_layer" or "full_layer" |
| `pinned_memory_pool_gb` | float | 100.0 | Total pinned host memory budget. **⚠️ 100GB default is excessive for consumer systems** (see §9.4) |
| `pinned_chunk_gb` | float | 4.0 | Pinned memory chunk size |

## Appendix C: ParallelConfig FSDP fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_fsdp` | bool | False | Enable FSDP wrapping |
| `fsdp_sharding_strategy` | str | "full" | "full", "hybrid", "no_shard", or "shard_grad_op" |
| `fsdp_offload_params` | bool | False | FSDP CPUOffload(offload_params=True) |
| `fsdp_use_orig_params` | bool | False | Preserve original parameter objects (required for Optimizer state offload+FSDP) |
| `fsdp_mixed_precision` | str | "native" | "native" or "mixed" |
| `use_distributed_optimizer` | bool | False | Enable ZeRO-1 DistributedOptimizer (DDP only) |
