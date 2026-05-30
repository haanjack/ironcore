# 13B Offload Capability Validation — Experiment Log

**Date:** 2026-05-30
**Branch:** feature/ram-host-optimizer-states
**Commit:** 71c7b24

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 3090 (24 GB VRAM) |
| CPU | AMD Ryzen, 12P / 24T |
| RAM | 123.5 GB DDR5 |
| PCIe | Gen4 x16 (~32 GB/s) |
| OS | Linux 6.8.0-101-generic |

## Model

LLaMA-13B style (12.9B params)

| Parameter | Value |
|-----------|-------|
| d_model | 5120 |
| d_ffn | 13824 |
| num_layers | 40 |
| num_attention_heads | 40 |
| num_attention_groups (GQA) | 8 |
| head_dim | 128 |
| max_seq_len | 1024 |
| precision | bfloat16 |
| normalization | RMSNorm |
| activation | SwiGLU |
| positional embedding | RoPE |

## Configuration

Full offload: optimizer_offload + weight_offload + activation_spill

| Setting | Value |
|---------|-------|
| weight_prefetch_layers | 2 |
| weight_storage_precision | bf16 |
| optimizer_state_precision | bf16 |
| activation_spill_granularity | sub_layer |
| micro_batch_size | 1 |
| seq_length | 1024 |

## Results

10 training steps, AdamW optimizer, cosine LR schedule.

| Metric | Value |
|--------|-------|
| Initial loss | ~12.0 |
| Final loss | ~11.3 |
| Peak VRAM | 22,710 MB |
| Steady-state VRAM | 1,594 MB |
| Peak RAM | ~92 GB |
| Avg step time | 60.8s |
| Total time | 608s |

## Findings

### Weight streaming keeps GPU small

GPU holds ~(prefetch_layers + 1) × layer_size at steady state (~1.6 GB).
Peak VRAM (22.7 GB) occurs during optimizer state transfer, not model weights.

### RAM is the bottleneck for 13B

- fp32 optimizer states (default): ~103 GB — exceeds 123 GB host RAM
- bf16 optimizer states: ~52 GB — fits comfortably
- Total with bf16: ~92 GB peak (model + pinned pool + optimizer states)

### Pinned pool fix impact

Fixed `PinnedMemoryPool._PinnedChunk.try_allocate()` — `.to(dtype)` was creating unpinned
copies, silently degrading all host-GPU async DMA to synchronous transfers. With `narrow()+view()`,
tensors now correctly retain `is_pinned()=True`.

## Prior Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| `tests/integration/offload/` (single GPU) | 26/26 | PASS |
| `tests/unit/offload/` (CPU + CUDA) | 138/138 | PASS |
| `tests/multi_gpu/offload/` (2x GPU, torchrun) | 5/5 | PASS |
