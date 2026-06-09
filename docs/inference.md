# Inference and Generation

> This guide covers generation configuration and KV cache options. For cache data structures,
> paged attention mechanics, and TP-aware layout, see the
> [KV Cache & Inference system design](design/kv-cache.md).

## Generation

`LanguageModel.generate()` follows the standard prefill → decode pattern: the prompt is
processed in one forward pass, then tokens are generated one at a time reusing cached
KV tensors.

### Sampling parameters

```yaml
alignment:
  generation:
    max_new_tokens: 512
    temperature: 1.0       # 1.0 = no scaling; lower = sharper
    top_p: 0.9             # nucleus sampling; 1.0 = disabled
    top_k: 0               # top-k cutoff; 0 = disabled
    do_sample: true        # false = greedy (argmax)
```

These settings are used by both the GRPO rollout and any direct `generate()` call.

## KV cache options

IronCore provides two KV cache implementations for different workloads:

| Cache | Use case | Prefix sharing |
|---|---|---|
| `KVCacheManager` | Standard inference, eval | No |
| `BlockKVCacheManager` | GRPO rollout generation | Yes — one copy of prompt KV per group |

### Standard KV cache (`KVCacheManager`)

Pre-allocates a dense `[batch, max_seq_len, kv_groups, head_dim]` buffer per layer.
Used during evaluation when `trainer.use_kv_cache_in_eval: true`.

```yaml
model:
  kv_cache:
    max_seq_length: 2048   # maximum cached sequence length
```

### Paged KV cache for GRPO (`BlockKVCacheManager`)

Stores prompt KV in fixed-size pages with reference counting. All G completions for a
prompt share a single copy of the prompt's KV — avoids storing G identical copies.
Activated automatically when using `generate_rollouts_paged()` in the GRPO rollout path.

The page size and block table dimensions are derived from `model.kv_cache.max_seq_length`.

## FlashAttention

FlashAttention is enabled by default and handles both standard sequences and bin-packed
SFT sequences (block-diagonal attention enforced via per-sample position IDs).

```yaml
trainer:
  use_flash_attn: true   # default
```

Disable only for debugging or when running without a CUDA-capable device.

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `trainer.use_flash_attn` | `true` | Use FlashAttention kernel |
| `trainer.use_kv_cache_in_eval` | `true` | Use `KVCacheManager` during evaluation |
| `model.kv_cache.max_seq_length` | `2048` | Max cached sequence length; also sets block table row size |
| `alignment.generation.max_new_tokens` | `512` | Maximum tokens to generate |
| `alignment.generation.temperature` | `1.0` | Sampling temperature |
| `alignment.generation.top_p` | `0.9` | Nucleus sampling threshold |
| `alignment.generation.top_k` | `0` | Top-k cutoff (0 = disabled) |
| `alignment.generation.do_sample` | `true` | Stochastic vs greedy decoding |
