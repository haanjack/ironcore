# Inference and Generation

## Generation loop

`LanguageModel.generate()` in `ironcore/language_model.py` follows the standard prefill → decode pattern:

1. **Prefill:** the full prompt `[B, prompt_len]` is processed in a single forward pass. Key/value tensors are computed for all prompt positions.
2. **Decode:** one token is generated per step. Only the last generated token is passed to the model; cached KV tensors from previous steps are reused.

The loop ends when all sequences have emitted an EOS token or `max_new_tokens` is reached.

## Stateless KV cache (`use_cache=True`)

When `model.forward(input_ids, use_cache=True, past_key_values=past_kv)` is called:
- The model appends new KV tensors to `past_key_values` and returns the updated tuple.
- The caller is responsible for passing the updated tuple back on the next call.

This is the mode used in `generate_rollouts_batched()` for GRPO rollouts.

## Stateful `KVCacheManager`

`KVCacheManager` in `ironcore/layers/kv_cache.py` provides a persistent, stateful cache suitable for continuous batching.

**Cache structure per layer:**
```
key_cache:   [batch, max_seq_len, num_local_kv_groups, head_dim]
value_cache: [batch, max_seq_len, num_local_kv_groups, head_dim]
cache_positions: [batch]   # current fill position per sequence
```

**TP-aware:** stores only the local KV groups for this TP rank. `num_local_kv_groups = num_attention_groups // tensor_model_parallel_size`.

**Key methods:**

| Method | Description |
|---|---|
| `initialize(batch_size, num_layers, device, dtype)` | Allocates cache buffers |
| `update_layer(layer_idx, key, value)` | Writes new KV into cache at current position, returns full KV slice `[:, :end_pos]` |
| `reset(batch_indices=None)` | Evicts specified sequences (or all if `None`) by zeroing `cache_positions` |
| `get_layer_kv(layer_idx, start_pos, end_pos)` | Read-only access to a slice of the cache |

Position tracking:
- Uniform positions: `update_layer()` with no explicit position argument — all sequences at the same position.
- Per-sequence positions: pass a `positions` tensor `[batch]` for continuous batching scenarios where sequences have different fill levels.

## Prefix caching for GRPO rollouts

`_expand_kv_cache()` in `ironcore/alignment/rollout.py` replicates the prompt's KV cache from `[B, prompt_len, num_heads, head_dim]` to `[B×G, prompt_len, num_heads, head_dim]` using `tensor.repeat_interleave(group_size, dim=0)`.

This avoids re-running the prompt through the model G times. The expanded cache is passed as `past_key_values` for the decode phase, which then generates G completions per prompt in parallel.

## FlashAttention integration

`Attention` in `ironcore/layers/attention.py` uses `[b, s, n, d]` layout (batch, sequence, heads, head_dim).

For packed SFT sequences, the collator produces `cu_seqlens` — cumulative sequence length boundaries — which are passed to FlashAttention to enforce block-diagonal attention (each sample attends only to its own tokens, not to other packed samples in the same row).

## Sampling

Temperature scaling, top-k, and top-p (nucleus) filtering are applied in `_sample_tokens_batched()` in `ironcore/alignment/rollout.py`:

1. Apply temperature: `logits /= temperature`
2. Top-k: mask all but the k largest logits
3. Top-p: mask tokens whose cumulative probability (after sorting) exceeds `top_p`
4. Sample from the filtered distribution via `torch.multinomial`

Under tensor parallelism, the sampled token is broadcast from TP rank 0 to all other ranks so every rank operates on the same next-token. Greedy decoding (`do_sample=False`) uses `argmax` and requires no broadcast.

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `trainer.use_flash_attn` | `true` | Use FlashAttention kernel |
| `trainer.use_kv_cache_in_eval` | `true` | Use `KVCacheManager` during evaluation |
| `alignment.generation.max_new_tokens` | `512` | Maximum tokens to generate per sequence |
| `alignment.generation.temperature` | `1.0` | Sampling temperature |
| `alignment.generation.top_p` | `0.9` | Nucleus sampling threshold |
| `alignment.generation.top_k` | `0` | Top-k cutoff (0 = disabled) |
| `alignment.generation.do_sample` | `true` | Stochastic vs greedy decoding |
| `model.kv_cache.max_seq_length` | — | Maximum cached sequence length for `KVCacheManager` |
