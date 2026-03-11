# HuggingFace Checkpoint Support Research

> Research date: 2026-03-10
> Purpose: Inform weight mapping implementation in ironcore for HF checkpoint interoperability.

---

## Core Convention

| Framework | Weight shape | Forward |
|-----------|-------------|---------|
| IronCore `ParallelLinear` | `[in, out]` | `y = x @ W` |
| HF `nn.Linear` (all modern models) | `[out, in]` | `y = F.linear(x, W) = x @ W.T` |
| HF GPT-2 `Conv1D` | `[in, out]` | `y = x @ W` |

**LLaMA and all derivatives require `.t()` on every linear weight when converting in either direction.**
**GPT-2 requires no transpose.**

---

## Model-by-Model Survey

### LLaMA-2 / LLaMA-3 / LLaMA-3.2

- `model_type`: `llama`
- Weight naming is **identical** across LLaMA-2, LLaMA-3, LLaMA-3.2.
- LLaMA-3 vocab: 128,256 (vs LLaMA-2: 32,000) — embedding shape differs, nothing else.
- No bias in attention/MLP.
- No tied embeddings.

```
model.embed_tokens.weight                         [vocab, hidden]
model.layers.{i}.input_layernorm.weight           [hidden]
model.layers.{i}.self_attn.q_proj.weight          [hn*hd, hidden]       ← [out, in]
model.layers.{i}.self_attn.k_proj.weight          [gn*hd, hidden]       ← [out, in]
model.layers.{i}.self_attn.v_proj.weight          [gn*hd, hidden]       ← [out, in]
model.layers.{i}.self_attn.o_proj.weight          [hidden, hn*hd]       ← [out, in]
model.layers.{i}.post_attention_layernorm.weight  [hidden]
model.layers.{i}.mlp.gate_proj.weight             [ffn, hidden]         ← [out, in]
model.layers.{i}.mlp.up_proj.weight               [ffn, hidden]         ← [out, in]
model.layers.{i}.mlp.down_proj.weight             [hidden, ffn]         ← [out, in]
model.norm.weight                                 [hidden]
lm_head.weight                                    [vocab, hidden]       ← [out, in]
```

- GQA: `num_key_value_heads` < `num_attention_heads`
- Activation: SwiGLU — separate `gate_proj` and `up_proj` (not fused in checkpoint)
- RMSNorm, RoPE

---

### Qwen2 / Qwen2.5

- `model_type`: `qwen2`
- Naming nearly identical to LLaMA. Key differences: has QKV bias (`attention_bias=True`), tied embeddings.

```
model.embed_tokens.weight                         [vocab, hidden]
model.layers.{i}.input_layernorm.weight           [hidden]
model.layers.{i}.self_attn.q_proj.weight          [hn*hd, hidden]       ← [out, in]
model.layers.{i}.self_attn.q_proj.bias            [hn*hd]
model.layers.{i}.self_attn.k_proj.weight          [gn*hd, hidden]       ← [out, in]
model.layers.{i}.self_attn.k_proj.bias            [gn*hd]
model.layers.{i}.self_attn.v_proj.weight          [gn*hd, hidden]       ← [out, in]
model.layers.{i}.self_attn.v_proj.bias            [gn*hd]
model.layers.{i}.self_attn.o_proj.weight          [hidden, hn*hd]       ← [out, in]
model.layers.{i}.post_attention_layernorm.weight  [hidden]
model.layers.{i}.mlp.gate_proj.weight             [ffn, hidden]         ← [out, in]
model.layers.{i}.mlp.up_proj.weight               [ffn, hidden]         ← [out, in]
model.layers.{i}.mlp.down_proj.weight             [hidden, ffn]         ← [out, in]
model.norm.weight                                 [hidden]
lm_head.weight                                    [vocab, hidden]       ← tied to embed_tokens
```

- GQA, SwiGLU, RMSNorm, RoPE
- QKV bias present; no o_proj bias

---

### Qwen3

- `model_type`: `qwen3`
- Same naming as Qwen2, **except**: QKV bias removed, QK-Norm added.

```
model.layers.{i}.self_attn.q_proj.weight          [hn*hd, hidden]       ← [out, in]  (no bias)
model.layers.{i}.self_attn.k_proj.weight          [gn*hd, hidden]       ← [out, in]  (no bias)
model.layers.{i}.self_attn.v_proj.weight          [gn*hd, hidden]       ← [out, in]  (no bias)
model.layers.{i}.self_attn.o_proj.weight          [hidden, hn*hd]       ← [out, in]
model.layers.{i}.self_attn.q_norm.weight          [hd]                  ← QK-Norm (RMSNorm per head)
model.layers.{i}.self_attn.k_norm.weight          [hd]                  ← QK-Norm (RMSNorm per head)
```

- Otherwise identical layout to Qwen2.
- GQA (e.g., 0.6B: 16 Q heads, 8 KV heads), SwiGLU, RMSNorm, RoPE

---

### Mistral

- `model_type`: `mistral`
- Naming **identical** to LLaMA. No biases anywhere. Sliding-window attention (config only, no extra weights).

```
(Same key names as LLaMA, same [out, in] shapes)
```

- GQA, SwiGLU, RMSNorm, RoPE
- `attention_bias=False`, `mlp_bias=False`

---

### Gemma 2

- `model_type`: `gemma2`
- Similar to LLaMA naming but with extra layernorm keys (pre/post attention and MLP).

```
model.embed_tokens.weight                              [vocab, hidden]   ← tied to lm_head
model.layers.{i}.input_layernorm.weight                [hidden]
model.layers.{i}.pre_feedforward_layernorm.weight      [hidden]          ← extra norm
model.layers.{i}.post_feedforward_layernorm.weight     [hidden]          ← extra norm
model.layers.{i}.post_attention_layernorm.weight       [hidden]
model.layers.{i}.self_attn.q_proj.weight               [hn*hd, hidden]   ← [out, in]
model.layers.{i}.self_attn.k_proj.weight               [gn*hd, hidden]   ← [out, in]
model.layers.{i}.self_attn.v_proj.weight               [gn*hd, hidden]   ← [out, in]
model.layers.{i}.self_attn.o_proj.weight               [hidden, hn*hd]   ← [out, in]
model.layers.{i}.mlp.gate_proj.weight                  [ffn, hidden]     ← [out, in]
model.layers.{i}.mlp.up_proj.weight                    [ffn, hidden]     ← [out, in]
model.layers.{i}.mlp.down_proj.weight                  [hidden, ffn]     ← [out, in]
model.norm.weight                                      [hidden]
```

- GQA, GELU activation (not SwiGLU), RMSNorm
- **Interleaved local/global attention** (sliding window every other layer) — config-only, no extra weights
- **Logit soft-capping** (config-only)
- `head_dim` is explicit in config (256 for 2B)
- Tied embeddings

---

### Phi-3 / Phi-3.5

- `model_type`: `phi3`
- **Fused QKV and fused gate+up** — major structural difference from LLaMA family.

```
model.embed_tokens.weight                         [vocab, hidden]
model.layers.{i}.input_layernorm.weight           [hidden]
model.layers.{i}.self_attn.qkv_proj.weight        [hn*hd + 2*gn*hd, hidden]  ← fused Q+K+V, [out, in]
model.layers.{i}.self_attn.o_proj.weight          [hidden, hn*hd]             ← [out, in]
model.layers.{i}.post_attention_layernorm.weight  [hidden]
model.layers.{i}.mlp.gate_up_proj.weight          [2*ffn, hidden]             ← fused gate+up, [out, in]
model.layers.{i}.mlp.down_proj.weight             [hidden, ffn]               ← [out, in]
model.norm.weight                                 [hidden]
lm_head.weight                                    [vocab, hidden]
```

Split logic when loading:
```python
# QKV split (along out dim=0):
q, k, v = qkv_proj.split([hn*hd, gn*hd, gn*hd], dim=0)

# Gate+Up split (along out dim=0):
gate, up = gate_up_proj.chunk(2, dim=0)
```

- GQA (8 KV heads vs 32 Q heads in mini)
- **QK-Norm**: `model.layers.{i}.self_attn.q_layernorm.weight` / `k_layernorm.weight`
- SwiGLU, RMSNorm, RoPE
- No bias

---

### Phi-4 / Phi-4-mini

- `model_type`: `phi4` (or `phi3` for mini variants)
- Same fused QKV + fused gate+up as Phi-3, same key names.
- Extended context (128K) via LongRoPE — no extra weights, position scaling in config.
- Larger vocab: 200,064.

---

### DeepSeek-V2 / DeepSeek-V3 (MLA)

- `model_type`: `deepseek_v2`
- Uses **Multi-head Latent Attention (MLA)** — fundamentally different from standard attention.
- KV is compressed to a low-rank latent vector; standard KV projection does not exist.

```
model.embed_tokens.weight                              [vocab, hidden]
model.layers.{i}.input_layernorm.weight                [hidden]
model.layers.{i}.self_attn.q_a_proj.weight             [q_lora_rank, hidden]         ← query down-proj
model.layers.{i}.self_attn.q_a_layernorm.weight        [q_lora_rank]
model.layers.{i}.self_attn.q_b_proj.weight             [hn*(nope_hd+rope_hd), q_lora_rank]  ← query up-proj
model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight   [kv_lora_rank + rope_hd, hidden]     ← KV down-proj + RoPE key
model.layers.{i}.self_attn.kv_a_layernorm.weight       [kv_lora_rank]
model.layers.{i}.self_attn.kv_b_proj.weight            [hn*(nope_hd + v_hd), kv_lora_rank]  ← KV up-proj
model.layers.{i}.self_attn.o_proj.weight               [hidden, hn*v_hd]             ← [out, in]
model.layers.{i}.post_attention_layernorm.weight       [hidden]
model.layers.{i}.mlp.gate_proj.weight                  [ffn, hidden]                 ← [out, in]  (dense layers)
model.layers.{i}.mlp.up_proj.weight                    [ffn, hidden]                 ← [out, in]
model.layers.{i}.mlp.down_proj.weight                  [hidden, ffn]                 ← [out, in]
model.norm.weight                                      [hidden]
lm_head.weight                                         [vocab, hidden]
```

MoE layers (DeepSeek-V2/V3 MoE):
```
model.layers.{i}.mlp.experts.{j}.gate_proj.weight
model.layers.{i}.mlp.experts.{j}.up_proj.weight
model.layers.{i}.mlp.experts.{j}.down_proj.weight
model.layers.{i}.mlp.gate.weight                       ← router
model.layers.{i}.mlp.shared_experts.gate_proj.weight  ← shared expert
```

Key MLA parameters (from config):
- `q_lora_rank`: 1536 (query compression dim)
- `kv_lora_rank`: 512 (KV compression dim)
- `qk_nope_head_dim`: 128 (non-RoPE head dim)
- `qk_rope_head_dim`: 64 (RoPE head dim)
- `v_head_dim`: 128

All weight shapes: `[out, in]` convention.

---

### DeepSeek-R1 Distill Variants

These are fine-tuned base models; architecture and weight naming are 100% inherited from the base:

| Variant | Base model | Weight naming |
|---------|-----------|---------------|
| R1-Distill-Qwen-1.5B/7B/14B/32B | Qwen2.5 | Qwen2 naming |
| R1-Distill-Llama-8B | LLaMA-3.1 | LLaMA naming |
| R1-Distill-Llama-70B | LLaMA-3.3 | LLaMA naming |

---

## Architecture Comparison Table

| Model | QKV | Gate+Up | QK-Norm | MLA | GQA | Bias (QKV) | Tied Embed |
|-------|-----|---------|---------|-----|-----|------------|------------|
| LLaMA-2/3 | separate | separate | No | No | Yes (L3) | No | No |
| Qwen2/2.5 | separate | separate | No | No | Yes | Yes | Yes |
| Qwen3 | separate | separate | **Yes** | No | Yes | No | Yes |
| Mistral | separate | separate | No | No | Yes | No | No |
| Gemma 2 | separate | separate | No | No | Yes | No | Yes |
| Phi-3/4 | **fused** | **fused** | **Yes** | No | Yes | No | No |
| DeepSeek-V2/V3 | MLA | separate | No | **Yes** | — | No | No |
| DeepSeek-R1 distill | (base) | (base) | (base) | No | (base) | (base) | (base) |

---

## Implementation Notes for ironcore `WeightMapper`

### Models mappable with current LLaMA path (after fixing transpose bugs)
- LLaMA-2, LLaMA-3, LLaMA-3.2
- Mistral, Mixtral
- Qwen2, Qwen2.5 (add bias support)
- DeepSeek-R1 Distill (Qwen and LLaMA variants)
- Gemma 2 (needs extra layernorm keys)

### Models needing new architecture entries
- **Qwen3**: LLaMA path + q_norm/k_norm per-head layernorm weights
- **Phi-3/4**: New path — fused QKV split and fused gate+up split
- **DeepSeek-V2/V3**: New path — MLA projections (q_a, q_b, kv_a, kv_b), MoE experts

### Transpose rule
All models listed (except GPT-2) use `nn.Linear` with `[out, in]` weights.
IronCore uses `[in, out]`. **Always `.t()` on linear weights in both directions.**
Bias vectors are 1D — never transpose.

### Fusion/split rules for ironcore's fused KV (`linear_kv`)
IronCore fuses K and V into a single `[in, 2*kv_out]` weight.

| Direction | Operation |
|-----------|-----------|
| HF → IC | `cat([k.t(), v.t()], dim=1)` or equivalently `cat([k, v], dim=0).t()` |
| IC → HF | `k_t, v_t = kv.chunk(2, dim=1); k=k_t.t(); v=v_t.t()` |
