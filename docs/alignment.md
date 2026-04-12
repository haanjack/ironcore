# Alignment (DPO and GRPO)

## Comparison

| | DPO | GRPO |
|---|---|---|
| Generation at train time | No | Yes |
| Reward source | Implicit (preference pairs) | Explicit reward functions |
| Advantage computation | N/A | Group-relative normalization |
| IS ratio clipping | No | Optional (multi-epoch replay) |
| Data source | Pre-collected (chosen, rejected) pairs | Prompts; responses generated on-the-fly |
| Trainer | `DPOTrainer` | `GRPOTrainer` |

---

## DPO

### Loss

Implemented in `ironcore/alignment/loss/dpo.py` — `dpo_loss()`.

```
L = -log(sigmoid(β · (log π_chosen − log π_ref_chosen − log π_rejected + log π_ref_rejected)))
```

`compute_logps()` computes per-sequence log probabilities from logits via a TP-safe log-softmax (`_compute_log_softmax_tp_safe()`). Under tensor parallelism the vocabulary is sharded across ranks; the function all-gathers logits before applying `F.log_softmax` to ensure correct normalization.

Label smoothing (`dpo_label_smoothing > 0`) replaces the hard `1` target with `1 − ε`.

### Efficiency: concat forward passes

When `alignment.concat_forward_passes: true`, chosen and rejected sequences are concatenated into a single batch `[2B, seq_len]` and processed in one forward pass instead of two. This avoids a second TP all-gather for log-softmax.

### Reference model

`DPOTrainer._post_checkpoint_load()` creates a frozen reference model. Under FSDP it uses a `state_dict` copy (not `deepcopy`) to avoid entangling internal FSDP metadata.

### DPO config fields

| Field | Default | Description |
|---|---|---|
| `alignment.dpo_beta` | `0.5` | β — preference strength |
| `alignment.dpo_label_smoothing` | `0.0` | Label smoothing ε |
| `alignment.concat_forward_passes` | `true` | Single forward pass for chosen+rejected |
| `alignment.metrics_interval` | `0` | Log extra metrics every N steps (0 = every step) |

---

## GRPO

### Rollout generation

`generate_rollouts_batched()` in `ironcore/alignment/rollout.py`:

1. **Prefill** prompts `[B, prompt_len]` → `prefill_logits`, `prefix_kv`
2. **Expand KV cache** `[B, …] → [B×G, …]` via `_expand_kv_cache()`, which calls `tensor.repeat_interleave(group_size, dim=0)` on each layer's key and value tensors. This avoids re-computing the prompt prefix for each of the G completions.
3. **Sample first tokens** from `prefill_logits[:, -1, :]`, expanded to `[B×G, vocab]`
4. **Autoregressive decode** all `B×G` sequences in parallel until `max_new_tokens` or all hit EOS

Under TP, stochastic sampling broadcasts the sampled token from rank 0 to all TP ranks so they remain in sync.

#### Chunked generation

`rollout_chunks = grpo_group_size / grpo_rollout_micro_group_size`. The `GRPOTrainer` calls `generate_rollouts_batched` once per chunk, then aggregates into a `RolloutBuffer`.

### Advantages

`compute_advantages()` in `ironcore/alignment/loss/grpo.py`:

```
A_i = (R_i − mean(R_group)) / (std(R_group) + ε)
```

In distributed settings, rewards are all-gathered from all DP ranks before normalization so that all completions for a given prompt are normalized together regardless of how they are distributed across GPUs. When all rewards in a group are equal (std < ε), advantages are set to 0.

### KL divergence

`kl_divergence_approx()` in `ironcore/alignment/loss/kl.py` uses the Schulman estimator:

```
KL ≈ exp(log_π_ref − log_π) − (log_π_ref − log_π) − 1
```

This avoids materializing the full-vocabulary softmax — it only requires per-token log probs of the response tokens already selected during rollout. The approximation is non-negative and unbiased.

### Loss

`grpo_loss()` in `ironcore/alignment/loss/grpo.py`:

**Online** (`grpo_num_epochs == 1`, `old_log_probs` not set):
```
L = −mean(A · log π_θ(y|x)) + β · KL
```

**Offline / multi-epoch** (`grpo_num_epochs > 1`, IS ratio clipping):
```
ratio = exp(log π_θ − log π_old)
L = −mean(min(ratio · A, clip(ratio, 1±ε) · A)) + β · KL
```

### Reward orchestration

`RewardManager` collects rewards from one or more `RewardFunction` implementations (rule-based or model-based) and averages them with configurable weights. See `docs/reward_manager.md` for full details.

### GRPO config fields

| Field | Default | Description |
|---|---|---|
| `alignment.grpo_group_size` | `4` | Total completions per prompt (G) |
| `alignment.grpo_rollout_micro_group_size` | `1` | Per-GPU parallel completions per prompt |
| `alignment.grpo_beta` | `0.1` | KL penalty coefficient |
| `alignment.grpo_eps` | `1e-8` | Advantage normalization ε |
| `alignment.grpo_num_epochs` | `1` | Gradient epochs per rollout (>1 = offline replay) |
| `alignment.grpo_clip_eps` | `0.2` | IS ratio clip range ε (0 = no clipping) |
| `alignment.generation.max_new_tokens` | `512` | Max tokens to generate |
| `alignment.generation.temperature` | `1.0` | Sampling temperature |
| `alignment.generation.top_p` | `0.9` | Nucleus sampling threshold |
| `alignment.generation.top_k` | `0` | Top-k cutoff (0 = disabled) |
| `alignment.generation.do_sample` | `true` | Stochastic sampling (false = greedy) |
| `alignment.reward_manager.*` | — | Reward function list and worker pool config |
