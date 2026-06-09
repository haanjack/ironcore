# Alignment (DPO and GRPO)

> This guide covers how to configure and run alignment training. For loss formulas,
> advantage computation, KL estimator, and rollout internals, see the
> [Alignment system design](design/alignment.md).

## DPO vs GRPO

| | DPO | GRPO |
|---|---|---|
| Generation at train time | No | Yes |
| Data requirement | Pre-collected (chosen, rejected) pairs | Prompts only |
| Reward source | Implicit (preference pairs) | Explicit reward functions |
| Memory overhead | Low — no rollout buffers | Higher — rollout buffer + reference model |
| Trainer | `DPOTrainer` | `GRPOTrainer` |

---

## DPO

Set `data.task_type: dpo` and provide a preference dataset with `chosen` and `rejected` columns.

### Key config decisions

`concat_forward_passes: true` (default) runs chosen and rejected sequences in a single forward pass, saving one TP all-gather. Leave it on unless you're debugging.

`dpo_label_smoothing` adds ε smoothing to the preference target. The standard DPO default is `0.0`; values around 0.1–0.2 can help when preference data is noisy.

`dpo_beta` controls how strongly the loss penalizes deviation from the reference policy. Typical range: 0.1–0.5.

### Minimal DPO config

```yaml
data:
  task_type: dpo
  datasets:
    - source: my-org/preference-dataset
      chosen_column: chosen
      rejected_column: rejected

alignment:
  dpo_beta: 0.5
  concat_forward_passes: true
```

### DPO config reference

| Field | Default | Description |
|---|---|---|
| `alignment.dpo_beta` | `0.5` | β — preference strength |
| `alignment.dpo_label_smoothing` | `0.0` | Label smoothing ε |
| `alignment.concat_forward_passes` | `true` | Single forward pass for chosen+rejected |
| `alignment.metrics_interval` | `0` | Log extra metrics every N steps (0 = every step) |

---

## GRPO

Set `data.task_type: grpo` and provide a prompt dataset. The trainer generates completions, scores them with reward functions, and trains the policy.

### Rollout configuration

`grpo_group_size` (G) is the total number of completions sampled per prompt. Advantages are computed group-relative — all G completions for a prompt must be visible before normalization. With DP > 1, rewards are all-gathered across ranks before normalization.

`grpo_rollout_micro_group_size` controls how many completions are generated in parallel per step. Set lower to reduce peak memory during generation.

```yaml
alignment:
  grpo_group_size: 8                 # G completions per prompt
  grpo_rollout_micro_group_size: 2   # generate 2 at a time
```

### Multi-epoch replay (offline GRPO)

Setting `grpo_num_epochs > 1` enables importance-sampled multi-epoch replay: the same rollout batch is used for multiple gradient update steps with IS ratio clipping. This improves sample efficiency at the cost of slightly stale advantage estimates.

```yaml
alignment:
  grpo_num_epochs: 4      # reuse each rollout for 4 gradient steps
  grpo_clip_eps: 0.2      # PPO-style IS ratio clipping range
```

### Generation parameters

```yaml
alignment:
  generation:
    max_new_tokens: 512
    temperature: 1.0
    top_p: 0.9
    top_k: 0          # 0 = disabled
    do_sample: true   # false = greedy
```

### Reference model CPU offload

Set `alignment.offload_ref_model: true` to keep the reference model on CPU between forward passes. Moves it to GPU only when computing reference log-probabilities. Useful for single-GPU GRPO where policy + reference model together exceed VRAM.

**Limitation:** incompatible with FSDP. If FSDP is active, the flag is ignored with a warning.

### Minimal GRPO config

```yaml
data:
  task_type: grpo
  datasets:
    - source: my-org/prompt-dataset
      prompt_column: prompt

alignment:
  grpo_group_size: 8
  grpo_beta: 0.1
  grpo_num_epochs: 1
  generation:
    max_new_tokens: 512
    temperature: 1.0
  reward_manager:
    num_workers: 4
    functions:
      - name: correctness
        type: rule_template
        weight: 1.0
        rule_template: configs/rewards/math_gsm8k.yaml
```

### GRPO config reference

| Field | Default | Description |
|---|---|---|
| `alignment.grpo_group_size` | `4` | Total completions per prompt (G) |
| `alignment.grpo_rollout_micro_group_size` | `1` | Parallel completions generated per chunk |
| `alignment.grpo_beta` | `0.1` | KL penalty coefficient |
| `alignment.grpo_eps` | `1e-8` | Advantage normalization ε |
| `alignment.grpo_num_epochs` | `1` | Gradient epochs per rollout (>1 = offline replay) |
| `alignment.grpo_clip_eps` | `0.2` | IS ratio clip range ε (0 = no clipping) |
| `alignment.offload_ref_model` | `false` | Keep reference model on CPU between passes |
| `alignment.generation.max_new_tokens` | `512` | Max tokens to generate |
| `alignment.generation.temperature` | `1.0` | Sampling temperature |
| `alignment.generation.top_p` | `0.9` | Nucleus sampling threshold |
| `alignment.generation.top_k` | `0` | Top-k cutoff (0 = disabled) |
| `alignment.generation.do_sample` | `true` | Stochastic sampling (false = greedy) |
| `alignment.reward_manager.*` | — | Reward function list and worker pool — see [reward_manager.md](reward_manager.md) |
