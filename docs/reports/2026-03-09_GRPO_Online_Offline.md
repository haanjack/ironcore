# Experiment Report: GRPO Online/Offline Training Mode

**Date:** 2026-03-09
**Commit Hash:** e38a5a7 (feature/grpo)

---

## 1. Objective

Wire up offline (multi-epoch) training mode for GRPO alongside the existing online mode. The goal is to allow the same rollout batch to be reused for multiple gradient updates using importance sampling (IS) correction, optionally with PPO-style ratio clipping — controlled by a single YAML knob (`grpo_num_epochs`).

---

## 2. Design

### Online vs Offline Distinction

| Mode | `grpo_num_epochs` | IS ratio | Clip | Rollout frequency |
|:---|:---:|:---:|:---:|:---|
| Online | 1 | no (ratio = 1) | n/a | every step |
| Offline, no clip | > 1 | yes | no (`clip_eps=0`) | once per step |
| Offline, PPO-clipped | > 1 | yes | yes | once per step |

Both modes generate fresh rollouts from the **current** policy at the start of every `train_step`. The distinction is how many gradient updates are applied to that batch before discarding it.

### Training Step Structure

```
train_step():
  ┌─ Phase 1: Rollout (always runs once) ──────────────────────────┐
  │  generate_rollouts_batched()  →  old_log_probs (frozen)        │
  │  reward_worker.score_batch()  →  rewards                       │
  │  compute_advantages()         →  advantages (detached)         │
  └────────────────────────────────────────────────────────────────┘
  ┌─ Phase 2: Update (× grpo_num_epochs) ──────────────────────────┐
  │  epoch 0, num_epochs=1:  online path, IS skipped               │
  │  epoch 0, num_epochs>1:  IS ratio ≈ 1.0 (policy unchanged)    │
  │  epoch 1+:               IS ratio = π_θ / π_old  (drifted)    │
  │    with clip_eps > 0:    ratio clamped to [1−ε, 1+ε]          │
  └────────────────────────────────────────────────────────────────┘
```

### Loss Formula

**Online** (`old_log_probs=None`):

```
L = −mean(A · log π_θ(y|x)) + β · KL(π_ref ‖ π_θ)
```

**Offline** (`old_log_probs` provided, no clipping):

```
ratio = exp(log π_θ − log π_old)
L = −mean(ratio · A) + β · KL
```

**Offline with PPO clip** (`clip_eps > 0`):

```
ratio  = exp(log π_θ − log π_old)
ratio' = clamp(ratio, 1−ε, 1+ε)
L = −mean(min(ratio · A, ratio' · A)) + β · KL
```

The `min` (pessimistic surrogate) prevents the policy from exploiting large positive advantages through unconstrained ratio growth.

---

## 3. Files Changed

### `ironcore/config/config_alignment.py`

Added two new fields to `AlignmentConfig`:

```python
grpo_num_epochs: int = 1    # gradient steps per rollout batch
grpo_clip_eps: float = 0.2  # PPO clip range (0 = disabled)
```

Validation:
- `grpo_num_epochs >= 1`
- `grpo_clip_eps >= 0`

### `ironcore/alignment/loss/grpo.py`

Extended `grpo_loss()` signature:

```python
def grpo_loss(
    policy_log_probs,
    ref_log_probs,
    advantages,
    kl_per_seq,
    beta,
    old_log_probs=None,   # NEW: frozen log probs from rollout
    clip_eps=0.0,         # NEW: PPO clip range
) -> (loss, metrics)
```

New metrics returned:
- `mean_ratio` — mean IS ratio across the batch (1.0 when online)
- `clip_fraction` — fraction of samples where ratio was clipped

### `ironcore/trainers/grpo_trainer.py`

- Added `self.num_epochs` and `self.clip_eps` from config in `__init__`
- Split `train_step` into Phase 1 (rollout, once) and Phase 2 (update loop)
- `_compute_grpo_loss` accepts and forwards `old_log_probs` and `clip_eps`
- Logging includes `mean_ratio` and `clip_fraction`

---

## 4. Configuration Reference

```yaml
alignment:
  method: grpo
  grpo_group_size: 4
  grpo_beta: 0.1
  grpo_eps: 1.0e-8

  # --- Online (default) ---
  grpo_num_epochs: 1
  grpo_clip_eps: 0.2      # has no effect when num_epochs=1

  # --- Offline, PPO-clipped (multi-epoch) ---
  # grpo_num_epochs: 4
  # grpo_clip_eps: 0.2

  # --- Offline, pure IS (no clipping) ---
  # grpo_num_epochs: 4
  # grpo_clip_eps: 0.0
```

---

## 5. Key Properties

**`old_log_probs` source:** Populated by `generate_rollouts_batched()` as the sum of per-token log probs during generation (already implemented before this change). No additional forward pass needed.

**Epoch 0 with `num_epochs > 1`:** The policy has not yet been updated, so `ratio ≈ 1.0` for all samples. The clip has no effect. This epoch is mathematically equivalent to the online update, but IS infrastructure is active to allow monitoring.

**Gradient isolation:** `old_log_probs` is `.detach()`-ed before the loop. Advantages are always detached inside `grpo_loss`. Gradients flow only through `policy_log_probs` (current policy forward pass).

**KL penalty:** Applied every epoch using the fixed reference model (`π_ref`), not `π_old`. This is intentional — the reference stays frozen at the pre-training checkpoint, not at the rollout snapshot.

---

## 6. Stability Notes

| Setting | Risk | Mitigation |
|:---|:---|:---|
| `num_epochs` too high | Policy drifts far from `π_old`, IS becomes inaccurate | Keep `num_epochs` ≤ 4; watch `mean_ratio` and `clip_fraction` |
| `clip_eps=0` (no clipping) | Unbounded ratio can spike loss | Use `clip_eps=0.2` unless debugging |
| `beta=0` + `clip_eps=0` | No constraint on policy drift | At least one of KL or clip must be active |
| `clip_fraction` > 0.3 | Too many samples being clipped; policy diverging too fast | Reduce `num_epochs` or lower learning rate |

Recommended monitoring during offline runs: `mean_ratio` should stay in `[0.8, 1.2]` and `clip_fraction` below `0.2`.

---

## 7. Next Steps

- Validate offline mode with toy test (keyword reward, Qwen2.5-0.5B)
- Compare reward curves: online (`num_epochs=1`) vs offline (`num_epochs=4`) for same compute budget
- Move to RLVR on GSM8K with math verifiable reward
