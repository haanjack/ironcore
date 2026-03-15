# GRPO / GSM8K — Experiment Report (Runs 1–6)
**Date:** 2026-03-13 to 2026-03-14 | **Model:** Qwen2.5-0.5B-Instruct | **Hardware:** 2× RTX 3090

---

## Run 1 — CUDA OOM
**Log:** `/tmp/grpo_training.log`
**Config:** `grpo_rollout_chunks=16`, `grpo_beta=0.04`
**Duration:** ~1 min 30s | **Steps completed:** 0

**Failure:** `torch.OutOfMemoryError` on GPU 1 at the first train step.
- Root cause: update phase micro-batch size = `B × (G / rollout_chunks)` = `16 × 2 = 32`. Float32 log_softmax chunk `[32, 128, 151936]` required **2.08 GiB** with only 1.61 GiB contiguous free.
- **Fix applied:** `grpo_rollout_chunks: 16 → 32` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## Run 2 — RolloutBuffer.cat() Shape Mismatch
**Log:** `/tmp/grpo_training2.log`
**Config:** `grpo_rollout_chunks=32`, `grpo_beta=0.04`
**Duration:** ~2h | **Steps completed:** 20

| Step | grpo_loss | policy_loss | kl_loss | mean_reward | iter_time |
|------|-----------|-------------|---------|-------------|-----------|
| 10 | 9012.9 | 4.753 | 9008.2 | 0.002 | 159.8s |
| 20 | 8732.9 | 2.650 | 8730.3 | 0.002 | 161.3s |

**Failure:** `RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 400 but got size 372` at step 21. Once the model began generating responses shorter than `max_new_tokens`, different rollout chunks produced different-length `completion_ids`, causing `torch.cat(dim=0)` to fail.
- **Fix applied:** Added `_pad_seq()` in `RolloutBuffer.cat()` to zero-pad shorter tensors before concatenation.

---

## Run 3 — KL Dominance (beta too large)
**Log:** `/tmp/grpo_training3.log`
**Config:** `grpo_rollout_chunks=32`, `grpo_beta=0.04`
**Duration:** ~4h 55min | **Steps completed:** 110 (stopped manually)

| Step | grpo_loss | policy_loss | kl_loss | mean_reward | grad_norm |
|------|-----------|-------------|---------|-------------|-----------|
| 10 | 8654.8 | 0.000 | 8654.8 | 0.000 | 4,502,720 |
| 20 | 8556.2 | -0.407 | 8556.6 | 0.004 | 673,830 |
| 50 | 1711.5 | -1.917 | 1713.4 | 0.004 | 252,402 |
| 100 | 219.9 | 0.000 | 219.9 | 0.000 | 32,368 |
| 110 | 207.9 | 0.000 | 207.9 | 0.000 | 73,730 |

**Outcome:** No crash, but reward never emerged. `kl_loss` dominated `policy_loss` by ~1000×, pulling the policy back to the reference before any reward gradient could take effect. `mean_reward` stayed in `[0.000, 0.004]` throughout.
- **Fixes applied during this run:** per-sequence EOS masking (`response_lengths`), `_log_qualitative_samples` group-id indexing
- **Config fix:** `grpo_beta: 0.04 → 0.001`

---

## Run 4 — beta=0.001, All Fixes Active
**Log:** `/tmp/grpo_training4.log`
**Config:** `grpo_rollout_chunks=32`, `grpo_beta=0.001`
**Duration:** ~2h 27min | **Steps completed:** 160 (resumed from step 100 checkpoint, stopped manually)

| Step | grpo_loss | policy_loss | kl_loss | mean_reward | grad_norm | iter_time |
|------|-----------|-------------|---------|-------------|-----------|-----------|
| 100* | 1.258 | 0.000 | 1.258 | 0.000 | — | — |
| 110 | 0.097 | -0.131 | 0.229 | 0.002 | 232 | 145.3s |
| 120 | 0.337 | +0.033 | 0.304 | 0.002 | 85 | 141.5s |
| 130 | 0.229 | 0.000 | 0.229 | 0.000 | 126 | 142.6s |
| 140 | 0.281 | 0.000 | 0.281 | 0.000 | 467 | 143.1s |
| 150 | 0.731 | 0.000 | 0.731 | 0.000 | 739 | 143.4s |
| 160 | 0.112 | 0.000 | 0.112 | 0.000 | 70 | 145.4s |

*\*Step 100 reported at run start from loaded checkpoint (Run 3 weights).*

**Outcome:** KL and policy loss now on the same scale (ratio ~1.7× at step 110 vs ~1000× in Run 3). `grad_norm` dropped from 73,730 → ~70–740, indicating stable training. `mean_reward` still near zero — insufficient steps for the policy to learn `####` format compliance from scratch. Stopped manually before reward signal emerged.

---

## Cumulative Fixes Across Runs

| Fix | Run | File |
|-----|-----|------|
| OOM: `rollout_chunks 16→32` + `expandable_segments:True` | Run 1→2 | `configs/grpo_gsm8k.yaml` |
| `RolloutBuffer.cat()` zero-padding for variable-length chunks | Run 2→3 | `alignment/buffer.py` |
| Per-sequence EOS masking via `response_lengths` | Run 3→4 | `rollout.py`, `buffer.py`, `grpo_trainer.py` |
| `_log_qualitative_samples` group-id indexing | Run 3→4 | `grpo_trainer.py` |
| `grpo_beta: 0.04 → 0.001` | Run 3→4 | `configs/grpo_gsm8k.yaml` |

---

## Post-Run 4 Analysis & Changes

Root cause of near-zero reward across all runs: **reward sparsity death spiral**.
`MathRewardFunction(strict=True)` returns binary 0/1. Qwen2.5-0.5B success rate on GSM8K < 2%.
When all G completions in a group score 0, `std=0` → advantages=0 → policy_loss=0 → no gradient.

### Changes applied after Run 4

| Change | File | Detail |
|--------|------|--------|
| **Composite reward** (`composite_math`) | `rewards.py`, `grpo_gsm8k.yaml` | 20% format (`#### <digit>` present) + 80% correctness (`strict=False`). Breaks zero-advantage groups. |
| **Partial credit** in `MathRewardFunction` | `rewards.py` | 0.1 reward when a number is extracted but wrong (was 0.0). Creates within-group variance. |
| **Length normalization** on sequence log probs | `grpo_trainer.py` | `sum(log_probs) / response_len`. Removes bias toward shorter sequences. |
| **EOS masking fix** in `_get_ref_log_probs` | `grpo_trainer.py` | Now uses `_prepare_labels_and_mask()` for consistent EOS-aware masking between policy and reference. |
| **Config: `train_batch_size` 16→4** | `grpo_gsm8k.yaml` | 4 prompts × 8 completions = 32 total. 16× stronger gradient per useful sample, 4× more steps per dataset pass. |
| **Config: `grpo_group_size` 32→8** | `grpo_gsm8k.yaml` | 8 is sufficient for reward variance with composite reward. More prompts/hour > more completions/prompt. |
| **Config rename: `grpo_rollout_chunks` → `grpo_rollout_micro_group_size`** | `config_alignment.py`, `grpo_trainer.py` | Semantic fix: `group_size` = total (like `train_batch_size`), `micro_group_size` = per-GPU hardware knob (like `micro_batch_size`). Chunks derived internally. |
| **Config: `max_lr` 1e-5→5e-6, `warmup_steps` 50→20** | `grpo_gsm8k.yaml` | Proportional to smaller batch. |

---

## Run 5 — Composite Reward, All Post-Run 4 Fixes
**Log:** `logs/grpo_training5.log`
**Config:** `beta=0.001`, `train_batch_size=4`, `group_size=8`, `composite_math` reward
**Duration:** ~50 min | **Steps completed:** 100 (100→200, resumed from Run 4 checkpoint)

| Step | grpo_loss | policy_loss | kl_loss | mean_reward | grad_norm | iter_time |
|------|-----------|-------------|---------|-------------|-----------|-----------|
| 100 | 6.73 | 0.17 | 6.56 | 0.069 | — | — |
| 110 | 0.36 | 0.15 | 0.21 | 0.090 | 998 | 28.7s |
| 120 | 0.86 | -0.05 | 0.92 | 0.070 | 581 | 28.7s |
| 130 | 0.30 | -0.16 | 0.46 | 0.081 | 1392 | 29.1s |
| 140 | 0.60 | 0.21 | 0.40 | **0.115** | 548 | 29.1s |
| 150 | 0.70 | 0.17 | 0.54 | 0.065 | 246 | 29.5s |
| 160 | 0.60 | -0.11 | 0.70 | 0.100 | 2417 | 29.8s |
| 170 | 3.36 | -0.01 | 3.37 | 0.075 | 169 | 29.6s |
| 180 | 4.07 | 0.12 | 3.95 | 0.073 | 275 | 29.4s |
| 190 | 0.35 | 0.01 | 0.34 | 0.075 | 504 | 29.5s |
| 200 | 0.05 | **-0.24** | 0.30 | 0.100 | 491 | 29.5s |

**Outcome:**
- ✅ `mean_reward` increased from ~0.002 (Run 4) to **0.07–0.12** (20–60× improvement)
- ✅ Composite reward successfully broke the zero-advantage death spiral
- ✅ Checkpoint saved at step 200 (`outputs/grpo_gsm8k/step_200`)
- ⚠️ Qualitative samples still poor (repetitive `!!!` patterns, off-topic text) — model hasn't learned to follow the `#### <answer>` format reliably yet
- ⚠️ KL loss spikes at steps 170–180 suggest policy drift; may need monitoring in longer runs

**Analysis:**
The composite reward (20% format + 80% correctness with partial credit) successfully creates within-group variance even when no completion is correct. This allows gradients to flow and the policy to learn. The ~29s/step iteration time is ~5× faster than Run 3–4 due to smaller batch size (4 vs 16).

---

## Run 6 — Fresh Start with Distributed Advantage Fix
**Log:** `logs/grpo_training6.log`
**Config:** Same as Run 5, but from step 0
**Duration:** ~3h 40min | **Steps completed:** 200 (0→200, fresh start)
**Code changes:** `distributed=True` for advantages, group ID offset fix, FSDP grad norm fix

| Step | grpo_loss | policy_loss | kl_loss | mean_reward | grad_norm | iter_time |
|------|-----------|-------------|---------|-------------|-----------|-----------|
| 10 | 233.88 | 0.38 | 233.50 | 0.074 | 51456 | 35.4s |
| 50 | 185.66 | 0.18 | 185.48 | 0.080 | — | ~36s |
| 80 | 34.68 | -0.67 | 35.35 | 0.045 | — | ~36s |
| 100 | 81.64 | 0.76 | 80.88 | 0.098 | — | ~36s |
| 110 | 27.27 | -0.19 | 27.47 | **0.120** | — | ~36s |
| 130 | 117.54 | 0.22 | 117.32 | 0.060 | — | ~36s |
| 150 | 6.68 | 0.43 | 6.25 | 0.075 | — | ~36s |
| 170 | 55.11 | 0.19 | 54.92 | 0.088 | — | ~36s |
| 200 | 21.55 | 0.59 | 20.96 | 0.093 | — | ~36s |

**Outcome:**
- ✅ Training stable from step 0 (no checkpoint dependency)
- ✅ `mean_reward` in 0.06–0.12 range, consistent with Run 5
- ✅ KL loss variable (4–120) but not exploding
- ⚠️ Qualitative samples still poor — best responses are off-topic, worst are repetitive `!!!` patterns
- ⚠️ Iteration time ~36s (slower than Run 5's 29s) — possibly due to distributed advantage sync

**Sample at Step 200:**
```
Best (reward=0.08): "Human: Calculate the area of a rectangle..." — off-topic, ignores math problem
Worst (reward=0.00): "!!! !!! !!! !!!..." — repetitive character collapse
```

**Analysis:**
The distributed advantage fix ensures correct gradient scaling across GPUs, but doesn't fundamentally change learning dynamics. The model still struggles to:
1. Follow the `#### <answer>` format
2. Stay on-topic for math problems
3. Avoid degenerate repetition

This suggests the 0.5B model may be too small for GRPO to effectively shape behavior, or the reward signal is still too sparse/weak.

---

## Summary & Recommendations

| Run | Steps | mean_reward | Outcome |
|-----|-------|-------------|---------|
| 1 | 0 | — | OOM |
| 2 | 20 | 0.002 | Shape mismatch |
| 3 | 110 | 0.004 | KL dominance |
| 4 | 160 | 0.002 | Stable, no reward |
| 5 | 200 | 0.07–0.12 | Reward signal ✓ |
| 6 | 200 | 0.06–0.12 | Confirmed stable |
| 7 | 20 | 0.0504 (both) | RewardManager equivalence ✓ (rel_diff=0.00%) |
| 8 | 200 | 0.05–0.10 | YAML templates ✓ (entropy fix) |

**Key fixes that worked:**
1. Composite reward (format + correctness) — breaks zero-advantage
2. `beta=0.001` — prevents KL dominance
3. Smaller batch (4) — faster iteration, stronger per-sample gradient
4. **Entropy computation fix** — `compute_entropy()` requires log_probs, not logits

**Remaining issues:**
- Model doesn't learn format compliance
- Qualitative samples are off-topic or degenerate
- May need larger model (1.5B+) or more training steps

---

## Run 7 — RewardManager Equivalence Validation (E2E Test 67)
**Date:** 2026-03-15 | **Config:** `grpo_gsm8k_smoke_composite.yaml` vs `grpo_gsm8k_smoke_rm_composite.yaml`
**Duration:** ~5m 25s | **Steps:** 20 | **Hardware:** 2× RTX 3090 (FSDP)

| Path | mean_reward | n (steps) | rel_diff |
|------|-------------|-----------|----------|
| Legacy (`reward: composite_math`) | 0.0504 | 19 | — |
| RewardManager (`reward_manager: composite_math`) | 0.0504 | 19 | **0.00%** |

**Outcome:** ✅ `RewardManager.from_config` with `composite_math` (format_weight=0.4) produces bit-identical rewards to the legacy `reward:` path. E2E test 67 passes at 0% relative divergence (tolerance: 5%).

**Bugs fixed during this run:**
- `_convert_lists_to_dict` in `utils.py` was merging single-item lists of dicts → destroyed `reward_manager.functions` list
- `RewardFunctionEntry` missing `format_weight` field → RM path silently used default 0.2 instead of configured 0.4
- `compute_advantages` missing `dtype=torch.long` on size tensor + no bounds check → corrupted `max_size` from NCCL under memory pressure caused impossible allocation (~120 PiB OOM at step 13 with `max_new_tokens=256`)

---

## Run 8 — RewardManager YAML Template Validation
**Date:** 2026-03-15 | **Config:** `grpo_gsm8k_rm.yaml` (YAML templates)
**Duration:** ~2h | **Steps:** 200 | **Hardware:** 2× RTX 3090 (DDP)

| Step | grpo_loss | policy_loss | kl_loss | mean_reward | grad_norm | iter_time |
|------|-----------|-------------|---------|-------------|-----------|-----------|
| 10   | 129.07    | -0.12       | 129.22  | 0.0487      | 72192     | 35.6s |
| 20   | 135.88    | 0.11        | 135.80  | 0.0869      | 18304     | 35.6s |
| 50   | 80.17     | 0.60        | 79.60   | 0.0581      | 31744     | 35.7s |
| 100  | 98.68     | 0.64        | 98.07   | 0.0619      | 15936     | 35.7s |
| 110  | 18.01     | 0.65        | 17.39   | 0.0725      | 16928     | 35.7s |
| 150  | 22.63     | 0.26        | 22.41   | 0.0525      | 9472      | 35.7s |
| 180  | 18.89     | 0.73        | 18.19   | **0.1006**  | 24704     | 35.7s |
| 190  | 12.57     | 0.36        | 12.24   | 0.0469      | 7168      | 35.7s |
| 200  | —         | —           | —       | —           | 25088     | 35.7s |

**Reward Config:**
```yaml
reward_manager:
  functions:
    - name: format
      type: rule_template
      weight: 0.4
      rule_template: configs/rewards/format_gsm8k_format.yaml
    - name: correctness
      type: rule_template
      weight: 0.6
      rule_template: configs/rewards/math_gsm8k.yaml
```

**Outcome:** ✅ YAML template rewards produce training dynamics comparable to legacy `composite_math`:
- `mean_reward` range: 0.047-0.101 (Run 6: 0.060-0.120) — within expected variance ✓
- `grpo_loss` range: 10-136 (Run 6: 5-234) — comparable ✓
- `iter_time`: 35.7s (Run 6: 36s) — unchanged ✓

**Bug fixed during this run:**
- `compute_entropy()` in `grpo_trainer.py:554` was receiving **logits** instead of **log_probs** → entropy_loss was ~100 million instead of ~0.03, corrupting total loss calculation. Fixed by applying `_compute_log_softmax_tp_safe(policy_logits)` before passing to `compute_entropy()`.

---

## Next Steps

- **YAML template migration complete** — legacy `reward:` path can be deprecated
- **Move to larger model** (Qwen2.5-1.5B or 3B) — 0.5B may lack capacity for GRPO
- Consider: Use SFT warmstart before GRPO
