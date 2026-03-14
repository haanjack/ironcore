# GRPO / GSM8K — Experiment Report (Runs 1–4)
**Date:** 2026-03-13 | **Model:** Qwen2.5-0.5B-Instruct | **Hardware:** 2× RTX 3090

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

### Next Steps

Run 5 with the above changes. Expected: `mean_reward` should show non-trivial signal within first 50 steps due to composite reward breaking the zero-advantage problem.
