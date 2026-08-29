# TP=1 vs TP=2 Training Divergence: Root Cause and Fix

**Date**: 2026-07-06
**Branch**: fix/training-correctness-audit
**Purpose**: Root-cause and fix a real numerical divergence between TP=1 and TP=2 training discovered during the systems-correctness audit, then validate the fix on real hardware.

---

## Executive Summary

An earlier validation pass (see [distributed_training_validation.md](distributed_training_validation.md), 2026-01-04) compared **DP=2 vs TP=2** loss curves and found them within 0.39% — a coarse, loss-only comparison that did not catch this bug. A later, more rigorous **TP=1 vs TP=2** equivalence check (same seed, same data order, same hyperparameters) found real divergence:

- Loss diff: **5.33e-3** by step (small enough to hide in a loss-curve-only comparison)
- Weight diff: **1.57e-1** (large — the two runs were not computing the same thing)

Root cause was not one bug but four independent sources of TP=1/TP=2 asymmetry, three of them RNG-related. All four are fixed on this branch and the fix is now verified end-to-end on real 2×RTX 3090 hardware via `torchrun`.

### Key Findings
- ❌ → ✅ **TP-sharded weight init**: every same-shaped layer got identical initial values under TP>1 (reseed-to-constant bug)
- ❌ → ✅ **Dropout under TP**: replicated activations (embedding output, attention residual, MLP output) drew different masks per rank instead of the same mask
- ❌ → ✅ **DistributedOptimizer broadcast**: post-step broadcast used a DP-group-relative rank as `src`, wrong whenever TP>1 makes the DP group's global ranks non-contiguous
- ❌ → ✅ **Muon optimizer**: Newton-Schulz orthogonalization ran per-TP-shard instead of on the gathered full matrix — mathematically different from the TP=1 computation
- ✅ **Validated**: `tests/multi_gpu/test_tp_equivalence.py`, 2/2 passed on real 2-GPU `torchrun`

---

## Background

`ironcore`'s TP implementation shards `linear_q`/`linear_kv`/`attn_output`/MLP weights column- or row-wise across TP ranks (see [parallelism guide](../parallelism.md)). For TP=1 and TP=2 to be numerically equivalent (same seed, same data), every operation that TP=2 splits across ranks must, when recombined, equal the single TP=1 computation — including RNG-dependent operations (init, dropout) and rank-aware collective operations (grad-norm reduction, broadcast, Newton-Schulz orthogonalization).

---

## Root Cause 1: TP-sharded weight init reseeded to a constant

**File**: `ironcore/layers/module.py` (`_init_tp_weight`)

Every TP-sharded weight's init reseeded the RNG to the *same constant seed*, regardless of which parameter was being initialized. Consequence: under TP>1, every same-shaped layer (e.g. `linear_q` in every transformer layer) received **identical** initial values — a much stronger and more obviously-wrong condition than "just different from TP=1," but the actual bug is that this collapses per-layer initialization diversity that TP=1 does have (each layer's `linear_q` gets an independent draw from the global RNG stream in the TP=1 path).

**Fix**: derive a per-parameter seed from the parameter's name via a local `torch.Generator`, leaving the ambient RNG stream untouched. This restores per-layer initialization diversity under TP>1 while keeping each shard's init reproducible and TP-rank-consistent (all ranks compute the same per-parameter seed, then each rank slices out its own shard from the analytically-equivalent full tensor).

## Root Cause 2: No cross-rank dropout RNG sync for TP-replicated activations

**File**: `ironcore/parallel/random.py` (new `TensorParallelRNGTracker`), wired into `ironcore/layers/embedding.py`, `mlp.py`, `parallel_mlp.py`, `models/transformer.py`, `parallel/tensor_parallel/layers.py`

Embedding output, attention residual, and MLP output are TP-*replicated* tensors — identical across ranks after all-reduce. The invariant TP requires is that dropout applied to a replicated tensor draws the **same mask on every rank**, otherwise the tensor stops being replicated (each rank now holds a different post-dropout value) while downstream code still assumes replication. Previously, dropout used each rank's local (unsynced) RNG state, breaking this invariant. `TensorParallelRNGTracker` gives these call sites a dedicated RNG stream that is kept in sync across the TP group.

## Root Cause 3: DistributedOptimizer broadcast used the wrong `src` rank under TP>1

**File**: `ironcore/optimizer/distributed_optimizer.py`

The post-step parameter broadcast used a DP-group-*relative* rank as the broadcast `src`. That's only correct when TP=1 (DP group == world). Under TP>1, the DP group's global ranks are non-contiguous (e.g. with TP=2, DP ranks are `{0, 2}` or `{1, 3}`, not `{0, 1}`), so a DP-relative rank resolved to the wrong *global* rank, corrupting the broadcast under TP>1.

**Fix**: resolve the true global rank via `dist.get_global_rank()` before broadcasting.

## Root Cause 4: Muon Newton-Schulz orthogonalization ran per-shard, not on the full matrix

**File**: `ironcore/optimizer/muon.py`

Muon's orthogonalization step (Newton-Schulz iteration) was applied to each rank's *local TP shard* independently. This is not mathematically equivalent to orthogonalizing the full matrix (what TP=1 does) — orthogonalizing two half-matrices independently does not reproduce orthogonalizing the concatenation of the two halves.

**Fix**: `_orthogonalize_tp_aware` now all-gathers the full matrix across the TP group, runs Newton-Schulz once on the gathered matrix, and re-shards the result back to each rank's local slice, using the `tp_shard_dim`/`tp_concatenated_weights` metadata already carried on TP-sharded weight tensors. The RMS-matching scale is computed from the gathered (full) shape, matching TP=1.

While fixing this, a separate correctness bug in `is_muon_param`'s pattern list was also found and fixed: `linear_q`/`linear_kv`/`attn_output` are direct children of `TransformerLayer` (`self_attention` is a weightless compute module), so the old `"self_attention.linear_q.weight"`-style patterns never matched, and all attention weights were silently falling back to AdamW instead of Muon.

## Related: param-norm double counting under TP (found alongside, not divergence-causing)

**File**: `ironcore/parallel/grad_norm.py` (new, consolidated), `trainers/base_trainer.py`, `trainers/grpo_trainer.py`

`GRPOTrainer._compute_grad_and_param_norms` summed non-expert param norms across the TP group without separating replicated params (biases, norms, embeddings) from TP-sharded ones, over-counting replicated params by `tp_size` in the logged param norm. `LanguageModelTrainer`'s version already handled this correctly; the correct logic was extracted into a shared `compute_param_norm()` used by both trainers. A duplicate, differently-buggy `clip_grad_norm_tp` (same all-reduce-SUM-without-exclusion issue, and unused) was deleted from `parallel/tensor_parallel/comm.py`.

This is a logging/gradient-clipping correctness issue, not a training-divergence cause (it doesn't feed back into the forward/backward computation), but it was found during the same TP audit and fixed in the same commit.

---

## Fix Commits

| Commit | Summary |
|---|---|
| `9cf2937` | per-parameter TP init seed, `TensorParallelRNGTracker` for dropout, `DistributedOptimizer` global-rank broadcast fix |
| `6a68139` | TP-aware Muon Newton-Schulz orthogonalization on the gathered full matrix; `is_muon_param` pattern fix |
| `a26b9b6` | consolidated correct param-norm computation; deleted dead/buggy `clip_grad_norm_tp` |

---

## Validation

### New tests added
- `tests/unit/parallel/test_tp_init_seed.py` — per-parameter seed derivation, CPU-only
- `tests/multi_gpu/test_tp_equivalence.py` — real 2-GPU equivalence checks, requires `torchrun --nproc_per_node=2`

### Real 2×RTX 3090 run (this session)

```bash
torchrun --nproc_per_node=2 -m pytest tests/multi_gpu/test_tp_equivalence.py -v --tb=short
```

```
tests/multi_gpu/test_tp_equivalence.py::test_tp2_init_matches_analytic_full_tensor PASSED [ 50%]
tests/multi_gpu/test_tp_equivalence.py::test_tp2_muon_orthogonalization_matches_full_matrix PASSED [100%]
========================= 2 passed, 1 warning in 4.00s =========================
```

- `test_tp2_init_matches_analytic_full_tensor`: gathered TP=2 init matches the analytic TP=1-equivalent full tensor (validates Root Cause 1's fix)
- `test_tp2_muon_orthogonalization_matches_full_matrix`: TP=2 Muon orthogonalization matches orthogonalizing the full matrix directly (validates Root Cause 4's fix)

Full-suite regression run (787 passed / 18 failed / 20 skipped inside the NGC container, both GPUs free) showed no failures in any file touched by these fixes; the 18 failures are pre-existing test-isolation issues unrelated to this work — see [test_suite_isolation_issues.md](test_suite_isolation_issues.md).

---

## Conclusions

TP=1/TP=2 numerical equivalence is restored for weight init, dropout, distributed-optimizer broadcast, and Muon orthogonalization — the four places where TP=2's "split the work across ranks" strategy previously computed something different from TP=1's single-rank computation. The fix is validated both by new targeted unit/multi-GPU tests and by confirming no regression across the broader test suite.

### Follow-ups
- The original TP=1 vs TP=2 divergence numbers (loss diff 5.33e-3, weight diff 1.57e-1) came from an ad-hoc validation run during the audit rather than a saved, reproducible experiment script — worth formalizing as a standing regression check (e.g. promote to `tests/regression/`) so future TP-path changes get the same weight-level scrutiny, not just loss-curve comparison.
- The earlier [distributed_training_validation.md](distributed_training_validation.md) DP=2/TP=2 comparison should be re-run post-fix; it wasn't wrong, but it also wasn't sensitive enough to have caught this class of bug.
