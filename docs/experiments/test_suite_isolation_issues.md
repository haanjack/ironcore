# Test Suite Isolation Issues (18 failures in full-suite runs)

**Date**: 2026-07-06
**Branch**: fix/training-correctness-audit (issues are pre-existing on `main`, not caused by this branch)
**Purpose**: Record test-infrastructure fragility discovered while validating the training-correctness-audit branch, for follow-up discussion. Not fixed yet — this is a findings writeup, not a fix.

---

## Executive Summary

Validating `fix/training-correctness-audit` required running the full test suite inside the NGC container on 2×RTX 3090. Two separate full-suite runs surfaced **18 failures + 1 hang**, none of them in files touched by this branch's 10 commits, and the most-failing file (`test_rollout_sanity.py`) passes 12/12 when run standalone. The common thread: this suite runs ~830 tests in a **single pytest process**, and several tests depend on process-global mutable state (`ironcore.parallel.parallel_states`, `os.environ["RANK"]`, `torch.backends.cudnn.deterministic`) that other tests set and never clean up. Test outcome becomes a function of *what ran before it*, not just its own logic.

This is orthogonal to the actual production-code fixes on the audit branch — confirmed via `git diff main` showing zero changes to any of the affected test files or their exercised code paths (see per-issue detail below) — but it undermines confidence in `./scripts/run_tests.sh` as a clean-room CI signal and is worth fixing on its own.

---

## Issue 1: `mp`-marked tests hang instead of skipping (1 hang, blocks the whole run)

**File**: `tests/integration/offload/test_weight_streaming_dp.py::TestDPWeightOffload::test_dp2_weight_offload_with_optimizer`

**Symptom**: `./scripts/run_tests.sh`'s first stage (plain `pytest tests/`, no `torchrun`) hung indefinitely — confirmed via `py-spy`-style investigation (blocked in `torch.distributed.init_process_group` → `eager_connect_single_device`, waiting for a second rank that will never appear in a single-process run).

**Expected behavior** (documented in `tests/test_suite.md`): "`mp` tests guard themselves with `pytest.skipif("RANK" not in os.environ …)` so they safely skip under plain `pytest` and are exercised by the `distributed-tests` CI job via per-file `torchrun`." This file does implement exactly that guard (`has_multi_gpu = cuda_available and device_count>=2 and os.environ.get("RANK") is not None`).

**What we confirmed**:
- Running this file **alone** (`pytest tests/integration/offload/test_weight_streaming_dp.py`): both tests correctly **SKIPPED** (`RANK` unset → guard works as designed).
- Running the **full suite**: `test_dp2_weight_offload_converges` (1st test in the class) **PASSED** (ran for real), then `test_dp2_weight_offload_with_optimizer` (2nd test, same class, same guard) **hung** in real `init_process_group(world_size=2)`.
- This is puzzling given both tests share one class-level `@skip_no_multi_gpu` decorator computed from a single `has_multi_gpu` boolean evaluated once at **module import (collection) time** — before any test executes. Both should see the same value.
- Grepped the whole repo for `os.environ["RANK"] =` / `os.environ['RANK'] =`: only 3 files do this (`tests/integration/optimizer/test_optimizer.py`, `tests/unit/attention/test_attention_parallel_validation.py`, `tests/benchmarks/benchmark_parallel_memory.py`), and all three execute **after** this test in the full run's collection order (confirmed by extracting the actual `pytest-shard` "Running N items in this shard" list and checking index positions — offload's test is at index 141-142, `test_optimizer.py` doesn't start until index 148).
- **Not resolved**: how `RANK` (or equivalent) ends up making the guard permissive for one test in the class but not consistently for both. Time-boxed the investigation and moved on rather than fully root-causing — worth another look.

**A separate, definite gap** found alongside: `scripts/run_tests.sh`'s stage 2 ("Integration Tests (torchrun)") explicitly re-runs a curated list of `mp`-marked files via `torchrun --nproc_per_node=2` (e.g. `test_attention_multi_gpu.py`, `test_kv_cache.py`, `test_lora_correctness.py`) — but **`test_weight_streaming_dp.py` and `test_weight_streaming_mp.py` are missing from that list**. So even when the skip guard works correctly, these two files never actually get exercised under real `torchrun` locally; they only ever run (or hang, or incorrectly skip) inside the single-process stage-1 sweep.

## Issue 2: `mp`-marked TP offload tests fail outright when not hung (2 failures)

**File**: `tests/integration/offload/test_weight_streaming_mp.py::TestTPWeightOffload::{test_tp2_weight_offload_converges, test_tp2_weight_offload_with_optimizer}`

Same file shape and same `has_multi_gpu`/`RANK` guard pattern as Issue 1. `git diff main` for this file is empty — unchanged. Same missing-from-torchrun-list gap applies. Grouped with Issue 1's root cause family rather than investigated separately.

## Issue 3: `test_rollout_sanity.py` — passes standalone, fails in full suite (12 failures)

**File**: `tests/unit/alignment/test_rollout_sanity.py`

**Symptom**: 12 of 13 tests failed in the full-suite run with:
```
ironcore/parallel/parallel_states.py:137: in get_tensor_model_parallel_world_size
    raise RuntimeError(...)
RuntimeError: Tensor model parallel not initialized. Call initialize_model_parallel() first.
```
(with an `INFO ironcore.logger: Torch distributed is already initialized: 1` line right before it, from an unrelated earlier test).

**Confirmed**:
- `git diff main -- tests/unit/alignment/test_rollout_sanity.py`: empty (unchanged).
- `git diff main -- ironcore/parallel/parallel_states.py`: this branch *does* touch this file (added `reset_tensor_parallel_rng_tracker()` calls inside `initialize_model_parallel()`/`destroy_model_parallel()` — part of the TP RNG divergence fix, see [tp_rng_divergence_fix.md](tp_rng_divergence_fix.md)) — but does **not** touch `get_tensor_model_parallel_world_size()` or the state variables it reads. Ruled out as the cause.
- Running the file **standalone**: `pytest tests/unit/alignment/test_rollout_sanity.py` → **12 passed, 1 skipped**. Confirms the test logic itself is correct; the failure is purely a full-suite ordering/state artifact.

**Working hypothesis** (not confirmed): `ironcore.parallel.parallel_states` holds module-level globals (`_TENSOR_MODEL_PARALLEL_WORLD_SIZE` etc.) shared across the entire pytest process. Some earlier test in the full run calls `initialize_model_parallel()` then either never calls `destroy_model_parallel()`, or an intervening test resets torch.distributed without resetting these ironcore-level globals, leaving them in a state this test doesn't expect. Needs a fixture (`autouse`, session- or function-scoped) that guarantees `destroy_model_parallel()` runs between tests, or these globals need to move to a context object instead of module state.

## Issue 4: Memory-threshold flake (1 failure)

**File**: `tests/integration/test_integration.py::TestMemoryEfficiency::test_no_memory_leak`
```
AssertionError: Peak memory too high: 887.7 MB
assert 887.671875 < 500
```
`git diff main`: empty. The 500MB threshold is almost certainly calibrated for this test running against a clean CUDA allocator state; ~830 other tests worth of allocations/fragmentation before it in the same process plausibly explain the overage. Not investigated further — flagged as another instance of the same "assumes a clean process" pattern.

## Issue 5: Gradient-parity flake under offload (3 failures)

**File**: `tests/unit/offload/test_activation_spill_convergence.py::TestActivationSpillGradientParity::{test_backward_gradient_parity_no_dropout, test_backward_gradient_parity_with_dropout, test_full_layer_granularity_gradient_parity}`

`git diff main`: empty. These compare gradients against a reference path under bitwise/near-bitwise tolerance; several files elsewhere in the suite toggle `torch.backends.cudnn.deterministic` / `torch.backends.cudnn.benchmark` at module level (e.g. `test_weight_streaming_dp.py` sets `cudnn.deterministic = True; cudnn.benchmark = False` at import time) and never restore the prior value. If a benchmark-mode-dependent test runs after one of those, algorithm selection (and thus exact gradient values) can differ from a clean-process run. Consistent with the same state-leakage pattern as Issues 3-4, not independently confirmed.

---

## Common Root-Cause Pattern

All 18 failures + the 1 hang fall into two buckets:

1. **`mp`-marker tests whose skip guard depends on `os.environ["RANK"]`, run outside `torchrun`** (Issues 1-2). The guard is documented and mostly works (proven by standalone runs skipping correctly), but something about running inside the full ~830-test process makes it behave inconsistently for at least one file. Also: `run_tests.sh`'s torchrun stage doesn't cover every `mp`-marked file that needs it.
2. **Process-global mutable state leaking across tests** (Issues 3-5): `ironcore.parallel.parallel_states` module globals, `torch.backends.cudnn.*` flags, and CUDA allocator fragmentation all persist for the life of the pytest process and are not guaranteed to be reset between tests by a shared fixture.

Neither bucket touches any file changed by the `fix/training-correctness-audit` branch's 10 commits.

---

## Suggested Directions (for discussion, not started)

- Add an `autouse` fixture (session- or module-scoped, at `tests/conftest.py` or `tests/integration/conftest.py`) that snapshots and restores `os.environ`, `ironcore.parallel.parallel_states` globals, and `torch.backends.cudnn.*` flags around every test, or at least around every test file boundary.
- Audit `scripts/run_tests.sh`'s stage-2 `torchrun` file list against `grep -rl 'pytest.mark.mp' tests/` to find every `mp` file that's currently orphaned (never correctly exercised locally) — `test_weight_streaming_dp.py` and `test_weight_streaming_mp.py` are confirmed missing; there may be others.
- Consider running `tests/unit/` and `tests/integration/` as separate pytest invocations (or with `pytest-forked`/`pytest-xdist` process isolation per file) rather than one giant process, trading some speed for isolation guarantees — would have caught/avoided all 5 issues here.
- Finish root-causing Issue 1's inconsistent-skip puzzle before deciding on a fix; the mechanism by which `has_multi_gpu` differs between two tests in the same decorated class is not yet understood.
