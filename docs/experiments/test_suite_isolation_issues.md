# Test Suite Isolation Issues (18 failures in full-suite runs)

**Date**: 2026-07-06 (findings), 2026-07-06 (root-caused + fixed on `fix/test-isolation`)
**Branch**: found while validating `fix/training-correctness-audit`; issues themselves are pre-existing on `main`. Fixed on `fix/test-isolation` (branched off the audit branch).
**Status**: All root causes below are now confirmed (no more hypotheses), fixed, and verified end-to-end (see "Verification" at the end). One issue (5) turned out to be an unrelated, out-of-scope pre-existing bug, not a test-isolation problem — left unfixed and flagged separately.

---

## Executive Summary

Validating `fix/training-correctness-audit` required running the full test suite inside the NGC container on 2×RTX 3090. Two separate full-suite runs surfaced **18 failures + 1 hang**, none of them in files touched by this branch's 10 commits, and the most-failing file (`test_rollout_sanity.py`) passes 12/12 when run standalone. The common thread: this suite runs ~830 tests in a **single pytest process**, and several tests depend on process-global mutable state (`ironcore.parallel.parallel_states`, `os.environ["RANK"]`, `torch.backends.cudnn.deterministic`) that other tests set and never clean up. Test outcome becomes a function of *what ran before it*, not just its own logic.

This is orthogonal to the actual production-code fixes on the audit branch — confirmed via `git diff main` (at finding time, before any fix below) showing zero changes to any of the affected test files or their exercised code paths (see per-issue detail below) — but it undermines confidence in `./scripts/run_tests.sh` as a clean-room CI signal, so it was fixed on its own branch (`fix/test-isolation`) rather than folded into the audit branch.

---

## Issue 1: `mp`-marked tests hang instead of skipping (1 hang, blocks the whole run)

**File**: `tests/integration/offload/test_weight_streaming_dp.py::TestDPWeightOffload::test_dp2_weight_offload_with_optimizer`

**Symptom**: `./scripts/run_tests.sh`'s first stage (plain `pytest tests/`, no `torchrun`) hung indefinitely — confirmed via `py-spy`-style investigation (blocked in `torch.distributed.init_process_group` → `eager_connect_single_device`, waiting for a second rank that will never appear in a single-process run).

**Expected behavior** (documented in `tests/test_suite.md`): "`mp` tests guard themselves with `pytest.skipif("RANK" not in os.environ …)` so they safely skip under plain `pytest` and are exercised by the `distributed-tests` CI job via per-file `torchrun`." This file does implement exactly that guard (`has_multi_gpu = cuda_available and device_count>=2 and os.environ.get("RANK") is not None`).

**What we confirmed**:
- Running this file **alone** (`pytest tests/integration/offload/test_weight_streaming_dp.py`): both tests correctly **SKIPPED** (`RANK` unset → guard works as designed).
- Running the **full suite**: `test_dp2_weight_offload_converges` (1st test in the class) **PASSED** (ran for real), then `test_dp2_weight_offload_with_optimizer` (2nd test, same class, same guard) **hung** in real `init_process_group(world_size=2)`.

**Root cause (confirmed)**: `tests/integration/moe/test_moe_{correctness,functional,layer}.py:19-21` each ran
```python
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
```
as **top-level, import-time statements** (not inside a function or fixture). Pytest imports every test module during collection — across the whole suite, before any test executes — and `tests/integration/moe/` sorts alphabetically **before** `tests/integration/offload/`. So by the time `test_weight_streaming_dp.py`'s module-level `has_multi_gpu = ... and os.environ.get("RANK") is not None` guard is evaluated (also at import time), `RANK` is already `"0"` from the moe modules' import-time side effect — the guard sees a "real" RANK and doesn't skip. Standalone, `tests/integration/moe/` is never imported, so `RANK` stays unset and the guard correctly skips. Three more files had the identical import-time pattern: `tests/unit/moe/test_expert.py:21-23`, `tests/unit/parallel/test_expert_parallel.py:19-21`, `tests/unit/parallel/test_comm_grad_correctness.py:18-20`.

This also explains the pass-then-hang: the first test (`test_dp2_weight_offload_converges`) reused an already-initialized `world_size=1` process group left behind by an earlier test in the same process (visible in the log as `INFO ironcore.logger: Torch distributed is already initialized: 1`), so its own `init_process_group(world_size=2)` call was actually a silent no-op against the existing group. The second test ran after some intervening cleanup had reset the process group, so its `init_process_group(world_size=2)` call was for real — and with only one process alive, it blocked forever waiting for a rank 1 that will never appear.

**A separate, definite gap** found alongside: `scripts/run_tests.sh`'s stage 2 ("Integration Tests (torchrun)") explicitly re-runs a curated list of `mp`-marked files via `torchrun --nproc_per_node=2` (e.g. `test_attention_multi_gpu.py`, `test_kv_cache.py`, `test_lora_correctness.py`) — but **`test_weight_streaming_dp.py` and `test_weight_streaming_mp.py` are missing from that list**. So even when the skip guard works correctly, these two files never actually get exercised under real `torchrun` locally; they only ever run (or hang, or incorrectly skip) inside the single-process stage-1 sweep. A full audit found 18 `mp`-marked files total were missing from `run_tests.sh`'s torchrun coverage (7 of the 18 were at least covered by CI's `scripts/run_distributed_tests.sh`; 11 ran under real `torchrun` nowhere at all — including `tests/multi_gpu/test_tp_equivalence.py`, the audit branch's own headline validation test).

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

**Root cause (confirmed)**: two independent bugs stacked together.

1. **Leaked `parallel_states`, and — the deeper layer found while re-verifying the fix — leaked `dist` itself**: `ironcore.parallel.parallel_states` holds module-level globals (`_TENSOR_MODEL_PARALLEL_WORLD_SIZE` etc.) shared across the entire pytest process. Several fixtures/tests call `initialize_model_parallel()` without a paired `destroy_model_parallel()` — notably `tests/unit/offload/conftest.py`'s `init_tp` autouse fixture (initializes TP=1 if not already initialized, but had no teardown) and `tests/integration/offload/conftest.py`'s `reset_state` autouse fixture (reset `ironcore.global_vars.GLOBAL_STATES` but never touched `parallel_states`, even though `run_training_step()` in the same file calls `trainer._initialize()` → `initialize_model_parallel()`).

   Fixing those closed most of Issue 3, but re-running the full suite afterward surfaced a **new**-looking failure in `tests/unit/parallel/test_tp_init_seed.py` (3 tests, added by the audit branch, previously passing) plus a residual `test_rollout_sanity.py` failure with a *different* error (`RuntimeError: No backend type associated with device type cpu`, from `rollout.py`'s `_fsdp_done_check` calling `dist.all_reduce` on a CPU tensor). Both traced to the same deeper leak: `tests/integration/optimizer/test_optimizer.py` has **three** separate setup/cleanup helper pairs (module-level `setup_distributed()`/`cleanup_distributed()`, `TestOptimizerTPIntegration._setup_tp_distributed()`/`_cleanup_tp_distributed()`, and `TestDistributedOptimizerIntegration._setup_distributed()`/`_cleanup_distributed()`). Only the module-level one called `dist.destroy_process_group()`; the other two only called `parallel_states.destroy_model_parallel()`, leaving `dist.is_initialized()` permanently `True` (PyTorch process groups can't be un-initialized without `destroy_process_group()`) while `parallel_states` gets reset to `None`. Confirmed independent of any `mp`/`TORCHELASTIC_RUN_ID` behavior — `pytest tests/integration/optimizer/test_optimizer.py tests/unit/parallel/test_tp_init_seed.py` reproduces the `test_tp_init_seed.py` failures in total isolation from the rest of the suite, with or without simulating a torchrun launch, because the offending test (`test_muon_tp1_with_trainer`) isn't `mp`-marked and always runs. Once any test leaves TP state initialized (or, worse, some other test calls `destroy_model_parallel()` and resets it to uninitialized) while `dist` stays initialized, later tests inherit whatever half-broken state is sitting in those globals.
2. **Stale exception filter in production code**: `ironcore/alignment/rollout.py`'s TP-group lookup (both `generate_rollouts_batched` and `generate_rollouts_paged`) was:
   ```python
   if dist.is_initialized():
       try:
           from ironcore.parallel.parallel_states import (
               get_tensor_model_parallel_group,
               get_tensor_model_parallel_world_size,
           )
           if get_tensor_model_parallel_world_size() > 1:
               tp_group = get_tensor_model_parallel_group()
       except (AssertionError, ImportError):
           pass
   ```
   But `get_tensor_model_parallel_world_size()` raises **`RuntimeError`** when `parallel_states` isn't initialized (`ironcore/parallel/parallel_states.py:134-140`), not `AssertionError`. So whenever an earlier test left `dist` initialized but `parallel_states` *not* initialized (exactly the leak described above), this call raised `RuntimeError`, the `except (AssertionError, ImportError)` didn't catch it, and it propagated straight out of `generate_rollouts_batched`/`generate_rollouts_paged` as an unhandled exception — which is exactly the traceback the 12 failures showed.

Either bug alone wouldn't reliably reproduce this; the leak needed to land `parallel_states` in the *uninitialized* state (not just "wrong TP size") for `get_tensor_model_parallel_world_size()` to hit its `raise` branch, and the stale `except` clause needed to fail to catch that specific exception type.

## Issue 4: Memory-threshold flake (1 failure)

**File**: `tests/integration/test_integration.py::TestMemoryEfficiency::test_no_memory_leak`
```
AssertionError: Peak memory too high: 887.7 MB
assert 887.671875 < 500
```
`git diff main`: empty. The 500MB threshold is almost certainly calibrated for this test running against a clean CUDA allocator state; ~830 other tests worth of allocations/fragmentation before it in the same process plausibly explain the overage. Not investigated further — flagged as another instance of the same "assumes a clean process" pattern.

## Issue 5: Gradient mismatch under offload — NOT a test-isolation issue (3 failures, out of scope)

**File**: `tests/unit/offload/test_activation_spill_convergence.py::TestActivationSpillGradientParity::{test_backward_gradient_parity_no_dropout, test_backward_gradient_parity_with_dropout, test_full_layer_granularity_gradient_parity}`

Originally hypothesized as cudnn-flag leakage (Issues 3-4's pattern). **That hypothesis was wrong** — checked while re-verifying the other fixes:
- `pytest tests/unit/offload/test_activation_spill_convergence.py` **standalone**: still fails, identically (3 failed, 3 passed). Not a cross-test artifact at all.
- `git diff main`: empty for both the test file and `ironcore/offload/`.
- The failure shape doesn't look like cudnn nondeterminism either: every parameter's gradient diff is exactly `0.0` except **one** (`mlp.down_proj.bias` or `attn_output.bias`, depending on the test), which differs by a suspiciously large, non-noise-like amount (`7.75`, `6.625`, `5.5625`). Nondeterministic algorithm selection would show small differences spread across many parameters, not 25/26 exact matches plus one large outlier — this looks like a genuine bug in the activation-spill/offload gradient path (e.g. a buffer-reuse or accumulation-order issue), not noise.

This is a real, pre-existing, standalone-reproducible bug, unrelated to test isolation and unrelated to the `fix/training-correctness-audit` branch. **Not fixed here** — flagged for separate investigation into `ironcore/offload/` (activation spill / gradient accumulation), out of scope for a test-infrastructure branch.

---

## Common Root-Cause Pattern

17 of the 18 original failures + the 1 hang fall into two buckets, both now confirmed (the 18th, Issue 5, turned out to be an unrelated real bug — see above):

1. **Import-time `os.environ` mutation defeats a collection-time skip guard** (Issues 1-2): `tests/integration/moe/test_moe_*.py` and 3 other files set `RANK`/`WORLD_SIZE`/`LOCAL_RANK` as top-level statements, which pytest executes for every module during collection regardless of what runs later. Any other module's `os.environ.get("RANK")`-based guard, evaluated at the same import time, sees this pollution. Also: `run_tests.sh`'s torchrun stage didn't cover every `mp`-marked file that needs real multi-process execution to be exercised at all.
2. **Process-global mutable state leaking across tests, surfaced by a stale exception filter** (Issues 3-4): `torch.distributed`'s initialized flag and `ironcore.parallel.parallel_states` module globals persist for the life of the pytest process — some cleanup helpers reset one but not the other, leaving a half-torn-down state; a separate absolute memory threshold (Issue 4) was also state-sensitive. In Issue 3's case, the leak was made visible by production code (`rollout.py`) catching the wrong exception type.

Neither bucket touches any file changed by the `fix/training-correctness-audit` branch's 10 commits.

---

## Fixes Applied (`fix/test-isolation` branch)

- **Import-time env pollution** (Issue 1 root cause): the 6 files with top-level `os.environ.setdefault(RANK/WORLD_SIZE/LOCAL_RANK)` now use a `single_gpu_env()` context manager (`tests/fixtures/utils.py`) wrapped in an autouse fixture — same effective values, but scoped to the test and restored on exit.
- **`mp` skip guard hardened centrally**: `tests/conftest.py`'s auto-skip for `@pytest.mark.mp` now additionally requires `TORCHELASTIC_RUN_ID` (set only by a real `torchrun` launch) instead of relying on file-local `RANK` checks that import-time pollution can defeat. The two `test_weight_streaming_{dp,mp}.py` local guards were switched to the same sentinel as belt-and-suspenders. A new autouse fixture in `tests/conftest.py` also snapshots/restores `RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT` around every test, closing the *runtime* leak class (e.g. `tests/integration/optimizer/test_optimizer.py`'s `setup_distributed()`, which hard-assigns these and never cleaned up).
- **rollout.py stale exception filter** (Issue 3 root cause #2): both TP-group lookups now check `dist.is_initialized() and parallel_states.is_model_parallel_initialized() and parallel_states.get_tensor_model_parallel_world_size() > 1` explicitly, with no try/except — there's nothing left to catch.
- **`parallel_states` leak sources** (Issue 3 root cause #1): `tests/unit/offload/conftest.py`'s `init_tp` and `tests/integration/offload/conftest.py`'s `reset_state` now pair every `initialize_model_parallel()` with a `destroy_model_parallel()`. `tests/unit/test_sft_masking.py` got the same treatment. (`tests/unit/attention/test_attention_parallel_validation.py` turned out to be a standalone script with zero `def test_*` functions — pytest imports it but nothing ever executes, so it wasn't actually a leak source despite matching the `test_*.py` naming pattern.)
- **`dist` leak source** (Issue 3 root cause #3, found while re-verifying): `tests/integration/optimizer/test_optimizer.py`'s `_cleanup_tp_distributed` and `TestDistributedOptimizerIntegration._cleanup_distributed` now also call `dist.destroy_process_group()`, matching the pattern the file's own module-level `cleanup_distributed()` already used correctly.
- **cudnn flag leakage** — turned out not to be the cause of anything (see corrected Issue 5), but the 7 files toggling `torch.backends.cudnn.deterministic`/`benchmark` at import time now do it through a `cudnn_determinism()` context manager (`tests/fixtures/utils.py`) in an autouse fixture anyway, save/restore instead of one-way set — a legitimate isolation improvement even though it wasn't the culprit here.
- **Memory test** (Issue 4): `test_no_memory_leak` now measures growth (`memory_allocated()` after a warmup pass vs. after 9 more passes, asserting near-zero delta) plus a generous 2GB absolute sanity cap, instead of a state-sensitive 500MB absolute peak.
- **`run_tests.sh` coverage + failure masking**: torchrun file lists for `run_tests.sh` and `run_distributed_tests.sh` are now a single sourced file (`scripts/distributed_test_files.sh`), with all 28 `@pytest.mark.mp` files (audited via `grep -rl 'pytest.mark.mp' tests/`) present in the NP2 list — including `tests/multi_gpu/test_tp_equivalence.py`, previously exercised under real `torchrun` nowhere locally. `run_tests.sh` no longer swallows pytest's exit code via `|| true`; a `run_pytest_capture` helper treats an abnormal exit code (crash/collection error) or a 0/0/0 passed+failed+skipped result as a failure instead of silently reporting "✓ PASSED — 0 tests".

---

## Verification (2×RTX 3090, NGC container)

- Full suite (`pytest tests/ --timeout=120`): **18 failed + 1 hang → 3 failed** (only Issue 5, confirmed out of scope). 762 passed, 61 skipped, 5 deselected (`e2e`), no hang.
- `test_weight_streaming_dp.py::test_dp2_weight_offload_with_optimizer` (the original hang): plain `pytest` → both tests **SKIPPED**, no hang. Real `torchrun --nproc_per_node=2` (its first-ever real execution — previously exercised under real torchrun nowhere) → **2 passed** in 132.77s, confirming the test's actual logic is correct, not just that it now skips safely.
- `tests/multi_gpu/test_tp_equivalence.py` under real `torchrun --nproc_per_node=2`: still **2 passed** — the audit branch's headline fix is unaffected.
- `test_rollout_sanity.py` standalone vs. full suite: 12 passed / 1 skipped in both, now consistent.
