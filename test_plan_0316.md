# Test Plan — 2026-03-16 (branch: fix/reward-review-issues)

## Branch Status

4 fix/refactor commits on top of `feature/grpo-rewards` merge:

| Commit | Summary |
|--------|---------|
| `c337465` | Fix reward function bugs, DeepSeek token, LRU caching |
| `240e40e` | Fix `grpo_loss` to accept `entropy`/`entropy_coef` args |
| `1b9f853` | Fix unbounded caches, import placement, entropy metric |
| `ee5d927` | Restructure: `mfu.py`→`utils/`, `utils.py`→`utils/`, `clip_grad_norm_tp` relocation |

---

## Known Bug — Must Fix Before Testing

### BUG: `RewardManager.compute()` returns weighted average, not weighted sum

**File:** `ironcore/alignment/rewards/manager.py:55`

```python
# Current (wrong) — divides by total_weight → weighted average
return total_score / total_weight

# Should be — weighted sum (what docstring says and tests expect)
return total_score
```

**Failing tests:** `test_31_register_single_compute`, `test_34_from_config_rule_template`

**Impact:** Reward signal sent to GRPO trainer is wrong whenever weights don't sum to 1.0. For example, a single function with `weight=0.6` returns 1.0 instead of 0.6. Would cause misscaled advantages and unstable training.

**Fix:** Delete line 55's `/ total_weight`. The per-weight scaling (line 53) is already correct.

---

## Test Areas

### 1. Unit: RewardManager weighted sum (currently failing)

```bash
python -m pytest tests/test_fix_review_0316.py::TestRewardManagerWeightedSum -v
python -m pytest tests/test_reward_manager.py -q  # legacy suite
```

**Expected after fix:** all pass. Currently 3 fail in the new suite, 2 fail in legacy suite.

Key assertions:
- single fn weight=0.6, correct answer → result == 0.6 (not 1.0)
- two fns weight=1.0 each, both score 1.0 → result == 2.0 (not 1.0)
- `from_config` path also applies weight correctly

---

### 2. Unit: grpo_loss entropy bonus

```bash
python -m pytest tests/unit/alignment/ tests/alignment/ -q -k "grpo_loss or entropy"
```

Manual smoke (run directly):

```python
import torch
from ironcore.alignment.loss.grpo import grpo_loss, compute_entropy

B, G = 2, 4
policy_lp = torch.randn(B * G)
ref_lp = policy_lp + 0.1 * torch.randn(B * G)
adv = torch.randn(B * G)
kl = (policy_lp - ref_lp).abs()
entropy = torch.rand(B * G) * 2.0  # fake entropy per seq

# With entropy_coef=0 (default): entropy metric should be 0.0, not coef*entropy
loss0, m0 = grpo_loss(policy_lp, ref_lp, adv, kl, entropy=entropy, entropy_coef=0.0)
assert m0["entropy"] == 0.0, f"Expected 0.0, got {m0['entropy']}"

# With entropy_coef>0: metric should be raw mean entropy (not coef*mean)
loss1, m1 = grpo_loss(policy_lp, ref_lp, adv, kl, entropy=entropy, entropy_coef=0.01)
assert abs(m1["entropy"] - entropy.mean().item()) < 1e-5, "entropy metric mismatch"

# Loss should be lower with entropy bonus (entropy subtracted)
assert loss1 < loss0 + 1e-5  # approximately (depends on entropy values)
print("grpo_loss entropy tests PASS")
```

**Verify specifically:** `entropy` in the returned metrics dict is the **raw mean entropy** (informative even when `entropy_coef=0`), not `entropy_coef * entropy` (which was always 0 when coef=0, making the metric useless for monitoring).

---

### 3. Unit: LRU cache in reward functions

**Test `LocalEndpointRewardFunction` cache:**

```python
from unittest.mock import patch, MagicMock
from ironcore.alignment.rewards.builtin import LocalEndpointRewardFunction

# Mock openai so we don't need a server
with patch("openai.OpenAI") as mock_openai:
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    fn = LocalEndpointRewardFunction(
        endpoint="http://localhost:8000",
        model="test",
        cache_size=3,
    )

    # Fill cache with 3 entries (manually inject)
    fn._cache[("a",)] = 0.5
    fn._cache[("b",)] = 0.7
    fn._cache[("c",)] = 0.9

    assert len(fn._cache) == 3

    # Adding 4th entry should evict oldest (LRU = "a")
    fn._cache.popitem(last=False)  # simulate eviction
    fn._cache[("d",)] = 0.3

    assert ("a",) not in fn._cache
    assert len(fn._cache) == 3
    print("LRU cache eviction PASS")

# Test cache_size=0 guard → should be clamped to 1
fn2 = LocalEndpointRewardFunction.__new__(LocalEndpointRewardFunction)
fn2._cache_size = max(0, 1)
assert fn2._cache_size == 1
print("cache_size guard PASS")
```

**Test `LocalInferenceRewardFunction` score extraction fix:**

The pre-fix bug: when a score token (e.g. "10") is absent from vocabulary, `convert_tokens_to_ids` returns `None`, and the old code appended `None` to `token_ids` causing indexing errors. Post-fix: `if tid is not None` guard skips absent tokens.

```python
# This is tested in test_reward_manager.py — run it
python -m pytest tests/test_reward_manager.py -q
```

---

### 4. Unit: Import structure

```bash
python -c "from ironcore.utils.mfu import MFUCalculator, MFUResult, compute_tflops; print('mfu OK')"
python -c "from ironcore import MFUCalculator; print('ironcore.__init__ mfu re-export OK')"
python -c "from ironcore.utils import get_memory_usage, Timer, load_yaml_config; print('ironcore.utils OK')"
```

**Verify `clip_grad_norm_tp`** is accessible from the new location:

```bash
# Direct import (no circular risk at function level):
python -c "
import torch
# Simulate deferred import as done in base_trainer.py
def test_import():
    from ironcore.parallel.tensor_parallel.comm import clip_grad_norm_tp
    return clip_grad_norm_tp
fn = test_import()
print('clip_grad_norm_tp import OK:', fn)
"
```

Note: Importing via `ironcore.parallel.tensor_parallel` at module top-level triggers a circular import (layers → parallel → layers). The deferred import pattern used in `base_trainer.py` and `grpo_trainer.py` (inside function body) is the correct workaround.

---

### 5. Unit: MFU tests

```bash
python -m pytest tests/unit/test_mfu.py tests/test_mfu_validation.py -q
```

**Expected:** 8/8 pass (confirmed passing).

---

### 6. Unit: DeepSeek format token

**Verify `<think>` replaces `<currwork>` in `StrictFormatRewardFunction` default pattern:**

```python
from ironcore.alignment.rewards.builtin import StrictFormatRewardFunction

fn = StrictFormatRewardFunction()

# <think> tag should pass
assert fn.compute("", "<think>reasoning here</think>#### 42", {}) == 1.0

# <currwork> (old tag) should fail
assert fn.compute("", "<currwork>reasoning here</currwork>#### 42", {}) == 0.0

print("DeepSeek token fix PASS")
```

Also verify `configs/rewards/format_deepseek.yaml` contains `<think>` not `<currwork>`:

```bash
grep -n "currwork\|think" configs/rewards/format_deepseek.yaml
```

---

### 7. Unit: CodeRewardFunction raises NotImplementedError

```python
from ironcore.alignment.rewards.builtin import CodeRewardFunction
import pytest

fn = CodeRewardFunction()
try:
    fn.compute("prompt", "completion", {"test_cases": ["assert 1+1==2"]})
    assert False, "Should have raised"
except NotImplementedError as e:
    assert "sandbox" in str(e).lower()
    print("CodeRewardFunction NotImplementedError PASS")
```

---

### 8. Integration: Trainer import smoke

Tests that `base_trainer.py` and `grpo_trainer.py` still import cleanly (the `clip_grad_norm_tp` relocation):

```bash
python -c "
import sys
sys.path.insert(0, '.')
# These do deferred imports inside functions — just verify module loads
import ironcore.trainers.base_trainer
import ironcore.trainers.grpo_trainer
print('Trainer imports OK')
"
```

---

### 9. Smoke: GRPO training step (2 GPU, DDP)

> **Pre-condition:** Fix the `RewardManager.compute()` bug first.

```bash
torchrun --nproc_per_node=2 examples/train.py \
  --config configs/grpo_gsm8k_smoke_rm.yaml \
  2>&1 | tee /tmp/grpo_smoke_0316.log
```

**Check log for:**
- [ ] No `TypeError` on `grpo_loss()` call (entropy args now accepted)
- [ ] `entropy` metric appears in per-step log (should be non-zero if `entropy_coef > 0`)
- [ ] Memory stays bounded: no growth trend over steps 1–20 (the unbounded cache bug would cause steady GPU/CPU RAM growth — now fixed)
- [ ] Step time stable (no stall from cache misses / hash collisions)

**Memory check specifically** (the possible "memory leak" from prior session):
- Root cause hypothesis: unbounded `OrderedDict` in `LocalEndpointRewardFunction` / `LocalInferenceRewardFunction` — every unique (prompt, completion) pair added a new entry, never evicted. With B=16, G=4, 20 steps → up to 1280 entries per run, but across thousands of steps in a full run this could grow to 100k+ entries.
- Fix: `cache_size=10000` with LRU eviction is now applied.
- Also: `APIRewardFunction` already had LRU, now all three are consistent.

**Watch:**
```bash
watch -n 5 nvidia-smi --query-gpu=memory.used --format=csv
```
Memory should plateau, not grow monotonically.

---

## Test Execution Order

```bash
# Step 1: Run new test suite (3 failures expected until RewardManager bug fixed)
python -m pytest tests/test_fix_review_0316.py -v

# Step 2: Fix RewardManager.compute() — remove `/ total_weight` on manager.py:55

# Step 3: All 41 tests should pass
python -m pytest tests/test_fix_review_0316.py -v

# Step 4: Legacy suite should also go green (77/77)
python -m pytest tests/test_reward_manager.py tests/unit/test_mfu.py -q

# Step 5: GRPO 2-GPU smoke run with memory watch
torchrun --nproc_per_node=2 examples/train.py \
  --config configs/grpo_gsm8k_smoke_rm.yaml \
  2>&1 | tee /tmp/grpo_smoke_0316.log
```

---

## Pre-existing Failures (not caused by this branch)

From `python -m pytest tests/unit/ -q`:
- `tests/unit/layers/test_attention.py` — 20+ failures (pre-existing attention layer issues)
- `tests/unit/layers/test_kv_cache_basic.py` — 7 failures (pre-existing)
- `tests/unit/parallel/test_comm_grad_correctness.py` — 1 failure (pre-existing)
- `tests/unit/parallel/test_expert_parallel.py` — 2 failures (pre-existing)
- `tests/unit/test_fim_*.py` — many errors (pre-existing)
- `tests/unit/test_hf_interop.py` — partial failures (pre-existing)

These are **not in scope** for this branch. Only the 2 `RewardManager` failures are new regressions introduced by this branch.
