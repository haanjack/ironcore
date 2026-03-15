# Test Plan: RewardManager Architecture

## 1. Unit Tests

### 1.1 TemplateRuleReward (`ironcore/alignment/reward_rules.py`)

#### `answer_match` mode
| # | Test Case | Input | Expected |
|---|---|---|---|
| 1 | Exact match via `####` pattern | completion=`"#### 42"`, answer=`"42"` | `1.0` |
| 2 | Exact match via `\boxed{}` pattern | completion=`"\\boxed{42}"`, answer=`"42"` | `1.0` |
| 3 | Exact match via `Answer:` pattern | completion=`"Answer: 42"`, answer=`"42"` | `1.0` |
| 4 | Exact match via `The answer is` pattern | completion=`"The answer is 42"`, answer=`"42"` | `1.0` |
| 5 | Wrong answer extracted | completion=`"#### 99"`, answer=`"42"` | `0.1` (partial) |
| 6 | No answer extracted (strict) | completion=`"no numbers"`, answer=`"42"`, `fallback_last_number: false` | `0.0` |
| 7 | Fallback last number | completion=`"I think 42 works"`, answer=`"42"`, `fallback_last_number: true` | `1.0` |
| 8 | No ground truth | completion=`"#### 42"`, answer=`""` | `0.5` |
| 9 | Normalization: case insensitive | completion=`"#### YES"`, answer=`"yes"` | `1.0` |
| 10 | Normalization: strip chars | completion=`"#### $1,000"`, answer=`"1000"` | `1.0` |
| 11 | Normalization: trailing period | completion=`"#### 42."`, answer=`"42"` | `1.0` |

#### `tag_check` mode
| # | Test Case | Input | Expected |
|---|---|---|---|
| 12 | All tags present | `"<thought>x</thought><answer>y</answer>"` | `0.0` (all_present) |
| 13 | All tags missing | `"plain text"` | `-0.4` (4 tags * -0.1) |
| 14 | Partial tags | `"<thought>x</thought> no answer tags"` | `-0.2` (2 missing * -0.1) |
| 15 | Custom scoring | `all_present: 0.5`, `per_missing_tag: -0.25` | Uses custom values |

#### `regex_match` mode
| # | Test Case | Input | Expected |
|---|---|---|---|
| 16 | Pattern matches | `"<think>reasoning</think> #### 42"` | `1.0` |
| 17 | Pattern doesn't match | `"just text"` | `0.0` |
| 18 | DOTALL flag works | `"<think>\nmultiline\n</think> #### 42"` | `1.0` |
| 19 | Custom scoring | `match: 0.5, no_match: -1.0` | Uses custom values |

#### Edge cases
| # | Test Case | Expected |
|---|---|---|
| 20 | Missing `mode` field in YAML | `ValueError` raised |
| 21 | Unknown mode | `ValueError` raised |
| 22 | Empty completion | Graceful handling (no crash) |
| 23 | `from_yaml` with nonexistent path | `FileNotFoundError` |

### 1.2 RewardModelFunction (`ironcore/alignment/reward_model.py`)

| # | Test Case | Expected |
|---|---|---|
| 24 | Init with unknown backend | `ValueError` |
| 25 | `local_inference` without model_path | `ValueError` |
| 26 | `local_endpoint` compute (mocked HTTP) | Returns parsed scalar from `{"reward": 0.8}` |
| 27 | `local_endpoint` retry on failure | Retries `max_retries` times, returns `0.5` on exhaustion |
| 28 | `local_endpoint` supports `{"score": float}` format | Returns parsed scalar |
| 29 | `api` compute (mocked OpenAI client) | Returns parsed scalar |
| 30 | `local_inference` compute (mocked model) | Returns `logits[0,0]` value |

### 1.3 RewardManager (`ironcore/alignment/reward_manager.py`)

| # | Test Case | Expected |
|---|---|---|
| 31 | Register single function, compute | Returns function's score * weight |
| 32 | Register multiple functions, compute | Returns weighted sum |
| 33 | No functions registered | `RuntimeError` on compute |
| 34 | `from_config` with `rule_template` type | Loads YAML, registers correctly |
| 35 | `from_config` with `reward_model` type | Creates `RewardModelFunction` |
| 36 | `from_config` with legacy type (`math`) | Delegates to `get_reward_function()` |
| 37 | `from_config` with missing `rule_template` path | `ValueError` |
| 38 | `from_legacy_config` with `type=math` | Creates `MathRewardFunction` via legacy path |
| 39 | `from_legacy_config` with `type=composite_math` | Creates `CompositeRewardFunction` |
| 40 | `from_legacy_config` with `type=keyword` | Creates `KeywordRewardFunction` with correct kwargs |

### 1.4 Config Dataclasses (`ironcore/config/config_alignment.py`)

| # | Test Case | Expected |
|---|---|---|
| 41 | `RewardManagerConfig` with dict functions | `__post_init__` converts to `RewardFunctionEntry` |
| 42 | `RewardManagerConfig` with `RewardFunctionEntry` list | Passes through unchanged |
| 43 | `AlignmentConfig` with `reward_manager` dict | `__post_init__` converts to `RewardManagerConfig` |
| 44 | `AlignmentConfig(method='grpo')` without reward_manager | `reward.type` validation runs |
| 45 | `AlignmentConfig(method='grpo', reward_manager=...)` | `reward.type` validation skipped |
| 46 | `AlignmentConfig.from_yaml` with `reward_manager` key | Parses correctly |
| 47 | `RewardFunctionEntry` default values | All defaults sensible |

---

## 2. Integration Tests

### 2.1 Backward Compatibility

| # | Test Case | Method | Expected |
|---|---|---|---|
| 48 | Existing `grpo_gsm8k_smoke.yaml` loads and runs | Load config, build RewardManager via legacy path, score known input | Same scores as before refactor |
| 49 | Legacy `type: math` produces identical scores | Compare `MathRewardFunction` directly vs `TemplateRuleReward` with `math_gsm8k.yaml` on 20 test cases | Identical results |
| 50 | Legacy `type: composite_math` produces identical scores | Compare `CompositeRewardFunction` vs two-entry RewardManager | Identical results |

### 2.2 New Config Path

| # | Test Case | Method | Expected |
|---|---|---|---|
| 51 | Config with `reward_manager` + two `rule_template` entries | Load config, build RewardManager, score batch | Weighted sum of both functions |
| 52 | Config with mixed types (`rule_template` + `code`) | Build RewardManager | Both functions registered and callable |
| 53 | Custom YAML template (copy + modify `math_gsm8k.yaml`) | Modify patterns, verify modified pattern is used | Uses new patterns |

### 2.3 GRPOTrainer Integration

| # | Test Case | Method | Expected |
|---|---|---|---|
| 54 | `_post_checkpoint_load` with `reward_manager` config | Mock model, verify `RewardManager.from_config` called | `reward_worker` initialized correctly |
| 55 | `_post_checkpoint_load` with legacy config | Mock model, verify `RewardManager.from_legacy_config` called | `reward_worker` initialized correctly |
| 56 | `RewardWorkerPool.score_batch` with `RewardManager` | Create pool with manager, score batch of 8 | Returns tensor of shape `[8]` |

---

## 3. YAML Template Tests

| # | Test Case | Method | Expected |
|---|---|---|---|
| 57 | `configs/rewards/math_gsm8k.yaml` loads | `TemplateRuleReward.from_yaml()` | No errors, mode=`answer_match` |
| 58 | `configs/rewards/format_cot.yaml` loads | `TemplateRuleReward.from_yaml()` | No errors, mode=`tag_check` |
| 59 | `configs/rewards/format_deepseek.yaml` loads | `TemplateRuleReward.from_yaml()` | No errors, mode=`regex_match` |
| 60 | All YAML files are valid YAML | `yaml.safe_load()` on each | No parse errors |

---

## 4. Deprecation Warning Tests

| # | Test Case | Expected |
|---|---|---|
| 61 | Calling `get_reward_function()` | Emits `DeprecationWarning` |
| 62 | Constructing `CompositeRewardFunction` | Emits `DeprecationWarning` |
| 63 | Using `RewardManager.from_legacy_config()` | Emits warning (via `get_reward_function`) |
| 64 | Using `RewardManager.from_config()` with `rule_template` | No deprecation warning |

---

## 5. Smoke / E2E Tests

| # | Test Case | Config(s) | Method | Expected |
|---|---|---|---|---|
| 65 | 20-step GRPO with new-style `reward_manager` config | `configs/grpo_gsm8k_smoke_rm.yaml` | `torchrun` 2-GPU | Runs without error, rewards logged |
| 66 | 20-step GRPO with legacy `reward` config (regression) | `configs/grpo_gsm8k_smoke_fsdp.yaml` | `torchrun` 2-GPU | Runs without error |
| 67 | RewardManager produces equivalent rewards to legacy path | `grpo_gsm8k_smoke_composite.yaml` vs `grpo_gsm8k_smoke_rm_composite.yaml` | Both use `composite_math`; compare mean_reward | Within 5% relative tolerance. Results recorded as Run 7 in `grpo_test_1303.md`. |

**Configs created for E2E tests:**
- `configs/grpo_gsm8k_smoke_rm.yaml` — new-style multi-function reward_manager (correctness 0.8 + format 0.2)
- `configs/grpo_gsm8k_smoke_composite.yaml` — smoke version of `grpo_gsm8k.yaml` (composite_math, legacy path)
- `configs/grpo_gsm8k_smoke_rm_composite.yaml` — same as above but via reward_manager (composite_math weight 1.0)
- `configs/grpo_gsm8k_smoke_rm_math.yaml` — single math reward via reward_manager (used for initial test 67 validation)

---

## 6. Error Handling Tests

| # | Test Case | Expected |
|---|---|---|
| 68 | `rule_template` path doesn't exist | Clear `FileNotFoundError` with path in message |
| 69 | YAML template has invalid mode | `ValueError` with mode name |
| 70 | `reward_manager.functions` is empty list | `RuntimeError` on first `compute()` call |
| 71 | `RewardModelFunction` endpoint unreachable | Returns `0.5` after retries (graceful degradation) |
| 72 | Malformed YAML template (invalid regex) | `re.error` at template load time |

---

## Test Priority

| Priority | Tests | Rationale |
|---|---|---|
| P0 (must pass) | 1-11, 12-19, 31-40, 48-50, 57-60, 65-66 | Core functionality and backward compat |
| P1 (should pass) | 20-23, 41-47, 51-56, 61-64, 67-72 | Edge cases, config loading, error handling |
| P2 (nice to have) | 24-30 | RewardModelFunction backends (require mocking or live endpoints) |

## Implementation Notes

- Unit tests go in `tests/test_reward_manager.py`
- E2E tests (65-67) go in `tests/test_e2e_grpo_smoke.py`, marked `@pytest.mark.e2e`
- Use `pytest.warns(DeprecationWarning)` for deprecation tests
- Mock HTTP calls for `RewardModelFunction` endpoint tests
- For backward compat tests (48-50), compute scores on a fixed set of (prompt, completion, answer) tuples and assert exact equality

## Bugs Found During Testing

### `_convert_lists_to_dict` in `ironcore/utils.py` (found during test 65)

`load_yaml_config()` calls `_convert_lists_to_dict()` which converts any list of dicts with non-overlapping keys into a merged dict. This destroyed the `reward_manager.functions` list:
- Multi-entry: `[{name: correctness, ...}, {name: format, ...}]` → `{name: format, ...}` (last entry wins)
- Single-entry: `[{name: math, ...}]` → `{name: math, ...}` (always merged for single items)

**Fix:** Require both `len(data) > 1` AND no overlapping keys for the merge to happen. Single-item lists and lists with shared keys are returned as-is. This preserves the dataset-config merge behavior while keeping reward function lists intact.
