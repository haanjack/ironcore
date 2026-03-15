# Migration Plan: Legacy RewardConfig to RewardManager

## Overview

Migrate existing training configs from the legacy `reward:` syntax to the new `reward_manager:` syntax. The legacy path remains functional via `RewardManager.from_legacy_config()`, so migration is **not urgent** but recommended for new configs.

## What Changes

| Before (legacy) | After (new) |
|---|---|
| `reward.type: math` | `reward_manager.functions[0].type: rule_template` + YAML |
| `reward.type: composite_math` | Two `rule_template` entries with weights |
| `reward.type: format` | `rule_template` with `format_cot.yaml` |
| `reward.type: strict_format` | `rule_template` with `format_deepseek.yaml` |
| `reward.type: keyword` | `type: keyword` (unchanged, just nested under `functions`) |
| `reward.type: api` | `type: api` (unchanged, just nested under `functions`) |
| `reward.type: code` | `type: code` (unchanged, just nested under `functions`) |

## Step-by-Step Migration

### Step 1: Identify all configs using legacy `reward:` block

```bash
grep -rn "reward:" configs/ --include="*.yaml" | grep -v "reward_manager"
```

Current configs to migrate:
- `configs/grpo_gsm8k_smoke.yaml`
- `configs/grpo_gsm8k.yaml`
- `configs/grpo_gsm8k_fsdp.yaml`
- `configs/grpo_gsm8k_smoke_fsdp.yaml`
- `configs/grpo_gsm8k_smoke_muon.yaml`
- `configs/grpo_keyword_toy.yaml`
- `configs/grpo_toy.yaml`
- `configs/grpo_verify.yaml`

### Step 2: Migrate each config type

#### 2a. `type: math` configs

Before:
```yaml
alignment:
  reward:
    type: math
    num_workers: 4
    timeout: 30
```

After:
```yaml
alignment:
  reward_manager:
    num_workers: 4
    timeout: 30
    functions:
      - name: correctness
        type: rule_template
        weight: 1.0
        rule_template: configs/rewards/math_gsm8k.yaml
```

#### 2b. `type: composite_math` configs

Before:
```yaml
alignment:
  reward:
    type: composite_math
    num_workers: 4
    timeout: 30
```

After:
```yaml
alignment:
  reward_manager:
    num_workers: 4
    timeout: 30
    functions:
      - name: correctness
        type: rule_template
        weight: 0.8
        rule_template: configs/rewards/math_gsm8k.yaml
      - name: format
        type: rule_template
        weight: 0.2
        rule_template: configs/rewards/format_deepseek.yaml
```

Note: `composite_math` hardcoded `format_weight=0.2` and `correctness_weight=0.8`. The YAML version makes these explicit and editable.

#### 2c. `type: keyword` / `type: soft_keyword` configs

Before:
```yaml
alignment:
  reward:
    type: keyword
    keyword: ironcore
    num_workers: 4
```

After:
```yaml
alignment:
  reward_manager:
    num_workers: 4
    functions:
      - name: keyword_check
        type: keyword
        weight: 1.0
        keyword: ironcore
```

#### 2d. `type: format` configs

Before:
```yaml
alignment:
  reward:
    type: format
    required_tags: ["<thought>", "</thought>", "<answer>", "</answer>"]
    format_penalty: -0.1
```

After:
```yaml
alignment:
  reward_manager:
    functions:
      - name: format
        type: rule_template
        weight: 1.0
        rule_template: configs/rewards/format_cot.yaml
```

If custom tags were used, create a new YAML template with those tags.

#### 2e. `type: api` / `type: local_endpoint` / `type: local_inference` configs

Before:
```yaml
alignment:
  reward:
    type: api
    api_provider: openai
    api_model: gpt-4o-mini
```

After:
```yaml
alignment:
  reward_manager:
    functions:
      - name: llm_judge
        type: api
        weight: 1.0
        api_provider: openai
        api_model: gpt-4o-mini
```

### Step 3: Validate each migrated config

For each migrated config:

1. **Dry run**: Instantiate the config and verify it parses
   ```python
   from ironcore.config.config_alignment import AlignmentConfig
   cfg = AlignmentConfig.from_yaml("configs/grpo_gsm8k_smoke.yaml")
   assert cfg.reward_manager is not None
   ```

2. **Reward sanity check**: Build the RewardManager and test with known inputs
   ```python
   from ironcore.alignment.reward_manager import RewardManager
   mgr = RewardManager.from_config(cfg.reward_manager)
   score = mgr.compute("What is 2+2?", "#### 4", {"answer": "4"})
   assert score > 0.9
   ```

3. **Smoke training run**: Run 5-10 steps and compare reward distribution to legacy
   ```bash
   torchrun --nproc_per_node=1 -m ironcore.main --config configs/grpo_gsm8k_smoke.yaml
   ```

### Step 4: Remove legacy `reward:` block from migrated configs

After validation, remove the `reward:` block from each config. The `reward:` field defaults to `RewardConfig()` which is ignored when `reward_manager` is set.

### Step 5 (future): Deprecation cleanup

Once all configs are migrated and validated:

1. Remove `from_legacy_config()` from `RewardManager`
2. Remove deprecation warnings from `get_reward_function()` and `CompositeRewardFunction`
3. Consider removing `RewardConfig` from `AlignmentConfig` (breaking change, needs major version bump)

This step is **not recommended now** -- keep backward compat until all experiment configs (including any user forks) have migrated.

## Implementation Checklist

- [ ] Migrate `configs/grpo_gsm8k_smoke.yaml`
- [ ] Migrate `configs/grpo_gsm8k.yaml`
- [ ] Migrate `configs/grpo_gsm8k_fsdp.yaml`
- [ ] Migrate `configs/grpo_gsm8k_smoke_fsdp.yaml`
- [ ] Migrate `configs/grpo_gsm8k_smoke_muon.yaml`
- [ ] Migrate `configs/grpo_keyword_toy.yaml`
- [ ] Migrate `configs/grpo_toy.yaml`
- [ ] Migrate `configs/grpo_verify.yaml`
- [ ] Smoke test each migrated config (5-10 steps)
- [ ] Verify reward distributions match legacy output
- [ ] Remove legacy `reward:` blocks from migrated configs

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Reward score divergence from legacy | Low | `from_legacy_config` uses identical code paths; YAML templates match hardcoded patterns |
| YAML template path resolution | Medium | Use relative paths from working directory; document that paths are relative to CWD |
| Config parsing failure on `list[RewardFunctionEntry]` | Low | `RewardManagerConfig.__post_init__` handles dict→dataclass conversion |
| Breaking existing experiment runs | None | Legacy path fully preserved; no forced migration |
