# RewardManager & YAML Rule Templates

## Architecture

```
AlignmentConfig
  ├── reward: RewardConfig          # Legacy (deprecated, still works)
  └── reward_manager: RewardManagerConfig   # New (takes precedence)
        └── functions: list[RewardFunctionEntry]
              ├── type: "rule_template"  → TemplateRuleReward.from_yaml(path)
              ├── type: "reward_model"   → RewardModelFunction(backend=...)
              └── type: "math"|"code"|…  → get_reward_function() (legacy classes)
```

### Component Roles

| Component | File | Role |
|---|---|---|
| `RewardManager` | `ironcore/alignment/reward_manager.py` | Weighted registry/orchestrator. Extends `RewardFunction` so `RewardWorkerPool` needs zero changes. |
| `TemplateRuleReward` | `ironcore/alignment/reward_rules.py` | YAML-driven rule reward. Replaces `MathRewardFunction`, `FormatRewardFunction`, `StrictFormatRewardFunction` with config-driven logic. |
| `RewardModelFunction` | `ironcore/alignment/reward_model.py` | Classifier-head RM scoring (scalar output). Backends: `local_endpoint`, `api`, `local_inference`. |
| `RewardFunctionEntry` | `ironcore/config/config_alignment.py` | Config dataclass for a single reward function entry. |
| `RewardManagerConfig` | `ironcore/config/config_alignment.py` | Config dataclass holding the list of entries + worker settings. |

### How It Plugs In

```
GRPOTrainer._post_checkpoint_load()
  │
  ├─ reward_manager config set?
  │   YES → RewardManager.from_config(reward_manager_cfg)
  │   NO  → RewardManager.from_legacy_config(reward_cfg)  # backward compat
  │
  └─ RewardWorkerPool(reward_fn=manager, ...)
       └─ manager.compute() called per (prompt, completion, metadata)
            └─ weighted sum of all registered functions
```

`RewardManager` extends `RewardFunction`, so the rest of the training loop is unchanged.

---

## Config Reference

### RewardManagerConfig

```yaml
alignment:
  method: grpo
  reward_manager:
    num_workers: 4        # Thread pool size for parallel reward computation
    timeout: 30           # Seconds before returning default reward (0.5)
    functions:            # List of RewardFunctionEntry
      - name: correctness
        type: rule_template
        weight: 0.6
        rule_template: configs/rewards/math_gsm8k.yaml
      - name: format
        type: rule_template
        weight: 0.4
        rule_template: configs/rewards/format_deepseek.yaml
```

### RewardFunctionEntry Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | str | `"default"` | Human-readable name (used in logging) |
| `type` | str | `"rule_template"` | `"rule_template"` \| `"reward_model"` \| `"math"` \| `"code"` \| `"api"` \| `"format"` \| ... |
| `weight` | float | `1.0` | Weight in the weighted sum |
| `rule_template` | str \| None | `None` | Path to YAML rule file (when `type="rule_template"`) |
| `rm_backend` | str | `"local_endpoint"` | Reward model backend (when `type="reward_model"`): `"local_endpoint"` \| `"api"` \| `"local_inference"` |
| `api_provider` | str | `"openai"` | API provider for `type="api"` or `rm_backend="api"` |
| `api_model` | str \| None | `None` | Model name for API calls |
| `local_endpoint` | str | `"http://localhost:8000/v1"` | URL for local endpoint |
| `local_model_path` | str \| None | `None` | HF model path for local inference |
| `local_device` | str | `"cuda:0"` | Device for local inference |
| `local_dtype` | str | `"bfloat16"` | Dtype for local inference |
| `keyword` | str | `""` | Keyword for `type="keyword"` or `type="soft_keyword"` |

---

## YAML Rule Templates

Rule templates live in `configs/rewards/` and are loaded by `TemplateRuleReward.from_yaml()`.

### Mode: `answer_match`

Extracts answer from completion using regex patterns, normalizes, compares to ground truth.

```yaml
# configs/rewards/math_gsm8k.yaml
mode: answer_match
answer_patterns:
  - '####\s*(.+)'
  - '\\boxed\{(.+?)\}'
  - '[Aa]nswer:\s*(.+)'
  - '[Tt]he answer is\s*(.+)'
normalization:
  lowercase: true
  strip_chars: ", _$"
  strip_trailing_period: true
fallback_last_number: true     # If no pattern matches, use last number in text
scoring:
  correct: 1.0                 # Exact match after normalization
  partial: 0.1                 # Extracted something but wrong
  no_answer: 0.0               # Failed to extract any answer
```

### Mode: `tag_check`

Checks for presence of required tags. Useful for enforcing structured output.

```yaml
# configs/rewards/format_cot.yaml
mode: tag_check
required_tags:
  - "<thought>"
  - "</thought>"
  - "<answer>"
  - "</answer>"
scoring:
  all_present: 0.0             # Score when all tags found
  per_missing_tag: -0.1        # Penalty per missing tag
```

### Mode: `regex_match`

Binary reward based on full regex match. Useful for strict format enforcement.

```yaml
# configs/rewards/format_deepseek.yaml
mode: regex_match
pattern: '<think>.*?</think>\s*####\s*.*'
pattern_flags:
  - DOTALL                     # . matches newlines
scoring:
  match: 1.0
  no_match: 0.0
```

---

## Usage Examples

### Example 1: GSM8K with format enforcement (common setup)

```yaml
alignment:
  method: grpo
  grpo_group_size: 8
  reward_manager:
    num_workers: 4
    timeout: 30
    functions:
      - name: correctness
        type: rule_template
        weight: 0.6
        rule_template: configs/rewards/math_gsm8k.yaml
      - name: format
        type: rule_template
        weight: 0.4
        rule_template: configs/rewards/format_deepseek.yaml
```

### Example 2: Reward model + rule-based format check

```yaml
alignment:
  method: grpo
  reward_manager:
    functions:
      - name: helpfulness_rm
        type: reward_model
        weight: 0.5
        rm_backend: local_endpoint
        local_endpoint: http://localhost:8000/v1
      - name: format
        type: rule_template
        weight: 0.3
        rule_template: configs/rewards/format_cot.yaml
      - name: correctness
        type: rule_template
        weight: 0.2
        rule_template: configs/rewards/math_gsm8k.yaml
```

### Example 3: Mixed legacy + new types

```yaml
alignment:
  method: grpo
  reward_manager:
    functions:
      - name: code_exec
        type: code
        weight: 0.7
      - name: format
        type: rule_template
        weight: 0.3
        rule_template: configs/rewards/format_cot.yaml
```

### Example 4: Custom template (copy and modify)

```bash
cp configs/rewards/math_gsm8k.yaml configs/rewards/math_custom.yaml
# Edit math_custom.yaml: add domain-specific patterns, change scoring
```

```yaml
alignment:
  method: grpo
  reward_manager:
    functions:
      - name: correctness
        type: rule_template
        weight: 1.0
        rule_template: configs/rewards/math_custom.yaml
```

### Example 5: Legacy config (still works, no changes needed)

```yaml
alignment:
  method: grpo
  reward:
    type: composite_math
    num_workers: 4
    timeout: 30
```

---

## Creating a Custom Rule Template

1. Choose a mode: `answer_match`, `tag_check`, or `regex_match`
2. Create a YAML file in `configs/rewards/`
3. Reference it in your training config

Example for a custom answer extraction:

```yaml
# configs/rewards/science_answer.yaml
mode: answer_match
answer_patterns:
  - 'Final answer:\s*(.+)'
  - 'Result:\s*(.+)'
  - '=\s*(.+?)(?:\s|$)'
normalization:
  lowercase: true
  strip_chars: " "
  strip_trailing_period: true
fallback_last_number: false
scoring:
  correct: 1.0
  partial: 0.0
  no_answer: 0.0
```

---

## RewardModelFunction Backends

| Backend | What it does | When to use |
|---|---|---|
| `local_endpoint` | POSTs to `{endpoint}/reward`, expects `{"reward": float}` or `{"score": float}` | vLLM/SGLang reward endpoint running locally |
| `api` | Sends prompt+completion as chat messages, parses scalar from response | OpenAI-compatible reward API |
| `local_inference` | Loads `AutoModelForSequenceClassification`, reads reward head output | HF reward model on GPU (no server needed) |
