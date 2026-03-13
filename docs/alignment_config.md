# Alignment Configuration

This document describes the configuration options for alignment training methods (DPO, GRPO) in IronCore.

## Overview

Alignment training configures how the model learns from preference data or reward signals. The configuration is defined in `ironcore/config/config_alignment.py` and can be specified via YAML files.

## Configuration Structure

```yaml
alignment:
  method: grpo  # "dpo" | "grpo"

  # DPO parameters (when method="dpo")
  dpo_beta: 0.5
  dpo_label_smoothing: 0.0

  # GRPO parameters (when method="grpo")
  grpo_group_size: 4
  grpo_beta: 0.1
  grpo_eps: 1e-8
  grpo_num_epochs: 1
  grpo_clip_eps: 0.2
  grpo_rollout_micro_group_size: 1

  # Generation config (GRPO only)
  generation:
    max_new_tokens: 512
    temperature: 1.0
    top_p: 0.9
    top_k: 0
    do_sample: true
    use_chat_template: false
    system_prompt: null

  # Reward config (GRPO only)
  reward:
    type: math
    num_workers: 4
    timeout: 30

  # Optimization
  concat_forward_passes: true
  metrics_interval: 0
```

---

## DPO Parameters

Direct Preference Optimization (DPO) trains the model directly on preference pairs without an explicit reward model.

### `dpo_beta`
- **Type**: `float`
- **Default**: `0.5`
- **Description**: KL divergence penalty coefficient. Higher values keep the policy closer to the reference model.
- **Range**: Must be positive

### `dpo_label_smoothing`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Label smoothing for the preference labels.
- **Range**: `[0.0, 1.0)`

---

## GRPO Parameters

Group Relative Policy Optimization (GRPO) generates multiple completions per prompt and optimizes based on group-relative advantages.

### `grpo_group_size`
- **Type**: `int`
- **Default**: `4`
- **Description**: Number of completions (G) generated per prompt. The advantage is computed relative to the group mean.
- **Range**: `>= 2`
- **Memory Impact**: Larger values require more memory for storing completions and log probabilities.

### `grpo_beta`
- **Type**: `float`
- **Default**: `0.1`
- **Description**: KL penalty coefficient. Controls how much the policy can deviate from the reference model.
- **Range**: `>= 0`

### `grpo_eps`
- **Type**: `float`
- **Default**: `1e-8`
- **Description**: Epsilon for advantage normalization stability.

### `grpo_num_epochs`
- **Type**: `int`
- **Default**: `1`
- **Description**: Number of gradient steps per rollout batch.
  - `1`: Online GRPO (generate, train once, discard)
  - `>1`: Offline/multi-epoch GRPO (reuse rollouts for multiple updates)
- **Range**: `>= 1`

### `grpo_clip_eps`
- **Type**: `float`
- **Default**: `0.2`
- **Description**: PPO-style importance sampling ratio clipping range.
  - `0.0`: No clipping (pure GRPO)
  - `>0`: Clip ratios outside `[1 - eps, 1 + eps]`
- **Range**: `>= 0`

### `grpo_rollout_micro_group_size`
- **Type**: `int`
- **Default**: `1`
- **Description**: Max completions generated in parallel per prompt per GPU (hardware knob, like `micro_batch_size`). Chunks are derived internally: `group_size / micro_group_size`.
  - `1`: Generate one completion at a time (minimum memory)
  - `group_size`: Generate all completions in one forward pass (maximum memory)
- **Range**: `>= 1`, must divide `grpo_group_size` evenly
- **Example**: `group_size=8, rollout_micro_group_size=2` → 4 chunks of 2 completions each

---

## Generation Configuration

Controls how completions are generated during GRPO rollouts.

### `max_new_tokens`
- **Type**: `int`
- **Default**: `512`
- **Description**: Maximum number of tokens to generate per completion.

### `temperature`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Sampling temperature. Higher values = more random.

### `top_p`
- **Type**: `float`
- **Default**: `0.9`
- **Description**: Nucleus sampling probability threshold.

### `top_k`
- **Type**: `int`
- **Default**: `0`
- **Description**: Top-k sampling. `0` = disabled.

### `do_sample`
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to sample or use greedy decoding.

### `use_chat_template`
- **Type**: `bool`
- **Default**: `false`
- **Description**: Apply the model's chat template to prompts.

### `system_prompt`
- **Type**: `str | None`
- **Default**: `null`
- **Description**: System prompt prepended to all inputs.

---

## Reward Configuration

Configures how rewards are computed for GRPO.

### `type`
- **Type**: `str`
- **Default**: `"math"`
- **Options**: `"math"`, `"code"`, `"api"`, `"local_endpoint"`, `"local_inference"`, `"format"`, `"keyword"`, `"soft_keyword"`

### `num_workers`
- **Type**: `int`
- **Default**: `4`
- **Description**: Number of parallel workers for reward computation.

### `timeout`
- **Type**: `int`
- **Default**: `30`
- **Description**: Timeout in seconds for reward computation.

### API Reward Options (when `type="api"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_provider` | str | `"openai"` | API provider: `"openai"`, `"anthropic"`, `"google"`, `"zhipu"` |
| `api_model` | str \| None | `null` | Model name to use |
| `api_key_env` | str \| None | `null` | Environment variable name for API key |
| `api_endpoint` | str \| None | `null` | Custom API endpoint |
| `prompt_template` | str | `"default"` | Prompt template: `"default"`, `"math"`, `"code"`, `"reasoning"` |
| `custom_prompt` | str \| None | `null` | Custom prompt template |
| `max_retries` | int | `3` | Maximum retry attempts |
| `cache_size` | int | `10000` | Response cache size |
| `rate_limit_delay` | float | `0.1` | Delay between requests |

### Local Endpoint Options (when `type="local_endpoint"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `local_endpoint` | str | `"http://localhost:8000/v1"` | Local API endpoint URL |

### Local Inference Options (when `type="local_inference"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `local_model_path` | str \| None | `null` | Path to local model |
| `local_device` | str | `"cuda:0"` | Device for inference |
| `local_dtype` | str | `"bfloat16"` | Data type: `"float32"`, `"float16"`, `"bfloat16"` |
| `load_in_8bit` | bool | `false` | Load model in 8-bit quantization |
| `load_in_4bit` | bool | `false` | Load model in 4-bit quantization |

### Format Reward Options (when `type="format"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `required_tags` | list[str] \| None | `null` | Required XML-style tags |
| `format_penalty` | float | `-0.1` | Penalty for missing tags |

### Keyword Reward Options (when `type="keyword"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keyword` | str | `"ironcore"` | Keyword to search for |
| `keyword_case_sensitive` | bool | `false` | Case-sensitive matching |

---

## Optimization Flags

### `concat_forward_passes`
- **Type**: `bool`
- **Default**: `true`
- **Description**: Concatenate policy and reference forward passes for efficiency.

### `metrics_interval`
- **Type**: `int`
- **Default**: `0`
- **Description**: Interval for computing detailed metrics.
  - `0`: Compute every step
  - `>0`: Compute every N steps

---

## Example Configurations

### DPO Training

```yaml
alignment:
  method: dpo
  dpo_beta: 0.1
  dpo_label_smoothing: 0.0
```

### GRPO for Math Reasoning (24GB GPUs)

```yaml
alignment:
  method: grpo
  grpo_group_size: 4
  grpo_beta: 0.04
  grpo_rollout_micro_group_size: 1
  generation:
    max_new_tokens: 128
    temperature: 0.7
    top_p: 0.95
    use_chat_template: true
    system_prompt: "Solve step by step."
  reward:
    type: math
    num_workers: 4
```

### GRPO with Chunked Rollouts (Memory-Constrained)

```yaml
alignment:
  method: grpo
  grpo_group_size: 8
  grpo_rollout_micro_group_size: 2  # Generate 2 per chunk, 4 chunks
  generation:
    max_new_tokens: 256
```
