# GRPO Implementation Plan

## Status (as of 2026-03-11)

**GRPO is fully implemented.** All core components below have been built and are in production on `feature/grpo`.

Completed:
- [x] KL divergence utilities (`alignment/loss/kl.py`)
- [x] GRPO loss + advantage computation (`alignment/loss/grpo.py`)
- [x] Batched rollout generation with prefix KV cache (`alignment/rollout.py`)
- [x] Rollout buffer (`alignment/buffer.py`)
- [x] GRPO dataset / data iterator (`alignment/dataset.py`)
- [x] Reward functions: math, code, API (multi-provider), local endpoint, local model, format, keyword, soft_keyword (`alignment/rewards.py`)
- [x] Reward worker pool (`alignment/rewards.py`)
- [x] `GRPOTrainer` with online rollout loop, multi-epoch support, IS clipping (`trainers/grpo_trainer.py`)
- [x] Alignment config: `AlignmentConfig`, `GenerationConfig`, `RewardConfig` (`config/config_alignment.py`)
- [x] GSM8K training configs (`configs/grpo_gsm8k.yaml`, `configs/data/grpo_gsm8k.yaml`)
- [x] Chat template + system prompt support for rollout generation
- [x] HF checkpoint fix for LLaMA weight transpose (`checkpointing/hf_interop.py`)

Remaining / out of scope for Phase 1:
- [ ] PPO (separate algorithm)
- [ ] Full vLLM-style inference engine
- [ ] Async TP / Pipeline Parallelism

---

## Context
Implement GRPO (Group Relative Policy Optimization) for Phase 1 completion of ironcore. The kvcache + generate() infrastructure is already merged, providing the foundation for rollout generation.

## Decisions (resolved)
- **Dataset format**: Support both verifiable (math/code) AND reward model formats
- **Prefix caching**: Implement proper prefix KV-cache sharing NOW
- **Batch layout**: Flatten to [B×G] completions
- **Reward computation**: Spawn separate worker processes (safer, can use other inference engines/models)
- **Memory optimization**: TP, activation recompute, FSDP

## Critical Optimizations (from technical review)

### 1. Batched Rollout Generation (CRITICAL)
**Problem**: Nested loops for B prompts × G samples = sequential bottleneck
**Solution**: Expand KV-cache to [B×G] after prefill, generate all at once

```
Before: for b in B: for g in G: generate()  # Slow!
After:  prefill(B) → expand_kv(B×G) → generate_batch(B×G)  # Fast!
```

### 2. Distributed Advantage Computation
**Problem**: Groups may be split across DP/FSDP ranks → wrong normalization
**Solution**: All-gather rewards before group normalization, or constrain G to single rank

### 3. Reference Model Memory Efficiency
**Problem**: Two full models = OOM risk
**Solutions**:
- CPU offload reference model during generation
- FSDP with separate process group for reference
- Share underlying weights with reference (advanced)

### 4. Reward Caching
**Problem**: API calls expensive, repeated completions waste money
**Solution**: LRU cache for (prompt_hash, completion_hash) → reward

### 5. Format Reward Function
**Problem**: Reasoning models need <thought>/<answer> structure
**Solution**: Add FormatRewardFunction that penalizes missing tags

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GRPO Training Loop                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌───────────────────┐    ┌─────────────────────┐   │
│  │ 1. Sample    │───▶│ 2. Rollout        │───▶│ 3. Reward           │   │
│  │    Prompts   │    │    (prefix cache) │    │    (worker pool)    │   │
│  └──────────────┘    └───────────────────┘    └─────────────────────┘   │
│         │                     │                        │                 │
│         │                     ▼                        ▼                 │
│         │            ┌───────────────────┐    ┌─────────────────────┐   │
│         │            │ completions [B*G] │    │ rewards [B*G]       │   │
│         │            │ log_probs [B*G]   │    │                     │   │
│         │            └───────────────────┘    └─────────────────────┘   │
│         │                        │                        │              │
│         │                        └──────────┬─────────────┘              │
│         ▼                                   ▼                            │
│  ┌──────────────┐    ┌───────────────────────────────────────────────┐  │
│  │ group_ids    │───▶│ 4. Compute Advantages (group-normalized)      │  │
│  │ [0,0,0,0,    │    │    A = (R - mean_group) / std_group            │  │
│  │  1,1,1,1]    │    └───────────────────────────────────────────────┘  │
│  └──────────────┘                              │                         │
│                                                ▼                         │
│         ┌──────────────────────────────────────────────────────────┐    │
│         │ 5. GRPO Loss: L = -mean(A * log_prob) + β * mean(KL)     │    │
│         │    - Policy log probs (current model, with grad)          │    │
│         │    - Reference log probs (frozen model, no grad)          │    │
│         │    - KL divergence per token                              │    │
│         └──────────────────────────────────────────────────────────┘    │
│                                                │                         │
│                                                ▼                         │
│                                    ┌───────────────────┐                 │
│                                    │ 6. Backward+Step  │                 │
│                                    └───────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: KL Divergence Utilities

**File**: `ironcore/alignment/loss/kl.py`

**Purpose**: Compute KL divergence between policy and reference distributions for the KL penalty term.

**Key Functions**:

```python
def kl_divergence(
    policy_log_probs: torch.Tensor,  # [batch, seq_len, vocab]
    ref_log_probs: torch.Tensor,     # [batch, seq_len, vocab]
    mask: torch.Tensor | None = None,  # [batch, seq_len] 1=valid, 0=pad/prompt
) -> torch.Tensor:
    """
    Compute KL(ref || policy) summed over valid tokens.

    KL(P||Q) = sum_x P(x) * (log P(x) - log Q(x))

    For numerical stability with log probs:
    KL = sum_x exp(ref_log_prob(x)) * (ref_log_prob(x) - policy_log_prob(x))

    Returns:
        [batch] - KL divergence per sequence
    """
    # Per-token KL: [batch, seq_len]
    kl_per_token = (ref_log_probs.exp() * (ref_log_probs - policy_log_probs)).sum(dim=-1)

    if mask is not None:
        kl_per_token = kl_per_token * mask

    return kl_per_token.sum(dim=-1)  # [batch]


def compute_sequence_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,        # [batch, seq_len]
    labels: torch.Tensor,            # [batch, seq_len] with -100 for ignore
    response_mask: torch.Tensor,     # [batch, seq_len] 1=response, 0=prompt
) -> torch.Tensor:
    """
    Compute sequence-level log probabilities for response tokens only.

    Reuses DPO's _compute_log_softmax_tp_safe() for TP compatibility.

    Returns:
        [batch] - sum of log probs over response tokens
    """
    logits = model(input_ids, labels=None)  # [batch, seq_len, vocab/tp]
    log_probs = _compute_log_softmax_tp_safe(logits)  # [batch, seq_len, full_vocab]

    # Extract log probs for labels, mask out non-response tokens
    return _extract_logps_from_log_probs(log_probs, labels, response_mask)
```

**Dependencies**:
- Reuses `_compute_log_softmax_tp_safe()` from `ironcore/alignment/loss/dpo.py`
- Reuses `_extract_logps_from_log_probs()` from `ironcore/alignment/loss/dpo.py`

---

## Component 2: Advantage Normalization + GRPO Loss

**File**: `ironcore/alignment/loss/grpo.py`

**Purpose**: Group-relative advantage computation and GRPO loss formula.

### Advantage Normalization

**Critical**: In distributed settings (DP/FSDP), groups may be split across ranks.
We must all-gather rewards before computing advantages.

```python
def compute_advantages(
    rewards: torch.Tensor,     # [B*G] flat tensor of rewards
    group_ids: torch.Tensor,   # [B*G] group index for each completion
    eps: float = 1e-8,
    distributed: bool = True,  # Set False for single-GPU
) -> torch.Tensor:
    """
    Compute group-normalized advantages with distributed support.

    For GRPO, we normalize rewards within each group (prompt):
        A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

    In distributed settings, we all-gather rewards first to ensure
    all samples in a group are normalized together.

    Edge cases:
    - If all rewards in a group are equal (std < eps), advantages = 0
    - Single-element groups: advantage = 0
    """
    device = rewards.device

    if distributed and dist.is_initialized():
        # All-gather rewards and group_ids across DP group
        from ironcore.parallel.parallel_states import get_data_parallel_world_size, get_data_parallel_group

        world_size = get_data_parallel_world_size()
        if world_size > 1:
            # Gather from all ranks
            gathered_rewards = [torch.zeros_like(rewards) for _ in range(world_size)]
            gathered_group_ids = [torch.zeros_like(group_ids) for _ in range(world_size)]

            dist.all_gather(gathered_rewards, rewards, group=get_data_parallel_group())
            dist.all_gather(gathered_group_ids, group_ids, group=get_data_parallel_group())

            rewards = torch.cat(gathered_rewards, dim=0)
            group_ids = torch.cat(gathered_group_ids, dim=0)

    # Compute advantages
    advantages = torch.zeros_like(rewards)

    for g in group_ids.unique():
        mask = group_ids == g
        group_rewards = rewards[mask]

        if len(group_rewards) > 1:
            mean = group_rewards.mean()
            std = group_rewards.std()
            if std < eps:
                # All rewards identical → zero advantage
                advantages[mask] = 0.0
            else:
                advantages[mask] = (group_rewards - mean) / (std + eps)
        # else: single element, advantage stays 0

    if distributed and dist.is_initialized() and world_size > 1:
        # Scatter back: only return our portion
        local_size = len(advantages) // world_size
        rank = dist.get_rank()
        advantages = advantages[rank * local_size : (rank + 1) * local_size]

    return advantages.to(device)
```

### GRPO Loss

```python
def grpo_loss(
    policy_log_probs: torch.Tensor,   # [B*G] sequence log probs (current)
    ref_log_probs: torch.Tensor,      # [B*G] sequence log probs (reference)
    advantages: torch.Tensor,          # [B*G] normalized advantages
    kl_per_seq: torch.Tensor,          # [B*G] KL divergence per sequence
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute GRPO loss.

    L = -mean(A * log π_θ(y|x)) + β * mean(KL(π_ref || π_θ))

    The first term is the policy gradient: maximize log prob weighted by advantage.
    The second term is the KL penalty: stay close to reference policy.

    Returns:
        (loss, metrics_dict)
    """
    # Policy gradient term (gradient flows through policy_log_probs)
    policy_loss = -(advantages.detach() * policy_log_probs).mean()

    # KL penalty term
    kl_loss = beta * kl_per_seq.mean()

    total_loss = policy_loss + kl_loss

    # Metrics for logging
    with torch.no_grad():
        metrics = {
            "grpo_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_per_seq": kl_per_seq.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item() if len(advantages) > 1 else 0.0,
            "mean_reward": (advantages * advantages.std() + advantages.mean()).mean().item() if advantages.std() > 0 else 0.0,
        }

    return total_loss, metrics
```

---

## Component 3: Rollout Utilities with Prefix Caching

**File**: `ironcore/alignment/rollout.py`

**Purpose**: Generate G completions per prompt efficiently using prefix KV-cache sharing.

### Critical Optimization: Batched Expansion

**Problem**: Nested loops (B × G) are sequential and don't use GPU parallelism.

**Solution**: Expand KV-cache from [B, ...] to [B×G, ...] after prefill, then generate all at once.

```
OLD (Slow):                      NEW (Fast):
for b in range(B):               prefill(prompts)           # [B]
  for g in range(G):             expand_kv_cache(B → B×G)   # Expand
    generate_one()               generate_batch(B×G)        # All at once!
```

### Batched Generation Algorithm

```python
@torch.no_grad()
def generate_rollouts_batched(
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,          # [B, prompt_len]
    group_size: int,
    metadata: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    do_sample: bool = True,
    eos_token_id: int | None = None,
) -> RolloutBuffer:
    """
    Generate G completions per prompt with BATCHED prefix KV-cache.

    Algorithm:
    1. Prefill prompts [B, prompt_len] → prefix_kv, prefill_logits
    2. Expand prefix_kv: [B, ...] → [B×G, ...] by repeating each G times
    3. Sample first tokens from prefill_logits (batched)
    4. Autoregressively generate all B×G sequences in parallel
    """
    B, prompt_len = prompt_ids.shape
    G = group_size
    device = prompt_ids.device
    total_samples = B * G

    # === Step 1: Prefill all prompts ===
    prefill_logits, prefix_kv = model.forward(
        prompt_ids, labels=None, use_cache=True, past_key_values=None
    )
    # prefix_kv: List of (key, value) per layer
    #   key: [B, num_heads, prompt_len, head_dim]

    # === Step 2: Expand KV-cache to [B×G, ...] ===
    # For each of B prompts, we want G copies in sequence
    # Result: [prompt_0, prompt_0, prompt_0, prompt_0, prompt_1, ...]
    expanded_kv = []
    for layer_kv in prefix_kv:
        key, value = layer_kv
        # key: [B, num_heads, prompt_len, head_dim]
        # Expand: repeat each sample G times
        # Method: unsqueeze → repeat → reshape
        expanded_key = key.unsqueeze(1).repeat(1, G, 1, 1, 1).reshape(total_samples, *key.shape[1:])
        expanded_value = value.unsqueeze(1).repeat(1, G, 1, 1, 1).reshape(total_samples, *value.shape[1:])
        expanded_kv.append((expanded_key, expanded_value))

    # === Step 3: Sample first tokens ===
    # prefill_logits: [B, prompt_len, vocab]
    last_logits = prefill_logits[:, -1, :]  # [B, vocab]
    # Expand logits for G samples per prompt
    expanded_logits = last_logits.unsqueeze(1).repeat(1, G, 1).reshape(total_samples, -1)

    # Sample first tokens for all B×G
    first_tokens = _sample_tokens_batched(expanded_logits, temperature, top_p, top_k, do_sample)
    # first_tokens: [B×G, 1]

    # === Step 4: Autoregressive generation (batched) ===
    generated = first_tokens  # [B×G, 1]
    past_kv = expanded_kv
    done_mask = torch.zeros(total_samples, dtype=torch.bool, device=device)

    log_probs_list = [_compute_token_log_probs_batched(expanded_logits, first_tokens.squeeze(-1))]

    for step in range(max_new_tokens - 1):
        if done_mask.all():
            break

        # Forward with cached KV
        logits, past_kv = model.forward(
            generated[:, -1:],  # Only last token
            labels=None,
            use_cache=True,
            past_key_values=past_kv,
        )
        # logits: [B×G, 1, vocab]

        next_tokens = _sample_tokens_batched(logits[:, 0, :], temperature, top_p, top_k, do_sample)
        # next_tokens: [B×G, 1]

        log_probs_list.append(_compute_token_log_probs_batched(logits[:, 0, :], next_tokens.squeeze(-1)))

        # Check EOS
        if eos_token_id is not None:
            done_mask = done_mask | (next_tokens.squeeze(-1) == eos_token_id)

        generated = torch.cat([generated, next_tokens], dim=1)

    # === Step 5: Build output ===
    # Expand prompt_ids to [B×G, prompt_len]
    expanded_prompts = prompt_ids.unsqueeze(1).repeat(1, G, 1).reshape(total_samples, prompt_len)

    # Concatenate prompts + completions
    completion_ids = torch.cat([expanded_prompts, generated], dim=1)  # [B×G, total_len]

    # Compute total log probs per sequence
    log_probs_stacked = torch.stack(log_probs_list, dim=1)  # [B×G, gen_len]
    old_log_probs = log_probs_stacked.sum(dim=1)  # [B×G]

    # Group IDs: [0,0,0,0, 1,1,1,1, ...]
    group_ids = torch.arange(B, device=device).unsqueeze(1).repeat(1, G).reshape(-1)

    # Expand metadata
    expanded_metadata = []
    for i, meta in enumerate(metadata):
        expanded_metadata.extend([meta.copy() for _ in range(G)])

    return RolloutBuffer(
        prompt_ids=prompt_ids,
        prompt_attention_mask=torch.ones_like(prompt_ids),
        completion_ids=completion_ids,
        response_ids=generated,
        old_log_probs=old_log_probs,
        rewards=torch.zeros(total_samples, device=device),
        advantages=torch.zeros(total_samples, device=device),
        group_ids=group_ids,
        metadata=expanded_metadata,
    )


def _sample_tokens_batched(logits, temperature, top_p, top_k, do_sample):
    """Sample tokens for entire batch at once. Returns [batch, 1]."""
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    if temperature != 1.0:
        logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth_vals, float('-inf'))

    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cumprobs = sorted_logits.softmax(-1).cumsum(-1)
        remove = cumprobs - sorted_logits.softmax(-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
        logits = logits.scatter(-1, sorted_idx, sorted_logits)

    probs = logits.softmax(-1)
    return torch.multinomial(probs, num_samples=1)


def _compute_token_log_probs_batched(logits, token_ids):
    """Compute log prob of token_ids. Returns [batch]."""
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
```

### RolloutBuffer

```python
@dataclass
class RolloutBuffer:
    """Storage for rollout data during a training step."""

    # Prompt data (original batch)
    prompt_ids: torch.Tensor           # [B, prompt_len]
    prompt_attention_mask: torch.Tensor  # [B, prompt_len]

    # Completion data (expanded: B×G)
    completion_ids: torch.Tensor       # [B*G, total_len]
    response_ids: torch.Tensor         # [B*G, gen_len]

    # Log probabilities
    old_log_probs: torch.Tensor        # [B*G]

    # Rewards and advantages
    rewards: torch.Tensor              # [B*G]
    advantages: torch.Tensor           # [B*G]

    # Group assignment
    group_ids: torch.Tensor            # [B*G]

    # Metadata for reward computation
    metadata: list[dict]               # [B*G]
```

---

## Component 4: Reward Pipeline with Workers

**File**: `ironcore/alignment/rewards.py`

**Purpose**: Compute rewards in separate worker processes for safety and flexibility.

### Reward Functions

```python
from abc import ABC, abstractmethod

class RewardFunction(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        """Compute reward for a completion given prompt and metadata."""
        pass


class MathRewardFunction(RewardFunction):
    """Reward for math problems with verifiable answers."""

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        answer = metadata.get("answer", "")

        # Extract answer from completion (look for common patterns)
        extracted = self._extract_answer(completion)

        # Normalize and compare
        if self._normalize_answer(extracted) == self._normalize_answer(answer):
            return 1.0
        return 0.0

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from completion."""
        # Try common patterns: "####", "\\boxed{", "Answer:", etc.
        patterns = [
            r'####\s*(.+)',
            r'\\boxed\{(.+?)\}',
            r'[Aa]nswer:\s*(.+)',
            r'[Tt]herefore,\s*(.+)',
        ]
        import re
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        # Fallback: last number in text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return numbers[-1] if numbers else ""

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        import re
        # Remove whitespace, convert to lowercase
        normalized = answer.strip().lower()
        # Remove common formatting
        normalized = re.sub(r'[, _$]', '', normalized)
        return normalized


class CodeRewardFunction(RewardFunction):
    """Reward for code problems with test cases."""

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        test_cases = metadata.get("test_cases", [])
        if not test_cases:
            return 0.0

        full_code = prompt + "\n" + completion
        passed = 0

        for test in test_cases:
            try:
                result = subprocess.run(
                    ["python", "-c", full_code + "\n" + test],
                    capture_output=True,
                    timeout=self.timeout,
                    text=True,
                )
                if result.returncode == 0:
                    passed += 1
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

        return passed / len(test_cases)


class RewardModelFunction(RewardFunction):
    """Reward using a local reward model (future)."""

    def __init__(self, model_path: str, device: str = "cuda"):
        # Load reward model
        # self.model = load_reward_model(model_path, device)
        pass

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        # Score completion with reward model
        # return self.model.score(prompt, completion)
        raise NotImplementedError("Local reward model not yet implemented")


class APIRewardFunction(RewardFunction):
    """Reward using external LLM API (OpenAI, Anthropic, Google, Zhipu).

    Features:
    - LRU cache for repeated completions (cost savings)
    - Rate limiting with exponential backoff
    - Configurable retry policy
    """

    PROVIDER_CONFIGS = {
        "openai": {
            "env_key": "OPENAI_API_KEY",
            "default_endpoint": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
        },
        "anthropic": {
            "env_key": "ANTHROPIC_API_KEY",
            "default_endpoint": None,
            "default_model": "claude-3-haiku-20240307",
        },
        "google": {
            "env_key": "GOOGLE_API_KEY",
            "default_endpoint": None,
            "default_model": "gemini-pro",
        },
        "zhipu": {
            "env_key": "ZHIPU_API_KEY",
            "default_endpoint": "https://open.bigmodel.cn/api/paas/v4",
            "default_model": "glm-4-flash",
        },
    }

    PROMPT_TEMPLATES = {
        "default": """Evaluate the following response on a scale of 0 to 1.

Question/Prompt:
{prompt}

Response:
{completion}

Score (0-1):""",

        "math": """Is this math answer correct?

Problem: {prompt}
Answer: {completion}
Expected: {answer}

Reply with only "1" if correct, "0" if incorrect.""",

        "code": """Evaluate this code solution.

Problem: {prompt}
Code:
{completion}

Test cases: {test_cases}

Score 1 if code passes all tests, 0 otherwise.
Score:""",

        "reasoning": """Evaluate the reasoning quality.

Question: {prompt}
Response: {completion}

Score 0-1 based on:
- Correctness of conclusion
- Quality of reasoning steps
- Completeness

Score:""",
    }

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        endpoint: str | None = None,
        prompt_template: str = "default",
        custom_prompt: str | None = None,
        max_retries: int = 3,
        timeout: int = 30,
        cache_size: int = 10000,  # LRU cache size
        rate_limit_delay: float = 0.1,  # Delay between API calls
    ):
        self.provider = provider.lower()
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self._last_call_time = 0

        config = self.PROVIDER_CONFIGS[self.provider]
        self.model = model or config["default_model"]

        self.api_key = api_key
        if self.api_key is None:
            env_name = api_key_env or config["env_key"]
            self.api_key = os.getenv(env_name)
        if self.api_key is None:
            raise ValueError(f"API key required. Set {config['env_key']} env var or pass api_key param.")

        self.endpoint = endpoint or config["default_endpoint"]

        if custom_prompt:
            self._prompt_template = custom_prompt
        else:
            self._prompt_template = self.PROMPT_TEMPLATES.get(
                prompt_template, self.PROMPT_TEMPLATES["default"]
            )

        self._client = self._init_client()

        # LRU cache for (prompt_hash, completion_hash) -> reward
        from functools import lru_cache
        self._compute_cached = lru_cache(maxsize=cache_size)(self._compute_uncached)

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        """Get reward score from API with caching."""
        # Create cache key
        cache_key = (hash(prompt), hash(completion))
        return self._compute_cached(cache_key, prompt, completion, metadata)

    def _compute_uncached(self, cache_key, prompt, completion, metadata):
        """Actual API call (uncached)."""
        # Rate limiting
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        # Build evaluation prompt
        try:
            eval_prompt = self._prompt_template.format(
                prompt=prompt,
                completion=completion,
                answer=metadata.get("answer", "N/A"),
                test_cases=metadata.get("test_cases", []),
                **metadata
            )
        except KeyError:
            eval_prompt = self._prompt_template.format(prompt=prompt, completion=completion)

        # Call API with retries
        for attempt in range(self.max_retries):
            try:
                self._last_call_time = time.time()
                response = self._call_api(eval_prompt)
                reward = self._parse_response(response)
                return reward
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return 0.5  # Neutral score on failure
                time.sleep(2 ** attempt)  # Exponential backoff

        return 0.5

    # ... rest of implementation same as before ...
```

### FormatRewardFunction

For reasoning models that need structured outputs:

```python
class FormatRewardFunction(RewardFunction):
    """Reward for enforcing structured output format.

    Useful for reasoning models that should output <thought>...</thought> <answer>...</answer>.
    Gives small penalty if required tags are missing.
    """

    def __init__(
        self,
        required_tags: list[str] | None = None,
        penalty: float = -0.1,  # Penalty for missing tags
        reward_for_present: float = 0.0,  # No reward for just having tags
    ):
        self.required_tags = required_tags or ["<thought>", "</thought>", "<answer>", "</answer>"]
        self.penalty = penalty
        self.reward_for_present = reward_for_present

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        """Check format and return penalty for missing tags."""
        missing_count = 0
        for tag in self.required_tags:
            if tag not in completion:
                missing_count += 1

        if missing_count > 0:
            return self.penalty * (missing_count / len(self.required_tags))
        return self.reward_for_present
```

### LocalEndpointRewardFunction

Connect to a local vLLM or SGLang server:

```python
class LocalEndpointRewardFunction(RewardFunction):
    """Reward using local inference server (vLLM, SGLang, TGI).

    This is more cost-effective than external APIs and provides:
    - Full control over the reward model
    - No rate limits
    - Lower latency (local network)
    - Privacy (data doesn't leave your infrastructure)
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8000/v1",  # vLLM/OpenAI-compatible
        model: str | None = None,  # Model name (some servers require it)
        prompt_template: str = "default",
        custom_prompt: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        cache_size: int = 10000,
        api_key: str = "EMPTY",  # Some servers need non-empty key
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key

        # Load prompt template
        if custom_prompt:
            self._prompt_template = custom_prompt
        else:
            self._prompt_template = APIRewardFunction.PROMPT_TEMPLATES.get(
                prompt_template, APIRewardFunction.PROMPT_TEMPLATES["default"]
            )

        # Initialize client (OpenAI-compatible)
        import openai
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=endpoint,
        )

        # LRU cache
        from functools import lru_cache
        self._compute_cached = lru_cache(maxsize=cache_size)(self._compute_uncached)

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        """Get reward score with caching."""
        cache_key = (hash(prompt), hash(completion))
        return self._compute_cached(cache_key, prompt, completion, metadata)

    def _compute_uncached(self, cache_key, prompt, completion, metadata):
        """Actual API call."""
        try:
            eval_prompt = self._prompt_template.format(
                prompt=prompt, completion=completion,
                answer=metadata.get("answer", "N/A"),
                **metadata
            )
        except KeyError:
            eval_prompt = self._prompt_template.format(prompt=prompt, completion=completion)

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model or "default",
                    messages=[{"role": "user", "content": eval_prompt}],
                    max_tokens=32,
                    temperature=0.0,
                )
                return self._parse_response(response.choices[0].message.content)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return 0.5
                time.sleep(2 ** attempt)

        return 0.5

    def _parse_response(self, response: str) -> float:
        """Parse reward from response."""
        import re
        numbers = re.findall(r'[\d.]+', response)
        if numbers:
            score = float(numbers[0])
            if score > 1:
                score = score / 10.0
            if score > 1:
                score = score / 100.0
            return min(max(score, 0.0), 1.0)

        response_lower = response.lower().strip()
        if response_lower in ["yes", "true", "correct", "1"]:
            return 1.0
        if response_lower in ["no", "false", "incorrect", "0"]:
            return 0.0
        return 0.5
```

### LocalInferenceRewardFunction

Spawn a local model process on a specified GPU:

```python
class LocalInferenceRewardFunction(RewardFunction):
    """Reward using a local model loaded on a specific GPU.

    This provides the most control and lowest latency, but requires:
    - GPU memory for the reward model
    - Model weights available locally

    Usage:
        reward_fn = LocalInferenceRewardFunction(
            model_path="/models/reward-model-7b",
            device="cuda:2",  # Use different GPU than training
            prompt_template="math",
        )
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",  # "float16", "bfloat16", "float32"
        prompt_template: str = "default",
        custom_prompt: str | None = None,
        max_length: int = 4096,
        cache_size: int = 10000,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16)

        # Load prompt template
        if custom_prompt:
            self._prompt_template = custom_prompt
        else:
            self._prompt_template = APIRewardFunction.PROMPT_TEMPLATES.get(
                prompt_template, APIRewardFunction.PROMPT_TEMPLATES["default"]
            )

        # Load model
        self._load_model(load_in_8bit, load_in_4bit)

        # LRU cache
        from functools import lru_cache
        self._compute_cached = lru_cache(maxsize=cache_size)(self._compute_uncached)

    def _load_model(self, load_in_8bit, load_in_4bit):
        """Load the reward model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        kwargs = {
            "pretrained_model_name_or_path": self.model_path,
            "torch_dtype": self.dtype,
            "device_map": self.device,
        }

        if load_in_8bit:
            kwargs["load_in_8bit"] = True
            kwargs.pop("torch_dtype")
            kwargs.pop("device_map")
        elif load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs.pop("torch_dtype")
            kwargs.pop("device_map")

        self.model = AutoModelForCausalLM.from_pretrained(**kwargs)
        self.model.eval()

        # Get reward token IDs if using special format
        self.reward_token_id = None
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            try:
                # Some reward models use special tokens
                self.reward_token_id = self.tokenizer.convert_tokens_to_ids(["+"])
            except:
                pass

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        """Get reward score with caching."""
        cache_key = (hash(prompt), hash(completion))
        return self._compute_cached(cache_key, prompt, completion, metadata)

    def _compute_uncached(self, cache_key, prompt, completion, metadata):
        """Actual model inference."""
        try:
            eval_prompt = self._prompt_template.format(
                prompt=prompt, completion=completion,
                answer=metadata.get("answer", "N/A"),
                **metadata
            )
        except KeyError:
            eval_prompt = self._prompt_template.format(prompt=prompt, completion=completion)

        with torch.no_grad():
            inputs = self.tokenizer(
                eval_prompt,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            ).to(self.device)

            outputs = self.model(**inputs)

            # Extract score from logits
            # Option 1: Last token logits → extract score token
            last_logits = outputs.logits[:, -1, :]

            # Option 2: Look for numeric tokens
            score = self._extract_score_from_logits(last_logits)

            return score

    def _extract_score_from_logits(self, logits: torch.Tensor) -> float:
        """Extract numerical score from logits."""
        # Common approach: look for number tokens
        number_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        token_ids = []
        for tok in number_tokens:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None:
                token_ids.append(tid)

        if token_ids:
            # Get probabilities for number tokens
            probs = torch.softmax(logits[0], dim=-1)
            number_probs = probs[token_ids]

            # Weighted average
            scores = torch.tensor([float(t) / 10.0 for t in number_tokens[:len(token_ids)]], device=probs.device)
            return (number_probs * scores).sum().item()

        # Fallback: use argmax and try to parse
        next_token = logits.argmax(dim=-1).item()
        decoded = self.tokenizer.decode([next_token])

        import re
        numbers = re.findall(r'[\d.]+', decoded)
        if numbers:
            score = float(numbers[0])
            if score > 1:
                score = score / 10.0
            return min(max(score, 0.0), 1.0)

        return 0.5

    def __del__(self):
        """Cleanup model memory."""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### Updated Reward Factory

```python
def get_reward_function(reward_type: str, **kwargs) -> RewardFunction:
    """Factory function to create reward functions.

    Supported types:
    - "math": Rule-based math verification
    - "code": Code execution with test cases
    - "api": External LLM API (OpenAI, Anthropic, etc.)
    - "local_endpoint": Local vLLM/SGLang server
    - "local_inference": Local model on specified GPU
    - "format": Check for required output tags
    """
    if reward_type == "math":
        return MathRewardFunction()
    elif reward_type == "code":
        return CodeRewardFunction(timeout=kwargs.get("timeout", 5))
    elif reward_type == "api":
        return APIRewardFunction(**kwargs)
    elif reward_type == "local_endpoint":
        return LocalEndpointRewardFunction(**kwargs)
    elif reward_type == "local_inference":
        return LocalInferenceRewardFunction(**kwargs)
    elif reward_type == "format":
        return FormatRewardFunction(**kwargs)
    elif reward_type == "reward_model":
        # Alias for local_inference with default settings
        return LocalInferenceRewardFunction(**kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}. "
                        f"Supported: math, code, api, local_endpoint, local_inference, format")
```
```

### Reward Worker Pool

```python
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable
import multiprocessing as mp


def _worker_compute(args):
    """Worker function for reward computation (must be top-level for pickle)."""
    reward_fn, prompt, completion, metadata = args
    try:
        return reward_fn(prompt, completion, metadata)
    except Exception as e:
        return 0.0  # Default to 0 on error


class RewardWorkerPool:
    """Pool of worker processes for reward computation."""

    def __init__(
        self,
        reward_fn: RewardFunction,
        num_workers: int = 4,
        timeout: int = 30,
    ):
        self.reward_fn = reward_fn
        self.num_workers = num_workers
        self.timeout = timeout
        self.pool = ProcessPoolExecutor(max_workers=num_workers)

    def score_batch(
        self,
        prompts: list[str],
        completions: list[str],
        metadata_list: list[dict],
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of completions.

        Args:
            prompts: List of prompts
            completions: List of completions (same length)
            metadata_list: List of metadata dicts

        Returns:
            Tensor of rewards [batch_size]
        """
        assert len(prompts) == len(completions) == len(metadata_list)

        # Submit all tasks
        args_list = [
            (self.reward_fn, p, c, m)
            for p, c, m in zip(prompts, completions, metadata_list)
        ]
        futures = [self.pool.submit(_worker_compute, args) for args in args_list]

        # Collect results with timeout
        rewards = []
        for future in futures:
            try:
                reward = future.result(timeout=self.timeout)
                rewards.append(float(reward))
            except FutureTimeoutError:
                rewards.append(0.0)
            except Exception:
                rewards.append(0.0)

        return torch.tensor(rewards, dtype=torch.float32)

    def shutdown(self):
        """Shutdown the worker pool."""
        self.pool.shutdown(wait=False)


def get_reward_function(reward_type: str, **kwargs) -> RewardFunction:
    """Factory function to create reward functions."""
    if reward_type == "math":
        return MathRewardFunction()
    elif reward_type == "code":
        return CodeRewardFunction(timeout=kwargs.get("timeout", 5))
    elif reward_type == "reward_model":
        return RewardModelFunction(**kwargs)
    elif reward_type == "api":
        return APIRewardFunction(**kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}. "
                        f"Supported: math, code, reward_model, api")


# === Test API Keys Setup ===
#
# For testing API reward functions, set environment variables:
#
#   export OPENAI_API_KEY="sk-..."
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export GOOGLE_API_KEY="AIza..."
#   export ZHIPU_API_KEY="..."
#
# Or create a .env file in project root:
#
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
#   GOOGLE_API_KEY=AIza...
#   ZHIPU_API_KEY=...
#
# Example usage in config:
#
#   reward:
#     type: "api"
#     provider: "openai"
#     model: "gpt-4o-mini"
#     prompt_template: "math"
#
#   reward:
#     type: "api"
#     provider: "anthropic"
#     model: "claude-3-haiku-20240307"
#     prompt_template: "code"
```

---

## Component 5: GRPO Dataset

**File**: `ironcore/alignment/dataset.py`

**Purpose**: Dataset that handles both verifiable and reward model formats.

### Dataset Format

**Verifiable (math)**:
```json
{"prompt": "Solve: 2x + 3 = 7", "answer": "x = 2", "type": "math"}
```

**Verifiable (code)**:
```json
{"prompt": "def fibonacci(n):\n    ", "test_cases": ["assert fibonacci(5) == 5", "assert fibonacci(10) == 55"], "type": "code"}
```

**Reward Model**:
```json
{"prompt": "Write a haiku about coding"}
```

### Implementation

```python
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from ironcore import get_tokenizer


@dataclass
class GRPOSample:
    """Single sample for GRPO training."""
    prompt: str
    input_ids: torch.Tensor       # Tokenized prompt
    attention_mask: torch.Tensor
    metadata: dict                # Contains answer/test_cases/type/etc.


class GRPODataset(IterableDataset):
    """
    Dataset for GRPO training supporting multiple formats.

    Handles:
    - Verifiable tasks (math, code) with ground truth
    - Reward model tasks without ground truth
    """

    def __init__(
        self,
        data_path: str | Path,
        max_prompt_length: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.max_prompt_length = max_prompt_length
        self.shuffle = shuffle
        self.seed = seed
        self.tokenizer = get_tokenizer()

        # Load and validate data
        self.samples = self._load_data()

    def _load_data(self) -> list[dict]:
        """Load data from file."""
        samples = []

        if self.data_path.suffix == ".jsonl":
            with open(self.data_path) as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        elif self.data_path.suffix == ".json":
            with open(self.data_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = [data]
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        return samples

    def __iter__(self) -> Iterator[GRPOSample]:
        """Iterate over samples."""
        indices = list(range(len(self.samples)))

        if self.shuffle:
            import random
            random.Random(self.seed).shuffle(indices)

        for idx in indices:
            sample = self.samples[idx]

            # Tokenize prompt
            prompt = sample["prompt"]
            encoded = self.tokenizer(
                prompt,
                max_length=self.max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Build metadata
            metadata = {
                "type": sample.get("type", "unknown"),
                "answer": sample.get("answer"),
                "test_cases": sample.get("test_cases", []),
                "original_prompt": prompt,
            }

            yield GRPOSample(
                prompt=prompt,
                input_ids=encoded["input_ids"].squeeze(0),
                attention_mask=encoded["attention_mask"].squeeze(0),
                metadata=metadata,
            )

    def __len__(self):
        return len(self.samples)


def get_grpo_data_iterator(
    config,
    split: str = "train",
) -> Iterator[dict]:
    """
    Create data iterator for GRPO training.

    Returns batches with:
    - input_ids: [B, prompt_len]
    - attention_mask: [B, prompt_len]
    - metadata: list[dict]
    """
    from torch.utils.data import DataLoader

    data_config = config.data
    data_path = data_config.train_file if split == "train" else data_config.eval_file

    dataset = GRPODataset(
        data_path=data_path,
        max_prompt_length=config.model.max_position_embeddings,
        shuffle=(split == "train"),
        seed=config.init.seed,
    )

    # Custom collate to handle metadata
    def collate_fn(samples):
        return {
            "input_ids": torch.stack([s.input_ids for s in samples]),
            "attention_mask": torch.stack([s.attention_mask for s in samples]),
            "metadata": [s.metadata for s in samples],
        }

    dataloader = DataLoader(
        dataset,
        batch_size=config.trainer.train_batch_size,
        collate_fn=collate_fn,
        num_workers=getattr(data_config, "num_workers", 0),
    )

    return iter(dataloader)
```

---

## Component 6: Configuration

**File**: `ironcore/config/config_alignment.py` (extend existing)

```python
@dataclass
class GenerationConfig:
    """Configuration for GRPO generation."""
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True


@dataclass
class RewardConfig:
    """Configuration for GRPO reward computation."""
    type: str = "math"              # "math" | "code" | "api" | "local_endpoint" | "local_inference" | "format"

    # Worker configuration
    num_workers: int = 4            # Worker processes
    timeout: int = 30               # Seconds before timeout

    # API reward configuration (when type="api")
    api_provider: str = "openai"    # "openai" | "anthropic" | "google" | "zhipu"
    api_model: str | None = None    # Model name (None = provider default)
    api_key_env: str | None = None  # Env var name for API key (None = provider default)
    api_endpoint: str | None = None # Custom endpoint (None = provider default)
    prompt_template: str = "default" # "default" | "math" | "code" | "reasoning"
    custom_prompt: str | None = None # Custom prompt template
    max_retries: int = 3            # API retry attempts
    cache_size: int = 10000         # LRU cache size
    rate_limit_delay: float = 0.1   # Delay between API calls

    # Local endpoint configuration (when type="local_endpoint")
    local_endpoint: str = "http://localhost:8000/v1"  # vLLM/SGLang endpoint

    # Local inference configuration (when type="local_inference")
    local_model_path: str | None = None  # Path to reward model
    local_device: str = "cuda:0"         # Device for reward model
    local_dtype: str = "bfloat16"        # "float16" | "bfloat16" | "float32"
    load_in_8bit: bool = False           # Use 8-bit quantization
    load_in_4bit: bool = False           # Use 4-bit quantization

    # Format reward configuration (when type="format")
    required_tags: list[str] | None = None  # e.g., ["<thought>", "</thought>", "<answer>", "</answer>"]
    format_penalty: float = -0.1            # Penalty for missing tags


@dataclass
class AlignmentConfig(BaseConfig):
    """Configuration for alignment training (DPO, GRPO, etc.)."""

    # Alignment method
    method: str = "dpo"  # "dpo" | "grpo"

    # DPO specific
    dpo_beta: float = 0.5
    dpo_label_smoothing: float = 0.0

    # GRPO specific
    grpo_group_size: int = 4          # G completions per prompt
    grpo_beta: float = 0.1            # KL penalty coefficient
    grpo_eps: float = 1e-8            # Advantage normalization epsilon

    # GRPO generation and reward config
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Shared
    concat_forward_passes: bool = True
    metrics_interval: int = 0

    def __post_init__(self):
        """Validate configuration."""
        if self.method not in ["dpo", "grpo"]:
            raise ValueError(f"Unknown alignment method: {self.method}")

        if self.method == "grpo":
            if self.grpo_group_size < 2:
                raise ValueError(f"grpo_group_size must be >= 2, got {self.grpo_group_size}")
            if self.grpo_beta < 0:
                raise ValueError(f"grpo_beta must be >= 0, got {self.grpo_beta}")
            if self.reward.type not in ["math", "code", "reward_model", "api"]:
                raise ValueError(f"Unknown reward type: {self.reward.type}")
```

### Example Configurations

**Using local math reward:**
```yaml
alignment:
  method: grpo
  grpo_group_size: 4
  grpo_beta: 0.1
  generation:
    max_new_tokens: 512
    temperature: 1.0
  reward:
    type: math
    num_workers: 4
```

**Using OpenAI API as reward:**
```yaml
alignment:
  method: grpo
  grpo_group_size: 4
  grpo_beta: 0.1
  generation:
    max_new_tokens: 512
    temperature: 1.0
  reward:
    type: api
    api_provider: openai
    api_model: gpt-4o-mini
    prompt_template: math
    num_workers: 8  # More workers for API latency
```

**Using Claude API as reward:**
```yaml
alignment:
  method: grpo
  grpo_group_size: 4
  grpo_beta: 0.1
  reward:
    type: api
    api_provider: anthropic
    api_model: claude-3-haiku-20240307
    prompt_template: default
```

**Using custom endpoint (e.g., local vLLM):**
```yaml
alignment:
  method: grpo
  reward:
    type: api
    api_provider: openai
    api_endpoint: http://localhost:8000/v1
    api_key_env: DUMMY_KEY
    prompt_template: default
```

**Using local vLLM/SGLang server:**
```yaml
alignment:
  method: grpo
  reward:
    type: local_endpoint
    local_endpoint: http://localhost:8000/v1
    prompt_template: math
    num_workers: 8
```

**Using local reward model on different GPU:**
```yaml
alignment:
  method: grpo
  reward:
    type: local_inference
    local_model_path: /models/Qwen2.5-Math-7B
    local_device: cuda:2  # Use GPU 2 for reward (training on GPU 0,1)
    local_dtype: bfloat16
    prompt_template: math
```

**Using 4-bit quantized reward model (saves memory):**
```yaml
alignment:
  method: grpo
  reward:
    type: local_inference
    local_model_path: /models/reward-model-7b
    local_device: cuda:2
    load_in_4bit: true  # ~4GB instead of ~14GB
```

**Combining format + correctness reward:**
```yaml
alignment:
  method: grpo
  reward:
    type: combined
    reward_weights:
      format: 0.1
      math: 0.9
    format:
      required_tags: ["<thought>", "</thought>", "<answer>", "</answer>"]
      format_penalty: -0.1
```

### Test API Keys Setup

For testing, set these environment variables:

```bash
# OpenAI (GPT-4, GPT-3.5)
export OPENAI_API_KEY="sk-..."

# Anthropic (Claude 3)
export ANTHROPIC_API_KEY="sk-ant-..."

# Google (Gemini)
export GOOGLE_API_KEY="AIza..."

# Zhipu (GLM-4)
export ZHIPU_API_KEY="..."
```

Or add to `.env` file in project root (add to .gitignore!).
```

---

## Component 7: GRPO Trainer

**File**: `ironcore/trainers/grpo_trainer.py`

```python
"""GRPO (Group Relative Policy Optimization) Trainer.

Reference:
    DeepSeek-AI et al., "DeepSeekMath: Pushing the Limits of Mathematical
    Reasoning in Open Language Models" (2024)
    https://arxiv.org/abs/2402.03300
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ironcore.alignment.loss.grpo import compute_advantages, grpo_loss
from ironcore.alignment.loss.kl import compute_sequence_log_probs, kl_divergence
from ironcore.alignment.rewards import get_reward_function, RewardWorkerPool
from ironcore.alignment.rollout import generate_rollouts_with_prefix_cache
from ironcore.global_vars import log_metric
from ironcore.utils import is_first_rank

from .base_trainer import BaseTrainer

if TYPE_CHECKING:
    from collections.abc import Iterator


class GRPOTrainer(BaseTrainer):
    """Trainer for Group Relative Policy Optimization (GRPO).

    GRPO improves reasoning by:
    1. Generating multiple completions per prompt
    2. Computing group-relative advantages
    3. Optimizing with policy gradient + KL penalty

    Key differences from DPO:
    - Online: generates completions from current policy
    - Group-based: normalizes rewards within groups
    - Reward-agnostic: works with verifiable rewards or reward models
    """

    def __init__(self, config, forward_step_func, loss_fn):
        super().__init__(config, forward_step_func, loss_fn)

        # GRPO hyperparameters
        self.group_size = config.alignment.grpo_group_size
        self.beta = config.alignment.grpo_beta
        self.eps = config.alignment.grpo_eps

        # Generation config
        gen_config = config.alignment.generation
        self.gen_kwargs = {
            "max_new_tokens": gen_config.max_new_tokens,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "do_sample": gen_config.do_sample,
        }

        # Reward worker
        reward_config = config.alignment.reward
        reward_fn = get_reward_function(
            reward_config.type,
            timeout=reward_config.timeout,
        )
        self.reward_worker = RewardWorkerPool(
            reward_fn=reward_fn,
            num_workers=reward_config.num_workers,
            timeout=reward_config.timeout,
        )

        # Reference model (created after checkpoint load)
        self.reference_model = None

        self.logger.info(
            f"GRPOTrainer initialized with group_size={self.group_size}, "
            f"beta={self.beta}, gen_kwargs={self.gen_kwargs}"
        )

    def _post_checkpoint_load(self, last_step: int) -> None:
        """Create reference model after checkpoint loading."""
        if dist.is_initialized():
            dist.barrier()

        self.logger.info("Creating reference model for GRPO...")
        self.reference_model = self._create_reference_model()

        if dist.is_initialized():
            dist.barrier()

    def _create_reference_model(self) -> torch.nn.Module:
        """Create frozen reference model from current policy.

        Reuses pattern from DPOTrainer.
        """
        # (Same implementation as DPOTrainer._create_reference_model)
        self.logger.info("Creating reference model from policy weights...")

        if isinstance(self.model, FSDP):
            from torch.distributed.fsdp import StateDictType
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                full_state_dict = self.model.state_dict()

            unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
            model_class = unwrapped_model.__class__
            reference_model = model_class(unwrapped_model.config)
            reference_model.load_state_dict(full_state_dict, strict=False)
        else:
            model_to_copy = self.model.module if hasattr(self.model, "module") else self.model
            reference_model = copy.deepcopy(model_to_copy)

        reference_model.eval()

        for param in reference_model.parameters():
            param.requires_grad = False

        device = self._get_compute_device()
        if isinstance(self.model, FSDP) and hasattr(self.model, "mixed_precision"):
            dtype = self.model.mixed_precision.param_dtype
        else:
            dtype = next(self.model.parameters()).dtype
        reference_model.to(device=device, dtype=dtype)

        return reference_model

    def _get_compute_device(self) -> torch.device:
        """Get the device where computation should happen."""
        if isinstance(self.model, FSDP):
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return next(self.model.parameters()).device

    def _move_batch_to_device(self, batch: dict) -> dict:
        """Move all tensors in batch to model device."""
        device = self._get_compute_device()
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def train_step(self, step: int) -> tuple[float, float, float]:
        """GRPO training step.

        1. Sample prompts from dataset
        2. Generate G completions per prompt (with prefix caching)
        3. Compute rewards via worker pool
        4. Compute group-wise advantages
        5. Compute GRPO loss
        6. Backward + optimizer step
        """
        self.timer.start(name="iter")

        # Get batch
        batch = next(self.data_iterator["train"])
        batch = self._move_batch_to_device(batch)

        prompt_ids = batch["input_ids"]
        metadata = batch["metadata"]
        B = prompt_ids.size(0)
        G = self.group_size

        # === Step 1: Generate rollouts ===
        self.model.eval()  # Switch to eval mode for generation
        with torch.no_grad():
            rollout = generate_rollouts_with_prefix_cache(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt_ids=prompt_ids,
                group_size=G,
                metadata=metadata,
                eos_token_id=self.tokenizer.eos_token_id,
                **self.gen_kwargs,
            )
        self.model.train()  # Back to train mode

        # === Step 2: Compute rewards ===
        # Decode completions for reward computation
        prompts_text = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.tokenizer.batch_decode(
            rollout.completion_ids, skip_special_tokens=True
        )

        rewards = self.reward_worker.score_batch(
            prompts=prompts_text * G,  # Repeat for each completion
            completions=completions_text,
            metadata_list=rollout.metadata,
        )
        rewards = rewards.to(prompt_ids.device)

        # === Step 3: Compute advantages ===
        advantages = compute_advantages(rewards, rollout.group_ids, self.eps)

        # === Step 4: Compute GRPO loss ===
        loss, metrics = self._compute_grpo_loss(rollout, advantages)

        # === Step 5: Backward + step ===
        self.scaler.scale(loss).backward()

        grad_norm, param_norm = self._compute_grad_and_param_norms(step)
        self._optimizer_step()

        self.timer.stop(name="iter")

        # Check for NaN
        self._check_loss_for_nan(metrics["grpo_loss"], step)

        # Logging
        if is_first_rank() and self.control.do_log(step):
            self._log_grpo_metrics(step, metrics, rewards, advantages)

        return metrics["grpo_loss"], grad_norm, param_norm

    def _compute_grpo_loss(
        self,
        rollout,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute GRPO loss with current policy."""
        device = self._get_compute_device()

        # Create labels (shift by 1 for next-token prediction)
        labels = rollout.completion_ids.clone()
        labels[:, :-1] = rollout.completion_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last position

        # Mask: only compute loss on response tokens (not prompt)
        prompt_len = rollout.prompt_ids.size(1)
        response_mask = torch.zeros_like(labels, dtype=torch.float)
        response_mask[:, prompt_len-1:-1] = 1.0  # Include last prompt token for first generated
        labels[:, :prompt_len-1] = -100  # Ignore prompt tokens

        # Compute policy log probs (with gradients)
        policy_logits = self.model(rollout.completion_ids, labels=None)
        policy_log_probs = self._compute_sequence_log_probs_from_logits(
            policy_logits, labels, response_mask
        )

        # Compute reference log probs (no gradients)
        with torch.no_grad():
            ref_logits = self.reference_model(rollout.completion_ids, labels=None)
            ref_log_probs = self._compute_sequence_log_probs_from_logits(
                ref_logits, labels, response_mask
        )

        # Compute KL divergence
        kl_per_seq = self._compute_kl_from_logits(policy_logits, ref_logits, response_mask)

        # GRPO loss
        loss, metrics = grpo_loss(
            policy_log_probs=policy_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            kl_per_seq=kl_per_seq,
            beta=self.beta,
        )

        return loss, metrics

    def _compute_sequence_log_probs_from_logits(self, logits, labels, mask):
        """Compute sequence log probs from logits (TP-safe)."""
        from ironcore.alignment.loss.dpo import _compute_log_softmax_tp_safe, _extract_logps_from_log_probs

        log_probs = _compute_log_softmax_tp_safe(logits)
        return _extract_logps_from_log_probs(log_probs, labels, mask)

    def _compute_kl_from_logits(self, policy_logits, ref_logits, mask):
        """Compute KL divergence from logits."""
        from ironcore.alignment.loss.dpo import _compute_log_softmax_tp_safe

        policy_log_probs = _compute_log_softmax_tp_safe(policy_logits)
        ref_log_probs = _compute_log_softmax_tp_safe(ref_logits)

        from ironcore.alignment.loss.kl import kl_divergence
        return kl_divergence(policy_log_probs, ref_log_probs, mask)

    def _log_grpo_metrics(self, step, metrics, rewards, advantages):
        """Log GRPO-specific metrics."""
        for name, value in metrics.items():
            log_metric(f"grpo/{name}", value, step)

        log_metric("grpo/mean_reward", rewards.mean().item(), step)
        log_metric("grpo/std_reward", rewards.std().item(), step)
        log_metric("grpo/mean_advantage", advantages.mean().item(), step)

        self.logger.info(
            f"step: {step}, grpo_loss: {metrics['grpo_loss']:.4f}, "
            f"policy_loss: {metrics['policy_loss']:.4f}, "
            f"kl_loss: {metrics['kl_loss']:.4f}, "
            f"mean_reward: {rewards.mean().item():.4f}"
        )

    def _eval_step(self, data_iterator: Iterator) -> tuple[float, float]:
        """Evaluation step (simplified - just compute loss on held-out prompts)."""
        # For GRPO, evaluation is tricky since we need to generate + reward
        # For now, return placeholder
        return 0.0, 0.0
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `ironcore/alignment/loss/kl.py` | CREATE | KL divergence utilities |
| `ironcore/alignment/loss/grpo.py` | CREATE | Advantage + GRPO loss |
| `ironcore/alignment/rollout.py` | CREATE | Rollout buffer + batched generation |
| `ironcore/alignment/rewards.py` | CREATE | Reward functions + worker pool + cache |
| `ironcore/alignment/dataset.py` | CREATE | GRPO dataset |
| `ironcore/config/config_alignment.py` | MODIFY | Add GRPO config fields |
| `ironcore/trainers/grpo_trainer.py` | CREATE | Main trainer |
| `ironcore/alignment/loss/__init__.py` | MODIFY | Export new loss functions |
| `ironcore/trainers/__init__.py` | MODIFY | Export GRPOTrainer |

## Implementation Checklist (Updated)

| Component | Status | Key Requirements |
|-----------|--------|------------------|
| **KL Divergence** | Planned | TP-safe via `_compute_log_softmax_tp_safe` |
| **Advantage Computation** | Planned | All-gather for DP/FSDP, handle std < eps |
| **Batched Rollout** | Planned | Expand KV-cache [B] → [B×G], generate in parallel |
| **TP Safety** | Planned | Broadcast sampled tokens from rank 0 |
| **Reward Workers** | Planned | LRU cache, rate limiting, retry policy |
| **Format Reward** | Planned | Penalize missing <thought>/<answer> tags |
| **Dataset** | Planned | Support arbitrary metadata keys |
| **FSDP Integration** | Planned | Reference model with sharded state dict |
| **Config** | Planned | API reward config, generation config |
| `ironcore/trainers/__init__.py` | MODIFY | Export GRPOTrainer |

---

## TP/Distributed Considerations

1. **Log softmax**: Use `_compute_log_softmax_tp_safe()` (already handles vocab sharding)
2. **Token sampling**: Already broadcast from rank 0 in `LanguageModel.generate()`
3. **KV cache**: Naturally per-device, no sync needed during generation
4. **Gradients**: Standard DDP/FSDP sync after backward
5. **Rewards**: Computed on main process, broadcast if needed

---

## Memory Optimization

1. **Prefix caching**: Reuse prompt KV cache (2× speedup for typical settings)
2. **No grad during rollout**: `torch.no_grad()` during generation
3. **KV cache cleanup**: Clear cache after each completion
4. **Gradient checkpointing**: Enable for policy loss backward if needed
5. **FSDP**: Shard model parameters across data parallel ranks

### Reference Model Memory Efficiency

**Problem**: GRPO keeps both policy (θ) and reference (θ_ref) in GPU memory.

**Solutions**:

#### Option 1: CPU Offload Reference Model
```python
# Keep reference on CPU, move to GPU only during forward pass
reference_model.to("cpu")

def compute_ref_forward(inputs):
    with torch.no_grad():
        # Move inputs to CPU, compute, move result back
        inputs_cpu = inputs.cpu()
        ref_logits = reference_model(inputs_cpu)
        return ref_logits.to("cuda")
```

#### Option 2: Share FSDP Process Group (Recommended)
```python
# Create reference model with same FSDP wrapping as policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

reference_model = FSDP(
    reference_model,
    process_group=data_parallel_group,  # Share same group
    device_id=torch.cuda.current_device(),
)
# Both models share memory via FSDP sharding
```

#### Option 3: Analytical KL Approximation (Advanced)
Instead of full forward pass through reference:
- Use `old_log_probs` from rollout (already computed under reference)
- Only compute policy log probs during loss
- Approximate KL using rollout statistics

**Recommendation for Phase 1**: Option 2 (FSDP sharing) with fallback to Option 1 if OOM.

---

## Verification Plan

### Unit Tests
1. `test_kl_divergence()` - Compare against scipy.stats.entropy
2. `test_advantage_normalization()` - Verify group-wise normalization
3. `test_grpo_loss()` - Verify loss decreases with gradient step
4. `test_prefix_cache_generation()` - Verify output matches non-cached generation
5. `test_reward_worker()` - Verify correct reward computation

### Integration Test
```bash
# Create toy dataset
echo '{"prompt": "What is 2+2?", "answer": "4", "type": "math"}' > test_grpo.jsonl

# Run single training step
python -m ironcore.train \
    --config configs/grpo_test.yaml \
    --data.train_file test_grpo.jsonl \
    --operation.train_steps 1
```

### Manual Verification
1. Print rewards/advantages to verify group normalization
2. Check GPU memory before/after rollout generation
3. Verify KL divergence is reasonable (not exploding)
4. Check training stability over 100+ steps
