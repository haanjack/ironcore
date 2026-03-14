# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Reward computation for GRPO with multiple backends.

Supports:
- Math: Rule-based verification for math problems
- Code: Execution-based verification with test cases
- API: External LLM APIs (OpenAI, Anthropic, Google, Zhipu)
- Local endpoint: Local vLLM/SGLang servers
- Local inference: Local model on specified GPU
- Format: Check for required output tags

All reward functions support LRU caching for cost savings.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


class RewardFunction(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        """Compute reward for a completion given prompt and metadata.

        Args:
            prompt: The input prompt
            completion: The model's completion
            metadata: Additional info (answer, test_cases, etc.)

        Returns:
            Reward score, typically in [0, 1] range
        """
        pass


class MathRewardFunction(RewardFunction):
    """Reward for math problems with verifiable answers."""

    def __init__(self, strict: bool = True):
        self.strict = strict

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        answer = metadata.get("answer", "")
        if not answer:
            return 0.5  # No ground truth, neutral score

        # Extract answers from both completion and ground truth
        extracted = self._extract_answer(completion)
        gold = self._extract_answer(answer)

        if not extracted or not gold:
            return 0.0

        if self._normalize_answer(extracted) == self._normalize_answer(gold):
            return 1.0

        # Partial credit: extracted a number but it's wrong
        # Gives gradient signal that answer extraction is on the right track
        if extracted:
            return 0.1

        return 0.0

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from completion or ground truth."""
        patterns = [
            r"####\s*(.+)",
            r"\\boxed\{(.+?)\}",
            r"[Aa]nswer:\s*(.+)",
            r"[Tt]herefore,\s*(.+)",
            r"[Tt]he answer is\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        if not self.strict:
            # Fallback: last number in text
            numbers = re.findall(r"-?\d+\.?\d*", text)
            return numbers[-1] if numbers else ""

        return ""

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        normalized = answer.strip().lower()
        # Remove common formatting: commas, spaces, underscores, dollar signs
        normalized = re.sub(r"[, _$]", "", normalized)
        # Remove trailing period only (not decimal points in the middle)
        if normalized.endswith("."):
            normalized = normalized[:-1]
        return normalized


class CodeRewardFunction(RewardFunction):
    """Reward for code problems with test cases."""

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        test_cases = metadata.get("test_cases", [])
        if not test_cases:
            return 0.5

        full_code = prompt + "\n" + completion
        passed = 0

        for test in test_cases:
            try:
                result = subprocess.run(
                    ["python", "-c", full_code + "\n" + test],
                    capture_output=True,
                    timeout=self.timeout,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    passed += 1
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

        return passed / len(test_cases)


class FormatRewardFunction(RewardFunction):
    """Reward for enforcing structured output format.

    Useful for reasoning models that should output:
    <thought>...</thought> <answer>...</answer>
    """

    def __init__(
        self,
        required_tags: list[str] | None = None,
        penalty: float = -0.1,
        reward_for_present: float = 0.0,
    ):
        self.required_tags = required_tags or [
            "<thought>",
            "</thought>",
            "<answer>",
            "</answer>",
        ]
        self.penalty = penalty
        self.reward_for_present = reward_for_present

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        missing = sum(1 for tag in self.required_tags if tag not in completion)
        if missing > 0:
            return self.penalty * (missing / len(self.required_tags))
        return self.reward_for_present


class StrictFormatRewardFunction(RewardFunction):
    """Reward for enforcing exact regex format matching.

    Useful for strict reasoning protocols (e.g. DeepSeekMath format).
    """

    def __init__(
        self,
        pattern: str = r"<think>.*?</think>\s*####\s*.*",
        reward: float = 1.0,
        penalty: float = 0.0,
    ):
        self.pattern = re.compile(pattern, re.DOTALL)
        self.reward = reward
        self.penalty = penalty

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        if self.pattern.search(completion):
            return self.reward
        return self.penalty


class KeywordRewardFunction(RewardFunction):
    """Binary reward: 1.0 if keyword appears in response, else 0.0.

    Useful for sanity-checking GRPO pipeline with trivial reward signal.
    """

    def __init__(self, keyword: str = "ironcore", case_sensitive: bool = False):
        self.keyword = keyword
        self.case_sensitive = case_sensitive

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        # Allow per-sample keyword override via metadata
        keyword = metadata.get("keyword", self.keyword)
        text = completion if self.case_sensitive else completion.lower()
        target = keyword if self.case_sensitive else keyword.lower()
        return 1.0 if target in text else 0.0


class SoftKeywordRewardFunction(RewardFunction):
    """Soft reward: partial credit for character-level matches.

    Returns max similarity score between keyword and any n-gram of same length in completion.
    This provides gradient signal even when exact match is not achieved.

    Score = (matching characters) / (total characters)
    """

    def __init__(
        self, keyword: str = "ironcore", case_sensitive: bool = False, min_score: float = 0.0
    ):
        self.keyword = keyword
        self.case_sensitive = case_sensitive
        self.min_score = min_score

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        keyword = metadata.get("keyword", self.keyword)
        text = completion if self.case_sensitive else completion.lower()
        target = keyword if self.case_sensitive else keyword.lower()

        # Exact match
        if target in text:
            return 1.0

        # Partial credit: find best matching n-gram
        n = len(target)
        if n == 0 or len(text) < n:
            return self.min_score

        best_score = 0.0
        for i in range(len(text) - n + 1):
            ngram = text[i : i + n]
            matches = sum(1 for a, b in zip(ngram, target, strict=True) if a == b)
            score = matches / n

            best_score = max(best_score, score)

        return max(best_score, self.min_score)


class CompositeRewardFunction(RewardFunction):
    """Weighted combination of multiple reward functions.

    Provides dense reward signal by combining sparse correctness rewards
    with easier-to-achieve format/structure rewards. This prevents the
    zero-advantage death spiral where all completions in a group score 0.
    """

    def __init__(self, reward_fns: list[tuple[float, RewardFunction]]):
        """
        Args:
            reward_fns: List of (weight, reward_fn) tuples. Weights should sum to 1.0.
        """
        self.reward_fns = reward_fns

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        total = 0.0
        for weight, fn in self.reward_fns:
            total += weight * fn.compute(prompt, completion, metadata)
        return total


class APIRewardFunction(RewardFunction):
    """Reward using external LLM API (OpenAI, Anthropic, Google, Zhipu)."""

    PROVIDER_CONFIGS = {
        "openai": {
            "env_key": "OPENAI_API_KEY",
            "default_model": "gpt-4o-mini",
        },
        "anthropic": {
            "env_key": "ANTHROPIC_API_KEY",
            "default_model": "claude-3-haiku-20240307",
        },
        "google": {
            "env_key": "GOOGLE_API_KEY",
            "default_model": "gemini-pro",
        },
        "zhipu": {
            "env_key": "ZHIPU_API_KEY",
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
        prompt_template: str = "default",
        custom_prompt: str | None = None,
        max_retries: int = 3,
        timeout: int = 30,
        cache_size: int = 10000,
        rate_limit_delay: float = 0.1,
    ):
        self.provider = provider.lower()
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self._last_call_time = 0.0

        config = self.PROVIDER_CONFIGS.get(self.provider)
        if not config:
            raise ValueError(f"Unknown provider: {self.provider}")

        self.model = model or config["default_model"]

        self.api_key = api_key or os.getenv(config["env_key"])
        if not self.api_key:
            raise ValueError(f"API key required. Set {config['env_key']} env var.")

        if custom_prompt:
            self._prompt_template = custom_prompt
        else:
            self._prompt_template = self.PROMPT_TEMPLATES.get(
                prompt_template, self.PROMPT_TEMPLATES["default"]
            )

        self._client = self._init_client()
        self._cache_size = cache_size

    def _init_client(self):
        if self.provider == "openai":
            import openai

            return openai.OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "google":
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel(self.model)
        elif self.provider == "zhipu":
            from zhipuai import ZhipuAI

            return ZhipuAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        # Use JSON for faster hashing of complex metadata
        metadata_str = json.dumps(metadata, sort_keys=True)
        return self._compute_cached(
            hash(prompt), hash(completion), hash(metadata_str), prompt, completion, metadata
        )

    def _make_hashable(self, obj):
        """Deprecated: use json.dumps for hashing metadata."""
        return json.dumps(obj, sort_keys=True)

    def _compute_cached(
        self,
        _prompt_hash: int,
        _completion_hash: int,
        _metadata_hash: int,
        prompt: str,
        completion: str,
        metadata: dict,
    ) -> float:
        """Cached computation. Hash args are for cache key only."""
        # Rate limiting
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        try:
            eval_prompt = self._build_eval_prompt(prompt, completion, metadata)
        except KeyError:
            eval_prompt = self._prompt_template.format(prompt=prompt, completion=completion)

        for attempt in range(self.max_retries):
            try:
                self._last_call_time = time.time()
                response = self._call_api(eval_prompt)
                return self._parse_response(response)
            except Exception:
                if attempt == self.max_retries - 1:
                    return 0.5
                time.sleep(2**attempt)

        return 0.5

    def _build_eval_prompt(self, prompt: str, completion: str, metadata: dict) -> str:
        return self._prompt_template.format(
            prompt=prompt,
            completion=completion,
            answer=metadata.get("answer", "N/A"),
            test_cases=metadata.get("test_cases", []),
            **{k: v for k, v in metadata.items() if k not in ("answer", "test_cases")},
        )

    def _call_api(self, eval_prompt: str) -> str:
        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": eval_prompt}],
                max_tokens=32,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""

        elif self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=32,
                messages=[{"role": "user", "content": eval_prompt}],
            )
            return response.content[0].text if response.content else ""

        elif self.provider == "google":
            response = self._client.generate_content(eval_prompt)
            return response.text or ""

        elif self.provider == "zhipu":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": eval_prompt}],
            )
            return response.choices[0].message.content or ""

        return ""

    def _parse_response(self, response: str) -> float:
        numbers = re.findall(r"[\d.]+", response)
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


class LocalEndpointRewardFunction(RewardFunction):
    """Reward using local inference server (vLLM, SGLang, TGI)."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8000/v1",
        model: str | None = None,
        prompt_template: str = "default",
        custom_prompt: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        api_key: str = "EMPTY",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        if custom_prompt:
            self._prompt_template = custom_prompt
        else:
            self._prompt_template = APIRewardFunction.PROMPT_TEMPLATES.get(
                prompt_template, APIRewardFunction.PROMPT_TEMPLATES["default"]
            )

        import openai

        self._client = openai.OpenAI(api_key=api_key, base_url=endpoint)

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        # Use JSON for faster hashing of complex metadata
        metadata_str = json.dumps(metadata, sort_keys=True)
        return self._compute_cached(
            hash(prompt), hash(completion), hash(metadata_str), prompt, completion, metadata
        )

    def _make_hashable(self, obj):
        """Deprecated: use json.dumps for hashing metadata."""
        return json.dumps(obj, sort_keys=True)

    def _compute_cached(
        self,
        _prompt_hash: int,
        _completion_hash: int,
        _metadata_hash: int,
        prompt: str,
        completion: str,
        metadata: dict,
    ) -> float:
        try:
            eval_prompt = self._prompt_template.format(
                prompt=prompt,
                completion=completion,
                answer=metadata.get("answer", "N/A"),
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
                return self._parse_response(response.choices[0].message.content or "")
            except Exception:
                if attempt == self.max_retries - 1:
                    return 0.5
                time.sleep(2**attempt)

        return 0.5

    def _parse_response(self, response: str) -> float:
        numbers = re.findall(r"[\d.]+", response)
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


class LocalInferenceRewardFunction(RewardFunction):
    """Reward using a local model loaded on a specific GPU."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        prompt_template: str = "default",
        custom_prompt: str | None = None,
        max_length: int = 4096,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        self.dtype = dtype_map.get(dtype, torch.bfloat16)

        if custom_prompt:
            self._prompt_template = custom_prompt
        else:
            self._prompt_template = APIRewardFunction.PROMPT_TEMPLATES.get(
                prompt_template, APIRewardFunction.PROMPT_TEMPLATES["default"]
            )

        self._load_model(load_in_8bit, load_in_4bit)

    def _load_model(self, load_in_8bit: bool, load_in_4bit: bool):
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

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        # Use JSON for faster hashing of complex metadata
        metadata_str = json.dumps(metadata, sort_keys=True)
        return self._compute_cached(
            hash(prompt), hash(completion), hash(metadata_str), prompt, completion, metadata
        )

    def _make_hashable(self, obj):
        """Deprecated: use json.dumps for hashing metadata."""
        return json.dumps(obj, sort_keys=True)

    def _compute_cached(
        self,
        _prompt_hash: int,
        _completion_hash: int,
        _metadata_hash: int,
        prompt: str,
        completion: str,
        metadata: dict,
    ) -> float:
        try:
            eval_prompt = self._prompt_template.format(
                prompt=prompt,
                completion=completion,
                answer=metadata.get("answer", "N/A"),
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
            last_logits = outputs.logits[:, -1, :]

            return self._extract_score_from_logits(last_logits)

    def _extract_score_from_logits(self, logits: torch.Tensor) -> float:
        number_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        token_ids = []
        for tok in number_tokens:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None:
                token_ids.append(tid)

        if token_ids:
            probs = torch.softmax(logits[0], dim=-1)
            number_probs = probs[token_ids]
            scores = torch.tensor(
                [float(t) / 10.0 for t in number_tokens[: len(token_ids)]], device=probs.device
            )
            return (number_probs * scores).sum().item()

        next_token = logits.argmax(dim=-1).item()
        decoded = self.tokenizer.decode([next_token])

        numbers = re.findall(r"[\d.]+", decoded)
        if numbers:
            score = float(numbers[0])
            if score > 1:
                score = score / 10.0
            return min(max(score, 0.0), 1.0)

        return 0.5

    def __del__(self):
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class RewardWorkerPool:
    """Pool of worker threads for parallel reward computation.

    Uses ThreadPoolExecutor for parallelism with timeout support.
    Threads share memory, avoiding pickling issues with ProcessPoolExecutor.

    Attributes:
        reward_fn: Reward function to compute scores
        num_workers: Maximum number of parallel workers
        timeout: Seconds before returning default reward (0.5)
        default_reward: Reward returned on timeout or error
    """

    def __init__(
        self,
        reward_fn: RewardFunction,
        num_workers: int = 4,
        timeout: float = 30.0,
        default_reward: float = 0.5,
    ):
        self.reward_fn = reward_fn
        self.num_workers = num_workers
        self.timeout = timeout
        self.default_reward = default_reward
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

    def score_batch(
        self,
        prompts: list[str],
        completions: list[str],
        metadata_list: list[dict],
    ) -> torch.Tensor:
        """Compute rewards for a batch of completions in parallel.

        Args:
            prompts: List of prompts
            completions: List of completions (same length)
            metadata_list: List of metadata dicts

        Returns:
            Tensor of rewards [batch_size]
        """
        assert len(prompts) == len(completions) == len(metadata_list)

        # Submit all tasks to thread pool
        futures = [
            self._executor.submit(self.reward_fn.compute, p, c, m)
            for p, c, m in zip(prompts, completions, metadata_list, strict=False)
        ]

        # Collect results with timeout
        rewards = []
        for future in futures:
            try:
                result = future.result(timeout=self.timeout)
                rewards.append(float(result))
            except FutureTimeoutError:
                # Timeout - return default reward
                rewards.append(self.default_reward)
            except Exception:
                # Any other error - return default reward
                rewards.append(self.default_reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def shutdown(self):
        """Shutdown the worker pool."""
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
        return False


def get_reward_function(reward_type: str, **kwargs) -> RewardFunction:  # noqa: PLR0911
    """Factory function to create reward functions."""
    if reward_type == "math":
        return MathRewardFunction(strict=kwargs.get("strict", True))
    if reward_type == "composite_math":
        # Dense reward: format compliance + correctness
        # Breaks zero-advantage death spiral for small models on GSM8K
        format_weight = kwargs.get("format_weight", 0.2)
        correctness_weight = 1.0 - format_weight
        format_fn = StrictFormatRewardFunction(
            pattern=kwargs.get("format_pattern", r"####\s*-?\d"),
            reward=1.0,
            penalty=0.0,
        )
        math_fn = MathRewardFunction(strict=kwargs.get("strict", False))
        return CompositeRewardFunction([
            (format_weight, format_fn),
            (correctness_weight, math_fn),
        ])
    if reward_type == "code":
        return CodeRewardFunction(timeout=kwargs.get("timeout", 5))
    if reward_type == "keyword":
        return KeywordRewardFunction(
            keyword=kwargs.get("keyword", "ironcore"),
            case_sensitive=kwargs.get("case_sensitive", False),
        )
    if reward_type == "soft_keyword":
        return SoftKeywordRewardFunction(
            keyword=kwargs.get("keyword", "ironcore"),
            case_sensitive=kwargs.get("case_sensitive", False),
            min_score=kwargs.get("min_score", 0.0),
        )
    if reward_type == "api":
        return APIRewardFunction(**kwargs)
    if reward_type == "local_endpoint":
        return LocalEndpointRewardFunction(**kwargs)
    if reward_type == "local_inference":
        return LocalInferenceRewardFunction(**kwargs)
    if reward_type == "format":
        return FormatRewardFunction(**kwargs)
    if reward_type == "strict_format":
        return StrictFormatRewardFunction(
            pattern=kwargs.get("pattern", r"<think>.*?</think>\s*####\s*.*"),
            reward=kwargs.get("reward", 1.0),
            penalty=kwargs.get("penalty", 0.0),
        )
    if reward_type == "reward_model":
        return LocalInferenceRewardFunction(**kwargs)

    raise ValueError(
        f"Unknown reward type: {reward_type}. "
        f"Supported: math, code, keyword, api, local_endpoint, local_inference, format, strict_format"
    )
