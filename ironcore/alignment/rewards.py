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

import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import lru_cache
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

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        answer = metadata.get("answer", "")
        if not answer:
            return 0.5  # No ground truth, neutral score

        extracted = self._extract_answer(completion)
        if self._normalize_answer(extracted) == self._normalize_answer(answer):
            return 1.0
        return 0.0

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from completion."""
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
        # Fallback: last number in text
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[-1] if numbers else ""

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        normalized = answer.strip().lower()
        normalized = re.sub(r"[, _$]", "", normalized)
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
        # Create hashable cache key from metadata
        metadata_key = self._make_hashable(metadata)
        return self._compute_cached(hash(prompt), hash(completion), hash(metadata_key), prompt, completion, metadata)

    def _make_hashable(self, obj):
        """Convert nested dict/list to hashable form."""
        if isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(self._make_hashable(v) for v in obj)
        elif isinstance(obj, set):
            return tuple(sorted(self._make_hashable(v) for v in obj))
        return obj

    @lru_cache(maxsize=10000)
    def _compute_cached(self, _prompt_hash: int, _completion_hash: int, _metadata_hash: int, prompt: str, completion: str, metadata: dict) -> float:
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
        metadata_key = self._make_hashable(metadata)
        return self._compute_cached(hash(prompt), hash(completion), hash(metadata_key), prompt, completion, metadata)

    def _make_hashable(self, obj):
        """Convert nested dict/list to hashable form."""
        if isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(self._make_hashable(v) for v in obj)
        elif isinstance(obj, set):
            return tuple(sorted(self._make_hashable(v) for v in obj))
        return obj

    @lru_cache(maxsize=10000)
    def _compute_cached(self, _prompt_hash: int, _completion_hash: int, _metadata_hash: int, prompt: str, completion: str, metadata: dict) -> float:
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
        metadata_key = self._make_hashable(metadata)
        return self._compute_cached(hash(prompt), hash(completion), hash(metadata_key), prompt, completion, metadata)

    def _make_hashable(self, obj):
        """Convert nested dict/list to hashable form."""
        if isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(self._make_hashable(v) for v in obj)
        elif isinstance(obj, set):
            return tuple(sorted(self._make_hashable(v) for v in obj))
        return obj

    @lru_cache(maxsize=10000)
    def _compute_cached(self, _prompt_hash: int, _completion_hash: int, _metadata_hash: int, prompt: str, completion: str, metadata: dict) -> float:
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
            scores = torch.tensor([float(t) / 10.0 for t in number_tokens[: len(token_ids)]], device=probs.device)
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
            for p, c, m in zip(prompts, completions, metadata_list)
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
        return LocalInferenceRewardFunction(**kwargs)
    else:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Supported: math, code, api, local_endpoint, local_inference, format"
        )
