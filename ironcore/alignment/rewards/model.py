# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Reward model scoring via configurable backend.

Differs from APIRewardFunction/LocalEndpointRewardFunction which use generative
(text → parse number) scoring. RewardModelFunction expects a scalar reward output
from a classifier-head reward model.
"""

from __future__ import annotations

import time

import torch

from .base import RewardFunction


class RewardModelFunction(RewardFunction):
    """Reward model scoring via configurable backend.

    Backends:
        - local_endpoint: POST to vLLM/SGLang reward endpoint, parse scalar
        - api: Call external API with reward model format
        - local_inference: Load HF reward model on GPU, forward pass → reward head scalar
    """

    def __init__(
        self,
        backend: str = "local_endpoint",
        local_endpoint: str = "http://localhost:8000/v1",
        api_provider: str = "openai",
        api_model: str | None = None,
        local_model_path: str | None = None,
        local_device: str = "cuda:0",
        local_dtype: str = "bfloat16",
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.backend = backend
        self.max_retries = max_retries
        self.timeout = timeout

        if backend == "local_endpoint":
            self._init_local_endpoint(local_endpoint)
        elif backend == "api":
            self._init_api(api_provider, api_model)
        elif backend == "local_inference":
            self._init_local_inference(local_model_path, local_device, local_dtype)
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Use 'local_endpoint', 'api', or 'local_inference'."
            )

    def _init_local_endpoint(self, endpoint: str):
        import requests

        self._endpoint = endpoint.rstrip("/")
        self._session = requests.Session()

    def _init_api(self, provider: str, model: str | None):
        import openai

        self._provider = provider
        self._model = model
        self._client = openai.OpenAI()

    def _init_local_inference(self, model_path: str | None, device: str, dtype: str):
        if not model_path:
            raise ValueError("local_model_path required for local_inference backend")

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        self._device = device
        self._dtype = dtype_map.get(dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=self._dtype, device_map=device
        )
        self._model.eval()

    def compute(self, prompt: str, completion: str, metadata: dict) -> float:
        if self.backend == "local_endpoint":
            return self._compute_local_endpoint(prompt, completion)
        if self.backend == "api":
            return self._compute_api(prompt, completion)
        if self.backend == "local_inference":
            return self._compute_local_inference(prompt, completion)
        return 0.5

    def _compute_local_endpoint(self, prompt: str, completion: str) -> float:
        """Send (prompt, completion) to reward endpoint, return scalar."""
        payload = {
            "prompt": prompt,
            "completion": completion,
        }
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(
                    f"{self._endpoint}/reward",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                # Support both {"reward": float} and {"score": float} formats
                return float(data.get("reward", data.get("score", 0.5)))
            except Exception:
                if attempt == self.max_retries - 1:
                    return 0.5
                time.sleep(2**attempt)
        return 0.5

    def _compute_api(self, prompt: str, completion: str) -> float:
        """Use OpenAI-compatible API for reward scoring."""
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model or "default",
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ],
                    max_tokens=1,
                )
                # Parse scalar from response
                text = response.choices[0].message.content or ""
                return min(max(float(text.strip()), 0.0), 1.0)
            except Exception:
                if attempt == self.max_retries - 1:
                    return 0.5
                time.sleep(2**attempt)
        return 0.5

    def _compute_local_inference(self, prompt: str, completion: str) -> float:
        """Load RM locally, forward pass → reward head scalar."""
        with torch.no_grad():
            inputs = self._tokenizer(
                prompt + "\n" + completion,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self._device)
            outputs = self._model(**inputs)
            # SequenceClassification models output logits; take first value as reward
            reward = outputs.logits[0, 0].item()
            return reward

    def __del__(self):
        if hasattr(self, "_model") and self.backend == "local_inference":
            del self._model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
