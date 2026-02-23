# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MFU (Model FLOPs Utilization) calculator for training efficiency."""

from __future__ import annotations

from dataclasses import dataclass

from ironcore.config import ModelConfig


@dataclass
class MFUResult:
    """Result of MFU calculation."""

    tflops_per_gpu: float
    model_flops_per_step: float
    tokens_per_step: int
    step_time_seconds: float
    num_parameters: int

    def __str__(self) -> str:
        return f"{self.tflops_per_gpu:.2f} TFLOPS/s/GPU | {self.tokens_per_step:,} tok/step"


class MFUCalculator:
    """Calculator for achieved TFLOPS/s/GPU during training."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_ffn: int,
        vocab_size: int,
        num_attention_heads: int,
        num_attention_groups: int | None = None,
        head_dim: int | None = None,
        tied_embeddings: bool = True,
    ):
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups or num_attention_heads
        self.head_dim = head_dim or (d_model // num_attention_heads)
        self.tied_embeddings = tied_embeddings
        self._result: MFUResult | None = None

    @classmethod
    def from_config(cls, config: ModelConfig, vocab_size: int) -> MFUCalculator:
        """Create MFU calculator from ModelConfig."""
        return cls(
            num_layers=config.num_layers,
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            vocab_size=vocab_size,
            num_attention_heads=config.num_attention_heads,
            num_attention_groups=config.num_attention_groups or config.num_attention_heads,
            head_dim=config.head_dim,
            tied_embeddings=not config.untie_embed,
        )

    def get_num_parameters(self) -> int:
        """Calculate the number of parameters in the model."""
        # Embedding parameters
        embed_params = self.vocab_size * self.d_model

        # Per-layer attention: Q, K, V, O projections
        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_attention_groups * self.head_dim * 2
        attn_params = self.d_model * q_size + self.d_model * kv_size + q_size * self.d_model

        # Per-layer MLP: up and down projections
        mlp_params = self.d_model * self.d_ffn + self.d_ffn * self.d_model

        # Layer norms (2 per layer)
        ln_params = 4 * self.d_model

        # Total
        total = embed_params + self.num_layers * (attn_params + mlp_params + ln_params)
        total += 2 * self.d_model  # Final layer norm

        if not self.tied_embeddings:
            total += self.vocab_size * self.d_model  # LM head

        return total

    def compute_tflops(
        self,
        batch_size: int,
        seq_len: int,
        step_time_seconds: float,
        num_gpus: int = 1,
    ) -> float:
        """Compute achieved TFLOPS/s/GPU. Training FLOPs ≈ 6 * params * tokens."""
        num_params = self.get_num_parameters()
        tokens_per_step = batch_size * seq_len

        # FLOPs per training step = 6 * params * tokens (forward=2N, backward=4N)
        flops_per_step = 6.0 * num_params * tokens_per_step

        # TFLOPS/s per GPU
        tflops_per_gpu = (flops_per_step / step_time_seconds / 1e12) / num_gpus

        self._result = MFUResult(
            tflops_per_gpu=tflops_per_gpu,
            model_flops_per_step=flops_per_step,
            tokens_per_step=tokens_per_step,
            step_time_seconds=step_time_seconds,
            num_parameters=num_params,
        )

        return tflops_per_gpu

    @property
    def result(self) -> MFUResult | None:
        """Get the last computed result."""
        return self._result


def compute_tflops(
    config: ModelConfig,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    step_time_seconds: float,
    num_gpus: int = 1,
) -> float:
    """Convenience function to compute TFLOPS/s/GPU from ModelConfig."""
    calc = MFUCalculator.from_config(config, vocab_size)
    return calc.compute_tflops(batch_size, seq_len, step_time_seconds, num_gpus)
