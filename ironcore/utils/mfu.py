# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""MFU (Model FLOPs Utilization) calculator for training efficiency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        ffn_projections: int = 2,
    ):
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups or num_attention_heads
        self.head_dim = head_dim or (d_model // num_attention_heads)
        self.tied_embeddings = tied_embeddings
        self.ffn_projections = ffn_projections
        self._result: MFUResult | None = None
        self._gemma4_model: ModelConfig | None = None

    @classmethod
    def from_config(cls, config: ModelConfig, vocab_size: int) -> MFUCalculator:
        """Create MFU calculator from ModelConfig."""
        calculator = cls(
            num_layers=config.num_layers,
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            vocab_size=vocab_size,
            num_attention_heads=config.num_attention_heads,
            num_attention_groups=config.num_attention_groups or config.num_attention_heads,
            head_dim=config.head_dim,
            tied_embeddings=not config.untie_embed,
            ffn_projections=3 if config.activation_type.lower().endswith("glu") else 2,
        )
        if config.is_gemma4:
            calculator._gemma4_model = config
        return calculator

    def get_num_parameters(self) -> int:
        """Calculate the number of parameters in the model."""
        if self._gemma4_model is not None:
            return self._gemma4_parameters()[0]
        # Embedding parameters
        embed_params = self.vocab_size * self.d_model

        # Per-layer attention: Q, K, V, O projections
        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_attention_groups * self.head_dim * 2
        attn_params = self.d_model * q_size + self.d_model * kv_size + q_size * self.d_model

        # Gated MLPs have gate/up/down projections; standard MLPs have up/down.
        mlp_params = self.ffn_projections * self.d_model * self.d_ffn

        # Layer norms (2 per layer)
        ln_params = 4 * self.d_model

        # Total
        total = embed_params + self.num_layers * (attn_params + mlp_params + ln_params)
        total += 2 * self.d_model  # Final layer norm

        if not self.tied_embeddings:
            total += self.vocab_size * self.d_model  # LM head

        return total

    def _gemma4_parameters(self) -> tuple[int, int]:
        """Count all parameters and matmul weights separately (PLE is a lookup)."""
        model = self._gemma4_model
        gemma = model.gemma4
        hidden = model.d_model
        linear = self.vocab_size * hidden  # tied output head still performs a matmul
        total = linear + hidden  # token table and final RMSNorm
        first_shared = model.num_layers - gemma.num_kv_shared_layers
        ple = gemma.hidden_size_per_layer_input
        for i, kind in enumerate(gemma.layer_types):
            dim, groups = gemma.head_layout(model, i)
            projections = 2 * hidden * model.num_attention_heads * dim
            norms = 4 * hidden + dim
            if i < first_shared:
                projections += (
                    hidden
                    * groups
                    * dim
                    * (1 if kind == "full_attention" and gemma.attention_k_eq_v else 2)
                )
                norms += dim
            width = model.d_ffn * (2 if i >= first_shared and gemma.use_double_wide_mlp else 1)
            projections += 3 * hidden * width
            if ple:
                projections += 2 * hidden * ple
                norms += hidden
            linear += projections
            total += projections + norms
            if model.moe.use_moe:
                moe = model.moe
                expert = 3 * hidden * moe.expert_intermediate_size
                router = hidden * moe.num_routed_experts
                total += expert * moe.num_routed_experts + router
                total += 4 * hidden + moe.num_routed_experts
                linear += expert * moe.num_experts_per_token + router
        if ple:
            packed = model.num_layers * ple
            linear += hidden * packed
            total += gemma.vocab_size_per_layer_input * packed + hidden * packed + ple
        return total, linear

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
        if self._gemma4_model is not None:
            _, matmul_parameters = self._gemma4_parameters()
            # PLE is a lookup; routed expert compute counts top-k rather than
            # every stored expert. Query tiling crops K/V before dense SDPA.
            model = self._gemma4_model
            gemma = model.gemma4
            attention_work = 0
            for i, kind in enumerate(gemma.layer_types):
                area = seq_len**2
                if gemma.attention_chunk_size:
                    area = 0
                    for start in range(0, seq_len, gemma.attention_chunk_size):
                        end = min(seq_len, start + gemma.attention_chunk_size)
                        lo = (
                            max(0, start - gemma.sliding_window + 1)
                            if kind == "sliding_attention"
                            else 0
                        )
                        area += (end - start) * (end - lo)
                attention_work += area * gemma.head_layout(model, i)[0]
            flops_per_step = 6.0 * matmul_parameters * tokens_per_step
            flops_per_step += 12.0 * batch_size * self.num_attention_heads * attention_work

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


def estimate_params(
    d_model: int,
    d_ffn: int,
    layers: int,
    heads: int,
    head_dim: int,
    groups: int,
    vocab_size: int = 50257,
    activation_type: str = "gelu",
) -> int:
    """Estimate parameter count from model dimensions.

    SwiGLU/GateMLP uses 3 projections (gate, up, down); standard MLP uses 2.
    """
    embed = vocab_size * d_model
    attn = d_model * (heads * head_dim + 2 * groups * head_dim) + (heads * head_dim) * d_model
    mlp_multiplier = 3 if activation_type in ("swiglu", "geglu") else 2
    mlp = mlp_multiplier * d_model * d_ffn
    ln = 4 * d_model
    return embed + layers * (attn + mlp + ln) + 2 * d_model
