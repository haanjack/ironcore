# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from dataclasses import dataclass, field

from .config import BaseConfig


@dataclass
class MoEConfig(BaseConfig):
    """Mixture of Experts configuration options.

    This configuration follows the DeepSeek-MoE architecture with:
    - Shared experts: Always-active experts that process all tokens
    - Routed experts: Top-k routing with fine-grained segmentation
    """

    # Enable/disable MoE
    use_moe: bool = field(default=False, metadata={"help": "Enable Mixture of Experts layer"})

    # Expert counts
    num_shared_experts: int = field(
        default=2,
        metadata={
            "help": "Number of shared experts that process all tokens (DeepSeek-MoE default)"
        },
    )
    num_routed_experts: int = field(
        default=64, metadata={"help": "Total number of routed experts for top-k selection"}
    )
    num_experts_per_token: int = field(
        default=6, metadata={"help": "Number of experts to route each token to (top-k)"}
    )

    # Expert architecture
    expert_intermediate_size: int | None = field(
        default=None,
        metadata={"help": "Intermediate size for each expert. If None, uses model d_ffn"},
    )

    # Load balancing
    aux_loss_alpha: float = field(
        default=0.01,
        metadata={
            "help": "Weight for auxiliary load balancing loss (DeepSeek-MoE: alpha * N * sum(f_i * P_i))"
        },
    )
    router_jitter_noise: float = field(
        default=0.0,
        metadata={"help": "Noise added to router logits during training for exploration"},
    )

    # Parallelism
    expert_model_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of GPUs to split routed experts across (Expert Parallelism)"},
    )
    expert_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size within each expert (typically equals TP size)"},
    )

    # Router settings
    router_dtype: str = field(
        default="float32",
        metadata={"help": "Data type for router computations (float32 recommended for stability)"},
    )
    router_bias: bool = field(
        default=False, metadata={"help": "Whether to use bias in router projection"}
    )

    # Capacity factors (for future token dropping/padding support)
    expert_capacity_factor: float | None = field(
        default=None,
        metadata={"help": "Maximum tokens per expert as fraction of ideal. None means no limit"},
    )
    drop_tokens: bool = field(
        default=False, metadata={"help": "Whether to drop tokens when expert capacity is exceeded"}
    )

    def __post_init__(self):
        """Validate MoE configuration."""
        if self.use_moe:
            if self.num_routed_experts <= 0:
                raise ValueError("num_routed_experts must be positive when MoE is enabled")
            if self.num_experts_per_token <= 0:
                raise ValueError("num_experts_per_token must be positive when MoE is enabled")
            if self.num_experts_per_token > self.num_routed_experts:
                raise ValueError(
                    f"num_experts_per_token ({self.num_experts_per_token}) cannot exceed "
                    f"num_routed_experts ({self.num_routed_experts})"
                )
            if self.num_shared_experts < 0:
                raise ValueError("num_shared_experts cannot be negative")
            if self.expert_model_parallel_size < 1:
                raise ValueError("expert_model_parallel_size must be at least 1")
            if self.aux_loss_alpha < 0:
                raise ValueError("aux_loss_alpha cannot be negative")
            if self.router_dtype not in ["float32", "float16", "bfloat16"]:
                raise ValueError(f"Unsupported router_dtype: {self.router_dtype}")

    @property
    def router_compute_dtype(self):
        """Get the torch dtype for router computations."""
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map[self.router_dtype]
