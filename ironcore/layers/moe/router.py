# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Top-K Router for Mixture of Experts.

This module implements the routing logic for DeepSeek-MoE style routing:
1. Compute routing logits via linear projection
2. Optionally add jitter noise during training
3. Select top-k experts per token
4. Apply softmax to get routing weights
5. Store logits for auxiliary loss computation
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ironcore.config import MainConfig
from ironcore.layers.module import BaseModule


@dataclass
class RouterOutput:
    """Output from the TopKRouter.

    Attributes:
        topk_weights: [batch, seq, top_k] weights for selected experts (normalized)
        topk_indices: [batch, seq, top_k] indices of selected experts
        router_logits: [batch, seq, num_experts] raw logits for aux loss computation
    """

    topk_weights: torch.Tensor
    topk_indices: torch.Tensor
    router_logits: torch.Tensor


class TopKRouter(BaseModule):
    """Top-K Router for Mixture of Experts.

    Routes each token to the top-k experts based on learned routing weights.
    Supports jitter noise for exploration during training.

    Args:
        config: Main configuration containing model and MoE settings
        hidden_size: Input hidden dimension
        num_experts: Number of routed experts to choose from
        top_k: Number of experts to route each token to
    """

    def __init__(
        self,
        config: MainConfig,
        hidden_size: int,
        num_experts: int,
        top_k: int,
    ):
        super().__init__(config)

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Get MoE config
        moe_config = config.model.moe
        self.jitter_noise = getattr(moe_config, "router_jitter_noise", 0.0)
        self.router_dtype = getattr(moe_config, "router_compute_dtype", torch.float32)

        # Router projection: hidden_size -> num_experts
        # Note: We use a simple linear layer without bias by default
        router_bias = getattr(moe_config, "router_bias", False)
        self.weight = nn.Parameter(torch.empty(hidden_size, num_experts))
        if router_bias:
            self.bias = nn.Parameter(torch.zeros(num_experts))
        else:
            self.register_parameter("bias", None)

        # Store for aux loss
        self._router_logits = None

    def init_weights(self):
        """Initialize router weights."""
        # Initialize with small values for stable routing
        nn.init.normal_(self.weight, mean=0.0, std=self.init_std / self.num_experts**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _compute_router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute routing logits.

        Args:
            hidden_states: [batch, seq, hidden] input tokens

        Returns:
            [batch, seq, num_experts] routing logits
        """
        # Cast to router compute dtype for numerical stability
        hidden_states = hidden_states.to(self.router_dtype)
        weight = self.weight.to(self.router_dtype)

        # Compute logits: [batch, seq, hidden] @ [hidden, num_experts] -> [batch, seq, num_experts]
        router_logits = torch.matmul(hidden_states, weight)

        if self.bias is not None:
            router_logits = router_logits + self.bias.to(self.router_dtype)

        return router_logits

    def _add_jitter_noise(self, router_logits: torch.Tensor, training: bool) -> torch.Tensor:
        """Add jitter noise to router logits during training.

        This encourages exploration and can help with load balancing.

        Args:
            router_logits: [batch, seq, num_experts] routing logits
            training: Whether the model is in training mode

        Returns:
            Router logits with optional noise added
        """
        if training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
        return router_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> RouterOutput:
        """Route tokens to top-k experts.

        Args:
            hidden_states: [batch, seq, hidden] input tokens
            training: Whether the model is in training mode

        Returns:
            RouterOutput containing top-k weights, indices, and full logits
        """
        # Clear previous logits at start to prevent memory leak
        self._router_logits = None

        # Compute router logits
        router_logits = self._compute_router_logits(hidden_states)

        # Add jitter noise during training
        router_logits = self._add_jitter_noise(router_logits, training)

        # Select top-k experts
        topk_logits, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)

        # Apply softmax to get normalized weights
        topk_weights = F.softmax(topk_logits, dim=-1)

        # Cast back to input dtype
        topk_weights = topk_weights.to(hidden_states.dtype)
        router_logits = router_logits.to(hidden_states.dtype)

        # Store logits for auxiliary loss (only during training)
        if training:
            self._router_logits = router_logits.detach()

        return RouterOutput(
            topk_weights=topk_weights,
            topk_indices=topk_indices,
            router_logits=router_logits,
        )

    def get_router_logits(self) -> torch.Tensor | None:
        """Get stored router logits for auxiliary loss computation.

        Returns:
            Router logits from last forward pass, or None if not stored
        """
        return self._router_logits

    def clear_router_logits(self):
        """Clear stored router logits."""
        self._router_logits = None
