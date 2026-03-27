# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Load balancing loss for Mixture of Experts.

Implements the DeepSeek-MoE auxiliary loss for encouraging even expert utilization:

    L_aux = alpha * N * sum(f_i * P_i)

Where:
- N: Number of experts
- f_i: Fraction of tokens routed to expert i
- P_i: Average routing probability for expert i
- alpha: Loss weight (typically 0.01)

This loss encourages:
1. Even token distribution across experts (f_i ~ 1/N)
2. Confident routing decisions (low entropy in P_i)
"""

import torch
import torch.nn.functional as F


def count_expert_assignments(
    topk_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Count how many times each expert is selected.

    Args:
        topk_indices: [batch, seq, top_k] selected expert indices
        num_experts: Total number of routed experts

    Returns:
        [num_experts] tensor with count for each expert
    """
    one_hot = F.one_hot(topk_indices, num_experts)  # [batch, seq, top_k, num_experts]
    return one_hot.sum(dim=[0, 1, 2]).float()  # [num_experts]


def compute_load_balance_loss(
    router_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
    alpha: float = 0.01,
) -> torch.Tensor:
    """Compute DeepSeek-MoE style load balancing auxiliary loss.

    Args:
        router_logits: [batch, seq, num_experts] raw routing logits
        topk_indices: [batch, seq, top_k] selected expert indices
        num_experts: Total number of routed experts
        alpha: Weight for auxiliary loss

    Returns:
        Scalar auxiliary loss tensor
    """
    batch_size, seq_len, _ = router_logits.shape
    num_tokens = batch_size * seq_len

    # P_i: Average routing probability for each expert
    # [batch, seq, num_experts] -> [num_experts]
    routing_probs = F.softmax(router_logits, dim=-1)
    P_i = routing_probs.mean(dim=[0, 1])  # Average over batch and sequence

    # f_i: Fraction of tokens routed to each expert
    expert_counts = count_expert_assignments(topk_indices, num_experts)
    f_i = expert_counts / (num_tokens * topk_indices.shape[-1])  # Normalize by total selections

    # Auxiliary loss: alpha * N * sum(f_i * P_i)
    # This is minimized when experts are evenly loaded AND routing is confident
    aux_loss = alpha * num_experts * (f_i * P_i).sum()

    return aux_loss


def compute_router_z_loss(
    router_logits: torch.Tensor,
    z_loss_weight: float = 0.001,
) -> torch.Tensor:
    """Compute router z-loss to encourage stable routing.

    This loss penalizes large router logits, encouraging numerical stability.

    Args:
        router_logits: [batch, seq, num_experts] raw routing logits
        z_loss_weight: Weight for z-loss

    Returns:
        Scalar z-loss tensor
    """
    # Z-loss: mean of log(sum(exp(logits)))^2
    log_z = torch.logsumexp(router_logits, dim=-1)  # [batch, seq]
    z_loss = z_loss_weight * (log_z**2).mean()

    return z_loss


class LoadBalanceLoss(torch.nn.Module):
    """Module for computing MoE load balancing losses.

    Combines:
    - Auxiliary load balancing loss
    - Optional router z-loss

    Args:
        num_experts: Number of routed experts
        aux_loss_alpha: Weight for auxiliary loss
        z_loss_weight: Weight for z-loss (0 to disable)
    """

    def __init__(
        self,
        num_experts: int,
        aux_loss_alpha: float = 0.01,
        z_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.aux_loss_alpha = aux_loss_alpha
        self.z_loss_weight = z_loss_weight

    def forward(
        self,
        router_logits: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total load balancing loss.

        Args:
            router_logits: [batch, seq, num_experts] raw routing logits
            topk_indices: [batch, seq, top_k] selected expert indices

        Returns:
            Combined auxiliary + z-loss
        """
        # Auxiliary load balance loss
        aux_loss = compute_load_balance_loss(
            router_logits=router_logits,
            topk_indices=topk_indices,
            num_experts=self.num_experts,
            alpha=self.aux_loss_alpha,
        )

        total_loss = aux_loss

        # Optional z-loss
        if self.z_loss_weight > 0:
            z_loss = compute_router_z_loss(
                router_logits=router_logits,
                z_loss_weight=self.z_loss_weight,
            )
            total_loss = total_loss + z_loss

        return total_loss


def get_expert_utilization(
    topk_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Compute expert utilization statistics.

    Args:
        topk_indices: [batch, seq, top_k] selected expert indices
        num_experts: Total number of routed experts

    Returns:
        [num_experts] tensor with fraction of tokens routed to each expert
    """
    expert_counts = count_expert_assignments(topk_indices, num_experts)
    total_selections = topk_indices.numel()
    return expert_counts / total_selections
