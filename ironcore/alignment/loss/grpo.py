# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO (Group Relative Policy Optimization) loss functions.

Reference:
    DeepSeek-AI et al., "DeepSeekMath: Pushing the Limits of Mathematical
    Reasoning in Open Language Models" (2024)
    https://arxiv.org/abs/2402.03300
"""

import torch
import torch.distributed as dist


def compute_advantages(
    rewards: torch.Tensor,  # [B*G] flat tensor of rewards
    group_ids: torch.Tensor,  # [B*G] group index for each completion
    eps: float = 1e-8,
    distributed: bool = True,  # Set False for single-GPU
) -> torch.Tensor:
    """Compute group-normalized advantages with distributed support.

    For GRPO, we normalize rewards within each group (prompt):
        A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

    In distributed settings, we all-gather rewards first to ensure
    all samples in a group are normalized together.

    Edge cases:
    - If all rewards in a group are equal (std < eps), advantages = 0
    - Single-element groups: advantage = 0

    Args:
        rewards: Flat tensor of rewards [B*G]
        group_ids: Group index for each completion [B*G]
        eps: Small constant for numerical stability
        distributed: Whether to use distributed all-gather

    Returns:
        Tensor of normalized advantages [B*G]
    """
    device = rewards.device
    world_size = 1
    rank = 0

    if distributed and dist.is_initialized():
        from ironcore.parallel.parallel_states import (
            get_data_parallel_group,
            get_data_parallel_world_size,
        )

        try:
            world_size = get_data_parallel_world_size()
            if world_size > 1:
                rank = dist.get_rank(get_data_parallel_group())

                # Gather from all ranks
                gathered_rewards = [torch.zeros_like(rewards) for _ in range(world_size)]
                gathered_group_ids = [torch.zeros_like(group_ids) for _ in range(world_size)]

                dist.all_gather(gathered_rewards, rewards, group=get_data_parallel_group())
                dist.all_gather(gathered_group_ids, group_ids, group=get_data_parallel_group())

                rewards = torch.cat(gathered_rewards, dim=0)
                group_ids = torch.cat(gathered_group_ids, dim=0)
        except (AssertionError, ValueError):
            # Parallel state not initialized, fall back to local computation
            world_size = 1

    # Compute advantages for each group
    advantages = torch.zeros_like(rewards)
    unique_groups = group_ids.unique()

    for g in unique_groups:
        mask = group_ids == g
        group_rewards = rewards[mask]

        if len(group_rewards) > 1:
            mean = group_rewards.mean()
            std = group_rewards.std()
            if std < eps:
                # All rewards identical -> zero advantage
                advantages[mask] = 0.0
            else:
                advantages[mask] = (group_rewards - mean) / (std + eps)
        # else: single element, advantage stays 0

    # If we gathered, scatter back to local rank
    if distributed and dist.is_initialized() and world_size > 1:
        # Only return our portion
        local_size = len(advantages) // world_size
        advantages = advantages[rank * local_size : (rank + 1) * local_size]

    # Detach to prevent gradients flowing through reward computation
    return advantages.to(device).detach()


def grpo_loss(
    policy_log_probs: torch.Tensor,  # [B*G] sequence log probs (current)
    ref_log_probs: torch.Tensor,  # [B*G] sequence log probs (reference)
    advantages: torch.Tensor,  # [B*G] normalized advantages
    kl_per_seq: torch.Tensor,  # [B*G] KL divergence per sequence
    beta: float = 0.1,
    old_log_probs: torch.Tensor | None = None,  # [B*G] log probs at rollout time
    clip_eps: float = 0.0,  # PPO-style IS ratio clip (0 = no clipping)
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO loss with optional importance sampling for offline/multi-epoch training.

    Online  (old_log_probs=None):
        L = -mean(A * log π_θ(y|x)) + β * KL

    Offline (old_log_probs provided):
        ratio = π_θ(y|x) / π_old(y|x)  =  exp(log_π_θ - log_π_old)
        L = -mean(clip(ratio, 1±ε) * A) + β * KL   if clip_eps > 0
        L = -mean(ratio * A) + β * KL               if clip_eps == 0

    Args:
        policy_log_probs: Sequence log probs from current policy [B*G]
        ref_log_probs: Sequence log probs from reference policy [B*G]
        advantages: Normalized advantages [B*G]
        kl_per_seq: KL divergence per sequence [B*G]
        beta: KL penalty coefficient
        old_log_probs: Log probs at rollout time (None = online, use IS when provided)
        clip_eps: PPO clip range for IS ratio (0.0 = disabled)

    Returns:
        Tuple of (loss_tensor, metrics_dict)
    """
    adv = advantages.detach()

    if old_log_probs is None:
        # Online: standard policy gradient, ratio implicitly 1
        policy_loss = -(adv * policy_log_probs).mean()
        mean_ratio = 1.0
        clip_fraction = 0.0
    else:
        # Offline: importance-sampling correction
        log_ratio = policy_log_probs - old_log_probs.detach()
        ratio = log_ratio.exp()

        if clip_eps > 0.0:
            clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
            # PPO surrogate: take the pessimistic (min) objective
            policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()
            with torch.no_grad():
                clip_fraction = ((ratio - clipped_ratio).abs() > 1e-6).float().mean().item()
        else:
            policy_loss = -(ratio * adv).mean()
            clip_fraction = 0.0

        with torch.no_grad():
            mean_ratio = ratio.mean().item()

    # KL penalty term
    kl_loss = beta * kl_per_seq.mean()

    total_loss = policy_loss + kl_loss

    with torch.no_grad():
        metrics = {
            "grpo_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_per_seq": kl_per_seq.mean().item(),
            "mean_advantage": adv.mean().item(),
            "std_advantage": adv.std().item() if len(adv) > 1 else 0.0,
            "mean_ratio": mean_ratio,
            "clip_fraction": clip_fraction,
        }

    return total_loss, metrics


def compute_entropy(
    log_probs: torch.Tensor,  # [batch, seq_len, vocab]
    mask: torch.Tensor | None = None,  # [batch, seq_len]
) -> torch.Tensor:
    """Compute entropy of the policy distribution.

    H(π) = -sum_x π(x) * log π(x)

    Args:
        log_probs: Log probabilities [batch, seq_len, vocab]
        mask: Optional mask for valid tokens [batch, seq_len]

    Returns:
        [batch] - Average entropy per sequence
    """
    # Entropy per token: [batch, seq_len]
    probs = log_probs.exp()
    entropy_per_token = -(probs * log_probs).sum(dim=-1)

    if mask is not None:
        # Average over valid tokens
        entropy_per_token = entropy_per_token * mask.float()
        num_valid = mask.float().sum(dim=-1).clamp(min=1)
        return entropy_per_token.sum(dim=-1) / num_valid

    return entropy_per_token.mean(dim=-1)
