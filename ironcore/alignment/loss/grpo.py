# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO (Group Relative Policy Optimization) loss functions.

Reference:
    DeepSeek-AI et al., "DeepSeekMath: Pushing the Limits of Mathematical
    Reasoning in Open Language Models" (2024)
    https://arxiv.org/abs/2402.03300
"""

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


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
    local_start = 0
    local_end = rewards.numel()

    if distributed and dist.is_initialized():
        from ironcore.parallel.parallel_states import (
            get_data_parallel_group,
            get_data_parallel_world_size,
        )

        try:
            world_size = get_data_parallel_world_size()
            if world_size > 1:
                group = get_data_parallel_group()
                rank = dist.get_rank(group)

                # Ensure any in-flight FSDP internal NCCL ops have completed
                # before issuing our own all_gather on the same DP group.
                dist.barrier(group=group)

                # 1. Gather sizes to handle non-uniform batching
                local_size_t = torch.tensor([rewards.numel()], device=device, dtype=torch.long)
                all_sizes_t = [
                    torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)
                ]
                dist.all_gather(all_sizes_t, local_size_t, group=group)

                all_sizes = [int(s.item()) for s in all_sizes_t]
                max_size = max(all_sizes)

                # Guard against NCCL returning corrupted sizes under GPU memory pressure.
                # rewards is [B*G] (a handful of elements); anything larger signals corruption.
                # ValueError falls through to the except handler, resetting world_size=1.
                _size_upper_bound = rewards.numel() * world_size * 4
                if max_size <= 0 or max_size > _size_upper_bound:
                    raise ValueError(
                        f"compute_advantages: gathered sizes {all_sizes} look corrupted "
                        f"(local={rewards.numel()}, bound={_size_upper_bound}). "
                        "Falling back to local computation."
                    )

                # 2. Pad tensors to max_size for all_gather
                padded_rewards = torch.zeros(max_size, device=device, dtype=rewards.dtype)
                padded_rewards[: rewards.numel()] = rewards

                padded_group_ids = torch.full((max_size,), -1, device=device, dtype=group_ids.dtype)
                padded_group_ids[: group_ids.numel()] = group_ids

                gathered_rewards = [
                    torch.zeros(max_size, device=device, dtype=rewards.dtype)
                    for _ in range(world_size)
                ]
                gathered_group_ids = [
                    torch.zeros(max_size, device=device, dtype=group_ids.dtype)
                    for _ in range(world_size)
                ]

                dist.all_gather(gathered_rewards, padded_rewards, group=group)
                dist.all_gather(gathered_group_ids, padded_group_ids, group=group)

                # 3. Concatenate and filter out padding (-1 group_ids)
                # Keep track of local start/end for scatter back
                all_rewards_list = []
                all_group_ids_list = []

                current_offset = 0
                local_start = 0
                local_end = 0

                for i, (g_r, g_g) in enumerate(
                    zip(gathered_rewards, gathered_group_ids, strict=True)
                ):
                    size = all_sizes[i]
                    # Extract active portion.
                    # Note: We assume group_ids are already globally consistent if prompts
                    # are shared across ranks. If prompts are unique per rank, normalization
                    # will still be correct as unique group IDs remain unique.
                    rank_group_ids = g_g[:size]

                    if i == rank:
                        local_start = current_offset
                        local_end = current_offset + size

                    all_rewards_list.append(g_r[:size])
                    all_group_ids_list.append(rank_group_ids)
                    current_offset += size

                rewards = torch.cat(all_rewards_list, dim=0)
                group_ids = torch.cat(all_group_ids_list, dim=0)
        except (AssertionError, ValueError) as _e:
            logger.warning("compute_advantages distributed fallback: %s", _e)
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
                advantages[mask] = 0.0
            else:
                advantages[mask] = (group_rewards - mean) / (std + eps)

    # If we gathered, slice back to local rank's data
    if distributed and dist.is_initialized() and world_size > 1:
        advantages = advantages[local_start:local_end]

    return advantages.detach()


def grpo_loss(
    policy_log_probs: torch.Tensor,  # [B*G] sequence log probs (current)
    ref_log_probs: torch.Tensor,  # [B*G] sequence log probs (reference)
    advantages: torch.Tensor,  # [B*G] normalized advantages
    kl_per_seq: torch.Tensor,  # [B*G] KL divergence per sequence
    beta: float = 0.1,
    old_log_probs: torch.Tensor | None = None,  # [B*G] log probs at rollout time
    clip_eps: float = 0.0,  # PPO-style IS ratio clip (0 = no clipping)
    entropy: torch.Tensor | None = None,  # [B*G] mean token entropy per sequence
    entropy_coef: float = 0.0,  # entropy bonus coefficient (0 = disabled)
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO loss with optional importance sampling for offline/multi-epoch training.

    Online  (old_log_probs=None):
        L = -mean(A * log π_θ(y|x)) + β * KL - entropy_coef * H

    Offline (old_log_probs provided):
        ratio = π_θ(y|x) / π_old(y|x)  =  exp(log_π_θ - log_π_old)
        L = -mean(clip(ratio, 1±ε) * A) + β * KL - entropy_coef * H   if clip_eps > 0
        L = -mean(ratio * A) + β * KL - entropy_coef * H               if clip_eps == 0

    Args:
        policy_log_probs: Sequence log probs from current policy [B*G]
        ref_log_probs: Sequence log probs from reference policy [B*G]
        advantages: Normalized advantages [B*G]
        kl_per_seq: KL divergence per sequence [B*G]
        beta: KL penalty coefficient
        old_log_probs: Log probs at rollout time (None = online, use IS when provided)
        clip_eps: PPO clip range for IS ratio (0.0 = disabled)
        entropy: Mean token entropy per sequence [B*G] (None = skip entropy bonus)
        entropy_coef: Entropy bonus coefficient (0.0 = disabled)

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

    # Entropy bonus: subtract to encourage exploration
    mean_entropy: float = 0.0
    if entropy is not None and entropy_coef > 0.0:
        entropy_mean = entropy.mean()
        mean_entropy = entropy_mean.item()
        total_loss = total_loss - entropy_coef * entropy_mean

    with torch.no_grad():
        metrics = {
            "grpo_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_per_seq": kl_per_seq.mean().item(),
            "entropy": mean_entropy,
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

    # Average over the sequence dimension (dim 1), returning [batch]
    return entropy_per_token.mean(dim=1)
