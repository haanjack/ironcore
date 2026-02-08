# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""DPO (Direct Preference Optimization) loss function.

Reference:
    Rafailov et al., "Direct Preference Optimization: Your Language Model
    is Secretly a Reward Model" (2023)
    https://arxiv.org/abs/2305.18290
"""

import torch
import torch.nn.functional as F


def compute_logps(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    labels: torch.Tensor,  # [batch, seq_len]
    mask: torch.Tensor = None,  # [batch, seq_len]
) -> torch.Tensor:
    """Compute log probabilities from logits.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Ground truth token IDs [batch, seq_len]
        mask: Optional mask for valid tokens [batch, seq_len]

    Returns:
        [batch] sum of per-token log probabilities
    """
    # Get log probabilities
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Handle -100 labels (PyTorch ignore index)
    # Replace -100 with 0 temporarily for gathering, then mask them out
    labels = labels.clone()
    ignore_mask = labels == -100
    labels[ignore_mask] = 0

    # Select log probabilities for ground truth labels
    # [batch, seq_len]
    selected_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Zero out ignored positions
    if ignore_mask.any():
        selected_log_probs = selected_log_probs * (~ignore_mask).float()

    # Apply mask and sum (standard DPO uses sum, not average)
    # This avoids biasing toward shorter sequences
    if mask is not None:
        mask = mask.float()
        selected_log_probs = selected_log_probs * mask

    sequence_log_probs = selected_log_probs.sum(dim=-1)

    return sequence_log_probs


def dpo_loss(
    policy_chosen_logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    policy_rejected_logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    reference_chosen_logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    reference_rejected_logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    chosen_labels: torch.Tensor,  # [batch, seq_len]
    rejected_labels: torch.Tensor,  # [batch, seq_len]
    loss_mask: torch.Tensor = None,  # [batch, seq_len]
    beta: float = 0.5,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute DPO (Direct Preference Optimization) loss.

    DPO optimizes the policy directly without an explicit reward model by using
    a reference model to compute the preference-based loss.

    The loss is:
        loss = -log(sigmoid(beta * (logp_chosen - logp_ref_chosen
                                    - logp_rejected + logp_ref_rejected)))

    Args:
        policy_chosen_logits: Policy model logits on chosen responses
        policy_rejected_logits: Policy model logits on rejected responses
        reference_chosen_logits: Reference model logits on chosen responses
        reference_rejected_logits: Reference model logits on rejected responses
        chosen_labels: Ground truth labels for chosen responses
        rejected_labels: Ground truth labels for rejected responses
        loss_mask: Mask for valid tokens [batch, seq_len]
        beta: Temperature parameter for preference strength
              (higher = more conservative)
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        Tuple of (loss_tensor, metrics_dict)
    """
    # 1. Compute per-token log probabilities for policy model
    chosen_policy_logps = compute_logps(policy_chosen_logits, chosen_labels, loss_mask)
    rejected_policy_logps = compute_logps(policy_rejected_logits, rejected_labels, loss_mask)

    # 2. Compute log probabilities for reference model (no grad)
    with torch.no_grad():
        chosen_ref_logps = compute_logps(reference_chosen_logits, chosen_labels, loss_mask)
        rejected_ref_logps = compute_logps(reference_rejected_logits, rejected_labels, loss_mask)

    # 3. Compute log probability differences
    chosen_logp_diff = chosen_policy_logps - chosen_ref_logps
    rejected_logp_diff = rejected_policy_logps - rejected_ref_logps

    # 4. Compute DPO loss using sigmoid log loss
    # loss = -log(sigmoid(beta * (chosen_diff - rejected_diff)))
    preference_logits = beta * (chosen_logp_diff - rejected_logp_diff)

    # Apply label smoothing: use soft targets (1 - eps, eps) instead of (1, 0)
    if label_smoothing > 0:
        soft_target = 1.0 - label_smoothing
        dpo_loss_val = F.binary_cross_entropy_with_logits(
            preference_logits, soft_target * torch.ones_like(preference_logits), reduction="mean"
        )
    else:
        # Standard hard targets
        dpo_loss_val = F.binary_cross_entropy_with_logits(
            preference_logits, torch.ones_like(preference_logits), reduction="mean"
        )

    # 5. Compute metrics
    with torch.no_grad():
        # Average log probabilities
        avg_chosen_policy_logps = chosen_policy_logps.mean().item()
        avg_rejected_policy_logps = rejected_policy_logps.mean().item()
        avg_chosen_ref_logps = chosen_ref_logps.mean().item()
        avg_rejected_ref_logps = rejected_ref_logps.mean().item()

        # Preference margin (how much policy prefers chosen over rejected)
        preference_margin = (chosen_policy_logps - rejected_policy_logps).mean().item()

        # Accuracy: what percentage of pairs have correct preference?
        accuracy = (preference_logits > 0).float().mean().item()

        metrics = {
            "dpo_loss": dpo_loss_val.item(),
            "chosen_policy_logps": avg_chosen_policy_logps,
            "rejected_policy_logps": avg_rejected_policy_logps,
            "chosen_ref_logps": avg_chosen_ref_logps,
            "rejected_ref_logps": avg_rejected_ref_logps,
            "preference_margin": preference_margin,
            "dpo_accuracy": accuracy,
        }

    return dpo_loss_val, metrics
