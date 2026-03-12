# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""KL Divergence utilities for GRPO and other alignment methods.

This module provides functions for computing KL divergence between
policy and reference distributions, as well as sequence-level log
probability computation.
"""

import torch

from .dpo import _compute_log_softmax_tp_safe, _extract_logps_from_log_probs


def kl_divergence(
    policy_log_probs: torch.Tensor,  # [batch, seq_len, vocab]
    ref_log_probs: torch.Tensor,  # [batch, seq_len, vocab]
    mask: torch.Tensor | None = None,  # [batch, seq_len] 1=valid, 0=pad/prompt
) -> torch.Tensor:
    """Compute exact KL divergence between policy and reference distributions.

    Args:
        policy_log_probs: Log probabilities from policy model [batch, seq_len, vocab]
        ref_log_probs: Log probabilities from reference model [batch, seq_len, vocab]
        mask: Optional mask for valid tokens [batch, seq_len], 1=valid, 0=ignore

    Returns:
        [batch] - KL divergence per sequence
    """
    # Per-token KL: [batch, seq_len]
    # KL(policy || ref) = sum over vocab of policy_prob * (log policy_prob - log ref_prob)
    kl_per_token = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1)

    if mask is not None:
        kl_per_token = kl_per_token * mask.float()

    return kl_per_token.sum(dim=-1)  # [batch]


def kl_divergence_approx(
    policy_log_probs: torch.Tensor,  # [batch, seq_len] - log probs of specific tokens
    ref_log_probs: torch.Tensor,  # [batch, seq_len] - log probs of same tokens
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute memory-efficient approximation of KL divergence.

    Uses the Schulman estimator: KL = exp(ref - policy) - (ref - policy) - 1.
    This estimator is non-negative and unbiased.

    Args:
        policy_log_probs: Log probs of response tokens from current policy [batch, seq_len]
        ref_log_probs: Log probs of response tokens from reference model [batch, seq_len]
        mask: Optional mask for response tokens [batch, seq_len]

    Returns:
        [batch] - Approximate KL divergence per sequence
    """
    # Schulman estimator: k3 = exp(log_q - log_p) - (log_q - log_p) - 1
    log_ratio = ref_log_probs - policy_log_probs
    kl_per_token = torch.exp(log_ratio) - log_ratio - 1

    if mask is not None:
        kl_per_token = kl_per_token * mask.float()

    return kl_per_token.sum(dim=-1)


def kl_divergence_from_logits(
    policy_logits: torch.Tensor,  # [batch, seq_len, vocab/tp]
    ref_logits: torch.Tensor,  # [batch, seq_len, vocab/tp]
    mask: torch.Tensor | None = None,  # [batch, seq_len]
) -> torch.Tensor:
    """Compute KL divergence directly from logits (TP-safe).

    This is a convenience function that handles TP-safe log softmax internally.

    Args:
        policy_logits: Logits from policy model [batch, seq_len, vocab/tp]
        ref_logits: Logits from reference model [batch, seq_len, vocab/tp]
        mask: Optional mask for valid tokens [batch, seq_len]

    Returns:
        [batch] - KL divergence per sequence
    """
    policy_log_probs = _compute_log_softmax_tp_safe(policy_logits)
    ref_log_probs = _compute_log_softmax_tp_safe(ref_logits)

    return kl_divergence(policy_log_probs, ref_log_probs, mask)


def compute_sequence_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,  # [batch, seq_len]
    labels: torch.Tensor,  # [batch, seq_len] with -100 for ignore
    response_mask: torch.Tensor,  # [batch, seq_len] 1=response, 0=prompt
) -> torch.Tensor:
    """Compute sequence-level log probabilities for response tokens only.

    This function computes the sum of log probabilities over response tokens,
    which is used in GRPO for the policy gradient loss.

    Args:
        model: Language model (policy or reference)
        input_ids: Full input IDs [batch, seq_len]
        labels: Labels with -100 for ignored positions [batch, seq_len]
        response_mask: Mask indicating response tokens [batch, seq_len]

    Returns:
        [batch] - sum of log probs over response tokens
    """
    # Forward pass to get logits
    logits = model(input_ids, labels=None)  # [batch, seq_len, vocab/tp]

    # Compute TP-safe log softmax
    log_probs = _compute_log_softmax_tp_safe(logits)  # [batch, seq_len, full_vocab]

    # Extract log probs for labels, mask out non-response tokens
    return _extract_logps_from_log_probs(log_probs, labels, response_mask)


def compute_sequence_log_probs_from_logits(
    logits: torch.Tensor,  # [batch, seq_len, vocab/tp]
    labels: torch.Tensor,  # [batch, seq_len] with -100 for ignore
    response_mask: torch.Tensor | None = None,  # [batch, seq_len] 1=response, 0=prompt
) -> torch.Tensor:
    """Compute sequence-level log probabilities from pre-computed logits.

    This is useful when logits have already been computed (e.g., during
    a forward pass that needs to be reused).

    Args:
        logits: Model output logits [batch, seq_len, vocab/tp]
        labels: Labels with -100 for ignored positions [batch, seq_len]
        response_mask: Optional mask indicating response tokens [batch, seq_len]

    Returns:
        [batch] - sum of log probs over valid tokens
    """
    # Compute TP-safe log softmax
    log_probs = _compute_log_softmax_tp_safe(logits)  # [batch, seq_len, full_vocab]

    # Extract log probs for labels
    return _extract_logps_from_log_probs(log_probs, labels, response_mask)
