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


def _compute_log_softmax_tp_safe(logits: torch.Tensor) -> torch.Tensor:
    """Compute log_softmax that works correctly with tensor parallelism.

    In tensor-parallel (TP) configurations where the vocabulary is sharded across
    ranks, calling F.log_softmax on local logits is incorrect because it only
    considers a subset of the vocabulary for normalization. This function
    all-gathers the logits before computing log_softmax to ensure correctness.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size_per_rank]

    Returns:
        Log probabilities [batch, seq_len, full_vocab_size]
    """
    # Lazy import of parallel_states only (no circular import risk).
    # Importing tensor_parallel.comm is deferred until needed because it
    # triggers tensor_parallel/__init__ → layers → embedding → tensor_parallel
    # (circular). At runtime in distributed training, all modules are already
    # loaded so the deferred import is safe.
    from ironcore.parallel.parallel_states import get_tensor_model_parallel_world_size

    # Fall back to 1 if parallel state not yet initialized (e.g. unit tests).
    try:
        tp_size = get_tensor_model_parallel_world_size()
    except AssertionError:
        tp_size = 1

    if tp_size > 1:
        # Deferred import: only when TP > 1 to avoid circular import at module load.
        from ironcore.parallel.tensor_parallel.comm import _gather_tensor_along_last_dim

        # Gather full vocabulary logits across TP group
        logits = _gather_tensor_along_last_dim(logits)

    # Now compute log_softmax with the full vocabulary
    # Use float32 for softmax stability
    return F.log_softmax(logits.float(), dim=-1)


def compute_logps(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    labels: torch.Tensor,  # [batch, seq_len]
    mask: torch.Tensor | None = None,  # [batch, seq_len]
) -> torch.Tensor:
    """Compute log probabilities from logits.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Ground truth token IDs [batch, seq_len]
        mask: Optional mask for valid tokens [batch, seq_len]

    Returns:
        [batch] sum of per-token log probabilities
    """
    # Get log probabilities (tensor-parallel safe)
    log_probs = _compute_log_softmax_tp_safe(logits)

    # Select log probabilities for ground truth labels
    return _extract_logps_from_log_probs(log_probs, labels, mask)


def _extract_logps_from_log_probs(
    log_probs: torch.Tensor,  # [batch, seq_len, full_vocab_size]
    labels: torch.Tensor,  # [batch, seq_len]
    mask: torch.Tensor | None = None,  # [batch, seq_len]
) -> torch.Tensor:
    """Extract per-token log probabilities for given labels from full log_probs.

    This function handles several edge cases:

    1. **-100 labels**: PyTorch uses -100 as the ignore index for loss computation.
       These positions are handled using masked operations without cloning.

    2. **Optional masking**: If a mask is provided, only positions where mask=1
       contribute to the final sum. This is useful for response-only computation
       where prompt tokens should be excluded.

    3. **Sum vs Mean**: Uses sum (not mean) to compute sequence-level log probs.
       This follows the standard DPO formulation which avoids biasing toward
       shorter sequences.

    Args:
        log_probs: Full log probabilities [batch, seq_len, vocab_size]
        labels: Ground truth token IDs [batch, seq_len], -100 for ignored positions
        mask: Optional mask for valid tokens [batch, seq_len], 1=valid, 0=ignore

    Returns:
        Sum of per-token log probabilities for each sequence [batch]
    """
    # Handle -100 labels (PyTorch ignore index)
    # Use masked operations instead of clone + index assignment
    ignore_mask = labels == -100

    # Clamp labels to valid range for gathering (0 is safe since we'll mask it out)
    safe_labels = torch.where(ignore_mask, torch.zeros_like(labels), labels)

    # Select log probabilities for ground truth labels
    # [batch, seq_len]
    selected_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(
        -1
    )

    # Zero out ignored positions using masked fill
    if ignore_mask.any():
        selected_log_probs = torch.where(
            ignore_mask, torch.zeros_like(selected_log_probs), selected_log_probs
        )

    # Apply mask and sum (standard DPO uses sum, not average)
    # This avoids biasing toward shorter sequences
    if mask is not None:
        selected_log_probs = selected_log_probs * mask.float()

    sequence_log_probs = selected_log_probs.sum(dim=-1)

    return sequence_log_probs


def dpo_loss(
    policy_chosen_logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    policy_rejected_logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    reference_chosen_logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    reference_rejected_logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    chosen_labels: torch.Tensor,  # [batch, seq_len]
    rejected_labels: torch.Tensor,  # [batch, seq_len]
    chosen_loss_mask: torch.Tensor | None = None,  # [batch, seq_len]
    rejected_loss_mask: torch.Tensor | None = None,  # [batch, seq_len]
    beta: float = 0.5,
    label_smoothing: float = 0.0,
    # Optimization: if already concatenated, we can pass them directly to avoid re-concatenating
    policy_concat_logits: torch.Tensor | None = None,
    reference_concat_logits: torch.Tensor | None = None,
    # Performance: skip metrics computation when not needed
    compute_metrics: bool = True,
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
        chosen_loss_mask: Mask for valid tokens in chosen responses [batch, seq_len]
        rejected_loss_mask: Mask for valid tokens in rejected responses [batch, seq_len]
        beta: Temperature parameter for preference strength
              (higher = more conservative)
        label_smoothing: Label smoothing factor (0.0 = no smoothing)
        policy_concat_logits: Optional concatenated policy logits [2*batch, seq, vocab]
        reference_concat_logits: Optional concatenated reference logits [2*batch, seq, vocab]
        compute_metrics: Whether to compute additional metrics (default True for compatibility)

    Returns:
        Tuple of (loss_tensor, metrics_dict)
    """
    batch_size = chosen_labels.size(0)

    # Pre-compute concatenated labels/mask for the concat-forward optimization
    # This avoids duplicating this logic in both policy and reference branches
    concat_labels = torch.cat([chosen_labels, rejected_labels], dim=0)
    concat_mask = (
        torch.cat([chosen_loss_mask, rejected_loss_mask], dim=0)
        if chosen_loss_mask is not None
        else None
    )

    # 1. Compute per-token log probabilities for policy model
    if policy_concat_logits is not None:
        # Optimization: compute logps for both chosen and rejected in one pass (single all-gather)
        concat_log_probs = _compute_log_softmax_tp_safe(policy_concat_logits)
        concat_logps = _extract_logps_from_log_probs(concat_log_probs, concat_labels, concat_mask)
        chosen_policy_logps = concat_logps[:batch_size]
        rejected_policy_logps = concat_logps[batch_size:]
    else:
        # Standard separate computation
        chosen_policy_logps = compute_logps(policy_chosen_logits, chosen_labels, chosen_loss_mask)
        rejected_policy_logps = compute_logps(
            policy_rejected_logits, rejected_labels, rejected_loss_mask
        )

    # 2. Compute log probabilities for reference model (no grad)
    with torch.no_grad():
        if reference_concat_logits is not None:
            # Optimization: single pass for reference model
            concat_log_probs = _compute_log_softmax_tp_safe(reference_concat_logits)
            concat_logps = _extract_logps_from_log_probs(
                concat_log_probs, concat_labels, concat_mask
            )
            chosen_ref_logps = concat_logps[:batch_size]
            rejected_ref_logps = concat_logps[batch_size:]
        else:
            chosen_ref_logps = compute_logps(
                reference_chosen_logits, chosen_labels, chosen_loss_mask
            )
            rejected_ref_logps = compute_logps(
                reference_rejected_logits, rejected_labels, rejected_loss_mask
            )

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

    # 5. Compute metrics (conditionally for performance)
    if compute_metrics:
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
    else:
        # Return minimal metrics when not computing
        metrics = {
            "dpo_loss": dpo_loss_val.item(),
            "dpo_accuracy": 0.0,  # Placeholder
        }

    return dpo_loss_val, metrics
