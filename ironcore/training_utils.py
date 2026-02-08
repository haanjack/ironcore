"""Common training utilities shared across training scripts."""

from typing import Union

import torch
import torch.distributed as dist

from ironcore.parallel import parallel_states


def loss_func_sft(output_tensor: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Per-sample loss averaging for SFT.

    Each sample contributes equally regardless of token count.
    This prevents longer sequences from dominating the loss.

    Args:
        output_tensor: [batch, seq_len] per-token losses
        loss_mask: [batch, seq_len] valid token mask (1=count, 0=ignore)

    Returns:
        Scalar loss (mean of per-sample losses)
    """
    token_losses = output_tensor.float()
    loss_mask = loss_mask.float()

    # Per-sample: sum tokens / count tokens for each row
    sample_token_sum = (token_losses * loss_mask).sum(dim=1)  # [batch]
    sample_token_count = loss_mask.sum(dim=1).clamp(min=1)  # [batch]
    sample_losses = sample_token_sum / sample_token_count  # [batch]

    return sample_losses.mean()


def compute_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> float:
    """Compute next-token prediction accuracy on valid tokens.

    Args:
        logits: [batch, seq_len, vocab_size] model output logits
        labels: [batch, seq_len] ground truth token IDs
        loss_mask: [batch, seq_len] mask for valid positions (1=count, 0=ignore)

    Returns:
        Accuracy as float (0.0 to 1.0)
    """
    if parallel_states.get_tensor_model_parallel_world_size() > 1:
        # TP mode: logits are sharded along vocab dimension [b, s, vocab/tp_size]

        # 1. Get local max and indices
        local_max_values, local_indices = torch.max(logits, dim=-1) # [b, s]

        # 2. Adjust local indices to global vocab indices
        rank = parallel_states.get_tensor_model_parallel_rank()
        partition_vocab_size = logits.size(-1)
        start_idx = rank * partition_vocab_size
        global_indices = local_indices + start_idx

        # 3. Gather max values and indices from all ranks
        # We need to find which rank has the true global max
        tp_group = parallel_states.get_tensor_model_parallel_group()
        world_size = parallel_states.get_tensor_model_parallel_world_size()

        # List to gather into
        gathered_max_values = [torch.zeros_like(local_max_values) for _ in range(world_size)]
        gathered_indices = [torch.zeros_like(global_indices) for _ in range(world_size)]

        dist.all_gather(gathered_max_values, local_max_values, group=tp_group)
        dist.all_gather(gathered_indices, global_indices, group=tp_group)

        # Stack: [world_size, b, s]
        all_max_values = torch.stack(gathered_max_values)
        all_indices = torch.stack(gathered_indices)

        # 4. Find max across ranks
        # [b, s] indices of the rank that has the max value
        max_rank_indices = torch.argmax(all_max_values, dim=0)

        # 5. Select the corresponding global token index
        # We use gather to select from the specific rank index for each position
        # all_indices: [world_size, b, s] -> gather -> [1, b, s]
        predictions = torch.gather(
            all_indices,
            dim=0,
            index=max_rank_indices.unsqueeze(0)
        ).squeeze(0)

    else:
        # Standard mode
        predictions = logits.argmax(dim=-1)  # [batch, seq_len]

    correct = (predictions == labels) & (loss_mask > 0)

    total_valid = loss_mask.sum()
    if total_valid == 0:
        return 0.0

    return (correct.sum() / total_valid).item()


def get_loss_func(task_type: str):
    """Get appropriate loss function based on task type.

    Args:
        task_type: Training task type ('pretrain', 'sft', 'dpo')

    Returns:
        Loss function callable
    """
    if task_type == "sft":
        return loss_func_sft
    # pretrain and dpo use per-token loss
    return loss_func


def get_batch(
    data_iterator,
) -> Union[torch.Tensor, torch.Tensor]:
    """Get batch from data iterator."""
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        batch = None

    # IronCore dataloader returns dict with 'input_ids' and 'labels'
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    return input_ids, labels


def forward_step(model, data_iterator) -> torch.Tensor:
    """Forward step."""
    input_ids, labels = get_batch(data_iterator=data_iterator)
    loss = model(input_ids, labels)
    return loss


def loss_func(output_tensor: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Loss function - computes mean over all valid tokens (nanoGPT style)."""
    token_losses = output_tensor.float()
    loss_mask = loss_mask.float()

    # Average over ALL valid tokens across the entire batch
    # This matches nanoGPT's F.cross_entropy behavior
    loss = torch.sum(token_losses * loss_mask) / torch.sum(loss_mask)

    return loss


def get_dpo_batch(data_iterator) -> dict:
    """Get DPO batch from data iterator.

    DPO batches contain both chosen and rejected samples with prefixed keys.

    Returns:
        Dictionary with 'chosen_*' and 'rejected_*' prefixed tensors
    """
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        batch = None

    # DPO dataloader returns dict with prefixed keys for chosen/rejected
    return batch


def forward_step_dpo(model, reference_model, data_iterator, dpo_beta=0.5, label_smoothing=0.0):
    """Forward step for DPO training.

    This is a convenience wrapper that can be used with the standard Trainer.
    For full DPO functionality, use DPOTrainer instead.

    Args:
        model: Policy model
        reference_model: Frozen reference model
        data_iterator: Data iterator yielding DPO batches
        dpo_beta: DPO temperature parameter
        label_smoothing: Label smoothing factor

    Returns:
        Dictionary with 'loss' tensor and 'metrics' dict
    """
    from ironcore.alignment.loss import dpo_loss

    batch = get_dpo_batch(data_iterator)

    # Extract chosen samples
    chosen_input_ids = batch['chosen_input_ids']
    chosen_labels = batch['chosen_labels']

    # Extract rejected samples
    rejected_input_ids = batch['rejected_input_ids']
    rejected_labels = batch['rejected_labels']

    # Get loss mask if available
    loss_mask = batch.get('chosen_loss_mask')

    # Forward pass on chosen (policy)
    chosen_policy_logits = model(chosen_input_ids, labels=None)

    # Forward pass on rejected (policy)
    rejected_policy_logits = model(rejected_input_ids, labels=None)

    # Forward pass on reference model (no grad)
    with torch.no_grad():
        chosen_ref_logits = reference_model(chosen_input_ids, labels=None)
        rejected_ref_logits = reference_model(rejected_input_ids, labels=None)

    # Compute DPO loss
    loss, metrics = dpo_loss(
        policy_chosen_logits=chosen_policy_logits,
        policy_rejected_logits=rejected_policy_logits,
        reference_chosen_logits=chosen_ref_logits,
        reference_rejected_logits=rejected_ref_logits,
        chosen_labels=chosen_labels,
        rejected_labels=rejected_labels,
        loss_mask=loss_mask,
        beta=dpo_beta,
        label_smoothing=label_smoothing,
    )

    return {'loss': loss, 'metrics': metrics}
