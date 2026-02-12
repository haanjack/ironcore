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
        local_max_values, local_indices = torch.max(logits, dim=-1)  # [b, s]

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
        predictions = torch.gather(all_indices, dim=0, index=max_rank_indices.unsqueeze(0)).squeeze(
            0
        )

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


def get_vla_batch(data_iterator) -> dict:
    """Get VLA batch from data iterator.

    Returns:
        Dictionary with:
        - input_ids: Token IDs
        - labels: Labels for language modeling
        - pixel_values: Processed images
        - actions: Target actions
        - attention_mask: Attention mask
        - image_token_mask: Mask for image token positions
    """
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        batch = None

    return batch


def forward_step_vla(model, data_iterator) -> torch.Tensor:
    """Forward step for VLA training.

    Handles vision, language, and action inputs for VLA models.

    Args:
        model: VLAModel instance
        data_iterator: Iterator yielding VLA batches

    Returns:
        Total loss (language loss + action loss)
    """
    batch = get_vla_batch(data_iterator)

    # Extract batch components
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    pixel_values = batch["pixel_values"]
    actions = batch["actions"]
    attention_mask = batch.get("attention_mask", None)
    image_token_mask = batch.get("image_token_mask", None)

    # Forward pass
    loss = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        labels=labels,
        actions=actions,
        attention_mask=attention_mask,
        image_token_mask=image_token_mask,
    )

    return loss


def compute_action_metrics(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
) -> dict[str, float]:
    """Compute metrics for action prediction.

    Args:
        pred_actions: [batch, action_dim * horizon] predicted actions
        target_actions: [batch, action_dim * horizon] target actions

    Returns:
        Dictionary with MSE, L1, and max error metrics
    """
    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(pred_actions, target_actions).item()
        l1 = torch.nn.functional.l1_loss(pred_actions, target_actions).item()
        max_error = (pred_actions - target_actions).abs().max().item()

    return {
        "action_mse": mse,
        "action_l1": l1,
        "action_max_error": max_error,
    }


def compute_vla_success_rate(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    position_tol: float = 0.05,
    rotation_tol: float = 0.1,
    gripper_tol: float = 0.1,
) -> dict[str, float]:
    """Compute success rate based on action tolerances.

    Success is determined by whether predictions are within tolerance
    for all action dimensions (position, rotation, gripper).

    Args:
        pred_actions: [batch, action_dim * horizon] predicted actions
        target_actions: [batch, action_dim * horizon] target actions
        position_tol: Position tolerance (default 5cm)
        rotation_tol: Rotation tolerance (default ~5.7 degrees)
        gripper_tol: Gripper state tolerance (default 10%)

    Returns:
        Dictionary with success rates
    """
    with torch.no_grad():
        errors = (pred_actions - target_actions).abs()

        # Position success (first 3 dims)
        if pred_actions.shape[1] >= 3:
            pos_errors = errors[:, :3].max(dim=1)[0]
            pos_success = (pos_errors < position_tol).float().mean().item()
        else:
            pos_success = 1.0

        # Rotation success (dims 3-5)
        if pred_actions.shape[1] >= 6:
            rot_errors = errors[:, 3:6].max(dim=1)[0]
            rot_success = (rot_errors < rotation_tol).float().mean().item()
        else:
            rot_success = 1.0

        # Gripper success (dim 6)
        if pred_actions.shape[1] >= 7:
            grip_errors = errors[:, 6]
            grip_success = (grip_errors < gripper_tol).float().mean().item()
        else:
            grip_success = 1.0

        # Overall success (all within tolerance)
        all_success = pos_success * rot_success * grip_success

    return {
        "success_rate": all_success * 100,
        "position_success": pos_success * 100,
        "rotation_success": rot_success * 100,
        "gripper_success": grip_success * 100,
    }


def eval_step_vla(model, data_iterator) -> dict[str, float]:
    """Evaluation step for VLA model.

    Computes validation loss and action prediction metrics.

    Args:
        model: VLA model
        data_iterator: Iterator over validation data

    Returns:
        Dictionary with loss and action metrics
    """
    batch = get_vla_batch(data_iterator)

    # Extract batch components
    input_ids = batch["input_ids"]
    labels = batch.get("labels")
    pixel_values = batch.get("pixel_values")
    actions = batch.get("actions")
    attention_mask = batch.get("attention_mask")

    # Forward pass
    loss = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        labels=labels,
        actions=actions,
        attention_mask=attention_mask,
    )

    metrics = {"val_loss": loss.item()}

    # Compute action metrics if actions present
    if actions is not None and pixel_values is not None:
        with torch.no_grad():
            pred_actions = model.predict_action(
                input_ids=input_ids,
                pixel_values=pixel_values,
            )

            action_metrics = compute_action_metrics(pred_actions, actions)
            success_metrics = compute_vla_success_rate(pred_actions, actions)

            metrics.update(action_metrics)
            metrics.update(success_metrics)

    return metrics
