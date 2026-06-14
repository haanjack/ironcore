# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Language Model Trainer for Pretraining and Supervised Fine-Tuning (SFT).

This trainer handles standard next-token prediction with cross-entropy loss.
It can be used for both:
- Pretraining: Training from scratch on large text corpora
- SFT: Fine-tuning on instruction/task-specific data

The training algorithm is the same for both use cases; only the data and
initialization differ (scratch vs. checkpoint).
"""

import torch

from ironcore.training_utils import compute_token_accuracy

from .base_trainer import BaseTrainer


class LanguageModelTrainer(BaseTrainer):
    """Trainer for language model pretraining and supervised fine-tuning.

    This trainer implements standard autoregressive language modeling with
    next-token prediction and cross-entropy loss. It supports:

    Use Cases:
    1. **Pretraining**: Train model from scratch on large text corpora
       - Initialize with random weights
       - Train on general domain text
       - Learn language patterns and knowledge

    2. **Supervised Fine-Tuning (SFT)**: Adapt pretrained model to specific tasks
       - Load pretrained checkpoint
       - Train on instruction/task data
       - Specialize model behavior

    The training algorithm is identical for both use cases. The difference is:
    - Pretraining: Start from random initialization, large-scale data
    - SFT: Start from pretrained checkpoint, task-specific data

    Key Features:
    - Standard autoregressive language modeling
    - Cross-entropy loss on next-token prediction
    - Gradient accumulation support
    - Mixed precision training
    - Distributed training (DP, TP, PP)
    - Automatic evaluation and checkpointing

    Example:
        # Pretraining
        config = load_config('pretrain_config.yaml')
        trainer = LanguageModelTrainer(config, forward_step_func, loss_fn)
        trainer.train()  # Trains from scratch

        # Supervised Fine-Tuning
        config = load_config('sft_config.yaml')  # with model_path set
        trainer = LanguageModelTrainer(config, forward_step_func, loss_fn)
        trainer.train()  # Loads checkpoint and continues training
    """

    def train_step(self, step: int) -> tuple[float, float, float]:
        """Single training step for language modeling.

        Uses the base class gradient accumulation loop with the default
        _forward_micro_batch implementation.

        Args:
            step: Current training step

        Returns:
            Tuple of (average_loss, grad_norm, param_norm)
        """
        self.timer.start(name="iter")

        # Use base class gradient accumulation (calls _forward_micro_batch)
        total_loss, _ = self._run_gradient_accumulation(step)

        # Compute gradient and parameter norms
        grad_norm, param_norm = self._compute_grad_and_param_norms(step)

        # Optimizer step
        self._optimizer_step()

        self.timer.stop(name="iter")

        num_steps = self.config.trainer.gradient_accumulation_steps
        avg_loss = total_loss / num_steps

        # Check for NaN/Inf loss
        self._check_loss_for_nan(avg_loss, step)

        return (avg_loss, grad_norm, param_norm)

    def _eval_step(self, batch) -> tuple:
        """Single evaluation step for language modeling.

        Computes loss and accuracy on evaluation batch.

        Args:
            batch: Batch dict with 'input_ids' and 'labels'

        Returns:
            Tuple of (loss, accuracy)
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        with torch.no_grad(), self.context["autocast"]:
            logits = self.model(input_ids, labels=None)

        loss_mask = (labels != -100).float()
        accuracy = compute_token_accuracy(logits, labels, loss_mask)

        shifted_logits = logits[..., :-1, :].contiguous().float()
        shifted_labels = labels[..., 1:].contiguous()
        shifted_mask = loss_mask[..., 1:]

        per_token_losses = torch.nn.functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            reduction="none",
        ).view(shifted_labels.shape)
        mask_sum = shifted_mask.sum()
        loss = (per_token_losses * shifted_mask).sum() / (mask_sum if mask_sum > 0 else 1.0)

        return loss.item(), accuracy


# Aliases for clarity - these are just references to LanguageModelTrainer
# Using direct assignment makes them true aliases (SFTTrainer is LanguageModelTrainer)
SFTTrainer = LanguageModelTrainer
"""Alias for LanguageModelTrainer emphasizing SFT use case.

This is the exact same as LanguageModelTrainer, just with a name that
emphasizes the supervised fine-tuning use case.

Under the hood, pretraining and SFT use the same algorithm (next-token
prediction), so they share the same trainer implementation.

Usage:
    # Explicitly for SFT (clearer intent)
    trainer = SFTTrainer(config, forward_step_func, loss_fn)

    # Or use the general name
    trainer = LanguageModelTrainer(config, forward_step_func, loss_fn)
"""

PretrainTrainer = LanguageModelTrainer
"""Alias for LanguageModelTrainer emphasizing pretraining use case.

This is the exact same as LanguageModelTrainer, just with a name that
emphasizes the pretraining use case.

Under the hood, pretraining and SFT use the same algorithm (next-token
prediction), so they share the same trainer implementation.

Usage:
    # Explicitly for pretraining (clearer intent)
    trainer = PretrainTrainer(config, forward_step_func, loss_fn)

    # Or use the general name
    trainer = LanguageModelTrainer(config, forward_step_func, loss_fn)
"""
