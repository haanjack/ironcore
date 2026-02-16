# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Language Model Trainer for Pretraining and Supervised Fine-Tuning (SFT).

This trainer handles standard next-token prediction with cross-entropy loss.
It can be used for both:
- Pretraining: Training from scratch on large text corpora
- SFT: Fine-tuning on instruction/task-specific data

The training algorithm is the same for both use cases; only the data and
initialization differ (scratch vs. checkpoint).
"""

from contextlib import nullcontext

import torch

from ironcore.checkpointing import load_checkpoint, save_checkpoint
from ironcore.training_utils import compute_token_accuracy, get_batch
from ironcore.utils import clip_grad_norm_tp

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

    def train(self):
        """Main training loop for language model training.

        This method handles both pretraining and SFT. It will:
        1. Load checkpoint if model_path is specified (SFT), or start from scratch (pretrain)
        2. Run training loop with forward/backward passes
        3. Log metrics, save checkpoints, run evaluation

        The training loop is standard for both pretraining and SFT.
        """
        # Load checkpoint and restore model and optimizer states
        # If no checkpoint exists, this starts training from scratch (pretraining)
        # If checkpoint exists, this resumes from checkpoint (SFT or resumed pretraining)
        last_step = load_checkpoint(self.config, self.model, self.optimizer, self.lr_scheduler)
        if last_step > -1:
            self.logger.info(
                f"Successfully loaded checkpoint: {self.config.trainer.model_path} "
                f"(resuming from step {last_step})"
            )
        else:
            self.logger.info("Training start from scratch (pretraining mode)")
            last_step = 0

        self.timer.start("total")

        # Set model to training mode
        self.model.train()

        # Start training
        step = last_step

        # Training loop
        if self.config.utils.profile_torch:
            self.context["profile"].start()  # pylint: disable=no-member

        self.logger.info(f"Training start from step: {step}")
        while step < self.config.operation.train_steps:
            if self.config.utils.profile_nsys and step >= self.config.utils.profile_step_start:
                torch.cuda.profiler.start()

            loss, grad_norm, param_norm = self.train_step(step)

            if self.config.utils.profile_nsys and step >= self.config.utils.profile_step_end:
                torch.cuda.profiler.stop()
                break
            if self.config.utils.profile_torch and step >= self.config.utils.profile_step_end:
                self.context["profile"].stop()  # pylint: disable=no-member
                break

            # Update step
            step += 1

            # Print and log training
            self.log_training(step, loss, grad_norm, param_norm, self.timer)

            # Update profiler step
            if self.config.utils.profile_torch:
                self.context["profile"].step()  # pylint: disable=no-member

            # Save checkpoint
            if self.control.do_checkpoint(step):
                save_checkpoint(self.config, self.model, self.optimizer, self.lr_scheduler, step)

                # Evaluation model
                if self.control.do_eval(step):
                    self.evaluate(step)
                    self.model.train()
                if self.control.do_eval_subtask(step):
                    self.evaluate_subtask(step)
                    self.model.train()

                # Check exit condition
                if self.control.do_exit(step):
                    self.logger.info(
                        f"Training is stopped by exit interval: {self.config.operation.exit_interval}"
                    )
                    break

        # Finish training
        # Save checkpoint in case of the total train step is not divisible by checkpoint save step
        if self.control.do_final_checkpoint(step, last_step):
            save_checkpoint(self.config, self.model, self.optimizer, self.lr_scheduler, step)

        if self.config.trainer.do_test:
            self.test()

        self.logger.info(f"Total training time: {(self.timer.get('total') / 3600):.2f} hours")
        self.logger.info("Finishing training")
        self._finialize_process()

    def train_step(self, step: int):
        """Single training step for language modeling.

        Performs:
        1. Forward pass through model with gradient accumulation
        2. Backward pass to compute gradients
        3. Gradient clipping (optional)
        4. Optimizer step to update weights
        5. Learning rate schedule update

        Args:
            step: Current training step

        Returns:
            Tuple of (average_loss, grad_norm, param_norm)
        """
        # Forward pass
        self.timer.start(name="iter")
        total_loss = 0.0

        for i in range(0, self.config.trainer.gradient_accumulation_steps):
            is_last_accum_step = i == self.config.trainer.gradient_accumulation_steps - 1

            backward_sync_ctx = (
                self.model.no_sync
                if not is_last_accum_step and hasattr(self.model, "no_sync")
                else nullcontext
            )

            with backward_sync_ctx():
                with self.context["autocast"]:
                    loss = self.forward_step_func(self.model, self.data_iterator["train"])
                    # loss is already a scalar (averaged over all valid tokens in micro-batch)
                    total_loss += loss
                    # For backprop: scale loss by gradient_accumulation_steps
                    # This ensures gradients are averaged, not summed
                    scaled_loss = loss / self.config.trainer.gradient_accumulation_steps

                # Backward pass
                self.scaler.scale(scaled_loss).backward()

        # Gradient clipping and norm computation
        self.scaler.unscale_(self.optimizer)

        grad_norm = 0.0
        if self.config.optim.clip_grad > 0.0:
            grad_norm = clip_grad_norm_tp(self.model.parameters(), self.config.optim.clip_grad)
        elif self.control.do_grad_norm(step):
            grad_norm = clip_grad_norm_tp(self.model.parameters(), float("inf"))

        param_norm = 0.0
        if self.control.do_param_norm(step):
            for p in self.model.parameters():
                if p.data is not None:
                    param_norm += p.data.norm() ** 2
            param_norm = param_norm**0.5

        # Update model
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        # Record iteration time
        self.timer.stop(name="iter")

        return (
            total_loss / self.config.trainer.gradient_accumulation_steps,
            grad_norm,
            param_norm,
        )

    def _eval_step(self, data_iterator) -> tuple:
        """Single evaluation step for language modeling.

        Computes loss and accuracy on evaluation batch.

        Args:
            data_iterator: Evaluation data iterator

        Returns:
            Tuple of (loss, accuracy)
        """
        batch = get_batch(data_iterator)

        # Extract batch data
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Forward pass
        with self.context["autocast"]:
            logits = self.model(input_ids, labels=None)
            loss = self.loss_fn(logits, labels)

        # Compute accuracy
        accuracy = compute_token_accuracy(logits, labels)

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
