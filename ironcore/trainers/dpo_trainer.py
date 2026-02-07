# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""DPO (Direct Preference Optimization) Trainer.

This trainer implements offline preference optimization using
frozen reference model and pairwise preference data.

Reference:
    Rafailov et al., "Direct Preference Optimization: Your Language Model
    is Secretly a Reward Model" (2023)
    https://arxiv.org/abs/2305.18290
"""

import copy
from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.distributed as dist

from ironcore.trainer import Trainer
from ironcore.alignment.loss import dpo_loss
from ironcore.global_vars import get_logger


def is_first_rank() -> bool:
    """Check if current process is rank 0."""
    return not dist.is_initialized() or dist.get_rank() == 0


class DPOTrainer(Trainer):
    """Trainer for Direct Preference Optimization (DPO).

    DPO optimizes a policy model using preference pairs (chosen vs rejected
    responses) without requiring an explicit reward model. Instead, it uses
    a frozen reference model to compute the preference-based loss.

    Key differences from standard Trainer:
    1. Maintains a frozen reference model copy
    2. Processes chosen/rejected pairs in each batch
    3. Uses DPO loss instead of standard language modeling loss
    4. Logs additional DPO-specific metrics

    Attributes:
        dpo_beta: Temperature parameter for preference strength
        dpo_label_smoothing: Label smoothing factor
        reference_model: Frozen copy of the initial policy
    """

    def __init__(
        self,
        config,
        forward_step_func,
        loss_fn,
    ):
        """Initialize DPO trainer.

        Args:
            config: Training configuration (must have dpo_beta attribute
                    or default 0.5)
            forward_step_func: Forward step function (not directly used,
                              overridden by train_step)
            loss_fn: Loss function (not directly used, DPO uses dpo_loss)
        """
        super().__init__(config, forward_step_func, loss_fn)

        # DPO-specific hyperparameters
        self.dpo_beta = getattr(config, 'dpo_beta', 0.5)
        self.dpo_label_smoothing = getattr(config, 'dpo_label_smoothing', 0.0)

        # Create reference model (frozen copy of initial policy)
        self.logger.info(f"Creating reference model with beta={self.dpo_beta}")
        self.reference_model = self._create_reference_model()

        self.logger.info("DPOTrainer initialized successfully")

    def _create_reference_model(self) -> torch.nn.Module:
        """Create frozen reference model from current policy.

        The reference model is a deep copy of the policy model at
        initialization time. It remains frozen throughout training
        and provides baseline log probabilities.

        Returns:
            Frozen copy of self.model
        """
        # Handle distributed training: get the underlying model
        model_to_copy = (
            self.model.module if hasattr(self.model, 'module') else self.model
        )

        # Create a deep copy of the model
        # Note: This can be memory-intensive for large models
        reference_model = copy.deepcopy(model_to_copy)
        reference_model.eval()

        # Freeze all parameters
        for param in reference_model.parameters():
            param.requires_grad = False

        # Move to same device as original model
        device = next(model_to_copy.parameters()).device
        reference_model.to(device)

        return reference_model

    def train_step(self, step: int) -> Tuple[float, float, float]:
        """DPO training step.

        Processes chosen and rejected pairs, computes DPO loss,
        and updates policy.

        The key difference from standard training:
        1. Batch contains both 'chosen_*' and 'rejected_*' prefixed tensors
        2. Forward pass through both policy and reference models
        3. DPO loss computation instead of standard LM loss

        Args:
            step: Current training step

        Returns:
            Tuple of (average_loss, grad_norm, param_norm)
        """
        self.timer.start(name="iter")
        total_loss = 0.0
        total_metrics: Dict[str, float] = {}

        for i in range(self.config.trainer.gradient_accumulation_steps):
            is_last_accum_step = (
                i == self.config.trainer.gradient_accumulation_steps - 1
            )

            backward_sync_ctx = (
                self.model.no_sync
                if not is_last_accum_step and hasattr(self.model, "no_sync")
                else nullcontext
            )

            with backward_sync_ctx():
                with self.context["autocast"]:
                    # Get DPO batch (contains chosen and rejected pairs)
                    batch = next(self.data_iterator["train"])

                    # Compute DPO loss
                    loss, metrics = self._dpo_forward_step(batch)

                    total_loss += loss.item()
                    scaled_loss = (
                        loss / self.config.trainer.gradient_accumulation_steps
                    )

                    # Accumulate metrics
                    for k, v in metrics.items():
                        total_metrics[k] = total_metrics.get(k, 0.0) + v

                # Backward pass
                self.scaler.scale(scaled_loss).backward()

        # Gradient clipping and norm computation
        self.scaler.unscale_(self.optimizer)

        grad_norm = 0.0
        if self.config.optim.clip_grad > 0.0:
            grad_norm = self._clip_grad_norm_tp(
                self.model.parameters(), self.config.optim.clip_grad
            )
        elif self.control.do_grad_norm(step):
            grad_norm = self._clip_grad_norm_tp(
                self.model.parameters(), float("inf")
            )

        param_norm = 0.0
        if self.control.do_param_norm(step):
            for p in self.model.parameters():
                if p.data is not None:
                    param_norm += p.data.norm() ** 2
            param_norm = param_norm ** 0.5

        # Update model
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        # Record iteration time
        self.timer.stop(name="iter")

        # Average metrics over accumulation steps
        num_steps = self.config.trainer.gradient_accumulation_steps
        avg_metrics = {k: v / num_steps for k, v in total_metrics.items()}

        # Log DPO-specific metrics
        if is_first_rank() and self.control.do_log(step):
            self._log_dpo_metrics(step, avg_metrics)

        return (
            total_loss / num_steps,
            grad_norm,
            param_norm,
        )

    def _dpo_forward_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward step for DPO training.

        Processes chosen and rejected pairs through both policy
        and reference models.

        Args:
            batch: Dictionary with 'chosen_*' and 'rejected_*' prefixed tensors

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Extract chosen samples
        chosen_input_ids = batch['chosen_input_ids']
        chosen_labels = batch['chosen_labels']

        # Extract rejected samples
        rejected_input_ids = batch['rejected_input_ids']
        rejected_labels = batch['rejected_labels']

        # Get loss mask if available
        loss_mask = batch.get('chosen_loss_mask')

        # Forward pass on chosen (policy)
        chosen_policy_logits = self.model(chosen_input_ids, labels=None)

        # Forward pass on rejected (policy)
        rejected_policy_logits = self.model(rejected_input_ids, labels=None)

        # Forward pass on reference model (no grad)
        with torch.no_grad():
            chosen_ref_logits = self.reference_model(
                chosen_input_ids, labels=None
            )
            rejected_ref_logits = self.reference_model(
                rejected_input_ids, labels=None
            )

        # Compute DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logits=chosen_policy_logits,
            policy_rejected_logits=rejected_policy_logits,
            reference_chosen_logits=chosen_ref_logits,
            reference_rejected_logits=rejected_ref_logits,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            loss_mask=loss_mask,
            beta=self.dpo_beta,
            label_smoothing=self.dpo_label_smoothing,
        )

        return loss, metrics

    def _clip_grad_norm_tp(self, parameters, max_norm: float) -> float:
        """Clip gradient norm with tensor parallelism support.

        This is the same as the parent class method but exposed
        for use in train_step.
        """
        from ironcore.utils import clip_grad_norm_tp
        return clip_grad_norm_tp(parameters, max_norm)

    def _log_dpo_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log DPO-specific metrics.

        Args:
            step: Current training step
            metrics: Dictionary of metrics to log
        """
        from ironcore.global_vars import log_metric

        for name, value in metrics.items():
            log_metric(name, value, step)

        # Also log to console
        self.logger.debug(
            f"step: {step}, dpo_loss: {metrics.get('dpo_loss', 0):.4f}, "
            f"margin: {metrics.get('preference_margin', 0):.4f}, "
            f"accuracy: {metrics.get('dpo_accuracy', 0):.4f}"
        )
