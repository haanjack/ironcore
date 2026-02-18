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

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import distributed as dist

from ironcore.alignment.loss import dpo_loss
from ironcore.global_vars import log_metric
from ironcore.utils import is_first_rank

from .base_trainer import BaseTrainer

if TYPE_CHECKING:
    from collections.abc import Iterator


class DPOTrainer(BaseTrainer):
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
        beta: Temperature parameter for preference strength
        label_smoothing: Label smoothing factor
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
            config: Training configuration (must have alignment config)
            forward_step_func: Forward step function (not directly used,
                              overridden by train_step)
            loss_fn: Loss function (not directly used, DPO uses dpo_loss)
        """
        super().__init__(config, forward_step_func, loss_fn)

        # DPO-specific hyperparameters from alignment config
        self.beta = config.alignment.dpo_beta
        self.label_smoothing = config.alignment.dpo_label_smoothing

        # Memory optimization: keep reference model on CPU to save GPU memory
        self.reference_model_on_cpu = config.alignment.reference_model_on_cpu

        # Efficiency: concatenate chosen/rejected for fewer forward passes
        self.concat_forward_passes = config.alignment.concat_forward_passes

        # Performance: compute metrics only every N steps (0 = every step)
        self.metrics_interval = config.alignment.metrics_interval

        # Reference model will be created after checkpoint loading in _post_checkpoint_load()
        self.reference_model = None

        self.logger.info(
            f"DPOTrainer initialized with beta={self.beta}, "
            f"reference_on_cpu={self.reference_model_on_cpu}, "
            f"concat_passes={self.concat_forward_passes}, "
            f"metrics_interval={self.metrics_interval}"
        )

    def _create_reference_model(self) -> torch.nn.Module:
        """Create frozen reference model from current policy.

        The reference model is a deep copy of the policy model after loading
        the SFT checkpoint weights. It remains frozen throughout training
        and provides baseline log probabilities.

        Returns:
            Frozen copy of self.model
        """
        from ironcore.parallel.parallel_states import get_tensor_model_parallel_world_size

        # Handle distributed training: get the underlying model
        model_to_copy = self.model.module if hasattr(self.model, "module") else self.model

        # Create a deep copy of the model
        self.logger.info("Creating reference model from policy weights...")
        reference_model = copy.deepcopy(model_to_copy)
        reference_model.eval()

        # Freeze all parameters
        for param in reference_model.parameters():
            param.requires_grad = False

        # Memory optimization: optionally keep reference model on CPU
        # Note: Only supported for TP=1. For TP>1, collective operations
        # require GPU tensors with NCCL backend.
        if self.reference_model_on_cpu:
            if get_tensor_model_parallel_world_size() > 1:
                self.logger.warning(
                    "reference_model_on_cpu is NOT supported with tensor parallelism > 1. "
                    "Force-disabling to avoid hangs."
                )
                self.reference_model_on_cpu = False
                device = next(model_to_copy.parameters()).device
                reference_model.to(device)
            else:
                self.logger.info("Moving reference model to CPU to save GPU memory")
                reference_model = reference_model.cpu()
        else:
            device = next(model_to_copy.parameters()).device
            reference_model.to(device)

        return reference_model

    def _move_batch_to_device(self, batch: dict) -> dict:
        """Move all tensors in batch to model device.

        Args:
            batch: Dictionary containing tensor values

        Returns:
            Dictionary with all tensors moved to model device
        """
        device = next(self.model.parameters()).device
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _compute_dpo_logits(
        self,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        enable_grad: bool = True,
    ) -> tuple[
        torch.Tensor,  # chosen_policy_logits
        torch.Tensor,  # rejected_policy_logits
        torch.Tensor,  # chosen_ref_logits
        torch.Tensor,  # rejected_ref_logits
        torch.Tensor | None,  # policy_concat_logits
        torch.Tensor | None,  # reference_concat_logits
    ]:
        """Compute policy and reference logits for DPO.

        This shared method handles both training (with gradients) and evaluation
        (without gradients) cases, supporting both concatenated and separate
        forward pass modes.

        Args:
            chosen_input_ids: Input IDs for chosen responses [batch, seq_len]
            rejected_input_ids: Input IDs for rejected responses [batch, seq_len]
            enable_grad: Whether to enable gradients for policy model

        Returns:
            Tuple of (chosen_policy_logits, rejected_policy_logits,
                     chosen_ref_logits, rejected_ref_logits,
                     policy_concat_logits or None, reference_concat_logits or None)
        """
        batch_size = chosen_input_ids.size(0)

        if self.concat_forward_passes:
            # Optimization: Concatenate chosen and rejected for fewer forward passes
            concat_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)

            # Policy model forward
            if enable_grad:
                concat_policy_logits = self.model(concat_input_ids, labels=None)
            else:
                with torch.no_grad():
                    concat_policy_logits = self.model(concat_input_ids, labels=None)

            # Reference model forward (always no_grad)
            with torch.no_grad():
                if self.reference_model_on_cpu:
                    device = concat_input_ids.device
                    concat_input_ids_cpu = concat_input_ids.cpu()
                    concat_ref_logits = self.reference_model(
                        concat_input_ids_cpu, labels=None
                    ).detach().to(device)
                    del concat_input_ids_cpu
                else:
                    concat_ref_logits = self.reference_model(
                        concat_input_ids, labels=None
                    ).detach()

            # Split back into chosen and rejected
            chosen_policy_logits = concat_policy_logits[:batch_size]
            rejected_policy_logits = concat_policy_logits[batch_size:]
            chosen_ref_logits = concat_ref_logits[:batch_size]
            rejected_ref_logits = concat_ref_logits[batch_size:]

            # Return concatenated logits for efficient TP computation
            policy_concat_logits = concat_policy_logits
            reference_concat_logits = concat_ref_logits
        else:
            # Standard approach: 4 separate forward passes
            if enable_grad:
                chosen_policy_logits = self.model(chosen_input_ids, labels=None)
                rejected_policy_logits = self.model(rejected_input_ids, labels=None)
            else:
                with torch.no_grad():
                    chosen_policy_logits = self.model(chosen_input_ids, labels=None)
                    rejected_policy_logits = self.model(rejected_input_ids, labels=None)

            # Reference model forward (always no_grad)
            with torch.no_grad():
                if self.reference_model_on_cpu:
                    device = chosen_input_ids.device
                    chosen_ref_logits = self.reference_model(
                        chosen_input_ids.cpu(), labels=None
                    ).detach().to(device)
                    rejected_ref_logits = self.reference_model(
                        rejected_input_ids.cpu(), labels=None
                    ).detach().to(device)
                else:
                    chosen_ref_logits = self.reference_model(
                        chosen_input_ids, labels=None
                    ).detach()
                    rejected_ref_logits = self.reference_model(
                        rejected_input_ids, labels=None
                    ).detach()

            policy_concat_logits = None
            reference_concat_logits = None

        return (
            chosen_policy_logits,
            rejected_policy_logits,
            chosen_ref_logits,
            rejected_ref_logits,
            policy_concat_logits,
            reference_concat_logits,
        )

    def _post_checkpoint_load(self, last_step: int) -> None:
        """Create reference model after checkpoint loading.

        This ensures the reference model is initialized with the SFT checkpoint
        weights rather than random initialization.

        Args:
            last_step: The step loaded from checkpoint (0 if fresh start)
        """
        # Synchronize after checkpoint loading before reference model creation
        if dist.is_initialized():
            dist.barrier()

        # Create reference model from the loaded weights
        self.logger.info(f"Creating reference model with beta={self.beta}")
        self.reference_model = self._create_reference_model()

        # Synchronize ranks after reference model creation
        if dist.is_initialized():
            dist.barrier()

    def _forward_micro_batch(self, step: int) -> tuple[torch.Tensor, dict[str, float] | None]:
        """Forward pass for a single micro-batch in DPO training.

        Overrides base class to use DPO-specific forward logic.

        Args:
            step: Current training step

        Returns:
            Tuple of (loss_tensor, metrics_dict)
        """
        batch = next(self.data_iterator["train"])

        # Determine if we should compute metrics this step
        compute_metrics = (
            self.metrics_interval == 0 or  # Always compute
            step % self.metrics_interval == 0  # Compute on interval
        )

        return self._dpo_forward_step(batch, compute_metrics=compute_metrics)

    def train_step(self, step: int) -> tuple[float, float, float]:
        """DPO training step.

        Uses the base class gradient accumulation loop with DPO-specific
        forward pass.

        Args:
            step: Current training step

        Returns:
            Tuple of (average_loss, grad_norm, param_norm)
        """
        self.timer.start(name="iter")

        # Use base class gradient accumulation (calls _forward_micro_batch)
        total_loss, total_metrics = self._run_gradient_accumulation(step)

        # Compute gradient and parameter norms
        grad_norm, param_norm = self._compute_grad_and_param_norms(step)

        # Optimizer step
        self._optimizer_step()

        self.timer.stop(name="iter")

        # Average metrics over accumulation steps
        num_steps = self.config.trainer.gradient_accumulation_steps
        avg_metrics = {k: v / num_steps for k, v in total_metrics.items()}
        avg_loss = total_loss / num_steps

        # Check for NaN/Inf loss
        self._check_loss_for_nan(avg_loss, step)

        if is_first_rank() and self.control.do_log(step):
            self._log_dpo_metrics(step, avg_metrics)

        return (avg_loss, grad_norm, param_norm)

    def _dpo_forward_step(
        self,
        batch: dict[str, torch.Tensor],
        compute_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward step for DPO training.

        Args:
            batch: Dictionary with 'chosen_*' and 'rejected_*' prefixed tensors
            compute_metrics: Whether to compute additional metrics

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Move batch to device
        batch = self._move_batch_to_device(batch)

        # Extract inputs
        chosen_input_ids = batch["chosen_input_ids"]
        rejected_input_ids = batch["rejected_input_ids"]
        chosen_labels = batch["chosen_labels"]
        rejected_labels = batch["rejected_labels"]
        chosen_loss_mask = batch.get("chosen_loss_mask")
        rejected_loss_mask = batch.get("rejected_loss_mask")

        # Compute logits using shared method
        (
            chosen_policy_logits,
            rejected_policy_logits,
            chosen_ref_logits,
            rejected_ref_logits,
            policy_concat_logits,
            reference_concat_logits,
        ) = self._compute_dpo_logits(
            chosen_input_ids,
            rejected_input_ids,
            enable_grad=True,
        )

        # Compute DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logits=chosen_policy_logits,
            policy_rejected_logits=rejected_policy_logits,
            reference_chosen_logits=chosen_ref_logits,
            reference_rejected_logits=rejected_ref_logits,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            chosen_loss_mask=chosen_loss_mask,
            rejected_loss_mask=rejected_loss_mask,
            beta=self.beta,
            label_smoothing=self.label_smoothing,
            policy_concat_logits=policy_concat_logits,
            reference_concat_logits=reference_concat_logits,
            compute_metrics=compute_metrics,
        )

        return loss, metrics

    def _log_dpo_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log DPO-specific metrics.

        Args:
            step: Current training step
            metrics: Dictionary of metrics to log
        """
        for name, value in metrics.items():
            log_metric(name, value, step)

        self.logger.info(
            f"step: {step}, dpo_loss: {metrics.get('dpo_loss', 0):.4f}, "
            f"margin: {metrics.get('preference_margin', 0):.4f}, "
            f"accuracy: {metrics.get('dpo_accuracy', 0):.4f}"
        )

    def _eval_step(self, data_iterator: Iterator) -> tuple[float, float]:
        """Single evaluation step for DPO.

        Args:
            data_iterator: Evaluation data iterator

        Returns:
            Tuple of (loss, accuracy)
        """
        batch = next(data_iterator)
        batch = self._move_batch_to_device(batch)

        # Extract inputs
        chosen_input_ids = batch["chosen_input_ids"]
        rejected_input_ids = batch["rejected_input_ids"]
        chosen_labels = batch["chosen_labels"]
        rejected_labels = batch["rejected_labels"]
        chosen_loss_mask = batch.get("chosen_loss_mask")
        rejected_loss_mask = batch.get("rejected_loss_mask")

        # Compute logits using shared method (no gradients)
        with torch.no_grad():
            (
                chosen_policy_logits,
                rejected_policy_logits,
                chosen_ref_logits,
                rejected_ref_logits,
                policy_concat_logits,
                reference_concat_logits,
            ) = self._compute_dpo_logits(
                chosen_input_ids,
                rejected_input_ids,
                enable_grad=False,
            )

            # Compute DPO loss
            loss, metrics = dpo_loss(
                policy_chosen_logits=chosen_policy_logits,
                policy_rejected_logits=rejected_policy_logits,
                reference_chosen_logits=chosen_ref_logits,
                reference_rejected_logits=rejected_ref_logits,
                chosen_labels=chosen_labels,
                rejected_labels=rejected_labels,
                chosen_loss_mask=chosen_loss_mask,
                rejected_loss_mask=rejected_loss_mask,
                beta=self.beta,
                label_smoothing=self.label_smoothing,
                policy_concat_logits=policy_concat_logits,
                reference_concat_logits=reference_concat_logits,
            )

        accuracy = metrics.get("dpo_accuracy", 0.0)
        return loss.item(), accuracy
