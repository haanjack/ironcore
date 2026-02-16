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
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from torch import distributed as dist

from ironcore.alignment.loss import dpo_loss
from ironcore.checkpointing import load_checkpoint, save_checkpoint
from ironcore.global_vars import log_metric
from ironcore.utils import clip_grad_norm_tp, is_first_rank

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

        # Reference model will be created after checkpoint loading in train()
        self.reference_model = None

        self.logger.info(
            f"DPOTrainer initialized with beta={self.beta}, "
            f"reference_on_cpu={self.reference_model_on_cpu}, "
            f"concat_passes={self.concat_forward_passes}"
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

    def train(self) -> None:
        """Override train() to create reference model after checkpoint loading.

        This ensures the reference model is initialized with the SFT checkpoint
        weights rather than random initialization.
        """
        # Synchronize all ranks before checkpoint loading to prevent race conditions
        if dist.is_initialized():
            dist.barrier()

        # Load checkpoint (this loads SFT weights into self.model)
        last_step = load_checkpoint(self.config, self.model, self.optimizer, self.lr_scheduler)
        if last_step > -1:
            self.logger.info(f"Successfully loaded checkpoint: {self.config.trainer.model_path}")
        else:
            self.logger.info("Training start from scratch")
            last_step = 0

        # Synchronize after checkpoint loading before reference model creation
        if dist.is_initialized():
            dist.barrier()

        # Create reference model from the loaded weights
        self.logger.info(f"Creating reference model with beta={self.beta}")
        self.reference_model = self._create_reference_model()

        # Synchronize ranks after reference model creation
        if dist.is_initialized():
            dist.barrier()

        self.timer.start("total")
        self.model.train()

        step = last_step

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

            step += 1
            self.log_training(step, loss, grad_norm, param_norm, self.timer)

            if self.config.utils.profile_torch:
                self.context["profile"].step()  # pylint: disable=no-member

            if self.control.do_checkpoint(step):
                save_checkpoint(self.config, self.model, self.optimizer, self.lr_scheduler, step)

                if self.control.do_eval(step):
                    self.evaluate(step)
                    self.model.train()
                if self.control.do_eval_subtask(step):
                    self.evaluate_subtask(step)
                    self.model.train()

                if self.control.do_exit(step):
                    self.logger.info(
                        f"Training stopped by exit interval: {self.config.operation.exit_interval}"
                    )
                    break

        # Final checkpoint if needed
        if self.control.do_final_checkpoint(step, last_step):
            save_checkpoint(self.config, self.model, self.optimizer, self.lr_scheduler, step)

        if self.config.trainer.do_test:
            self.test()

        self.logger.info(f"Total training time: {(self.timer.get('total') / 3600):.2f} hours")
        self.logger.info("Finishing training")
        self._finialize_process()

    def train_step(self, step: int) -> tuple[float, float, float]:
        """DPO training step.

        Processes chosen and rejected pairs, computes DPO loss,
        and updates policy.

        Args:
            step: Current training step

        Returns:
            Tuple of (average_loss, grad_norm, param_norm)
        """
        self.timer.start(name="iter")
        total_loss = 0.0
        total_metrics: dict[str, float] = {}

        for i in range(self.config.trainer.gradient_accumulation_steps):
            is_last_accum_step = i == self.config.trainer.gradient_accumulation_steps - 1

            backward_sync_ctx = (
                self.model.no_sync
                if not is_last_accum_step and hasattr(self.model, "no_sync")
                else nullcontext
            )

            with backward_sync_ctx():
                with self.context["autocast"]:
                    batch = next(self.data_iterator["train"])
                    loss, metrics = self._dpo_forward_step(batch)

                    total_loss += loss.item()
                    scaled_loss = loss / self.config.trainer.gradient_accumulation_steps

                    for k, v in metrics.items():
                        total_metrics[k] = total_metrics.get(k, 0.0) + v

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

        self.timer.stop(name="iter")

        # Average metrics over accumulation steps
        num_steps = self.config.trainer.gradient_accumulation_steps
        avg_metrics = {k: v / num_steps for k, v in total_metrics.items()}

        if is_first_rank() and self.control.do_log(step):
            self._log_dpo_metrics(step, avg_metrics)

        return (total_loss / num_steps, grad_norm, param_norm)

    def _dpo_forward_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward step for DPO training.

        Args:
            batch: Dictionary with 'chosen_*' and 'rejected_*' prefixed tensors

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
