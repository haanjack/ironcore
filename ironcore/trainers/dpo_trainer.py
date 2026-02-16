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

import torch
from torch import distributed as dist

from ironcore.alignment.loss import dpo_loss
from ironcore.checkpointing import load_checkpoint, save_checkpoint
from ironcore.global_vars import log_metric
from ironcore.utils import clip_grad_norm_tp, is_first_rank

from .base_trainer import BaseTrainer


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

        # DPO-specific hyperparameters from alignment config
        self.dpo_beta = config.alignment.dpo_beta
        self.dpo_label_smoothing = config.alignment.dpo_label_smoothing

        # Memory optimization: keep reference model on CPU to save GPU memory
        # Set to True to offload reference model to CPU (slower but uses less GPU memory)
        self.reference_model_on_cpu = config.alignment.reference_model_on_cpu

        # Efficiency: concatenate chosen/rejected for fewer forward passes
        self.concat_forward_passes = config.alignment.concat_forward_passes

        # Reference model will be created after checkpoint loading in train()
        self.reference_model = None

        self.logger.info(
            f"DPOTrainer initialized with beta={self.dpo_beta}, "
            f"reference_on_cpu={self.reference_model_on_cpu}, "
            f"concat_passes={self.concat_forward_passes}"
        )

    def _create_reference_model(self) -> torch.nn.Module:
        """Create frozen reference model from current policy.

        The reference model is a deep copy of the policy model after loading
        the SFT checkpoint weights. It remains frozen throughout training
        and provides baseline log probabilities.

        For large models, the reference model can be kept on CPU to save GPU
        memory, with the trade-off of slower forward passes due to transfers.

        Returns:
            Frozen copy of self.model
        """
        from ironcore.parallel.parallel_states import get_tensor_model_parallel_world_size

        # Handle distributed training: get the underlying model
        model_to_copy = self.model.module if hasattr(self.model, "module") else self.model

        # Create a deep copy of the model
        # WARNING: This doubles memory usage temporarily during the copy
        self.logger.info("Creating reference model from policy weights...")
        reference_model = copy.deepcopy(model_to_copy)
        reference_model.eval()

        # Freeze all parameters
        for param in reference_model.parameters():
            param.requires_grad = False

        # Memory optimization: optionally keep reference model on CPU
        # CRITICAL: This is only supported for TP=1. For TP>1, it causes hangs
        # because collective operations (like all-reduce in RowParallelLinear)
        # will be called on CPU tensors with NCCL backend.
        if self.reference_model_on_cpu:
            if get_tensor_model_parallel_world_size() > 1:
                self.logger.warning(
                    "reference_model_on_cpu is NOT supported with tensor parallelism > 1. "
                    "Force-disabling to avoid hangs."
                )
                self.reference_model_on_cpu = False
                # Keep on same device as policy model
                device = next(model_to_copy.parameters()).device
                reference_model.to(device)
            else:
                self.logger.info("Moving reference model to CPU to save GPU memory")
                reference_model = reference_model.cpu()
        else:
            # Keep on same device as policy model
            device = next(model_to_copy.parameters()).device
            reference_model.to(device)

        return reference_model

    def train(self):
        """Override train() to create reference model after checkpoint loading.

        This ensures the reference model is initialized with the SFT checkpoint
        weights rather than random initialization.
        """
        # Synchronize all ranks before checkpoint loading to prevent race conditions
        if dist.is_initialized():
            dist.barrier()

        # First, load checkpoint (this loads SFT weights into self.model)
        last_step = load_checkpoint(self.config, self.model, self.optimizer, self.lr_scheduler)
        if last_step > -1:
            self.logger.info(f"Successfully loaded checkpoint: {self.config.trainer.model_path}")
        else:
            self.logger.info("Training start from scratch")
            last_step = 0

        # Synchronize after checkpoint loading before reference model creation
        if dist.is_initialized():
            dist.barrier()

        # NOW create reference model from the loaded weights
        self.logger.info(f"Creating reference model with beta={self.dpo_beta}")
        self.reference_model = self._create_reference_model()

        # Synchronize ranks after reference model creation
        if dist.is_initialized():
            dist.barrier()

        self.timer.start("total")

        # Set model to training mode
        self.model.train()

        # Start training loop
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

    def train_step(self, step: int) -> tuple[float, float, float]:
        """DPO training step.

        Processes chosen and rejected pairs, computes DPO loss,
        and updates policy.

        The key difference from standard training:
        1. Batch contains both 'chosen_*' and 'rejected_*' prefixed tensors
        2. Forward pass through both policy and reference models
        3. DPO loss computation instead of standard language modeling loss

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
                    # Get DPO batch (contains chosen and rejected pairs)
                    batch = next(self.data_iterator["train"])

                    # Compute DPO loss
                    loss, metrics = self._dpo_forward_step(batch)

                    total_loss += loss.item()
                    scaled_loss = loss / self.config.trainer.gradient_accumulation_steps

                    # Accumulate metrics
                    for k, v in metrics.items():
                        total_metrics[k] = total_metrics.get(k, 0.0) + v

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
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward step for DPO training.

        Processes chosen and rejected pairs through both policy
        and reference models.

        Args:
            batch: Dictionary with 'chosen_*' and 'rejected_*' prefixed tensors

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Extract chosen samples
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_labels = batch["chosen_labels"]

        # Extract rejected samples
        rejected_input_ids = batch["rejected_input_ids"]
        rejected_labels = batch["rejected_labels"]

        # Get loss masks if available (separate masks for chosen and rejected)
        chosen_loss_mask = batch.get("chosen_loss_mask")
        rejected_loss_mask = batch.get("rejected_loss_mask")

        # Ensure all tensors are on the same device as the model
        # Get device from model parameters (input_ids may be on CPU from dataloader)
        device = next(self.model.parameters()).device
        chosen_input_ids = chosen_input_ids.to(device)
        rejected_input_ids = rejected_input_ids.to(device)
        chosen_labels = chosen_labels.to(device)
        rejected_labels = rejected_labels.to(device)
        if chosen_loss_mask is not None:
            chosen_loss_mask = chosen_loss_mask.to(device)
        if rejected_loss_mask is not None:
            rejected_loss_mask = rejected_loss_mask.to(device)

        if self.concat_forward_passes:
            # Optimization: Concatenate chosen and rejected for fewer forward passes
            # This reduces kernel launch overhead and improves GPU utilization
            concat_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)

            # Forward pass on policy model (single call for both chosen and rejected)
            concat_policy_logits = self.model(concat_input_ids, labels=None)

            # Split back into chosen and rejected (for compatibility if needed)
            batch_size = chosen_input_ids.size(0)
            chosen_policy_logits = concat_policy_logits[:batch_size]
            rejected_policy_logits = concat_policy_logits[batch_size:]

            # Forward pass on reference model (single call)
            with torch.no_grad():
                # Move to GPU if reference model is on CPU
                if self.reference_model_on_cpu:
                    device = concat_input_ids.device
                    concat_input_ids_cpu = concat_input_ids.cpu()
                    concat_ref_logits = self.reference_model(concat_input_ids_cpu, labels=None).detach()
                    concat_ref_logits = concat_ref_logits.to(device)
                    del concat_input_ids_cpu  # Explicit cleanup
                else:
                    concat_ref_logits = self.reference_model(concat_input_ids, labels=None).detach()

                # Split back
                chosen_ref_logits = concat_ref_logits[:batch_size]
                rejected_ref_logits = concat_ref_logits[batch_size:]

            # Pass concatenated logits to dpo_loss for efficient TP computation
            policy_concat_logits = concat_policy_logits
            reference_concat_logits = concat_ref_logits
        else:
            # Standard approach: 4 separate forward passes (less efficient)
            chosen_policy_logits = self.model(chosen_input_ids, labels=None)
            rejected_policy_logits = self.model(rejected_input_ids, labels=None)

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
                    chosen_ref_logits = self.reference_model(chosen_input_ids, labels=None).detach()
                    rejected_ref_logits = self.reference_model(rejected_input_ids, labels=None).detach()

            policy_concat_logits = None
            reference_concat_logits = None

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
            beta=self.dpo_beta,
            label_smoothing=self.dpo_label_smoothing,
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

        # Also log to console
        self.logger.info(
            f"step: {step}, dpo_loss: {metrics.get('dpo_loss', 0):.4f}, "
            f"margin: {metrics.get('preference_margin', 0):.4f}, "
            f"accuracy: {metrics.get('dpo_accuracy', 0):.4f}"
        )

    def _eval_step(self, data_iterator) -> tuple:
        """Single evaluation step for DPO.

        Computes DPO loss and metrics on evaluation batch.

        Args:
            data_iterator: Evaluation data iterator

        Returns:
            Tuple of (loss, accuracy)
        """
        batch = next(data_iterator)

        # Ensure all tensors are on the same device as the model
        device = next(self.model.parameters()).device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # Extract batch data
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_labels = batch["chosen_labels"]
        rejected_input_ids = batch["rejected_input_ids"]
        rejected_labels = batch["rejected_labels"]
        chosen_loss_mask = batch.get("chosen_loss_mask")
        rejected_loss_mask = batch.get("rejected_loss_mask")

        # Compute DPO loss (without gradient)
        with torch.no_grad():
            if self.concat_forward_passes:
                # Optimization: Concatenate chosen and rejected for fewer forward passes
                # This is especially important for TP>1 to reduce all-gather operations
                concat_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)

                # Policy model forward (single call)
                concat_policy_logits = self.model(concat_input_ids, labels=None)
                batch_size = chosen_input_ids.size(0)
                chosen_policy_logits = concat_policy_logits[:batch_size]
                rejected_policy_logits = concat_policy_logits[batch_size:]

                # Reference model forward (single call)
                if self.reference_model_on_cpu:
                    device = concat_input_ids.device
                    concat_input_ids_cpu = concat_input_ids.cpu()
                    concat_ref_logits = self.reference_model(
                        concat_input_ids_cpu, labels=None
                    ).detach().to(device)
                    del concat_input_ids_cpu  # Explicit cleanup
                else:
                    concat_ref_logits = self.reference_model(concat_input_ids, labels=None).detach()

                chosen_ref_logits = concat_ref_logits[:batch_size]
                rejected_ref_logits = concat_ref_logits[batch_size:]

                # Use optimized concatenated logits
                policy_concat_logits = concat_policy_logits
                reference_concat_logits = concat_ref_logits
            else:
                # Standard approach: 4 separate forward passes
                chosen_policy_logits = self.model(chosen_input_ids, labels=None)
                rejected_policy_logits = self.model(rejected_input_ids, labels=None)

                if self.reference_model_on_cpu:
                    device = chosen_input_ids.device
                    chosen_ref_logits = self.reference_model(
                        chosen_input_ids.cpu(), labels=None
                    ).detach().to(device)
                    rejected_ref_logits = self.reference_model(
                        rejected_input_ids.cpu(), labels=None
                    ).detach().to(device)
                else:
                    chosen_ref_logits = self.reference_model(chosen_input_ids, labels=None).detach()
                    rejected_ref_logits = self.reference_model(rejected_input_ids, labels=None).detach()

                policy_concat_logits = None
                reference_concat_logits = None

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
                beta=self.dpo_beta,
                label_smoothing=self.dpo_label_smoothing,
                policy_concat_logits=policy_concat_logits,
                reference_concat_logits=reference_concat_logits,
            )

        # Return loss and accuracy (preference accuracy)
        accuracy = metrics.get("dpo_accuracy", 0.0)
        return loss.item(), accuracy
