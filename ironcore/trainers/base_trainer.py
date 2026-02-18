# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Base trainer class with common functionality for all training methods."""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Union

import torch
from torch import distributed as dist

from ironcore.checkpointing import load_checkpoint, save_checkpoint
from ironcore.config import MainConfig
from ironcore.controller import TrainingControl
from ironcore.dataloader import get_data_iterator
from ironcore.eval import get_evaluators
from ironcore.global_vars import (
    get_logger,
    get_timer,
    global_states_cleanup,
    log_metric,
    log_metrics,
    set_global_states,
)
from ironcore.language_model import LanguageModel
from ironcore.optimizer import get_optimizer
from ironcore.optimizer.lr_scheduler import get_lr_scheduler
from ironcore.parallel import initialize_parallelism, initialize_process
from ironcore.parallel.parallel_states import (
    get_data_parallel_group,
    get_data_parallel_world_size,
    initialize_model_parallel,
)
from ironcore.utils import (
    get_device,
    get_memory_usage,
    get_model_dtype,
    is_first_rank,
)


class BaseTrainer(ABC):
    """Abstract base trainer with common functionality.

    This class provides:
    - Model and optimizer initialization
    - Data loader setup
    - Distributed training infrastructure
    - Evaluation framework
    - Logging utilities
    - Checkpointing support
    - Template method for training loop

    Subclasses must implement:
    - train_step(): Single training step
    - _eval_step(): Custom evaluation logic

    Subclasses can override:
    - _pre_train_setup(): Hook for setup before training (e.g., checkpoint loading)
    - _post_checkpoint_load(): Hook called after checkpoint load
    - _on_checkpoint_save(): Hook called when checkpoint is saved
    """

    def __init__(
        self,
        config: MainConfig,
        forward_step_func,
        loss_fn,
    ):
        """Initialize base trainer.

        Args:
            config: Training configuration
            forward_step_func: Forward step function (may be unused by some trainers)
            loss_fn: Loss function for model
        """
        self.config = config

        set_global_states(config)

        self.timer = get_timer()
        self.logger = get_logger()
        self.forward_step_func = forward_step_func
        self.loss_fn = loss_fn

        # training control
        self.control = TrainingControl(config)

        initialize_process(config)

        initialize_model_parallel(
            config.trainer.tensor_model_parallel_size,
            timeout_in_minutes=int(config.parallel.timeout_minute)
            if config.parallel.timeout_minute is not None
            else 10.0,
        )

        # initialize data loader
        self.data_iterator = get_data_iterator(config)

        self.evaluators = get_evaluators(
            config.data.eval_datasets,
            config.trainer.eval_batch_size,
            config.operation.eval_samples,
        )

        # contexts control training process
        self.context: dict[str, Union[nullcontext, torch.autocast]] = {
            "autocast": nullcontext(),
            "profile": nullcontext(),
        }

        # initialize model and optimizer
        self.model, self.optimizer = self._build_model_and_optimizer()
        self.lr_scheduler = get_lr_scheduler(config, self.optimizer)

        if self.model.device != "mps":
            self.context["autocast"] = torch.autocast(
                device_type=get_device(), dtype=get_model_dtype(self.config)
            )

        self.scaler = torch.amp.GradScaler(enabled=(get_model_dtype(config) == torch.float16))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finialize_process()

    def _finialize_process(self):
        """Cleanup resources."""
        # Close loggers before exiting
        global_states_cleanup()

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _build_model_and_optimizer(self):
        """Build model and optimizer.

        Returns:
            Tuple of (model, optimizer)
        """
        # Set random seed for reproducibility (critical for TP initialization)
        import random

        import numpy as np

        seed = self.config.init.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.logger.info(f"Set random seed to {seed} for model initialization")

        device = get_device()

        model = LanguageModel(self.config, self.loss_fn).to(device=device)
        self.logger.info("Created Language Model")

        model = model.to(dtype=get_model_dtype(self.config))

        optimizer = get_optimizer(self.config, model, device_type=device)
        self.logger.info("Created Optimizer")

        # Enable profiling if requested
        if self.config.profiler.gpu_profiler:
            model.register_profile_hooks(profile_nsys=True)

        # Apply torch.compile BEFORE parallelism wrapping (DDP/FSDP)
        if self.config.trainer.compile_model:
            compile_options = {
                "backend": self.config.trainer.compile_backend,
                "dynamic": self.config.trainer.compile_dynamic,
                "fullgraph": self.config.trainer.compile_fullgraph,
            }
            if self.config.trainer.compile_mode is not None:
                compile_options["mode"] = self.config.trainer.compile_mode
            try:
                model = torch.compile(model, **compile_options)
                self.logger.info(f"Compiled model with options: {compile_options}")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}. Running without compilation.")

        if device not in ["cpu", "mps"]:
            model = initialize_parallelism(self.config, model)
        self.rank = dist.get_rank()

        return model, optimizer

    @staticmethod
    def average_loss(loss):
        """Average loss across data parallel ranks."""
        if dist.is_initialized() and get_data_parallel_world_size() > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=get_data_parallel_group())
            loss /= get_data_parallel_world_size()
        return loss.item()

    def _pre_train_setup(self) -> int:
        """Hook for setup before training starts.

        Override this method to perform custom setup (e.g., checkpoint loading,
        reference model creation for DPO).

        Returns:
            Starting step number (0 for fresh training, or checkpoint step + 1)
        """
        # Default implementation: load checkpoint if available
        try:
            last_step = load_checkpoint(self.config, self.model, self.optimizer, self.lr_scheduler)
            if last_step > -1:
                self.logger.info(
                    f"Successfully loaded checkpoint: {self.config.trainer.model_path} "
                    f"(resuming from step {last_step})"
                )
            else:
                self.logger.info("Training start from scratch")
                last_step = 0
        except FileNotFoundError as e:
            self.logger.warning(f"Checkpoint not found: {e}. Starting from scratch.")
            last_step = 0
        except RuntimeError as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(
                f"Checkpoint loading failed. If the checkpoint is corrupted, "
                f"remove or rename {self.config.trainer.model_path} and restart."
            ) from e

        self._post_checkpoint_load(last_step)
        return last_step

    def _post_checkpoint_load(self, last_step: int) -> None:  # noqa: B027
        """Hook called after checkpoint loading.

        Override this method to perform post-load setup (e.g., reference model
        creation for DPO).

        Args:
            last_step: The step loaded from checkpoint (0 if fresh start)
        """

    def _on_checkpoint_save(self, step: int) -> None:  # noqa: B027
        """Hook called when a checkpoint is about to be saved.

        Override this method to perform actions before checkpoint save.

        Args:
            step: Current training step
        """

    def train(self):
        """Main training loop (template method).

        This method implements the common training loop structure:
        1. Call _pre_train_setup() for subclass-specific setup
        2. Run training loop with train_step()
        3. Handle checkpointing, evaluation, and exit conditions
        4. Save final checkpoint if needed

        Subclasses should override _pre_train_setup() and _post_checkpoint_load()
        for custom behavior rather than overriding this method.
        """
        # Synchronize all ranks before setup
        if dist.is_initialized():
            dist.barrier()

        # Subclass setup (checkpoint loading, reference model creation, etc.)
        last_step = self._pre_train_setup()

        # Synchronize after setup
        if dist.is_initialized():
            dist.barrier()

        self.timer.start("total")
        self.model.train()

        step = last_step

        if self.config.profiler.torch_profiler:
            self.context["profile"].start()  # pylint: disable=no-member

        self.logger.info(f"Training start from step: {step}")
        while step < self.config.operation.train_steps:
            if self.config.profiler.gpu_profiler and step >= self.config.profiler.start:
                torch.cuda.profiler.start()

            loss, grad_norm, param_norm = self.train_step(step)

            if self.config.profiler.gpu_profiler and step >= self.config.profiler.end:
                torch.cuda.profiler.stop()
                break
            if self.config.profiler.torch_profiler and step >= self.config.profiler.end:
                self.context["profile"].stop()  # pylint: disable=no-member
                break

            step += 1
            self.log_training(step, loss, grad_norm, param_norm, self.timer)

            if self.config.profiler.torch_profiler:
                self.context["profile"].step()  # pylint: disable=no-member

            if self.control.do_checkpoint(step):
                self._on_checkpoint_save(step)
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

    @abstractmethod
    def train_step(self, step: int) -> tuple[float, float, float]:
        """Single training step.

        Args:
            step: Current training step

        Returns:
            Tuple of (loss, grad_norm, param_norm)

        Subclasses must implement this method with their specific step logic.
        """
        pass

    def _run_gradient_accumulation(
        self,
        step: int,
    ) -> tuple[float, dict[str, float]]:
        """Run gradient accumulation loop (shared between trainers).

        This template method handles:
        - Gradient accumulation over micro-batches
        - DDP/FSDP gradient synchronization control
        - Mixed precision (autocast and gradient scaling)
        - Backward pass

        Args:
            step: Current training step

        Returns:
            Tuple of (total_loss, additional_metrics)
            - total_loss: Sum of losses over all micro-batches
            - additional_metrics: Dict of metrics to average over accumulation steps
        """
        total_loss = 0.0
        total_metrics: dict[str, float] = {}

        for i in range(self.config.trainer.gradient_accumulation_steps):
            is_last_accum_step = i == self.config.trainer.gradient_accumulation_steps - 1

            # Disable gradient sync for intermediate accumulation steps (DDP/FSDP)
            backward_sync_ctx = (
                self.model.no_sync
                if not is_last_accum_step and hasattr(self.model, "no_sync")
                else nullcontext
            )

            with backward_sync_ctx():
                with self.context["autocast"]:
                    loss, metrics = self._forward_micro_batch(step)

                    total_loss += loss.item()
                    scaled_loss = loss / self.config.trainer.gradient_accumulation_steps

                    # Accumulate metrics if provided
                    if metrics:
                        for k, v in metrics.items():
                            total_metrics[k] = total_metrics.get(k, 0.0) + v

                # Backward pass with gradient scaling
                self.scaler.scale(scaled_loss).backward()

        return total_loss, total_metrics

    def _forward_micro_batch(self, step: int) -> tuple[torch.Tensor, dict[str, float] | None]:
        """Forward pass for a single micro-batch.

        Subclasses must implement this method to define their forward logic.

        Args:
            step: Current training step

        Returns:
            Tuple of (loss_tensor, metrics_dict or None)
            - loss_tensor: Loss for this micro-batch (not averaged)
            - metrics_dict: Optional dict of metrics to accumulate
        """
        # Default implementation for standard language modeling
        loss = self.forward_step_func(self.model, self.data_iterator["train"])
        return loss, None

    def _compute_grad_and_param_norms(self, step: int) -> tuple[float, float]:
        """Compute gradient and parameter norms after gradient accumulation.

        This method unscales gradients and optionally clips them.

        Args:
            step: Current training step

        Returns:
            Tuple of (grad_norm, param_norm)
        """
        from ironcore.utils import clip_grad_norm_tp

        # Unscale gradients before clipping/norm computation
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

        return grad_norm, param_norm

    def _optimizer_step(self):
        """Perform optimizer step after gradient accumulation."""
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

    def _check_loss_for_nan(self, loss: float, step: int) -> None:
        """Check if loss is NaN or Inf and raise error if so.

        Args:
            loss: Loss value to check
            step: Current training step

        Raises:
            RuntimeError: If loss is NaN or Inf
        """
        import math

        if math.isnan(loss) or math.isinf(loss):
            self.logger.error(f"NaN/Inf loss detected at step {step}: loss={loss}")
            raise RuntimeError(
                f"Training stopped due to {'NaN' if math.isnan(loss) else 'Inf'} loss at step {step}. "
                f"Possible causes: learning rate too high, gradient explosion, or data issues."
            )

    def _handle_training_error(self, error: Exception, step: int) -> None:
        """Handle training errors with appropriate logging and cleanup.

        Args:
            error: The exception that occurred
            step: Current training step

        Raises:
            The original error after logging and cleanup
        """
        import torch.cuda

        self.logger.error(f"Training error at step {step}: {error}")

        # Log GPU memory state if CUDA is available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                self.logger.error(
                    f"GPU {i}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"
                )

        # Try to save emergency checkpoint
        try:
            emergency_path = f"{self.config.trainer.model_path}_emergency_step{step}"
            self.logger.info(f"Attempting to save emergency checkpoint to {emergency_path}")
            # Note: We intentionally don't save here to avoid overwriting good checkpoints
            # Users can manually save if needed
        except Exception as e:
            self.logger.error(f"Failed to save emergency checkpoint: {e}")

        raise error

    def log_training(
        self,
        step: int,
        loss: float,
        grad_norm: float = 0.0,
        param_norm: float = 0.0,
        timer=None,
    ):
        """Log training metrics.

        Args:
            step: Current training step
            loss: Training loss
            grad_norm: Gradient norm
            param_norm: Parameter norm
            timer: Optional timer object
        """
        if not is_first_rank():
            return

        # Basic metrics
        # Get LR - handle case where scheduler hasn't been stepped yet
        try:
            lr = self.lr_scheduler.get_last_lr()[0]
        except (AttributeError, IndexError):
            # Fallback to optimizer's current LR
            lr = self.optimizer.param_groups[0]["lr"]

        metrics = {
            "loss": loss,
            "step": step,
            "lr": lr,
        }

        # Optional metrics
        if grad_norm > 0:
            metrics["grad_norm"] = grad_norm
        if param_norm > 0:
            metrics["param_norm"] = param_norm

        # Timing metrics
        if timer is not None:
            iter_time = timer.get("iter")
            metrics["iter_time"] = iter_time
            if iter_time > 0:
                tokens_per_sec = (
                    self.config.trainer.train_batch_size
                    * self.config.model.max_position_embeddings
                    / iter_time
                )
                metrics["tokens_per_sec"] = tokens_per_sec

        # Memory metrics (on log interval)
        if self.control.do_log(step):
            gpu_mem = get_memory_usage()
            if gpu_mem is not None:
                metrics["gpu_memory_mb"] = gpu_mem

        # Log to console and tracking
        if self.control.do_log(step):
            log_msg = f"step: {step}, loss: {loss:.4f}, lr: {metrics['lr']:.6f}"
            if grad_norm > 0:
                log_msg += f", grad_norm: {grad_norm:.4f}"
            if timer is not None:
                log_msg += f", iter_time: {iter_time:.3f}s"
                if "tokens_per_sec" in metrics:
                    log_msg += f", tok/s: {metrics['tokens_per_sec']:.1f}"
            self.logger.info(log_msg)

            # Log all metrics to tracking system
            log_metrics(metrics, step)

    @abstractmethod
    def _eval_step(self, data_iterator) -> tuple:
        """Single evaluation step.

        Args:
            data_iterator: Evaluation data iterator

        Returns:
            Tuple of evaluation metrics

        Subclasses should implement this for their specific evaluation logic.
        """
        pass

    def evaluate(self, global_step: int):
        """Run evaluation on eval datasets.

        Args:
            global_step: Current training step
        """
        if is_first_rank():
            self.logger.info(f"Running evaluation at step {global_step}")

        self.model.eval()

        # Evaluation using data iterator (built-in evaluation)
        if "eval" in self.data_iterator:
            total_loss = 0.0
            total_accuracy = 0.0
            num_batches = self.config.operation.eval_samples // self.config.trainer.eval_batch_size
            if num_batches == 0:
                num_batches = 1

            with torch.no_grad():
                for _ in range(num_batches):
                    loss, accuracy = self._eval_step(self.data_iterator["eval"])
                    total_loss += loss
                    total_accuracy += accuracy

            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches

            # Aggregate across data parallel ranks
            metrics = {"eval_loss": avg_loss, "eval_accuracy": avg_accuracy}
            if dist.is_initialized() and get_data_parallel_world_size() > 1:
                for k, v in metrics.items():
                    v_tensor = torch.tensor(v, device=get_device())
                    dist.all_reduce(v_tensor, op=dist.ReduceOp.SUM, group=get_data_parallel_group())
                    metrics[k] = v_tensor.item() / get_data_parallel_world_size()

            if is_first_rank():
                self.logger.info(
                    f"Evaluation results - step: {global_step}, "
                    f"loss: {metrics['eval_loss']:.4f}, "
                    f"accuracy: {metrics['eval_accuracy']:.4f}"
                )
                log_metrics(metrics, global_step)

        # External evaluators (if any)
        for evaluator in self.evaluators:
            evaluator_name = getattr(evaluator, "name", "external_eval")
            if is_first_rank():
                self.logger.info(f"Evaluating {evaluator_name}")

            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for _ in range(getattr(evaluator, "num_eval_steps", 10)):
                    loss, _ = self._eval_step(evaluator.data_iterator)
                    total_loss += loss
                    num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

            # Aggregate across data parallel ranks
            if dist.is_initialized() and get_data_parallel_world_size() > 1:
                v_tensor = torch.tensor(avg_loss, device=get_device())
                dist.all_reduce(v_tensor, op=dist.ReduceOp.SUM, group=get_data_parallel_group())
                avg_loss = v_tensor.item() / get_data_parallel_world_size()

            if is_first_rank():
                self.logger.info(f"{evaluator_name} - loss: {avg_loss:.4f}")
                log_metric(f"eval/{evaluator_name}/loss", avg_loss, global_step)

    def evaluate_subtask(self, global_step: int):
        """Run evaluation on subtasks (e.g., specific benchmarks).

        Args:
            global_step: Current training step
        """
        self.logger.info(
            f"Subtask evaluation at step {global_step} (default implementation - no-op)"
        )

    def test(self):
        """Run final test evaluation.

        Can be overridden by subclasses for specific test logic.
        """
        self.logger.info("Running final test evaluation")
        self.model.eval()
        # Placeholder - subclasses can implement specific test logic
