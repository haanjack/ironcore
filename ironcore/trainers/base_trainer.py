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

    Subclasses must implement:
    - train(): Main training loop
    - train_step(): Single training step

    Subclasses can override:
    - _eval_step(): Custom evaluation logic
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
        if self.config.utils.profile_nsys:
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

    @abstractmethod
    def train(self):
        """Main training loop.

        Subclasses must implement this method with their specific training logic.
        """
        pass

    @abstractmethod
    def train_step(self, step: int):
        """Single training step.

        Args:
            step: Current training step

        Returns:
            Tuple of (loss, grad_norm, param_norm)

        Subclasses must implement this method with their specific step logic.
        """
        pass

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
        metrics = {
            "loss": loss,
            "step": step,
            "lr": self.lr_scheduler.get_last_lr()[0],
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
