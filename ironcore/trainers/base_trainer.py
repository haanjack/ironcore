# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Base trainer class with common functionality for all training methods."""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Union

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ironcore.checkpointing import load_checkpoint, save_checkpoint
from ironcore.config import MainConfig
from ironcore.controller import TrainingControl
from ironcore.dataloader import get_data_iterator
from ironcore.eval import get_evaluators
from ironcore.global_vars import (
    get_logger,
    get_timer,
    get_tokenizer,
    global_states_cleanup,
    log_metric,
    log_metrics,
    set_global_states,
)
from ironcore.language_model import LanguageModel
from ironcore.mfu import MFUCalculator
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
        """Initialize base trainer configuration.

        The __init__ method is lightweight and primarily for configuration.
        Resource-intensive initialization (distributed process group, model
        creation, etc.) is deferred to __enter__ or explicitly calling
        _initialize().

        Args:
            config: Training configuration
            forward_step_func: Forward step function (may be unused by some trainers)
            loss_fn: Loss function for model
        """
        self.config = config
        self.forward_step_func = forward_step_func
        self.loss_fn = loss_fn

        # State flags
        self._initialized = False
        self._memory_reported = False

        # Configuration that doesn't acquire heavy resources
        set_global_states(config)
        self.timer = get_timer()
        self.logger = get_logger()
        self.control = TrainingControl(config)

    def _initialize(self):
        """Acquire heavy resources needed for training.

        This method initializes distributed process groups, creates the model,
        optimizer, and data loaders. It is idempotent.
        """
        if self._initialized:
            return

        self.logger.info("Acquiring training resources...")

        # Initialize distributed environment
        initialize_process(self.config)

        initialize_model_parallel(
            self.config.trainer.tensor_model_parallel_size,
            timeout_in_minutes=int(self.config.parallel.timeout_minute)
            if self.config.parallel.timeout_minute is not None
            else 10,
        )

        # Initialize expert parallelism if MoE is enabled with EP > 1
        if self.config.model.moe.use_moe and self.config.model.moe.expert_model_parallel_size > 1:
            from ironcore.parallel.expert_parallel import initialize_expert_parallel

            initialize_expert_parallel(
                expert_model_parallel_size=self.config.model.moe.expert_model_parallel_size,
                tensor_model_parallel_size=self.config.trainer.tensor_model_parallel_size,
            )

        # Initialize data loader
        self.data_iterator = get_data_iterator(self.config)

        self.evaluators = get_evaluators(
            self.config.data.eval_datasets,
            self.config.trainer.eval_batch_size,
            self.config.operation.eval_samples,
        )

        # Initialize Profile Manager
        from ironcore.profiler import ProfileManager

        self.profiler = ProfileManager(self.config)

        # Wrap train iterator for data loading profiling (F5)
        if self.config.profiler.data_load_profiler and "train" in self.data_iterator:
            self.data_iterator["train"] = self.profiler.wrap_data_iterator(
                self.data_iterator["train"]
            )

        # Build model and optimizer
        self.model, self.optimizer = self._build_model_and_optimizer()
        self.lr_scheduler = get_lr_scheduler(self.config, self.optimizer)
        self._init_mfu_calculator()

        # Contexts control training process
        self.context: dict[str, Union[nullcontext, torch.autocast]] = {
            "autocast": nullcontext(),
        }

        if self.model.device != "mps":
            self.context["autocast"] = torch.autocast(
                device_type=get_device(), dtype=get_model_dtype(self.config)
            )

        self.scaler = torch.amp.GradScaler(enabled=(get_model_dtype(self.config) == torch.float16))

        self._initialized = True
        self.logger.info("Resources acquired successfully.")

    def __enter__(self):
        self._initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._initialized:
            self._finalize_process()

    def _finalize_process(self):
        """Cleanup resources."""
        # Close loggers before exiting
        global_states_cleanup()

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        self._initialized = False

    def _init_mfu_calculator(self):
        """Initialize MFU calculator for TFLOPS/s/GPU reporting."""
        try:
            tokenizer = get_tokenizer()
            self.mfu_calculator = MFUCalculator.from_config(
                self.config.model,
                vocab_size=tokenizer.padded_vocab_size,
            )
        except Exception as e:
            self.logger.debug(f"MFU calculator unavailable: {e}")
            self.mfu_calculator = None

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

        # Load pretrained weights from HuggingFace if specified
        if self.config.trainer.load_from_hf:
            from ironcore.checkpointing import load_from_huggingface

            hf_model_name = self.config.trainer.load_from_hf
            self.logger.info(f"Loading pretrained weights from HuggingFace: {hf_model_name}")

            # Download checkpoint if needed (handled by huggingface_hub)
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(hf_model_name)
            self.logger.info(f"Downloaded/loaded from cache: {cache_dir}")

            # Load weights into ironcore model
            result = load_from_huggingface(
                checkpoint_path=cache_dir,
                model=model,
                architecture=self.config.model.hf_model_type,
                strict=False,
            )

            self.logger.info(
                f"HF weights loaded: {len(result['loaded_keys'])} keys, "
                f"{len(result['missing_keys'])} missing, "
                f"{len(result['unexpected_keys'])} unexpected"
            )

        optimizer = get_optimizer(self.config, model, device_type=device)
        self.logger.info("Created Optimizer")

        # Wrap with DistributedOptimizer if requested (after optimizer creation, before parallelism)
        if self.config.parallel.use_distributed_optimizer:
            from ironcore.optimizer import DistributedOptimizer

            optimizer = DistributedOptimizer(
                optimizer,
                process_group=get_data_parallel_group(),
                bucket_cap_mb=self.config.parallel.dist_opt_bucket_cap_mb,
            )
            self.logger.info(
                f"Wrapped optimizer with DistributedOptimizer (bucket_cap={self.config.parallel.dist_opt_bucket_cap_mb}MB)"
            )

        # Enable profiling if requested
        if (
            self.config.profiler.gpu_profiler
            or self.config.profiler.torch_profiler
            or self.config.profiler.layer_timing
        ):
            model.register_profile_hooks(
                torch_profiler=self.config.profiler.torch_profiler,
                gpu_profiler=self.config.profiler.gpu_profiler,
                layer_timing=self.config.profiler.layer_timing,
            )

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
                f"Checkpoint loading failed due to a runtime error: {e}. "
                f"Please check the integrity of the checkpoint file or the storage medium. "
                f"If the checkpoint is corrupted, remove or rename {self.config.trainer.model_path} and restart."
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
        # Ensure resources are acquired if not using context manager
        self._initialize()

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

        self.logger.info(f"Training start from step: {step}")
        while step < self.config.operation.train_steps:
            loss, grad_norm, param_norm = self.train_step(step)

            step += 1
            self.log_training(step, loss, grad_norm, param_norm, self.timer)

            self.profiler.step(step)
            if (
                self.config.profiler.stop_at_end
                and step >= self.config.profiler.end
                and not self.profiler.is_active
            ):
                self.logger.info("Stopping training as requested by stop_at_end")
                break

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

        from ironcore.parallel import parallel_states
        from ironcore.utils import clip_grad_norm_tp

        # Unscale gradients before clipping/norm computation
        self.scaler.unscale_(self.optimizer)

        grad_norm = 0.0
        if self.config.optim.clip_grad > 0.0:
            # Active gradient clipping: clip to specified threshold
            if isinstance(self.model, FSDP):
                grad_norm = self.model.clip_grad_norm_(self.config.optim.clip_grad).item()
            else:
                grad_norm = clip_grad_norm_tp(self.model.parameters(), self.config.optim.clip_grad)
        elif self.control.do_grad_norm(step):
            # No clipping, but compute norm for logging (clip_grad=inf means compute but don't clip)
            if isinstance(self.model, FSDP):
                grad_norm = self.model.clip_grad_norm_(float("inf")).item()
            else:
                grad_norm = clip_grad_norm_tp(self.model.parameters(), float("inf"))

        param_norm = 0.0
        if self.control.do_param_norm(step):
            # Compute local squared norm.
            # For FSDP, p.data is the local shard.
            param_norm_sq = 0.0
            for p in self.model.parameters():
                if p.data is not None:
                    param_norm_sq += p.data.norm() ** 2

            # All-reduce once after accumulating all parameter norms
            param_norm_tensor = torch.tensor(param_norm_sq, device=get_device())

            # For FSDP, sync across DP group (parameters sharded across DP ranks)
            if isinstance(self.model, FSDP):
                dist.all_reduce(
                    param_norm_tensor, op=dist.ReduceOp.SUM, group=get_data_parallel_group()
                )
            # For TP, sync across TP group (parameters sharded across TP ranks)
            elif parallel_states.get_tensor_model_parallel_world_size() > 1:
                dist.all_reduce(
                    param_norm_tensor,
                    op=dist.ReduceOp.SUM,
                    group=parallel_states.get_tensor_model_parallel_group(),
                )

            param_norm_sq = param_norm_tensor.item()
            param_norm = param_norm_sq**0.5

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
                f"Possible causes: learning rate too high, gradient explosion, or data issues. "
                f"Consider enabling `torch.autograd.set_detect_anomaly(True)` for more debugging information."
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
        iter_time: float = 0.0
        if timer is not None:
            iter_time = timer.get("iter")
            metrics["iter_time"] = iter_time
            if iter_time > 0:
                tokens_per_sec = (
                    self.config.trainer.train_batch_size * self.config.model.max_seq_len / iter_time
                )
                metrics["tokens_per_sec"] = tokens_per_sec

                if self.mfu_calculator is not None:
                    dp_world_size = get_data_parallel_world_size() if dist.is_initialized() else 1
                    micro_batch_size = self.config.trainer.micro_batch_size or 1
                    gradient_accumulation_steps = (
                        self.config.trainer.gradient_accumulation_steps or 1
                    )
                    global_batch_size = (
                        micro_batch_size * gradient_accumulation_steps * dp_world_size
                    )
                    tflops = self.mfu_calculator.compute_tflops(
                        batch_size=global_batch_size,
                        seq_len=self.config.model.max_seq_len,
                        step_time_seconds=iter_time,
                        num_gpus=dp_world_size,
                    )
                    metrics["tflops_per_gpu"] = tflops

        # Memory report: on first log step, then at every checkpoint interval
        if (
            is_first_rank()
            and self.config.utils.report_memory_usage
            and (
                (self.control.do_log(step) and not self._memory_reported)
                or self.control.do_checkpoint(step)
            )
        ):
            from ironcore.utils import (
                format_memory_report,
                get_detailed_memory_breakdown,
            )

            breakdown = get_detailed_memory_breakdown(self.model, self.optimizer, in_mib=True)
            report = format_memory_report(breakdown, "Memory Breakdown")
            print(report, flush=True)
            self._memory_reported = True

        # Log to console and tracking
        if self.control.do_log(step):
            # Accumulate data loading stats across the full log interval before logging
            if self.config.profiler.data_load_profiler:
                dl_stats = self.profiler.get_data_load_stats()
                if dl_stats is not None and dl_stats["count"] > 0:
                    metrics["data_load_ms_per_step"] = dl_stats["total_ms"] / dl_stats["count"]
                    if timer is not None and iter_time > 0:
                        metrics["data_load_ratio"] = metrics["data_load_ms_per_step"] / (
                            iter_time * 1000.0
                        )

            log_msg = f"step: {step}, loss: {loss:.4f}, lr: {metrics['lr']:.6f}"
            if grad_norm > 0:
                log_msg += f", grad_norm: {grad_norm:.4f}"
            if timer is not None:
                log_msg += f", iter_time: {iter_time:.3f}s"
                if "tokens_per_sec" in metrics:
                    log_msg += f", tok/s: {metrics['tokens_per_sec']:.1f}"
                if "tflops_per_gpu" in metrics:
                    log_msg += f", TFLOPS/s/GPU: {metrics['tflops_per_gpu']:.2f}"
            if "data_load_ms_per_step" in metrics:
                log_msg += f", data_load: {metrics['data_load_ms_per_step']:.1f}ms/step"
                if "data_load_ratio" in metrics:
                    log_msg += f" ({metrics['data_load_ratio'] * 100:.1f}%)"
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
                    dist.all_reduce(
                        v_tensor,
                        op=dist.ReduceOp.SUM,
                        group=get_data_parallel_group(),
                    )
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
                for batch in evaluator.data_loader:
                    loss, _ = self._eval_step(batch)
                    total_loss += loss
                    num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

            # Aggregate across data parallel ranks
            if dist.is_initialized() and get_data_parallel_world_size() > 1:
                v_tensor = torch.tensor(avg_loss, device=get_device())
                dist.all_reduce(
                    v_tensor,
                    op=dist.ReduceOp.SUM,
                    group=get_data_parallel_group(),
                )
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
