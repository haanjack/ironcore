# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Base trainer class with common functionality for all training methods."""

import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Union

import torch
import torch._dynamo
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
from ironcore.utils.mfu import MFUCalculator


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

    def _setup_data_iterators(self):
        """Initialize data iterators. Override in subclasses for custom data loading."""
        self.data_iterator = get_data_iterator(self.config)

    def _initialize(self):
        """Acquire heavy resources needed for training.

        This method initializes distributed process groups, creates the model,
        optimizer, and data loaders. It is idempotent.
        """
        if self._initialized:
            return
        from ironcore.config.config_blockwise import validate_blockwise_mlp
        from ironcore.config.config_context_parallel import validate_context_parallel

        validate_blockwise_mlp(self.config)
        validate_context_parallel(self.config)
        if self.config.parallel.use_fsdp and self.config.parallel.use_distributed_optimizer:
            raise ValueError(
                "FSDP and DistributedOptimizer cannot partition the same optimizer states"
            )

        if self.config.parallel.use_fsdp and self.config.trainer.tensor_model_parallel_size != 1:
            raise ValueError("FSDP native trainer checkpoints currently require TP=1")
        if self.config.model.moe.use_moe and self.config.model.moe.expert_model_parallel_size > 1:
            if (
                self.config.model.moe.expert_model_parallel_size != 2
                or self.config.trainer.tensor_model_parallel_size != 1
                or self.config.parallel.world_size != 2 * self.config.trainer.context_parallel_size
                or self.config.parallel.use_fsdp
                or self.config.parallel.use_distributed_optimizer
                or self.config.offload.enabled
            ):
                raise ValueError(
                    "EP>1 trainer requires EP=2, TP=1, world=2*CP without FSDP/distributed optimizer/offload"
                )

        self.logger.info("Acquiring training resources...")

        # Initialize distributed environment
        initialize_process(self.config)

        initialize_model_parallel(
            self.config.trainer.tensor_model_parallel_size,
            timeout_in_minutes=int(self.config.parallel.timeout_minute)
            if self.config.parallel.timeout_minute is not None
            else 10,
            context_parallel_size=self.config.trainer.context_parallel_size,
        )

        # Initialize expert parallelism if MoE is enabled with EP > 1
        if self.config.model.moe.use_moe and self.config.model.moe.expert_model_parallel_size > 1:
            from ironcore.parallel.expert_parallel import initialize_expert_parallel

            initialize_expert_parallel(
                expert_model_parallel_size=self.config.model.moe.expert_model_parallel_size,
                tensor_model_parallel_size=self.config.trainer.tensor_model_parallel_size,
                context_parallel_size=self.config.trainer.context_parallel_size,
            )

        # Initialize data loader
        self.data_iterator = self._get_data_iterator()

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

        device_type = torch.device(get_device()).type
        if device_type != "mps" and get_model_dtype(self.config) != torch.float32:
            self.context["autocast"] = torch.autocast(
                device_type=device_type, dtype=get_model_dtype(self.config)
            )

        self.scaler = torch.amp.GradScaler(enabled=(get_model_dtype(self.config) == torch.float16))

        # Initialize weight streaming scheduler if enabled
        self._offload_scheduler = None
        offload_needs_scheduler = self.config.offload.enabled and (
            self.config.offload.weight_offload or self.config.offload.activation_spill
        )
        if offload_needs_scheduler:
            from ironcore.offload.scheduler import ExecutionScheduler

            # Unwrap torch.compile + DDP + LanguageModel to get TransformerModel
            inner_model = self.model
            if hasattr(inner_model, "_orig_mod"):
                inner_model = inner_model._orig_mod
            if isinstance(inner_model, torch.nn.parallel.DistributedDataParallel):
                inner_model = inner_model.module
            # LanguageModel wraps TransformerModel in LanguageModel.model
            if hasattr(inner_model, "model") and hasattr(inner_model.model, "layers"):
                inner_model = inner_model.model

            # Get DP group for ZeRO-3 parameter sharding
            dp_group = None
            if self.config.offload.weight_offload and dist.is_initialized():
                dp_world_size = get_data_parallel_world_size()
                if dp_world_size > 1:
                    dp_group = get_data_parallel_group()

            self._offload_scheduler = ExecutionScheduler.from_model(
                model=inner_model,
                config=self.config.offload,
                device=torch.device(get_device()),
                dp_group=dp_group,
            )
            if self._offload_scheduler is not None and self._offload_scheduler.is_active:
                # Attach scheduler to model so forward pass can call per-layer hooks
                inner_model._offload_scheduler = self._offload_scheduler
                self.logger.info(f"Weight streaming scheduler attached: {self._offload_scheduler}")
            elif (
                self._offload_scheduler is not None
                and self._offload_scheduler.spill_manager is not None
            ):
                # Scheduler created for activation spilling only (no weight streaming)
                inner_model._offload_scheduler = self._offload_scheduler
                self.logger.info(f"Activation spill scheduler attached: {self._offload_scheduler}")
            else:
                self._offload_scheduler = None

            # Set gradient accumulation steps for activation spill manager
            if self._offload_scheduler is not None:
                self._offload_scheduler.set_gradient_accumulation_steps(
                    self.config.trainer.gradient_accumulation_steps
                )

            # Configure CPU thread count for optimizer offload compute
            if self.config.offload.optimizer_cpu_threads > 0:
                from ironcore.offload.optimizer_helpers import configure_cpu_threads

                configure_cpu_threads(self.config.offload.optimizer_cpu_threads)
                self.logger.info(
                    f"Optimizer CPU threads set to {self.config.offload.optimizer_cpu_threads}"
                )

        self._initialized = True
        self.logger.info("Resources acquired successfully.")

    def __enter__(self):
        self._initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._initialized:
            self._finalize_process()

    def _finalize_process(self, destroy_process_group: bool = True):
        """Cleanup resources.

        Args:
            destroy_process_group: Tear the global process group down as well.
                True for a real run, which exits straight afterwards. Tests that
                call this between cases in one torchrun session must pass False:
                the next case re-creates the group on the same MASTER_PORT, and
                a rank that gets there before rank 0 has re-hosted the TCPStore
                fails with "Connection refused" while its peer sails on.
        """
        # Shutdown weight streaming scheduler (releases GPU staging buffers)
        if hasattr(self, "_offload_scheduler") and self._offload_scheduler is not None:
            self._offload_scheduler.shutdown()
            self._offload_scheduler = None

        # Close loggers before exiting
        global_states_cleanup()

        if dist.is_initialized():
            dist.barrier()
            if destroy_process_group:
                dist.destroy_process_group()

        if destroy_process_group:
            from ironcore.parallel.parallel_states import destroy_model_parallel

            destroy_model_parallel()

        self._initialized = False

    def _get_data_iterator(self):
        """Return data iterators for train/eval/test splits.

        Subclasses can override this to use task-specific data pipelines
        (e.g., GRPOTrainer uses get_grpo_data_iterator).
        """
        return get_data_iterator(self.config)

    def _init_mfu_calculator(self):
        """Initialize MFU calculator for TFLOPS/s/GPU reporting."""
        if self.config.model.moe.use_moe:
            self.mfu_calculator = None
            self.logger.info("Dense FLOP estimate disabled for MoE; use measured tokens/s.")
            return
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
        parameter_dtype = (
            torch.float32
            if self.config.trainer.parameter_precision == "float32"
            else get_model_dtype(self.config)
        )
        weight_streaming = self.config.offload.enabled and self.config.offload.weight_offload

        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(parameter_dtype)
            model = LanguageModel(self.config, self.loss_fn)
        finally:
            torch.set_default_dtype(original_dtype)
        if (
            self.config.peft.method == "lora"
            and self.config.peft.lora.parameter_precision == "float32"
        ):
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.data = parameter.data.float()

        if weight_streaming:
            # Weight streaming: Keep model on CPU — ExecutionScheduler manages per-layer GPU staging.
            # This avoids OOM for models whose weights exceed GPU memory (e.g. 13B on 24GB).

            # With TP > 1, the embedding and output head must stay on GPU because
            # VocabParallelEmbedding and vocab_parallel_cross_entropy call dist.all_reduce
            # which requires CUDA tensors when using NCCL backend.
            tp_size = self.config.trainer.tensor_model_parallel_size
            if tp_size > 1 or self.config.model.is_gemma4:
                gpu = torch.device(device)
                model.embedding = model.embedding.to(gpu)
                model.output_layernorm = model.output_layernorm.to(gpu)
                if hasattr(model, "output_layer"):
                    model.output_layer = model.output_layer.to(gpu)
                if self.config.model.is_gemma4:
                    for module in model.model.modules():
                        for name, buffer in module.named_buffers(recurse=False):
                            module._buffers[name] = buffer.to(gpu)
                self.logger.info(
                    f"TP={tp_size}: embedding + output head on {gpu}, "
                    "transformer layers on CPU for weight streaming"
                )

            self.logger.info("Created Language Model on CPU (weight streaming mode)")
        else:
            model = model.to(device=device)
            self.logger.info("Created Language Model")
        if self.config.model.moe.use_moe and self.config.model.moe.expert_model_parallel_size > 1:
            import copy

            from ironcore.checkpointing.expert import global_parameter_names

            full_config = copy.deepcopy(self.config)
            full_config.model.moe.expert_model_parallel_size = 1
            torch.manual_seed(seed)
            full_model = LanguageModel(full_config, self.loss_fn).to(
                device=device, dtype=parameter_dtype
            )
            full_state = full_model.state_dict()
            names = global_parameter_names(model)
            model.load_state_dict(
                {name: full_state[global_name] for name, global_name in names.items()}
            )
            del full_state, full_model

        # Load pretrained weights from HuggingFace if specified
        if self.config.trainer.load_from_hf:
            from ironcore.checkpointing import load_from_huggingface

            hf_model_name = self.config.trainer.load_from_hf
            self.logger.info(f"Loading pretrained weights from HuggingFace: {hf_model_name}")

            # Download checkpoint if needed (handled by huggingface_hub)
            from huggingface_hub import snapshot_download

            cache_dir = (
                str(Path(hf_model_name).resolve())
                if Path(hf_model_name).is_dir()
                else snapshot_download(hf_model_name)
            )
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
            from ironcore.checkpointing.hf_interop import validate_imported_base_parameters

            validate_imported_base_parameters(model, result["missing_keys"])

        if self.config.peft.method == "lora" and self.config.peft.lora.adapter_path:
            from ironcore.peft import load_lora_adapter

            load_lora_adapter(model, self.config.peft.lora.adapter_path)
        optimizer = get_optimizer(self.config, model)
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

        # Defensive: verify optimizer holds references to original model params for FSDP+offload
        if self.config.offload.optimizer_offload and self.config.parallel.use_fsdp:
            optim_params = {p for group in optimizer.param_groups for p in group["params"]}
            model_params = set(model.parameters())
            if not optim_params.issubset(model_params):
                raise RuntimeError(
                    "Optimizer parameters don't match model parameters. "
                    "This will break with FSDP + optimizer_offload."
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
            tp_size = self.config.trainer.tensor_model_parallel_size
            if tp_size > 1 or self.config.trainer.context_parallel_size > 1:
                self.logger.info(
                    f"torch.compile skipped for TP={tp_size}, CP={self.config.trainer.context_parallel_size}: "
                    f"dynamo tracing is incompatible with parallel collective communication "
                    f"(all-reduce/all-gather in custom autograd Functions)."
                )
            else:
                torch._dynamo.config.optimize_ddp = False
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

        if device != "mps" and not weight_streaming:
            model = initialize_parallelism(self.config, model)
        self.rank = dist.get_rank()

        return model, optimizer

    def _skip_consumed_train_samples(self, last_step: int) -> None:
        """Advance the train iterator past samples already seen before resume.

        Without this, a resumed run replays the dataset from the start,
        re-training on samples the original run already consumed. The streaming
        dataset is seeded deterministically, so advancing the iterator by exactly
        what the original run drew reproduces its trajectory. (Fable issue #62.)

        Count in micro-batches, not samples: each ``next()`` yields a whole
        micro-batch, so burning ``last_step * train_batch_size`` of them
        overshot by the micro-batch size and landed the resumed run on entirely
        different data.
        """
        if last_step <= 0 or not self.data_iterator or "train" not in self.data_iterator:
            return
        if hasattr(self.data_iterator["train"], "load_state_dict"):
            rank = dist.get_rank() if dist.is_initialized() else 0
            path = (
                Path(self.config.trainer.model_path)
                / f"step_{last_step}"
                / f"trainer_rank{rank}.pt"
            )
            if path.exists() and "train" in torch.load(
                path, map_location="cpu", weights_only=True
            ).get("data", {}):
                return
        accum_steps = self.config.trainer.gradient_accumulation_steps
        if not accum_steps:
            return
        skip = last_step * accum_steps
        self.logger.info(
            f"Skipping {skip} consumed micro-batches to resume data position "
            f"({skip * self.config.trainer.micro_batch_size} samples on this rank)."
        )
        train_iter = self.data_iterator["train"]
        for _ in range(skip):
            try:
                next(train_iter)
            except StopIteration:
                break

    def _pre_train_setup(self) -> int:
        """Hook for setup before training starts.

        Override this method to perform custom setup (e.g., checkpoint loading,
        reference model creation for DPO).

        Returns:
            Starting step number (0 for fresh training, or checkpoint step + 1)
        """  # Default implementation: load checkpoint if available
        try:
            last_step = load_checkpoint(self.config, self.model, self.optimizer, self.lr_scheduler)
            if last_step > -1:
                self.logger.info(
                    f"Successfully loaded checkpoint: {self.config.trainer.model_path} "
                    f"(resuming from step {last_step})"
                )
                self._skip_consumed_train_samples(last_step)
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
        from .checkpoint_state import load_trainer_state

        load_trainer_state(self, last_step)
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
        from .checkpoint_state import save_trainer_state

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
        self._train_wall_start = time.time()
        self._train_step_start = last_step

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
                save_trainer_state(self, step)
                save_checkpoint(self.config, self.model, self.optimizer, self.lr_scheduler, step)

                if self.control.do_eval_subtask(step):
                    self.evaluate_subtask(step)
                    self.model.train()

                if self.control.do_exit(step):
                    self.logger.info(
                        f"Training stopped by exit interval: {self.config.operation.exit_interval}"
                    )
                    break

            if self.control.do_eval(step):
                self.evaluate(step)
                self.model.train()

        # Final checkpoint if needed
        if self.control.do_final_checkpoint(step, last_step):
            self._on_checkpoint_save(step)
            save_trainer_state(self, step)
            save_checkpoint(self.config, self.model, self.optimizer, self.lr_scheduler, step)

        # Eval-only mode: train_steps == 0 means skip training, run eval once.
        if self.config.operation.train_steps == 0 and self.config.trainer.do_eval_subtask:
            self.logger.info("Eval-only mode (train_steps=0): running evaluation.")
            self.evaluate(global_step=0)
            self.model.train()

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
        from ironcore.training_utils import batch_objective_count

        accumulation = self.config.trainer.gradient_accumulation_steps
        source_iterator = self.data_iterator["train"]
        batches = [next(source_iterator) for _ in range(accumulation)]
        if "input_ids" in batches[0]:
            self._last_sequence_length = batches[0]["input_ids"].size(1)
        task = getattr(getattr(self.config, "data", None), "task_type", None)
        if task is None:
            task = "sft" if self.loss_fn.__name__ == "loss_func_sft" else "pretrain"
        counts = [batch_objective_count(batch, task) for batch in batches]
        dp_size = get_data_parallel_world_size() if dist.is_initialized() else 1
        denominator = torch.tensor(sum(counts), dtype=torch.float64, device=get_device())
        if dp_size > 1:
            dist.all_reduce(denominator, group=get_data_parallel_group())
        denominator = denominator.item()
        if denominator == 0:
            raise ValueError("Training update contains no valid objective tokens/samples")
        total_loss = torch.zeros((), dtype=torch.float64, device=get_device())
        total_metrics: dict[str, float] = {}

        # Weight streaming: prefetch first layers before forward pass
        if self._offload_scheduler is not None:
            self._offload_scheduler.on_training_step_start()

        for i in range(accumulation):
            # Prefetch inputs only, not activation graphs. Global denominators
            # account for unequal token/sample counts across microbatches/ranks.
            self.data_iterator["train"] = iter([batches[i]])
            is_last_accum_step = i == self.config.trainer.gradient_accumulation_steps - 1

            # Notify spill manager of micro-batch forward start
            if self._offload_scheduler is not None:
                self._offload_scheduler.on_microbatch_forward_start(i)

            # Disable gradient sync for intermediate accumulation steps (DDP/FSDP)
            backward_sync_ctx = (
                self.model.no_sync
                if not is_last_accum_step and hasattr(self.model, "no_sync")
                else nullcontext
            )

            with backward_sync_ctx():
                with self.context["autocast"]:
                    try:
                        loss, metrics = self._forward_micro_batch(step)
                    finally:
                        self.data_iterator["train"] = source_iterator

                    weight = counts[i] * dp_size / denominator
                    total_loss += loss.detach().to(total_loss) * weight * accumulation
                    scaled_loss = loss * weight

                    # Accumulate metrics if provided
                    if metrics:
                        for k, v in metrics.items():
                            total_metrics[k] = total_metrics.get(k, 0.0) + v * weight * accumulation

                # Notify spill manager that forward is done for this micro-batch
                if self._offload_scheduler is not None:
                    self._offload_scheduler.on_microbatch_forward_end()

                # Notify spill manager of micro-batch backward start
                if self._offload_scheduler is not None:
                    self._offload_scheduler.on_microbatch_backward_start(i)

                # Backward pass with gradient scaling
                self.scaler.scale(scaled_loss).backward()

                # Notify spill manager that backward is done for this micro-batch
                if self._offload_scheduler is not None:
                    self._offload_scheduler.on_microbatch_backward_end()

        self.data_iterator["train"] = source_iterator

        # Weight streaming: zero GPU staging buffers after all micro-batches' backward passes.
        # Actual param.data stays on GPU until next step's on_layer_start overwrites it.
        if self._offload_scheduler is not None:
            self._offload_scheduler.on_backward_pass_end()

        # Report the objective averaged over DP workers, matching DDP's
        # gradient averaging. TP ranks are evaluating the same objective.
        if dist.is_initialized():
            loss_tensor = total_loss
            if get_data_parallel_world_size() > 1:
                dist.all_reduce(loss_tensor, group=get_data_parallel_group())
                loss_tensor /= get_data_parallel_world_size()
            total_loss = loss_tensor
        total_loss = total_loss.item()
        self._validate_distributed_loss(total_loss, step)
        total_metrics = self._average_dp_metrics(total_metrics)

        return total_loss, total_metrics

    def _average_dp_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Average metrics from equally sized local batches across DP workers."""
        if not metrics or not dist.is_initialized() or get_data_parallel_world_size() == 1:
            return metrics
        keys = sorted(metrics)
        values = torch.tensor(
            [metrics[key] for key in keys], device=get_device(), dtype=torch.float64
        )
        dist.all_reduce(values, group=get_data_parallel_group())
        values /= get_data_parallel_world_size()
        return dict(zip(keys, values.tolist(), strict=True))

    def _validate_distributed_loss(self, loss: float, step: int) -> None:
        """Stop all workers before an update if any worker saw a bad loss."""
        import math

        if dist.is_initialized():
            invalid = torch.tensor(int(not math.isfinite(loss)), device=get_device())
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
            if invalid.item():
                loss = float("nan")
        self._check_loss_for_nan(loss, step)

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

    def _prepare_gradients(self):
        if getattr(self, "_gradients_prepared", False):
            return
        # Unscale gradients before clipping/norm computation
        self.scaler.unscale_(self.optimizer)
        if hasattr(self.model, "synchronize_gradients"):
            self.model.synchronize_gradients()
        from ironcore.parallel.context_parallel import synchronize_context_parallel_gradients

        synchronize_context_parallel_gradients(self.model)
        # A finite objective can have a nonfinite derivative. BF16 has no
        # GradScaler, so loss checks alone cannot protect the optimizer state.
        invalid = torch.zeros((), dtype=torch.int32, device=get_device())
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                invalid.copy_(
                    torch.maximum(invalid, (~torch.isfinite(parameter.grad).all()).to(invalid))
                )
        if dist.is_initialized():
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        if invalid.item():
            self.optimizer.zero_grad()
            raise RuntimeError(
                "NaN/Inf gradient detected; optimizer and scheduler were not updated"
            )

        self._gradients_prepared = True

    def _compute_grad_and_param_norms(self, step: int) -> tuple[float, float]:
        """Compute gradient and parameter norms after gradient accumulation.

        This method unscales gradients and optionally clips them.

        Args:
            step: Current training step

        Returns:
            Tuple of (grad_norm, param_norm)
        """

        from ironcore.parallel.grad_norm import clip_grad_norm, compute_param_norm

        self._prepare_gradients()

        grad_norm = 0.0
        if self.config.optim.clip_grad > 0.0:
            # Active gradient clipping: clip to specified threshold
            if isinstance(self.model, FSDP):
                grad_norm = self.model.clip_grad_norm_(self.config.optim.clip_grad).item()
            else:
                grad_norm = clip_grad_norm(
                    self.model.parameters(), self.config.optim.clip_grad
                ).item()
        elif self.control.do_grad_norm(step):
            # No clipping, but compute norm for logging (clip_grad=inf means compute but don't clip)
            if isinstance(self.model, FSDP):
                grad_norm = self.model.clip_grad_norm_(float("inf")).item()
            else:
                grad_norm = clip_grad_norm(self.model.parameters(), float("inf")).item()

        param_norm = 0.0
        if self.control.do_param_norm(step):
            param_norm = compute_param_norm(
                self.model.parameters(), is_fsdp=isinstance(self.model, FSDP)
            )

        return grad_norm, param_norm

    def _optimizer_step(self):
        """Perform optimizer step after gradient accumulation."""
        try:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            # GradScaler reduces its scale when it skips an overflowing update.
            if self.scaler.get_scale() >= scale_before:
                self.lr_scheduler.step()
        finally:
            self._gradients_prepared = False
            # Weight streaming: synchronize all transfers, prepare for next step.
            # Must run even on exception to free pinned grad buffers and
            # prevent pool budget exhaustion.
            if self._offload_scheduler is not None:
                self._offload_scheduler.on_training_step_end()

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

        # Timing metrics — only computed on logging steps.
        # get_interval() resets the buffer, so calling it on every step would leave
        # only one sample in the interval instead of log_interval samples.
        iter_time: float = 0.0
        if timer is not None and self.control.do_log(step):
            iter_time = timer.get_interval("iter")  # avg over this log interval, resets buffer
            metrics["iter_time"] = iter_time

            # Overall average from wall clock since training (re)started
            train_start = getattr(self, "_train_wall_start", None)
            train_step_start = getattr(self, "_train_step_start", 0)
            steps_done = step - train_step_start
            if train_start is not None and steps_done > 0:
                metrics["avg_iter_time"] = (time.time() - train_start) / steps_done

            sequence_length = getattr(self, "_last_sequence_length", self.config.model.max_seq_len)
            tokens_per_step = self.config.trainer.train_batch_size * sequence_length

            if iter_time > 0:
                metrics["tokens_per_sec"] = tokens_per_step / iter_time

            avg_iter_time = metrics.get("avg_iter_time", 0.0)
            if avg_iter_time > 0:
                metrics["avg_tokens_per_sec"] = tokens_per_step / avg_iter_time

            if self.mfu_calculator is not None:
                dp_world_size = get_data_parallel_world_size() if dist.is_initialized() else 1
                gpu_world_size = dist.get_world_size() if dist.is_initialized() else 1
                micro_batch_size = self.config.trainer.micro_batch_size or 1
                gradient_accumulation_steps = self.config.trainer.gradient_accumulation_steps or 1
                global_batch_size = micro_batch_size * gradient_accumulation_steps * dp_world_size

                if iter_time > 0:
                    metrics["tflops_per_gpu"] = self.mfu_calculator.compute_tflops(
                        batch_size=global_batch_size,
                        seq_len=sequence_length,
                        step_time_seconds=iter_time,
                        num_gpus=gpu_world_size,
                    )
                if avg_iter_time > 0:
                    metrics["avg_tflops_per_gpu"] = self.mfu_calculator.compute_tflops(
                        batch_size=global_batch_size,
                        seq_len=sequence_length,
                        step_time_seconds=avg_iter_time,
                        num_gpus=gpu_world_size,
                    )

        # Offload metrics: log when scheduler is active
        if self._offload_scheduler is not None:
            offload_metrics = self._offload_scheduler.get_metrics()
            metrics["step_elapsed_ms"] = offload_metrics["step_elapsed_ms"]
            metrics["h2d_ms"] = offload_metrics["h2d_ms"]
            metrics["d2h_snapshot_ms"] = offload_metrics["d2h_snapshot_ms"]
            metrics["host_pool_used_mb"] = offload_metrics["host_pool_used_mb"]

            # Host RAM (current RSS)
            try:
                from ironcore.utils.memory import get_host_memory_usage

                host_mem = get_host_memory_usage()
                metrics["host_rss_mb"] = host_mem["rss_mb"]
            except Exception:
                pass

            # VRAM
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                metrics["vram_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
                metrics["vram_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 * 1024)

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
                log_msg += f", step_time: {iter_time:.3f}s"
                if "avg_iter_time" in metrics:
                    log_msg += f" (avg: {metrics['avg_iter_time']:.3f}s)"
                if "tokens_per_sec" in metrics:
                    tok_s = metrics["tokens_per_sec"]
                    avg_tok_s = metrics.get("avg_tokens_per_sec")
                    log_msg += f", tok/s: {tok_s:.1f}"
                    if avg_tok_s is not None:
                        log_msg += f" (avg: {avg_tok_s:.1f})"
                if "tflops_per_gpu" in metrics:
                    tflops = metrics["tflops_per_gpu"]
                    avg_tflops = metrics.get("avg_tflops_per_gpu")
                    log_msg += f", TFLOPS/s/GPU: {tflops:.2f}"
                    if avg_tflops is not None:
                        log_msg += f" (avg: {avg_tflops:.2f})"
            if "data_load_ms_per_step" in metrics:
                log_msg += f", data_load: {metrics['data_load_ms_per_step']:.1f}ms/step"
                if "data_load_ratio" in metrics:
                    log_msg += f" ({metrics['data_load_ratio'] * 100:.1f}%)"
            if "step_elapsed_ms" in metrics:
                log_msg += f", step_elapsed: {metrics['step_elapsed_ms']:.1f}ms"
                log_msg += f" (h2d={metrics['h2d_ms']:.1f}, d2h={metrics['d2h_snapshot_ms']:.1f})"
                if "host_rss_mb" in metrics:
                    log_msg += f", host_rss: {metrics['host_rss_mb']:.0f}MB"
                if "vram_allocated_mb" in metrics:
                    log_msg += f", vram: {metrics['vram_allocated_mb']:.0f}MB"
            self.logger.info(log_msg)

            # Log all metrics to tracking system
            log_metrics(metrics, step)

    @abstractmethod
    def _eval_step(self, batch) -> tuple:
        """Single evaluation step.

        Args:
            batch: A batch dict from the data loader

        Returns:
            Tuple of evaluation metrics

        Subclasses should implement this for their specific evaluation logic.
        """
        pass

    def _evaluation_batches(self, count):
        """Restart held-out data and stop all ranks together at its finite end."""
        iterator = self._get_data_iterator()["eval"]
        self.data_iterator["eval"] = iterator
        for _ in range(count):
            try:
                batch = next(iterator)
                present = 1
            except StopIteration:
                batch, present = None, 0
            available = torch.tensor(present, device=get_device())
            if dist.is_initialized():
                dist.all_reduce(available, op=dist.ReduceOp.MIN)
            if not available.item():
                break
            yield batch

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
            from ironcore.training_utils import batch_objective_count

            task = self.config.data.task_type
            totals = torch.zeros(4, dtype=torch.float64, device=get_device())
            num_batches = max(
                1,
                self.config.operation.eval_samples
                // (self.config.trainer.eval_batch_size * get_data_parallel_world_size()),
            )
            with torch.no_grad():
                for batch in self._evaluation_batches(num_batches):
                    loss, accuracy = self._eval_step(batch)
                    units = batch_objective_count(batch, task)
                    accuracy_units = (
                        int((batch["labels"] != -100).sum()) if task != "dpo" else units
                    )
                    totals += torch.tensor(
                        [loss * units, units, accuracy * accuracy_units, accuracy_units],
                        device=get_device(),
                        dtype=torch.float64,
                    )
            if dist.is_initialized() and get_data_parallel_world_size() > 1:
                dist.all_reduce(totals, group=get_data_parallel_group())
            if not totals[1].item():
                raise ValueError("Evaluation produced no valid objective units")
            metrics = {
                "eval_loss": (totals[0] / totals[1].clamp(min=1)).item(),
                "eval_accuracy": (totals[2] / totals[3].clamp(min=1)).item(),
            }

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

            from ironcore.training_utils import batch_objective_count

            totals = torch.zeros(2, dtype=torch.float64, device=get_device())

            with torch.no_grad():
                for batch in evaluator.data_loader:
                    loss, _ = self._eval_step(batch)
                    units = batch_objective_count(batch, self.config.data.task_type)
                    totals += totals.new_tensor([loss * units, units])

            # Aggregate across data parallel ranks
            if dist.is_initialized() and get_data_parallel_world_size() > 1:
                dist.all_reduce(
                    totals,
                    op=dist.ReduceOp.SUM,
                    group=get_data_parallel_group(),
                )
            avg_loss = (totals[0] / totals[1].clamp(min=1)).item()

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
