# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import math
from abc import ABC
from contextlib import nullcontext
from typing import Dict, Union

import torch
from torch import distributed as dist
from torch import nn
from torch.profiler import ProfilerActivity, profile

from ironcore.checkpointing import load_checkpoint, save_checkpoint
from ironcore.config import MainConfig
from ironcore.dataloader import get_data_iterator
from ironcore.eval import get_evaluators
from ironcore.global_vars import (
    global_states_cleanup,
    get_logger,
    get_timer,
    log_histogram,
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
from ironcore.controller import TrainingControl
from ironcore.utils import (
    Timer,
    get_device,
    get_memory_usage,
    get_model_dtype,
    is_first_rank,
    clip_grad_norm_tp,
)
from ironcore.training_utils import compute_token_accuracy, get_batch


class Trainer(ABC):
    """Trainer for given language model."""

    def __init__(
        self,
        config: MainConfig,
        forward_step_func,
        loss_fn,
    ):
        """Create Trainer"""

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
            timeout_in_minutes=int(config.parallel.timeout_minute) if config.parallel.timeout_minute is not None else 10.0,
        )

        # initialize data loader
        self.data_iterator = get_data_iterator(config)

        self.evaluators = get_evaluators(
            config.data.eval_datasets,
            config.trainer.eval_batch_size,
            config.operation.eval_samples,
        )

        # contexts contols training process
        self.context: Dict[str, Union[nullcontext, torch.autocast]] = {
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

        self.scaler = torch.amp.GradScaler(
            enabled=(get_model_dtype(config) == torch.float16)
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finialize_process()

    def _finialize_process(self):
        # Close loggers before exiting
        global_states_cleanup()

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _build_model_and_optimizer(self):
        """Build model and optimizer."""

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

        if self.config.utils.profile_torch:
            self.logger.info("Enabled PyTorch profiler")
            self.context["profile"] = profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.config.utils.tensorboard_dir),
                schedule=torch.profiler.schedule(
                    skip_first=0,
                    wait=1,
                    warmup=self.config.utils.profile_step_start-1,
                    active=self.config.utils.profile_step_end -
                    self.config.utils.profile_step_start,
                ),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                with_flops=True,
            )
            # experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            model.register_profile_hooks(profile_torch=True)

        if self.config.utils.profile_nsys:
            self.logger.info("Enabled nsys profiler")
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
        if dist.is_initialized() and get_data_parallel_world_size() > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM,
                            group=get_data_parallel_group())
            loss /= dist.get_world_size()
        return loss.item()

    def train(self):
        """Train language model."""
        # load checkpoint and restore model and optimizer states
        last_step = load_checkpoint(
            self.config, self.model, self.optimizer, self.lr_scheduler
        )
        if last_step > -1:
            self.logger.info(
                f"Successfully loaded checkpoint: {self. config.trainer.model_path}"
            )
        else:
            self.logger.info("Training start from scratch")
            last_step = 0

        self.timer.start("total")

        # set model to training mode
        self.model.train()

        # start training
        step = last_step

        # training loop
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

            # update step
            step += 1

            # print and log training
            self.log_training(step, loss, grad_norm, param_norm, self.timer)

            # update profiler step
            if self.config.utils.profile_torch:
                self.context["profile"].step()  # pylint: disable=no-member

            # save checkpoint
            if self.control.do_checkpoint(step):
                save_checkpoint(
                    self.config, self.model, self.optimizer, self.lr_scheduler, step
                )

                # evaluation model
                if self.control.do_eval(step):
                    self.evaluate(step)
                    self.model.train()
                if self.control.do_eval_subtask(step):
                    self.evaluate_subtask(step)
                    self.model.train()

                # check exit condition
                if self.control.do_exit(step):
                    self.logger.info(
                        f"Training is stopped by exit interval: {self.config.operation.exit_interval}"
                    )
                    break

        # finish training
        # save checkpoint in case of the total train step is not divisible by the checkpoint save step
        if self.control.do_final_checkpoint(step, last_step):
            save_checkpoint(self.config, self.model,
                            self.optimizer, self.lr_scheduler, step)

        if self.config.trainer.do_test:
            self.test()

        self.logger.info(
            f"Total training time: {(self.timer.get('total') / 3600):.2f} hours"
        )
        self.logger.info("Finishing training")
        self._finialize_process()

    def train_step(self, step: int):
        """Trainer for given language model."""
        # forward pass
        self.timer.start(name="iter")
        total_loss = 0.0

        for i in range(0, self.config.trainer.gradient_accumulation_steps):
            is_last_accum_step = (
                i == self.config.trainer.gradient_accumulation_steps - 1)

            backward_sync_ctx = (
                self.model.no_sync
                if not is_last_accum_step and hasattr(self.model, "no_sync")
                else nullcontext
            )

            with backward_sync_ctx():
                with self.context["autocast"]:
                    loss = self.forward_step_func(
                        self.model, self.data_iterator["train"]
                    )
                    # loss is already a scalar (averaged over all valid tokens in micro-batch)
                    total_loss += loss
                    # For backprop: scale loss by gradient_accumulation_steps
                    # This ensures gradients are averaged, not summed
                    scaled_loss = loss / self.config.trainer.gradient_accumulation_steps

                # backward pass
                self.scaler.scale(scaled_loss).backward()

        # gradient clipping and norm computation
        self.scaler.unscale_(self.optimizer)

        grad_norm = 0.0
        if self.config.optim.clip_grad > 0.0:
            # Clipping enabled: always compute grad_norm
            grad_norm = clip_grad_norm_tp(
                self.model.parameters(), self.config.optim.clip_grad
            )
        elif self.control.do_grad_norm(step):
            # No clipping: compute grad_norm only when control says so
            grad_norm = clip_grad_norm_tp(self.model.parameters(), float("inf"))

        param_norm = 0.0
        if self.control.do_param_norm(step):
            for p in self.model.parameters():
                if p.data is not None:
                    param_norm += p.data.norm() ** 2
            param_norm = param_norm ** 0.5

        # update model
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # update lr_scheduler
        self.lr_scheduler.step()

        # record iteration time
        self.timer.stop(name="iter")

        # Return average loss and norms
        return (
            total_loss / self.config.trainer.gradient_accumulation_steps,
            grad_norm,
            param_norm,
        )

    def log_training(
        self, step: int, loss: Union[float, torch.Tensor], grad_norm: float, param_norm: float, timer: Timer
    ):
        # log metric
        if not self.control.do_log(step):
            return

        loss = self.average_loss(loss)

        timer_summary = timer.get_summary()
        iteration_time = timer_summary["iter"]

        # report memory usage
        if self.config.utils.report_memory_usage:
            mem_report = get_memory_usage(in_mib=True)
            message = ["Memory Usage::"]
            message += [f"{k}:, {v} MiB, " for k, v in mem_report.items()]
            self.logger.info(" ".join(message))
            # self.logger.info(f'Memory Usage:: Allocated: {mem_report["memory_allocated"]} MiB | Reserved: {mem_report["memory_reserved"]} MiB | Max Reserved: {mem_report["max_memory_reserved"]} MiB')
            log_metrics(mem_report, step)
            self.config.utils.report_memory_usage = False

        # log training progress
        self.logger.info(
            f"step: {step:6d}/{self.config.operation.train_steps}, train loss: {loss:.4f}, "
            f"grad norm: {grad_norm:.4f}, param norm: {param_norm:.4f}, "
            f"iteration time: {iteration_time:.3f}s, lr: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
        log_metric("lm_loss", loss, step)
        if self.control.do_grad_norm(step):
            log_metric("grad_norm", grad_norm, step)
        if self.control.do_param_norm(step):
            log_metric("param_norm", param_norm, step)
        log_metric("lr", self.optimizer.param_groups[0]["lr"], step)

    @torch.no_grad()
    def evaluate(self, global_step: int):
        """Evaluate language model using validation set"""

        # set model to evaluation mode
        self.model.eval()

        if "eval" in self.data_iterator and self.data_iterator["eval"] is not None:
            # evaluation with splitted dataset
            total_loss = 0
            total_accuracy = 0
            task_type = getattr(self.config.data, 'task_type', 'pretrain')

            for _ in range(self.config.operation.eval_samples):
                if task_type == "sft":
                    # SFT mode: compute both loss and accuracy on same batch
                    loss, acc = self._eval_step_sft(self.data_iterator["eval"])
                    total_loss += loss
                    total_accuracy += acc
                else:
                    # Pretrain mode: use forward_step_func (original behavior)
                    with self.context["autocast"]:
                        loss = self.forward_step_func(
                            self.model, self.data_iterator["eval"]
                        ).mean()
                        total_loss += loss.item()

            avg_loss = total_loss / self.config.operation.eval_samples

            # print loss and perplexity
            if task_type == "sft":
                avg_accuracy = total_accuracy / self.config.operation.eval_samples
                self.logger.info(
                    f"{'-'* 18} eval loss: {avg_loss:.4f}, ppl: {math.exp(avg_loss):.4f}, "
                    f"accuracy: {avg_accuracy:.4f} {'-'*18}"
                )
            else:
                self.logger.info(
                    f"{'-'* 18} eval loss: {avg_loss:.4f}, ppl: {math.exp(avg_loss):.4f} {'-'*18}"
                )

            if is_first_rank():
                log_metric("eval_loss", avg_loss, global_step)
                log_metric("eval_ppl", math.exp(avg_loss), global_step)
                if task_type == "sft":
                    log_metric("eval_accuracy", avg_accuracy, global_step)
        else:
            self.logger.warning(
                "Evaluation is requested, but there is no evaluation splits."
            )

    @torch.no_grad()
    def _eval_step_sft(self, data_iterator) -> tuple:
        """Evaluate single batch for SFT mode.

        Returns both loss and accuracy computed on the same batch.

        Returns:
            tuple: (loss_value, accuracy_value)
        """
        # Get batch
        input_ids, labels = get_batch(data_iterator)

        # Get device from model (handle DDP/FSDP wrapping)
        model_for_forward = self.model.module if hasattr(self.model, 'module') else self.model
        device = next(model_for_forward.parameters()).device

        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with self.context["autocast"]:
            # Single forward pass to get logits
            logits = self.model(input_ids, labels=None)

            # Get loss mask
            _, _, loss_mask = model_for_forward.get_masks_and_position_ids(input_ids, labels)

            # Reuse model's loss computation logic (handles both TP=1 and TP>1)
            padding_start_idx = getattr(model_for_forward, 'padding_start_idx', None)
            fp16_lm_cross_entropy = getattr(model_for_forward, 'fp16_lm_cross_entropy', False)

            loss = model_for_forward.compute_loss_from_logits(
                logits=logits,
                labels=labels,
                loss_mask=loss_mask,
                fp16_lm_cross_entropy=fp16_lm_cross_entropy,
                padding_start_idx=padding_start_idx
            )

            # Compute accuracy
            acc = compute_token_accuracy(logits, labels, loss_mask)

        return loss.item(), acc

    @torch.no_grad()
    def evaluate_subtask(self, global_step: int):
        """Evaluate language model using subtasks"""

        # set model to evaluation mode
        self.model.eval()

        if self.config.data.eval_datasets:
            eval_metrics = {}
            for evaluator in self.evaluators:
                task_name = evaluator.task_name
                score = evaluator.process(self.model)
                eval_metrics.update({task_name: score})

            # log metric
            if is_first_rank():
                for task_name, metrics in eval_metrics.items():
                    log_metric(
                        f"{task_name}/{metrics['metric']}",
                        metrics["score"],
                        global_step,
                    )
                self.logger.log_metrics(eval_metrics)
        else:
            self.logger.warning(
                "Subtask evaluation is requested, but no subtask is specified"
            )

    @torch.no_grad()
    def test(self):
        """Test trained language model"""

        # set model to evaluation mode
        self.model.eval()

        self.logger.info("Testing model before finalization..")

        if self.data_iterator.get("test"):
            # test with splited test set
            for step in range(self.config.operation.test_samples):

                # forward pass
                loss = self.forward_step_func(
                    self.model, self.data_iterator["test"]
                ).mean()
                # loss = loss_func(output_tensor, loss_mask)

                # update step
                step += 1

            # print loss and perplexity
            self.logger.info(f"loss: {loss}, ppl: {torch.exp(loss)}")

        if self.config.data.test_datasets:
            testers = get_evaluators(
                self.config.data.test_datasets,
                self.config.trainer.test_batch_size,
                self.config.operation.test_samples,
            )
            for tester in testers:
                tester.process(self.model)
