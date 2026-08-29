# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import math

from torch import optim
from torch.optim.lr_scheduler import LRScheduler

from ironcore.global_vars import get_logger


class LinearDecayLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
    ):
        """
        Initializes the linear warmup + linear decay LR scheduler.

        Args:
            optimizer: The optimizer for the scheduler.
            warmup_steps: The number of linear warmup steps.
            total_steps: The total number of training steps (decay reaches 0 here).

        Returns:
            None
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch=-1)
        self._step_count = 0

    def get_lr(self):
        if self.warmup_steps > 0 and self._step_count <= self.warmup_steps:
            warmup_factor = self._step_count / self.warmup_steps
            lr = [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            decay_steps = max(self.total_steps - self.warmup_steps, 1)
            decay_factor = max(0.0, 1 - (self._step_count - self.warmup_steps) / decay_steps)
            lr = [base_lr * decay_factor for base_lr in self.base_lrs]
        return lr

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        self._step_count = self.last_epoch + 1
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
            param_group["lr"] = lr
        # Store _last_lr for get_last_lr() compatibility
        self._last_lr = lrs


class CosineAnnealingLR(LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        annealing_steps: int,
        total_steps: int,
        max_lr: float = 1e-5,
        min_lr: float = 1e-8,
        last_epoch: int = -1,
    ):
        # max_lr is validation-only here: actual peak LR values come from the
        # optimizer's own base_lrs (set when the optimizer was constructed from
        # config.optim.max_lr), not from this parameter.
        if max_lr < min_lr:
            raise ValueError("max_lr should be larger than min_lr")

        self.warmup_steps = warmup_steps
        self.annealing_steps = annealing_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

        self._step_count = 0

    def get_lr(self):
        if self.warmup_steps > 0 and self._step_count <= self.warmup_steps:
            warmup_factor = self._step_count / self.warmup_steps
            lr = [base_lr * warmup_factor for base_lr in self.base_lrs]
        elif self._step_count >= self.annealing_steps + self.warmup_steps:
            lr = [self.min_lr for _ in self.base_lrs]
        else:
            # cosine annealing
            cos_inner = math.pi * (self._step_count - self.warmup_steps) / self.annealing_steps
            cos_out = (1 + math.cos(cos_inner)) / 2
            lr = [self.min_lr + (base_lr - self.min_lr) * cos_out for base_lr in self.base_lrs]
        return lr

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        self._step_count = self.last_epoch + 1
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
            param_group["lr"] = lr
        # Store _last_lr for get_last_lr() compatibility
        self._last_lr = lrs


def get_lr_scheduler(config, optimizer):
    """Returns the learning rate scheduler."""

    logger = get_logger()

    if config.optim.annealing_steps == 0:
        config.optim.annealing_steps = config.operation.train_steps

    # lr scheduler arguments
    lr_scheduler_kwargs = {
        "max_lr": config.optim.max_lr,
        "min_lr": config.optim.min_lr,
        "warmup_steps": config.optim.warmup_steps,
        "annealing_steps": config.optim.annealing_steps,
        "total_steps": config.operation.train_steps,
    }

    if config.optim.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, **lr_scheduler_kwargs)
    elif config.optim.lr_scheduler == "linear":
        scheduler = LinearDecayLRScheduler(
            optimizer,
            warmup_steps=config.optim.warmup_steps,
            total_steps=config.operation.train_steps,
        )
    else:
        message = f"lr_scheduler {config.optim.lr_scheduler} is not implemented"
        logger.error(message)
        raise NotImplementedError(message)

    return scheduler
