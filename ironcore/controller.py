# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Helper for training-time cadence and actions

from typing import Optional, Literal

from ironcore.config import MainConfig


class TrainingControl:
    """Encapsulates training-time cadence decisions.

    This keeps cadence policy out of config objects and centralizes logic
    for logging, checkpointing, and selective norm computations.
    """

    def __init__(self, config: MainConfig):
        self.config = config
        self.trainer = config.trainer
        self.operation = config.operation
        self.utils = config.utils

    @staticmethod
    def _cadence_now(every: Optional[int], step: int) -> bool:
        return isinstance(every, int) and every > 0 and step > 0 and (step % every == 0)

    @staticmethod
    def _cadence_next(every: Optional[int], step: int) -> bool:
        return isinstance(every, int) and every > 0 and ((step + 1) % every == 0)

    def _cadence_flag(self, mode: Optional[Literal['log', 'checkpoint']], step: int) -> bool:
        if mode == 'log':
            return self._cadence_next(self.trainer.log_interval, step)
        if mode == 'checkpoint':
            return self._cadence_next(self.trainer.save_checkpoint_steps, step)
        return False

    # Public API (do_*)
    def do_log(self, step: int) -> bool:
        return self._cadence_now(self.trainer.log_interval, step)

    def do_checkpoint(self, step: int) -> bool:
        return self._cadence_now(self.trainer.save_checkpoint_steps, step)

    def do_grad_norm(self, step: int) -> bool:
        return self._cadence_flag(self.trainer.grad_norm_log_interval, step)

    def do_param_norm(self, step: int) -> bool:
        return self._cadence_flag(self.trainer.param_norm_log_interval, step)

    def do_eval(self, step: int) -> bool:
        """True when evaluation should run."""
        return (
            self.trainer.do_eval
            and self._cadence_now(self.operation.eval_interval, step)
        )

    def do_eval_subtask(self, step: int) -> bool:
        """True when subtask evaluation should run."""
        return (
            self.trainer.do_eval_subtask
            and self._cadence_now(self.operation.eval_interval, step)
        )

    def do_exit(self, step: int) -> bool:
        """True when training should exit early."""
        return self._cadence_now(self.operation.exit_interval, step)

    def do_final_checkpoint(self, step: int, last_step: int) -> bool:
        """True when final checkpoint should be saved (if not already on cadence)."""
        return (
            not self.do_checkpoint(step)
            and step - last_step > 1
            and not self.utils.profile_nsys
            and not self.utils.profile_torch
        )
