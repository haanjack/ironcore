# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from abc import ABC
from typing import Dict

from ironcore.logger import MLFlowLogger, IronCoreLogger, TensorboardLogger
from ironcore.tokenizer import Tokenizer, build_tokenizer
from ironcore.utils import Timer

GLOBAL_STATES = None


class GlobalStates:

    def __init__(self, config):
        self.config = config

        # logger initialize
        self.logger = IronCoreLogger(config=config, level=config.utils.log_level)
        if config.utils.tensorboard_dir:
            self.tensorboard_logger = TensorboardLogger(config)
        if config.utils.mlflow_experiment_name:
            self.mlflow_logger = MLFlowLogger(config)

        self.timer = Timer()

        self._tokenizer = build_tokenizer(config)
        self.config.model.padded_vocab_size = self._tokenizer.padded_vocab_size

    def get_config(self):
        """returns config"""
        return self.config

    def get_logger(self):
        """returns logger"""
        return self.logger

    def get_tokenizer(self) -> Tokenizer:
        """returns tokenizer"""
        return self._tokenizer

    def cleanup(self):
        """Clean up resources like closing loggers."""
        if hasattr(self, "tensorboard_logger"):
            self.tensorboard_logger.close()
        if hasattr(self, "mlflow_logger"):
            self.mlflow_logger.close()


def set_global_states(config):
    """set global states"""
    global GLOBAL_STATES  # pylint: disable=global-statement
    assert GLOBAL_STATES is None, "global states should not be initialized"
    GLOBAL_STATES = GlobalStates(config)


def get_global_states():
    """get global states"""
    assert GLOBAL_STATES is not None, "global states should not be None"
    return GLOBAL_STATES


def get_config():
    """get config"""
    assert GLOBAL_STATES is not None, "global states should not be None"
    return GLOBAL_STATES.get_config()


def get_logger():
    """get logger"""
    assert GLOBAL_STATES is not None, "global states should not be None"
    return GLOBAL_STATES.get_logger()


def get_tokenizer() -> Tokenizer:
    assert GLOBAL_STATES is not None, "global states should not be None"
    return GLOBAL_STATES.get_tokenizer()


def log_metric(name: str, value: float, step: int):
    assert GLOBAL_STATES is not None, "global states should not be None"
    tensorboard_logger = getattr(GLOBAL_STATES, "tensorboard_logger", None)
    mlflow_logger = getattr(GLOBAL_STATES, "mlflow_logger", None)

    if tensorboard_logger:
        tensorboard_logger.add_scalar(name, value, step)
    if mlflow_logger:
        mlflow_logger.log_metric(name, value, step)


def log_metrics(metrics: Dict[str, float], step: int):
    assert GLOBAL_STATES is not None, "global states should not be None"
    tensorboard_logger = getattr(GLOBAL_STATES, "tensorboard_logger", None)
    mlflow_logger = getattr(GLOBAL_STATES, "mlflow_logger", None)

    for k, v in metrics.items():
        if tensorboard_logger:
            tensorboard_logger.add_scalar(k, v, step)
        if mlflow_logger:
            mlflow_logger.log_metric(k, v, step)


def log_histogram(name: str, values, step: int):
    """Log histogram to tensorboard."""
    assert GLOBAL_STATES is not None, "global states should not be None"
    tensorboard_logger = getattr(GLOBAL_STATES, "tensorboard_logger", None)

    if tensorboard_logger:
        tensorboard_logger.add_histogram(name, values, step)


def global_states_cleanup():
    """Clean up global resources like closing loggers."""
    if GLOBAL_STATES is not None:
        GLOBAL_STATES.cleanup()


def get_timer():
    assert GLOBAL_STATES is not None, "global states should be initialized"
    return GLOBAL_STATES.timer
