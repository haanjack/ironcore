# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from torch.utils import tensorboard


class LevelFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, level_formats=None):
        super().__init__(fmt, datefmt)
        self.level_formats = level_formats or {}
        self.default_format = fmt

    def format(self, record):
        log_fmt = self.level_formats.get(record.levelno, self.default_format)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class IronCoreLogger:
    """trainer's logger for global control of logging"""

    def __init__(self, config, name=__name__, level=logging.INFO):
        """init logger"""
        super().__init__()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.rank = config.parallel.rank

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            level_formats = {
                logging.INFO: "[%(asctime)s] %(message)s",
                logging.WARNING: "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                logging.ERROR: "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            }
            formatter = LevelFormatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                level_formats=level_formats,
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def _log(self, level, message):
        """log message"""

        if self.rank > 0:
            return

        # collect filename and line number
        frame = sys._getframe(2)
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno

        extra = {"filename": filename, "lineno": lineno}
        self.logger.log(level, message, {"extra": extra})

    def set_log_level(self, level):
        """set log level"""
        self.logger.setLevel(level)

    def info(self, message):
        """log info message"""
        self._log(logging.INFO, message)

    def warning(self, message):  # pylintg: disable=invalid-name
        """log warning message"""
        self._log(logging.WARNING, message)

    def error(self, message):  # pylint: disable=arguments-differ, arguments-renamed
        """log error message"""
        self._log(logging.ERROR, message)

    def debug(self, message):
        """log debug message"""
        self._log(logging.DEBUG, message)

    def log_metrics(self, metrics: Dict[str, Any], msg: str = "evaluation score"):
        """log metrics in table"""
        df = pd.DataFrame(metrics).transpose()
        df = df.infer_objects(copy=False)
        df = df.round(2)
        self._log(logging.INFO, f"{msg}:\n{df.to_markdown()}\n")


class TensorboardLogger:
    def __init__(self, config):
        self.tensorboard = tensorboard.SummaryWriter(
            log_dir=f"{config.utils.tensorboard_dir}/{config.trainer.model_name}"
        )
        self.rank = config.parallel.rank

    def add_scalar(self, name, value, step):
        if self.rank > 0:
            return
        try:
            self.tensorboard.add_scalar(name, value, step)
        except Exception as e:  # pylint: disable=broad-except
            # Silently fail to avoid disrupting training
            pass

    def add_histogram(self, name, values, step):
        if self.rank > 0:
            return
        try:
            self.tensorboard.add_histogram(name, values, step)
        except Exception as e:  # pylint: disable=broad-except
            # Silently fail to avoid disrupting training
            pass

    def close(self):
        try:
            self.tensorboard.close()
        except Exception:  # pylint: disable=broad-except
            # Silently fail during cleanup
            pass


class MLFlowLogger:
    def __init__(self, config):
        if config.parallel.rank > 0:
            return

        try:
            import mlflow
        except ImportError:
            raise ImportError("MLFlowLogger requires mlflow package")
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(config.utils.mlflow_tracking_uri)

        # check if hash run id exists
        exp_info_filename = "exp_info.json"
        exp_file = (
            Path(config.utils.tensorboard_dir)
            / config.trainer.model_name
            / exp_info_filename
        )
        if exp_file.exists():
            self.mlflow.get_experiment_by_name(
                config.utils.mlflow_experiment_name)
            self.mlflow.set_experiment(config.utils.mlflow_experiment_name)

            # load run_id and start run with it
            with open(exp_file, "r", encoding="utf-8") as f:
                exp_info = json.load(f)
            exp_info["run_count"] += 1
            self.mlflow.start_run(
                run_name=f'{config.trainer.model_name}-{exp_info["run_count"]}'
            )
            self.mlflow.set_tag("run_count", exp_info["run_count"])
            with open(exp_file, "w", encoding="utf-8") as f:
                json.dump(exp_info, f)
        else:
            if (
                self.mlflow.get_experiment_by_name(
                    config.utils.mlflow_experiment_name)
                is None
            ):
                self.mlflow.create_experiment(
                    config.utils.mlflow_experiment_name)
            active_run = self.mlflow.start_run(
                run_name=config.trainer.model_name)

            # start run and save run_id
            exp_info = {"run_id": active_run.info.run_id, "run_count": 0}
            with open(exp_file, "w", encoding="utf-8") as f:
                json.dump(exp_info, f)
            self.mlflow.set_tag("run_count", 1)

    def set_tags(self, tags):
        if self.mlflow:
            self.mlflow.set_tags(tags)

    def set_params(self, params):
        if self.mlflow:
            self.mlflow.set_tags(params)

    def log_params(self, params):
        if self.mlflow:
            self.mlflow.log_params(params)

    def log_metric(self, key, value, step):
        if self.mlflow:
            self.mlflow.log_metric(key, value, step)

    def log_artifact(self, path):
        if self.mlflow:
            self.mlflow.log_artifact(path)

    def log_artifacts(self, path):
        if self.mlflow:
            self.mlflow.log_artifacts(path)

    def close(self):
        if self.mlflow:
            self.mlflow.end_run()
