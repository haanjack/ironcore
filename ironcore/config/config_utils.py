# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from dataclasses import dataclass, field

from .config import BaseConfig


@dataclass
class UtilsConfig(BaseConfig):
    """config for trainer's utilities"""

    log_level: str = field(default="INFO", metadata={"help": "log level"})

    profile_nsys: bool = field(
        default=False,
        metadata={
            "help": "Enable nsys profiling. When profile use this command: "
            "nsys profile -s none -t nvtx,cuda,cudnn,cublas,osrt -o <path/to/output_file> --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
        },
    )
    profile_torch: bool = field(
        default=False,
        metadata={
            "help": "Enable torch profiler. When profile use this command: torch.profiler.profile"
        },
    )
    profile_step_start: int | None = field(default=10, metadata={"help": "nsys profile start step"})
    profile_step_end: int | None = field(default=12, metadata={"help": "nsys profile end step"})
    profile_ranks: list[int] | None = field(
        default_factory=lambda: [0], metadata={"help": "global ranks nsys profile"}
    )
    stop_on_profile_end: bool = field(
        default=False, metadata={"help": "stop training on profile end"}
    )

    deterministic: bool = field(default=False, metadata={"help": "Enable deterministic mode"})

    report_memory_usage: bool = field(
        default=True, metadata={"help": "Enable memory report at the first log step"}
    )

    # logger
    tensorboard_dir: str | None = field(default=None, metadata={"help": "tensorboard path"})
    mlflow_tracking_uri: str | None = field(default=None, metadata={"help": "mlflow tracking uri"})
    mlflow_experiment_name: str | None = field(
        default=None, metadata={"help": "mlflow experiment name"}
    )
