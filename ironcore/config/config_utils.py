# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from .config import BaseConfig


@dataclass
class ProfilerConfig(BaseConfig):
    """Configuration for profiling and performance monitoring."""

    gpu_profiler: bool = field(default=False, metadata={"help": "Enable hardware-level profiling"})
    torch_profiler: bool = field(default=False, metadata={"help": "Enable torch profiler"})

    start: int = field(default=10, metadata={"help": "Profile start step"})
    end: int = field(default=12, metadata={"help": "Profile end step"})
    ranks: list[int] = field(
        default_factory=lambda: [0], metadata={"help": "Global ranks to profile"}
    )
    stop_at_end: bool = field(default=False, metadata={"help": "Stop training on profile end"})

    name: str = field(default="profile", metadata={"help": "Prefix for profile output files"})
    output_dir: str = field(
        default="./logs/profile/", metadata={"help": "Directory to save profile traces"}
    )

    wait_steps: int = field(default=1, metadata={"help": "Steps to wait before starting"})
    warmup_steps: int = field(default=1, metadata={"help": "Warmup steps before active capture"})
    active_steps: int = field(
        default=1, metadata={"help": "Number of steps to capture active data"}
    )
    repeat: int = field(default=1, metadata={"help": "Number of times to repeat the capture cycle"})

    oom_monitor: bool = field(
        default=False, metadata={"help": "Enable automatic profiling on high memory usage"}
    )
    oom_threshold: float = field(default=95.0, metadata={"help": "Memory threshold percentage"})


@dataclass
class UtilsConfig(BaseConfig):
    """config for trainer's utilities"""

    log_level: str = field(default="INFO", metadata={"help": "log level"})

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
    wandb_project: str | None = field(default=None, metadata={"help": "wandb project name"})
    wandb_name: str | None = field(default=None, metadata={"help": "wandb run name"})
    wandb_entity: str | None = field(default=None, metadata={"help": "wandb entity/username"})
