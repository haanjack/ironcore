# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Alignment configuration for DPO, GRPO, and other alignment methods."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import yaml

from .config import BaseConfig


@dataclass
class GenerationConfig(BaseConfig):
    """Configuration for GRPO generation."""

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True

    # Chat template settings
    use_chat_template: bool = False
    system_prompt: str | None = None


@dataclass
class RewardFunctionEntry(BaseConfig):
    """Configuration for a single reward function in RewardManager."""

    name: str = "default"
    type: str = "rule_template"  # "rule_template" | "reward_model" | "math" | "code" | "api" | ...
    weight: float = 1.0
    rule_template: str | None = None  # path to YAML (e.g. "configs/rewards/math_gsm8k.yaml")

    # Reward model backend (when type="reward_model")
    rm_backend: str = "local_endpoint"  # "local_endpoint" | "api" | "local_inference"

    # Type-specific params (reuse same flat fields)
    api_provider: str = "openai"
    api_model: str | None = None
    local_endpoint: str = "http://localhost:8000/v1"
    local_model_path: str | None = None
    local_device: str = "cuda:0"
    local_dtype: str = "bfloat16"
    keyword: str = ""
    format_weight: float = 0.2  # for composite_math (fraction of reward from format check)


@dataclass
class RewardManagerConfig(BaseConfig):
    """Configuration for the RewardManager orchestrator."""

    functions: list[RewardFunctionEntry] = field(default_factory=list)
    num_workers: int = 4
    timeout: int = 30

    def __post_init__(self):
        """Convert function dicts to RewardFunctionEntry instances."""
        converted = []
        for entry in self.functions:
            if isinstance(entry, dict):
                converted.append(RewardFunctionEntry(**entry))
            else:
                converted.append(entry)
        self.functions = converted


@dataclass
class AlignmentConfig(BaseConfig):
    """Configuration for alignment training (DPO, GRPO, etc.)."""

    # Alignment method
    method: str = "dpo"  # "dpo" | "grpo"

    # DPO specific parameters
    dpo_beta: float = 0.5
    dpo_label_smoothing: float = 0.0

    # GRPO specific parameters
    # grpo_group_size: total completions per prompt (world-size agnostic, like train_batch_size)
    # grpo_rollout_micro_group_size: completions generated in parallel per prompt per GPU
    #   (hardware knob, like micro_batch_size). chunks derived: group_size / micro_group_size
    grpo_group_size: int = 4  # Total completions per prompt
    grpo_rollout_micro_group_size: int = 1  # Per-GPU parallel completions per prompt
    grpo_use_paged_rollout: bool = False  # Use block-based paged KV cache for rollouts
    grpo_beta: float = 0.1  # KL penalty coefficient
    grpo_eps: float = 1e-8  # Advantage normalization epsilon
    grpo_num_epochs: int = 1  # Gradient steps per rollout batch (>1 = offline/multi-epoch)
    grpo_clip_eps: float = 0.2  # PPO-style IS ratio clip range (0.0 = no clipping)

    # GRPO generation and reward config
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reward_manager: RewardManagerConfig | None = None  # Required for GRPO

    # Optimization flags
    concat_forward_passes: bool = True

    # GRPO reference model offloading
    offload_ref_model: bool = False  # Move reference model to CPU between forward passes

    # Metrics computation interval (0 = compute every step)
    metrics_interval: int = 0

    def __post_init__(self):
        """Validate alignment configuration parameters."""
        # Convert reward_manager dict to RewardManagerConfig if needed
        if isinstance(self.reward_manager, dict):
            self.reward_manager = RewardManagerConfig(**self.reward_manager)

        if self.method not in ("dpo", "grpo"):
            raise ValueError(f"method must be 'dpo' or 'grpo', got {self.method}")

        # DPO validation
        if self.method == "dpo":
            if self.dpo_beta <= 0:
                raise ValueError(f"dpo_beta must be positive, got {self.dpo_beta}")
            if not (0.0 <= self.dpo_label_smoothing < 1.0):
                raise ValueError(
                    f"dpo_label_smoothing must be in [0, 1), got {self.dpo_label_smoothing}"
                )

        # GRPO validation
        if self.method == "grpo":
            if self.grpo_group_size < 2:
                raise ValueError(f"grpo_group_size must be >= 2, got {self.grpo_group_size}")
            if self.grpo_beta < 0:
                raise ValueError(f"grpo_beta must be >= 0, got {self.grpo_beta}")
            if self.grpo_num_epochs < 1:
                raise ValueError(f"grpo_num_epochs must be >= 1, got {self.grpo_num_epochs}")
            if self.grpo_clip_eps < 0:
                raise ValueError(f"grpo_clip_eps must be >= 0, got {self.grpo_clip_eps}")
            if self.grpo_rollout_micro_group_size < 1:
                raise ValueError(
                    f"grpo_rollout_micro_group_size must be >= 1, got {self.grpo_rollout_micro_group_size}"
                )
            if self.grpo_group_size % self.grpo_rollout_micro_group_size != 0:
                raise ValueError(
                    f"grpo_group_size ({self.grpo_group_size}) must be divisible by "
                    f"grpo_rollout_micro_group_size ({self.grpo_rollout_micro_group_size})"
                )
            if self.reward_manager is None:
                raise ValueError("GRPO requires reward_manager configuration")

        if self.metrics_interval < 0:
            raise ValueError(f"metrics_interval must be >= 0, got {self.metrics_interval}")

    @classmethod
    def from_yaml(cls, filename: Union[str, Path]) -> "AlignmentConfig":
        """Load alignment config from YAML file."""
        with open(filename, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Handle nested configs
        if "generation" in config_dict and isinstance(config_dict["generation"], dict):
            config_dict["generation"] = GenerationConfig(**config_dict["generation"])
        if "reward_manager" in config_dict and isinstance(config_dict["reward_manager"], dict):
            config_dict["reward_manager"] = RewardManagerConfig(**config_dict["reward_manager"])

        return cls(**config_dict)


def get_alignment_config(config_name: str = "dpo_default") -> AlignmentConfig:
    """Get alignment configuration by name.

    Args:
        config_name: Name of alignment config (e.g., 'dpo_default', 'grpo_default')

    Returns:
        AlignmentConfig object
    """
    config_path = Path("configs/alignment") / f"{config_name}.yaml"
    if config_path.exists():
        return AlignmentConfig.from_yaml(config_path)
    else:
        # Return default config
        return AlignmentConfig()
