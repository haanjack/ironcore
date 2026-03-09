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

    # System prompt prepended to all prompts
    system_prompt: str | None = None


@dataclass
class RewardConfig(BaseConfig):
    """Configuration for GRPO reward computation."""

    type: str = "math"  # "math" | "code" | "api" | "local_endpoint" | "local_inference" | "format"

    # Worker configuration
    num_workers: int = 4
    timeout: int = 30

    # API reward configuration (when type="api")
    api_provider: str = "openai"  # "openai" | "anthropic" | "google" | "zhipu"
    api_model: str | None = None
    api_key_env: str | None = None
    api_endpoint: str | None = None
    prompt_template: str = "default"  # "default" | "math" | "code" | "reasoning"
    custom_prompt: str | None = None
    max_retries: int = 3
    cache_size: int = 10000
    rate_limit_delay: float = 0.1

    # Local endpoint configuration (when type="local_endpoint")
    local_endpoint: str = "http://localhost:8000/v1"

    # Local inference configuration (when type="local_inference")
    local_model_path: str | None = None
    local_device: str = "cuda:0"
    local_dtype: str = "bfloat16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Format reward configuration (when type="format")
    required_tags: list[str] | None = None
    format_penalty: float = -0.1

    # Keyword reward configuration (when type="keyword")
    keyword: str = "ironcore"
    keyword_case_sensitive: bool = False


@dataclass
class AlignmentConfig(BaseConfig):
    """Configuration for alignment training (DPO, GRPO, etc.)."""

    # Alignment method
    method: str = "dpo"  # "dpo" | "grpo"

    # DPO specific parameters
    dpo_beta: float = 0.5
    dpo_label_smoothing: float = 0.0

    # GRPO specific parameters
    grpo_group_size: int = 4  # G completions per prompt
    grpo_beta: float = 0.1  # KL penalty coefficient
    grpo_eps: float = 1e-8  # Advantage normalization epsilon
    grpo_num_epochs: int = 1  # Gradient steps per rollout batch (>1 = offline/multi-epoch)
    grpo_clip_eps: float = 0.2  # PPO-style IS ratio clip range (0.0 = no clipping)

    # GRPO generation and reward config
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Optimization flags
    concat_forward_passes: bool = True

    # Metrics computation interval (0 = compute every step)
    metrics_interval: int = 0

    def __post_init__(self):
        """Validate alignment configuration parameters."""
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
            valid_reward_types = ("math", "code", "api", "local_endpoint", "local_inference", "format", "keyword", "soft_keyword")
            if self.reward.type not in valid_reward_types:
                raise ValueError(
                    f"reward.type must be one of {valid_reward_types}, got {self.reward.type}"
                )

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
        if "reward" in config_dict and isinstance(config_dict["reward"], dict):
            config_dict["reward"] = RewardConfig(**config_dict["reward"])

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
