"""Alignment configuration for DPO and other alignment methods."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

from .config import BaseConfig


@dataclass
class AlignmentConfig(BaseConfig):
    """Configuration for alignment training (DPO, PPO, etc.)."""

    # DPO specific parameters
    dpo_beta: float = 0.5
    dpo_label_smoothing: float = 0.0

    # Optimization flags
    # concat_forward_passes=True batches chosen+rejected into a single forward pass
    # for both policy and reference models (2 passes total).
    # When False, uses 4 separate passes (chosen policy, rejected policy,
    # chosen ref, rejected ref) which is ~2× slower but useful for debugging
    # or when memory is extremely constrained.
    concat_forward_passes: bool = True

    # Metrics computation interval (0 = compute every step)
    # Set to higher value (e.g., 10-50) to reduce overhead
    metrics_interval: int = 0

    def __post_init__(self):
        """Validate alignment configuration parameters."""
        if self.dpo_beta <= 0:
            raise ValueError(f"dpo_beta must be positive, got {self.dpo_beta}")
        if not (0.0 <= self.dpo_label_smoothing < 1.0):
            raise ValueError(
                f"dpo_label_smoothing must be in [0, 1), got {self.dpo_label_smoothing}"
            )
        if self.metrics_interval < 0:
            raise ValueError(f"metrics_interval must be >= 0, got {self.metrics_interval}")

    @classmethod
    def from_yaml(cls, filename: Union[str, Path]) -> "AlignmentConfig":
        """Load alignment config from YAML file."""
        with open(filename) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


def get_alignment_config(config_name: str = "dpo_default") -> AlignmentConfig:
    """
    Get alignment configuration by name.

    Args:
        config_name: Name of alignment config (e.g., 'dpo_default')

    Returns:
        AlignmentConfig object
    """
    config_path = Path("configs/alignment") / f"{config_name}.yaml"
    if config_path.exists():
        return AlignmentConfig.from_yaml(config_path)
    else:
        # Return default config
        return AlignmentConfig()
