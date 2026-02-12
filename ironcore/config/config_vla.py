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
class VisionConfig(BaseConfig):
    """Vision encoder configuration for VLA models."""

    encoder_type: str = field(
        default="siglip",
        metadata={"help": "Vision encoder type: siglip, clip, or vit"},
    )
    model_name: str = field(
        default="google/siglip-so400m-patch14-384",
        metadata={"help": "HuggingFace model name for vision encoder"},
    )
    image_size: int = field(
        default=384,
        metadata={"help": "Input image size"},
    )
    patch_size: int = field(
        default=14,
        metadata={"help": "Vision patch size"},
    )
    hidden_size: int = field(
        default=1152,
        metadata={"help": "Vision encoder hidden dimension"},
    )
    num_hidden_layers: int = field(
        default=27,
        metadata={"help": "Number of vision encoder layers"},
    )
    num_attention_heads: int = field(
        default=16,
        metadata={"help": "Number of attention heads in vision encoder"},
    )
    intermediate_size: int = field(
        default=4304,
        metadata={"help": "Vision encoder FFN intermediate size"},
    )
    freeze_vision: bool = field(
        default=True,
        metadata={"help": "Whether to freeze vision encoder weights"},
    )
    layer_norm_eps: float = field(
        default=1e-6,
        metadata={"help": "Layer normalization epsilon for vision encoder"},
    )
    # Device placement
    device: str = field(
        default="auto",
        metadata={"help": "Device for vision encoder: 'auto', 'cuda:0', 'cuda:1', 'cuda:2' (e-GPU), 'cpu'"},
    )
    prefer_cpu_with_avx512: bool = field(
        default=False,
        metadata={"help": "Prefer CPU for vision when AVX-512 is available"},
    )
    offload_memory_mb: int = field(
        default=0,
        metadata={"help": "CPU offload threshold in MB (0=disabled)"},
    )


@dataclass
class ProjectorConfig(BaseConfig):
    """Vision-Language projector configuration."""

    projector_type: str = field(
        default="mlp",
        metadata={"help": "Projector type: mlp or linear"},
    )
    hidden_size: int = field(
        default=4096,
        metadata={"help": "Projector hidden dimension"},
    )
    num_layers: int = field(
        default=2,
        metadata={"help": "Number of projector layers"},
    )
    activation: str = field(
        default="gelu",
        metadata={"help": "Activation function: gelu or relu"},
    )


@dataclass
class FusionConfig(BaseConfig):
    """Vision-Language fusion configuration."""

    fusion_type: str = field(
        default="gated_cross_attention",
        metadata={"help": "Fusion type: gated_cross_attention, qformer, or concat"},
    )
    num_layers: int = field(
        default=1,
        metadata={"help": "Number of cross-attention layers"},
    )
    # For Q-Former
    num_query_tokens: int = field(
        default=32,
        metadata={"help": "Number of learnable query tokens for Q-Former"},
    )
    # Cross-attention settings
    cross_attn_every_n_layers: int = field(
        default=1,
        metadata={"help": "Apply cross-attention every N language layers"},
    )


@dataclass
class ActionConfig(BaseConfig):
    """Action head configuration for VLA models."""

    action_dim: int = field(
        default=7,
        metadata={"help": "Action dimension (e.g., 7 for xyz + quaternion)"},
    )
    hidden_size: int = field(
        default=1024,
        metadata={"help": "Action head hidden dimension"},
    )
    num_layers: int = field(
        default=3,
        metadata={"help": "Number of action head MLP layers"},
    )
    prediction_horizon: int = field(
        default=1,
        metadata={"help": "Number of future actions to predict"},
    )
    loss_type: str = field(
        default="mse",
        metadata={"help": "Action loss type: mse, l1, or smooth_l1"},
    )
    action_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for action loss in total loss"},
    )
    use_normalizer: bool = field(
        default=True,
        metadata={"help": "Whether to normalize actions"},
    )


@dataclass
class VLAConfig(BaseConfig):
    """VLA (Vision-Language-Action) model configuration."""

    # Sub-configs
    vision: VisionConfig = field(default_factory=VisionConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    action: ActionConfig = field(default_factory=ActionConfig)

    # VLA-specific settings
    image_token_id: int = field(
        default=-200,
        metadata={"help": "Token ID for <image> special token"},
    )
    action_token_id: int = field(
        default=-201,
        metadata={"help": "Token ID for <action> special token"},
    )
    num_image_tokens: int = field(
        default=729,  # (384/14)^2 for 384x384 images with patch_size=14
        metadata={"help": "Number of vision tokens per image"},
    )

    # Loss weights
    vision_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for vision-language alignment loss"},
    )
    language_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for language modeling loss"},
    )


__all__ = [
    "VisionConfig",
    "ProjectorConfig",
    "ActionConfig",
    "VLAConfig",
]
