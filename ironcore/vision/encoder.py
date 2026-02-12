# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import math

import torch
from torch import nn

from ironcore.config import MainConfig
from ironcore.config.config_vla import VisionConfig
from ironcore.layers.module import BaseModule


class VisionMultiHeadAttention(BaseModule):
    """Multi-head attention for vision encoder.

    Standard ViT-style attention with separate Q, K, V projections.
    """

    def __init__(self, config: MainConfig, vision_config: VisionConfig):
        super().__init__(config)

        self.num_heads = vision_config.num_attention_heads
        self.head_dim = vision_config.hidden_size // vision_config.num_attention_heads
        self.scale = self.head_dim**-0.5

        hidden_size = vision_config.hidden_size

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, num_patches, hidden_size]

        Returns:
            [batch, num_patches, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Output
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.proj(out)

        return out


class VisionMLP(BaseModule):
    """MLP for vision encoder."""

    def __init__(self, config: MainConfig, vision_config: VisionConfig):
        super().__init__(config)

        hidden_size = vision_config.hidden_size
        intermediate_size = vision_config.intermediate_size

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class VisionTransformerLayer(BaseModule):
    """Single ViT transformer layer."""

    def __init__(self, config: MainConfig, vision_config: VisionConfig):
        super().__init__(config)

        hidden_size = vision_config.hidden_size
        eps = vision_config.layer_norm_eps

        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attn = VisionMultiHeadAttention(config, vision_config)
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = VisionMLP(config, vision_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEmbeddings(BaseModule):
    """Vision embeddings: patch embeddings + position embeddings."""

    def __init__(self, config: MainConfig, vision_config: VisionConfig):
        super().__init__(config)

        self.image_size = vision_config.image_size
        self.patch_size = vision_config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        hidden_size = vision_config.hidden_size

        # Patch embedding via convolution
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )

        # Position embeddings (learnable)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )

        self._init_position_embeddings()

    def _init_position_embeddings(self):
        # Initialize with sinusoidal pattern for better convergence
        with torch.no_grad():
            position = torch.arange(self.num_patches).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, self.position_embedding.shape[-1], 2).float()
                * (-math.log(10000.0) / self.position_embedding.shape[-1])
            )
            self.position_embedding[:, :, 0::2] = torch.sin(position * div_term)
            self.position_embedding[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Convert images to patch embeddings.

        Args:
            pixel_values: [batch, C, H, W] preprocessed images

        Returns:
            [batch, num_patches, hidden_size] patch embeddings
        """
        # Patch embedding: [batch, C, H, W] -> [batch, hidden, h, w]
        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten spatial dimensions: [batch, hidden, h, w] -> [batch, hidden, num_patches]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # Add position embeddings
        embeddings = patch_embeds + self.position_embedding

        return embeddings


class VisionEncoder(BaseModule):
    """Vision encoder for VLA models.

    Loads pretrained weights from HuggingFace transformers (SigLIP/CLIP)
    and wraps in IronCore's BaseModule pattern. Supports frozen mode
    and device placement for hybrid training.
    """

    def __init__(self, config: MainConfig):
        super().__init__(config)

        self.vision_config = config.vla.vision
        self.freeze_vision = self.vision_config.freeze_vision

        # Device placement
        self.vision_device = self._resolve_device(self.vision_config.device)

        # Build encoder components
        self.embeddings = VisionEmbeddings(config, self.vision_config)
        self.layers = nn.ModuleList([
            VisionTransformerLayer(config, self.vision_config)
            for _ in range(self.vision_config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(
            self.vision_config.hidden_size,
            eps=self.vision_config.layer_norm_eps,
        )

        # Load pretrained weights
        self._load_pretrained_weights()

        # Move to designated device
        self.to(self.vision_device)

        # Freeze if specified
        if self.freeze_vision:
            self._freeze_encoder()

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            if not torch.cuda.is_available():
                return torch.device("cpu")

            num_gpus = torch.cuda.device_count()
            if num_gpus >= 2:
                # Use second GPU for vision (first GPU for language TP)
                return torch.device("cuda:1")
            else:
                # Single GPU: use it
                return torch.device("cuda:0")

        return torch.device(device)

    def _load_pretrained_weights(self):
        """Load weights from HuggingFace transformers."""
        try:
            from transformers import AutoModel

            # Load pretrained model
            pretrained = AutoModel.from_pretrained(self.vision_config.model_name)

            # Map weights based on encoder type
            if "siglip" in self.vision_config.model_name.lower():
                self._load_siglip_weights(pretrained)
            elif "clip" in self.vision_config.model_name.lower():
                self._load_clip_weights(pretrained)
            else:
                # Generic ViT loading
                self._load_vit_weights(pretrained)

        except ImportError:
            print("Warning: transformers not installed. Using random initialization.")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")

    def _load_siglip_weights(self, pretrained_model):
        """Load weights from SigLIP model."""
        # SigLIP uses vision_model prefix
        vision_model = pretrained_model.vision_model

        # Load embeddings
        state_dict = {}
        pretrained_state = vision_model.state_dict()

        # Map patch embedding
        if "embeddings.patch_embedding.weight" in pretrained_state:
            state_dict["embeddings.patch_embedding.weight"] = pretrained_state[
                "embeddings.patch_embedding.weight"
            ]
            state_dict["embeddings.patch_embedding.bias"] = pretrained_state.get(
                "embeddings.patch_embedding.bias", torch.zeros(1)
            )

        # Map position embedding
        if "embeddings.position_embedding.weight" in pretrained_state:
            state_dict["embeddings.position_embedding"] = pretrained_state[
                "embeddings.position_embedding.weight"
            ]

        # Load encoder layers
        for i in range(min(len(self.layers), len(vision_model.encoder.layers))):
            layer_prefix = f"encoder.layers.{i}"
            our_prefix = f"layers.{i}"

            # Layer norms
            state_dict[f"{our_prefix}.norm1.weight"] = pretrained_state[
                f"{layer_prefix}.layer_norm1.weight"
            ]
            state_dict[f"{our_prefix}.norm1.bias"] = pretrained_state[
                f"{layer_prefix}.layer_norm1.bias"
            ]
            state_dict[f"{our_prefix}.norm2.weight"] = pretrained_state[
                f"{layer_prefix}.layer_norm2.weight"
            ]
            state_dict[f"{our_prefix}.norm2.bias"] = pretrained_state[
                f"{layer_prefix}.layer_norm2.bias"
            ]

            # Attention
            state_dict[f"{our_prefix}.attn.qkv.weight"] = pretrained_state[
                f"{layer_prefix}.self_attn.qkv_proj.weight"
            ]
            state_dict[f"{our_prefix}.attn.qkv.bias"] = pretrained_state[
                f"{layer_prefix}.self_attn.qkv_proj.bias"
            ]
            state_dict[f"{our_prefix}.attn.proj.weight"] = pretrained_state[
                f"{layer_prefix}.self_attn.projection.weight"
            ]
            state_dict[f"{our_prefix}.attn.proj.bias"] = pretrained_state[
                f"{layer_prefix}.self_attn.projection.bias"
            ]

            # MLP
            state_dict[f"{our_prefix}.mlp.fc1.weight"] = pretrained_state[
                f"{layer_prefix}.mlp.fc1.weight"
            ]
            state_dict[f"{our_prefix}.mlp.fc1.bias"] = pretrained_state[
                f"{layer_prefix}.mlp.fc1.bias"
            ]
            state_dict[f"{our_prefix}.mlp.fc2.weight"] = pretrained_state[
                f"{layer_prefix}.mlp.fc2.weight"
            ]
            state_dict[f"{our_prefix}.mlp.fc2.bias"] = pretrained_state[
                f"{layer_prefix}.mlp.fc2.bias"
            ]

        # Final norm
        if "post_layernorm.weight" in pretrained_state:
            state_dict["norm.weight"] = pretrained_state["post_layernorm.weight"]
            state_dict["norm.bias"] = pretrained_state["post_layernorm.bias"]

        # Load mapped weights
        self.load_state_dict(state_dict, strict=False)

    def _load_clip_weights(self, pretrained_model):
        """Load weights from CLIP model."""
        # Similar to SigLIP but with CLIP-specific mapping
        vision_model = pretrained_model.vision_model
        state_dict = {}
        pretrained_state = vision_model.state_dict()

        # Map embeddings
        if "embeddings.patch_embedding.weight" in pretrained_state:
            state_dict["embeddings.patch_embedding.weight"] = pretrained_state[
                "embeddings.patch_embedding.weight"
            ]
        if "embeddings.position_embedding.weight" in pretrained_state:
            state_dict["embeddings.position_embedding"] = pretrained_state[
                "embeddings.position_embedding.weight"
            ]

        # Load encoder layers
        for i in range(min(len(self.layers), len(vision_model.encoder.layers))):
            layer_prefix = f"encoder.layers.{i}"
            our_prefix = f"layers.{i}"

            # Layer norms (CLIP uses different naming)
            state_dict[f"{our_prefix}.norm1.weight"] = pretrained_state[
                f"{layer_prefix}.layer_norm1.weight"
            ]
            state_dict[f"{our_prefix}.norm1.bias"] = pretrained_state[
                f"{layer_prefix}.layer_norm1.bias"
            ]
            state_dict[f"{our_prefix}.norm2.weight"] = pretrained_state[
                f"{layer_prefix}.layer_norm2.weight"
            ]
            state_dict[f"{our_prefix}.norm2.bias"] = pretrained_state[
                f"{layer_prefix}.layer_norm2.bias"
            ]

            # Attention
            state_dict[f"{our_prefix}.attn.qkv.weight"] = pretrained_state[
                f"{layer_prefix}.self_attn.qkv_proj.weight"
            ]
            state_dict[f"{our_prefix}.attn.qkv.bias"] = pretrained_state[
                f"{layer_prefix}.self_attn.qkv_proj.bias"
            ]
            state_dict[f"{our_prefix}.attn.proj.weight"] = pretrained_state[
                f"{layer_prefix}.self_attn.projection.weight"
            ]
            state_dict[f"{our_prefix}.attn.proj.bias"] = pretrained_state[
                f"{layer_prefix}.self_attn.projection.bias"
            ]

            # MLP
            state_dict[f"{our_prefix}.mlp.fc1.weight"] = pretrained_state[
                f"{layer_prefix}.mlp.fc1.weight"
            ]
            state_dict[f"{our_prefix}.mlp.fc1.bias"] = pretrained_state[
                f"{layer_prefix}.mlp.fc1.bias"
            ]
            state_dict[f"{our_prefix}.mlp.fc2.weight"] = pretrained_state[
                f"{layer_prefix}.mlp.fc2.weight"
            ]
            state_dict[f"{our_prefix}.mlp.fc2.bias"] = pretrained_state[
                f"{layer_prefix}.mlp.fc2.bias"
            ]

        self.load_state_dict(state_dict, strict=False)

    def _load_vit_weights(self, pretrained_model):
        """Load weights from generic ViT model."""
        # Generic ViT weight loading
        if hasattr(pretrained_model, "vit"):
            vision_model = pretrained_model.vit
        else:
            vision_model = pretrained_model

        state_dict = {}
        pretrained_state = vision_model.state_dict()

        # Try to map weights generically
        for key, value in pretrained_state.items():
            # Map common patterns
            new_key = key.replace("encoder.layer", "layers")
            new_key = new_key.replace("attention.output.dense", "attn.proj")
            new_key = new_key.replace("intermediate.dense", "mlp.fc1")
            new_key = new_key.replace("output.dense", "mlp.fc2")
            new_key = new_key.replace("layernorm_before", "norm1")
            new_key = new_key.replace("layernorm_after", "norm2")
            new_key = new_key.replace("embeddings.patch_embedding.projection", "embeddings.patch_embedding")
            new_key = new_key.replace("embeddings.position_embedding", "embeddings.position_embedding")

            if new_key in self.state_dict():
                state_dict[new_key] = value

        self.load_state_dict(state_dict, strict=False)

    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        """Encode images to vision features.

        Args:
            pixel_values: [batch, C, H, W] preprocessed images
            output_hidden_states: Whether to return all layer outputs

        Returns:
            [batch, num_patches, hidden_size] vision features
        """
        # Patch embedding
        hidden_states = self.embeddings(pixel_values)

        # Store hidden states if requested
        all_hidden_states = [hidden_states] if output_hidden_states else None

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            return hidden_states, all_hidden_states

        return hidden_states

    def get_num_patches(self) -> int:
        """Get the number of patches per image."""
        return self.embeddings.num_patches

    def get_hidden_size(self) -> int:
        """Get the vision encoder hidden size."""
        return self.vision_config.hidden_size
