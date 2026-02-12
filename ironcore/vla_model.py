# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import torch
import torch.nn.functional as F
from torch import nn

from ironcore.action.head import ActionHead
from ironcore.action.loss import ActionLoss
from ironcore.config import MainConfig
from ironcore.layers.cross_attention import VisionLanguageFusion
from ironcore.layers.embedding import LanguageModelEmbedding
from ironcore.layers.module import BaseModule
from ironcore.layers.multimodal.projection import VisionLanguageProjector
from ironcore.models.transformer import TransformerModel
from ironcore.vision.device_manager import DeviceManager, get_optimal_device_config
from ironcore.vision.encoder import VisionEncoder


class VLAModel(BaseModule):
    """Vision-Language-Action Model for robotics.

    Integrates:
    - VisionEncoder: Encodes images to vision features (configurable device)
    - VisionLanguageProjector: Projects vision features to language space
    - VisionLanguageFusion: Cross-attention for vision-language fusion
    - LanguageModelEmbedding: Token embeddings with position embeddings
    - TransformerModel: Language model backbone
    - ActionHead: Predicts continuous robot actions

    Supports hybrid device placement:
    - Vision encoder can run on CPU, separate GPU, or same GPU
    - Language model uses tensor parallelism
    - Automatic tensor transfer between devices

    Fusion strategies:
    - "gated_cross_attention": Flamingo-style gated cross-attention
    - "qformer": BLIP-2 style Q-Former
    - "concat": Simple concatenation (baseline)
    """

    def __init__(self, config: MainConfig):
        super().__init__(config)

        self.vla_config = config.vla
        self.model_config = config.model

        # Initialize device manager for hybrid placement
        self._setup_device_manager()

        # Vision components (placed on vision_device)
        self.vision_encoder = VisionEncoder(config)
        self.vision_projector = VisionLanguageProjector(config)

        # Fusion module (cross-attention for vision-language)
        self.fusion = VisionLanguageFusion(
            config,
            num_layers=config.vla.fusion.num_layers,
            fusion_type=config.vla.fusion.fusion_type,
        )

        # Language components (placed on language_device / TP devices)
        self.embedding = LanguageModelEmbedding(config)
        self.model = TransformerModel(config)

        # Action components
        self.action_head = ActionHead(config)
        self.action_loss_fn = ActionLoss(config.vla.action)

        # Special tokens
        self.image_token_id = config.vla.image_token_id
        self.num_image_tokens = config.vla.num_image_tokens

        # Initialize weights
        self.init_weights()

    def _setup_device_manager(self):
        """Setup device manager based on configuration."""
        vision_device = self.vla_config.vision.device

        # Get optimal config if auto
        if vision_device == "auto":
            tp_size = self.config.trainer.tensor_model_parallel_size
            optimal = get_optimal_device_config(tensor_parallel_size=tp_size)
            vision_device = optimal["vision_device"]
            print(f"[VLA] Auto device config: {optimal['recommendation']}")

        self.device_manager = DeviceManager(
            vision_device=vision_device,
            language_device="cuda:0",  # Language model uses TP, managed separately
        )

        print(f"[VLA] Vision encoder on: {self.device_manager.vision_device}")
        print(f"[VLA] Language model on: {self.device_manager.language_device}")

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for VLA model with cross-attention fusion.

        Architecture:
        1. Vision: Images -> VisionEncoder -> Projector -> Vision tokens
        2. Language: Input IDs -> Embedding -> Text embeddings
        3. Fusion: Text embeddings <-Cross-Attention-> Vision tokens
        4. Transformer: Fused embeddings -> Transformer layers
        5. Output: Language loss + Action loss

        Handles device transfers automatically:
        - Vision processing on vision_device (CPU or separate GPU)
        - Language processing on language_device with TP

        Args:
            input_ids: [batch, seq_len] token IDs
            pixel_values: [batch, C, H, W] preprocessed images
            labels: [batch, seq_len] labels for language modeling loss
            actions: [batch, action_dim * horizon] target actions
            attention_mask: [batch, seq_len] attention mask
            position_ids: [batch, seq_len] position IDs
            vision_mask: [batch, vision_len] mask for vision tokens (optional)

        Returns:
            Total loss (language loss + action loss)
        """
        batch_size = input_ids.size(0)
        device = input_ids.device  # Language device

        # 1. Encode images if provided (with device transfer)
        if pixel_values is not None:
            # Move images to vision device
            pixel_values = self.device_manager.move_tensor(pixel_values, "vision")

            # Encode on vision device
            vision_features = self.vision_encoder(pixel_values)

            # Move features back to language device
            vision_features = self.device_manager.move_tensor(vision_features, "language")

            # Project vision features to language dimension
            vision_tokens = self.vision_projector(vision_features)
        else:
            vision_tokens = None

        # 2. Get text embeddings (on language device)
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0)

        text_embeds = self.embedding(input_ids, position_ids)

        # 3. Vision-Language Fusion via Cross-Attention
        if vision_tokens is not None:
            # Use cross-attention to fuse vision and language
            hidden_states = self.fusion(
                text_embeds,
                vision_tokens,
                vision_mask,
            )
        else:
            hidden_states = text_embeds

        # 4. Create attention mask for transformer
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.size(1), device=device).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, -1)

        # Expand attention mask for transformer
        extended_attention_mask = self._prepare_attention_mask(
            attention_mask, hidden_states.size(1)
        )

        # 5. Process through transformer
        hidden_states = self.model(
            hidden_states,
            extended_attention_mask,
            rotary_pos_emb=None,
        )

        # 6. Compute losses
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Language modeling loss
        if labels is not None:
            lm_loss = self._compute_lm_loss(hidden_states, labels)
            total_loss = total_loss + self.vla_config.language_weight * lm_loss

        # Action prediction loss
        if actions is not None:
            action_loss = self._compute_action_loss(hidden_states, actions)
            total_loss = total_loss + self.vla_config.action.action_weight * action_loss

        return total_loss

    def _fuse_embeddings(
        self,
        text_embeds: torch.Tensor,
        vision_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        image_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse vision tokens with text embeddings at <image> positions.

        Args:
            text_embeds: [batch, seq_len, d_model]
            vision_tokens: [batch, num_patches, d_model]
            input_ids: [batch, seq_len]
            image_token_mask: Optional precomputed mask

        Returns:
            Fused embeddings with vision tokens replacing <image> placeholders
        """
        batch_size, seq_len, d_model = text_embeds.shape
        num_image_tokens = vision_tokens.size(1)

        if image_token_mask is None:
            # Find image token positions
            image_token_mask = (input_ids == self.image_token_id)

        # Count image tokens per sample
        image_counts = image_token_mask.sum(dim=1)

        # If no image tokens, return text embeddings as-is
        if image_counts.sum() == 0:
            return text_embeds

        # Build new sequence with vision tokens inserted
        # For simplicity, we assume single image per sample with contiguous <image> tokens
        # More sophisticated fusion can be added later

        # Find first image token position in each sample
        first_image_pos = image_token_mask.int().argmax(dim=1)

        # Create output tensor
        # New length: original - num_image_placeholder + num_vision_tokens
        placeholder_count = image_counts[0].item()  # Assume same for all samples

        if placeholder_count == num_image_tokens:
            # Direct replacement: replace placeholder tokens with vision tokens
            fused = text_embeds.clone()
            for b in range(batch_size):
                start_pos = first_image_pos[b].item()
                if start_pos < seq_len and image_counts[b] > 0:
                    fused[b, start_pos:start_pos + num_image_tokens] = vision_tokens[b]
            return fused

        # For different sizes, we need to insert/expand
        # Create new sequence with proper size
        new_seq_len = seq_len - placeholder_count + num_image_tokens
        fused = torch.zeros(batch_size, new_seq_len, d_model, device=text_embeds.device, dtype=text_embeds.dtype)

        for b in range(batch_size):
            start_pos = first_image_pos[b].item()
            if start_pos > 0:
                # Copy tokens before image
                fused[b, :start_pos] = text_embeds[b, :start_pos]

            # Insert vision tokens
            fused[b, start_pos:start_pos + num_image_tokens] = vision_tokens[b]

            # Copy tokens after image placeholder
            end_placeholder = start_pos + placeholder_count
            if end_placeholder < seq_len:
                fused[b, start_pos + num_image_tokens:] = text_embeds[b, end_placeholder:]

        return fused

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Prepare attention mask for transformer.

        Args:
            attention_mask: [batch, seq_len]
            seq_len: Sequence length (may differ from attention_mask if vision tokens inserted)

        Returns:
            Attention mask in format expected by transformer
        """
        # Expand to [batch, 1, 1, seq_len] for broadcasting
        extended_mask = attention_mask[:, None, None, :]

        # Convert 0s to -inf for masking
        extended_mask = (1.0 - extended_mask.float()) * torch.finfo(extended_mask.dtype).min

        return extended_mask

    def _compute_lm_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute language modeling loss.

        Args:
            hidden_states: [batch, seq_len, d_model]
            labels: [batch, seq_len]

        Returns:
            Scalar loss
        """
        # Use tied embeddings if available, otherwise use separate lm_head
        # For now, use a simple linear projection
        if not hasattr(self, "lm_head"):
            self.lm_head = nn.Linear(
                self.model_config.d_model,
                self.embedding.word_embeddings.weight.size(0),
                bias=False,
            ).to(hidden_states.device)

        # Compute logits
        logits = self.lm_head(hidden_states)  # [batch, seq_len, vocab_size]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return loss

    def _compute_action_loss(
        self,
        hidden_states: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute action prediction loss.

        Args:
            hidden_states: [batch, seq_len, d_model]
            target_actions: [batch, action_dim * horizon]

        Returns:
            Scalar loss
        """
        # Predict actions from final hidden state
        pred_actions = self.action_head(hidden_states)

        # Compute loss
        loss = self.action_loss_fn(pred_actions, target_actions)

        return loss

    def predict_action(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference: predict actions from image and text.

        Args:
            input_ids: [batch, seq_len] token IDs
            pixel_values: [batch, C, H, W] preprocessed images
            attention_mask: [batch, seq_len] attention mask

        Returns:
            [batch, action_dim * horizon] predicted actions
        """
        # Encode vision
        vision_features = self.vision_encoder(pixel_values)
        vision_tokens = self.vision_projector(vision_features)

        # Get text embeddings
        device = input_ids.device
        position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
        text_embeds = self.embedding(input_ids, position_ids)

        # Fuse embeddings
        fused_embeds = self._fuse_embeddings(text_embeds, vision_tokens, input_ids)

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
        extended_mask = self._prepare_attention_mask(attention_mask, fused_embeds.size(1))

        # Forward through transformer
        hidden_states = self.model(fused_embeds, extended_mask, None)

        # Predict actions
        actions = self.action_head(hidden_states)

        return actions

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to vision tokens.

        Args:
            pixel_values: [batch, C, H, W] preprocessed images

        Returns:
            [batch, num_patches, d_model] vision tokens
        """
        vision_features = self.vision_encoder(pixel_values)
        vision_tokens = self.vision_projector(vision_features)
        return vision_tokens
