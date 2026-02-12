# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Asynchronous VLA model with producer-consumer vision pipeline.

Extends VLAModel with async vision processing for maximum GPU utilization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from ironcore.vision.async_pipeline import (
    HybridAsyncVisionPipeline,
)
from ironcore.vision.encoder import VisionEncoder

if TYPE_CHECKING:
    pass  # Type-only imports can be added here as needed


class VLAModelAsync(BaseModule):
    """VLA Model with asynchronous vision processing.

    Uses producer-consumer pattern:
    - Producer: Vision encoder on CPU (background thread)
    - Consumer: Language model on GPU (main thread)

    This maximizes GPU utilization by pre-computing vision features
    while the GPU processes the language model.

    Usage:
        >>> model = VLAModelAsync(config)
        >>> model.start_async()  # Start vision encoder threads
        >>>
        >>> for batch in dataloader:
        ...     # Submit vision for async encoding
        ...     model.submit_vision(batch["pixel_values"], batch_idx=i)
        ...
        ...     # Get pre-computed features (may briefly wait)
        ...     vision_features = model.get_vision_features(batch_idx=i)
        ...
        ...     # Forward with pre-computed features
        ...     loss = model.forward_with_vision(
        ...         batch["input_ids"],
        ...         vision_features=vision_features,
        ...         labels=batch["labels"],
        ...         actions=batch["actions"],
        ...     )
        >>>
        >>> model.stop_async()  # Stop background threads

    For training integration:
        >>> # In training_utils.py
        >>> def forward_step_vla_async(model, data_iterator):
        ...     batch = next(data_iterator)
        ...     return model.forward_with_vision(
        ...         batch["input_ids"],
        ...         vision_features=batch.get("vision_features"),
        ...         pixel_values=batch.get("pixel_values"),
        ...         labels=batch.get("labels"),
        ...         actions=batch.get("actions"),
        ...     )
    """

    def __init__(self, config: MainConfig):
        super().__init__(config)

        self.vla_config = config.vla
        self.model_config = config.model

        # Vision components
        self.vision_encoder = VisionEncoder(config)
        self.vision_projector = VisionLanguageProjector(config)

        # Async vision pipeline
        self.async_pipeline = HybridAsyncVisionPipeline(
            config,
            self.vision_encoder,
            queue_size=config.trainer.async_vision_queue_size
            if hasattr(config.trainer, "async_vision_queue_size")
            else 4,
            num_cpu_workers=config.trainer.async_vision_workers
            if hasattr(config.trainer, "async_vision_workers")
            else 2,
        )

        # Fusion module
        self.fusion = VisionLanguageFusion(
            config,
            num_layers=config.vla.fusion.num_layers,
            fusion_type=config.vla.fusion.fusion_type,
        )

        # Language components
        self.embedding = LanguageModelEmbedding(config)
        self.model = TransformerModel(config)

        # Action components
        self.action_head = ActionHead(config)
        self.action_loss_fn = ActionLoss(config.vla.action)

        # Special tokens
        self.image_token_id = config.vla.image_token_id
        self.num_image_tokens = config.vla.num_image_tokens

        # Batch tracking for async
        self._current_batch_idx = 0
        self._vision_cache: dict[int, torch.Tensor] = {}

        # Initialize weights
        self.init_weights()

    def start_async(self):
        """Start async vision processing pipeline."""
        self.async_pipeline.start()
        self._current_batch_idx = 0
        self._vision_cache.clear()

    def stop_async(self, timeout: float = 5.0):
        """Stop async vision processing."""
        self.async_pipeline.stop(timeout=timeout)
        self._vision_cache.clear()

    def submit_vision(
        self,
        pixel_values: torch.Tensor,
        batch_idx: int | None = None,
    ) -> int:
        """Submit images for async encoding.

        Call this BEFORE you need the features to allow
        background processing while other work happens.

        Args:
            pixel_values: [batch, C, H, W] images
            batch_idx: Optional batch index for tracking

        Returns:
            Batch index for later retrieval
        """
        if batch_idx is None:
            batch_idx = self._current_batch_idx
            self._current_batch_idx += 1

        self.async_pipeline.submit(pixel_values, batch_idx)
        return batch_idx

    def get_vision_features(
        self,
        batch_idx: int,
        timeout: float = 5.0,
    ) -> torch.Tensor | None:
        """Get pre-computed vision features.

        Args:
            batch_idx: Batch index from submit_vision
            timeout: Max wait time

        Returns:
            Vision features or None if timeout
        """
        # Check cache first
        if batch_idx in self._vision_cache:
            return self._vision_cache.pop(batch_idx)

        # Get from pipeline
        features = self.async_pipeline.get_features(batch_idx, timeout)

        if features is not None:
            # Project to language dimension
            features = self.vision_projector(features)

        return features

    def prefetch_vision(
        self,
        pixel_values: torch.Tensor,
        batch_idx: int,
    ):
        """Prefetch and cache vision features.

        Encodes and caches for later retrieval without waiting.

        Args:
            pixel_values: Images to encode
            batch_idx: Batch index
        """
        self.submit_vision(pixel_values, batch_idx)

        # Encode synchronously and cache for immediate use
        with torch.no_grad():
            features = self.vision_encoder(pixel_values)
            features = self.vision_projector(features)
            self._vision_cache[batch_idx] = features

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        vision_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        batch_idx: int | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional async vision features.

        Supports two modes:
        1. Pre-computed vision_features: Skip vision encoding entirely
        2. pixel_values provided: Encode synchronously (or use cached)

        For best performance, use submit_vision() before forward(),
        then pass vision_features directly.

        Args:
            input_ids: [batch, seq_len] token IDs
            pixel_values: [batch, C, H, W] images (optional if vision_features provided)
            vision_features: Pre-computed vision features [batch, num_patches, d_model]
            labels: [batch, seq_len] labels for LM loss
            actions: [batch, action_dim * horizon] target actions
            attention_mask: [batch, seq_len] attention mask
            position_ids: [batch, seq_len] position IDs
            vision_mask: [batch, vision_len] mask for vision tokens
            batch_idx: Batch index for async feature retrieval

        Returns:
            Total loss (language loss + action loss)
        """
        device = input_ids.device
        batch_size = input_ids.size(0)

        # 1. Get vision features
        if vision_features is not None:
            # Use pre-computed features (best for async)
            pass
        elif batch_idx is not None and batch_idx in self._vision_cache:
            # Use cached features
            vision_features = self._vision_cache.pop(batch_idx)
        elif pixel_values is not None:
            # Synchronous encoding (fallback)
            raw_features = self.vision_encoder(pixel_values)
            vision_features = self.vision_projector(raw_features)
        else:
            vision_features = None

        # 2. Get text embeddings
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0)

        text_embeds = self.embedding(input_ids, position_ids)

        # 3. Vision-Language Fusion
        if vision_features is not None:
            hidden_states = self.fusion(
                text_embeds,
                vision_features,
                vision_mask,
            )
        else:
            hidden_states = text_embeds

        # 4. Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.size(1), device=device).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, -1)

        extended_attention_mask = self._prepare_attention_mask(
            attention_mask, hidden_states.size(1)
        )

        # 5. Transformer forward
        hidden_states = self.model(
            hidden_states,
            extended_attention_mask,
            rotary_pos_emb=None,
        )

        # 6. Compute losses
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if labels is not None:
            lm_loss = self._compute_lm_loss(hidden_states, labels)
            total_loss = total_loss + self.vla_config.language_weight * lm_loss

        if actions is not None:
            action_loss = self._compute_action_loss(hidden_states, actions)
            total_loss = total_loss + self.vla_config.action.action_weight * action_loss

        return total_loss

    def forward_with_vision(
        self,
        input_ids: torch.Tensor,
        vision_features: torch.Tensor,
        labels: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Optimized forward with pre-computed vision features.

        Use this when vision features are already computed
        (e.g., from async pipeline or dataloader prefetch).

        Args:
            input_ids: [batch, seq_len] token IDs
            vision_features: [batch, num_patches, d_model] pre-computed features
            labels: Labels for LM loss
            actions: Target actions
            attention_mask: Attention mask

        Returns:
            Total loss
        """
        return self.forward(
            input_ids=input_ids,
            vision_features=vision_features,
            labels=labels,
            actions=actions,
            attention_mask=attention_mask,
        )

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Prepare attention mask for transformer."""
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask.float()) * torch.finfo(extended_mask.dtype).min
        return extended_mask

    def _compute_lm_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute language modeling loss."""
        if not hasattr(self, "lm_head"):
            self.lm_head = nn.Linear(
                self.model_config.d_model,
                self.embedding.word_embeddings.weight.size(0),
                bias=False,
            ).to(hidden_states.device)

        logits = self.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

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
        """Compute action prediction loss."""
        pred_actions = self.action_head(hidden_states)
        loss = self.action_loss_fn(pred_actions, target_actions)
        return loss

    def get_pipeline_stats(self) -> dict:
        """Get async pipeline statistics."""
        return self.async_pipeline.get_stats()


class VLATrainingIterator:
    """Iterator that prefetches vision features during training.

    Wraps a dataloader and pre-computes vision features for the
    next batch while the current batch is being processed.

    Example:
        >>> model = VLAModelAsync(config)
        >>> model.start_async()
        >>>
        >>> train_iter = VLATrainingIterator(dataloader, model)
        >>> for batch in train_iter:
        ...     # batch already contains pre-computed vision_features
        ...     loss = model.forward_with_vision(
        ...         batch["input_ids"],
        ...         batch["vision_features"],
        ...         labels=batch["labels"],
        ...         actions=batch["actions"],
        ...     )
        >>>
        >>> model.stop_async()
    """

    def __init__(
        self,
        dataloader,
        model: VLAModelAsync,
        prefetch_batches: int = 1,
    ):
        """Initialize training iterator.

        Args:
            dataloader: PyTorch dataloader
            model: Async VLA model
            prefetch_batches: Number of batches to prefetch
        """
        self.dataloader = dataloader
        self.model = model
        self.prefetch_batches = prefetch_batches
        self._iterator = None
        self._prefetch_queue: list[dict] = []
        self._batch_idx = 0

    def __iter__(self):
        self._iterator = iter(self.dataloader)
        self._batch_idx = 0
        self._prefetch_queue.clear()

        # Prime the prefetch queue
        for _ in range(self.prefetch_batches):
            try:
                batch = next(self._iterator)
                self._submit_for_prefetch(batch)
                self._prefetch_queue.append(batch)
            except StopIteration:
                break

        return self

    def __next__(self) -> dict:
        if not self._prefetch_queue:
            raise StopIteration

        # Get next batch
        batch = self._prefetch_queue.pop(0)

        # Get pre-computed vision features
        if "pixel_values" in batch:
            vision_features = self.model.get_vision_features(
                batch_idx=batch.get("_batch_idx", 0),
                timeout=10.0,
            )
            if vision_features is not None:
                batch["vision_features"] = vision_features

        # Prefetch next batch
        try:
            next_batch = next(self._iterator)
            self._submit_for_prefetch(next_batch)
            self._prefetch_queue.append(next_batch)
        except StopIteration:
            pass

        return batch

    def _submit_for_prefetch(self, batch: dict):
        """Submit batch for async vision encoding."""
        if "pixel_values" in batch:
            batch["_batch_idx"] = self._batch_idx
            self.model.submit_vision(batch["pixel_values"], self._batch_idx)
            self._batch_idx += 1
