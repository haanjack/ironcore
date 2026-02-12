# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from typing import Any

import numpy as np
import torch
from PIL import Image

from ironcore.config.config_vla import VisionConfig
from ironcore.vision.image_processor import ImageProcessor


class VLACollator:
    """Collator for VLA batches.

    Handles:
    - Padding sequences to uniform length
    - Processing images into tensors
    - Stacking action tensors
    - Creating attention masks
    """

    def __init__(
        self,
        vision_config: VisionConfig,
        pad_token_id: int = 0,
        max_seq_len: int = 512,
        image_token_id: int = -200,
        action_token_id: int = -201,
    ):
        self.vision_config = vision_config
        self.image_processor = ImageProcessor(vision_config)
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.image_token_id = image_token_id
        self.action_token_id = action_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Dictionary with:
            - input_ids: [batch, seq_len] padded token IDs
            - labels: [batch, seq_len] padded labels
            - pixel_values: [batch, C, H, W] processed images
            - actions: [batch, action_dim * horizon] action tensors
            - attention_mask: [batch, seq_len] attention mask
            - image_token_mask: [batch, seq_len] mask for image tokens
        """
        # Process each sample
        processed_batch = []

        for sample in batch:
            processed = self._process_sample(sample)
            processed_batch.append(processed)

        # Stack and pad batch
        return self._collate_batch(processed_batch)

    def _process_sample(self, sample: dict) -> dict:
        """Process a single sample."""
        # Handle input_ids
        input_ids = sample["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        elif isinstance(input_ids, str):
            # Tokenize if still string
            input_ids = torch.tensor([self.pad_token_id], dtype=torch.long)

        # Handle labels
        labels = sample.get("labels")
        if labels is not None:
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = torch.full_like(input_ids, -100)

        # Handle image
        image = sample["image"]
        if isinstance(image, Image.Image):
            pixel_values = self.image_processor.preprocess(image)
        elif isinstance(image, torch.Tensor):
            pixel_values = image
        else:
            # Create dummy image if missing
            pixel_values = torch.zeros(1, 3, self.vision_config.image_size, self.vision_config.image_size)

        # Handle actions
        actions = sample["actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.clone()
        elif isinstance(actions, (list, np.ndarray)):
            actions = torch.tensor(actions, dtype=torch.float32)

        # Create image token mask
        image_token_mask = (input_ids == self.image_token_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "actions": actions,
            "image_token_mask": image_token_mask,
        }

    def _collate_batch(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate processed samples into batch tensors."""
        # Find max sequence length
        max_seq_len = max(sample["input_ids"].size(0) for sample in batch)
        max_seq_len = min(max_seq_len, self.max_seq_len)

        # Truncate or pad sequences
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        image_token_mask_list = []
        pixel_values_list = []
        actions_list = []

        for sample in batch:
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            image_token_mask = sample["image_token_mask"]

            seq_len = input_ids.size(0)

            # Truncate if needed
            if seq_len > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                labels = labels[:max_seq_len]
                image_token_mask = image_token_mask[:max_seq_len]
                seq_len = max_seq_len

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones(seq_len, dtype=torch.float32)

            # Pad to max_seq_len
            padding_length = max_seq_len - seq_len
            if padding_length > 0:
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=torch.long),
                ])
                labels = torch.cat([
                    labels,
                    torch.full((padding_length,), -100, dtype=torch.long),
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=torch.float32),
                ])
                image_token_mask = torch.cat([
                    image_token_mask,
                    torch.zeros(padding_length, dtype=torch.bool),
                ])

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            image_token_mask_list.append(image_token_mask)
            pixel_values_list.append(sample["pixel_values"].squeeze(0))
            actions_list.append(sample["actions"])

        # Stack into batch tensors
        return {
            "input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list),
            "pixel_values": torch.stack(pixel_values_list),
            "actions": torch.stack(actions_list),
            "attention_mask": torch.stack(attention_mask_list),
            "image_token_mask": torch.stack(image_token_mask_list),
        }
