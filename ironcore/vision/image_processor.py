# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from typing import Union

import torch
from PIL import Image

from ironcore.config.config_vla import VisionConfig


class ImageProcessor:
    """Image preprocessing for VLA vision encoder.

    Handles image loading, resizing, normalization, and conversion to tensors
    compatible with the vision encoder.
    """

    def __init__(self, config: VisionConfig):
        self.config = config
        self.image_size = config.image_size

        # Standard normalization for SigLIP/CLIP
        self.mean = torch.tensor([0.5, 0.5, 0.5])
        self.std = torch.tensor([0.5, 0.5, 0.5])

    def preprocess(
        self,
        image: Union[Image.Image, torch.Tensor, str],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Preprocess a single image.

        Args:
            image: Input image (PIL Image, tensor, or path)
            dtype: Output tensor dtype

        Returns:
            Preprocessed image tensor [1, C, H, W]
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Convert PIL to tensor
        if isinstance(image, Image.Image):
            image = self._pil_to_tensor(image)

        # Ensure correct shape [C, H, W]
        if image.dim() == 4:
            image = image.squeeze(0)

        # Resize
        image = self._resize(image, self.image_size)

        # Normalize
        image = self._normalize(image)

        # Add batch dimension
        image = image.unsqueeze(0)

        return image.to(dtype)

    def preprocess_batch(
        self,
        images: list[Union[Image.Image, torch.Tensor, str]],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Preprocess a batch of images.

        Args:
            images: List of input images
            dtype: Output tensor dtype

        Returns:
            Preprocessed image tensor [batch, C, H, W]
        """
        processed = [self.preprocess(img, dtype) for img in images]
        return torch.cat(processed, dim=0)

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        import numpy as np

        img_array = np.array(image)
        # HWC -> CHW
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        # Convert to float [0, 1]
        return img_tensor.float() / 255.0

    def _resize(self, image: torch.Tensor, size: int) -> torch.Tensor:
        """Resize image to target size."""
        # image: [C, H, W]
        image = image.unsqueeze(0)  # [1, C, H, W]
        image = torch.nn.functional.interpolate(
            image,
            size=(size, size),
            mode="bicubic",
            align_corners=False,
        )
        return image.squeeze(0)  # [C, H, W]

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image with mean and std."""
        mean = self.mean.to(image.device, image.dtype)
        std = self.std.to(image.device, image.dtype)
        return (image - mean[:, None, None]) / std[:, None, None]
