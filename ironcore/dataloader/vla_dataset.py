# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from ironcore.config.config import BaseConfig
from ironcore.config.config_vla import VisionConfig


@dataclass
class VLADatasetConfig(BaseConfig):
    """Configuration for VLA dataset."""

    name: str = ""
    source: str = ""
    task_type: Literal["vla"] = "vla"
    ratio: float = 1.0
    samples: int | None = None

    # Column names in dataset
    image_column: str = "image"
    instruction_column: str = "instruction"
    action_column: str = "action"
    state_column: str | None = None

    # Image settings
    image_size: int = 384

    # Data format
    format: Literal["bridge", "droid", "rtx", "custom"] = "bridge"

    # Action settings
    action_dim: int = 7
    prediction_horizon: int = 1


class VLADataset(IterableDataset):
    """Iterable dataset for VLA training.

    Supports common robotics dataset formats:
    - Bridge v2
    - DROID
    - RT-X
    - Custom formats
    """

    def __init__(
        self,
        config: VLADatasetConfig,
        vision_config: VisionConfig,
        split: Literal["train", "eval", "test"] = "train",
        tokenizer=None,
        image_token_id: int = -200,
        action_token_id: int = -201,
    ):
        super().__init__()

        self.config = config
        self.vision_config = vision_config
        self.split = split
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.action_token_id = action_token_id

        self.source_path = Path(config.source)

        # Determine data format
        self._detect_format()

    def _detect_format(self):
        """Detect dataset format from directory structure."""
        if self.source_path.suffix == ".json":
            # Single JSON file
            self.format = "json"
            self.data_files = [self.source_path]
        elif self.source_path.is_dir():
            # Directory of data files
            if (self.source_path / "metadata.json").exists():
                # Bridge-like format
                self.format = "bridge"
                with open(self.source_path / "metadata.json") as f:
                    self.metadata = json.load(f)
                self.data_files = sorted(self.source_path.glob("*.json"))
            elif (self.source_path / "data").exists():
                # RT-X like format with data subdirectory
                self.format = "rtx"
                self.data_files = sorted((self.source_path / "data").glob("*.tfrecord"))
            else:
                # Default: look for JSON/parquet files
                self.format = "custom"
                self.data_files = (
                    list(self.source_path.glob("*.json"))
                    + list(self.source_path.glob("*.parquet"))
                )
        else:
            raise ValueError(f"Unknown source format: {self.source_path}")

    def __iter__(self) -> Iterator[dict]:
        """Iterate over dataset samples.

        Yields:
            Dictionary with:
            - input_ids: Tokenized instruction with image/action tokens
            - labels: Labels for language modeling
            - image: PIL Image
            - actions: Action tensor
        """
        if self.format == "bridge":
            yield from self._iter_bridge()
        elif self.format == "json":
            yield from self._iter_json()
        else:
            yield from self._iter_custom()

    def _iter_bridge(self) -> Iterator[dict]:
        """Iterate over Bridge v2 format."""
        for data_file in self.data_files:
            with open(data_file) as f:
                data = json.load(f)

            for episode in data.get("episodes", [data]):
                for step in episode.get("steps", []):
                    sample = self._process_sample(step, data_file.parent)
                    if sample is not None:
                        yield sample

    def _iter_json(self) -> Iterator[dict]:
        """Iterate over single JSON file."""
        for data_file in self.data_files:
            with open(data_file) as f:
                data = json.load(f)

            if isinstance(data, list):
                for sample in data:
                    processed = self._process_sample(sample, data_file.parent)
                    if processed is not None:
                        yield processed
            else:
                processed = self._process_sample(data, data_file.parent)
                if processed is not None:
                    yield processed

    def _iter_custom(self) -> Iterator[dict]:
        """Iterate over custom format."""
        for data_file in self.data_files:
            if data_file.suffix == ".json":
                with open(data_file) as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for sample in data:
                        processed = self._process_sample(sample, data_file.parent)
                        if processed is not None:
                            yield processed

    def _process_sample(self, sample: dict, base_path: Path) -> dict | None:
        """Process a single sample.

        Args:
            sample: Raw sample dictionary
            base_path: Base path for resolving relative paths

        Returns:
            Processed sample or None if invalid
        """
        try:
            # Load image
            image_path = sample.get(self.config.image_column)
            if image_path is not None:
                if not Path(image_path).is_absolute():
                    image_path = base_path / image_path
                image = Image.open(image_path).convert("RGB")
            else:
                # Try direct image data
                image_data = sample.get("image_data")
                if image_data is not None:
                    import io
                    image = Image.open(io.BytesIO(image_data)).convert("RGB")
                else:
                    return None

            # Get instruction
            instruction = sample.get(self.config.instruction_column, "")

            # Get action
            action = sample.get(self.config.action_column)
            if action is None:
                return None

            # Convert action to tensor
            if isinstance(action, str):
                action = json.loads(action)
            if isinstance(action, list):
                action = torch.tensor(action, dtype=torch.float32)
            elif isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()

            # Ensure action has correct shape
            if action.dim() == 1:
                action = action.unsqueeze(0)  # Add horizon dimension
            if action.size(0) < self.config.prediction_horizon:
                # Pad with last action
                pad_size = self.config.prediction_horizon - action.size(0)
                action = torch.cat([action, action[-1:].repeat(pad_size, 1)])
            action = action[:self.config.prediction_horizon].flatten()

            # Tokenize instruction with special tokens
            if self.tokenizer is not None:
                # Format: <image> What should I do? <action>
                text = f"<image>{instruction}"
                tokens = self.tokenizer.encode(text)

                # Insert image token
                input_ids = [self.image_token_id] + tokens

                # Create labels (shift for next token prediction)
                labels = [-100] + tokens[1:]  # Don't predict image token
            else:
                # Return raw text for later tokenization
                input_ids = instruction
                labels = None

            return {
                "input_ids": input_ids,
                "labels": labels,
                "image": image,
                "actions": action,
            }

        except Exception as e:
            print(f"Error processing sample: {e}")
            return None


class VLAShuffledDataset(IterableDataset):
    """VLA dataset with shuffle buffer for better training."""

    def __init__(
        self,
        dataset: VLADataset,
        buffer_size: int = 1000,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        """Iterate with shuffling."""
        import random

        rng = random.Random(self.seed)
        buffer = []

        for sample in self.dataset:
            buffer.append(sample)
            if len(buffer) >= self.buffer_size:
                idx = rng.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)

        # Drain remaining buffer
        while buffer:
            idx = rng.randint(0, len(buffer) - 1)
            yield buffer.pop(idx)
