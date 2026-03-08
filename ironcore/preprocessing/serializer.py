# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Universal Data Serializer

Handles downloading, tokenizing, and serializing datasets into a unified binary format.
Supports pretrain, SFT, DPO, and FIM task types.
"""

import json
from pathlib import Path
from random import Random

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from ironcore.dataloader.data_config import DataConfig, DatasetConfig


class DataSerializer:
    """
    Serializes datasets into unified binary format (.bin + .idx).

    Output Format:
        .bin: Flattened uint16/uint32 array of token IDs
        .idx: NumPy structured array with metadata:
            - offset: Byte offset in .bin
            - length: Token count
            - type: Task type (pretrain/sft/dpo_chosen/dpo_rejected)
            - group_id: For linking DPO pairs (-1 if not paired)
            - mask_ranges: JSON string of [[start, end], ...] for SFT masking
    """

    # Metadata dtype for .idx files
    METADATA_DTYPE = np.dtype(
        [
            ("offset", np.uint64),  # Byte offset in .bin file
            ("length", np.uint32),  # Number of tokens
            ("type", "U20"),  # Task type string
            ("group_id", np.int64),  # For DPO pairing (-1 = not paired)
            ("mask_ranges", "U500"),  # JSON string of masking ranges
        ]
    )

    def __init__(self, data_config: DataConfig, tokenizer, verbose: bool = True):
        """
        Initialize serializer.

        Args:
            data_config: Data configuration
            tokenizer: Tokenizer instance (HF or tiktoken)
            verbose: Whether to print progress
        """
        self.config = data_config
        self.tokenizer = tokenizer
        self.verbose = verbose

        # Create output directories
        self.config.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def serialize_all(self):
        """Serialize all datasets defined in config."""
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Starting serialization of {len(self.config.datasets)} dataset(s)")
            print(f"{'=' * 60}\n")

        for dataset_config in self.config.datasets:
            self.serialize_dataset(dataset_config)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print("Serialization complete!")
            print(f"{'=' * 60}\n")

    def serialize_dataset(self, dataset_config: DatasetConfig):
        """
        Serialize a single dataset.

        Args:
            dataset_config: Configuration for this dataset
        """
        output_path = self.config.get_dataset_output_path(dataset_config)
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx"

        if self.verbose:
            print(f"\n[{dataset_config.name}] Task: {dataset_config.task_type}")
            print(f"  Source: {dataset_config.source}")
            print(f"  Output: {output_path}")

        # Check if already processed
        if bin_path.exists() and idx_path.exists():
            if self.verbose:
                print("  ⚠️  Already processed. Skipping...")
            return

        # Load dataset
        dataset = self._load_dataset(dataset_config)

        if self.verbose:
            # Handle both streaming and non-streaming datasets
            if hasattr(dataset, "__len__"):
                print(f"  Loaded {len(dataset)} samples")
            else:
                print(
                    f"  Streaming dataset (max {dataset_config.max_samples or 'unlimited'} samples)"
                )

        # Serialize based on task type
        if dataset_config.task_type == "pretrain":
            self._serialize_pretrain(dataset, dataset_config, bin_path, idx_path)
        elif dataset_config.task_type == "sft":
            self._serialize_sft(dataset, dataset_config, bin_path, idx_path)
        elif dataset_config.task_type == "dpo":
            self._serialize_dpo(dataset, dataset_config, bin_path, idx_path)
        elif dataset_config.task_type == "grpo":
            self._serialize_grpo(dataset, dataset_config, bin_path, idx_path)
        else:
            raise ValueError(f"Unknown task type: {dataset_config.task_type}")

        if self.verbose:
            print("  ✓ Serialization complete")

    def _load_dataset(self, dataset_config: DatasetConfig):
        """Load dataset from HuggingFace or local source.

        Uses streaming mode when max_samples is specified to avoid downloading
        the full dataset when only a subset is needed.
        """
        use_streaming = dataset_config.max_samples is not None and dataset_config.max_samples > 0

        if Path(dataset_config.source).exists():
            # Local dataset
            if dataset_config.source.endswith(".json") or dataset_config.source.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=dataset_config.source, split="train")
            else:
                dataset = load_dataset(dataset_config.source, split=dataset_config.split)
        # HuggingFace dataset with streaming
        elif use_streaming:
            if dataset_config.subset:
                dataset = load_dataset(
                    dataset_config.source,
                    dataset_config.subset,
                    split=dataset_config.split,
                    cache_dir=str(self.config.cache_dir),
                    streaming=True,
                )
            else:
                dataset = load_dataset(
                    dataset_config.source,
                    split=dataset_config.split,
                    cache_dir=str(self.config.cache_dir),
                    streaming=True,
                )
            # For streaming datasets, take only the needed samples
            if dataset_config.max_samples:
                dataset = dataset.take(dataset_config.max_samples)
            return dataset
        # HuggingFace dataset without streaming (full download)
        elif dataset_config.subset:
            dataset = load_dataset(
                dataset_config.source,
                dataset_config.subset,
                split=dataset_config.split,
                cache_dir=str(self.config.cache_dir),
            )
        else:
            dataset = load_dataset(
                dataset_config.source,
                split=dataset_config.split,
                cache_dir=str(self.config.cache_dir),
            )

        # Limit samples if specified (for non-streaming datasets)
        if dataset_config.max_samples:
            dataset = dataset.select(range(min(dataset_config.max_samples, len(dataset))))

        return dataset

    def _serialize_pretrain(
        self, dataset, dataset_config: DatasetConfig, bin_path: Path, idx_path: Path
    ):
        """
        Serialize pretrain dataset.

        For pretrain, we tokenize raw text and optionally apply FIM transformation.
        No masking metadata needed.
        """
        text_column = dataset_config.text_column

        # Check if FIM is enabled (read from global config, not per-dataset)
        fim_enabled = self.config.fim_rate > 0

        # Get FIM special token IDs if enabled
        if fim_enabled:
            fim_prefix_id = self._get_token_id(self.config.fim_prefix_token)
            fim_suffix_id = self._get_token_id(self.config.fim_suffix_token)
            fim_middle_id = self._get_token_id(self.config.fim_middle_token)
            rng = Random(1337)

            if self.verbose:
                print(f"  FIM enabled: {self.config.fim_rate:.0%} of sequences will be transformed")

        # Open binary file for writing
        all_tokens = []
        metadata = []
        current_offset = 0

        desc = "  Tokenizing" + (" (FIM)" if fim_enabled else "")
        if self.verbose:
            dataset_iter = tqdm(dataset, desc=desc, unit="docs")
        else:
            dataset_iter = dataset

        for sample in dataset_iter:
            text = sample[text_column]

            # Tokenize
            token_ids = self._tokenize(text)

            # Apply FIM transformation with probability fim_rate
            if fim_enabled and rng.random() < self.config.fim_rate:
                token_ids = self._apply_fim_transformation(
                    token_ids, fim_prefix_id, fim_suffix_id, fim_middle_id, rng
                )

            # Add EOS token
            token_ids.append(self.tokenizer.eos_token_id)

            # Append to token stream
            all_tokens.extend(token_ids)

            # Record metadata
            metadata.append(
                (
                    current_offset,  # offset
                    len(token_ids),  # length
                    "pretrain",  # type
                    -1,  # group_id (not used)
                    "[]",  # mask_ranges (empty)
                )
            )

            current_offset += len(token_ids)

        # Save .bin file
        tokens_array = np.array(
            all_tokens, dtype=np.uint16 if max(all_tokens) < 65535 else np.uint32
        )
        with open(bin_path, "wb") as f:
            tokens_array.tofile(f)

        # Save .idx file
        metadata_array = np.array(metadata, dtype=self.METADATA_DTYPE)
        np.save(idx_path, metadata_array)

        if self.verbose:
            print(f"  Tokens: {len(tokens_array):,}")
            print(f"  Documents: {len(metadata):,}")

    def _serialize_sft(
        self, dataset, dataset_config: DatasetConfig, bin_path: Path, idx_path: Path
    ):
        """
        Serialize SFT dataset.

        For SFT, we apply chat template and store masking ranges for user prompts.
        """
        messages_column = dataset_config.messages_column

        # Check if FIM is enabled (read from global config, not per-dataset)
        fim_enabled = self.config.fim_rate > 0

        # Get FIM special token IDs if enabled
        if fim_enabled:
            fim_prefix_id = self._get_token_id(self.config.fim_prefix_token)
            fim_suffix_id = self._get_token_id(self.config.fim_suffix_token)
            fim_middle_id = self._get_token_id(self.config.fim_middle_token)
            rng = Random(1337)

            if self.verbose:
                print(
                    f"  FIM enabled: {self.config.fim_rate:.0%} of conversations will be transformed"
                )

        all_tokens = []
        metadata = []
        current_offset = 0

        desc = "  Tokenizing" + (" (FIM)" if fim_enabled else "")
        if self.verbose:
            dataset_iter = tqdm(dataset, desc=desc, unit="convs")
        else:
            dataset_iter = dataset

        for sample in dataset_iter:
            messages = sample[messages_column]

            # Apply chat template and get token IDs + mask ranges
            token_ids, mask_ranges = self._apply_chat_template_and_get_masks(
                messages, dataset_config.chat_template
            )

            # Apply FIM transformation with probability fim_rate
            if fim_enabled and rng.random() < self.config.fim_rate:
                token_ids = self._apply_fim_transformation(
                    token_ids, fim_prefix_id, fim_suffix_id, fim_middle_id, rng
                )
                # FIM sequences don't use standard SFT masking as tokens are reordered
                mask_ranges = []

            # Append to token stream
            all_tokens.extend(token_ids)

            # Record metadata with mask ranges
            metadata.append(
                (
                    current_offset,  # offset
                    len(token_ids),  # length
                    "sft",  # type
                    -1,  # group_id (not used)
                    json.dumps(mask_ranges),  # mask_ranges as JSON
                )
            )

            current_offset += len(token_ids)

        # Save .bin file
        tokens_array = np.array(
            all_tokens, dtype=np.uint16 if max(all_tokens) < 65535 else np.uint32
        )
        with open(bin_path, "wb") as f:
            tokens_array.tofile(f)

        # Save .idx file
        metadata_array = np.array(metadata, dtype=self.METADATA_DTYPE)
        np.save(idx_path, metadata_array)

        if self.verbose:
            print(f"  Tokens: {len(tokens_array):,}")
            print(f"  Conversations: {len(metadata):,}")

    def _serialize_dpo(
        self, dataset, dataset_config: DatasetConfig, bin_path: Path, idx_path: Path
    ):
        """
        Serialize DPO dataset.

        For DPO, we process chosen and rejected responses separately,
        linking them via group_id.
        """
        chosen_column = dataset_config.chosen_column
        rejected_column = dataset_config.rejected_column

        # Check if FIM is enabled (read from global config, not per-dataset)
        fim_enabled = self.config.fim_rate > 0

        # Get FIM special token IDs if enabled
        if fim_enabled:
            fim_prefix_id = self._get_token_id(self.config.fim_prefix_token)
            fim_suffix_id = self._get_token_id(self.config.fim_suffix_token)
            fim_middle_id = self._get_token_id(self.config.fim_middle_token)
            rng = Random(1337)

            if self.verbose:
                print(f"  FIM enabled: {self.config.fim_rate:.0%} of DPO pairs will be transformed")

        all_tokens = []
        metadata = []
        current_offset = 0

        desc = "  Tokenizing" + (" (FIM)" if fim_enabled else "")
        if self.verbose:
            dataset_iter = tqdm(dataset, desc=desc, unit="pairs")
        else:
            dataset_iter = dataset

        for pair_idx, sample in enumerate(dataset_iter):
            # Process chosen response
            chosen_messages = sample[chosen_column]
            chosen_token_ids, chosen_mask_ranges = self._apply_chat_template_and_get_masks(
                chosen_messages, dataset_config.chat_template
            )

            # Process rejected response
            rejected_messages = sample[rejected_column]
            rejected_token_ids, rejected_mask_ranges = self._apply_chat_template_and_get_masks(
                rejected_messages, dataset_config.chat_template
            )

            # Apply FIM transformation to both if enabled (same roll for the pair)
            if fim_enabled and rng.random() < self.config.fim_rate:
                # Transform chosen
                chosen_token_ids = self._apply_fim_transformation(
                    chosen_token_ids, fim_prefix_id, fim_suffix_id, fim_middle_id, rng
                )
                chosen_mask_ranges = []

                # Transform rejected
                rejected_token_ids = self._apply_fim_transformation(
                    rejected_token_ids, fim_prefix_id, fim_suffix_id, fim_middle_id, rng
                )
                rejected_mask_ranges = []

            # Add chosen to stream
            all_tokens.extend(chosen_token_ids)
            metadata.append(
                (
                    current_offset,
                    len(chosen_token_ids),
                    "dpo_chosen",
                    pair_idx,
                    json.dumps(chosen_mask_ranges),
                )
            )
            current_offset += len(chosen_token_ids)

            # Add rejected to stream
            all_tokens.extend(rejected_token_ids)
            metadata.append(
                (
                    current_offset,
                    len(rejected_token_ids),
                    "dpo_rejected",
                    pair_idx,
                    json.dumps(rejected_mask_ranges),
                )
            )
            current_offset += len(rejected_token_ids)

        # Save .bin file
        tokens_array = np.array(
            all_tokens, dtype=np.uint16 if max(all_tokens) < 65535 else np.uint32
        )
        with open(bin_path, "wb") as f:
            tokens_array.tofile(f)

        # Save .idx file
        metadata_array = np.array(metadata, dtype=self.METADATA_DTYPE)
        np.save(idx_path, metadata_array)

        if self.verbose:
            print(f"  Tokens: {len(tokens_array):,}")
            print(f"  Pairs: {len(dataset):,}")

    def _serialize_grpo(
        self, dataset, dataset_config: DatasetConfig, bin_path: Path, idx_path: Path
    ):
        """
        Serialize GRPO dataset.

        For GRPO, we only store prompts. The model generates completions during training.
        Metadata (answer, test_cases, etc.) is stored in mask_ranges as JSON.

        Expected input format (JSON/JSONL):
        {
            "prompt": "Solve: 2x + 3 = 7",
            "answer": "x = 2",
            "type": "math"
        }

        Or for code:
        {
            "prompt": "def fibonacci(n):\\n    ",
            "test_cases": ["assert fib(5)==5"],
            "type": "code"
        }
        """
        all_tokens = []
        metadata = []
        current_offset = 0

        if self.verbose:
            dataset_iter = tqdm(dataset, desc="  Tokenizing prompts", unit="prompts")
        else:
            dataset_iter = dataset

        for sample in dataset_iter:
            # Get prompt
            prompt = sample.get("prompt", "")
            if not prompt:
                continue

            # Tokenize prompt only (no completion - model generates during training)
            token_ids = self._tokenize(prompt)

            # Add EOS token
            token_ids.append(self.tokenizer.eos_token_id)

            # Build metadata dict for reward computation
            sample_metadata = {}
            for key in ["answer", "test_cases", "type", "difficulty", "category"]:
                if key in sample:
                    sample_metadata[key] = sample[key]

            # Append to token stream
            all_tokens.extend(token_ids)

            # Record metadata with sample info in mask_ranges
            metadata.append(
                (
                    current_offset,  # offset
                    len(token_ids),  # length
                    "grpo",  # type
                    -1,  # group_id (not used)
                    json.dumps(sample_metadata),  # metadata as JSON
                )
            )

            current_offset += len(token_ids)

        # Save .bin file
        tokens_array = np.array(
            all_tokens, dtype=np.uint16 if max(all_tokens) < 65535 else np.uint32
        )
        with open(bin_path, "wb") as f:
            tokens_array.tofile(f)

        # Save .idx file
        metadata_array = np.array(metadata, dtype=self.METADATA_DTYPE)
        np.save(idx_path, metadata_array)

        if self.verbose:
            print(f"  Tokens: {len(tokens_array):,}")
            print(f"  Prompts: {len(metadata):,}")

    def _tokenize(self, text: str) -> list[int]:
        """
        Tokenize text using the configured tokenizer.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        # Handle different tokenizer types
        if hasattr(self.tokenizer, "encode"):
            # HuggingFace or tiktoken
            if hasattr(self.tokenizer, "add_special_tokens"):
                # HuggingFace
                return self.tokenizer.encode(text, add_special_tokens=False)
            else:
                # tiktoken
                return self.tokenizer.encode(text)
        else:
            raise ValueError(f"Unsupported tokenizer type: {type(self.tokenizer)}")

    def _apply_chat_template_and_get_masks(
        self, messages: list[dict[str, str]], chat_template: str | None = None
    ) -> tuple[list[int], list[list[int]]]:
        """
        Apply chat template to messages and compute masking ranges.

        Args:
            messages: List of messages [{"role": "user/assistant", "content": "..."}]
            chat_template: Optional custom chat template

        Returns:
            Tuple of (token_ids, mask_ranges)
            mask_ranges: List of [start, end] ranges for user prompts to mask
        """
        # Try to use HuggingFace apply_chat_template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Use tokenizer's built-in chat template
            token_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )

            # Compute mask ranges by tokenizing each message separately
            mask_ranges = []
            current_pos = 0

            for msg in messages:
                msg_tokens = self.tokenizer.apply_chat_template(
                    [msg], tokenize=True, add_generation_prompt=False
                )

                msg_length = len(msg_tokens)

                # Mask user messages (set labels to -100 for these ranges)
                if msg["role"] == "user" or msg["role"] == "system":
                    mask_ranges.append([current_pos, current_pos + msg_length])

                current_pos += msg_length

            return token_ids, mask_ranges

        else:
            # Manual chat template application for tiktoken or custom tokenizers
            # Default template: <|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>

            full_text = ""
            mask_ranges = []
            current_pos = 0

            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                # Simple template (customize based on your needs)
                msg_text = f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"

                msg_tokens = self._tokenize(msg_text)
                msg_length = len(msg_tokens)

                # Mask non-assistant messages
                if role in ["user", "system"]:
                    mask_ranges.append([current_pos, current_pos + msg_length])

                full_text += msg_text
                current_pos += msg_length

            # Final tokenization
            token_ids = self._tokenize(full_text)

            return token_ids, mask_ranges

    def _apply_fim_transformation(
        self,
        token_ids: list[int],
        fim_prefix_id: int,
        fim_suffix_id: int,
        fim_middle_id: int,
        rng: Random,
    ) -> list[int]:
        """
        Apply FIM transformation to token IDs.

        Transforms: [1,2,3,4,5,6,7,8]
        Into: [FP,1,2,FS,6,7,8,FM,3,4,5] where FP=fim_prefix, FS=fim_suffix, FM=fim_middle

        Args:
            token_ids: Original token IDs
            fim_prefix_id: Token ID for <fim_prefix>
            fim_suffix_id: Token ID for <fim_suffix>
            fim_middle_id: Token ID for <fim_middle>
            rng: Random number generator

        Returns:
            Transformed token IDs in PSM format
        """
        length = len(token_ids)

        # Skip if too short (< 10 tokens) for meaningful FIM
        if length < 10:
            return token_ids

        # Random split: choose 2 split points
        split_points = sorted(rng.sample(range(1, length), 2))
        split1, split2 = split_points[0], split_points[1]

        # Ensure different split points (should be guaranteed by sample, but double-check)
        if split1 == split2:
            split2 = min(split1 + 1, length - 1)

        # Split sequence
        prefix = token_ids[:split1]
        middle = token_ids[split1:split2]
        suffix = token_ids[split2:]

        # Handle edge case: empty middle (shouldn't happen with sample, but be safe)
        if not middle:
            middle = [token_ids[split1]]

        # Construct PSM format: [fim_prefix] + prefix + [fim_suffix] + suffix + [fim_middle] + middle
        return [fim_prefix_id] + prefix + [fim_suffix_id] + suffix + [fim_middle_id] + middle

    def _get_token_id(self, token: str) -> int:
        """
        Get token ID from string, with clear error if not found.

        Args:
            token: Token string (e.g., "<fim_prefix>")

        Returns:
            Token ID

        Raises:
            ValueError: If token not found in tokenizer
        """
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            # HuggingFace tokenizer
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            unk_id = getattr(self.tokenizer, "unk_token_id", None)

            # Check if token is unknown/UNK
            if unk_id is not None and token_id == unk_id:
                raise ValueError(
                    f"FIM token '{token}' not found in tokenizer vocabulary. "
                    f"Please add FIM tokens before preprocessing:\n"
                    f"  tokenizer.add_special_tokens({{'additional_special_tokens': "
                    f"['<fim_prefix>', '<fim_suffix>', '<fim_middle>']}})\n"
                    f"  tokenizer.save_pretrained(...)"
                )
            return token_id

        # Fallback for other tokenizer types
        raise TypeError(
            f"Unsupported tokenizer type: {type(self.tokenizer)}. "
            f"Please use a HuggingFace tokenizer for FIM."
        )
