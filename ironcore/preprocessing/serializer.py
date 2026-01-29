"""
Universal Data Serializer

Handles downloading, tokenizing, and serializing datasets into a unified binary format.
Supports pretrain, SFT, and DPO task types.
"""

from typing import Dict, List, Optional, Tuple, Union

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
    METADATA_DTYPE = np.dtype([
        ('offset', np.uint64),      # Byte offset in .bin file
        ('length', np.uint32),       # Number of tokens
        ('type', 'U20'),             # Task type string
        ('group_id', np.int64),      # For DPO pairing (-1 = not paired)
        ('mask_ranges', 'U500'),     # JSON string of masking ranges
    ])

    def __init__(
        self,
        data_config: DataConfig,
        tokenizer,
        verbose: bool = True
    ):
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
            print(f"\n{'='*60}")
            print(f"Starting serialization of {len(self.config.datasets)} dataset(s)")
            print(f"{'='*60}\n")

        for dataset_config in self.config.datasets:
            self.serialize_dataset(dataset_config)

        if self.verbose:
            print(f"\n{'='*60}")
            print("Serialization complete!")
            print(f"{'='*60}\n")

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
                print(f"  ⚠️  Already processed. Skipping...")
            return

        # Load dataset
        dataset = self._load_dataset(dataset_config)

        if self.verbose:
            print(f"  Loaded {len(dataset)} samples")

        # Serialize based on task type
        if dataset_config.task_type == "pretrain":
            self._serialize_pretrain(dataset, dataset_config, bin_path, idx_path)
        elif dataset_config.task_type == "sft":
            self._serialize_sft(dataset, dataset_config, bin_path, idx_path)
        elif dataset_config.task_type == "dpo":
            self._serialize_dpo(dataset, dataset_config, bin_path, idx_path)
        else:
            raise ValueError(f"Unknown task type: {dataset_config.task_type}")

        if self.verbose:
            print(f"  ✓ Serialization complete")

    def _load_dataset(self, dataset_config: DatasetConfig):
        """Load dataset from HuggingFace or local source."""
        if Path(dataset_config.source).exists():
            # Local dataset
            if dataset_config.source.endswith('.json') or dataset_config.source.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=dataset_config.source, split='train')
            else:
                dataset = load_dataset(dataset_config.source, split=dataset_config.split)
        else:
            # HuggingFace dataset
            if dataset_config.subset:
                dataset = load_dataset(
                    dataset_config.source,
                    dataset_config.subset,
                    split=dataset_config.split,
                    cache_dir=str(self.config.cache_dir)
                )
            else:
                dataset = load_dataset(
                    dataset_config.source,
                    split=dataset_config.split,
                    cache_dir=str(self.config.cache_dir)
                )

        # Limit samples if specified
        if dataset_config.max_samples:
            dataset = dataset.select(range(min(dataset_config.max_samples, len(dataset))))

        return dataset

    def _serialize_pretrain(
        self,
        dataset,
        dataset_config: DatasetConfig,
        bin_path: Path,
        idx_path: Path
    ):
        """
        Serialize pretrain dataset.

        For pretrain, we simply tokenize raw text and append to .bin.
        No masking metadata needed.
        """
        text_column = dataset_config.text_column

        # Open binary file for writing
        all_tokens = []
        metadata = []
        current_offset = 0

        if self.verbose:
            dataset_iter = tqdm(dataset, desc="  Tokenizing", unit="docs")
        else:
            dataset_iter = dataset

        for sample in dataset_iter:
            text = sample[text_column]

            # Tokenize
            token_ids = self._tokenize(text)

            # Add EOS token
            token_ids.append(self.tokenizer.eos_token_id)

            # Append to token stream
            all_tokens.extend(token_ids)

            # Record metadata
            metadata.append((
                current_offset,          # offset
                len(token_ids),          # length
                'pretrain',              # type
                -1,                      # group_id (not used)
                '[]'                     # mask_ranges (empty)
            ))

            current_offset += len(token_ids)

        # Save .bin file
        tokens_array = np.array(all_tokens, dtype=np.uint16 if max(all_tokens) < 65535 else np.uint32)
        with open(bin_path, 'wb') as f:
            tokens_array.tofile(f)

        # Save .idx file
        metadata_array = np.array(metadata, dtype=self.METADATA_DTYPE)
        np.save(idx_path, metadata_array)

        if self.verbose:
            print(f"  Tokens: {len(tokens_array):,}")
            print(f"  Documents: {len(metadata):,}")

    def _serialize_sft(
        self,
        dataset,
        dataset_config: DatasetConfig,
        bin_path: Path,
        idx_path: Path
    ):
        """
        Serialize SFT dataset.

        For SFT, we apply chat template and store masking ranges for user prompts.
        """
        messages_column = dataset_config.messages_column

        all_tokens = []
        metadata = []
        current_offset = 0

        if self.verbose:
            dataset_iter = tqdm(dataset, desc="  Tokenizing", unit="convs")
        else:
            dataset_iter = dataset

        for sample in dataset_iter:
            messages = sample[messages_column]

            # Apply chat template and get token IDs + mask ranges
            token_ids, mask_ranges = self._apply_chat_template_and_get_masks(
                messages,
                dataset_config.chat_template
            )

            # Append to token stream
            all_tokens.extend(token_ids)

            # Record metadata with mask ranges
            metadata.append((
                current_offset,                    # offset
                len(token_ids),                    # length
                'sft',                             # type
                -1,                                # group_id (not used)
                json.dumps(mask_ranges)            # mask_ranges as JSON
            ))

            current_offset += len(token_ids)

        # Save .bin file
        tokens_array = np.array(all_tokens, dtype=np.uint16 if max(all_tokens) < 65535 else np.uint32)
        with open(bin_path, 'wb') as f:
            tokens_array.tofile(f)

        # Save .idx file
        metadata_array = np.array(metadata, dtype=self.METADATA_DTYPE)
        np.save(idx_path, metadata_array)

        if self.verbose:
            print(f"  Tokens: {len(tokens_array):,}")
            print(f"  Conversations: {len(metadata):,}")

    def _serialize_dpo(
        self,
        dataset,
        dataset_config: DatasetConfig,
        bin_path: Path,
        idx_path: Path
    ):
        """
        Serialize DPO dataset.

        For DPO, we process chosen and rejected responses separately,
        linking them via group_id.
        """
        chosen_column = dataset_config.chosen_column
        rejected_column = dataset_config.rejected_column

        all_tokens = []
        metadata = []
        current_offset = 0

        if self.verbose:
            dataset_iter = tqdm(dataset, desc="  Tokenizing", unit="pairs")
        else:
            dataset_iter = dataset

        for pair_idx, sample in enumerate(dataset_iter):
            # Process chosen response
            chosen_messages = sample[chosen_column]
            chosen_token_ids, chosen_mask_ranges = self._apply_chat_template_and_get_masks(
                chosen_messages,
                dataset_config.chat_template
            )

            all_tokens.extend(chosen_token_ids)
            metadata.append((
                current_offset,
                len(chosen_token_ids),
                'dpo_chosen',
                pair_idx,                          # group_id links chosen/rejected
                json.dumps(chosen_mask_ranges)
            ))
            current_offset += len(chosen_token_ids)

            # Process rejected response
            rejected_messages = sample[rejected_column]
            rejected_token_ids, rejected_mask_ranges = self._apply_chat_template_and_get_masks(
                rejected_messages,
                dataset_config.chat_template
            )

            all_tokens.extend(rejected_token_ids)
            metadata.append((
                current_offset,
                len(rejected_token_ids),
                'dpo_rejected',
                pair_idx,                          # Same group_id
                json.dumps(rejected_mask_ranges)
            ))
            current_offset += len(rejected_token_ids)

        # Save .bin file
        tokens_array = np.array(all_tokens, dtype=np.uint16 if max(all_tokens) < 65535 else np.uint32)
        with open(bin_path, 'wb') as f:
            tokens_array.tofile(f)

        # Save .idx file
        metadata_array = np.array(metadata, dtype=self.METADATA_DTYPE)
        np.save(idx_path, metadata_array)

        if self.verbose:
            print(f"  Tokens: {len(tokens_array):,}")
            print(f"  Pairs: {len(dataset):,}")

    def _tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using the configured tokenizer.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        # Handle different tokenizer types
        if hasattr(self.tokenizer, 'encode'):
            # HuggingFace or tiktoken
            if hasattr(self.tokenizer, 'add_special_tokens'):
                # HuggingFace
                return self.tokenizer.encode(text, add_special_tokens=False)
            else:
                # tiktoken
                return self.tokenizer.encode(text)
        else:
            raise ValueError(f"Unsupported tokenizer type: {type(self.tokenizer)}")

    def _apply_chat_template_and_get_masks(
        self,
        messages: List[Dict[str, str]],
        chat_template: Optional[str] = None
    ) -> Tuple[List[int], List[List[int]]]:
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
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Use tokenizer's built-in chat template
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False
            )

            # Compute mask ranges by tokenizing each message separately
            mask_ranges = []
            current_pos = 0

            for msg in messages:
                msg_tokens = self.tokenizer.apply_chat_template(
                    [msg],
                    tokenize=True,
                    add_generation_prompt=False
                )

                msg_length = len(msg_tokens)

                # Mask user messages (set labels to -100 for these ranges)
                if msg['role'] == 'user' or msg['role'] == 'system':
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
                role = msg['role']
                content = msg['content']

                # Simple template (customize based on your needs)
                msg_text = f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"

                msg_tokens = self._tokenize(msg_text)
                msg_length = len(msg_tokens)

                # Mask non-assistant messages
                if role in ['user', 'system']:
                    mask_ranges.append([current_pos, current_pos + msg_length])

                full_text += msg_text
                current_pos += msg_length

            # Final tokenization
            token_ids = self._tokenize(full_text)

            return token_ids, mask_ranges
