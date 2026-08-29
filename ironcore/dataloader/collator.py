# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Universal Collator for Pretrain and SFT modes.

Implements:
- Simple batching for pretrain
- First-Fit Decreasing bin-packing for SFT
- FlashAttention-compatible outputs (cu_seqlens)
- Fallback to full attention masks
"""

from typing import Literal

import torch


class UniversalCollator:
    """
    Collator supporting both pretrain and SFT modes.

    For pretrain: Simple stacking of sequences
    For SFT: Bin-packing with attention masks and position IDs
    For GRPO: Return prompts with metadata for online generation
    """

    def __init__(
        self,
        mode: Literal["pretrain", "sft", "dpo", "grpo"],
        max_seq_len: int,
        pad_token_id: int = 0,
        use_flash_attention: bool = True,
        return_full_attention_mask: bool = False,
    ):
        """
        Initialize collator.

        Args:
            mode: Training mode
            max_seq_len: Maximum sequence length
            pad_token_id: Padding token ID
            use_flash_attention: Whether to output FlashAttention format
            return_full_attention_mask: Whether to return full attention mask
                                       (for non-FlashAttention models)
        """
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.use_flash_attention = use_flash_attention
        self.return_full_attention_mask = return_full_attention_mask

    def __call__(self, batch: list) -> dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            batch: List of samples from dataset

        Returns:
            Dict with collated tensors
        """
        if self.mode == "pretrain":
            return self._collate_pretrain(batch)
        elif self.mode == "sft":
            return self._collate_sft(batch)
        elif self.mode == "dpo":
            return self._collate_dpo(batch)
        elif self.mode == "grpo":
            return self._collate_grpo(batch)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _collate_pretrain(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Collate pretrain batch.

        Simple stacking since all sequences are already max_seq_len + 1.
        """
        # Stack sequences
        tokens = torch.stack(batch)  # [batch_size, max_seq_len + 1]

        # Split into input_ids and labels
        input_ids = tokens[:, :-1]  # [batch_size, max_seq_len]
        labels = tokens[:, 1:]  # [batch_size, max_seq_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def _collate_sft(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate SFT batch with bin-packing.

        Implements First-Fit Decreasing algorithm:
        1. Sort samples by length (descending)
        2. Pack into bins (sequences) using first-fit
        3. Generate attention masks and position IDs
        """
        # Extract token_ids and metadata
        samples = [(sample["token_ids"], sample["metadata"]) for sample in batch]

        # Sort by length (descending) for better packing
        samples.sort(key=lambda x: len(x[0]), reverse=True)

        # Pre-create range tensor for position_ids (avoids torch.arange in loop)
        position_range = torch.arange(self.max_seq_len, dtype=torch.long)

        # Bin-packing: First-Fit Decreasing
        bins = []  # Each bin: [(token_ids, metadata), ...]
        bin_lengths = []  # Current length of each bin

        for token_ids, metadata in samples:
            sample_len = len(token_ids)

            # Find first bin with enough space
            placed = False
            for i, current_len in enumerate(bin_lengths):
                if current_len + sample_len <= self.max_seq_len:
                    bins[i].append((token_ids, metadata))
                    bin_lengths[i] += sample_len
                    placed = True
                    break

            # If no bin has space, create new bin
            if not placed:
                bins.append([(token_ids, metadata)])
                bin_lengths.append(sample_len)

        # Now construct tensors from bins
        batch_size = len(bins)

        # Initialize tensors
        input_ids = torch.full((batch_size, self.max_seq_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full(
            (batch_size, self.max_seq_len),
            -100,  # Ignore index for loss
            dtype=torch.long,
        )
        position_ids = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long)

        # For FlashAttention: cumulative sequence lengths
        cu_seqlens_list = []

        # For full attention mask (fallback)
        if self.return_full_attention_mask:
            attention_mask = torch.zeros(
                (batch_size, self.max_seq_len, self.max_seq_len), dtype=torch.bool
            )

        # Fill tensors
        for batch_idx, bin_samples in enumerate(bins):
            current_pos = 0
            cu_seqlens = [0]  # Start of first sequence

            for token_ids, metadata in bin_samples:
                sample_len = len(token_ids)
                mask_ranges = metadata.get("mask_ranges", [])

                # Truncate if sample exceeds remaining space in this row.
                # position_ids needs sample_len slots; input_ids/labels need sample_len-1.
                # So the binding constraint is sample_len <= max_seq_len - current_pos.
                available = self.max_seq_len - current_pos
                if sample_len > available:
                    token_ids = token_ids[:available]
                    sample_len = available

                # written_len: number of (input, label) pairs written for this
                # sample = sample_len - 1. ALL per-sample bookkeeping below
                # (position_ids, cu_seqlens, attention mask block, current_pos)
                # must advance by written_len, not sample_len, otherwise a PAD
                # slot is left inside the sample's attention block with a valid
                # position id — confusing FlashAttention and inflating cu_seqlens.
                # (Fable issue #63.)
                written_len = sample_len - 1

                # Copy tokens
                input_ids[batch_idx, current_pos : current_pos + written_len] = token_ids[:-1]
                labels[batch_idx, current_pos : current_pos + written_len] = token_ids[1:]

                # Apply masking for user/system prompt tokens. mask_ranges are
                # token-space indices, but labels are shifted by one relative to
                # input_ids (labels[j] = token_ids[j+1]: predicting token j+1 from
                # position j), so the label-space window is [start-1, end-1),
                # clamped to this sample's own span.
                for start, end in mask_ranges:
                    mask_start = current_pos + max(start - 1, 0)
                    mask_end = current_pos + max(min(end, sample_len) - 1, 0)
                    labels[batch_idx, mask_start:mask_end] = -100

                # Position IDs reset for each sample — advance by written_len.
                position_ids[batch_idx, current_pos : current_pos + written_len] = position_range[
                    :written_len
                ]

                # Block-diagonal attention mask
                if self.return_full_attention_mask:
                    # This sample attends only to itself
                    sample_end = current_pos + written_len
                    attention_mask[batch_idx, current_pos:sample_end, current_pos:sample_end] = True

                # Update cumulative sequence lengths — advance by written_len.
                current_pos += written_len
                cu_seqlens.append(current_pos)

            cu_seqlens_list.append(torch.tensor(cu_seqlens, dtype=torch.int32))

        # Prepare output dict
        output = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
        }

        if self.use_flash_attention:
            # FlashAttention format: list of cu_seqlens per batch element
            output["cu_seqlens"] = cu_seqlens_list

        if self.return_full_attention_mask:
            output["attention_mask"] = attention_mask

        return output

    def _collate_grpo(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate GRPO batch.

        For GRPO, we only need prompts with their metadata.
        The model generates completions during training.

        Returns:
            Dict with:
            - input_ids: [batch_size, prompt_len] tokenized prompts
            - attention_mask: [batch_size, prompt_len] mask for valid tokens
            - metadata: list of dicts with answer/test_cases/type/etc.
        """
        # Extract prompts and metadata
        prompts = []
        metadata_list = []

        for sample in batch:
            token_ids = sample["token_ids"]
            meta = sample.get("metadata", {})

            # Convert to list if tensor
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()

            # Pad to max_seq_len
            if len(token_ids) < self.max_seq_len:
                token_ids = token_ids + [self.pad_token_id] * (self.max_seq_len - len(token_ids))
            else:
                token_ids = token_ids[: self.max_seq_len]

            prompts.append(token_ids)
            metadata_list.append(meta)

        # Stack prompts
        input_ids = torch.tensor(prompts, dtype=torch.long)

        # Create attention mask (1 for valid tokens, 0 for padding)
        # Use the original lengths before padding
        attention_mask = torch.zeros_like(input_ids)
        for i, sample in enumerate(batch):
            orig_token_ids = sample["token_ids"]
            if isinstance(orig_token_ids, torch.Tensor):
                orig_token_ids = orig_token_ids.tolist()
            original_len = min(len(orig_token_ids), self.max_seq_len)
            attention_mask[i, :original_len] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "metadata": metadata_list,
        }

    def _collate_dpo(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate DPO batch.

        Groups chosen/rejected pairs and returns separate tensors, with row i of
        chosen_* guaranteed to correspond to row i of rejected_* (both sides are
        sorted by group_id, the pairing key the serializer stamps onto every
        dpo_chosen/dpo_rejected sample).

        Each side is collated per-row (one sample per row, right-padded) rather
        than bin-packed: dpo_loss pairs chosen/rejected by row index, and
        bin-packing (as used for SFT) can both reorder rows independently on
        each side and merge multiple samples into one row, which breaks that
        row-index correspondence.

        Raises:
            ValueError: If batch doesn't contain paired chosen/rejected samples
        """
        # Separate chosen and rejected
        chosen_samples = [s for s in batch if s["metadata"]["type"] == "dpo_chosen"]
        rejected_samples = [s for s in batch if s["metadata"]["type"] == "dpo_rejected"]

        # Extract metadata for better error messages
        all_sample_types = [s["metadata"].get("type", "unknown") for s in batch]
        type_counts = {}
        for sample_type in all_sample_types:
            type_counts[sample_type] = type_counts.get(sample_type, 0) + 1

        # Validation: DPO requires paired data
        if len(chosen_samples) == 0:
            raise ValueError(
                f"DPO batch contains no chosen samples (dpo_chosen). "
                f"Batch contains {len(batch)} total samples with types: {type_counts}. "
                f"Check data pipeline: all pairs must have 'dpo_chosen' type."
            )
        if len(rejected_samples) == 0:
            raise ValueError(
                f"DPO batch contains no rejected samples (dpo_rejected). "
                f"Batch contains {len(batch)} total samples with types: {type_counts}. "
                f"Check data pipeline: all pairs must have 'dpo_rejected' type."
            )
        if len(chosen_samples) != len(rejected_samples):
            raise ValueError(
                f"DPO batch has mismatched pairs: {len(chosen_samples)} chosen vs {len(rejected_samples)} rejected. "
                f"Batch composition: {type_counts}. "
                f"Each chosen sample must have a corresponding rejected pair. "
                f"Total samples in batch: {len(batch)}"
            )

        # Sort both sides by group_id so row i is the same preference pair on
        # both sides, regardless of the order samples arrived in this batch.
        missing_group_id = [s for s in batch if s["metadata"].get("group_id") is None]
        if missing_group_id:
            raise ValueError(
                f"{len(missing_group_id)} DPO sample(s) are missing 'group_id' in metadata, "
                f"which is required to pair chosen/rejected rows correctly. "
                f"The serializer stamps group_id on every dpo_chosen/dpo_rejected sample; "
                f"check the data pipeline that produced this batch."
            )

        chosen_samples.sort(key=lambda s: s["metadata"]["group_id"])
        rejected_samples.sort(key=lambda s: s["metadata"]["group_id"])

        chosen_group_ids = [s["metadata"]["group_id"] for s in chosen_samples]
        rejected_group_ids = [s["metadata"]["group_id"] for s in rejected_samples]
        if chosen_group_ids != rejected_group_ids:
            raise ValueError(
                f"DPO batch chosen/rejected group_ids do not form matching pairs after "
                f"sorting: chosen={chosen_group_ids} vs rejected={rejected_group_ids}. "
                f"Each dpo_chosen sample must have exactly one dpo_rejected sample "
                f"sharing the same group_id."
            )

        chosen_batch = self._collate_dpo_side(chosen_samples)
        rejected_batch = self._collate_dpo_side(rejected_samples)

        # Prefix keys
        output = {}
        for k, v in chosen_batch.items():
            output[f"chosen_{k}"] = v
        for k, v in rejected_batch.items():
            output[f"rejected_{k}"] = v

        return output

    def _collate_dpo_side(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        """Per-row (unpacked) collation for one side of a DPO batch.

        Unlike _collate_sft, this never packs multiple samples into one row and
        never reorders samples — the caller controls row order via group_id so
        chosen/rejected stay aligned by index.
        """
        batch_size = len(samples)
        position_range = torch.arange(self.max_seq_len, dtype=torch.long)

        input_ids = torch.full((batch_size, self.max_seq_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, self.max_seq_len), -100, dtype=torch.long)
        position_ids = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long)

        for row, sample in enumerate(samples):
            token_ids = sample["token_ids"]
            mask_ranges = sample["metadata"].get("mask_ranges", [])

            sample_len = len(token_ids)
            if sample_len > self.max_seq_len:
                token_ids = token_ids[: self.max_seq_len]
                sample_len = self.max_seq_len

            input_ids[row, : sample_len - 1] = torch.as_tensor(token_ids[:-1], dtype=torch.long)
            labels[row, : sample_len - 1] = torch.as_tensor(token_ids[1:], dtype=torch.long)

            # See _collate_sft for why the label-space window is [start-1, end-1).
            for start, end in mask_ranges:
                mask_start = max(start - 1, 0)
                mask_end = max(min(end, sample_len) - 1, 0)
                labels[row, mask_start:mask_end] = -100

            position_ids[row, :sample_len] = position_range[:sample_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
        }


def create_collator(
    mode: Literal["pretrain", "sft", "dpo", "grpo"],
    max_seq_len: int,
    pad_token_id: int = 0,
    **kwargs,
) -> UniversalCollator:
    """
    Factory function to create collator.

    Args:
        mode: Training mode
        max_seq_len: Maximum sequence length
        pad_token_id: Padding token ID
        **kwargs: Additional arguments passed to UniversalCollator

    Returns:
        UniversalCollator instance
    """
    return UniversalCollator(
        mode=mode, max_seq_len=max_seq_len, pad_token_id=pad_token_id, **kwargs
    )
