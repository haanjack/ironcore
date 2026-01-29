"""
Universal Collator for Pretrain and SFT modes.

Implements:
- Simple batching for pretrain
- First-Fit Decreasing bin-packing for SFT
- FlashAttention-compatible outputs (cu_seqlens)
- Fallback to full attention masks
"""

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F


class UniversalCollator:
    """
    Collator supporting both pretrain and SFT modes.

    For pretrain: Simple stacking of sequences
    For SFT: Bin-packing with attention masks and position IDs
    """

    def __init__(
        self,
        mode: Literal["pretrain", "sft", "dpo"],
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

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
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
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _collate_pretrain(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Collate pretrain batch.

        Simple stacking since all sequences are already max_seq_len + 1.
        """
        # Stack sequences
        tokens = torch.stack(batch)  # [batch_size, max_seq_len + 1]

        # Split into input_ids and labels
        input_ids = tokens[:, :-1]   # [batch_size, max_seq_len]
        labels = tokens[:, 1:]        # [batch_size, max_seq_len]

        return {
            'input_ids': input_ids,
            'labels': labels,
        }

    def _collate_sft(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate SFT batch with bin-packing.

        Implements First-Fit Decreasing algorithm:
        1. Sort samples by length (descending)
        2. Pack into bins (sequences) using first-fit
        3. Generate attention masks and position IDs
        """
        # Extract token_ids and metadata
        samples = [
            (sample['token_ids'], sample['metadata'])
            for sample in batch
        ]

        # Sort by length (descending) for better packing
        samples.sort(key=lambda x: len(x[0]), reverse=True)

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
        input_ids = torch.full(
            (batch_size, self.max_seq_len),
            self.pad_token_id,
            dtype=torch.long
        )
        labels = torch.full(
            (batch_size, self.max_seq_len),
            -100,  # Ignore index for loss
            dtype=torch.long
        )
        position_ids = torch.zeros(
            (batch_size, self.max_seq_len),
            dtype=torch.long
        )

        # For FlashAttention: cumulative sequence lengths
        cu_seqlens_list = []

        # For full attention mask (fallback)
        if self.return_full_attention_mask:
            attention_mask = torch.zeros(
                (batch_size, self.max_seq_len, self.max_seq_len),
                dtype=torch.bool
            )

        # Fill tensors
        for batch_idx, bin_samples in enumerate(bins):
            current_pos = 0
            cu_seqlens = [0]  # Start of first sequence

            for token_ids, metadata in bin_samples:
                sample_len = len(token_ids)
                mask_ranges = metadata.get('mask_ranges', [])

                # Copy tokens
                input_ids[batch_idx, current_pos:current_pos + sample_len - 1] = token_ids[:-1]
                labels[batch_idx, current_pos:current_pos + sample_len - 1] = token_ids[1:]

                # Apply masking for user prompts
                for start, end in mask_ranges:
                    # Adjust for position in packed sequence
                    mask_start = current_pos + start
                    mask_end = current_pos + min(end, sample_len)
                    labels[batch_idx, mask_start:mask_end] = -100

                # Position IDs reset for each sample
                position_ids[batch_idx, current_pos:current_pos + sample_len] = torch.arange(sample_len)

                # Block-diagonal attention mask
                if self.return_full_attention_mask:
                    # This sample attends only to itself
                    sample_end = current_pos + sample_len
                    attention_mask[batch_idx, current_pos:sample_end, current_pos:sample_end] = True

                # Update cumulative sequence lengths
                current_pos += sample_len
                cu_seqlens.append(current_pos)

            cu_seqlens_list.append(torch.tensor(cu_seqlens, dtype=torch.int32))

        # Prepare output dict
        output = {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': position_ids,
        }

        if self.use_flash_attention:
            # FlashAttention format: list of cu_seqlens per batch element
            output['cu_seqlens'] = cu_seqlens_list

        if self.return_full_attention_mask:
            output['attention_mask'] = attention_mask

        return output

    def _collate_dpo(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate DPO batch.

        Groups chosen/rejected pairs and returns separate tensors.
        """
        # Separate chosen and rejected
        chosen_samples = [s for s in batch if s['metadata']['type'] == 'dpo_chosen']
        rejected_samples = [s for s in batch if s['metadata']['type'] == 'dpo_rejected']

        # Collate each separately using SFT logic
        chosen_batch = self._collate_sft(chosen_samples) if chosen_samples else None
        rejected_batch = self._collate_sft(rejected_samples) if rejected_samples else None

        # Prefix keys
        output = {}
        if chosen_batch:
            for k, v in chosen_batch.items():
                output[f'chosen_{k}'] = v
        if rejected_batch:
            for k, v in rejected_batch.items():
                output[f'rejected_{k}'] = v

        return output


def create_collator(
    mode: Literal["pretrain", "sft", "dpo"],
    max_seq_len: int,
    pad_token_id: int = 0,
    **kwargs
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
        mode=mode,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
        **kwargs
    )
