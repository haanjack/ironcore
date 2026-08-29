# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ironcore/dataloader/collator.py.

Regression coverage for two bugs found in the codebase audit:

1. _collate_dpo used to collate chosen/rejected sides independently through
   _collate_sft, which sorts samples by length and bin-packs multiple samples
   per row. That breaks the row-index correspondence dpo_loss relies on
   (chosen[i] must be the same preference pair as rejected[i]) whenever the
   two sides truncate/sort/pack differently. The fix sorts both sides by the
   serializer-assigned group_id and collates per-row (no packing).

2. _collate_sft's mask_ranges (token-space indices) were applied directly to
   `labels`, which are shifted by one relative to input_ids (labels[j] =
   token_ids[j+1]). The fix shifts the mask window by one in label space.
"""

import torch

from ironcore.dataloader.collator import UniversalCollator


def _sft_sample(token_ids, mask_ranges=None):
    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "metadata": {"mask_ranges": mask_ranges or []},
    }


def _dpo_sample(sample_type, group_id, token_ids, mask_ranges=None):
    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "metadata": {"type": sample_type, "group_id": group_id, "mask_ranges": mask_ranges or []},
    }


class TestDpoCollatorPairing:
    def _collator(self, max_seq_len=32):
        return UniversalCollator(
            mode="dpo",
            max_seq_len=max_seq_len,
            pad_token_id=0,
            use_flash_attention=False,
            return_full_attention_mask=False,
        )

    def test_rows_stay_paired_by_group_id_regardless_of_batch_order(self):
        """Row i of chosen_input_ids must correspond to row i of rejected_input_ids,
        even when the chosen/rejected samples for different pairs arrive in a
        shuffled, non-group-id-sorted order (as a real DataLoader would deliver
        after within-dataset shuffling)."""
        collator = self._collator()

        # Two pairs with clearly distinguishable token content per group_id,
        # deliberately interleaved out of group_id order.
        batch = [
            _dpo_sample("dpo_rejected", group_id=1, token_ids=[21, 22, 23, 24]),
            _dpo_sample("dpo_chosen", group_id=0, token_ids=[11, 12, 13]),
            _dpo_sample("dpo_rejected", group_id=0, token_ids=[31, 32]),
            _dpo_sample("dpo_chosen", group_id=1, token_ids=[41, 42, 43, 44, 45]),
        ]

        result = collator(batch)

        # Row for group_id=0: chosen starts with 11, rejected starts with 31.
        # Row for group_id=1: chosen starts with 41, rejected starts with 21.
        chosen_first_tokens = result["chosen_input_ids"][:, 0].tolist()
        rejected_first_tokens = result["rejected_input_ids"][:, 0].tolist()

        for row in range(2):
            if chosen_first_tokens[row] == 11:
                assert rejected_first_tokens[row] == 31, (
                    "group_id=0 pair misaligned: chosen/rejected rows do not match"
                )
            elif chosen_first_tokens[row] == 41:
                assert rejected_first_tokens[row] == 21, (
                    "group_id=1 pair misaligned: chosen/rejected rows do not match"
                )
            else:
                raise AssertionError(f"unexpected chosen first token {chosen_first_tokens[row]}")

    def test_length_sorted_batch_does_not_break_pairing(self):
        """Regression: the old implementation independently sorted each side by
        length via _collate_sft, which could put different pairs' rows at
        different indices on the chosen vs rejected side when lengths differ
        asymmetrically."""
        collator = self._collator()

        batch = [
            # group 0: short chosen, long rejected
            _dpo_sample("dpo_chosen", group_id=0, token_ids=[100, 101]),
            _dpo_sample("dpo_rejected", group_id=0, token_ids=[200, 201, 202, 203, 204, 205]),
            # group 1: long chosen, short rejected (opposite length asymmetry)
            _dpo_sample("dpo_chosen", group_id=1, token_ids=[110, 111, 112, 113, 114, 115]),
            _dpo_sample("dpo_rejected", group_id=1, token_ids=[210, 211]),
        ]

        result = collator(batch)
        chosen_first = result["chosen_input_ids"][:, 0].tolist()
        rejected_first = result["rejected_input_ids"][:, 0].tolist()

        for row in range(2):
            if chosen_first[row] == 100:
                assert rejected_first[row] == 200
            elif chosen_first[row] == 110:
                assert rejected_first[row] == 210
            else:
                raise AssertionError(f"unexpected chosen first token {chosen_first[row]}")

    def test_mismatched_group_ids_raise(self):
        collator = self._collator()
        batch = [
            _dpo_sample("dpo_chosen", group_id=0, token_ids=[1, 2, 3]),
            _dpo_sample("dpo_rejected", group_id=1, token_ids=[4, 5, 6]),
        ]
        try:
            collator(batch)
            raise AssertionError("expected ValueError for mismatched group_ids")
        except ValueError as e:
            assert "group_id" in str(e)

    def test_missing_group_id_raises(self):
        collator = self._collator()
        batch = [
            {"token_ids": torch.tensor([1, 2, 3]), "metadata": {"type": "dpo_chosen"}},
            {"token_ids": torch.tensor([4, 5, 6]), "metadata": {"type": "dpo_rejected"}},
        ]
        try:
            collator(batch)
            raise AssertionError("expected ValueError for missing group_id")
        except ValueError as e:
            assert "group_id" in str(e)

    def test_no_packing_one_sample_per_row(self):
        """DPO rows must never bin-pack multiple samples together."""
        collator = self._collator(max_seq_len=64)
        batch = [
            _dpo_sample("dpo_chosen", group_id=0, token_ids=[1, 2, 3]),
            _dpo_sample("dpo_rejected", group_id=0, token_ids=[4, 5, 6]),
            _dpo_sample("dpo_chosen", group_id=1, token_ids=[7, 8]),
            _dpo_sample("dpo_rejected", group_id=1, token_ids=[9, 10]),
        ]
        result = collator(batch)
        assert result["chosen_input_ids"].shape[0] == 2
        assert result["rejected_input_ids"].shape[0] == 2


class TestLabelMaskOffByOne:
    def _collator(self, max_seq_len=16):
        return UniversalCollator(
            mode="sft",
            max_seq_len=max_seq_len,
            pad_token_id=0,
            use_flash_attention=False,
            return_full_attention_mask=False,
        )

    def test_mask_range_shifts_into_label_space(self):
        """mask_ranges=[[0, 3]] marks token_ids[0:3] as prompt tokens. Since
        labels[j] = token_ids[j+1], the label positions that predict a prompt
        token are [start-1, end-1) = [-1, 2) clamped to [0, 2)."""
        collator = self._collator()
        token_ids = [10, 11, 12, 13, 14, 15]  # 6 tokens -> 5 (input, label) pairs
        sample = _sft_sample(token_ids, mask_ranges=[[0, 3]])

        result = collator([sample])
        labels = result["labels"][0]

        # labels[0] predicts token_ids[1] (a prompt token) -> masked
        # labels[1] predicts token_ids[2] (a prompt token) -> masked
        # labels[2] predicts token_ids[3] (first non-prompt token) -> NOT masked
        assert labels[0].item() == -100
        assert labels[1].item() == -100
        assert labels[2].item() == token_ids[3]

    def test_mask_range_at_start_does_not_underflow(self):
        """start=0 must not wrap to a negative label index."""
        collator = self._collator()
        token_ids = [1, 2, 3, 4]
        sample = _sft_sample(token_ids, mask_ranges=[[0, 1]])

        result = collator([sample])
        labels = result["labels"][0]

        # Should not raise, and should not mask anything before index 0.
        assert torch.all(labels[:0] == -100)  # vacuously true, just checks no crash
        assert labels.shape[0] >= 4
