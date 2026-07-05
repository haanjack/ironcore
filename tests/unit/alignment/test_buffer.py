# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ironcore/alignment/buffer.py.

Regression coverage: RolloutBuffer.group_size used to be computed as
len(metadata) // batch_size, where batch_size reads prompt_ids.size(0).
select() deliberately keeps prompt_ids as the original full prompt set (not
sliced), so after select() that ratio no longer reflects the actual
completions-per-group in the returned sub-buffer.
"""

import torch

from ironcore.alignment.buffer import RolloutBuffer


def _make_buffer(batch_size: int, group_size: int) -> RolloutBuffer:
    total = batch_size * group_size
    prompt_len, resp_len = 4, 6

    group_ids = torch.arange(batch_size).repeat_interleave(group_size)

    return RolloutBuffer(
        prompt_ids=torch.zeros(batch_size, prompt_len, dtype=torch.long),
        prompt_attention_mask=torch.ones(batch_size, prompt_len, dtype=torch.long),
        completion_ids=torch.zeros(total, prompt_len + resp_len, dtype=torch.long),
        response_ids=torch.zeros(total, resp_len, dtype=torch.long),
        old_log_probs=torch.zeros(total),
        rewards=torch.zeros(total),
        advantages=torch.zeros(total),
        group_ids=group_ids,
        metadata=[{"idx": i} for i in range(total)],
    )


class TestRolloutBufferGroupSize:
    def test_group_size_on_full_buffer(self):
        buf = _make_buffer(batch_size=4, group_size=8)
        assert buf.group_size == 8
        assert buf.batch_size == 4
        assert buf.total_samples == 32

    def test_group_size_after_select_one_full_group(self):
        """Selecting exactly one prompt's worth of completions should report
        that group's actual size, not a ratio against the untouched
        original batch_size."""
        buf = _make_buffer(batch_size=4, group_size=8)

        # Select all 8 completions belonging to group 0 (indices 0..7).
        indices = torch.arange(0, 8)
        sub = buf.select(indices)

        assert sub.total_samples == 8
        assert sub.group_size == 8

    def test_group_size_after_select_multiple_groups(self):
        buf = _make_buffer(batch_size=4, group_size=8)

        # Select all completions for groups 0 and 1 (16 samples, 2 groups).
        indices = torch.arange(0, 16)
        sub = buf.select(indices)

        assert sub.total_samples == 16
        assert sub.group_size == 8  # 16 samples / 2 groups = 8 per group

    def test_select_preserves_original_prompt_ids(self):
        """select() intentionally does not slice prompt_ids; batch_size stays
        the original prompt count even for a partial selection."""
        buf = _make_buffer(batch_size=4, group_size=8)
        sub = buf.select(torch.arange(0, 8))

        assert sub.batch_size == buf.batch_size == 4

    def test_summary_does_not_raise_after_select(self):
        buf = _make_buffer(batch_size=4, group_size=8)
        sub = buf.select(torch.arange(0, 8))
        summary = sub.summary()
        assert summary["group_size"] == 8
