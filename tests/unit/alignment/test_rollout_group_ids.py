# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for GRPO's distributed advantage-normalization group_id collision.

Root cause (fixed in ironcore/alignment/rollout.py `_build_rollout_output`):
group_ids used to be built as arange(B) on every DP rank, identical
regardless of rank. compute_advantages() all-gathers rewards/group_ids
across the DP group and normalizes per unique group_id, so group 0 from rank
0 and group 0 from rank 1 (different prompts) were pooled together and
normalized as if they were the same group. Fixed by offsetting group_ids by
dp_rank * B so they are globally unique across the DP group.
"""

import torch

from ironcore.alignment import rollout as rollout_module
from ironcore.alignment.rollout import _build_rollout_output


def _call_build_rollout_output(batch_size: int, group_size: int, resp_len: int = 3):
    prompt_len = 2
    prompt_ids = torch.zeros(batch_size, prompt_len, dtype=torch.long)
    total = batch_size * group_size
    generated = torch.zeros(total, resp_len, dtype=torch.long)
    log_probs_list = [torch.zeros(total) for _ in range(resp_len)]
    response_lengths = torch.full((total,), resp_len, dtype=torch.long)
    metadata = [{"i": i} for i in range(batch_size)]

    return _build_rollout_output(
        prompt_ids, generated, log_probs_list, response_lengths, group_size, metadata
    )


class TestRolloutGroupIdOffset:
    def test_group_ids_offset_by_dp_rank(self, monkeypatch):
        monkeypatch.setattr(
            rollout_module.parallel_states, "get_data_parallel_group_rank", lambda: 0
        )
        buf_rank0 = _call_build_rollout_output(batch_size=4, group_size=2)
        assert buf_rank0.group_ids.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]

        monkeypatch.setattr(
            rollout_module.parallel_states, "get_data_parallel_group_rank", lambda: 1
        )
        buf_rank1 = _call_build_rollout_output(batch_size=4, group_size=2)
        assert buf_rank1.group_ids.tolist() == [4, 4, 5, 5, 6, 6, 7, 7]

    def test_rank0_and_rank1_group_ids_never_collide(self, monkeypatch):
        """The actual bug: pre-fix, both ranks produced identical group_ids
        (arange(B)), so a naive concatenation (simulating all_gather) would
        pool different prompts' rewards into the same 'group'."""
        monkeypatch.setattr(
            rollout_module.parallel_states, "get_data_parallel_group_rank", lambda: 0
        )
        buf_rank0 = _call_build_rollout_output(batch_size=3, group_size=2)

        monkeypatch.setattr(
            rollout_module.parallel_states, "get_data_parallel_group_rank", lambda: 1
        )
        buf_rank1 = _call_build_rollout_output(batch_size=3, group_size=2)

        rank0_groups = set(buf_rank0.group_ids.tolist())
        rank1_groups = set(buf_rank1.group_ids.tolist())
        assert rank0_groups.isdisjoint(rank1_groups), (
            "rank 0 and rank 1 produced overlapping group_ids — advantage "
            "normalization would pool unrelated prompts together"
        )
