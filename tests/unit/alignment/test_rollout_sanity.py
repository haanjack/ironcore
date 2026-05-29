# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Sanity tests for GRPO rollout output correctness.

Smoke tests only check crash-free completion.  These tests verify three
properties the smoke tests miss:

1. paged vs batched numerical equivalence — same prompt+model+greedy decoding
   must produce identical old_log_probs and generated tokens.
2. old_log_probs are non-positive and finite — each per-token log prob is a
   log_softmax value (≤ 0), so the summed sequence log prob must be ≤ 0 with
   no NaN / overflow.
3. response_lengths matches EOS positions — for sequences that generated an EOS
   token, completion_ids[i, prompt_len + response_lengths[i] - 1] must equal
   eos_token_id.
"""

from typing import cast

import pytest
import torch
from torch import nn

from ironcore.alignment.rollout import generate_rollouts_batched, generate_rollouts_paged

# ── Mock infrastructure ────────────────────────────────────────────────────────


class _MockBlockKVCache:
    """No-op block KV cache that satisfies the generate_rollouts_paged interface."""

    def __init__(self, max_seqs: int = 64, block_size: int = 16) -> None:
        self.block_size = block_size
        self.is_initialized = True
        self.token_positions = torch.zeros(max_seqs, dtype=torch.long)

    def allocate_blocks(self, seq_id: int, count: int) -> None:  # noqa: ARG002
        pass

    def advance_position(self, seq_id: int, tokens: int) -> None:
        self.token_positions[seq_id] += tokens

    def advance_positions_batched(self, seq_ids: list[int], tokens: int) -> None:
        idx = torch.tensor(seq_ids, dtype=torch.long)
        self.token_positions[idx] += tokens

    def share_prefix(self, src: int, dsts: list[int]) -> None:
        for dst in dsts:
            self.token_positions[dst] = self.token_positions[src].clone()

    def free_sequence(self, seq_id: int) -> None:
        self.token_positions[seq_id] = 0


class _MockModel(nn.Module):
    """Deterministic context-free model for rollout sanity tests.

    Logits depend only on the *current* input token — no attention over past
    context.  With greedy decoding (do_sample=False) this makes both paged
    and batched paths produce identical token sequences and log probs, enabling
    a clean numerical equivalence check.
    """

    VOCAB_SIZE = 32

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.register_buffer("_logit_table", torch.randn(self.VOCAB_SIZE, self.VOCAB_SIZE))
        self.block_kv_cache_manager = _MockBlockKVCache()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels=None,  # noqa: ARG002
        use_cache: bool = False,
        past_key_values=None,  # noqa: ARG002
        seq_id=None,  # noqa: ARG002
    ):
        B, T = input_ids.shape
        table = cast(torch.Tensor, self._logit_table)
        logits = table[input_ids.view(-1)].view(B, T, self.VOCAB_SIZE)
        if use_cache:
            return logits, []
        return logits, None

    def share_prefix_cache(self, src: int, dsts: list[int]) -> None:
        self.block_kv_cache_manager.share_prefix(src, dsts)

    def free_sequence_cache(self, seq_id: int) -> None:
        self.block_kv_cache_manager.free_sequence(seq_id)

    def advance_cache_position(self, seq_ids, tokens: int) -> None:
        if isinstance(seq_ids, list):
            self.block_kv_cache_manager.advance_positions_batched(seq_ids, tokens)
        else:
            self.block_kv_cache_manager.advance_position(seq_ids, tokens)


def _make_eos_model(eos_token_id: int, trigger_token: int) -> _MockModel:
    """Return a MockModel whose logit table guarantees EOS after one hop.

    Path: trigger_token → relay_token → eos_token_id.
    Any prompt ending with trigger_token will generate EOS at step 2.
    """
    model = _MockModel()
    relay_token = (trigger_token + 1) % _MockModel.VOCAB_SIZE
    # Ensure relay != eos
    if relay_token == eos_token_id:
        relay_token = (relay_token + 1) % _MockModel.VOCAB_SIZE

    with torch.no_grad():
        table = cast(torch.Tensor, model._logit_table)
        table[trigger_token].fill_(-10.0)
        table[trigger_token][relay_token] = 10.0
        table[relay_token].fill_(-10.0)
        table[relay_token][eos_token_id] = 10.0

    return model


# ── Test classes ───────────────────────────────────────────────────────────────


class TestLogProbRange:
    """old_log_probs must be non-positive and finite after generation."""

    def _run_batched(self, max_new_tokens: int = 8) -> torch.Tensor:
        model = _MockModel()
        model.eval()
        torch.manual_seed(0)
        B, prompt_len = 2, 4
        prompt_ids = torch.randint(0, _MockModel.VOCAB_SIZE, (B, prompt_len))
        buf = generate_rollouts_batched(
            model,
            prompt_ids,
            group_size=2,
            metadata=[{} for _ in range(B)],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return buf.old_log_probs

    def test_old_log_probs_nonpositive(self):
        log_probs = self._run_batched()
        assert (log_probs <= 0).all(), (
            f"old_log_probs must be ≤ 0; got max={log_probs.max().item():.4f}"
        )

    def test_old_log_probs_finite(self):
        log_probs = self._run_batched()
        assert log_probs.isfinite().all(), f"old_log_probs must be finite; got {log_probs}"

    def test_old_log_probs_not_extreme(self):
        """Sum of log probs must not underflow to -inf territory."""
        log_probs = self._run_batched(max_new_tokens=8)
        lower_bound = -8 * torch.log(torch.tensor(float(_MockModel.VOCAB_SIZE))) - 1.0
        assert (log_probs >= lower_bound).all(), (
            f"old_log_probs={log_probs} below expected lower bound {lower_bound:.2f}"
        )

    def test_no_nan_in_log_probs(self):
        log_probs = self._run_batched()
        assert not log_probs.isnan().any(), "old_log_probs must not contain NaN"


class TestResponseLengthsConsistency:
    """For sequences that hit EOS, completion_ids must contain EOS at
    the position indicated by response_lengths."""

    EOS_ID = 5
    TRIGGER_TOKEN = 9

    def _run(self, group_size: int = 2) -> tuple:
        model = _make_eos_model(self.EOS_ID, self.TRIGGER_TOKEN)
        model.eval()
        B, prompt_len = 2, 3
        # Prompt 0 ends with TRIGGER_TOKEN → will generate EOS after 2 tokens
        # Prompt 1 ends with a different token that does not trigger EOS
        other_token = (self.TRIGGER_TOKEN + 5) % _MockModel.VOCAB_SIZE
        if other_token == self.EOS_ID:
            other_token = (other_token + 1) % _MockModel.VOCAB_SIZE
        prompt_ids = torch.tensor(
            [
                [1, 2, self.TRIGGER_TOKEN],
                [1, 2, other_token],
            ]
        )
        buf = generate_rollouts_batched(
            model,
            prompt_ids,
            group_size=group_size,
            metadata=[{} for _ in range(B)],
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=self.EOS_ID,
        )
        return buf, prompt_len

    def test_eos_sequences_have_response_lengths_set(self):
        buf, prompt_len = self._run()
        # Sequences from prompt 0 (indices 0 and 1 with G=2) should hit EOS.
        # response_lengths[i] < max_new_tokens means EOS was generated.
        for i in range(2):  # G=2 completions from prompt 0
            rl = buf.response_lengths[i].item()
            assert rl < 6, f"Sequence {i} should have hit EOS but response_lengths={rl}"

    def test_eos_at_response_length_position(self):
        buf, prompt_len = self._run()
        # For every sequence that hit EOS, the token at
        # completion_ids[i, prompt_len + response_lengths[i] - 1] must be EOS.
        for i in range(buf.total_samples):
            rl = buf.response_lengths[i].item()
            if rl < 6:  # EOS was generated
                eos_pos = prompt_len + rl - 1
                token_at_pos = buf.completion_ids[i, eos_pos].item()
                assert token_at_pos == self.EOS_ID, (
                    f"Sequence {i}: expected EOS ({self.EOS_ID}) at position {eos_pos}, "
                    f"got {token_at_pos}. completion_ids={buf.completion_ids[i].tolist()}, "
                    f"response_lengths={rl}"
                )

    def test_response_lengths_within_bounds(self):
        buf, _ = self._run()
        max_len = buf.response_ids.size(1)
        assert (buf.response_lengths >= 1).all(), "response_lengths must be ≥ 1"
        assert (buf.response_lengths <= max_len).all(), f"response_lengths must be ≤ {max_len}"

    def test_no_eos_means_max_length(self):
        """Sequences from the non-triggering prompt should use max_new_tokens."""
        buf, _ = self._run(group_size=1)
        # With G=1: index 0 = prompt 0 (hits EOS), index 1 = prompt 1 (no EOS)
        no_eos_rl = buf.response_lengths[1].item()
        gen_len = buf.response_ids.size(1)
        assert no_eos_rl == gen_len, (
            f"Non-EOS sequence should have response_lengths={gen_len}, got {no_eos_rl}"
        )


class TestPagedVsBatchedEquivalence:
    """Paged and batched rollouts must produce numerically identical outputs
    under greedy decoding with the same context-free model."""

    def _run_both(  # noqa: N803
        self,
        B: int = 2,  # noqa: N803
        G: int = 2,  # noqa: N803
        max_new_tokens: int = 6,
    ):
        torch.manual_seed(7)
        prompt_ids = torch.randint(1, _MockModel.VOCAB_SIZE, (B, 4))
        metadata = [{} for _ in range(B)]

        batched_model = _MockModel()
        batched_model.eval()
        buf_batched = generate_rollouts_batched(
            batched_model,
            prompt_ids,
            G,
            metadata,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        paged_model = _MockModel()  # fresh instance, same seed → same _logit_table
        paged_model.eval()
        buf_paged = generate_rollouts_paged(
            paged_model,
            prompt_ids,
            G,
            metadata,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        return buf_batched, buf_paged

    def test_generated_tokens_identical(self):
        bb, bp = self._run_both()
        assert torch.equal(bb.response_ids, bp.response_ids), (
            f"Greedy-decoded tokens differ between paged and batched.\n"
            f"Batched:\n{bb.response_ids}\nPaged:\n{bp.response_ids}"
        )

    def test_old_log_probs_identical(self):
        bb, bp = self._run_both()
        assert torch.allclose(bb.old_log_probs, bp.old_log_probs, atol=1e-5), (
            f"old_log_probs differ between paged and batched.\n"
            f"Batched: {bb.old_log_probs}\nPaged:   {bp.old_log_probs}\n"
            f"Max diff: {(bb.old_log_probs - bp.old_log_probs).abs().max().item():.2e}"
        )

    def test_completion_ids_identical(self):
        bb, bp = self._run_both()
        assert torch.equal(bb.completion_ids, bp.completion_ids), (
            "completion_ids (prompt + generated) differ between paged and batched"
        )

    def test_shapes_consistent(self):
        bb, bp = self._run_both(B=2, G=3, max_new_tokens=4)
        assert bb.old_log_probs.shape == bp.old_log_probs.shape
        assert bb.response_ids.shape == bp.response_ids.shape
        assert bb.completion_ids.shape == bp.completion_ids.shape


class TestDoneSequenceBlockUsage:
    """Verify done sequences don't waste block pool capacity.

    PR #37 (RAM-host optimizer states / CPU offloading) will implement
    FSDP-compatible offloading and needs to verify that FSDP-wrapped
    models correctly handle done sequences during paged rollout.
    """

    @pytest.mark.skip(reason="Requires FSDP + offloading from PR #37")
    def test_done_sequences_no_extra_blocks_under_fsdp(self):
        """Done sequences should not allocate new blocks beyond their last token.

        When FSDP is active, the full batch must still be forwarded for shape
        consistency, but block allocation should be skipped for done sequences.
        This test verifies the offloading-aware path in PR #37.
        """
