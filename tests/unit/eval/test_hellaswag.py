# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ironcore/eval/tasks/hellaswag.py's per-token loss masking.

Regression coverage: HellaSwag._do_predict used to score each candidate
ending with cross_entropy(..., reduction="mean") over the *entire* sequence
(shared prompt + ending + right-padding), rather than the ending tokens
alone. Since HellaSwag's 4 candidate endings for the same example have
different lengths, that dilutes the comparison by whatever fraction of each
candidate's sequence happens to be prompt/padding — making the accuracy
metric unreliable. Fixed by masking to [prompt_len-1, real_length) in
label-space.
"""

import pytest
import torch

from ironcore.eval.tasks.hellaswag import HellaSwag


def _fake_model_with_logits(logits: torch.Tensor):
    """A stand-in for the language model: ignores input_ids, always returns
    the given pre-built logits tensor (as HellaSwag._do_predict expects from
    `model(input_ids, labels=None)`)."""

    def _model(input_ids, labels=None):
        return logits

    return _model


def _build_logits(labels: torch.Tensor, correct_positions: set[int], vocab_size: int = 5):
    """Build [1, seq_len, vocab_size] logits that are confidently *correct*
    (near-zero CE loss) at `correct_positions` and confidently *wrong*
    (high CE loss) everywhere else."""
    seq_len = labels.size(0)
    logits = torch.zeros(1, seq_len, vocab_size)
    for j in range(seq_len):
        target = labels[j].item()
        if j in correct_positions:
            logits[0, j, target] = 20.0
        else:
            logits[0, j, (target + 1) % vocab_size] = 20.0
    return logits


class TestHellaSwagMasking:
    def test_masked_loss_ignores_prompt_and_padding(self):
        # Raw (unshifted) token sequence: prompt=[0,1,2], ending=[3,4,0,1].
        tokens = torch.tensor([0, 1, 2, 3, 4, 0, 1])
        tokenized_inputs = tokens.unsqueeze(0)  # [1, 7]
        labels = tokenized_inputs[:, 1:].squeeze(0)  # [1,2,3,4,0,1], seq_len=6
        prompt_len = 3  # tokens[0:3] is the shared prompt

        # ending_start in label-space = prompt_len - 1 = 2, i.e. positions 2..5.
        logits = _build_logits(labels, correct_positions={2, 3, 4, 5})
        model = _fake_model_with_logits(logits)

        task = HellaSwag.__new__(HellaSwag)
        loss = task._do_predict(
            model,
            tokenized_inputs=tokenized_inputs,
            attention_mask=torch.ones_like(tokenized_inputs),
            prompt_lens=[prompt_len],
        )

        # All positions counted in the loss (2..5) are confidently correct.
        assert loss.item() < 0.1

    def test_unmasked_loss_is_diluted_by_wrong_prompt_positions(self):
        """Same scenario, but without prompt_lens/attention_mask: falls back
        to the old whole-sequence mean, which is dragged up by the
        (deliberately wrong) prompt positions."""
        tokens = torch.tensor([0, 1, 2, 3, 4, 0, 1])
        tokenized_inputs = tokens.unsqueeze(0)
        labels = tokenized_inputs[:, 1:].squeeze(0)
        logits = _build_logits(labels, correct_positions={2, 3, 4, 5})
        model = _fake_model_with_logits(logits)

        task = HellaSwag.__new__(HellaSwag)
        loss_masked = task._do_predict(
            model,
            tokenized_inputs=tokenized_inputs,
            attention_mask=torch.ones_like(tokenized_inputs),
            prompt_lens=[3],
        )
        loss_unmasked = task._do_predict(model, tokenized_inputs=tokenized_inputs)

        assert loss_unmasked.item() > loss_masked.item() + 1.0

    def test_padding_positions_excluded_via_attention_mask(self):
        """Two real tokens followed by padding; the padded label positions
        must not affect the loss even if the model is 'wrong' there."""
        tokens = torch.tensor([0, 1, 2, 0, 0])  # last two are pad_token_id=0
        tokenized_inputs = tokens.unsqueeze(0)
        labels = tokenized_inputs[:, 1:].squeeze(0)  # [1,2,0,0], seq_len=4
        # attention_mask over the *unshifted* sequence: [1,1,1,0,0]
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

        # Correct at the two real label positions (0,1); deliberately wrong
        # at the padded label positions (2,3) to prove they get excluded.
        logits = _build_logits(labels, correct_positions={0, 1})
        model = _fake_model_with_logits(logits)

        task = HellaSwag.__new__(HellaSwag)
        loss = task._do_predict(
            model,
            tokenized_inputs=tokenized_inputs,
            attention_mask=attention_mask,
            prompt_lens=[0],
        )

        assert loss.item() < 0.1


class TestHellaSwagScoreGuard:
    def test_missing_correct_label_raises(self):
        task = HellaSwag.__new__(HellaSwag)
        with pytest.raises(ValueError, match="no correct"):
            task._get_score(
                batch_prompts=["p", "p", "p", "p"],
                all_losses=[1.0, 2.0, 3.0, 4.0],
                all_labels=[0, 0, 0, 0],  # no label == 1
            )
