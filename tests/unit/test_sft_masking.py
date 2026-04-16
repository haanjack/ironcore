# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test: SFT loss masking must exclude prompt tokens (labels == -100).

This tests the fix in LanguageModel.get_masks_and_position_ids() which was
returning all-ones loss_mask regardless of label masking. The fix derives
loss_mask from labels: 0 where labels == -100 (prompt), 1 otherwise.
"""

import torch
from tests.fixtures.config_fixtures import create_small_test_config

from ironcore import global_vars
from ironcore.language_model import LanguageModel
from ironcore.parallel import parallel_states


def _make_model():
    """Create a LanguageModel with all required global state."""
    config = create_small_test_config()

    # Initialize global state for tokenizer
    if global_vars.GLOBAL_STATES is None:
        global_vars.set_global_states(config)

    # Single GPU mode: ensure parallel state is initialized for TP=1
    if parallel_states._TENSOR_MODEL_PARALLEL_GROUP is None:
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=1, timeout_in_minutes=10.0
        )

    model = LanguageModel(config)
    model.eval()
    return model


class TestSFTLossMasking:
    """Verify get_masks_and_position_ids derives loss_mask from labels."""

    def test_loss_mask_ignores_prompt_tokens(self):
        """Prompt tokens (labels=-100) should have loss_mask=0, response tokens should have 1."""
        model = _make_model()
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        # First 8 tokens are prompt (masked), last 8 are response
        labels = input_ids.clone()
        labels[:, :8] = -100

        _, _, loss_mask = model.get_masks_and_position_ids(input_ids, labels=labels)

        assert (loss_mask[:, :8] == 0.0).all(), "Prompt tokens should have loss_mask=0"
        assert (loss_mask[:, 8:] == 1.0).all(), "Response tokens should have loss_mask=1"

    def test_loss_mask_all_ones_when_no_labels(self):
        """When labels is None, loss_mask should be all ones (pretrain behavior)."""
        model = _make_model()
        input_ids = torch.randint(0, 100, (2, 16))

        _, _, loss_mask = model.get_masks_and_position_ids(input_ids, labels=None)

        assert (loss_mask == 1.0).all(), "Without labels, all tokens should count"

    def test_loss_mask_all_ones_when_no_masking(self):
        """When labels has no -100 values, loss_mask should be all ones."""
        model = _make_model()
        input_ids = torch.randint(0, 100, (2, 16))
        labels = input_ids.clone()

        _, _, loss_mask = model.get_masks_and_position_ids(input_ids, labels=labels)

        assert (loss_mask == 1.0).all(), "No masked tokens means all ones"
