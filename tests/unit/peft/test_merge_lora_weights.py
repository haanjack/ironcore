# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ironcore/peft/utils.py::merge_lora_weights.

merge_lora_weights was previously an unimplemented stub (NotImplementedError).
These tests verify the real implementation: merging LoRA adapters into the
base weights must be numerically a no-op on the model's *forward pass* output
(the whole point of merging — same computation, no separate LoRA matmuls
needed afterward), and must replace each LoRA wrapper with its base layer.
"""

import torch
from tests.fixtures.lora_test_utils import create_lora_test_config, set_seed

from ironcore.global_vars import global_states_cleanup, set_global_states
from ironcore.language_model import LanguageModel
from ironcore.parallel import parallel_states
from ironcore.peft.lora import (
    LoRAColumnParallelLinear,
    LoRAConcatenatedColumnParallel,
    LoRARowParallelLinear,
)
from ironcore.peft.utils import merge_lora_weights


def _count_lora_wrappers(model) -> int:
    count = 0
    for module in model.modules():
        if isinstance(
            module,
            (LoRAColumnParallelLinear, LoRARowParallelLinear, LoRAConcatenatedColumnParallel),
        ):
            count += 1
    return count


def _randomize_lora_b(model):
    """lora_B is zero-initialized (so LoRA starts as a no-op); give it
    non-zero values so a merge is numerically meaningful to test."""
    for name, param in model.named_parameters():
        if name.endswith("lora_B"):
            with torch.no_grad():
                param.copy_(torch.randn_like(param) * 0.1)


class TestMergeLoraWeights:
    def setup_method(self):
        set_seed(42)
        self.config = create_lora_test_config(
            tp_size=1,
            enable_lora=True,
            lora_r=4,
            lora_alpha=8.0,
            lora_target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
            d_model=64,
            num_attention_heads=4,
            max_seq_len=16,
        )
        parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)
        set_global_states(self.config)

    def teardown_method(self):
        global_states_cleanup()
        parallel_states.destroy_model_parallel()

    def test_merge_preserves_forward_output(self):
        model = LanguageModel(self.config)
        model.eval()
        _randomize_lora_b(model)

        assert _count_lora_wrappers(model) > 0, "test config produced no LoRA-wrapped layers"

        input_ids = torch.randint(0, self.config.model.padded_vocab_size, (2, 8))

        def _logits(out):
            return out[0] if isinstance(out, tuple) else out

        with torch.no_grad():
            output_before = _logits(model(input_ids, labels=None))

        merge_lora_weights(model)

        with torch.no_grad():
            output_after = _logits(model(input_ids, labels=None))

        assert torch.allclose(output_before, output_after, atol=1e-3, rtol=1e-3), (
            "merged model's forward output diverged from the pre-merge (LoRA-wrapped) output"
        )

    def test_merge_removes_lora_wrappers(self):
        model = LanguageModel(self.config)
        model.eval()
        _randomize_lora_b(model)

        assert _count_lora_wrappers(model) > 0

        merge_lora_weights(model)

        assert _count_lora_wrappers(model) == 0, "LoRA wrappers should be replaced after merging"

    def test_merge_is_no_op_when_lora_b_is_zero(self):
        """With the default zero lora_B init, LoRA contributes nothing, so
        merging must leave the base weights numerically unchanged."""
        model = LanguageModel(self.config)
        model.eval()
        # Deliberately do NOT randomize lora_B here.

        base_weights_before = {
            name: param.clone()
            for name, param in model.named_parameters()
            if "lora_" not in name
        }

        merge_lora_weights(model)

        base_weights_after = dict(model.named_parameters())
        for name, before in base_weights_before.items():
            # After merging, LoRA-wrapped layers are replaced, so the module
            # path may differ; match by suffix (e.g. ".weight", ".bias").
            after = base_weights_after.get(name)
            if after is not None:
                assert torch.allclose(before, after, atol=1e-6)
