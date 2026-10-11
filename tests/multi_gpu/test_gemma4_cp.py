# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""CP2 Gemma A4B masked sample loss and adapter gradients against CP1."""

import os

import pytest
import torch
import torch.distributed as dist
from tests.fixtures.gemma4 import gemma4_pair

from ironcore.parallel import parallel_states
from ironcore.parallel.context_parallel import synchronize_context_parallel_gradients
from ironcore.training_utils import loss_func_sft

pytestmark = [pytest.mark.mp, pytest.mark.skipif("RANK" not in os.environ, reason="Needs torchrun")]


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_gemma4_cp2_sft_loss_and_adapter_gradient_parity(device_type):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = (
        torch.device(device_type, int(os.environ["LOCAL_RANK"]))
        if device_type == "cuda"
        else torch.device("cpu")
    )
    if device_type == "cuda":
        torch.cuda.set_device(device)
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo")
    parallel_states.initialize_model_parallel(1, timeout_in_minutes=5, context_parallel_size=1)
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7], [2, 8, 9, 10, 11, 12]], device=device)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    labels = tokens.clone()
    labels[0, :5] = -100
    labels[1, :1] = -100
    try:
        with pytest.MonkeyPatch.context() as patch:
            torch.manual_seed(42)
            reference, _, _ = gemma4_pair(patch, "A4B", lora=True, lora_targets=targets)
            reference.to(device)
            with torch.no_grad():
                for name, parameter in reference.named_parameters():
                    if name.endswith("lora_B"):
                        parameter.fill_(0.01)
            reference.loss_fn = loss_func_sft
            expected = reference(tokens, labels=labels)
            expected.backward()
            gradients = {
                name: p.grad.clone() if p.grad is not None else None
                for name, p in reference.named_parameters()
                if p.requires_grad
            }
        parallel_states.destroy_model_parallel()
        parallel_states.initialize_model_parallel(1, timeout_in_minutes=5, context_parallel_size=2)
        with pytest.MonkeyPatch.context() as patch:
            torch.manual_seed(42)
            native, _, config = gemma4_pair(
                patch, "A4B", lora=True, cp_size=2, lora_targets=targets
            )
            native.to(device)
            with torch.no_grad():
                for name, parameter in native.named_parameters():
                    if name.endswith("lora_B"):
                        parameter.fill_(0.01)
            config.trainer.recompute_linear_ce = True
            config.trainer.loss_chunk_size = 2
            config.model.moe.expert_backend = "grouped"
            config.model.moe.blockwise_backend = "triton" if device_type == "cuda" else "scheduled"
            config.model.moe.grouped_token_budget = 5
            native.loss_fn = loss_func_sft
            actual = native(tokens, labels=labels)
            actual.backward()
            synchronize_context_parallel_gradients(native)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            for name, p in native.named_parameters():
                if p.requires_grad:
                    if gradients[name] is None:
                        assert p.grad is None, name
                        continue
                    torch.testing.assert_close(
                        p.grad, gradients[name], atol=5e-5, rtol=5e-4, msg=name
                    )
                else:
                    assert p.grad is None
    finally:
        parallel_states.destroy_model_parallel()
        dist.destroy_process_group()
