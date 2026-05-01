# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Reproduction test for BUG-006: bf16 + M2+M3 dtype crash."""

import pytest
import torch

from tests.fixtures.config_fixtures import create_small_test_config

cuda_available = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

DEVICE = torch.device("cuda:0")


def _make_config(precision="bfloat16", weight_offload=True):
    config = create_small_test_config()
    config.model.precision = precision
    config.offload.enabled = weight_offload
    config.offload.weight_offload = weight_offload
    config.trainer.gradient_accumulation_steps = 1
    return config


@skip_no_cuda
class TestBf16Offload:
    """Reproduce BUG-006: bf16 model + M2+M3 crash."""

    def test_bf16_weight_streaming_forward_backward(self):
        """bf16 model + weight streaming should not crash during forward/backward."""
        from ironcore.models.transformer import TransformerModel
        from ironcore.offload.scheduler import ExecutionScheduler

        config = _make_config(precision="bfloat16")
        dtype = torch.bfloat16

        torch.manual_seed(42)
        model = TransformerModel(config).to(device=DEVICE, dtype=dtype)
        model.train()

        scheduler = ExecutionScheduler.from_model(
            model=model, config=config.offload, device=DEVICE
        )
        model._offload_scheduler = scheduler
        scheduler.set_gradient_accumulation_steps(1)

        hidden = torch.randn(2, 8, 128, device=DEVICE, dtype=dtype)
        mask = torch.ones(2, 1, 8, 8, device=DEVICE)

        # Forward
        scheduler.on_microbatch_forward_start(0)
        out = model(hidden, mask, None)
        scheduler.on_microbatch_forward_end()

        loss = out.sum()

        # Backward
        scheduler.on_microbatch_backward_start(0)
        loss.backward()
        scheduler.on_microbatch_backward_end()
        scheduler.on_training_step_end()

        # Verify output is valid
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert out.dtype == dtype, f"Output dtype {out.dtype} != expected {dtype}"

    def test_bf16_weight_streaming_multi_step(self):
        """bf16 + M2+M3 should survive multiple forward/backward steps."""
        from ironcore.models.transformer import TransformerModel
        from ironcore.offload.scheduler import ExecutionScheduler

        config = _make_config(precision="bfloat16")
        dtype = torch.bfloat16

        torch.manual_seed(42)
        model = TransformerModel(config).to(device=DEVICE, dtype=dtype)
        model.train()

        scheduler = ExecutionScheduler.from_model(
            model=model, config=config.offload, device=DEVICE
        )
        model._offload_scheduler = scheduler
        scheduler.set_gradient_accumulation_steps(1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for step in range(3):
            torch.manual_seed(step)
            hidden = torch.randn(2, 8, 128, device=DEVICE, dtype=dtype)
            mask = torch.ones(2, 1, 8, 8, device=DEVICE)

            scheduler.on_microbatch_forward_start(0)
            out = model(hidden, mask, None)
            scheduler.on_microbatch_forward_end()

            loss = out.square().mean()
            assert not torch.isnan(loss), f"NaN loss at step {step}"

            scheduler.on_microbatch_backward_start(0)
            loss.backward()
            scheduler.on_microbatch_backward_end()
            scheduler.on_training_step_end()

            optimizer.step()
            optimizer.zero_grad()

    def test_fp32_vs_bf16_forward_parity(self):
        """Forward output should match between fp32 and bf16 with weight streaming."""
        from ironcore.models.transformer import TransformerModel
        from ironcore.offload.scheduler import ExecutionScheduler

        # fp32 reference
        config_fp32 = _make_config(precision="float32")
        torch.manual_seed(42)
        model_fp32 = TransformerModel(config_fp32).to(device=DEVICE, dtype=torch.float32)
        model_fp32.eval()

        # bf16 offload
        config_bf16 = _make_config(precision="bfloat16")
        torch.manual_seed(42)
        model_bf16 = TransformerModel(config_bf16).to(device=DEVICE, dtype=torch.bfloat16)
        model_bf16.train()

        scheduler = ExecutionScheduler.from_model(
            model=model_bf16, config=config_bf16.offload, device=DEVICE
        )
        model_bf16._offload_scheduler = scheduler
        scheduler.set_gradient_accumulation_steps(1)

        # Same inputs (cast to appropriate dtype)
        torch.manual_seed(99)
        hidden_fp32 = torch.randn(2, 8, 128, device=DEVICE, dtype=torch.float32)
        mask = torch.ones(2, 1, 8, 8, device=DEVICE)

        with torch.no_grad():
            out_fp32 = model_fp32(hidden_fp32, mask, None)

        scheduler.on_microbatch_forward_start(0)
        with torch.no_grad():
            out_bf16 = model_bf16(hidden_fp32.to(torch.bfloat16), mask, None)
        scheduler.on_microbatch_forward_end()

        # bf16 has limited precision — compare at bf16 tolerance
        out_fp32_bf16 = out_fp32.to(torch.bfloat16)
        max_diff = (out_fp32_bf16 - out_bf16).abs().max().item()
        assert max_diff < 0.1, f"Forward parity failed: max diff={max_diff}"
