# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test for grad_accum>1 + weight streaming + activation spill device mismatch.

Reproduces the bug where on_layer_start skips loading weights when no
prefetch is in flight — which happens at the start of every micro-batch
after the first, because backward consumes all prefetches.
"""

import pytest
import torch
from tests.fixtures.config_fixtures import create_small_test_config

cuda_available = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

DEVICE = torch.device("cuda:0")


def _make_config(grad_accum=2):
    config = create_small_test_config()
    config.offload.enabled = True
    config.offload.weight_offload = True
    config.offload.activation_spill = True
    config.trainer.gradient_accumulation_steps = grad_accum
    return config


@skip_no_cuda
class TestGradAccumDeviceMismatch:
    """Regression: grad_accum>1 + weight streaming + activation spill should not produce device mismatch."""

    def test_two_microbatches_forward_backward(self):
        """Two micro-batches of forward/backward with weight streaming + activation spill."""
        from ironcore.models.transformer import TransformerModel
        from ironcore.offload.scheduler import ExecutionScheduler

        config = _make_config(grad_accum=2)
        grad_accum = config.trainer.gradient_accumulation_steps

        torch.manual_seed(42)
        model = TransformerModel(config).to(device=DEVICE, dtype=torch.float32)
        model.train()

        scheduler = ExecutionScheduler.from_model(model=model, config=config.offload, device=DEVICE)
        model._offload_scheduler = scheduler
        scheduler.set_gradient_accumulation_steps(grad_accum)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler.on_training_step_start()

        for mb in range(grad_accum):
            torch.manual_seed(mb)
            hidden = torch.randn(2, 8, 128, device=DEVICE)
            mask = torch.ones(2, 1, 8, 8, device=DEVICE)

            scheduler.on_microbatch_forward_start(mb)
            out = model(hidden, mask, None)
            scheduler.on_microbatch_forward_end()

            loss = out.square().mean()
            assert not torch.isnan(loss), f"NaN loss at micro-batch {mb}"

            scheduler.on_microbatch_backward_start(mb)
            loss.backward()
            scheduler.on_microbatch_backward_end()

        scheduler.on_backward_pass_end()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.on_training_step_end()

    def test_four_microbatches_converge(self):
        """Four micro-batches should produce stable losses (no crash, no NaN)."""
        from ironcore.models.transformer import TransformerModel
        from ironcore.offload.scheduler import ExecutionScheduler

        config = _make_config(grad_accum=4)
        grad_accum = config.trainer.gradient_accumulation_steps

        torch.manual_seed(42)
        model = TransformerModel(config).to(device=DEVICE, dtype=torch.float32)
        model.train()

        scheduler = ExecutionScheduler.from_model(model=model, config=config.offload, device=DEVICE)
        model._offload_scheduler = scheduler
        scheduler.set_gradient_accumulation_steps(grad_accum)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        for step in range(3):
            scheduler.on_training_step_start()

            for mb in range(grad_accum):
                torch.manual_seed(step * grad_accum + mb)
                hidden = torch.randn(2, 8, 128, device=DEVICE)
                mask = torch.ones(2, 1, 8, 8, device=DEVICE)

                scheduler.on_microbatch_forward_start(mb)
                out = model(hidden, mask, None)
                scheduler.on_microbatch_forward_end()

                loss = out.square().mean()
                assert not torch.isnan(loss), f"NaN at step={step} mb={mb}"

                scheduler.on_microbatch_backward_start(mb)
                loss.backward()
                scheduler.on_microbatch_backward_end()

            scheduler.on_backward_pass_end()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.on_training_step_end()

            losses.append(loss.item())

        assert all(not torch.isnan(torch.tensor(loss)) for loss in losses)
