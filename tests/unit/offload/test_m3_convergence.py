# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Diagnostic tests for BUG-001: activation spill breaks training convergence.

Tests compare forward output and backward gradients with and without activation spill
to isolate exactly where the divergence occurs.
"""

import pytest
import torch

cuda_available = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

DTYPE = torch.bfloat16
DEVICE = torch.device("cuda:0")


def _make_model_config():
    """Create minimal model config for testing."""
    from ironcore.config import MainConfig
    from ironcore.config.config_alignment import AlignmentConfig
    from ironcore.config.config_data import DataConfig
    from ironcore.config.config_model import ModelConfig
    from ironcore.config.config_optim import OptimConfig
    from ironcore.config.config_parallel import ParallelConfig
    from ironcore.config.config_peft import PEFTConfig
    from ironcore.config.config_trainer import InitConfig, OperationConfig, TrainerConfig
    from ironcore.config.config_utils import ProfilerConfig, UtilsConfig
    from ironcore.offload.config import OffloadConfig

    model_config = ModelConfig()
    model_config.num_layers = 2
    model_config.num_attention_heads = 4
    model_config.num_attention_groups = 4

    config = MainConfig(
        model=model_config,
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(),
        operation=OperationConfig(train_steps=10),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
        alignment=AlignmentConfig(),
        offload=OffloadConfig(),
    )
    config.trainer.micro_batch_size = 2
    config.trainer.train_batch_size = 2
    config.trainer.gradient_accumulation_steps = 1
    config.parallel.world_size = 1
    return config


def _make_scheduler(model, config, granularity="sub_layer"):
    """Create an ExecutionScheduler with activation spill enabled."""
    from ironcore.offload.config import OffloadConfig
    from ironcore.offload.scheduler import ExecutionScheduler

    offload = OffloadConfig(
        enabled=True,
        activation_spill=True,
        activation_spill_granularity=granularity,
        pinned_chunk_gb=0.05,
        pinned_memory_pool_gb=0.2,
    )
    scheduler = ExecutionScheduler.from_model(
        model=model,
        config=offload,
        device=DEVICE,
    )
    return scheduler, offload


def _make_inputs(device, dtype):
    hidden = torch.randn(2, 8, 512, device=device, dtype=dtype)
    mask = torch.ones(2, 1, 8, 8, device=device)
    return hidden, mask


@skip_no_cuda
class TestM3GradientParity:
    """Compare forward output and backward gradients with/without activation spill."""

    def test_forward_output_parity_no_dropout(self):
        """Forward output should be identical with and without spill (no dropout)."""
        from ironcore.models.transformer import TransformerModel

        config = _make_model_config()
        config.model.dropout_attn = 0.0

        torch.manual_seed(42)
        model_ref = TransformerModel(config).to(device=DEVICE, dtype=DTYPE)
        torch.manual_seed(42)
        model_spill = TransformerModel(config).to(device=DEVICE, dtype=DTYPE)

        # Verify identical weights
        for (n1, p1), (n2, p2) in zip(
            model_ref.named_parameters(), model_spill.named_parameters(), strict=False
        ):
            assert torch.equal(p1, p2), f"Weight mismatch: {n1}"

        scheduler, _ = _make_scheduler(model_spill, config)
        assert scheduler is not None
        model_spill._offload_scheduler = scheduler
        scheduler.set_gradient_accumulation_steps(1)

        # Same inputs
        torch.manual_seed(99)
        hidden_ref, mask_ref = _make_inputs(DEVICE, DTYPE)
        torch.manual_seed(99)
        hidden_spill, mask_spill = _make_inputs(DEVICE, DTYPE)

        # Forward without spill
        model_ref.eval()
        with torch.no_grad():
            out_ref = model_ref(hidden_ref, mask_ref, None)

        # Forward with spill (training mode for spill to activate)
        model_spill.train()
        scheduler.on_microbatch_forward_start(0)
        with torch.no_grad():
            out_spill = model_spill(hidden_spill, mask_spill, None)
        scheduler.on_microbatch_forward_end()

        assert out_ref.shape == out_spill.shape
        max_diff = (out_ref - out_spill).abs().max().item()
        assert max_diff == 0.0, f"Forward output differs by {max_diff}"

    def test_backward_gradient_parity_no_dropout(self):
        """Gradients should be identical with and without spill (no dropout)."""
        from ironcore.models.transformer import TransformerModel

        config = _make_model_config()
        config.model.dropout_attn = 0.0

        torch.manual_seed(42)
        model_ref = TransformerModel(config).to(device=DEVICE, dtype=DTYPE)
        torch.manual_seed(42)
        model_spill = TransformerModel(config).to(device=DEVICE, dtype=DTYPE)

        scheduler, _ = _make_scheduler(model_spill, config)
        assert scheduler is not None
        model_spill._offload_scheduler = scheduler
        scheduler.set_gradient_accumulation_steps(1)

        # Forward + backward without spill
        model_ref.train()
        torch.manual_seed(99)
        hidden_ref, mask_ref = _make_inputs(DEVICE, DTYPE)
        out_ref = model_ref(hidden_ref, mask_ref, None)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # Forward + backward with spill
        model_spill.train()
        torch.manual_seed(99)
        hidden_spill, mask_spill = _make_inputs(DEVICE, DTYPE)

        scheduler.on_microbatch_forward_start(0)
        out_spill = model_spill(hidden_spill, mask_spill, None)
        scheduler.on_microbatch_forward_end()

        loss_spill = out_spill.sum()
        scheduler.on_microbatch_backward_start(0)
        loss_spill.backward()
        scheduler.on_microbatch_backward_end()

        # Compare gradients for ALL parameters
        grad_diffs = {}
        for (n1, p1), (n2, p2) in zip(
            model_ref.named_parameters(), model_spill.named_parameters(), strict=False
        ):
            assert p1.grad is not None, f"ref grad is None for {n1}"
            assert p2.grad is not None, f"spill grad is None for {n2}"
            diff = (p1.grad - p2.grad).abs().max().item()
            grad_diffs[n1] = diff

        max_overall = max(grad_diffs.values())
        worst_param = max(grad_diffs, key=lambda k: grad_diffs[k])

        assert max_overall < 1e-1, (
            f"Gradient mismatch: max diff={max_overall:.2e} at {worst_param}.\n"
            f"All diffs: {grad_diffs}"
        )

    def test_backward_gradient_parity_with_dropout(self):
        """Gradients should be identical with dropout (RNG state must match)."""
        from ironcore.models.transformer import TransformerModel

        config = _make_model_config()
        config.model.dropout_attn = 0.1

        torch.manual_seed(42)
        model_ref = TransformerModel(config).to(device=DEVICE, dtype=DTYPE)
        torch.manual_seed(42)
        model_spill = TransformerModel(config).to(device=DEVICE, dtype=DTYPE)

        scheduler, _ = _make_scheduler(model_spill, config)
        assert scheduler is not None
        model_spill._offload_scheduler = scheduler
        scheduler.set_gradient_accumulation_steps(1)

        # Forward + backward without spill
        model_ref.train()
        torch.manual_seed(99)
        hidden_ref, mask_ref = _make_inputs(DEVICE, DTYPE)
        out_ref = model_ref(hidden_ref, mask_ref, None)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # Forward + backward with spill
        model_spill.train()
        torch.manual_seed(99)
        hidden_spill, mask_spill = _make_inputs(DEVICE, DTYPE)

        scheduler.on_microbatch_forward_start(0)
        out_spill = model_spill(hidden_spill, mask_spill, None)
        scheduler.on_microbatch_forward_end()

        loss_spill = out_spill.sum()
        scheduler.on_microbatch_backward_start(0)
        loss_spill.backward()
        scheduler.on_microbatch_backward_end()

        grad_diffs = {}
        for (n1, p1), (n2, p2) in zip(
            model_ref.named_parameters(), model_spill.named_parameters(), strict=False
        ):
            diff = (p1.grad - p2.grad).abs().max().item()
            grad_diffs[n1] = diff

        max_overall = max(grad_diffs.values())
        worst_param = max(grad_diffs, key=lambda k: grad_diffs[k])

        assert max_overall < 1e-1, (
            f"Gradient mismatch with dropout: max diff={max_overall:.2e} at {worst_param}.\n"
            f"All diffs: {grad_diffs}"
        )

    def test_convergence_5_step(self):
        """Training should converge over 5 steps with activation spill (loss should decrease)."""
        from ironcore.models.transformer import TransformerModel

        config = _make_model_config()
        config.model.dropout_attn = 0.0

        torch.manual_seed(42)
        model = TransformerModel(config).to(device=DEVICE, dtype=DTYPE)
        model.train()

        scheduler, _ = _make_scheduler(model, config)
        assert scheduler is not None
        model._offload_scheduler = scheduler
        scheduler.set_gradient_accumulation_steps(1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        for step in range(5):
            torch.manual_seed(step)
            hidden = torch.randn(2, 8, 512, device=DEVICE, dtype=DTYPE)
            mask = torch.ones(2, 1, 8, 8, device=DEVICE)

            scheduler.on_microbatch_forward_start(0)
            out = model(hidden, mask, None)
            scheduler.on_microbatch_forward_end()

            loss = out.sum()
            losses.append(loss.item())

            scheduler.on_microbatch_backward_start(0)
            loss.backward()
            scheduler.on_microbatch_backward_end()
            scheduler.on_training_step_end()

            optimizer.step()
            optimizer.zero_grad()

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_full_layer_granularity_gradient_parity(self):
        """full_layer granularity should also produce identical gradients."""
        from ironcore.models.transformer import TransformerModel

        config = _make_model_config()
        config.model.dropout_attn = 0.0

        torch.manual_seed(42)
        model_ref = TransformerModel(config).to(device=DEVICE, dtype=DTYPE)
        torch.manual_seed(42)
        model_spill = TransformerModel(config).to(device=DEVICE, dtype=DTYPE)

        scheduler, _ = _make_scheduler(model_spill, config, granularity="full_layer")
        assert scheduler is not None
        model_spill._offload_scheduler = scheduler
        scheduler.set_gradient_accumulation_steps(1)

        model_ref.train()
        torch.manual_seed(99)
        hidden_ref, mask_ref = _make_inputs(DEVICE, DTYPE)
        out_ref = model_ref(hidden_ref, mask_ref, None)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        model_spill.train()
        torch.manual_seed(99)
        hidden_spill, mask_spill = _make_inputs(DEVICE, DTYPE)

        scheduler.on_microbatch_forward_start(0)
        out_spill = model_spill(hidden_spill, mask_spill, None)
        scheduler.on_microbatch_forward_end()

        loss_spill = out_spill.sum()
        scheduler.on_microbatch_backward_start(0)
        loss_spill.backward()
        scheduler.on_microbatch_backward_end()

        grad_diffs = {}
        for (n1, p1), (n2, p2) in zip(
            model_ref.named_parameters(), model_spill.named_parameters(), strict=False
        ):
            diff = (p1.grad - p2.grad).abs().max().item()
            grad_diffs[n1] = diff

        max_overall = max(grad_diffs.values())
        assert max_overall < 1e-1, (
            f"Full-layer granularity gradient mismatch: max diff={max_overall:.2e}\n"
            f"All diffs: {grad_diffs}"
        )
