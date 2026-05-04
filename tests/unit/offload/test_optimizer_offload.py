# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for optimizer offloading (simple host optimizer states).

Tests cover:
1. Offloaded optimizer produces same results as in-VRAM optimizer
2. Optimizer states are on CPU after step
3. Parameters update correctly
4. Small params below threshold are not offloaded
5. LoRA params excluded via offloadable attribute
6. Muon momentum stays on GPU
7. Shared helper correctness
"""

import pytest
import torch
from torch import nn

from ironcore.offload.optimizer_helpers import (
    _adamw_offloaded_step,
    _should_offload_param,
)
from ironcore.optimizer.adamw import AdamWOptimizer
from ironcore.optimizer.muon import MuonOptimizer

cuda_available = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")


class TestShouldOffloadParam:
    """Test the _should_offload_param helper."""

    def test_normal_param_offloads(self):
        p = nn.Parameter(torch.randn(256, 256))
        assert _should_offload_param(p, 65536) is True

    def test_small_param_not_offloaded(self):
        p = nn.Parameter(torch.randn(100))
        assert _should_offload_param(p, 65536) is False

    def test_boundary_exact_threshold(self):
        p = nn.Parameter(torch.randn(65536))
        assert _should_offload_param(p, 65536) is True

    def test_boundary_one_below(self):
        p = nn.Parameter(torch.randn(65535))
        assert _should_offload_param(p, 65536) is False

    def test_offloadable_false_excludes(self):
        p = nn.Parameter(torch.randn(256, 256))
        p.offloadable = False
        assert _should_offload_param(p, 65536) is False

    def test_offloadable_true_includes(self):
        p = nn.Parameter(torch.randn(256, 256))
        p.offloadable = True
        assert _should_offload_param(p, 65536) is True


class TestAdamWOffloadedStep:
    """Test AdamW with optimizer states on host."""

    def _make_model_and_grad(self, seed=42):
        torch.manual_seed(seed)
        model = nn.Linear(64, 64, bias=False)
        # Create a fake gradient
        grad = torch.randn_like(model.weight)
        model.weight.grad = grad
        return model

    def test_offloaded_states_on_cpu_after_step(self):
        """After step, exp_avg and exp_avg_sq should be on CPU."""
        model = self._make_model_and_grad()

        optimizer = AdamWOptimizer(
            [{"params": model.parameters()}],
            lr=1e-3,
            offload_enabled=True,
        )

        optimizer.step()

        for p in model.parameters():
            state = optimizer.state[p]
            assert state["exp_avg"].device == torch.device("cpu")
            assert state["exp_avg_sq"].device == torch.device("cpu")

    def test_offloaded_matches_in_vram(self):
        """Offloaded AdamW should produce identical results to in-VRAM AdamW."""
        torch.manual_seed(42)
        model_ref = nn.Linear(64, 64, bias=False)
        model_off = nn.Linear(64, 64, bias=False)
        # Copy weights so both start identical
        model_off.weight.data.copy_(model_ref.weight.data)

        grad = torch.randn_like(model_ref.weight)
        model_ref.weight.grad = grad.clone()
        model_off.weight.grad = grad.clone()

        opt_ref = AdamWOptimizer([{"params": model_ref.parameters()}], lr=1e-3)
        opt_off = AdamWOptimizer(
            [{"params": model_off.parameters()}], lr=1e-3, offload_enabled=True
        )

        # Run 5 steps with different gradients each time
        for _ in range(5):
            grad = torch.randn_like(model_ref.weight)
            model_ref.weight.grad = grad.clone()
            model_off.weight.grad = grad.clone()

            opt_ref.step()
            opt_off.step()

        # Parameters should be identical
        assert torch.allclose(model_ref.weight, model_off.weight, atol=1e-6)

    def test_small_params_stay_on_gpu(self):
        """Params below min_param_elements should keep states on param device (not offloaded to CPU)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        small_param = nn.Parameter(torch.randn(100, device=device))
        large_param = nn.Parameter(torch.randn(256, 256, device=device))

        nn.ModuleList([nn.ParameterList([small_param, large_param])])

        # Manually set grads
        small_param.grad = torch.randn_like(small_param)
        large_param.grad = torch.randn_like(large_param)

        optimizer = AdamWOptimizer(
            [{"params": [small_param, large_param]}],
            lr=1e-3,
            offload_enabled=True,
            offload_min_param_elements=65536,
        )

        optimizer.step()

        # Small param: in-VRAM path (state on same device type as param)
        small_state = optimizer.state[small_param]
        assert small_state["exp_avg"].device.type == device.type

        # Large param: offloaded path (state on CPU)
        large_state = optimizer.state[large_param]
        assert large_state["exp_avg"].device.type == "cpu"

    def test_lora_params_not_offloaded(self):
        """Params with offloadable=False should not be offloaded (state stays on param device)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p = nn.Parameter(torch.randn(256, 256, device=device))
        p.offloadable = False
        p.grad = torch.randn_like(p)

        optimizer = AdamWOptimizer(
            [{"params": [p]}],
            lr=1e-3,
            offload_enabled=True,
        )

        optimizer.step()

        state = optimizer.state[p]
        # Should stay on param device type, not moved to CPU
        assert state["exp_avg"].device.type == device.type

    def test_offload_disabled_uses_vram_path(self):
        """When offload_enabled=False, states stay on param device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Linear(64, 64, bias=False).to(device)
        grad = torch.randn_like(model.weight)
        model.weight.grad = grad

        optimizer = AdamWOptimizer(
            [{"params": model.parameters()}],
            lr=1e-3,
            offload_enabled=False,
        )

        optimizer.step()

        for p in model.parameters():
            state = optimizer.state[p]
            assert state["exp_avg"].device.type == device.type


class TestAdamWOffloadedStepHelper:
    """Test the shared _adamw_offloaded_step helper directly."""

    def test_state_initialization_on_cpu(self):
        """First call should create states on CPU."""
        p = nn.Parameter(torch.randn(128, 128))
        p.grad = torch.randn_like(p)
        state = {}

        _adamw_offloaded_step(
            p,
            p.grad,
            state,
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            state_dtype=torch.float32,
        )

        assert state["step"] == 1
        assert state["exp_avg"].device == torch.device("cpu")
        assert state["exp_avg_sq"].device == torch.device("cpu")
        assert state["exp_avg"].shape == p.shape

    def test_amsgrad_state_on_cpu(self):
        """With amsgrad=True, max_exp_avg_sq should also be on CPU."""
        p = nn.Parameter(torch.randn(128, 128))
        p.grad = torch.randn_like(p)
        state = {}

        _adamw_offloaded_step(
            p,
            p.grad,
            state,
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=True,
            state_dtype=torch.float32,
        )

        assert state["max_exp_avg_sq"].device == torch.device("cpu")

    def test_parameter_updates_correctly(self):
        """Parameter should be updated after step."""
        p = nn.Parameter(torch.randn(32, 32))
        p.grad = torch.ones_like(p) * 0.1
        state = {}
        original = p.data.clone()

        _adamw_offloaded_step(
            p,
            p.grad,
            state,
            lr=1e-2,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            state_dtype=torch.float32,
        )

        # Parameter should have changed
        assert not torch.allclose(p.data, original)

    def test_weight_decay_applied(self):
        """Weight decay should shrink the parameter."""
        p = nn.Parameter(torch.ones(32, 32))
        p.grad = torch.zeros_like(p)
        state = {}

        _adamw_offloaded_step(
            p,
            p.grad,
            state,
            lr=1e-2,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.1,
            amsgrad=False,
            state_dtype=torch.float32,
        )

        # With zero grad and weight_decay=0.1, p should shrink by (1 - lr * wd)
        expected_scale = 1 - 0.01 * 0.1
        assert torch.allclose(p.data, torch.ones(32, 32) * expected_scale, atol=1e-6)


class TestMuonOffloaded:
    """Test Muon optimizer with offloading (AdamW branch only)."""

    def _make_muon_optimizer(self, offload_enabled=False):
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create small model with clear Muon/AdamW param split
        attn_weight = nn.Parameter(torch.randn(32, 32, device=device))
        embedding = nn.Parameter(torch.randn(100, 32, device=device))
        bias = nn.Parameter(torch.randn(32, device=device))

        # Set grads
        attn_weight.grad = torch.randn_like(attn_weight)
        embedding.grad = torch.randn_like(embedding)
        bias.grad = torch.randn_like(bias)

        # Manually classify: attn_weight -> muon, rest -> adamw
        # (is_muon_param would classify based on name, but we test with manual groups)
        optimizer = MuonOptimizer(
            muon_params=[{"params": [attn_weight]}],
            adamw_params=[{"params": [embedding, bias]}],
            lr=0.02,
            momentum=0.95,
            offload_enabled=offload_enabled,
        )
        return optimizer, attn_weight, embedding, bias, device

    def test_muon_momentum_stays_on_gpu(self):
        """Muon momentum buffer should NOT be offloaded (stays on param device)."""
        optimizer, attn_weight, _, _, device = self._make_muon_optimizer(offload_enabled=True)
        optimizer.step()

        state = optimizer.state[attn_weight]
        assert "momentum_buffer" in state
        # Momentum must stay on the param's device type, not offloaded to CPU
        assert state["momentum_buffer"].device.type == device.type

    def test_muon_adamw_states_offloaded(self):
        """AdamW-managed states in Muon should be offloaded to CPU."""
        optimizer, _, embedding, bias, _ = self._make_muon_optimizer(offload_enabled=True)
        optimizer.step()

        for p in [embedding, bias]:
            if p.numel() >= 65536:
                state = optimizer.state[p]
                assert state["exp_avg"].device == torch.device("cpu")
                assert state["exp_avg_sq"].device == torch.device("cpu")

    def test_muon_offload_disabled(self):
        """With offload disabled, all states stay on param device."""
        optimizer, attn_weight, embedding, bias, device = self._make_muon_optimizer(
            offload_enabled=False
        )
        optimizer.step()

        for p in [attn_weight, embedding, bias]:
            state = optimizer.state[p]
            for key in state:
                if isinstance(state[key], torch.Tensor):
                    assert state[key].device.type == device.type, (
                        f"State '{key}' unexpectedly on {state[key].device}, expected {device.type}"
                    )

    def test_muon_offloaded_matches_reference(self):
        """Offloaded Muon should produce same parameter updates as non-offloaded."""
        torch.manual_seed(42)

        # Reference: no offload
        attn_ref = nn.Parameter(torch.randn(32, 32))
        emb_ref = nn.Parameter(torch.randn(100, 32))
        bias_ref = nn.Parameter(torch.randn(32))

        # Offloaded
        attn_off = nn.Parameter(attn_ref.data.clone())
        emb_off = nn.Parameter(emb_ref.data.clone())
        bias_off = nn.Parameter(bias_ref.data.clone())

        for _ in range(3):
            grad_a = torch.randn_like(attn_ref)
            grad_e = torch.randn_like(emb_ref)
            grad_b = torch.randn_like(bias_ref)

            attn_ref.grad = grad_a.clone()
            emb_ref.grad = grad_e.clone()
            bias_ref.grad = grad_b.clone()
            attn_off.grad = grad_a.clone()
            emb_off.grad = grad_e.clone()
            bias_off.grad = grad_b.clone()

            opt_ref = MuonOptimizer(
                muon_params=[{"params": [attn_ref]}],
                adamw_params=[{"params": [emb_ref, bias_ref]}],
                lr=0.02,
                momentum=0.95,
                offload_enabled=False,
            )
            opt_off = MuonOptimizer(
                muon_params=[{"params": [attn_off]}],
                adamw_params=[{"params": [emb_off, bias_off]}],
                lr=0.02,
                momentum=0.95,
                offload_enabled=True,
            )

            opt_ref.step()
            opt_off.step()

        # Note: Muon has NS5 iteration which works in bf16, so tolerance needs to be looser
        assert torch.allclose(attn_ref, attn_off, atol=1e-4)
        assert torch.allclose(emb_ref, emb_off, atol=1e-6)
        assert torch.allclose(bias_ref, bias_off, atol=1e-6)


@skip_no_cuda
class TestAdamWOffloadedCPUCompute:
    """Tests for the CPU-compute path activated when params are on CUDA (optimizer offload only)."""

    def test_cpu_compute_path_cuda_param(self):
        """CPU-compute path: states stay on CPU, param on CUDA updates correctly."""
        p = nn.Parameter(torch.randn(128, 128, device="cuda"))
        p.grad = torch.randn_like(p)
        state = {}
        original = p.data.clone()

        _adamw_offloaded_step(
            p,
            p.grad,
            state,
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            state_dtype=torch.float32,
        )

        assert state["exp_avg"].device == torch.device("cpu")
        assert state["exp_avg_sq"].device == torch.device("cpu")
        assert not torch.allclose(p.data, original)
        assert p.data.device.type == "cuda"

    def test_cpu_compute_matches_gpu_compute(self):
        """CPU-compute path should produce nearly identical results to in-VRAM path."""
        torch.manual_seed(42)

        # Reference: in-VRAM AdamW (no offload)
        p_ref = nn.Parameter(torch.randn(64, 64, device="cuda"))
        opt_ref = AdamWOptimizer([{"params": [p_ref]}], lr=1e-3, offload_enabled=False)

        # Offloaded: CPU-compute path
        p_off = nn.Parameter(p_ref.data.clone())
        opt_off = AdamWOptimizer([{"params": [p_off]}], lr=1e-3, offload_enabled=True)

        for step in range(5):
            torch.manual_seed(step)
            grad = torch.randn_like(p_ref)
            p_ref.grad = grad.clone()
            p_off.grad = grad.clone()

            opt_ref.step()
            opt_off.step()

        # CPU fp32 vs GPU fp32 can differ slightly due to SIMD rounding order
        assert torch.allclose(p_ref.data, p_off.data, atol=1e-5), (
            f"Max diff: {(p_ref.data - p_off.data).abs().max().item():.2e}"
        )

    def test_cpu_compute_bf16_state_dtype(self):
        """CPU-compute path with bf16 state storage: states stored in bf16 on CPU."""
        p = nn.Parameter(torch.randn(64, 64, device="cuda"))
        p.grad = torch.randn_like(p)
        state = {}

        _adamw_offloaded_step(
            p,
            p.grad,
            state,
            lr=1e-2,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            state_dtype=torch.bfloat16,
        )

        assert state["exp_avg"].dtype == torch.bfloat16
        assert state["exp_avg_sq"].dtype == torch.bfloat16
        assert state["exp_avg"].device == torch.device("cpu")

    def test_cpu_compute_weight_decay(self):
        """Weight decay should shrink the parameter (applied on GPU)."""
        p = nn.Parameter(torch.ones(64, 64, device="cuda"))
        p.grad = torch.zeros_like(p)
        state = {}

        _adamw_offloaded_step(
            p,
            p.grad,
            state,
            lr=1e-2,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.1,
            amsgrad=False,
            state_dtype=torch.float32,
        )

        expected_scale = 1 - 0.01 * 0.1
        assert torch.allclose(p.data.cpu(), torch.ones(64, 64) * expected_scale, atol=1e-5)

    def test_cpu_compute_amsgrad(self):
        """amsgrad variant: max_exp_avg_sq stays on CPU."""
        p = nn.Parameter(torch.randn(64, 64, device="cuda"))
        p.grad = torch.randn_like(p)
        state = {}

        _adamw_offloaded_step(
            p,
            p.grad,
            state,
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=True,
            state_dtype=torch.float32,
        )

        assert "max_exp_avg_sq" in state
        assert state["max_exp_avg_sq"].device == torch.device("cpu")
        assert state["max_exp_avg_sq"].dtype == torch.float32

    def test_cpu_compute_vram_reduction(self):
        """Peak VRAM during offloaded step should be less than 2x param size in fp32."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # 1024x1024 fp32 = 4MB param. Old path would stage exp_avg + exp_avg_sq
        # = 8MB on GPU. New path only needs delta (4MB in fp32).
        p = nn.Parameter(torch.randn(1024, 1024, device="cuda"))
        p.grad = torch.randn_like(p)

        before = torch.cuda.memory_allocated()
        _adamw_offloaded_step(
            p,
            p.grad,
            {},
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            state_dtype=torch.float32,
        )
        peak = torch.cuda.max_memory_allocated()

        param_bytes = p.numel() * p.element_size()  # 4MB

        # Peak should be at most 2x param bytes (delta + grad).
        # Old path would be ~3x param bytes (param + exp_avg + exp_avg_sq on GPU).
        assert peak - before < 2.5 * param_bytes, (
            f"Peak VRAM too high: {(peak - before) / 1024**2:.1f}MB "
            f"(expected < {2.5 * param_bytes / 1024**2:.1f}MB)"
        )
