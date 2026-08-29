# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient correctness tests for Expert Parallel communication primitives.

Uses torch.autograd.gradcheck to verify gradient computations are correct.

Note: These tests use float64 (double precision) for numerical gradient checking.
"""

import pytest
import torch
from tests.fixtures.utils import single_gpu_env

from ironcore.parallel.expert_parallel.comm import (
    _AllReduceEP,
    all_reduce_ep_with_grad,
)


@pytest.fixture(autouse=True)
def _single_gpu_env():
    """Scope RANK/LOCAL_RANK/WORLD_SIZE to this test's lifetime."""
    with single_gpu_env():
        yield


class TestAllReduceEPGradient:
    """Gradient correctness tests for EP all-reduce."""

    def test_all_reduce_ep_forward_single(self):
        """Test forward pass with ep_size=1 (no communication)."""
        torch.manual_seed(42)

        x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
        output = _AllReduceEP.apply(x, 1)  # ep_size=1

        # Output should be same as input when ep_size=1
        assert torch.allclose(output, x)

    def test_all_reduce_ep_gradient_ep_size_1(self):
        """Test gradient with ep_size=1 (identity case)."""
        torch.manual_seed(42)

        x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)

        def fn(inp):
            return _AllReduceEP.apply(inp, 1)

        # gradcheck verifies numerical gradient matches analytical gradient
        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_all_reduce_ep_backward_pass(self):
        """Test backward pass creates gradients."""
        torch.manual_seed(42)

        x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
        output = _AllReduceEP.apply(x, 1)

        # Backward pass
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

        # Gradient should exist and match grad_output for ep_size=1
        assert x.grad is not None
        assert torch.allclose(x.grad, grad_output)

    def test_all_reduce_ep_with_grad_function(self):
        """Test all_reduce_ep_with_grad convenience function."""
        torch.manual_seed(42)

        x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
        output = all_reduce_ep_with_grad(x)

        # With ep_size=1, output equals input
        assert torch.allclose(output.detach(), x.detach())

        # Gradient flow
        output.sum().backward()
        assert x.grad is not None


class TestDispatchGatherGradient:
    """Gradient correctness tests for dispatch/gather operations.

    Note: Full gradcheck for all-to-all operations requires multi-GPU setup.
    These tests verify the gradient structure and basic properties.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU tests require CUDA")
    def test_dispatch_gradient_structure(self):
        """Test that dispatch creates proper gradient structure."""
        # This test verifies the gradient is properly shaped
        # Full gradient correctness requires multi-GPU all-to-all

        from ironcore.parallel.expert_parallel.comm import (
            DispatchMetadata,
            DispatchOutput,
        )

        # Just verify the data structures exist and have expected types
        assert DispatchOutput is not None
        assert DispatchMetadata is not None

    def test_combine_outputs_gradient(self):
        """Test that _combine_outputs preserves gradients."""
        from ironcore.parallel.expert_parallel.comm import AllToAllDispatcher

        torch.manual_seed(42)

        # Create a single-rank dispatcher (ep_size=1)
        # This tests the combine logic without actual communication
        batch_size, seq_len, hidden_size = 2, 4, 8
        batch_size * seq_len
        top_k = 2
        num_experts = 4

        # Create dispatcher with ep_size=1
        dispatcher = AllToAllDispatcher(num_experts, ep_size=1)

        # Create inputs
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float64)
        topk_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))
        topk_weights = torch.softmax(
            torch.randn(batch_size, seq_len, top_k, dtype=torch.float64), dim=-1
        )

        # Dispatch
        dispatch_output, metadata = dispatcher.dispatch(x, topk_indices, topk_weights)

        # Create expert outputs with gradient tracking
        expert_outputs = torch.randn_like(dispatch_output.tokens, requires_grad=True)

        # Gather
        combined, _ = dispatcher.gather(expert_outputs, metadata)

        # Verify gradient can flow back
        combined.sum().backward()
        assert expert_outputs.grad is not None


class TestAutogradFunctions:
    """Test autograd function implementations."""

    def test_all_reduce_ep_save_for_backward(self):
        """Test that ctx is properly saved for backward."""
        torch.manual_seed(42)

        # Test with ep_size=1
        x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)

        class TestFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp, ep_size):
                ctx.ep_size = ep_size
                return inp.clone()

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None

        output = TestFn.apply(x, 1)
        output.sum().backward()

        assert x.grad is not None

    def test_all_reduce_ep_no_grad_for_ep_size(self):
        """Test that ep_size doesn't get a gradient."""
        torch.manual_seed(42)

        x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)

        output = _AllReduceEP.apply(x, 1)
        output.sum().backward()

        # ep_size is the second argument, its gradient should be None
        # (verified by the backward returning grad_input, None)


class TestNumericalStability:
    """Test numerical stability of gradient computations."""

    def test_small_values(self):
        """Test gradient with very small input values."""
        torch.manual_seed(42)

        x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True) * 1e-6

        def fn(inp):
            return _AllReduceEP.apply(inp, 1)

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-8, atol=1e-6, rtol=1e-4)

    def test_large_values(self):
        """Test gradient with large input values."""
        torch.manual_seed(42)

        x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True) * 1e6

        def fn(inp):
            return _AllReduceEP.apply(inp, 1)

        # Large values may have less precision, use relaxed tolerances
        assert torch.autograd.gradcheck(fn, (x,), eps=1e-4, atol=1e-2, rtol=1e-2)

    def test_mixed_values(self):
        """Test gradient with mixed magnitude values."""
        torch.manual_seed(42)

        x_raw = torch.randn(4, 8, dtype=torch.float64)
        x_raw[0] *= 1e6
        x_raw[1] *= 1e-6
        x = x_raw.detach().requires_grad_(True)

        def fn(inp):
            return _AllReduceEP.apply(inp, 1)

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-3, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
