# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Muon optimizer.

Tests cover:
1. Newton-Schulz orthogonalization correctness
2. Parameter classification (Muon vs AdamW)
3. Momentum buffer updates
4. Weight decay application
5. State dict save/load
6. Gradient handling
"""

import pytest
import torch
import torch.nn as nn

from ironcore.optimizer.muon import (
    MuonOptimizer,
    is_muon_param,
    zeropower_via_newtonschulz5,
)


class TestNewtonSchulz:
    """Test the Newton-Schulz 5 iteration for orthogonalization."""

    def test_orthogonalization_identity_matrix(self):
        """Identity matrix should be processed and produce semi-orthogonal output."""
        # Identity-like matrix (already orthogonal)
        G = torch.eye(64, dtype=torch.float32)

        result = zeropower_via_newtonschulz5(G, steps=5)

        # Result should have the same shape
        assert result.shape == G.shape

        # The result should be valid (no NaN/Inf)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        # The result should be approximately semi-orthogonal (singular values pushed toward 1)
        # Note: NS iteration pushes singular values toward 1, but is an approximation
        S = torch.linalg.svdvals(result.float())
        # Singular values should be in a reasonable range (between 0.3 and 1.5 for 5 iterations)
        assert (S > 0.3).all() and (S < 1.5).all()

    def test_orthogonalization_random_matrix(self):
        """Random matrix should become approximately orthogonal."""
        torch.manual_seed(42)
        G = torch.randn(64, 32, dtype=torch.float32) * 0.1

        result = zeropower_via_newtonschulz5(G, steps=5)

        # Result should have the same shape
        assert result.shape == G.shape

        # Result should be valid
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        # Check that singular values are pushed toward 1 (the goal of NS iteration)
        S = torch.linalg.svdvals(result.float())
        # Singular values should be in a reasonable range - NS pushes them toward 1
        # but is an approximation, so we check they're in a reasonable band
        assert (S > 0.5).all() and (S < 1.5).all()

    def test_tall_matrix_handling(self):
        """Tall matrices (rows > cols) should be handled correctly."""
        G = torch.randn(128, 32, dtype=torch.float32)

        result = zeropower_via_newtonschulz5(G, steps=5)

        assert result.shape == G.shape
        # Should produce valid output without NaN
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_wide_matrix_handling(self):
        """Wide matrices (rows < cols) should be handled correctly."""
        G = torch.randn(32, 128, dtype=torch.float32)

        result = zeropower_via_newtonschulz5(G, steps=5)

        assert result.shape == G.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_numerical_stability_large_values(self):
        """Large gradient values should not cause NaN."""
        G = torch.randn(64, 64, dtype=torch.float32) * 1000.0

        result = zeropower_via_newtonschulz5(G, steps=5)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_numerical_stability_small_values(self):
        """Small gradient values should not cause NaN."""
        G = torch.randn(64, 64, dtype=torch.float32) * 1e-10

        result = zeropower_via_newtonschulz5(G, steps=5)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_bfloat16_precision(self):
        """Function should work with bfloat16 internally."""
        G = torch.randn(64, 32, dtype=torch.float32)

        result = zeropower_via_newtonschulz5(G, steps=5)

        # Result should be valid
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()


class TestMuonParameterClassification:
    """Test parameter classification for Muon vs AdamW."""

    def test_attention_weights_are_muon(self):
        """Attention projection weights should use Muon."""
        # Create a fake parameter tensor (2D weight)
        weight = torch.randn(768, 768)

        # Test attention patterns
        assert is_muon_param("layers.0.self_attention.linear_q.weight", weight)
        assert is_muon_param("layers.0.self_attention.linear_kv.weight", weight)
        assert is_muon_param("layers.0.self_attention.attn_output.weight", weight)
        assert is_muon_param("layers.5.self_attention.linear_q.weight", weight)

    def test_mlp_weights_are_muon(self):
        """MLP projection weights should use Muon."""
        weight = torch.randn(3072, 768)

        assert is_muon_param("layers.0.mlp.up_proj.weight", weight)
        assert is_muon_param("layers.0.mlp.down_proj.weight", weight)
        assert is_muon_param("layers.5.mlp.up_proj.weight", weight)

    def test_embedding_uses_adamw(self):
        """Embedding weights should NOT use Muon."""
        weight = torch.randn(50257, 768)  # Vocab size x hidden dim

        assert not is_muon_param("embedding.word_embeddings.weight", weight)
        assert not is_muon_param("model.embedding.weight", weight)

    def test_output_layer_uses_adamw(self):
        """Output layer weights should NOT use Muon."""
        weight = torch.randn(50257, 768)

        assert not is_muon_param("output_layer.weight", weight)
        assert not is_muon_param("lm_head.weight", weight)

    def test_biases_use_adamw(self):
        """Bias parameters (1D) should NOT use Muon."""
        bias = torch.randn(768)  # 1D tensor

        assert not is_muon_param("layers.0.self_attention.linear_q.bias", bias)
        assert not is_muon_param("layers.0.mlp.up_proj.bias", bias)

    def test_layernorm_uses_adamw(self):
        """LayerNorm parameters should NOT use Muon."""
        weight = torch.randn(768)

        assert not is_muon_param("layers.0.input_layernorm.weight", weight)
        assert not is_muon_param("layers.0.post_attention_layernorm.weight", weight)

    def test_position_embedding_uses_adamw(self):
        """Position embedding should NOT use Muon."""
        weight = torch.randn(1024, 768)

        assert not is_muon_param("embedding.position_embedding.weight", weight)
        assert not is_muon_param("pos_embedding.weight", weight)

    def test_non_2d_uses_adamw(self):
        """Non-2D tensors should NOT use Muon."""
        # 1D tensor
        tensor_1d = torch.randn(768)
        assert not is_muon_param("layers.0.mlp.up_proj.weight", tensor_1d)

        # 3D tensor
        tensor_3d = torch.randn(12, 64, 768)
        assert not is_muon_param("layers.0.mlp.up_proj.weight", tensor_3d)

    def test_alternative_attention_naming(self):
        """Alternative attention naming patterns should work."""
        weight = torch.randn(768, 768)

        # Test alternative naming conventions
        assert is_muon_param("layers.0.attention.linear_q.weight", weight)
        assert is_muon_param("layers.0.attention.linear_kv.weight", weight)
        assert is_muon_param("layers.0.attention.attn_output.weight", weight)


class TestMuonOptimizerStep:
    """Test the optimizer step functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model with Muon and AdamW params."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                # These should use AdamW (embedding)
                self.embedding = nn.Embedding(1000, 64)
                # These should use Muon (2D hidden weights)
                self.linear1 = nn.Linear(64, 128)
                self.linear2 = nn.Linear(128, 64)
                # This should use AdamW (output layer)
                self.output = nn.Linear(64, 1000)

            def forward(self, x):
                x = self.embedding(x)
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                x = self.output(x)
                return x

        return SimpleModel()

    @pytest.fixture
    def muon_optimizer(self, simple_model):
        """Create a Muon optimizer for the simple model."""
        # Manually split params for the optimizer
        muon_params = []
        adamw_params = []

        for name, param in simple_model.named_parameters():
            if not param.requires_grad:
                continue

            if is_muon_param(name, param):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        return MuonOptimizer(
            muon_params=[{"params": muon_params, "weight_decay": 0.01}],
            adamw_params=[{"params": adamw_params, "weight_decay": 0.01}],
            lr=0.02,
            momentum=0.95,
        )

    def test_momentum_buffer_initialization(self, simple_model, muon_optimizer):
        """State should initialize momentum buffer correctly."""
        # Run one step to initialize state
        output = simple_model(torch.randint(0, 1000, (2, 10)))
        loss = output.sum()
        loss.backward()
        muon_optimizer.step()

        # Check that momentum buffers were created for Muon params
        for name, param in simple_model.named_parameters():
            if is_muon_param(name, param) and param.dim() == 2:
                state = muon_optimizer.state[param]
                assert "momentum_buffer" in state
                assert state["momentum_buffer"].shape == param.shape

    def test_adamw_state_initialization(self, simple_model, muon_optimizer):
        """AdamW params should have exp_avg and exp_avg_sq states."""
        output = simple_model(torch.randint(0, 1000, (2, 10)))
        loss = output.sum()
        loss.backward()
        muon_optimizer.step()

        # Check AdamW state for embedding
        param = simple_model.embedding.weight
        state = muon_optimizer.state[param]
        assert "exp_avg" in state
        assert "exp_avg_sq" in state

    def test_optimizer_step_updates_params(self, simple_model, muon_optimizer):
        """Parameters should change after optimizer step."""
        # Store initial params
        initial_params = {name: param.clone() for name, param in simple_model.named_parameters()}

        # Forward + backward + step
        output = simple_model(torch.randint(0, 1000, (2, 10)))
        loss = output.sum()
        loss.backward()
        muon_optimizer.step()

        # Check that params changed
        params_changed = False
        for name, param in simple_model.named_parameters():
            if not torch.equal(initial_params[name], param):
                params_changed = True
                break

        assert params_changed, "Parameters should change after optimizer step"

    def test_weight_decay_application(self, simple_model):
        """Weight decay should be applied decoupled from gradients."""
        # Create optimizer with weight decay
        muon_params = []
        adamw_params = []

        for name, param in simple_model.named_parameters():
            if is_muon_param(name, param):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        optimizer = MuonOptimizer(
            muon_params=[{"params": muon_params, "weight_decay": 0.1}],
            adamw_params=[{"params": adamw_params, "weight_decay": 0.1}],
            lr=0.01,
            momentum=0.95,
        )

        # Get a param with weight decay
        param = simple_model.linear1.weight
        initial_value = param.clone()

        # Zero gradient (so only weight decay should apply)
        param.grad = torch.zeros_like(param)
        optimizer.step()

        # Weight should have decayed (multiplied by (1 - lr * weight_decay))
        expected = initial_value * (1 - 0.01 * 0.1)
        assert torch.allclose(param, expected, atol=1e-6)

    def test_state_dict_roundtrip(self, simple_model, muon_optimizer):
        """State dict save/load should preserve optimizer state."""
        # Run a few steps to build state
        for _ in range(3):
            output = simple_model(torch.randint(0, 1000, (2, 10)))
            loss = output.sum()
            loss.backward()
            muon_optimizer.step()
            muon_optimizer.zero_grad()

        # Save state
        state_dict = muon_optimizer.state_dict()

        # Create new optimizer and load state
        muon_params = []
        adamw_params = []
        for name, param in simple_model.named_parameters():
            if is_muon_param(name, param):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        new_optimizer = MuonOptimizer(
            muon_params=[{"params": muon_params}],
            adamw_params=[{"params": adamw_params}],
            lr=0.02,
            momentum=0.95,
        )
        new_optimizer.load_state_dict(state_dict)

        # Verify states match
        for param in simple_model.parameters():
            if param in muon_optimizer.state:
                old_state = muon_optimizer.state[param]
                new_state = new_optimizer.state[param]
                for key in old_state:
                    if isinstance(old_state[key], torch.Tensor):
                        assert torch.equal(old_state[key], new_state[key])

    def test_zero_grad(self, simple_model, muon_optimizer):
        """zero_grad should clear all gradients."""
        output = simple_model(torch.randint(0, 1000, (2, 10)))
        loss = output.sum()
        loss.backward()

        # Check grads exist
        has_grad = any(p.grad is not None for p in simple_model.parameters())
        assert has_grad

        # Zero grads
        muon_optimizer.zero_grad()

        # Check grads are cleared (set_to_none=True by default)
        for param in simple_model.parameters():
            assert param.grad is None


class TestMuonOptimizerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_param_groups_raises(self):
        """Optimizer should raise error for empty param groups (PyTorch requirement)."""
        # PyTorch optimizer requires at least one parameter
        with pytest.raises(ValueError, match="empty parameter"):
            MuonOptimizer(
                muon_params=[],
                adamw_params=[],
                lr=0.02,
            )

    def test_muon_only_params(self):
        """Optimizer should work with only Muon params."""
        param = torch.randn(10, 10, requires_grad=True)
        optimizer = MuonOptimizer(
            muon_params=[{"params": [param]}],
            adamw_params=[],
            lr=0.02,
        )
        param.grad = torch.randn_like(param)
        optimizer.step()  # Should not raise
        assert not torch.isnan(param).any()

    def test_adamw_only_params(self):
        """Optimizer should work with only AdamW params."""
        param = torch.randn(10, requires_grad=True)
        optimizer = MuonOptimizer(
            muon_params=[],
            adamw_params=[{"params": [param]}],
            lr=0.02,
        )
        param.grad = torch.randn_like(param)
        optimizer.step()  # Should not raise
        assert not torch.isnan(param).any()

    def test_sparse_gradient_raises(self):
        """AdamW params with sparse gradients should raise error."""
        model = nn.Embedding(100, 10, sparse=True)
        param = model.weight

        optimizer = MuonOptimizer(
            muon_params=[],
            adamw_params=[{"params": [param]}],
            lr=0.02,
        )

        # Create sparse gradient
        x = torch.randint(0, 100, (5,))
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Should raise for sparse gradients
        with pytest.raises(RuntimeError, match="sparse"):
            optimizer.step()

    def test_repr(self):
        """Repr should show useful information."""
        optimizer = MuonOptimizer(
            muon_params=[{"params": [torch.randn(10, 10)]}],
            adamw_params=[{"params": [torch.randn(10)]}],
            lr=0.02,
            momentum=0.95,
        )

        repr_str = repr(optimizer)
        assert "MuonOptimizer" in repr_str
        assert "muon_groups" in repr_str
        assert "adamw_groups" in repr_str
        assert "momentum" in repr_str


class TestMuonVsAdamBehavior:
    """Compare Muon behavior with expected mathematical behavior."""

    def test_nesterov_momentum_direction(self):
        """Nesterov momentum should look ahead in gradient direction."""
        # Create a simple 2D parameter with known gradient
        param = torch.randn(4, 4, requires_grad=True)

        optimizer = MuonOptimizer(
            muon_params=[{"params": [param]}],
            adamw_params=[],
            lr=1.0,  # Large LR to see effect clearly
            momentum=0.9,
            nesterov=True,
        )

        # Set a gradient
        param.grad = torch.ones_like(param)

        # Store momentum buffer after first step
        optimizer.step()
        momentum_after_step1 = optimizer.state[param]["momentum_buffer"].clone()

        # Reset and do another step
        param.grad = torch.ones_like(param) * 2.0
        optimizer.step()
        momentum_after_step2 = optimizer.state[param]["momentum_buffer"].clone()

        # Momentum should have accumulated
        assert torch.norm(momentum_after_step2) > torch.norm(momentum_after_step1)

    def test_orthogonalization_effect(self):
        """Newton-Schulz should produce different updates than raw gradient."""
        torch.manual_seed(42)

        # Create parameter with gradient
        param = torch.randn(16, 16, requires_grad=True)
        param.grad = torch.randn_like(param)

        optimizer = MuonOptimizer(
            muon_params=[{"params": [param]}],
            adamw_params=[],
            lr=1.0,
            momentum=0.0,  # No momentum to isolate NS effect
            nesterov=False,
        )

        # Store original param
        original_param = param.clone()

        # Take step
        optimizer.step()

        # Param should have changed
        assert not torch.equal(param, original_param)

        # The change should not be just -lr * grad (due to orthogonalization)
        direct_update = original_param - param.grad
        assert not torch.equal(param, direct_update)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
