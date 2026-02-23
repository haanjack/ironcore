# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for load balancing loss functions in MoE.

Tests cover:
- compute_load_balance_loss() correctness
- compute_router_z_loss() correctness
- LoadBalanceLoss class forward/backward
- get_expert_utilization() correctness
- Edge cases: empty batches, single token, all tokens to one expert
- z_loss_weight > 0 path
"""

import pytest
import torch

from ironcore.layers.moe.load_balance_loss import (
    LoadBalanceLoss,
    compute_load_balance_loss,
    compute_router_z_loss,
    get_expert_utilization,
)


class TestComputeLoadBalanceLoss:
    """Tests for compute_load_balance_loss function."""

    def test_basic_load_balance_loss(self):
        """Test basic load balance loss computation."""
        batch_size, seq_len, num_experts, top_k = 2, 16, 8, 2

        # Create uniform routing logits (equal probability for all experts)
        router_logits = torch.zeros(batch_size, seq_len, num_experts)
        topk_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))

        loss = compute_load_balance_loss(
            router_logits=router_logits,
            topk_indices=topk_indices,
            num_experts=num_experts,
            alpha=0.01,
        )

        # Loss should be positive
        assert loss.item() >= 0.0
        # Loss should be a scalar
        assert loss.ndim == 0

    def test_perfect_balance_lower_loss(self):
        """Test that perfectly balanced routing has lower loss."""
        batch_size, seq_len, num_experts = 1, 16, 4

        # Uniform routing logits
        router_logits = torch.zeros(batch_size, seq_len, num_experts)

        # Perfectly balanced: each expert gets exactly 4 tokens
        topk_indices_balanced = torch.tensor([[[0], [1], [2], [3]] * 4])

        # Imbalanced: all tokens go to expert 0
        topk_indices_imbalanced = torch.zeros(1, 16, 1, dtype=torch.long)

        loss_balanced = compute_load_balance_loss(
            router_logits, topk_indices_balanced, num_experts, alpha=0.01
        )
        loss_imbalanced = compute_load_balance_loss(
            router_logits, topk_indices_imbalanced, num_experts, alpha=0.01
        )

        # Balanced should have lower loss
        assert loss_balanced.item() < loss_imbalanced.item()

    def test_alpha_scales_loss(self):
        """Test that alpha parameter scales the loss."""
        batch_size, seq_len, num_experts, top_k = 2, 8, 4, 2

        router_logits = torch.randn(batch_size, seq_len, num_experts)
        topk_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))

        loss_alpha_01 = compute_load_balance_loss(
            router_logits, topk_indices, num_experts, alpha=0.01
        )
        loss_alpha_10 = compute_load_balance_loss(
            router_logits, topk_indices, num_experts, alpha=0.10
        )

        # Higher alpha should give proportionally higher loss
        assert torch.isclose(loss_alpha_10, loss_alpha_01 * 10, rtol=1e-4)

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        batch_size, seq_len, num_experts, top_k = 2, 8, 4, 2

        router_logits = torch.randn(batch_size, seq_len, num_experts, requires_grad=True)
        topk_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))

        loss = compute_load_balance_loss(router_logits, topk_indices, num_experts, alpha=0.01)
        loss.backward()

        assert router_logits.grad is not None
        assert router_logits.grad.shape == router_logits.shape

    def test_single_token(self):
        """Test with single token (edge case)."""
        batch_size, seq_len, num_experts, top_k = 1, 1, 4, 1

        router_logits = torch.randn(batch_size, seq_len, num_experts)
        topk_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))

        loss = compute_load_balance_loss(router_logits, topk_indices, num_experts, alpha=0.01)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss)


class TestComputeRouterZLoss:
    """Tests for compute_router_z_loss function."""

    def test_basic_z_loss(self):
        """Test basic z-loss computation."""
        batch_size, seq_len, num_experts = 2, 8, 4

        router_logits = torch.randn(batch_size, seq_len, num_experts)

        z_loss = compute_router_z_loss(router_logits, z_loss_weight=0.001)

        # Z-loss should be positive
        assert z_loss.item() >= 0.0
        # Should be scalar
        assert z_loss.ndim == 0

    def test_large_logits_higher_loss(self):
        """Test that larger logits produce higher z-loss."""
        batch_size, seq_len, num_experts = 2, 8, 4

        small_logits = torch.randn(batch_size, seq_len, num_experts) * 0.1
        large_logits = torch.randn(batch_size, seq_len, num_experts) * 10.0

        z_loss_small = compute_router_z_loss(small_logits, z_loss_weight=0.001)
        z_loss_large = compute_router_z_loss(large_logits, z_loss_weight=0.001)

        # Larger logits should produce higher z-loss
        assert z_loss_large.item() > z_loss_small.item()

    def test_z_loss_weight_scales(self):
        """Test that z_loss_weight scales the loss."""
        batch_size, seq_len, num_experts = 2, 8, 4

        router_logits = torch.randn(batch_size, seq_len, num_experts)

        z_loss_1 = compute_router_z_loss(router_logits, z_loss_weight=0.001)
        z_loss_2 = compute_router_z_loss(router_logits, z_loss_weight=0.002)

        assert torch.isclose(z_loss_2, z_loss_1 * 2, rtol=1e-4)

    def test_gradient_flow(self):
        """Test that gradients flow through z-loss."""
        batch_size, seq_len, num_experts = 2, 8, 4

        router_logits = torch.randn(batch_size, seq_len, num_experts, requires_grad=True)

        z_loss = compute_router_z_loss(router_logits, z_loss_weight=0.001)
        z_loss.backward()

        assert router_logits.grad is not None


class TestLoadBalanceLoss:
    """Tests for LoadBalanceLoss module."""

    def test_forward_without_z_loss(self):
        """Test forward pass without z-loss."""
        num_experts = 8
        batch_size, seq_len, top_k = 2, 16, 2

        loss_fn = LoadBalanceLoss(num_experts=num_experts, aux_loss_alpha=0.01)

        router_logits = torch.randn(batch_size, seq_len, num_experts)
        topk_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))

        loss = loss_fn(router_logits, topk_indices)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_forward_with_z_loss(self):
        """Test forward pass with z-loss enabled."""
        num_experts = 8
        batch_size, seq_len, top_k = 2, 16, 2

        loss_fn = LoadBalanceLoss(
            num_experts=num_experts,
            aux_loss_alpha=0.01,
            z_loss_weight=0.001,
        )

        router_logits = torch.randn(batch_size, seq_len, num_experts)
        topk_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))

        loss = loss_fn(router_logits, topk_indices)

        assert loss.item() >= 0.0
        # With z_loss, loss should be higher than aux_loss alone
        aux_loss_only = compute_load_balance_loss(
            router_logits, topk_indices, num_experts, alpha=0.01
        )
        assert loss.item() >= aux_loss_only.item()

    def test_backward_pass(self):
        """Test backward pass."""
        num_experts = 8
        batch_size, seq_len, top_k = 2, 16, 2

        loss_fn = LoadBalanceLoss(
            num_experts=num_experts,
            aux_loss_alpha=0.01,
            z_loss_weight=0.001,
        )

        router_logits = torch.randn(batch_size, seq_len, num_experts, requires_grad=True)
        topk_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))

        loss = loss_fn(router_logits, topk_indices)
        loss.backward()

        assert router_logits.grad is not None


class TestGetExpertUtilization:
    """Tests for get_expert_utilization function."""

    def test_basic_utilization(self):
        """Test basic expert utilization computation."""
        num_experts = 4
        _batch_size, _seq_len, _top_k = 2, 8, 2

        # Create indices that route to experts 0, 1, 2, 3 evenly
        topk_indices = torch.tensor(
            [
                [[0, 1], [2, 3], [0, 1], [2, 3]],
                [[0, 1], [2, 3], [0, 1], [2, 3]],
            ]
        )

        utilization = get_expert_utilization(topk_indices, num_experts)

        # Check shape
        assert utilization.shape == (num_experts,)
        # Check sum is 1
        assert torch.isclose(utilization.sum(), torch.tensor(1.0), atol=1e-5)
        # Check all non-negative
        assert torch.all(utilization >= 0)

    def test_perfect_balance(self):
        """Test perfectly balanced expert utilization."""
        num_experts = 4
        _batch_size, _seq_len, _top_k = 1, 4, 1

        # Each expert gets exactly one token
        topk_indices = torch.tensor([[[0], [1], [2], [3]]])

        utilization = get_expert_utilization(topk_indices, num_experts)

        # Each expert should have 0.25 utilization
        expected = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert torch.allclose(utilization, expected, atol=1e-5)

    def test_all_to_one_expert(self):
        """Test all tokens going to one expert."""
        num_experts = 4
        batch_size, seq_len, top_k = 2, 4, 1

        # All tokens go to expert 0
        topk_indices = torch.zeros(batch_size, seq_len, top_k, dtype=torch.long)

        utilization = get_expert_utilization(topk_indices, num_experts)

        assert utilization[0].item() == 1.0
        assert torch.all(utilization[1:] == 0.0)

    def test_single_token(self):
        """Test with single token."""
        num_experts = 4

        topk_indices = torch.tensor([[[0]]])

        utilization = get_expert_utilization(topk_indices, num_experts)

        assert utilization[0].item() == 1.0
        assert torch.all(utilization[1:] == 0.0)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_batch(self):
        """Test with minimal batch (batch_size=1, seq_len=1)."""
        num_experts, top_k = 4, 1

        router_logits = torch.randn(1, 1, num_experts)
        topk_indices = torch.randint(0, num_experts, (1, 1, top_k))

        loss = compute_load_balance_loss(router_logits, topk_indices, num_experts)
        utilization = get_expert_utilization(topk_indices, num_experts)

        assert not torch.isnan(loss)
        assert not torch.any(torch.isnan(utilization))

    def test_large_num_experts(self):
        """Test with large number of experts."""
        num_experts = 128
        batch_size, seq_len, top_k = 2, 16, 4

        router_logits = torch.randn(batch_size, seq_len, num_experts)
        topk_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))

        loss = compute_load_balance_loss(router_logits, topk_indices, num_experts)
        utilization = get_expert_utilization(topk_indices, num_experts)

        assert not torch.isnan(loss)
        assert utilization.shape == (num_experts,)

    def test_high_top_k(self):
        """Test with high top_k value."""
        num_experts = 8
        batch_size, seq_len = 2, 16
        top_k = 6  # Route to most experts

        router_logits = torch.randn(batch_size, seq_len, num_experts)
        topk_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))

        loss = compute_load_balance_loss(router_logits, topk_indices, num_experts)

        assert not torch.isnan(loss)
        assert loss.item() >= 0.0

    def test_zero_alpha(self):
        """Test with zero alpha (should give zero loss)."""
        num_experts = 4
        batch_size, seq_len, top_k = 2, 8, 2

        router_logits = torch.randn(batch_size, seq_len, num_experts)
        topk_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))

        loss = compute_load_balance_loss(router_logits, topk_indices, num_experts, alpha=0.0)

        assert torch.isclose(loss, torch.tensor(0.0))

    def test_deterministic_output(self):
        """Test that same inputs give same outputs."""
        num_experts = 8
        batch_size, seq_len, top_k = 2, 16, 2

        torch.manual_seed(42)
        router_logits = torch.randn(batch_size, seq_len, num_experts)
        topk_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))

        loss1 = compute_load_balance_loss(router_logits, topk_indices, num_experts)
        loss2 = compute_load_balance_loss(router_logits, topk_indices, num_experts)

        assert torch.isclose(loss1, loss2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
