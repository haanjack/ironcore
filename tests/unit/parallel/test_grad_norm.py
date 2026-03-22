# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for gradient norm computation."""

import pytest
import torch

from ironcore.parallel.grad_norm import clip_grad_norm


class SimpleModel(torch.nn.Module):
    """Simple test model."""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.mark.parametrize("norm_type", [2.0, float("inf")])
def test_clip_grad_norm_basic(norm_type):
    """Test basic gradient clipping without distributed setup."""
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Forward pass
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Compute norm before clipping
    norm_before = clip_grad_norm(model.parameters(), max_norm=float("inf"), norm_type=norm_type)
    assert norm_before > 0, "Gradient norm should be positive"
    assert not torch.isnan(norm_before), "Gradient norm should not be NaN"

    # New forward/backward pass for second test
    optimizer.zero_grad()
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()

    # Test clipping with max_norm=0.5
    max_norm = 0.5
    clip_grad_norm(model.parameters(), max_norm=max_norm, norm_type=norm_type)

    # Compute actual norm of clipped gradients
    if norm_type == 2.0:
        norm_clipped = torch.norm(
            torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        )
    else:  # inf
        norm_clipped = max(
            p.grad.abs().max().item() for p in model.parameters() if p.grad is not None
        )

    # For L2 norm, clipped should be <= max_norm (with small tolerance)
    assert norm_clipped <= max_norm + 1e-5, (
        f"Actual clipped norm {norm_clipped} exceeds max_norm {max_norm}"
    )


def test_clip_grad_norm_no_gradients():
    """Test clip_grad_norm with parameters that have no gradients."""
    model = SimpleModel()

    # Parameters without gradients
    norm = clip_grad_norm(model.parameters(), max_norm=1.0)

    assert norm == 0.0 or torch.isclose(norm, torch.tensor(0.0)), (
        "Norm should be 0 when no gradients exist"
    )


def test_clip_grad_norm_single_tensor():
    """Test clip_grad_norm with a single tensor as input."""
    x = torch.randn(10, requires_grad=True)
    y = torch.randn(10)

    loss = torch.nn.functional.mse_loss(x, y)
    loss.backward()

    norm = clip_grad_norm(x, max_norm=1.0)
    assert norm > 0, "Gradient norm should be positive"
    assert not torch.isnan(norm), "Gradient norm should not be NaN"


def test_clip_grad_norm_clipping_effect():
    """Test that clipping actually reduces gradient magnitude."""
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Large gradients
    x = torch.randn(32, 10)
    y = torch.randn(32, 5)
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y) * 100  # Large loss to get large gradients

    optimizer.zero_grad()
    loss.backward()

    # Store gradient norms before clipping
    grad_norms_before = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]

    # Clip with aggressive max_norm
    max_norm = 0.1
    clip_grad_norm(model.parameters(), max_norm=max_norm, norm_type=2.0)

    # Check that individual parameter gradients are reduced
    grad_norms_after = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]

    for before, after in zip(grad_norms_before, grad_norms_after, strict=False):
        assert after <= before + 1e-5, "Clipping should not increase gradient magnitudes"


if __name__ == "__main__":
    # Run basic smoke tests
    print("Testing clip_grad_norm with different norm types...")
    for norm_type in [2.0, float("inf")]:
        print(f"  Testing with norm_type={norm_type}")
        test_clip_grad_norm_basic(norm_type)
    print("✓ Basic tests passed")

    print("Testing with no gradients...")
    test_clip_grad_norm_no_gradients()
    print("✓ No gradient test passed")

    print("Testing with single tensor...")
    test_clip_grad_norm_single_tensor()
    print("✓ Single tensor test passed")

    print("Testing clipping effect...")
    test_clip_grad_norm_clipping_effect()
    print("✓ Clipping effect test passed")

    print("\n✓ All smoke tests passed!")
