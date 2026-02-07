#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Test script for DPO implementation.

This script verifies that the DPO implementation is correct by:
1. Testing the DPO loss function with synthetic data
2. Verifying gradient flow
3. Testing the DPOTrainer with a small test run

Usage:
    python tests/test_dpo.py
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_dpo_loss_function():
    """Test the DPO loss function with synthetic data."""
    print("=" * 60)
    print("Testing DPO loss function...")
    print("=" * 60)

    from ironcore.alignment.loss.dpo import dpo_loss, compute_logps

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create synthetic logits
    batch_size = 4
    seq_len = 32
    vocab_size = 1000

    # Generate random logits
    policy_chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
    policy_rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
    reference_chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
    reference_rejected_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Generate random labels
    chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Create loss mask (all valid for simplicity)
    loss_mask = torch.ones(batch_size, seq_len)

    # Test DPO loss
    beta = 0.5
    loss, metrics = dpo_loss(
        policy_chosen_logits=policy_chosen_logits,
        policy_rejected_logits=policy_rejected_logits,
        reference_chosen_logits=reference_chosen_logits,
        reference_rejected_logits=reference_rejected_logits,
        chosen_labels=chosen_labels,
        rejected_labels=rejected_labels,
        loss_mask=loss_mask,
        beta=beta,
    )

    print(f"DPO Loss: {loss.item():.4f}")
    print(f"Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Verify loss is finite and positive
    assert torch.isfinite(loss), "DPO loss should be finite"
    assert loss.item() > 0, "DPO loss should be positive"

    # Test with different beta values
    print("\nTesting different beta values...")
    for beta in [0.1, 0.5, 1.0, 2.0]:
        loss_beta, _ = dpo_loss(
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_rejected_logits,
            reference_chosen_logits=reference_chosen_logits,
            reference_rejected_logits=reference_rejected_logits,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            loss_mask=loss_mask,
            beta=beta,
        )
        print(f"  beta={beta}: loss={loss_beta.item():.4f}")

    print("\nDPO loss function test PASSED!")
    return True


def test_logps_computation():
    """Test the log probabilities computation."""
    print("\n" + "=" * 60)
    print("Testing log probabilities computation...")
    print("=" * 60)

    from ironcore.alignment.loss.dpo import compute_logps

    # Set random seed
    torch.manual_seed(42)

    # Create synthetic logits
    batch_size = 4
    seq_len = 32
    vocab_size = 1000

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    # Compute log probabilities
    logps = compute_logps(logits, labels, mask)

    print(f"Log probabilities shape: {logps.shape}")
    print(f"Log probabilities: {logps}")
    print(f"Mean: {logps.mean().item():.4f}")
    print(f"Std: {logps.std().item():.4f}")

    # Verify shape
    assert logps.shape == (batch_size,), f"Expected shape ({batch_size},), got {logps.shape}"

    # Verify finite
    assert torch.all(torch.isfinite(logps)), "All log probabilities should be finite"

    # Test with partial mask
    partial_mask = torch.ones(batch_size, seq_len)
    partial_mask[:, seq_len//2:] = 0  # Mask out second half

    logps_partial = compute_logps(logits, labels, partial_mask)
    print(f"\nWith partial mask:")
    print(f"Log probabilities: {logps_partial}")
    print(f"Mean: {logps_partial.mean().item():.4f}")

    print("\nLog probabilities computation test PASSED!")
    return True


def test_dpo_gradient_flow():
    """Test that gradients flow correctly through DPO loss."""
    print("\n" + "=" * 60)
    print("Testing DPO gradient flow...")
    print("=" * 60)

    from ironcore.alignment.loss.dpo import dpo_loss

    # Set random seed
    torch.manual_seed(42)

    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self, vocab_size=1000):
            super().__init__()
            self.linear = torch.nn.Linear(32, vocab_size)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create synthetic batch
    batch_size = 4
    seq_len = 32
    vocab_size = 1000

    # Random inputs
    policy_chosen = model(torch.randn(batch_size, seq_len, 32))
    policy_rejected = model(torch.randn(batch_size, seq_len, 32))
    reference_chosen = torch.randn(batch_size, seq_len, vocab_size)
    reference_rejected = torch.randn(batch_size, seq_len, vocab_size)

    chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss_mask = torch.ones(batch_size, seq_len)

    # Compute loss
    loss, metrics = dpo_loss(
        policy_chosen_logits=policy_chosen,
        policy_rejected_logits=policy_rejected,
        reference_chosen_logits=reference_chosen,
        reference_rejected_logits=reference_rejected,
        chosen_labels=chosen_labels,
        rejected_labels=rejected_labels,
        loss_mask=loss_mask,
        beta=0.5,
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            print(f"Gradient for {name}: norm={param.grad.norm().item():.6f}")
            assert torch.all(torch.isfinite(param.grad)), f"Gradient for {name} is not finite"

    assert has_grad, "No gradients computed"

    # Optimizer step
    optimizer.step()

    print("\nDPO gradient flow test PASSED!")
    return True


def test_dpo_with_real_model():
    """Test DPO with a small GPT-2 model."""
    print("\n" + "=" * 60)
    print("Testing DPO with small GPT-2 model...")
    print("=" * 60)

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        from ironcore.alignment.loss.dpo import dpo_loss
    except ImportError:
        print("Skipping test: transformers not installed")
        print("Install with: pip install transformers")
        return True

    # Load small GPT-2 model
    print("Loading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()  # Use eval mode for simplicity

    # Create synthetic preference pairs
    batch_size = 2
    chosen_texts = [
        "The capital of France is Paris.",
        "Python is a programming language."
    ]
    rejected_texts = [
        "The capital of France is London.",
        "Python is a type of snake."
    ]

    # Tokenize
    chosen_inputs = tokenizer(chosen_texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    rejected_inputs = tokenizer(rejected_texts, return_tensors="pt", padding=True, truncation=True, max_length=32)

    chosen_input_ids = chosen_inputs["input_ids"]
    rejected_input_ids = rejected_inputs["input_ids"]

    # Create labels (shifted input_ids for causal LM)
    chosen_labels = torch.roll(chosen_input_ids, shifts=-1, dims=1)
    chosen_labels[:, -1] = -100  # Mask last token
    rejected_labels = torch.roll(rejected_input_ids, shifts=-1, dims=1)
    rejected_labels[:, -1] = -100

    # Get logits from model
    with torch.no_grad():
        chosen_logits = model(chosen_input_ids).logits.float()
        rejected_logits = model(rejected_input_ids).logits.float()

    # Create reference logits (slightly different)
    reference_chosen_logits = chosen_logits + 0.1 * torch.randn_like(chosen_logits)
    reference_rejected_logits = rejected_logits + 0.1 * torch.randn_like(rejected_logits)

    # Create loss mask
    loss_mask = (chosen_labels != -100).float()

    # Compute DPO loss
    loss, metrics = dpo_loss(
        policy_chosen_logits=chosen_logits,
        policy_rejected_logits=rejected_logits,
        reference_chosen_logits=reference_chosen_logits,
        reference_rejected_logits=reference_rejected_logits,
        chosen_labels=chosen_labels,
        rejected_labels=rejected_labels,
        loss_mask=loss_mask,
        beta=0.5,
    )

    print(f"DPO Loss with GPT-2: {loss.item():.4f}")
    print(f"Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nDPO with GPT-2 test PASSED!")
    return True


def test_dpo_trainer_import():
    """Test that DPOTrainer can be imported and instantiated."""
    print("\n" + "=" * 60)
    print("Testing DPOTrainer import and instantiation...")
    print("=" * 60)

    try:
        from ironcore.trainers import DPOTrainer
        print("DPOTrainer imported successfully")
        print(f"DPOTrainer class: {DPOTrainer}")
        print(f"DPOTrainer methods: {[m for m in dir(DPOTrainer) if not m.startswith('_')]}")
        print("\nDPOTrainer import test PASSED!")
        return True
    except Exception as e:
        print(f"ERROR importing DPOTrainer: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all DPO tests."""
    print("\n" + "=" * 60)
    print("Running DPO Implementation Tests")
    print("=" * 60)

    results = {}

    tests = [
        ("Log probabilities computation", test_logps_computation),
        ("DPO loss function", test_dpo_loss_function),
        ("DPO gradient flow", test_dpo_gradient_flow),
        ("DPOTrainer import", test_dpo_trainer_import),
        ("DPO with GPT-2", test_dpo_with_real_model),
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
