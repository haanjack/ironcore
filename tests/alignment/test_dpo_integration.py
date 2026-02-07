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

"""Integration test for DPO training.

This script tests the full DPO training pipeline with a small GPT-2 model
and synthetic preference pairs.

Usage:
    python tests/test_dpo_integration.py
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class SyntheticDPODataset(IterableDataset):
    """Synthetic DPO dataset for testing."""

    def __init__(self, num_samples=10, seq_len=64, vocab_size=50257):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_token_id = 50256  # GPT-2 EOS token

    def __iter__(self):
        for i in range(self.num_samples):
            group_id = i // 2  # Pair up samples

            # Chosen sample
            chosen_tokens = torch.randint(0, self.vocab_size - 1, (self.seq_len,))
            chosen_metadata = {
                'type': 'dpo_chosen',
                'group_id': group_id,
                'mask_ranges': [],
            }

            # Rejected sample
            rejected_tokens = torch.randint(0, self.vocab_size - 1, (self.seq_len,))
            rejected_metadata = {
                'type': 'dpo_rejected',
                'group_id': group_id,
                'mask_ranges': [],
            }

            yield {'token_ids': chosen_tokens, 'metadata': chosen_metadata}
            yield {'token_ids': rejected_tokens, 'metadata': rejected_metadata}


def create_dpo_collator(max_seq_len=64, pad_token_id=50256):
    """Create a simple DPO collator for testing."""
    from ironcore.dataloader.collator import UniversalCollator
    return UniversalCollator(
        mode="dpo",
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
        use_flash_attention=False,
        return_full_attention_mask=True,
    )


def test_dpo_training_loop():
    """Test the DPO training loop with a small model."""
    print("=" * 60)
    print("Testing DPO training loop...")
    print("=" * 60)

    from ironcore.alignment.loss.dpo import dpo_loss

    # Set random seed
    torch.manual_seed(42)

    # Create a small model
    class SimpleGPT2(torch.nn.Module):
        def __init__(self, vocab_size=1000, d_model=128, num_layers=2):
            super().__init__()
            self.vocab_size = vocab_size
            self.embedding = torch.nn.Embedding(vocab_size, d_model)
            self.layers = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
                for _ in range(num_layers)
            ])
            self.lm_head = torch.nn.Linear(d_model, vocab_size)

        def forward(self, input_ids, labels=None):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            logits = self.lm_head(x)
            return logits

    # Create policy and reference models
    vocab_size = 1000
    policy_model = SimpleGPT2(vocab_size=vocab_size)
    reference_model = SimpleGPT2(vocab_size=vocab_size)

    # Copy policy to reference and freeze
    reference_model.load_state_dict(policy_model.state_dict())
    for param in reference_model.parameters():
        param.requires_grad = False

    # Create optimizer
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-4)

    # Training loop
    num_steps = 5
    beta = 0.5
    batch_size = 4  # 2 chosen + 2 rejected
    seq_len = 64

    print(f"\nRunning {num_steps} training steps...")
    for step in range(num_steps):
        # Generate synthetic batch directly (simpler than using collator)
        chosen_input_ids = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
        rejected_input_ids = torch.randint(0, vocab_size - 1, (batch_size, seq_len))

        # For causal LM, labels are shifted input_ids
        chosen_labels = torch.cat([chosen_input_ids[:, 1:], torch.full((batch_size, 1), vocab_size - 1)], dim=1)
        rejected_labels = torch.cat([rejected_input_ids[:, 1:], torch.full((batch_size, 1), vocab_size - 1)], dim=1)

        # Forward pass through policy
        policy_chosen_logits = policy_model(chosen_input_ids, labels=None)
        policy_rejected_logits = policy_model(rejected_input_ids, labels=None)

        # Forward pass through reference (no grad)
        with torch.no_grad():
            reference_chosen_logits = reference_model(chosen_input_ids, labels=None)
            reference_rejected_logits = reference_model(rejected_input_ids, labels=None)

        # Compute DPO loss
        loss_mask = None
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

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print metrics
        print(f"  Step {step + 1}/{num_steps}: loss={loss.item():.4f}, "
              f"margin={metrics['preference_margin']:.4f}, "
              f"acc={metrics['dpo_accuracy']:.2f}")

    print("\nDPO training loop test PASSED!")
    return True


def test_dpo_collation():
    """Test DPO batch collation."""
    print("\n" + "=" * 60)
    print("Testing DPO batch collation...")
    print("=" * 60)

    from ironcore.dataloader.collator import UniversalCollator

    # Create collator
    collator = UniversalCollator(
        mode="dpo",
        max_seq_len=64,
        pad_token_id=50256,
        use_flash_attention=False,
        return_full_attention_mask=True,
    )

    # Create synthetic batch
    batch_samples = []
    for i in range(4):  # 2 pairs
        group_id = i // 2
        chosen_tokens = torch.randint(0, 50256, (32,))
        rejected_tokens = torch.randint(0, 50256, (32,))

        chosen_metadata = {'type': 'dpo_chosen', 'group_id': group_id, 'mask_ranges': []}
        rejected_metadata = {'type': 'dpo_rejected', 'group_id': group_id, 'mask_ranges': []}

        batch_samples.append({'token_ids': chosen_tokens, 'metadata': chosen_metadata})
        batch_samples.append({'token_ids': rejected_tokens, 'metadata': rejected_metadata})

    # Collate
    batch = collator(batch_samples)

    # Verify output
    assert 'chosen_input_ids' in batch, "Missing 'chosen_input_ids' in batch"
    assert 'rejected_input_ids' in batch, "Missing 'rejected_input_ids' in batch"
    assert 'chosen_labels' in batch, "Missing 'chosen_labels' in batch"
    assert 'rejected_labels' in batch, "Missing 'rejected_labels' in batch"

    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  chosen_input_ids shape: {batch['chosen_input_ids'].shape}")
    print(f"  rejected_input_ids shape: {batch['rejected_input_ids'].shape}")
    print(f"  chosen_labels shape: {batch['chosen_labels'].shape}")
    print(f"  rejected_labels shape: {batch['rejected_labels'].shape}")

    print("\nDPO collation test PASSED!")
    return True


def test_dpo_with_hf_model():
    """Test DPO with a HuggingFace GPT-2 model."""
    print("\n" + "=" * 60)
    print("Testing DPO with HuggingFace GPT-2...")
    print("=" * 60)

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("Skipping: transformers not installed")
        print("Install with: pip install transformers")
        return True

    from ironcore.alignment.loss.dpo import dpo_loss

    # Load model
    print("Loading GPT-2 (small)...")
    policy = GPT2LMHeadModel.from_pretrained("gpt2")
    reference = GPT2LMHeadModel.from_pretrained("gpt2")

    # Freeze reference
    for param in reference.parameters():
        param.requires_grad = False

    # Create optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)

    # Create synthetic DPO batch
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = 2
    seq_len = 32

    # Create random input sequences
    chosen_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    rejected_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))

    # Create labels (shifted input for causal LM)
    chosen_labels = torch.cat([chosen_input_ids[:, 1:], torch.full((batch_size, 1), -100)], dim=1)
    rejected_labels = torch.cat([rejected_input_ids[:, 1:], torch.full((batch_size, 1), -100)], dim=1)

    # Training loop
    print("\nRunning 3 training steps...")
    for step in range(3):
        # Forward pass
        policy_chosen_logits = policy(chosen_input_ids).logits
        policy_rejected_logits = policy(rejected_input_ids).logits

        with torch.no_grad():
            ref_chosen_logits = reference(chosen_input_ids).logits
            ref_rejected_logits = reference(rejected_input_ids).logits

        # Compute loss
        loss, metrics = dpo_loss(
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_rejected_logits,
            reference_chosen_logits=ref_chosen_logits,
            reference_rejected_logits=ref_rejected_logits,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            loss_mask=None,
            beta=0.5,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Step {step + 1}: loss={loss.item():.4f}, margin={metrics['preference_margin']:.4f}")

    print("\nDPO with HF GPT-2 test PASSED!")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("Running DPO Integration Tests")
    print("=" * 60)

    results = {}

    tests = [
        ("DPO collation", test_dpo_collation),
        ("DPO training loop", test_dpo_training_loop),
        ("DPO with HF GPT-2", test_dpo_with_hf_model),
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
    print("Integration Test Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL INTEGRATION TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
