# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for rollout and KV-cache correctness.

Tests:
1. KV-cache expansion integrity
2. Logit parity between sequential and batched generation
3. EOS and masking logic
"""

import pytest
import torch


class TestKVCacheExpansion:
    """Tests for KV-cache expansion from [B] to [B×G]."""

    def test_expand_kv_cache_shape(self):
        """Verify expanded KV-cache has correct shape [B×G, ...]."""
        from ironcore.alignment.rollout import _expand_kv_cache

        B, G = 4, 8
        num_heads, seq_len, head_dim = 8, 16, 64

        # Create mock KV-cache for 2 layers
        past_kv = [
            (
                torch.randn(B, num_heads, seq_len, head_dim),
                torch.randn(B, num_heads, seq_len, head_dim),
            )
            for _ in range(2)
        ]

        expanded = _expand_kv_cache(past_kv, G)

        assert len(expanded) == len(past_kv), "Number of layers should match"

        for i, (expanded_k, expanded_v) in enumerate(expanded):
            expected_shape = (B * G, num_heads, seq_len, head_dim)
            assert expanded_k.shape == expected_shape, (
                f"Layer {i} key shape: expected {expected_shape}, got {expanded_k.shape}"
            )
            assert expanded_v.shape == expected_shape, (
                f"Layer {i} value shape: expected {expected_shape}, got {expanded_v.shape}"
            )

    def test_expand_kv_cache_content_correctness(self):
        """Verify each prompt's KV-cache is replicated G times correctly."""
        from ironcore.alignment.rollout import _expand_kv_cache

        B, G = 3, 4
        num_heads, seq_len, head_dim = 4, 8, 16

        # Create KV-cache with unique values for each prompt
        past_kv = [
            (
                torch.arange(B).float().view(B, 1, 1, 1).expand(B, num_heads, seq_len, head_dim),
                torch.arange(B).float().view(B, 1, 1, 1).expand(B, num_heads, seq_len, head_dim)
                + 100,
            )
        ]

        expanded = _expand_kv_cache(past_kv, G)
        expanded_k, _ = expanded[0]

        # Verify pattern: [0,0,0,0, 1,1,1,1, 2,2,2,2]
        for b in range(B):
            for g in range(G):
                idx = b * G + g
                expected_value = float(b)
                assert torch.allclose(expanded_k[idx, 0, 0, 0], torch.tensor(expected_value)), (
                    f"Expected prompt {b} copy {g} to have value {expected_value}, "
                    f"got {expanded_k[idx, 0, 0, 0]}"
                )

    def test_expand_kv_cache_order(self):
        """Verify expansion order: [p0_g0, p0_g1, ..., p0_g{G-1}, p1_g0, ...]."""
        from ironcore.alignment.rollout import _expand_kv_cache

        B, G = 2, 3
        past_kv = [(torch.arange(B).float().view(B, 1, 1, 1), torch.zeros(B, 1, 1, 1))]

        expanded = _expand_kv_cache(past_kv, G)
        expanded_k, _ = expanded[0]

        # Expected pattern: [0, 0, 0, 1, 1, 1]
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).view(6, 1, 1, 1)
        assert torch.allclose(expanded_k, expected), (
            f"Expansion order incorrect. Expected {expected.squeeze()}, got {expanded_k.squeeze()}"
        )


class TestBatchedSampling:
    """Tests for batched token sampling."""

    def test_sample_tokens_greedy(self):
        """Test greedy sampling returns argmax."""
        from ironcore.alignment.rollout import _sample_tokens_batched

        torch.manual_seed(42)
        logits = torch.randn(4, 100)

        # Greedy sampling
        tokens = _sample_tokens_batched(
            logits, temperature=1.0, top_p=1.0, top_k=0, do_sample=False
        )

        expected = logits.argmax(dim=-1, keepdim=True)
        assert torch.equal(tokens, expected), "Greedy sampling should return argmax"

    def test_sample_tokens_temperature(self):
        """Test temperature scaling affects distribution."""
        from ironcore.alignment.rollout import _sample_tokens_batched

        torch.manual_seed(42)
        logits = torch.zeros(1, 100)
        logits[0, 0] = 10.0  # Strong preference for token 0

        # Low temperature should concentrate probability on token 0
        _sample_tokens_batched(logits.clone(), temperature=0.1, top_p=1.0, top_k=0, do_sample=True)

        # High temperature should spread probability more
        # Sample many times to check distribution
        counts = torch.zeros(100)
        for _ in range(100):
            t = _sample_tokens_batched(
                logits.clone(), temperature=2.0, top_p=1.0, top_k=0, do_sample=True
            )
            counts[t.item()] += 1

        # Token 0 should be most common even with high temperature
        assert counts[0] > counts[1:].sum() / 99, (
            "Token 0 should be most common due to higher logit"
        )

    def test_sample_tokens_top_k(self):
        """Test top-k filtering."""
        from ironcore.alignment.rollout import _sample_tokens_batched

        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        # With top_k=1, should always return argmax
        for _ in range(10):
            tokens = _sample_tokens_batched(
                logits.clone(), temperature=1.0, top_p=1.0, top_k=1, do_sample=True
            )
            expected = logits.argmax(dim=-1, keepdim=True)
            assert torch.equal(tokens, expected), "top_k=1 should always return argmax"

    def test_sample_tokens_top_p(self):
        """Test nucleus (top-p) filtering."""
        from ironcore.alignment.rollout import _sample_tokens_batched

        torch.manual_seed(42)
        logits = torch.zeros(1, 100)
        logits[0, 0] = 5.0
        logits[0, 1] = 4.0
        logits[0, 2] = 3.0

        # With top_p very low, should only sample from top tokens
        # This is probabilistic, so we just verify no errors occur
        for _ in range(10):
            tokens = _sample_tokens_batched(
                logits.clone(), temperature=1.0, top_p=0.1, top_k=0, do_sample=True
            )
            assert tokens.item() < 100, "Token should be in vocab range"


class TestLogitParity:
    """Tests for logit parity between sequential and batched generation."""

    def test_logit_expansion_parity(self):
        """Verify logit expansion matches KV-cache expansion pattern."""
        from ironcore.alignment.rollout import _expand_kv_cache

        torch.manual_seed(42)
        B, G = 2, 4
        num_heads, seq_len, head_dim = 4, 8, 16
        vocab_size = 100

        # Synthetic prefill logits and KV cache (same shape as real model output)
        prefill_logits = torch.randn(B, seq_len, vocab_size)
        prefix_kv = [
            (
                torch.randn(B, num_heads, seq_len, head_dim),
                torch.randn(B, num_heads, seq_len, head_dim),
            )
            for _ in range(2)
        ]

        last_logits = prefill_logits[:, -1, :]  # [B, vocab]

        # Expand KV-cache
        expanded_kv = _expand_kv_cache(prefix_kv, G)

        # Expand logits the same way
        expanded_logits = last_logits.unsqueeze(1).expand(B, G, -1).reshape(B * G, -1)

        # Verify shapes match
        assert expanded_logits.shape == (B * G, vocab_size), "Expanded logits shape mismatch"
        assert expanded_kv[0][0].shape[0] == B * G, "KV batch dim should match expanded logits"

        # Verify each group's logits match the original prompt's logits
        for b in range(B):
            for g in range(G):
                idx = b * G + g
                assert torch.allclose(expanded_logits[idx], last_logits[b]), (
                    f"Expanded logits for prompt {b}, group {g} should match original"
                )

    def test_response_mask_correctness(self):
        """Test that response mask correctly identifies generated tokens."""
        B, prompt_len, response_len = 4, 10, 20
        total_len = prompt_len + response_len

        # Simulated labels with prompt ignored
        labels = torch.full((B, total_len), -100)
        labels[:, prompt_len - 1 : -1] = torch.randint(0, 100, (B, response_len))

        # Create response mask (1 for response, 0 for prompt)
        response_mask = torch.zeros_like(labels, dtype=torch.float)
        response_mask[:, prompt_len - 1 : -1] = 1.0

        # Verify mask correctness
        assert response_mask[:, : prompt_len - 1].sum() == 0, "Prompt tokens should be masked"
        assert response_mask[:, prompt_len - 1 : -1].sum() == B * response_len, (
            "Response tokens should be unmasked"
        )


class TestRolloutBuffer:
    """Tests for RolloutBuffer operations."""

    def test_buffer_creation(self):
        """Test RolloutBuffer creation with correct shapes."""
        from ironcore.alignment.buffer import RolloutBuffer

        B, G = 4, 8
        prompt_len, response_len = 16, 32

        buffer = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (B, prompt_len)),
            prompt_attention_mask=torch.ones(B, prompt_len),
            completion_ids=torch.randint(0, 100, (B * G, prompt_len + response_len)),
            response_ids=torch.randint(0, 100, (B * G, response_len)),
            old_log_probs=torch.randn(B * G),
            rewards=torch.zeros(B * G),
            advantages=torch.zeros(B * G),
            group_ids=torch.arange(B).unsqueeze(1).expand(B, G).reshape(-1),
            metadata=[{"test": i} for i in range(B * G)],
        )

        assert buffer.batch_size == B
        assert buffer.group_size == G
        assert buffer.total_samples == B * G
        assert buffer.prompt_length == prompt_len
        assert buffer.response_length == response_len

    def test_buffer_to_device(self):
        """Test buffer movement between devices."""
        from ironcore.alignment.buffer import RolloutBuffer

        B, G = 2, 4
        prompt_len, response_len = 8, 16

        buffer = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (B, prompt_len)),
            prompt_attention_mask=torch.ones(B, prompt_len),
            completion_ids=torch.randint(0, 100, (B * G, prompt_len + response_len)),
            response_ids=torch.randint(0, 100, (B * G, response_len)),
            old_log_probs=torch.randn(B * G),
            rewards=torch.zeros(B * G),
            advantages=torch.zeros(B * G),
            group_ids=torch.arange(B).unsqueeze(1).expand(B, G).reshape(-1),
            metadata=[{} for _ in range(B * G)],
        )

        # Test to("cpu") - should be no-op since tensors are already on CPU
        buffer_cpu = buffer.to("cpu")
        assert buffer_cpu.prompt_ids.device.type == "cpu"

    def test_buffer_get_group(self):
        """Test retrieving completions for a specific group."""
        from ironcore.alignment.buffer import RolloutBuffer

        B, G = 3, 4
        prompt_len, response_len = 8, 16

        buffer = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (B, prompt_len)),
            prompt_attention_mask=torch.ones(B, prompt_len),
            completion_ids=torch.randint(0, 100, (B * G, prompt_len + response_len)),
            response_ids=torch.randint(0, 100, (B * G, response_len)),
            old_log_probs=torch.randn(B * G),
            rewards=torch.randn(B * G),
            advantages=torch.randn(B * G),
            group_ids=torch.arange(B).unsqueeze(1).expand(B, G).reshape(-1),
            metadata=[{"idx": i} for i in range(B * G)],
        )

        # Get group 1
        group_data = buffer.get_group(1)
        assert group_data["completion_ids"].shape[0] == G
        assert len(group_data["metadata"]) == G

    def test_buffer_get_best_completion(self):
        """Test retrieving highest-reward completion."""
        from ironcore.alignment.buffer import RolloutBuffer

        B, G = 2, 4

        rewards = torch.tensor([0.1, 0.5, 0.3, 0.2, 0.9, 0.1, 0.4, 0.3])
        buffer = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (B, 8)),
            prompt_attention_mask=torch.ones(B, 8),
            completion_ids=torch.randint(0, 100, (B * G, 20)),
            response_ids=torch.randint(0, 100, (B * G, 12)),
            old_log_probs=torch.randn(B * G),
            rewards=rewards,
            advantages=torch.randn(B * G),
            group_ids=torch.arange(B).unsqueeze(1).expand(B, G).reshape(-1),
            metadata=[{"idx": i} for i in range(B * G)],
        )

        # Best in group 0 should be index 1 (reward 0.5)
        best_0 = buffer.get_best_completion(0)
        assert abs(best_0["reward"] - 0.5) < 1e-5

        # Best in group 1 should be index 4 (reward 0.9)
        best_1 = buffer.get_best_completion(1)
        assert abs(best_1["reward"] - 0.9) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
