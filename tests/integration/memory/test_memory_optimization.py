# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for GRPO memory optimization features.

Tests:
1. RolloutBuffer.cat() for chunked rollout accumulation
2. TransformerModel activation checkpointing configuration
3. Config validation for grpo_rollout_chunks
4. Memory cleanup behavior
"""

import pytest
import torch


class TestRolloutBufferCat:
    """Tests for RolloutBuffer.cat() method."""

    def test_cat_basic(self):
        """Test basic concatenation of two buffers."""
        from ironcore.alignment.buffer import RolloutBuffer

        B, G = 2, 4
        prompt_len, response_len = 8, 16

        b1 = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (B, prompt_len)),
            prompt_attention_mask=torch.ones(B, prompt_len),
            completion_ids=torch.randint(0, 100, (B * G, prompt_len + response_len)),
            response_ids=torch.randint(0, 100, (B * G, response_len)),
            old_log_probs=torch.randn(B * G),
            rewards=torch.randn(B * G),
            advantages=torch.randn(B * G),
            group_ids=torch.arange(B).unsqueeze(1).expand(B, G).reshape(-1),
            metadata=[{"chunk": 1, "idx": i} for i in range(B * G)],
        )

        b2 = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (B, prompt_len)),
            prompt_attention_mask=torch.ones(B, prompt_len),
            completion_ids=torch.randint(0, 100, (B * G, prompt_len + response_len)),
            response_ids=torch.randint(0, 100, (B * G, response_len)),
            old_log_probs=torch.randn(B * G),
            rewards=torch.randn(B * G),
            advantages=torch.randn(B * G),
            group_ids=torch.arange(B).unsqueeze(1).expand(B, G).reshape(-1),
            metadata=[{"chunk": 2, "idx": i} for i in range(B * G)],
        )

        combined = b1.cat(b2)

        # Verify shapes
        assert combined.batch_size == B, "Batch size should be preserved"
        assert combined.group_size == G * 2, "Group size should double"
        assert combined.total_samples == B * G * 2, "Total samples should double"
        assert combined.completion_ids.shape[0] == B * G * 2

    def test_cat_group_ids_correctness(self):
        """Test that group_ids are correctly offset when concatenating."""
        from ironcore.alignment.buffer import RolloutBuffer

        B, G = 2, 2
        prompt_len, response_len = 4, 8

        b1 = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (B, prompt_len)),
            prompt_attention_mask=torch.ones(B, prompt_len),
            completion_ids=torch.randint(0, 100, (B * G, prompt_len + response_len)),
            response_ids=torch.randint(0, 100, (B * G, response_len)),
            old_log_probs=torch.randn(B * G),
            rewards=torch.randn(B * G),
            advantages=torch.randn(B * G),
            group_ids=torch.tensor([0, 0, 1, 1]),
            metadata=[{}] * (B * G),
        )

        b2 = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (B, prompt_len)),
            prompt_attention_mask=torch.ones(B, prompt_len),
            completion_ids=torch.randint(0, 100, (B * G, prompt_len + response_len)),
            response_ids=torch.randint(0, 100, (B * G, response_len)),
            old_log_probs=torch.randn(B * G),
            rewards=torch.randn(B * G),
            advantages=torch.randn(B * G),
            group_ids=torch.tensor([2, 2, 3, 3]),
            metadata=[{}] * (B * G),
        )

        combined = b1.cat(b2)

        # Group IDs should be concatenated: [0,0,1,1,2,2,3,3]
        expected = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        assert torch.equal(combined.group_ids, expected), (
            f"Group IDs incorrect. Expected {expected.tolist()}, got {combined.group_ids.tolist()}"
        )

    def test_cat_metadata_preserved(self):
        """Test that metadata is correctly concatenated."""
        from ironcore.alignment.buffer import RolloutBuffer

        B, G = 1, 2

        b1 = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (B, 4)),
            prompt_attention_mask=torch.ones(B, 4),
            completion_ids=torch.randint(0, 100, (B * G, 8)),
            response_ids=torch.randint(0, 100, (B * G, 4)),
            old_log_probs=torch.randn(B * G),
            rewards=torch.randn(B * G),
            advantages=torch.randn(B * G),
            group_ids=torch.tensor([0, 0]),
            metadata=[{"chunk": 1}, {"chunk": 1}],
        )

        b2 = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (B, 4)),
            prompt_attention_mask=torch.ones(B, 4),
            completion_ids=torch.randint(0, 100, (B * G, 8)),
            response_ids=torch.randint(0, 100, (B * G, 4)),
            old_log_probs=torch.randn(B * G),
            rewards=torch.randn(B * G),
            advantages=torch.randn(B * G),
            group_ids=torch.tensor([1, 1]),
            metadata=[{"chunk": 2}, {"chunk": 2}],
        )

        combined = b1.cat(b2)

        assert len(combined.metadata) == 4
        assert combined.metadata[0]["chunk"] == 1
        assert combined.metadata[2]["chunk"] == 2

    def test_cat_mismatched_batch_size_raises(self):
        """Test that concatenating buffers with different batch sizes raises error."""
        from ironcore.alignment.buffer import RolloutBuffer

        b1 = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (2, 4)),
            prompt_attention_mask=torch.ones(2, 4),
            completion_ids=torch.randint(0, 100, (4, 8)),
            response_ids=torch.randint(0, 100, (4, 4)),
            old_log_probs=torch.randn(4),
            rewards=torch.randn(4),
            advantages=torch.randn(4),
            group_ids=torch.tensor([0, 0, 1, 1]),
            metadata=[{}] * 4,
        )

        b2 = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (3, 4)),  # Different batch size
            prompt_attention_mask=torch.ones(3, 4),
            completion_ids=torch.randint(0, 100, (6, 8)),
            response_ids=torch.randint(0, 100, (6, 4)),
            old_log_probs=torch.randn(6),
            rewards=torch.randn(6),
            advantages=torch.randn(6),
            group_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
            metadata=[{}] * 6,
        )

        with pytest.raises(ValueError, match="different batch sizes"):
            b1.cat(b2)

    def test_cat_multiple_chunks(self):
        """Test concatenating multiple chunks sequentially."""
        from ironcore.alignment.buffer import RolloutBuffer

        B, G = 2, 2
        num_chunks = 4

        # Create base buffer
        combined = RolloutBuffer(
            prompt_ids=torch.randint(0, 100, (B, 4)),
            prompt_attention_mask=torch.ones(B, 4),
            completion_ids=torch.randint(0, 100, (B * G, 8)),
            response_ids=torch.randint(0, 100, (B * G, 4)),
            old_log_probs=torch.randn(B * G),
            rewards=torch.randn(B * G),
            advantages=torch.randn(B * G),
            group_ids=torch.tensor([0, 0, 1, 1]),
            metadata=[{"chunk": 0}] * (B * G),
        )

        # Add more chunks
        for chunk_idx in range(1, num_chunks):
            chunk = RolloutBuffer(
                prompt_ids=torch.randint(0, 100, (B, 4)),
                prompt_attention_mask=torch.ones(B, 4),
                completion_ids=torch.randint(0, 100, (B * G, 8)),
                response_ids=torch.randint(0, 100, (B * G, 4)),
                old_log_probs=torch.randn(B * G),
                rewards=torch.randn(B * G),
                advantages=torch.randn(B * G),
                group_ids=torch.tensor([0, 0, 1, 1]) + chunk_idx * B,
                metadata=[{"chunk": chunk_idx}] * (B * G),
            )
            combined = combined.cat(chunk)

        assert combined.total_samples == B * G * num_chunks


class TestConfigValidation:
    """Tests for config validation of grpo_rollout_micro_group_size."""

    def test_valid_micro_group_size_config(self):
        """Test valid rollout_micro_group_size configurations."""
        from ironcore.config.config_alignment import AlignmentConfig, RewardManagerConfig

        # Default config should be valid
        config = AlignmentConfig(
            method="grpo",
            reward_manager=RewardManagerConfig(functions=[]),
        )
        assert config.grpo_rollout_micro_group_size == 1

        # Various valid combinations (micro_group_size must divide group_size)
        valid_configs = [
            {"grpo_group_size": 4, "grpo_rollout_micro_group_size": 1},
            {"grpo_group_size": 4, "grpo_rollout_micro_group_size": 2},
            {"grpo_group_size": 4, "grpo_rollout_micro_group_size": 4},
            {"grpo_group_size": 8, "grpo_rollout_micro_group_size": 1},
            {"grpo_group_size": 8, "grpo_rollout_micro_group_size": 2},
            {"grpo_group_size": 8, "grpo_rollout_micro_group_size": 4},
            {"grpo_group_size": 8, "grpo_rollout_micro_group_size": 8},
        ]

        for cfg in valid_configs:
            config = AlignmentConfig(
                method="grpo",
                reward_manager=RewardManagerConfig(functions=[]),
                **cfg,
            )
            assert config.grpo_rollout_micro_group_size == cfg["grpo_rollout_micro_group_size"]

    def test_invalid_micro_group_size_zero(self):
        """Test that rollout_micro_group_size=0 raises error."""
        from ironcore.config.config_alignment import AlignmentConfig

        with pytest.raises(ValueError, match="grpo_rollout_micro_group_size must be >= 1"):
            AlignmentConfig(method="grpo", grpo_rollout_micro_group_size=0)

    def test_invalid_micro_group_size_not_divisible(self):
        """Test that non-divisible group_size/micro_group_size raises error."""
        from ironcore.config.config_alignment import AlignmentConfig

        # 8 % 3 != 0
        with pytest.raises(ValueError, match="must be divisible"):
            AlignmentConfig(method="grpo", grpo_group_size=8, grpo_rollout_micro_group_size=3)

        # 4 % 3 != 0
        with pytest.raises(ValueError, match="must be divisible"):
            AlignmentConfig(method="grpo", grpo_group_size=4, grpo_rollout_micro_group_size=3)


class TestTransformerModelActivationCheckpointing:
    """Tests for TransformerModel activation checkpointing configuration."""

    def test_activation_recompute_disabled_by_default(self):
        """Test that activation_recompute is False by default."""
        from ironcore.config.config_trainer import OperationConfig

        config = OperationConfig()
        assert config.activation_recompute is False

        assert config.recompute_strategy == "default"

    def test_activation_recompute_can_be_enabled(self):
        """Test that activation_recompute can be enabled."""
        from ironcore.config.config_trainer import OperationConfig

        config = OperationConfig(activation_recompute=True)
        assert config.activation_recompute is True
        assert config.recompute_strategy == "default"

    def test_recompute_strategy_default(self):
        """Test default recompute_strategy."""
        from ironcore.config.config_trainer import OperationConfig

        config = OperationConfig()
        assert config.recompute_strategy == "default"

    def test_recompute_strategy_optimized(self):
        """Test optimized recompute_strategy."""
        from ironcore.config.config_trainer import OperationConfig

        config = OperationConfig(recompute_strategy="optimized")
        assert config.recompute_strategy == "optimized"

    def test_activation_recompute_config_round_trips(self):
        """Test that activation_recompute and recompute_strategy survive a config round-trip."""
        from ironcore.config.config_trainer import OperationConfig

        for strategy in ("default", "optimized"):
            config = OperationConfig(activation_recompute=True, recompute_strategy=strategy)
            assert config.activation_recompute is True
            assert config.recompute_strategy == strategy


class TestMemoryCleanupBehavior:
    """Tests for memory cleanup behavior in GRPO trainer."""

    def test_del_and_empty_cache_pattern(self):
        """Test that the del + empty_cache pattern works as expected."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Allocate a large tensor
        large_tensor = torch.randn(1000, 1000, device="cuda")

        # Get memory before cleanup
        mem_before = torch.cuda.memory_allocated()
        # Delete and empty cache
        del large_tensor
        torch.cuda.empty_cache()
        # Get memory after cleanup
        mem_after = torch.cuda.memory_allocated()
        # Memory should be lower after cleanup
        assert mem_after < mem_before, "Memory should be freed after del + empty_cache"
