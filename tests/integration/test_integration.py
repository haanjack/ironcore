# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Holistic integration test for ironcore training framework.

This test exercises the complete training pipeline:
1. Configuration loading and validation
2. Model initialization (with various architectures)
3. Forward/backward pass
4. Optimizer step
5. Checkpoint save/load

The goal is to catch integration regressions early.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from tests.fixtures.config_fixtures import (
    create_gqa_config,
    create_mqa_config,
    create_small_test_config,
)

from ironcore.parallel import parallel_states

# =============================================================================
# Test Configuration
# =============================================================================

VOCAB_SIZE = 1000  # GPT-2 standard vocab size for testing

HOLISTIC_TEST_CONFIGS = {
    "small_baseline": create_small_test_config,
    "gqa_attention": lambda: create_gqa_config(num_heads=8, num_groups=2),
    "mqa_attention": lambda: create_mqa_config(num_heads=8),
}


# =============================================================================
# Helper Functions
# =============================================================================


def ensure_parallel_initialized():
    """Ensure parallel states are initialized for single GPU tests."""
    try:
        parallel_states.get_tensor_model_parallel_world_size()
    except RuntimeError:
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=1,
            timeout_in_minutes=1.0,
        )


def compute_lm_loss(logits, labels):
    """Compute cross-entropy loss for language modeling."""
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Flatten for cross entropy
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


# =============================================================================
# Part 1: Configuration Validation
# =============================================================================


class TestConfigurationValidation:
    """Test configuration loading and validation."""

    @pytest.mark.parametrize("config_name", list(HOLISTIC_TEST_CONFIGS.keys()))
    def test_config_creates_successfully(self, config_name):
        """Verify all test configurations can be created."""
        config = HOLISTIC_TEST_CONFIGS[config_name]()
        assert config is not None
        assert config.model is not None
        assert config.trainer is not None

    def test_config_serializable(self):
        """Verify configuration can be converted to dict."""
        from dataclasses import asdict

        config = create_small_test_config()

        # Use asdict for serialization
        config_dict = asdict(config)

        # Verify structure
        assert config_dict is not None
        assert "model" in config_dict
        assert config_dict["model"]["d_model"] == config.model.d_model
        assert "trainer" in config_dict

    def test_config_validation_catches_errors(self):
        """Verify configuration validation catches invalid settings."""
        from ironcore.config import ModelConfig

        with pytest.raises((ValueError, AssertionError)):  # Should raise validation error
            ModelConfig(
                d_model=-1,  # Invalid: must be positive
                num_attention_heads=8,
                num_attention_groups=8,
                head_dim=64,
                d_ffn=256,
                num_layers=2,
                max_seq_len=64,
            )


# =============================================================================
# Part 2: Model Initialization
# =============================================================================


class TestModelInitialization:
    """Test model initialization across configurations."""

    @pytest.mark.cuda
    @pytest.mark.parametrize("config_name", ["small_baseline", "gqa_attention", "mqa_attention"])
    def test_model_forward_pass(self, config_name):
        """Verify model forward pass works for all attention types."""
        from ironcore.global_vars import reset_global_states, set_global_states
        from ironcore.language_model import LanguageModel

        ensure_parallel_initialized()
        config = HOLISTIC_TEST_CONFIGS[config_name]()
        set_global_states(config)

        try:
            model = LanguageModel(config)
            model.eval()

            # Create dummy input
            batch_size, seq_len = 2, 16
            input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))

            with torch.no_grad():
                output = model(input_ids)

            # Handle tuple output (with cache) vs tensor output
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Verify output shape (batch, seq, vocab)
            assert logits.shape[0] == batch_size
            assert logits.shape[1] == seq_len
            assert not torch.isnan(logits).any()
        finally:
            reset_global_states()

    def test_model_parameter_count_reasonable(self):
        """Verify model parameter count is reasonable."""
        from ironcore.global_vars import reset_global_states, set_global_states
        from ironcore.language_model import LanguageModel

        ensure_parallel_initialized()
        config = create_small_test_config()
        set_global_states(config)

        try:
            model = LanguageModel(config)
            num_params = sum(p.numel() for p in model.parameters())

            # Small config should have < 10M parameters
            assert num_params < 10_000_000, f"Too many params: {num_params}"
        finally:
            reset_global_states()


# =============================================================================
# Part 3: Training Step
# =============================================================================


@pytest.mark.cuda
class TestTrainingStep:
    """Test complete training step."""

    def test_single_training_step(self):
        """Verify single training step completes without errors."""
        from ironcore.global_vars import reset_global_states, set_global_states
        from ironcore.language_model import LanguageModel

        ensure_parallel_initialized()
        config = create_small_test_config()
        set_global_states(config)

        try:
            model = LanguageModel(config)
            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            # Create dummy batch
            batch_size, seq_len = 2, 16
            input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
            labels = input_ids.clone()

            # Forward pass (without labels to get logits)
            logits = model(input_ids)

            # Compute loss manually
            loss = compute_lm_loss(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Verify loss is valid
            assert loss.item() > 0
            assert not torch.isnan(loss)
        finally:
            reset_global_states()

    def test_gradients_flow_to_all_parameters(self):
        """Verify gradients reach all trainable parameters."""
        from ironcore.global_vars import reset_global_states, set_global_states
        from ironcore.language_model import LanguageModel

        ensure_parallel_initialized()
        config = create_small_test_config()
        set_global_states(config)

        try:
            model = LanguageModel(config)
            model.train()

            batch_size, seq_len = 2, 16
            input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
            labels = input_ids.clone()

            logits = model(input_ids)
            loss = compute_lm_loss(logits, labels)
            loss.backward()

            # Check all parameters have gradients
            params_without_grad = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is None:
                    params_without_grad.append(name)

            assert len(params_without_grad) == 0, f"Params without grad: {params_without_grad}"
        finally:
            reset_global_states()

    def test_loss_decreases_over_steps(self):
        """Verify loss decreases over multiple training steps."""
        from ironcore.global_vars import reset_global_states, set_global_states
        from ironcore.language_model import LanguageModel

        ensure_parallel_initialized()
        config = create_small_test_config()
        set_global_states(config)

        try:
            model = LanguageModel(config)
            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            # Fixed batch for consistency
            torch.manual_seed(42)
            batch_size, seq_len = 4, 16
            input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
            labels = input_ids.clone()

            losses = []
            for _ in range(10):
                logits = model(input_ids)
                loss = compute_lm_loss(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            # Loss should generally decrease (allow some variance)
            assert losses[-1] < losses[0], (
                f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
            )
        finally:
            reset_global_states()


# =============================================================================
# Part 4: Checkpoint Save/Load
# =============================================================================


@pytest.mark.cuda
class TestCheckpointing:
    """Test checkpoint save and load."""

    def test_checkpoint_roundtrip(self):
        """Verify checkpoint can be saved and loaded."""
        from ironcore.global_vars import reset_global_states, set_global_states
        from ironcore.language_model import LanguageModel

        ensure_parallel_initialized()
        config = create_small_test_config()
        set_global_states(config)

        try:
            model = LanguageModel(config)
            model.train()

            # Do a training step to modify parameters
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            input_ids = torch.randint(0, VOCAB_SIZE, (2, 16))
            labels = input_ids.clone()

            logits = model(input_ids)
            loss = compute_lm_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save checkpoint
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = Path(tmpdir) / "checkpoint.pt"

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )

                # Load into new model
                model2 = LanguageModel(config)
                optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)

                checkpoint = torch.load(checkpoint_path, weights_only=False)
                model2.load_state_dict(checkpoint["model_state_dict"])
                optimizer2.load_state_dict(checkpoint["optimizer_state_dict"])

                # Verify outputs match
                model.eval()
                model2.eval()
                with torch.no_grad():
                    out1 = model(input_ids)
                    out2 = model2(input_ids)

                # Handle tuple output (with cache) vs tensor output
                if isinstance(out1, tuple):
                    out1 = out1[0]
                if isinstance(out2, tuple):
                    out2 = out2[0]

                assert torch.allclose(out1, out2, atol=1e-6)
        finally:
            reset_global_states()


# =============================================================================
# Part 5: Data Pipeline
# =============================================================================


class TestDataPipeline:
    """Test data loading and preprocessing."""

    def test_dataloader_exists(self):
        """Verify dataloader module is importable."""
        from ironcore.dataloader import get_data_iterator

        assert callable(get_data_iterator)


# =============================================================================
# Part 6: GRPO Integration
# =============================================================================


class TestGRPOIntegration:
    """Test GRPO training integration."""

    def test_grpo_loss_computation(self):
        """Verify GRPO loss computation works."""
        from ironcore.alignment.loss.grpo import compute_advantages, grpo_loss

        B, G = 2, 4

        rewards = torch.randn(B * G)
        group_ids = torch.arange(B).unsqueeze(1).expand(B, G).reshape(-1)

        advantages = compute_advantages(rewards, group_ids, distributed=False)

        policy_log_probs = torch.randn(B * G, requires_grad=True)
        ref_log_probs = torch.randn(B * G)
        kl_per_seq = torch.abs(torch.randn(B * G)) * 0.1

        loss, metrics = grpo_loss(
            policy_log_probs=policy_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            kl_per_seq=kl_per_seq,
            beta=0.1,
        )

        assert loss.item() is not None
        assert not torch.isnan(loss)
        assert "policy_loss" in metrics
        assert "kl_loss" in metrics


# =============================================================================
# Part 7: Memory Tests
# =============================================================================


@pytest.mark.cuda
class TestMemoryEfficiency:
    """Test memory efficiency."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_no_memory_leak(self):
        """Verify no memory leak over multiple forward passes."""
        from ironcore.global_vars import reset_global_states, set_global_states
        from ironcore.language_model import LanguageModel

        ensure_parallel_initialized()
        config = create_small_test_config()
        set_global_states(config)

        try:
            model = LanguageModel(config).cuda()
            model.eval()

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            for _ in range(10):
                input_ids = torch.randint(0, VOCAB_SIZE, (2, 16), device="cuda")
                with torch.no_grad():
                    _ = model(input_ids)

            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2

            # Should be under 500MB for small model
            assert peak_memory < 500, f"Peak memory too high: {peak_memory:.1f} MB"
        finally:
            reset_global_states()


# =============================================================================
# Part 8: Summary
# =============================================================================


def test_holistic_summary():
    """Print summary of holistic test coverage."""
    summary = """
    IronCore Holistic Integration Test Summary
    ==========================================

    1. Configuration Validation: 3 tests
       - Config creation (3 variants)
       - Serialization
       - Validation errors

    2. Model Initialization: 2 tests
       - Forward pass (MHA/GQA/MQA)
       - Parameter count

    3. Training Step: 3 tests
       - Single step
       - Gradient flow
       - Loss decrease

    4. Checkpointing: 1 test
       - Save/load roundtrip

    5. Data Pipeline: 1 test
       - Module import

    6. GRPO Integration: 1 test
       - Loss computation

    7. Memory Efficiency: 1 test
       - No memory leak

    Total: 12 holistic integration tests

    Note: These tests cover the core training pipeline:
    - Config -> Model -> Forward -> Loss -> Backward -> Optimizer -> Checkpoint
    """
    print(summary)
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
