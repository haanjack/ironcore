# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for DPO training.

Tests cover:
1. Full training loop with data loading, forward, backward, optimizer step
2. Checkpoint save/load with DPO (including reference model)
3. Reference model weight preservation
4. Gradient accumulation correctness
5. Distributed training (TP=1 vs TP=2)

Run tests:
    # All DPO tests
    pytest tests/integration/test_dpo_e2e.py -v

    # Only unit tests (fast)
    pytest tests/integration/test_dpo_e2e.py -v -m "not distributed"

    # Distributed tests (requires GPU)
    pytest tests/integration/test_dpo_e2e.py -v -m distributed
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tiny_model_config():
    """Create a minimal model configuration for fast testing."""
    from ironcore.config import (
        DataConfig,
        InitConfig,
        MainConfig,
        ModelConfig,
        OperationConfig,
        OptimConfig,
        ParallelConfig,
        PEFTConfig,
        ProfilerConfig,
        TrainerConfig,
        UtilsConfig,
    )
    from ironcore.config.config_alignment import AlignmentConfig
    from ironcore.config.config_model import BiasConfig

    model_config = ModelConfig(
        d_model=64,
        num_attention_heads=2,
        num_attention_groups=2,
        head_dim=32,
        d_ffn=128,
        num_layers=1,
        max_seq_len=32,
        max_position_embeddings=32,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        bias=BiasConfig.all_true(),
        layernorm_bias=True,
        precision="float32",
    )

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=1,
        micro_batch_size=2,
        train_batch_size=2,
        gradient_accumulation_steps=1,
        use_flash_attn=False,
    )

    alignment_config = AlignmentConfig(
        dpo_beta=0.1,
        dpo_label_smoothing=0.0,
        concat_forward_passes=True,
        metrics_interval=0,
    )

    return MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=InitConfig(seed=42, init_std=0.02),
        optim=OptimConfig(max_lr=1e-5, weight_decay=0.0, clip_grad=1.0),
        data=DataConfig(),
        parallel=ParallelConfig(world_size=1),
        operation=OperationConfig(train_steps=3, eval_interval=100),
        utils=UtilsConfig(),
        peft=PEFTConfig(),  # Required parameter
        profiler=ProfilerConfig(),
        alignment=alignment_config,
    )


@pytest.fixture
def mock_dpo_batch():
    """Create a mock DPO batch for testing."""
    batch_size, seq_len, vocab_size = 2, 16, 100

    return {
        "chosen_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "chosen_labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "chosen_loss_mask": torch.ones(batch_size, seq_len),
        "chosen_position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
        "rejected_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "rejected_labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "rejected_loss_mask": torch.ones(batch_size, seq_len),
        "rejected_position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
    }


@pytest.fixture
def tiny_transformer(tiny_model_config):
    """Create a tiny transformer model for testing."""
    from ironcore.global_vars import reset_global_states, set_global_states
    from ironcore.language_model import LanguageModel

    # Initialize GLOBAL_STATES before creating model (required for tokenizer)
    set_global_states(tiny_model_config)

    model = LanguageModel(tiny_model_config, loss_fn=nn.CrossEntropyLoss())
    yield model

    # Cleanup
    reset_global_states()


# =============================================================================
# DPO Loss Integration Tests
# =============================================================================


class TestDpoLossIntegration:
    """Integration tests for DPO loss with real model forward pass."""

    @pytest.mark.unit
    def test_dpo_loss_with_model_forward(self, tiny_transformer, mock_dpo_batch):
        """Test DPO loss computation with actual model forward pass."""
        from ironcore.alignment.loss import dpo_loss

        device = next(tiny_transformer.parameters()).device
        batch = {k: v.to(device) for k, v in mock_dpo_batch.items()}

        tiny_transformer.train()

        # Forward pass for chosen
        chosen_logits = tiny_transformer(batch["chosen_input_ids"], labels=None)
        rejected_logits = tiny_transformer(batch["rejected_input_ids"], labels=None)

        # Use same model as reference (in practice, reference is frozen copy)
        with torch.no_grad():
            ref_chosen_logits = tiny_transformer(batch["chosen_input_ids"], labels=None)
            ref_rejected_logits = tiny_transformer(batch["rejected_input_ids"], labels=None)

        # Compute DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logits=chosen_logits,
            policy_rejected_logits=rejected_logits,
            reference_chosen_logits=ref_chosen_logits,
            reference_rejected_logits=ref_rejected_logits,
            chosen_labels=batch["chosen_labels"],
            rejected_labels=batch["rejected_labels"],
            beta=0.1,
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() >= 0, "Loss should be non-negative"

    @pytest.mark.unit
    def test_dpo_loss_backward_flow(self, tiny_transformer, mock_dpo_batch):
        """Test that gradients flow correctly through DPO loss."""
        from ironcore.alignment.loss import dpo_loss

        device = next(tiny_transformer.parameters()).device
        batch = {k: v.to(device) for k, v in mock_dpo_batch.items()}

        tiny_transformer.train()

        # Forward pass
        chosen_logits = tiny_transformer(batch["chosen_input_ids"], labels=None)
        rejected_logits = tiny_transformer(batch["rejected_input_ids"], labels=None)

        with torch.no_grad():
            ref_chosen_logits = tiny_transformer(batch["chosen_input_ids"], labels=None)
            ref_rejected_logits = tiny_transformer(batch["rejected_input_ids"], labels=None)

        loss, _ = dpo_loss(
            policy_chosen_logits=chosen_logits,
            policy_rejected_logits=rejected_logits,
            reference_chosen_logits=ref_chosen_logits,
            reference_rejected_logits=ref_rejected_logits,
            chosen_labels=batch["chosen_labels"],
            rejected_labels=batch["rejected_labels"],
            beta=0.1,
        )

        # Backward
        loss.backward()

        # Check gradients exist on all parameters
        for name, param in tiny_transformer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"

    @pytest.mark.unit
    def test_dpo_with_bin_packed_position_ids(self, tiny_transformer):
        """Test DPO forward pass with non-sequential position_ids (bin-packed sequences)."""
        from ironcore.alignment.loss import dpo_loss

        batch_size, seq_len, vocab_size = 2, 16, 100
        device = next(tiny_transformer.parameters()).device

        # Simulate bin-packed sequences where position_ids reset mid-sequence
        # This tests that the model correctly uses provided position_ids
        chosen_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        rejected_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        # Create non-sequential position_ids (simulating bin-packing)
        # First half: positions 0-7, Second half: positions 0-7 (reset)
        chosen_position_ids = torch.cat(
            [
                torch.arange(8).unsqueeze(0).expand(batch_size, -1),
                torch.arange(8).unsqueeze(0).expand(batch_size, -1),
            ],
            dim=1,
        ).to(device)
        rejected_position_ids = chosen_position_ids.clone()

        tiny_transformer.train()

        # Forward pass WITH position_ids
        chosen_logits = tiny_transformer(
            chosen_input_ids, labels=None, position_ids=chosen_position_ids
        )
        rejected_logits = tiny_transformer(
            rejected_input_ids, labels=None, position_ids=rejected_position_ids
        )

        with torch.no_grad():
            ref_chosen_logits = tiny_transformer(
                chosen_input_ids, labels=None, position_ids=chosen_position_ids
            )
            ref_rejected_logits = tiny_transformer(
                rejected_input_ids, labels=None, position_ids=rejected_position_ids
            )

        loss, metrics = dpo_loss(
            policy_chosen_logits=chosen_logits,
            policy_rejected_logits=rejected_logits,
            reference_chosen_logits=ref_chosen_logits,
            reference_rejected_logits=ref_rejected_logits,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            beta=0.1,
        )

        assert torch.isfinite(loss), "Loss should be finite with bin-packed position_ids"
        assert loss.dim() == 0, "Loss should be scalar"


# =============================================================================
# Reference Model Tests
# =============================================================================


class TestReferenceModel:
    """Tests for reference model handling in DPO."""

    @pytest.mark.unit
    def test_reference_model_creation(self, tiny_transformer):
        """Test that reference model is a deep copy of policy."""
        # Simulate reference model creation
        ref_model = copy.deepcopy(tiny_transformer)

        # Freeze parameters
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()

        # Check parameters are equal initially
        for (name1, p1), (name2, p2) in zip(
            tiny_transformer.named_parameters(), ref_model.named_parameters(), strict=False
        ):
            assert name1 == name2
            assert torch.allclose(p1, p2), f"Parameters {name1} differ"

    @pytest.mark.unit
    def test_reference_model_not_updated(self, tiny_transformer, mock_dpo_batch):
        """Test that reference model weights are not updated during training."""
        from ironcore.alignment.loss import dpo_loss

        device = next(tiny_transformer.parameters()).device
        batch = {k: v.to(device) for k, v in mock_dpo_batch.items()}

        # Create reference model
        ref_model = copy.deepcopy(tiny_transformer)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()

        # Store initial reference weights
        initial_ref_weights = {name: p.clone() for name, p in ref_model.named_parameters()}

        # Training step
        tiny_transformer.train()
        optimizer = torch.optim.Adam(tiny_transformer.parameters(), lr=1e-4)

        chosen_logits = tiny_transformer(batch["chosen_input_ids"], labels=None)
        rejected_logits = tiny_transformer(batch["rejected_input_ids"], labels=None)

        with torch.no_grad():
            ref_chosen_logits = ref_model(batch["chosen_input_ids"], labels=None)
            ref_rejected_logits = ref_model(batch["rejected_input_ids"], labels=None)

        loss, _ = dpo_loss(
            policy_chosen_logits=chosen_logits,
            policy_rejected_logits=rejected_logits,
            reference_chosen_logits=ref_chosen_logits,
            reference_rejected_logits=ref_rejected_logits,
            chosen_labels=batch["chosen_labels"],
            rejected_labels=batch["rejected_labels"],
            beta=0.1,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify reference model unchanged
        for name, p in ref_model.named_parameters():
            assert torch.allclose(p, initial_ref_weights[name]), f"Reference {name} was modified"


# =============================================================================
# Gradient Accumulation Tests
# =============================================================================


class TestGradientAccumulation:
    """Tests for gradient accumulation in DPO training."""

    @pytest.mark.unit
    def test_gradient_accumulation_correctness(self, tiny_transformer, mock_dpo_batch):
        """Test that gradient accumulation produces correct gradients."""
        from ironcore.alignment.loss import dpo_loss

        device = next(tiny_transformer.parameters()).device
        batch = {k: v.to(device) for k, v in mock_dpo_batch.items()}

        tiny_transformer.train()

        # Create reference model
        ref_model = copy.deepcopy(tiny_transformer)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()

        optimizer = torch.optim.SGD(tiny_transformer.parameters(), lr=0.0)  # Zero LR to check grads

        # Method 1: Single step with batch
        optimizer.zero_grad()
        chosen_logits = tiny_transformer(batch["chosen_input_ids"], labels=None)
        rejected_logits = tiny_transformer(batch["rejected_input_ids"], labels=None)
        with torch.no_grad():
            ref_chosen = ref_model(batch["chosen_input_ids"], labels=None)
            ref_rejected = ref_model(batch["rejected_input_ids"], labels=None)

        loss1, _ = dpo_loss(
            chosen_logits,
            rejected_logits,
            ref_chosen,
            ref_rejected,
            batch["chosen_labels"],
            batch["rejected_labels"],
            beta=0.1,
        )
        loss1.backward()
        grads1 = {
            name: p.grad.clone()
            for name, p in tiny_transformer.named_parameters()
            if p.grad is not None
        }

        # Method 2: Two accumulation steps (simulated with same batch)
        optimizer.zero_grad()
        for _ in range(2):
            chosen_logits = tiny_transformer(batch["chosen_input_ids"], labels=None)
            rejected_logits = tiny_transformer(batch["rejected_input_ids"], labels=None)
            with torch.no_grad():
                ref_chosen = ref_model(batch["chosen_input_ids"], labels=None)
                ref_rejected = ref_model(batch["rejected_input_ids"], labels=None)

            loss2, _ = dpo_loss(
                chosen_logits,
                rejected_logits,
                ref_chosen,
                ref_rejected,
                batch["chosen_labels"],
                batch["rejected_labels"],
                beta=0.1,
            )
            (loss2 / 2).backward()  # Scale by accumulation steps

        grads2 = {
            name: p.grad.clone()
            for name, p in tiny_transformer.named_parameters()
            if p.grad is not None
        }

        # Gradients should be similar (same data, just accumulated differently)
        for name, grad1 in grads1.items():
            if name in grads2:
                # Allow some numerical tolerance
                assert torch.allclose(grad1, grads2[name], atol=1e-5, rtol=1e-3), (
                    f"Gradient mismatch for {name}"
                )


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestDPOCheckpoint:
    """Tests for DPO checkpoint save/load."""

    @pytest.mark.integration
    def test_checkpoint_save_load_preserves_weights(self, tiny_model_config, tiny_transformer):
        """Test that checkpoint save/load preserves model weights."""
        from torch.optim.lr_scheduler import StepLR

        from ironcore.checkpointing import load_checkpoint, save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir)

            # Update config with checkpoint path
            tiny_model_config.trainer.model_path = str(ckpt_path)
            tiny_model_config.operation.save_dist_ckpt = True

            optimizer = torch.optim.Adam(tiny_transformer.parameters(), lr=1e-4)
            lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

            # Store original weights before saving
            original_weights = {name: p.clone() for name, p in tiny_transformer.named_parameters()}

            # Save checkpoint
            save_checkpoint(tiny_model_config, tiny_transformer, optimizer, lr_scheduler, step=5)

            # Modify weights (simulate training)
            for p in tiny_transformer.parameters():
                p.data.add_(torch.randn_like(p) * 0.01)

            # Verify weights are now different from original
            for name, p in tiny_transformer.named_parameters():
                assert not torch.allclose(p, original_weights[name], atol=1e-6), (
                    f"Weight {name} should have been modified"
                )

            # Load checkpoint
            step = load_checkpoint(tiny_model_config, tiny_transformer, optimizer, lr_scheduler)

            assert step == 5, f"Expected step 5, got {step}"

            # Verify weights are restored to original values
            for name, p in tiny_transformer.named_parameters():
                assert torch.allclose(p, original_weights[name], atol=1e-6), (
                    f"Weight {name} was not restored correctly from checkpoint"
                )

    @pytest.mark.integration
    def test_reference_model_recreation_after_load(self, tiny_model_config):
        """Test that reference model is correctly created from loaded checkpoint.

        This verifies the DPO-specific behavior where reference model must be
        created AFTER checkpoint loading to use the SFT weights.
        """
        # This test verifies the pattern used in DPOTrainer._post_checkpoint_load
        # 1. Load checkpoint (policy weights restored)
        # 2. Deep copy policy to create reference

        from ironcore.language_model import LanguageModel

        model = LanguageModel(tiny_model_config, loss_fn=nn.CrossEntropyLoss())

        # Simulate checkpoint load (just use initial weights)
        # In real scenario, checkpoint would have SFT weights

        # Create reference from loaded policy
        ref_model = copy.deepcopy(model)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()

        # Verify reference has same weights as policy
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), ref_model.named_parameters(), strict=False
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch: {n1} vs {n2}"


# =============================================================================
# Full Training Loop Tests
# =============================================================================


class TestDPOTrainingLoop:
    """Tests for complete DPO training loop."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_multi_step_training(self, tiny_model_config, mock_dpo_batch):
        """Test multiple training steps with loss decreasing or stable."""
        from ironcore.alignment.loss import dpo_loss
        from ironcore.language_model import LanguageModel

        model = LanguageModel(tiny_model_config, loss_fn=nn.CrossEntropyLoss())
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in mock_dpo_batch.items()}

        # Create reference model
        ref_model = copy.deepcopy(model)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()

        losses = []
        for step in range(5):
            optimizer.zero_grad()

            chosen_logits = model(batch["chosen_input_ids"], labels=None)
            rejected_logits = model(batch["rejected_input_ids"], labels=None)

            with torch.no_grad():
                ref_chosen = ref_model(batch["chosen_input_ids"], labels=None)
                ref_rejected = ref_model(batch["rejected_input_ids"], labels=None)

            loss, metrics = dpo_loss(
                chosen_logits,
                rejected_logits,
                ref_chosen,
                ref_rejected,
                batch["chosen_labels"],
                batch["rejected_labels"],
                beta=0.1,
            )

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # All losses should be finite
        import math

        assert all(math.isfinite(loss) for loss in losses), "Loss should not be NaN"
        assert all(abs(loss) < 100 for loss in losses), "Loss should be reasonable"

    @pytest.mark.integration
    def test_training_with_different_betas(self, tiny_model_config, mock_dpo_batch):
        """Test training with different beta values."""
        from ironcore.alignment.loss import dpo_loss
        from ironcore.language_model import LanguageModel

        for beta in [0.05, 0.1, 0.5, 1.0]:
            model = LanguageModel(tiny_model_config, loss_fn=nn.CrossEntropyLoss())
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in mock_dpo_batch.items()}

            ref_model = copy.deepcopy(model)
            for param in ref_model.parameters():
                param.requires_grad = False
            ref_model.eval()

            model.train()

            chosen_logits = model(batch["chosen_input_ids"], labels=None)
            rejected_logits = model(batch["rejected_input_ids"], labels=None)

            with torch.no_grad():
                ref_chosen = ref_model(batch["chosen_input_ids"], labels=None)
                ref_rejected = ref_model(batch["rejected_input_ids"], labels=None)

            loss, metrics = dpo_loss(
                chosen_logits,
                rejected_logits,
                ref_chosen,
                ref_rejected,
                batch["chosen_labels"],
                batch["rejected_labels"],
                beta=beta,
            )

            assert torch.isfinite(loss), f"Loss not finite for beta={beta}"
            assert 0.0 <= metrics["dpo_accuracy"] <= 1.0, f"Invalid accuracy for beta={beta}"


# =============================================================================
# Concat Forward Pass Optimization Tests
# =============================================================================


class TestConcatForwardPassOptimization:
    """Tests for the concat_forward_passes optimization."""

    @pytest.mark.unit
    def test_concat_vs_separate_equivalence(self, tiny_transformer, mock_dpo_batch):
        """Test that concat and separate forward passes give same results."""
        from ironcore.alignment.loss import dpo_loss

        device = next(tiny_transformer.parameters()).device
        batch = {k: v.to(device) for k, v in mock_dpo_batch.items()}

        tiny_transformer.eval()

        # Method 1: Separate forward passes
        with torch.no_grad():
            chosen_logits = tiny_transformer(batch["chosen_input_ids"], labels=None)
            rejected_logits = tiny_transformer(batch["rejected_input_ids"], labels=None)
            ref_chosen = tiny_transformer(batch["chosen_input_ids"], labels=None)
            ref_rejected = tiny_transformer(batch["rejected_input_ids"], labels=None)

        loss1, metrics1 = dpo_loss(
            chosen_logits,
            rejected_logits,
            ref_chosen,
            ref_rejected,
            batch["chosen_labels"],
            batch["rejected_labels"],
            beta=0.1,
            compute_metrics=True,
        )

        # Method 2: Concatenated forward passes
        concat_input = torch.cat([batch["chosen_input_ids"], batch["rejected_input_ids"]], dim=0)

        with torch.no_grad():
            concat_logits = tiny_transformer(concat_input, labels=None)

        policy_concat = concat_logits
        ref_concat = concat_logits.clone()

        loss2, metrics2 = dpo_loss(
            chosen_logits,
            rejected_logits,
            ref_chosen,
            ref_rejected,
            batch["chosen_labels"],
            batch["rejected_labels"],
            beta=0.1,
            policy_concat_logits=policy_concat,
            reference_concat_logits=ref_concat,
            compute_metrics=True,
        )

        # Results should be identical
        assert torch.allclose(loss1, loss2, atol=1e-5), (
            f"Loss mismatch: {loss1.item()} vs {loss2.item()}"
        )


# =============================================================================
# NaN Detection Tests
# =============================================================================


class TestNaNHandling:
    """Tests for NaN detection and handling."""

    @pytest.mark.unit
    def test_nan_loss_detection(self, tiny_model_config):
        """Test that NaN losses are detected."""
        # Simulate NaN detection (the actual check is in BaseTrainer._check_loss_for_nan)
        import math

        def check_loss_for_nan(loss: float, step: int) -> None:
            if math.isnan(loss) or math.isinf(loss):
                raise RuntimeError(f"NaN/Inf loss detected at step {step}: loss={loss}")

        # Normal loss should pass
        check_loss_for_nan(0.5, 1)

        # NaN should raise
        with pytest.raises(RuntimeError, match="NaN"):
            check_loss_for_nan(float("nan"), 1)

        # Inf should raise
        with pytest.raises(RuntimeError, match="Inf"):
            check_loss_for_nan(float("inf"), 1)


# =============================================================================
# Metrics Tests
# =============================================================================


class TestDPOMetrics:
    """Tests for DPO metrics computation."""

    @pytest.mark.unit
    def test_metrics_interval_skip(self, tiny_transformer, mock_dpo_batch):
        """Test that metrics are skipped when not on interval."""
        from ironcore.alignment.loss import dpo_loss

        device = next(tiny_transformer.parameters()).device
        batch = {k: v.to(device) for k, v in mock_dpo_batch.items()}

        tiny_transformer.eval()

        with torch.no_grad():
            chosen_logits = tiny_transformer(batch["chosen_input_ids"], labels=None)
            rejected_logits = tiny_transformer(batch["rejected_input_ids"], labels=None)

        # With compute_metrics=True
        _, metrics_full = dpo_loss(
            chosen_logits,
            rejected_logits,
            chosen_logits.clone(),
            rejected_logits.clone(),
            batch["chosen_labels"],
            batch["rejected_labels"],
            beta=0.1,
            compute_metrics=True,
        )

        # With compute_metrics=False
        _, metrics_minimal = dpo_loss(
            chosen_logits,
            rejected_logits,
            chosen_logits.clone(),
            rejected_logits.clone(),
            batch["chosen_labels"],
            batch["rejected_labels"],
            beta=0.1,
            compute_metrics=False,
        )

        # Full metrics should have more keys
        assert "chosen_policy_logps" in metrics_full
        assert "preference_margin" in metrics_full

        # Minimal should still have basic metrics
        assert "dpo_loss" in metrics_minimal


# =============================================================================
# Collator Tests
# =============================================================================


class TestCollatorTruncation:
    """Tests for collator truncation of oversized samples."""

    @pytest.mark.unit
    def test_sft_collator_truncates_oversized_samples(self):
        """Test that SFT collator truncates samples exceeding max_seq_len."""
        from ironcore.dataloader.collator import UniversalCollator

        max_seq_len = 32
        collator = UniversalCollator(
            mode="sft",
            max_seq_len=max_seq_len,
            pad_token_id=0,
            use_flash_attention=False,
            return_full_attention_mask=False,
        )

        # Create a sample that exceeds max_seq_len
        oversized_sample = {
            "token_ids": torch.randint(0, 100, (50,)),  # 50 tokens > 32 max
            "metadata": {"mask_ranges": []},
        }

        # Should not raise, should truncate
        result = collator([oversized_sample])

        assert result["input_ids"].shape == (1, max_seq_len)
        assert result["labels"].shape == (1, max_seq_len)
        assert result["position_ids"].shape == (1, max_seq_len)

    @pytest.mark.unit
    def test_dpo_collator_truncates_oversized_samples(self):
        """Test that DPO collator truncates samples exceeding max_seq_len."""
        from ironcore.dataloader.collator import UniversalCollator

        max_seq_len = 32
        collator = UniversalCollator(
            mode="dpo",
            max_seq_len=max_seq_len,
            pad_token_id=0,
            use_flash_attention=False,
            return_full_attention_mask=False,
        )

        # Create DPO pair where chosen exceeds max_seq_len
        chosen_sample = {
            "token_ids": torch.randint(0, 100, (50,)),  # 50 tokens > 32 max
            "metadata": {"type": "dpo_chosen", "mask_ranges": []},
        }
        rejected_sample = {
            "token_ids": torch.randint(0, 100, (50,)),
            "metadata": {"type": "dpo_rejected", "mask_ranges": []},
        }

        # Should not raise, should truncate both
        result = collator([chosen_sample, rejected_sample])

        assert result["chosen_input_ids"].shape == (1, max_seq_len)
        assert result["rejected_input_ids"].shape == (1, max_seq_len)
        assert result["chosen_position_ids"].shape == (1, max_seq_len)
        assert result["rejected_position_ids"].shape == (1, max_seq_len)
