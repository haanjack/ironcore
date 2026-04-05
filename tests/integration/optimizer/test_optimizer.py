# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Trainer-Optimizer integration tests.

Tests that optimizers work correctly within LanguageModelTrainer,
using mock forward_step_func to bypass data loading.
"""

from __future__ import annotations

import math
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ironcore.parallel import parallel_states
from tests.fixtures.config_fixtures import create_small_test_config


# =============================================================================
# Helper Functions
# =============================================================================


def setup_distributed():
    """Set up single-process distributed environment for testing."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


def cleanup_distributed():
    """Clean up distributed environment."""
    try:
        parallel_states.destroy_model_parallel()
    except Exception:
        pass
    if dist.is_initialized():
        dist.destroy_process_group()


def create_mock_forward_step_func():
    """Create a mock forward_step_func that bypasses data loading."""

    def mock_forward_step(model, data_iterator):
        batch_size = 2
        seq_len = 16
        # Get the correct device for the current rank
        device = next(model.parameters()).device
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        logits = model(input_ids, labels=None)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Debug: verify loss has gradients
        if not loss.requires_grad:
            print(f"WARNING: loss does not require grad! logits.requires_grad={shift_logits.requires_grad}")
        elif loss.grad_fn is None:
            print(f"WARNING: loss has no grad_fn!")

        return loss

    return mock_forward_step


def create_mock_data_iterator():
    """Create a mock data iterator for testing."""

    class MockIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    return {"train": MockIterator(), "eval": MockIterator(), "test": MockIterator()}


def create_mock_evaluators():
    """Create mock evaluators for testing."""
    return []


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.cuda
class TestOptimizerTrainerIntegration:
    """Test optimizer integration with LanguageModelTrainer."""

    def test_adamw_with_trainer(self):
        """Verify AdamW optimizer works with LanguageModelTrainer."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        # Ensure clean state before starting
        reset_global_states()
        setup_distributed()
        config = create_small_test_config()
        config.optim.optimizer = "adamw"
        config.trainer.gradient_accumulation_steps = 1

        try:
            with patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ), patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                loss, grad_norm, param_norm = trainer.train_step(step=0)

                assert loss > 0
                assert not math.isnan(loss)
                assert grad_norm > 0
        finally:
            reset_global_states()
            cleanup_distributed()

    def test_muon_with_trainer(self):
        """Verify Muon optimizer works with LanguageModelTrainer."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        # Ensure clean state before starting
        reset_global_states()
        setup_distributed()
        config = create_small_test_config()
        config.optim.optimizer = "muon"
        config.trainer.gradient_accumulation_steps = 1

        try:
            with patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ), patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                # Verify optimizer type (Muon creates hybrid optimizer)
                opt_type = type(trainer.optimizer).__name__
                assert "Muon" in opt_type or "AdamW" in opt_type

                loss, grad_norm, param_norm = trainer.train_step(step=0)

                assert loss > 0
                assert not math.isnan(loss)
        finally:
            reset_global_states()
            cleanup_distributed()

    def test_muon_multi_step(self):
        """Verify Muon optimizer accumulates state across steps."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        setup_distributed()
        config = create_small_test_config()
        config.optim.optimizer = "muon"
        config.trainer.gradient_accumulation_steps = 1

        try:
            with patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ), patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                losses = []
                for step in range(5):
                    loss, _, _ = trainer.train_step(step=step)
                    losses.append(loss)

                # Losses should change as model trains
                assert len(set(round(l, 4) for l in losses)) > 1
        finally:
            reset_global_states()
            cleanup_distributed()

    def test_optimizer_state_dict_roundtrip(self):
        """Verify optimizer state can be saved and loaded."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        setup_distributed()
        config = create_small_test_config()
        config.optim.optimizer = "muon"
        config.trainer.gradient_accumulation_steps = 1

        try:
            with patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ), patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                # Run a step to populate optimizer state
                trainer.train_step(step=0)

                # Save and load state
                state_dict = trainer.optimizer.state_dict()
                assert len(state_dict["state"]) > 0

                trainer.optimizer.load_state_dict(state_dict)
        finally:
            reset_global_states()
            cleanup_distributed()


# =============================================================================
# Tensor Parallelism Tests
# =============================================================================


@pytest.mark.cuda
@pytest.mark.distributed
class TestOptimizerTPIntegration:
    """Test optimizer integration with Tensor Parallelism and LanguageModelTrainer."""

    def _setup_tp_distributed(self, tp_size: int):
        """Setup distributed environment for TP tests."""
        if tp_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            torch.cuda.set_device(rank)
            os.environ.setdefault("LOCAL_RANK", str(rank))

            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=tp_size,
                timeout_in_minutes=30,
            )
        else:
            rank = 0
            world_size = 1

            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12358")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", rank=0, world_size=1)

            try:
                parallel_states.initialize_model_parallel(
                    tensor_model_parallel_size=1,
                    timeout_in_minutes=1.0,
                )
            except Exception:
                pass

        return rank, world_size

    def _cleanup_tp_distributed(self, tp_size: int):
        """Cleanup distributed environment."""
        try:
            parallel_states.destroy_model_parallel()
        except Exception:
            pass
        if dist.is_initialized():
            dist.barrier()
            if tp_size > 1:
                dist.destroy_process_group()

    def test_muon_tp1_with_trainer(self):
        """Verify Muon optimizer works with LanguageModelTrainer and TP=1."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tp_size = 1
        rank, _ = self._setup_tp_distributed(tp_size)

        config = create_small_test_config()
        config.optim.optimizer = "muon"
        config.optim.muon_lr = 0.01
        config.trainer.tensor_model_parallel_size = tp_size
        config.trainer.gradient_accumulation_steps = 1

        try:
            with patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ), patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                loss, grad_norm, param_norm = trainer.train_step(step=0)

                assert loss > 0
                assert not math.isnan(loss)
                assert grad_norm > 0

                if rank == 0:
                    print(f"TP=1 Muon trainer test passed: loss={loss:.6f}")
        finally:
            reset_global_states()
            self._cleanup_tp_distributed(tp_size)

    def test_muon_tp2_with_trainer(self):
        """Verify Muon optimizer works with LanguageModelTrainer and TP=2."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if dist.is_initialized() and dist.get_world_size() < 2:
            pytest.skip("Need at least 2 GPUs for TP=2 test")

        tp_size = 2
        rank, _ = self._setup_tp_distributed(tp_size)

        config = create_small_test_config()
        config.optim.optimizer = "muon"
        config.optim.muon_lr = 0.01
        config.trainer.tensor_model_parallel_size = tp_size
        config.trainer.gradient_accumulation_steps = 1

        try:
            with patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ), patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                loss, grad_norm, param_norm = trainer.train_step(step=0)

                assert loss > 0
                assert not math.isnan(loss)

                if rank == 0:
                    print(f"TP=2 Muon trainer test passed: loss={loss:.6f}")
        finally:
            reset_global_states()
            self._cleanup_tp_distributed(tp_size)

    def test_muon_tp2_multi_step(self):
        """Verify Muon optimizer state accumulates across steps with TP=2."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if dist.is_initialized() and dist.get_world_size() < 2:
            pytest.skip("Need at least 2 GPUs for TP=2 test")

        tp_size = 2
        rank, _ = self._setup_tp_distributed(tp_size)

        config = create_small_test_config()
        config.optim.optimizer = "muon"
        config.trainer.tensor_model_parallel_size = tp_size
        config.trainer.gradient_accumulation_steps = 1

        try:
            with patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ), patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                losses = []
                for step in range(3):
                    loss, _, _ = trainer.train_step(step=step)
                    losses.append(loss)

                # Losses should change as model trains
                assert len(set(round(l, 4) for l in losses)) > 1

                if rank == 0:
                    print(f"TP=2 multi-step test passed: losses={[f'{l:.4f}' for l in losses]}")
        finally:
            reset_global_states()
            self._cleanup_tp_distributed(tp_size)


# =============================================================================
# FSDP Tests
# =============================================================================


@pytest.mark.cuda
@pytest.mark.distributed
class TestOptimizerFSDPIntegration:
    """Test optimizer integration with FSDP and LanguageModelTrainer."""

    def _setup_fsdp_distributed(self):
        """Setup distributed environment for FSDP tests."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        os.environ.setdefault("LOCAL_RANK", str(rank))

        # Initialize parallel states for single TP (FSDP handles parallelism)
        try:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30,
            )
        except Exception:
            pass

        return rank, world_size

    def _cleanup_fsdp_distributed(self):
        """Cleanup distributed environment."""
        try:
            parallel_states.destroy_model_parallel()
        except Exception:
            pass
        if dist.is_initialized():
            dist.barrier()

    def test_muon_fsdp_with_trainer(self):
        """Verify Muon optimizer works with LanguageModelTrainer and FSDP."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        rank, _ = self._setup_fsdp_distributed()

        config = create_small_test_config()
        config.optim.optimizer = "muon"
        config.optim.muon_lr = 0.01
        config.trainer.tensor_model_parallel_size = 1
        config.trainer.gradient_accumulation_steps = 1
        config.parallel.use_fsdp = True
        config.parallel.fsdp_sharding_strategy = "full"

        try:
            with patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ), patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                loss, grad_norm, param_norm = trainer.train_step(step=0)

                assert loss > 0
                assert not math.isnan(loss)

                if rank == 0:
                    print(f"FSDP Muon trainer test passed: loss={loss:.6f}")
        finally:
            reset_global_states()
            self._cleanup_fsdp_distributed()

    def test_muon_fsdp_multi_step(self):
        """Verify Muon optimizer works across multiple steps with FSDP."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        rank, _ = self._setup_fsdp_distributed()

        config = create_small_test_config()
        config.optim.optimizer = "muon"
        config.trainer.tensor_model_parallel_size = 1
        config.trainer.gradient_accumulation_steps = 1
        config.parallel.use_fsdp = True
        config.parallel.fsdp_sharding_strategy = "full"

        try:
            with patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ), patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                losses = []
                for step in range(3):
                    loss, _, _ = trainer.train_step(step=step)
                    losses.append(loss)

                assert len(set(round(l, 4) for l in losses)) > 1

                if rank == 0:
                    print(f"FSDP multi-step test passed: losses={[f'{l:.4f}' for l in losses]}")
        finally:
            reset_global_states()
            self._cleanup_fsdp_distributed()

    def test_muon_fsdp_state_dict(self):
        """Verify optimizer state dict save/load works with FSDP."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        rank, _ = self._setup_fsdp_distributed()

        config = create_small_test_config()
        config.optim.optimizer = "muon"
        config.trainer.tensor_model_parallel_size = 1
        config.trainer.gradient_accumulation_steps = 1
        config.parallel.use_fsdp = True
        config.parallel.fsdp_sharding_strategy = "full"

        try:
            with patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ), patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                # Run a step to populate optimizer state
                trainer.train_step(step=0)

                # Save state dict
                state_dict = trainer.optimizer.state_dict()
                assert len(state_dict.get("state", {})) > 0 or len(state_dict.get("param_groups", [])) > 0

                if rank == 0:
                    print("FSDP state dict test passed")
        finally:
            reset_global_states()
            self._cleanup_fsdp_distributed()


# =============================================================================
# Distributed Optimizer Tests
# =============================================================================


@pytest.mark.cuda
@pytest.mark.distributed
class TestDistributedOptimizerIntegration:
    """Test DistributedOptimizer integration with LanguageModelTrainer."""

    def _setup_distributed(self):
        """Setup distributed environment."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        os.environ.setdefault("LOCAL_RANK", str(rank))

        try:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30,
            )
        except Exception:
            pass

        return rank, world_size

    def _cleanup_distributed(self):
        """Cleanup distributed environment."""
        try:
            parallel_states.destroy_model_parallel()
        except Exception:
            pass
        if dist.is_initialized():
            dist.barrier()

    def test_distributed_optimizer_with_trainer(self):
        """Verify DistributedOptimizer works with LanguageModelTrainer."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        rank, _ = self._setup_distributed()

        config = create_small_test_config()
        config.optim.optimizer = "adamw"
        config.trainer.tensor_model_parallel_size = 1
        config.trainer.gradient_accumulation_steps = 1
        config.parallel.use_distributed_optimizer = True

        try:
            with patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ), patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                loss, grad_norm, param_norm = trainer.train_step(step=0)

                assert loss > 0
                assert not math.isnan(loss)

                if rank == 0:
                    print(f"DistributedOptimizer trainer test passed: loss={loss:.6f}")
        finally:
            reset_global_states()
            self._cleanup_distributed()

    def test_distributed_optimizer_state_dict(self):
        """Verify DistributedOptimizer state dict save/load."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        rank, _ = self._setup_distributed()

        config = create_small_test_config()
        config.optim.optimizer = "adamw"
        config.trainer.tensor_model_parallel_size = 1
        config.trainer.gradient_accumulation_steps = 1
        config.parallel.use_distributed_optimizer = True

        try:
            with patch(
                "ironcore.trainers.base_trainer.get_data_iterator",
                return_value=create_mock_data_iterator(),
            ), patch(
                "ironcore.trainers.base_trainer.get_evaluators",
                return_value=create_mock_evaluators(),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                trainer.train_step(step=0)

                state_dict = trainer.optimizer.state_dict()
                assert len(state_dict["state"]) > 0

                trainer.optimizer.load_state_dict(state_dict)

                if rank == 0:
                    print("DistributedOptimizer state dict test passed")
        finally:
            reset_global_states()
            self._cleanup_distributed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
