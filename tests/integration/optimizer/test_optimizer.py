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
from tests.fixtures.config_fixtures import create_small_test_config

from ironcore.parallel import parallel_states

# =============================================================================
# Helper Functions
# =============================================================================


def _get_free_port() -> str:
    """Find a free TCP port to avoid conflicts between test classes."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


def setup_distributed():
    """Set up single-process distributed environment for testing."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = _get_free_port()
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
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


def create_mock_forward_step_func(deterministic: bool = False):
    """Create a mock forward_step_func that bypasses data loading.

    Args:
        deterministic: If True, seeds RNG before each call so all ranks
            in a TP group produce identical inputs.
    """

    def mock_forward_step(model, data_iterator):
        batch_size = 2
        seq_len = 16
        device = next(model.parameters()).device

        if deterministic:
            torch.manual_seed(42)

        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        logits, _ = model(input_ids, labels=None)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
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


def assert_loss_valid(loss, grad_norm=None):
    """Assert loss is a positive finite scalar."""
    assert isinstance(loss, float), f"loss should be float, got {type(loss)}"
    assert loss > 0, f"loss should be positive, got {loss}"
    assert math.isfinite(loss), f"loss should be finite, got {loss}"
    if grad_norm is not None:
        assert grad_norm > 0, f"grad_norm should be positive, got {grad_norm}"


def assert_tp_loss_consistent(loss: float, tp_group=None):
    """Verify all TP ranks produce the same loss value."""
    if not dist.is_initialized():
        return
    ws = dist.get_world_size()
    if ws < 2:
        return
    tensor = torch.tensor([loss], device="cuda")
    gathered = [torch.zeros(1, device="cuda") for _ in range(ws)]
    dist.all_gather(gathered, tensor, group=tp_group)
    values = [t.item() for t in gathered]
    for i, v in enumerate(values):
        assert math.isclose(v, values[0], rel_tol=1e-4), (
            f"TP rank {i} loss {v} != rank 0 loss {values[0]}"
        )


def assert_params_changed(model, snapshot: dict, label: str = ""):
    """Assert that at least one parameter has changed since snapshot."""
    changed = False
    for name, p in model.named_parameters():
        if not p.requires_grad or name not in snapshot:
            continue
        if not torch.equal(p.data, snapshot[name]):
            changed = True
            break
    assert changed, f"Parameters did not change after training step {label}"


def snapshot_params(model) -> dict:
    """Snapshot model parameters for later comparison."""
    return {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}


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

        reset_global_states()
        setup_distributed()
        config = create_small_test_config()
        config.optim.optimizer = "adamw"
        config.trainer.gradient_accumulation_steps = 1

        try:
            with (
                patch(
                    "ironcore.trainers.base_trainer.get_data_iterator",
                    return_value=create_mock_data_iterator(),
                ),
                patch(
                    "ironcore.trainers.base_trainer.get_evaluators",
                    return_value=create_mock_evaluators(),
                ),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                before = snapshot_params(trainer.model)
                loss, grad_norm, param_norm = trainer.train_step(step=0)

                assert_loss_valid(loss, grad_norm)
                assert_params_changed(trainer.model, before, "(AdamW single step)")
        finally:
            reset_global_states()
            cleanup_distributed()

    def test_muon_with_trainer(self):
        """Verify Muon optimizer works with LanguageModelTrainer."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        reset_global_states()
        setup_distributed()
        config = create_small_test_config()
        config.optim.optimizer = "muon"
        config.trainer.gradient_accumulation_steps = 1

        try:
            with (
                patch(
                    "ironcore.trainers.base_trainer.get_data_iterator",
                    return_value=create_mock_data_iterator(),
                ),
                patch(
                    "ironcore.trainers.base_trainer.get_evaluators",
                    return_value=create_mock_evaluators(),
                ),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                opt_type = type(trainer.optimizer).__name__
                assert "Muon" in opt_type or "AdamW" in opt_type

                before = snapshot_params(trainer.model)
                loss, grad_norm, param_norm = trainer.train_step(step=0)

                assert_loss_valid(loss, grad_norm)
                assert_params_changed(trainer.model, before, "(Muon single step)")
        finally:
            reset_global_states()
            cleanup_distributed()

    def test_muon_multi_step(self):
        """Verify Muon optimizer accumulates state across steps."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        reset_global_states()
        setup_distributed()
        config = create_small_test_config()
        config.optim.optimizer = "muon"
        config.trainer.gradient_accumulation_steps = 1

        try:
            with (
                patch(
                    "ironcore.trainers.base_trainer.get_data_iterator",
                    return_value=create_mock_data_iterator(),
                ),
                patch(
                    "ironcore.trainers.base_trainer.get_evaluators",
                    return_value=create_mock_evaluators(),
                ),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                before = snapshot_params(trainer.model)
                losses = []
                for step in range(5):
                    loss, grad_norm, _ = trainer.train_step(step=step)
                    assert_loss_valid(loss, grad_norm)
                    losses.append(loss)

                assert_params_changed(trainer.model, before, "(Muon 5 steps)")

                # Optimizer should have accumulated state
                state = trainer.optimizer.state_dict()["state"]
                assert len(state) > 0, "Optimizer state should be non-empty after 5 steps"
        finally:
            reset_global_states()
            cleanup_distributed()

    def test_optimizer_state_dict_roundtrip(self):
        """Verify optimizer state can be saved and loaded."""
        from ironcore.global_vars import reset_global_states
        from ironcore.trainers import LanguageModelTrainer

        reset_global_states()
        setup_distributed()
        config = create_small_test_config()
        config.optim.optimizer = "muon"
        config.trainer.gradient_accumulation_steps = 1

        try:
            with (
                patch(
                    "ironcore.trainers.base_trainer.get_data_iterator",
                    return_value=create_mock_data_iterator(),
                ),
                patch(
                    "ironcore.trainers.base_trainer.get_evaluators",
                    return_value=create_mock_evaluators(),
                ),
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

                # Verify state contains actual tensor values
                for param_state in state_dict["state"].values():
                    assert any(isinstance(v, torch.Tensor) for v in param_state.values()), (
                        "Optimizer state should contain tensors"
                    )

                trainer.optimizer.load_state_dict(state_dict)
        finally:
            reset_global_states()
            cleanup_distributed()


# =============================================================================
# Tensor Parallelism Tests
# =============================================================================


class TestOptimizerTPIntegration:
    """Test optimizer integration with Tensor Parallelism and LanguageModelTrainer."""

    def _setup_tp_distributed(self, tp_size: int):
        """Setup distributed environment for TP tests."""
        if tp_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            # torchrun sets CUDA_VISIBLE_DEVICES per rank, so only set device
            # when local_rank doesn't match current device
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            if torch.cuda.current_device() != local_rank:
                torch.cuda.set_device(local_rank)
            os.environ.setdefault("LOCAL_RANK", str(rank))

            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=tp_size,
                timeout_in_minutes=30,
            )
        else:
            rank = 0
            world_size = 1

            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = _get_free_port()
            os.environ["LOCAL_RANK"] = "0"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
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
        if not dist.is_initialized():
            return

        if tp_size > 1:
            # Under torchrun the process group is session scoped. Tearing it down
            # per test and rebuilding it in the next setup deadlocks the run: the
            # rank that gets there first re-enters the rendezvous while the other
            # is still in the previous test's collective, and from then on they
            # are one test apart. Which rank errors and which blocks forever
            # varies run to run. Only the parallel_states above are per test.
            dist.barrier()
            return

        # Single process: safe to tear down, and necessary — otherwise dist stays
        # initialized (get_tensor_model_parallel_rank() then takes the
        # dist.get_rank() branch instead of the world_size=1 fast path) while
        # parallel_states is reset to None above, leaking a broken "initialized
        # but groupless" state into later tests in the same pytest process
        # (see tests/unit/parallel/test_tp_init_seed.py).
        dist.destroy_process_group()

    @pytest.mark.cuda
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
            with (
                patch(
                    "ironcore.trainers.base_trainer.get_data_iterator",
                    return_value=create_mock_data_iterator(),
                ),
                patch(
                    "ironcore.trainers.base_trainer.get_evaluators",
                    return_value=create_mock_evaluators(),
                ),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                before = snapshot_params(trainer.model)
                loss, grad_norm, param_norm = trainer.train_step(step=0)

                assert_loss_valid(loss, grad_norm)
                assert_params_changed(trainer.model, before, "(TP=1 Muon)")

                if rank == 0:
                    print(f"TP=1 Muon trainer test passed: loss={loss:.6f}")
        finally:
            reset_global_states()
            self._cleanup_tp_distributed(tp_size)

    @pytest.mark.mp
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
            with (
                patch(
                    "ironcore.trainers.base_trainer.get_data_iterator",
                    return_value=create_mock_data_iterator(),
                ),
                patch(
                    "ironcore.trainers.base_trainer.get_evaluators",
                    return_value=create_mock_evaluators(),
                ),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(deterministic=True),
                    F.cross_entropy,
                )
                trainer._initialize()

                before = snapshot_params(trainer.model)
                loss, grad_norm, param_norm = trainer.train_step(step=0)

                assert_loss_valid(loss, grad_norm)
                assert_tp_loss_consistent(loss)
                assert_params_changed(trainer.model, before, "(TP=2 Muon)")

                if rank == 0:
                    print(f"TP=2 Muon trainer test passed: loss={loss:.6f}")
        finally:
            reset_global_states()
            self._cleanup_tp_distributed(tp_size)

    @pytest.mark.mp
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
            with (
                patch(
                    "ironcore.trainers.base_trainer.get_data_iterator",
                    return_value=create_mock_data_iterator(),
                ),
                patch(
                    "ironcore.trainers.base_trainer.get_evaluators",
                    return_value=create_mock_evaluators(),
                ),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(deterministic=True),
                    F.cross_entropy,
                )
                trainer._initialize()

                before = snapshot_params(trainer.model)
                losses = []
                for step in range(3):
                    loss, grad_norm, _ = trainer.train_step(step=step)
                    assert_loss_valid(loss, grad_norm)
                    losses.append(loss)

                assert_params_changed(trainer.model, before, "(TP=2 3 steps)")
                assert_tp_loss_consistent(losses[-1])

                state = trainer.optimizer.state_dict()["state"]
                assert len(state) > 0, "Optimizer state should be non-empty after 3 steps"

                if rank == 0:
                    print(f"TP=2 multi-step test passed: losses={[f'{v:.4f}' for v in losses]}")
        finally:
            reset_global_states()
            self._cleanup_tp_distributed(tp_size)


# =============================================================================
# FSDP Tests
# =============================================================================


@pytest.mark.mp
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

        return rank, world_size

    def _cleanup_fsdp_distributed(self):
        """Cleanup distributed environment."""
        try:
            parallel_states.destroy_model_parallel()
        except Exception:
            pass

    def test_muon_fsdp_with_trainer(self):
        """Verify Muon optimizer works with LanguageModelTrainer and FSDP."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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
        config.parallel.fsdp_use_orig_params = True

        try:
            with (
                patch(
                    "ironcore.trainers.base_trainer.get_data_iterator",
                    return_value=create_mock_data_iterator(),
                ),
                patch(
                    "ironcore.trainers.base_trainer.get_evaluators",
                    return_value=create_mock_evaluators(),
                ),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                assert isinstance(trainer.model, FSDP), (
                    "Model should be wrapped in FSDP when use_fsdp=True"
                )

                before = snapshot_params(trainer.model)
                loss, grad_norm, param_norm = trainer.train_step(step=0)

                assert_loss_valid(loss, grad_norm)
                assert_params_changed(trainer.model, before, "(FSDP Muon)")

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
        config.parallel.fsdp_use_orig_params = True

        try:
            with (
                patch(
                    "ironcore.trainers.base_trainer.get_data_iterator",
                    return_value=create_mock_data_iterator(),
                ),
                patch(
                    "ironcore.trainers.base_trainer.get_evaluators",
                    return_value=create_mock_evaluators(),
                ),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                before = snapshot_params(trainer.model)
                losses = []
                for step in range(3):
                    loss, grad_norm, _ = trainer.train_step(step=step)
                    assert_loss_valid(loss, grad_norm)
                    losses.append(loss)

                assert_params_changed(trainer.model, before, "(FSDP 3 steps)")

                state = trainer.optimizer.state_dict()["state"]
                assert len(state) > 0, "Optimizer state should be non-empty after 3 steps"

                if rank == 0:
                    print(f"FSDP multi-step test passed: losses={[f'{v:.4f}' for v in losses]}")
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
        config.parallel.fsdp_use_orig_params = True

        try:
            with (
                patch(
                    "ironcore.trainers.base_trainer.get_data_iterator",
                    return_value=create_mock_data_iterator(),
                ),
                patch(
                    "ironcore.trainers.base_trainer.get_evaluators",
                    return_value=create_mock_evaluators(),
                ),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                trainer.train_step(step=0)

                state_dict = trainer.optimizer.state_dict()
                assert len(state_dict["state"]) > 0, "Optimizer state should not be empty"

                for param_state in state_dict["state"].values():
                    assert any(isinstance(v, torch.Tensor) for v in param_state.values()), (
                        "Optimizer state should contain tensors"
                    )

                trainer.optimizer.load_state_dict(state_dict)

                if rank == 0:
                    print("FSDP state dict test passed")
        finally:
            reset_global_states()
            self._cleanup_fsdp_distributed()


# =============================================================================
# Distributed Optimizer Tests
# =============================================================================


@pytest.mark.mp
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

        return rank, world_size

    def _cleanup_distributed(self):
        """Cleanup distributed environment."""
        try:
            parallel_states.destroy_model_parallel()
        except Exception:
            pass
        # These run under torchrun, so the process group is session scoped for the
        # same reason as _cleanup_tp_distributed: destroying it here desynchronises
        # the ranks against the next test's setup.
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
            with (
                patch(
                    "ironcore.trainers.base_trainer.get_data_iterator",
                    return_value=create_mock_data_iterator(),
                ),
                patch(
                    "ironcore.trainers.base_trainer.get_evaluators",
                    return_value=create_mock_evaluators(),
                ),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                before = snapshot_params(trainer.model)
                loss, grad_norm, param_norm = trainer.train_step(step=0)

                assert_loss_valid(loss, grad_norm)
                assert_params_changed(trainer.model, before, "(DistributedOptimizer)")

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
            with (
                patch(
                    "ironcore.trainers.base_trainer.get_data_iterator",
                    return_value=create_mock_data_iterator(),
                ),
                patch(
                    "ironcore.trainers.base_trainer.get_evaluators",
                    return_value=create_mock_evaluators(),
                ),
            ):
                trainer = LanguageModelTrainer(
                    config,
                    create_mock_forward_step_func(),
                    F.cross_entropy,
                )
                trainer._initialize()

                trainer.train_step(step=0)

                state_dict = trainer.optimizer.state_dict()
                assert len(state_dict["state"]) > 0, "Optimizer state should not be empty"

                for param_state in state_dict["state"].values():
                    assert any(isinstance(v, torch.Tensor) for v in param_state.values()), (
                        "Optimizer state should contain tensors"
                    )

                trainer.optimizer.load_state_dict(state_dict)

                if rank == 0:
                    print("DistributedOptimizer state dict test passed")
        finally:
            reset_global_states()
            self._cleanup_distributed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
