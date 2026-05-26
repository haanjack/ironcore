# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive checkpoint switching tests for DistributedOptimizer.

Tests the following scenarios:
1. Universal checkpoint -> Universal checkpoint (baseline)
2. Distributed checkpoint -> Distributed checkpoint (baseline)
3. Universal checkpoint -> Distributed checkpoint (mode switch)
4. Distributed checkpoint -> Universal checkpoint (mode switch)

Verifies that training loss continues correctly regardless of checkpoint mode changes.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from ironcore.optimizer.distributed_optimizer import DistributedOptimizer

import pytest

pytestmark = [pytest.mark.mp, pytest.mark.checkpointing]


def get_shared_tmp_dir():
    """Get a shared temporary directory for all ranks."""
    # Use /tmp with a fixed name based on PID to share across ranks
    tmp_base = Path("/tmp") / f"ironcore_ckpt_test_{os.getppid()}"
    tmp_base.mkdir(parents=True, exist_ok=True)
    return tmp_base


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


def create_training_data(num_samples=32, hidden_size=64, device="cuda"):
    """Create reproducible training data."""
    torch.manual_seed(12345)
    x = torch.randn(num_samples, hidden_size, device=device)
    y = torch.randn(num_samples, 1, device=device)
    return x, y


def compute_loss(model, x, y):
    """Compute MSE loss."""
    pred = model(x)
    return nn.MSELoss()(pred, y)


def train_step(model, optimizer, x, y, scaler=None):
    """Single training step."""
    optimizer.zero_grad()
    loss = compute_loss(model, x, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def save_checkpoint_universal(path, model, optimizer, step, loss_history):
    """Save checkpoint in universal format (single file)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Get underlying model if DDP wrapped
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

    # Get optimizer state - if DistributedOptimizer, gather from all ranks
    is_dist_opt = hasattr(optimizer, "optimizer") and hasattr(optimizer, "local_param_indices")

    if is_dist_opt:
        # Gather optimizer states from all ranks
        dp_group = optimizer.process_group
        dp_size = dist.get_world_size(group=dp_group) if dp_group else 1
        dp_rank = dist.get_rank(group=dp_group) if dp_group else 0

        all_params = []
        for group in optimizer.optimizer.param_groups:
            for p in group["params"]:
                all_params.append(p)

        # Get underlying model for param names
        underlying_model = model.module if isinstance(model, DDP) else model
        param_to_name = {p: n for n, p in underlying_model.named_parameters()}

        full_state = {"state": {}, "param_groups": optimizer.optimizer.state_dict()["param_groups"]}

        for param_idx, param in enumerate(all_params):
            owner_rank = param_idx % dp_size if dp_size > 1 else 0
            param_name = param_to_name.get(param, None)

            if param_name is None:
                continue

            if dp_rank == owner_rank:
                local_state = optimizer.optimizer.state.get(param, {})
                state_dict = {}
                for k, v in local_state.items():
                    if isinstance(v, torch.Tensor):
                        state_dict[k] = v.cpu()
                    else:
                        state_dict[k] = v
            else:
                state_dict = None

            if dp_size > 1:
                state_list = [state_dict]
                dist.broadcast_object_list(state_list, src=owner_rank, group=dp_group)
                full_state["state"][param_name] = state_list[0]
            else:
                full_state["state"][param_name] = state_dict

        optimizer_state = full_state
    else:
        optimizer_state = optimizer.state_dict()

    # Only rank 0 saves for universal checkpoint
    if dist.get_rank() == 0:
        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "step": step,
            "loss_history": loss_history,
        }
        ckpt_file = path / "pytorch_model.bin"
        torch.save(checkpoint, ckpt_file)


def save_checkpoint_distributed(path, model, optimizer, step, loss_history, local_rank):
    """Save checkpoint in distributed format (per-rank files)."""
    path = Path(path)
    rank_path = path / f"rank{local_rank}"
    rank_path.mkdir(parents=True, exist_ok=True)

    # Get underlying model if DDP wrapped
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

    # Get optimizer state - if DistributedOptimizer, save local partition
    is_dist_opt = hasattr(optimizer, "optimizer") and hasattr(optimizer, "local_param_indices")

    if is_dist_opt:
        optimizer_state = optimizer.optimizer.state_dict()
    else:
        optimizer_state = optimizer.state_dict()

    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "step": step,
        "loss_history": loss_history,
    }
    ckpt_file = rank_path / "pytorch_model.bin"
    torch.save(checkpoint, ckpt_file)


def load_checkpoint_universal(path, model, optimizer, local_rank):
    """Load checkpoint from universal format."""
    path = Path(path)
    ckpt_file = path / "pytorch_model.bin"

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    checkpoint = torch.load(ckpt_file, map_location=f"cuda:{local_rank}", weights_only=False)

    # Load model state
    model_state = checkpoint["model_state_dict"]
    if isinstance(model, DDP):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    # Load optimizer state
    is_dist_opt = hasattr(optimizer, "optimizer") and hasattr(optimizer, "local_param_indices")

    if is_dist_opt:
        # Partition full state for this rank
        full_state = checkpoint["optimizer_state_dict"]
        underlying_model = model.module if isinstance(model, DDP) else model
        param_to_name = {p: n for n, p in underlying_model.named_parameters()}
        {n: p for n, p in underlying_model.named_parameters()}

        all_params = []
        for group in optimizer.optimizer.param_groups:
            for p in group["params"]:
                all_params.append(p)

        dp_size = optimizer.dp_size

        partitioned_state = {
            "state": {},
            "param_groups": full_state["param_groups"],
        }

        for param_idx, param in enumerate(all_params):
            if param_idx % dp_size != optimizer.dp_rank:
                continue

            param_name = param_to_name.get(param, None)
            if param_name is None or param_name not in full_state["state"]:
                continue

            state = full_state["state"][param_name]
            partitioned_state["state"][param] = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    partitioned_state["state"][param][k] = v.to(param.device)
                else:
                    partitioned_state["state"][param][k] = v

        optimizer.optimizer.load_state_dict(partitioned_state)
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["step"], checkpoint.get("loss_history", [])


def load_checkpoint_distributed(path, model, optimizer, local_rank):
    """Load checkpoint from distributed format (per-rank files)."""
    path = Path(path)
    rank_path = path / f"rank{local_rank}"
    ckpt_file = rank_path / "pytorch_model.bin"

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    checkpoint = torch.load(ckpt_file, map_location=f"cuda:{local_rank}", weights_only=False)

    # Load model state
    model_state = checkpoint["model_state_dict"]
    if isinstance(model, DDP):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    # Load optimizer state
    is_dist_opt = hasattr(optimizer, "optimizer") and hasattr(optimizer, "local_param_indices")

    if is_dist_opt:
        optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["step"], checkpoint.get("loss_history", [])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs",
)
class TestCheckpointSwitching:
    """Tests for checkpoint mode switching with DistributedOptimizer."""

    @pytest.fixture(scope="class")
    def distributed_env(self):
        """Setup distributed environment (class-scoped)."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # Create shared temp directory
        tmp_dir = get_shared_tmp_dir()

        yield local_rank, tmp_dir

        # Cleanup - only rank 0 removes
        if local_rank == 0:
            import shutil

            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        dist.barrier()

    def _create_model_and_optimizer(self, local_rank, use_dist_optimizer=True):
        """Create model, optimizer, and training data."""
        torch.manual_seed(42)  # Same seed for initial weights
        model = SimpleModel().cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])

        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        if use_dist_optimizer:
            optimizer = DistributedOptimizer(base_optimizer)
        else:
            optimizer = base_optimizer

        x, y = create_training_data(device=f"cuda:{local_rank}")
        return model, optimizer, x, y

    def _train_n_steps(self, model, optimizer, x, y, n_steps):
        """Train for N steps and return loss history."""
        losses = []
        for _ in range(n_steps):
            loss = train_step(model, optimizer, x, y)
            losses.append(loss)
        return losses

    def test_universal_to_universal(self, distributed_env):
        """Test saving and loading with universal checkpoint format."""
        local_rank, tmp_dir = distributed_env

        # Phase 1: Train and save universal checkpoint
        model, optimizer, x, y = self._create_model_and_optimizer(local_rank)

        losses_phase1 = self._train_n_steps(model, optimizer, x, y, 5)
        final_loss_phase1 = losses_phase1[-1]

        save_checkpoint_universal(
            tmp_dir / "universal_ckpt", model, optimizer, step=5, loss_history=losses_phase1
        )
        dist.barrier()

        # Phase 2: Load and continue training
        model2, optimizer2, _, _ = self._create_model_and_optimizer(local_rank)
        loaded_step, loaded_history = load_checkpoint_universal(
            tmp_dir / "universal_ckpt", model2, optimizer2, local_rank
        )

        assert loaded_step == 5
        assert len(loaded_history) == 5

        losses_phase2 = self._train_n_steps(model2, optimizer2, x, y, 5)

        # Verify loss continues (should be similar to end of phase 1, not reset)
        # Loss should be in similar range, not drastically different
        assert abs(losses_phase2[0] - final_loss_phase1) < 1.0, (
            f"Loss jumped unexpectedly: {losses_phase2[0]} vs {final_loss_phase1}"
        )

    def test_distributed_to_distributed(self, distributed_env):
        """Test saving and loading with distributed checkpoint format."""
        local_rank, tmp_dir = distributed_env

        # Phase 1: Train and save distributed checkpoint
        model, optimizer, x, y = self._create_model_and_optimizer(local_rank)

        losses_phase1 = self._train_n_steps(model, optimizer, x, y, 5)
        final_loss_phase1 = losses_phase1[-1]

        save_checkpoint_distributed(
            tmp_dir / "distributed_ckpt",
            model,
            optimizer,
            step=5,
            loss_history=losses_phase1,
            local_rank=local_rank,
        )
        dist.barrier()

        # Phase 2: Load and continue training
        model2, optimizer2, _, _ = self._create_model_and_optimizer(local_rank)
        loaded_step, loaded_history = load_checkpoint_distributed(
            tmp_dir / "distributed_ckpt", model2, optimizer2, local_rank
        )

        assert loaded_step == 5
        assert len(loaded_history) == 5

        losses_phase2 = self._train_n_steps(model2, optimizer2, x, y, 5)

        # Verify loss continues
        assert abs(losses_phase2[0] - final_loss_phase1) < 1.0, (
            f"Loss jumped unexpectedly: {losses_phase2[0]} vs {final_loss_phase1}"
        )

    def test_universal_to_distributed(self, distributed_env):
        """Test switching from universal checkpoint to distributed mode."""
        local_rank, tmp_dir = distributed_env

        # Phase 1: Train and save as UNIVERSAL checkpoint
        model, optimizer, x, y = self._create_model_and_optimizer(local_rank)

        losses_phase1 = self._train_n_steps(model, optimizer, x, y, 5)

        save_checkpoint_universal(
            tmp_dir / "universal_ckpt", model, optimizer, step=5, loss_history=losses_phase1
        )
        dist.barrier()

        # Phase 2: Load from UNIVERSAL, save as DISTRIBUTED
        model2, optimizer2, _, _ = self._create_model_and_optimizer(local_rank)
        loaded_step, _ = load_checkpoint_universal(
            tmp_dir / "universal_ckpt", model2, optimizer2, local_rank
        )

        # Verify state loaded correctly
        assert loaded_step == 5

        # Train a few steps
        losses_phase2 = self._train_n_steps(model2, optimizer2, x, y, 3)
        final_loss_phase2 = losses_phase2[-1]

        # Save as distributed
        save_checkpoint_distributed(
            tmp_dir / "distributed_ckpt",
            model2,
            optimizer2,
            step=8,
            loss_history=losses_phase1 + losses_phase2,
            local_rank=local_rank,
        )
        dist.barrier()

        # Phase 3: Load from DISTRIBUTED and continue
        model3, optimizer3, _, _ = self._create_model_and_optimizer(local_rank)
        loaded_step3, _ = load_checkpoint_distributed(
            tmp_dir / "distributed_ckpt", model3, optimizer3, local_rank
        )

        assert loaded_step3 == 8

        losses_phase3 = self._train_n_steps(model3, optimizer3, x, y, 3)

        # Verify loss continuity
        assert abs(losses_phase3[0] - final_loss_phase2) < 1.0, (
            f"Loss jumped after checkpoint mode switch: {losses_phase3[0]} vs {final_loss_phase2}"
        )

    def test_distributed_to_universal(self, distributed_env):
        """Test switching from distributed checkpoint to universal mode."""
        local_rank, tmp_dir = distributed_env

        # Phase 1: Train and save as DISTRIBUTED checkpoint
        model, optimizer, x, y = self._create_model_and_optimizer(local_rank)

        losses_phase1 = self._train_n_steps(model, optimizer, x, y, 5)

        save_checkpoint_distributed(
            tmp_dir / "distributed_ckpt",
            model,
            optimizer,
            step=5,
            loss_history=losses_phase1,
            local_rank=local_rank,
        )
        dist.barrier()

        # Phase 2: Load from DISTRIBUTED, save as UNIVERSAL
        model2, optimizer2, _, _ = self._create_model_and_optimizer(local_rank)
        loaded_step, _ = load_checkpoint_distributed(
            tmp_dir / "distributed_ckpt", model2, optimizer2, local_rank
        )

        assert loaded_step == 5

        # Train a few steps
        losses_phase2 = self._train_n_steps(model2, optimizer2, x, y, 3)
        final_loss_phase2 = losses_phase2[-1]

        # Save as universal
        save_checkpoint_universal(
            tmp_dir / "universal_ckpt",
            model2,
            optimizer2,
            step=8,
            loss_history=losses_phase1 + losses_phase2,
        )
        dist.barrier()

        # Phase 3: Load from UNIVERSAL and continue
        model3, optimizer3, _, _ = self._create_model_and_optimizer(local_rank)
        loaded_step3, _ = load_checkpoint_universal(
            tmp_dir / "universal_ckpt", model3, optimizer3, local_rank
        )

        assert loaded_step3 == 8

        losses_phase3 = self._train_n_steps(model3, optimizer3, x, y, 3)

        # Verify loss continuity
        assert abs(losses_phase3[0] - final_loss_phase2) < 1.0, (
            f"Loss jumped after checkpoint mode switch: {losses_phase3[0]} vs {final_loss_phase2}"
        )

    def test_optimizer_state_correctness_after_switch(self, distributed_env):
        """Verify optimizer states (Adam moments) are preserved correctly across mode switches."""
        local_rank, tmp_dir = distributed_env

        # Phase 1: Train with DistributedOptimizer, save universal
        model1, optimizer1, x, y = self._create_model_and_optimizer(local_rank)
        self._train_n_steps(model1, optimizer1, x, y, 10)

        # Get optimizer states before saving
        states_before = {}
        underlying_model = model1.module if isinstance(model1, DDP) else model1
        for name, param in underlying_model.named_parameters():
            if param in optimizer1.optimizer.state:
                state = optimizer1.optimizer.state[param]
                if "exp_avg" in state:
                    states_before[name] = {
                        "exp_avg": state["exp_avg"].clone(),
                        "exp_avg_sq": state["exp_avg_sq"].clone(),
                    }

        save_checkpoint_universal(
            tmp_dir / "universal_ckpt", model1, optimizer1, step=10, loss_history=[]
        )
        dist.barrier()

        # Phase 2: Load and verify states
        model2, optimizer2, _, _ = self._create_model_and_optimizer(local_rank)
        load_checkpoint_universal(tmp_dir / "universal_ckpt", model2, optimizer2, local_rank)

        # Verify optimizer states match
        underlying_model2 = model2.module if isinstance(model2, DDP) else model2
        for name, param in underlying_model2.named_parameters():
            if name in states_before and param in optimizer2.optimizer.state:
                state_after = optimizer2.optimizer.state[param]
                state_before = states_before[name]

                # Check moments are approximately preserved
                assert torch.allclose(
                    state_before["exp_avg"], state_after["exp_avg"], rtol=1e-4, atol=1e-6
                ), f"exp_avg mismatch for {name}"

                assert torch.allclose(
                    state_before["exp_avg_sq"], state_after["exp_avg_sq"], rtol=1e-4, atol=1e-6
                ), f"exp_avg_sq mismatch for {name}"

    def test_loss_trajectory_consistency(self, distributed_env):
        """Test that training loss trajectory is consistent across checkpoint mode switches."""
        local_rank, tmp_dir = distributed_env

        # Reference: Train 15 steps without any checkpoint
        model_ref, optimizer_ref, x_ref, y_ref = self._create_model_and_optimizer(local_rank)
        reference_losses = self._train_n_steps(model_ref, optimizer_ref, x_ref, y_ref, 15)

        # Test: Train 5 steps, checkpoint, train 5 steps, switch mode, train 5 steps
        model, optimizer, x, y = self._create_model_and_optimizer(local_rank)

        # Phase 1: Train 5 steps
        losses_1 = self._train_n_steps(model, optimizer, x, y, 5)
        save_checkpoint_universal(
            tmp_dir / "ckpt_switch", model, optimizer, step=5, loss_history=losses_1
        )
        dist.barrier()

        # Phase 2: Load, train 5 steps, switch mode
        model2, optimizer2, _, _ = self._create_model_and_optimizer(local_rank)
        load_checkpoint_universal(tmp_dir / "ckpt_switch", model2, optimizer2, local_rank)
        losses_2 = self._train_n_steps(model2, optimizer2, x, y, 5)

        # Save as distributed
        save_checkpoint_distributed(
            tmp_dir / "ckpt_switch_dist",
            model2,
            optimizer2,
            step=10,
            loss_history=losses_1 + losses_2,
            local_rank=local_rank,
        )
        dist.barrier()

        # Phase 3: Load from distributed, train 5 steps
        model3, optimizer3, _, _ = self._create_model_and_optimizer(local_rank)
        load_checkpoint_distributed(tmp_dir / "ckpt_switch_dist", model3, optimizer3, local_rank)
        losses_3 = self._train_n_steps(model3, optimizer3, x, y, 5)

        # Compare loss trajectories
        test_losses = losses_1 + losses_2 + losses_3

        # Allow some numerical difference due to floating point
        for i, (ref, test) in enumerate(zip(reference_losses, test_losses, strict=False)):
            relative_diff = abs(ref - test) / max(abs(ref), 1e-8)
            assert relative_diff < 0.01, (
                f"Step {i}: loss diverged - reference={ref}, test={test}, diff={relative_diff:.4f}"
            )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs",
)
class TestCheckpointEdgeCases:
    """Edge case tests for checkpoint switching."""

    @pytest.fixture(scope="class")
    def distributed_env(self):
        """Setup distributed environment (class-scoped)."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        tmp_dir = get_shared_tmp_dir()
        yield local_rank, tmp_dir

        if local_rank == 0:
            import shutil

pytestmark = [pytest.mark.mp, pytest.mark.checkpointing]

            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_empty_optimizer_state(self, distributed_env):
        """Test saving/loading when optimizer has no state (fresh optimizer)."""
        local_rank, tmp_dir = distributed_env

        model = SimpleModel().cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])
        optimizer = DistributedOptimizer(AdamW(model.parameters(), lr=1e-3))

        # Save without any training (no optimizer state)
        save_checkpoint_universal(tmp_dir / "empty_ckpt", model, optimizer, step=0, loss_history=[])
        dist.barrier()

        # Load should work
        model2 = SimpleModel().cuda(local_rank)
        model2 = DDP(model2, device_ids=[local_rank])
        optimizer2 = DistributedOptimizer(AdamW(model2.parameters(), lr=1e-3))

        loaded_step, _ = load_checkpoint_universal(
            tmp_dir / "empty_ckpt", model2, optimizer2, local_rank
        )
        assert loaded_step == 0
