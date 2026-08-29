# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for activation spill activation offloading (merged forward spill + backward prefetch).

Tests cover:
1. ActivationSpillManager: spill, prefetch, free-after-consume
2. Pool budget enforcement for activations
3. Gradient accumulation: correct ordering across micro-batches
4. Config validation: activation_spill vs activation_recompute mutual exclusion
5. Coexistence with weight streaming
6. Scheduler integration: lifecycle hooks

CUDA tests are gated on torch.cuda.is_available().
"""

import pytest
import torch

from ironcore.config import OffloadConfig
from ironcore.offload.hooks import ActivationSpillManager
from ironcore.offload.memory_pool import PinnedMemoryPool

cuda_available = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

pytestmark = [pytest.mark.cuda]


# ---------------------------------------------------------------------------
# ActivationSpillManager (CUDA required)
# ---------------------------------------------------------------------------


@skip_no_cuda
class TestActivationSpillManager:
    """Test ActivationSpillManager spill and prefetch mechanics."""

    def _make_manager(self, num_layers=2, grad_accum=1):
        from ironcore.offload.transfer_engine import MemoryTransferEngine

        pool = PinnedMemoryPool(chunk_bytes=4 * 1024 * 1024)
        device = torch.device("cuda:0")
        engine = MemoryTransferEngine(device=device)
        manager = ActivationSpillManager(
            pool=pool,
            engine=engine,
            num_layers=num_layers,
            gradient_accumulation_steps=grad_accum,
        )
        return manager, pool, engine

    def test_spill_and_prefetch_roundtrip(self):
        """Spill a tensor D2H, then prefetch H2D. Data should match."""
        manager, pool, engine = self._make_manager(num_layers=1, grad_accum=1)

        device = torch.device("cuda:0")
        original = torch.randn(32, 32, device=device)

        # Forward: spill
        manager.on_microbatch_forward_start(0)
        manager.on_sublayer_forward(0, 0, original)
        manager.on_microbatch_forward_end()

        assert manager.pending_count == 1
        assert pool.total_used_bytes > 0

        # Backward: prefetch
        manager.on_microbatch_backward_start(0)
        result = manager.on_sublayer_backward(0, 0, original.shape, original.dtype, original.device)
        manager.on_microbatch_backward_end()

        assert manager.pending_count == 0
        assert torch.allclose(original, result, atol=1e-6)

    def test_two_sublayers_per_layer(self):
        """Spill both sub-layers (layer input + post-attention residual)."""
        manager, _, _ = self._make_manager(num_layers=1, grad_accum=1)

        device = torch.device("cuda:0")
        hidden = torch.randn(16, 16, device=device)
        norm_input = torch.randn(16, 16, device=device)

        # Forward: spill both sub-layers
        manager.on_microbatch_forward_start(0)
        manager.on_sublayer_forward(0, 0, hidden)
        manager.on_sublayer_forward(0, 1, norm_input)
        manager.on_microbatch_forward_end()

        assert manager.pending_count == 2

        # Backward: prefetch in reverse order
        manager.on_microbatch_backward_start(0)
        result1 = manager.on_sublayer_backward(
            0, 1, norm_input.shape, norm_input.dtype, norm_input.device
        )
        result0 = manager.on_sublayer_backward(0, 0, hidden.shape, hidden.dtype, hidden.device)
        manager.on_microbatch_backward_end()

        assert manager.pending_count == 0
        assert torch.allclose(hidden, result0, atol=1e-6)
        assert torch.allclose(norm_input, result1, atol=1e-6)

    def test_gradient_accumulation(self):
        """Multiple micro-batches: each has its own set of spilled activations."""
        grad_accum = 3
        manager, _, _ = self._make_manager(num_layers=1, grad_accum=grad_accum)

        device = torch.device("cuda:0")
        originals = [torch.randn(8, 8, device=device) for _ in range(grad_accum)]

        # Forward: spill all micro-batches
        for mb in range(grad_accum):
            manager.on_microbatch_forward_start(mb)
            manager.on_sublayer_forward(0, 0, originals[mb])
            manager.on_microbatch_forward_end()

        assert manager.pending_count == grad_accum

        # Backward: prefetch each micro-batch
        for mb in range(grad_accum):
            orig = originals[mb]
            manager.on_microbatch_backward_start(mb)
            result = manager.on_sublayer_backward(0, 0, orig.shape, orig.dtype, orig.device)
            manager.on_microbatch_backward_end()
            assert torch.allclose(orig, result, atol=1e-6)

        assert manager.pending_count == 0

    def test_free_after_consume(self):
        """Pinned memory is freed after backward consumes activation."""
        manager, pool, engine = self._make_manager(num_layers=1, grad_accum=1)

        device = torch.device("cuda:0")
        tensor = torch.randn(32, 32, device=device)

        used_before = pool.total_used_bytes

        # Forward: spill
        manager.on_microbatch_forward_start(0)
        manager.on_sublayer_forward(0, 0, tensor)
        manager.on_microbatch_forward_end()

        used_after_spill = pool.total_used_bytes
        assert used_after_spill > used_before

        # Backward: prefetch + free
        manager.on_microbatch_backward_start(0)
        manager.on_sublayer_backward(0, 0, tensor.shape, tensor.dtype, tensor.device)
        manager.on_microbatch_backward_end()

        assert pool.total_used_bytes == used_before

    def test_from_config(self):
        from ironcore.offload.transfer_engine import MemoryTransferEngine

        config = OffloadConfig(
            enabled=True,
            activation_spill=True,
            pinned_chunk_gb=0.01,
            pinned_memory_pool_gb=0.1,
        )
        pool = PinnedMemoryPool.from_config(config)
        device = torch.device("cuda:0")
        engine = MemoryTransferEngine(device=device)

        manager = ActivationSpillManager.from_config(
            pool=pool,
            engine=engine,
            num_layers=4,
            gradient_accumulation_steps=2,
        )
        assert manager._num_layers == 4
        assert manager._gradient_accumulation_steps == 2

    def test_shutdown_cleans_up(self):
        """Shutdown frees all pending activations."""
        manager, pool, _ = self._make_manager(num_layers=1, grad_accum=1)

        device = torch.device("cuda:0")
        tensor = torch.randn(16, 16, device=device)

        manager.on_microbatch_forward_start(0)
        manager.on_sublayer_forward(0, 0, tensor)
        manager.on_microbatch_forward_end()

        assert manager.pending_count == 1
        manager.shutdown()
        assert manager.pending_count == 0

    def test_repr(self):
        manager, _, _ = self._make_manager()
        r = repr(manager)
        assert "ActivationSpillManager" in r
        assert "layers=2" in r


# ---------------------------------------------------------------------------
# Scheduler integration (CUDA required)
# ---------------------------------------------------------------------------


@skip_no_cuda
class TestSchedulerActivationSpill:
    """Test scheduler with activation spilling enabled."""

    def _make_simple_model(self, num_layers=2):
        from ironcore.config import MainConfig
        from ironcore.models.transformer import TransformerModel

        model_config = __import__(
            "ironcore.config.config_model", fromlist=["ModelConfig"]
        ).ModelConfig()
        model_config.num_layers = num_layers
        model_config.num_attention_heads = 4
        model_config.num_attention_groups = 4
        # d_ffn, not ffn_hidden_size — the latter is not a ModelConfig field. Same
        # value as the default here, so this only ever looked like it was setting it.
        model_config.d_ffn = 512 * 4

        config = MainConfig(
            model=model_config,
            init=__import__("ironcore.config.config_trainer", fromlist=["InitConfig"]).InitConfig(),
            optim=__import__(
                "ironcore.config.config_optim", fromlist=["OptimConfig"]
            ).OptimConfig(),
            data=__import__("ironcore.config.config_data", fromlist=["DataConfig"]).DataConfig(),
            parallel=__import__(
                "ironcore.config.config_parallel", fromlist=["ParallelConfig"]
            ).ParallelConfig(),
            trainer=__import__(
                "ironcore.config.config_trainer", fromlist=["TrainerConfig"]
            ).TrainerConfig(),
            operation=__import__(
                "ironcore.config.config_trainer", fromlist=["OperationConfig"]
            ).OperationConfig(
                train_steps=10,
            ),
            utils=__import__(
                "ironcore.config.config_utils", fromlist=["UtilsConfig"]
            ).UtilsConfig(),
            profiler=__import__(
                "ironcore.config.config_utils", fromlist=["ProfilerConfig"]
            ).ProfilerConfig(),
            peft=__import__("ironcore.config.config_peft", fromlist=["PEFTConfig"]).PEFTConfig(),
            alignment=__import__(
                "ironcore.config.config_alignment", fromlist=["AlignmentConfig"]
            ).AlignmentConfig(),
            offload=OffloadConfig(
                enabled=True,
                activation_spill=True,
                pinned_chunk_gb=0.05,
                pinned_memory_pool_gb=0.2,
            ),
        )
        config.trainer.micro_batch_size = 1
        config.trainer.train_batch_size = 1
        config.trainer.gradient_accumulation_steps = 1
        config.parallel.world_size = 1

        model = TransformerModel(config)
        model = model.to(device=torch.device("cuda:0"), dtype=torch.bfloat16)
        return model, config

    def test_spill_manager_created(self):
        from ironcore.offload.scheduler import ExecutionScheduler

        model, config = self._make_simple_model()
        scheduler = ExecutionScheduler.from_model(
            model=model,
            config=config.offload,
            device=torch.device("cuda:0"),
        )
        assert scheduler is not None
        assert scheduler.spill_manager is not None
        assert scheduler.spill_manager._num_layers == 2

    def test_forward_spills_activations(self):
        """Forward pass should spill activations via scheduler hooks."""
        from ironcore.offload.scheduler import ExecutionScheduler

        model, config = self._make_simple_model()
        scheduler = ExecutionScheduler.from_model(
            model=model,
            config=config.offload,
            device=torch.device("cuda:0"),
        )
        assert scheduler is not None

        # Attach scheduler to model (normally done by trainer)
        model._offload_scheduler = scheduler

        scheduler.on_microbatch_forward_start(0)

        # Run forward pass manually
        device = torch.device("cuda:0")
        hidden = torch.randn(1, 4, 512, device=device, dtype=torch.bfloat16)
        mask = torch.ones(1, 1, 4, 4, device=device)

        # After forward, spill manager should have activations
        with torch.no_grad():
            model(hidden, mask, None)

        scheduler.on_microbatch_forward_end()

        # 2 layers * 2 sub-layers = 4 spilled activations
        assert scheduler.spill_manager.pending_count == 4

    def test_spill_disables_checkpointing(self):
        """When activation_spill is enabled, checkpointing is disabled."""
        from ironcore.offload.scheduler import ExecutionScheduler

        model, config = self._make_simple_model()
        # Enable checkpointing in config
        config.operation.activation_recompute = True
        model.activation_recompute = True

        scheduler = ExecutionScheduler.from_model(
            model=model,
            config=config.offload,
            device=torch.device("cuda:0"),
        )
        assert scheduler is not None

        # Attach scheduler
        model._offload_scheduler = scheduler

        # Forward should NOT use checkpointing because spill is active
        # (The forward loop checks activation_spill_active)
        device = torch.device("cuda:0")
        hidden = torch.randn(1, 4, 512, device=device, dtype=torch.bfloat16)
        mask = torch.ones(1, 1, 4, 4, device=device)

        scheduler.on_microbatch_forward_start(0)
        with torch.no_grad():
            output = model(hidden, mask, None)
        scheduler.on_microbatch_forward_end()

        # Should still work (no crash from checkpoint + spill conflict)
        assert output.shape[0] == 1

    def test_full_layer_granularity_spills(self):
        """full_layer granularity should spill 1 activation per layer."""
        from ironcore.offload.scheduler import ExecutionScheduler

        model, config = self._make_simple_model()
        config.offload.activation_spill_granularity = "full_layer"

        scheduler = ExecutionScheduler.from_model(
            model=model,
            config=config.offload,
            device=torch.device("cuda:0"),
        )
        assert scheduler is not None
        assert scheduler._activation_spill_granularity == "full_layer"

        # Attach scheduler to model (normally done by trainer)
        model._offload_scheduler = scheduler

        scheduler.on_microbatch_forward_start(0)

        device = torch.device("cuda:0")
        hidden = torch.randn(1, 4, 512, device=device, dtype=torch.bfloat16)
        mask = torch.ones(1, 1, 4, 4, device=device)

        with torch.no_grad():
            model(hidden, mask, None)

        scheduler.on_microbatch_forward_end()

        # 2 layers * 1 sub-layer = 2 spilled activations (vs 4 for sub_layer)
        assert scheduler.spill_manager.pending_count == 2


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestActivationSpillConfigValidation:
    """Test activation spill config validation rules."""

    def test_activation_spill_requires_enabled(self):
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.activation_spill = True
        # enabled defaults to False
        with pytest.raises(ValueError, match="requires offload.enabled"):
            _config_validation(config)

    def test_invalid_granularity_raises(self):
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.enabled = True
        config.offload.activation_spill = True
        config.offload.activation_spill_granularity = "per_token"
        with pytest.raises(ValueError, match="must be 'sub_layer' or 'full_layer'"):
            _config_validation(config)

    def test_full_layer_granularity_accepted(self):
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.enabled = True
        config.offload.activation_spill = True
        config.offload.activation_spill_granularity = "full_layer"
        _config_validation(config)  # should not raise
        assert config.offload.activation_spill_granularity == "full_layer"

    def test_spill_auto_disables_recompute(self):
        """activation_spill=true auto-disables activation_recompute."""
        import warnings

        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.enabled = True
        config.offload.activation_spill = True
        config.operation.activation_recompute = True

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _config_validation(config)

        assert config.operation.activation_recompute is False
        assert len(w) == 1
        assert "Disabling activation_recompute" in str(w[0].message)

    def test_valid_spill_config(self):
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.enabled = True
        config.offload.activation_spill = True
        # Should not raise
        _config_validation(config)


def _make_minimal_main_config():
    """Create a minimal MainConfig with valid required fields for testing."""
    from ironcore.config import MainConfig

    config = MainConfig(
        model=__import__("ironcore.config.config_model", fromlist=["ModelConfig"]).ModelConfig(),
        init=__import__("ironcore.config.config_trainer", fromlist=["InitConfig"]).InitConfig(),
        optim=__import__("ironcore.config.config_optim", fromlist=["OptimConfig"]).OptimConfig(),
        data=__import__("ironcore.config.config_data", fromlist=["DataConfig"]).DataConfig(),
        parallel=__import__(
            "ironcore.config.config_parallel", fromlist=["ParallelConfig"]
        ).ParallelConfig(),
        trainer=__import__(
            "ironcore.config.config_trainer", fromlist=["TrainerConfig"]
        ).TrainerConfig(),
        operation=__import__(
            "ironcore.config.config_trainer", fromlist=["OperationConfig"]
        ).OperationConfig(
            train_steps=100,
        ),
        utils=__import__("ironcore.config.config_utils", fromlist=["UtilsConfig"]).UtilsConfig(),
        profiler=__import__(
            "ironcore.config.config_utils", fromlist=["ProfilerConfig"]
        ).ProfilerConfig(),
        peft=__import__("ironcore.config.config_peft", fromlist=["PEFTConfig"]).PEFTConfig(),
        alignment=__import__(
            "ironcore.config.config_alignment", fromlist=["AlignmentConfig"]
        ).AlignmentConfig(),
        offload=OffloadConfig(),
    )
    config.trainer.micro_batch_size = 4
    config.trainer.train_batch_size = 4
    config.trainer.gradient_accumulation_steps = 1
    config.parallel.world_size = 1
    return config
