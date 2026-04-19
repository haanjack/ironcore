# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for M2 weight streaming components.

Tests cover:
1. PinnedMemoryPool: allocation, free, reuse, budget enforcement
2. MemoryTransferEngine: H2D/D2H transfers, event synchronization
3. TileManager: register params, apply tiles, precision conversion
4. ExecutionScheduler: from_model, lifecycle hooks, layer prefetch
5. Config validation for M2 fields

CUDA tests are gated on torch.cuda.is_available().
"""

import pytest
import torch
from torch import nn

from ironcore.offload.config import OffloadConfig
from ironcore.offload.memory_pool import PinnedMemoryPool
from ironcore.offload.tile_manager import TileManager

# ---------------------------------------------------------------------------
# PinnedMemoryPool
# ---------------------------------------------------------------------------


class TestPinnedMemoryPool:
    """Test PinnedMemoryPool allocation and free mechanics."""

    def test_allocate_returns_pinned_tensor(self):
        pool = PinnedMemoryPool(chunk_bytes=1024 * 1024)  # 1MB chunks
        t = pool.allocate(256, torch.float32)
        assert t.shape == (256,)
        assert t.dtype == torch.float32
        assert t.device.type == "cpu"

    def test_allocate_and_free_reuse(self):
        pool = PinnedMemoryPool(chunk_bytes=1024 * 1024)
        t1 = pool.allocate(256, torch.float32)
        pool.free(t1)
        pool.allocate(256, torch.float32)
        # Same chunk, should reuse freed space
        assert pool.total_used_bytes == 256 * 4

    def test_allocate_exceeds_budget_raises(self):
        pool = PinnedMemoryPool(chunk_bytes=1024, max_total_bytes=512)
        with pytest.raises(RuntimeError, match="budget exceeded"):
            pool.allocate(256, torch.float32)  # 1024 bytes > 512 budget

    def test_multiple_dtypes(self):
        pool = PinnedMemoryPool(chunk_bytes=1024 * 1024)
        t32 = pool.allocate(64, torch.float32)
        t16 = pool.allocate(64, torch.float16)
        assert t32.dtype == torch.float32
        assert t16.dtype == torch.float16

    def test_from_config(self):
        config = OffloadConfig(pinned_chunk_gb=1.0, pinned_memory_pool_gb=10.0)
        pool = PinnedMemoryPool.from_config(config)
        assert pool._chunk_bytes == 1 * 1024**3
        assert pool._max_total_bytes == 10 * 1024**3

    def test_allocate_temp_context(self):
        pool = PinnedMemoryPool(chunk_bytes=1024 * 1024)
        with pool.allocate_temp(128, torch.float32) as t:
            assert t.shape == (128,)
        # After context exit, memory should be freed
        assert pool.total_used_bytes == 0

    def test_repr(self):
        pool = PinnedMemoryPool(chunk_bytes=1024)
        assert "PinnedMemoryPool" in repr(pool)


# ---------------------------------------------------------------------------
# MemoryTransferEngine (CUDA required)
# ---------------------------------------------------------------------------


cuda_available = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")


@skip_no_cuda
class TestMemoryTransferEngine:
    """Test async DMA transfers."""

    def test_h2d_transfer(self):
        from ironcore.offload.transfer_engine import MemoryTransferEngine

        device = torch.device("cuda:0")
        engine = MemoryTransferEngine(device=device)

        src = torch.randn(64, pin_memory=True)
        dst = torch.empty(64, device=device)

        handle = engine.submit_h2d(src, dst)
        engine.wait(handle)
        assert handle.completed
        assert torch.allclose(src, dst.cpu())

    def test_d2h_transfer(self):
        from ironcore.offload.transfer_engine import MemoryTransferEngine

        device = torch.device("cuda:0")
        engine = MemoryTransferEngine(device=device)

        src = torch.randn(64, device=device)
        dst = torch.empty(64, pin_memory=True)

        handle = engine.submit_d2h(src, dst)
        engine.wait(handle)
        assert handle.completed
        assert torch.allclose(src.cpu(), dst)

    def test_synchronize_all(self):
        from ironcore.offload.transfer_engine import MemoryTransferEngine

        device = torch.device("cuda:0")
        engine = MemoryTransferEngine(device=device)

        handles = []
        for _ in range(3):
            src = torch.randn(32, pin_memory=True)
            dst = torch.empty(32, device=device)
            handles.append(engine.submit_h2d(src, dst))

        engine.synchronize()
        for h in handles:
            assert h.completed

    def test_pending_count(self):
        from ironcore.offload.transfer_engine import MemoryTransferEngine

        device = torch.device("cuda:0")
        engine = MemoryTransferEngine(device=device)

        src = torch.randn(32, pin_memory=True)
        dst = torch.empty(32, device=device)
        engine.submit_h2d(src, dst)
        assert engine.pending_count >= 1

        engine.synchronize()
        assert engine.drain_completed() == 0

    def test_shape_mismatch_raises(self):
        from ironcore.offload.transfer_engine import MemoryTransferEngine

        device = torch.device("cuda:0")
        engine = MemoryTransferEngine(device=device)

        src = torch.randn(16, pin_memory=True)
        dst = torch.empty(32, device=device)
        with pytest.raises(AssertionError, match="Shape mismatch"):
            engine.submit_h2d(src, dst)


# ---------------------------------------------------------------------------
# TileManager
# ---------------------------------------------------------------------------


@skip_no_cuda
class TestTileManager:
    """Test weight tiling and parameter application."""

    def test_register_layer(self):
        pool = PinnedMemoryPool(chunk_bytes=4 * 1024 * 1024)
        device = torch.device("cuda:0")
        tm = TileManager(pool=pool, device=device, precision="fp32")

        params = [nn.Parameter(torch.randn(64, 64, device=device))]
        group = tm.register_layer(layer_idx=0, params=params)

        assert group.layer_idx == 0
        assert len(group.tiles) == 1
        assert tm.num_groups == 1

    def test_apply_tiles_roundtrip(self):
        pool = PinnedMemoryPool(chunk_bytes=4 * 1024 * 1024)
        device = torch.device("cuda:0")
        tm = TileManager(pool=pool, device=device, precision="fp32")

        original = torch.randn(32, 32, device=device)
        param = nn.Parameter(original.clone())
        group = tm.register_layer(layer_idx=0, params=[param])

        # Simulate: modify host tensor, apply back
        group.tiles[0].host_tensor.fill_(0.0)
        group.tiles[0].gpu_tensor.copy_(group.tiles[0].host_tensor.to(device))

        tm.apply_tiles_to_params(group)
        assert torch.all(param.data == 0)

    def test_precision_conversion(self):
        pool = PinnedMemoryPool(chunk_bytes=4 * 1024 * 1024)
        device = torch.device("cuda:0")
        tm = TileManager(pool=pool, device=device, precision="bf16")

        param = nn.Parameter(torch.randn(16, 16, device=device))
        group = tm.register_layer(layer_idx=0, params=[param])

        # Host tensor should be bf16
        assert group.tiles[0].host_tensor.dtype == torch.bfloat16
        # GPU staging should be original dtype
        assert group.tiles[0].gpu_tensor.dtype == torch.float32

    def test_invalid_precision_raises(self):
        pool = PinnedMemoryPool(chunk_bytes=1024)
        with pytest.raises(ValueError, match="Invalid precision"):
            TileManager(pool=pool, device=torch.device("cpu"), precision="fp8")

    def test_from_config_uses_weight_storage_precision(self):
        config = OffloadConfig(weight_storage_precision="bf16")
        pool = PinnedMemoryPool(chunk_bytes=4 * 1024 * 1024)
        tm = TileManager.from_config(config, pool, torch.device("cpu"))
        assert tm._storage_dtype == torch.bfloat16

    def test_from_config_defaults_to_fp32(self):
        config = OffloadConfig()
        pool = PinnedMemoryPool(chunk_bytes=4 * 1024 * 1024)
        tm = TileManager.from_config(config, pool, torch.device("cpu"))
        assert tm._storage_dtype == torch.float32


# ---------------------------------------------------------------------------
# ExecutionScheduler
# ---------------------------------------------------------------------------


@skip_no_cuda
class TestExecutionScheduler:
    """Test scheduler lifecycle and weight streaming."""

    def _make_simple_model(self, num_layers=2, hidden=32):
        """Create a minimal model with TransformerLayers."""
        from ironcore.config import MainConfig
        from ironcore.models.transformer import TransformerModel

        # Build minimal config
        model_config = __import__(
            "ironcore.config.config_model", fromlist=["ModelConfig"]
        ).ModelConfig()
        model_config.num_layers = num_layers
        model_config.hidden_size = hidden
        model_config.num_attention_heads = 4
        model_config.num_attention_groups = 4
        model_config.ffn_hidden_size = hidden * 4

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
                weight_offload=True,
                pinned_chunk_gb=0.05,  # 50MB chunks
                pinned_memory_pool_gb=0.2,  # 200MB budget
            ),
        )
        config.trainer.micro_batch_size = 1
        config.trainer.train_batch_size = 1
        config.trainer.gradient_accumulation_steps = 1
        config.parallel.world_size = 1

        model = TransformerModel(config)
        model = model.to(device=torch.device("cuda:0"))
        return model, config

    def test_from_model_creates_scheduler(self):
        from ironcore.offload.scheduler import ExecutionScheduler

        model, config = self._make_simple_model()
        scheduler = ExecutionScheduler.from_model(
            model=model,
            config=config.offload,
            device=torch.device("cuda:0"),
        )
        assert scheduler is not None
        assert scheduler.is_active
        assert scheduler.num_registered_layers > 0

    def test_from_model_disabled_returns_none(self):
        from ironcore.offload.scheduler import ExecutionScheduler

        model, _ = self._make_simple_model()
        config = OffloadConfig(enabled=False, weight_offload=False)
        scheduler = ExecutionScheduler.from_model(
            model=model, config=config, device=torch.device("cuda:0")
        )
        assert scheduler is None

    def test_lifecycle_hooks_no_error(self):
        from ironcore.offload.scheduler import ExecutionScheduler

        model, config = self._make_simple_model()
        scheduler = ExecutionScheduler.from_model(
            model=model, config=config.offload, device=torch.device("cuda:0")
        )
        assert scheduler is not None

        # Full lifecycle should not raise
        scheduler.on_training_step_start()
        for i in range(len(model.layers)):
            scheduler.on_layer_start(i)
            scheduler.on_layer_end(i)
        scheduler.on_backward_pass_end()
        scheduler.on_training_step_end()

    def test_lora_params_excluded(self):
        from ironcore.offload.scheduler import ExecutionScheduler

        model, config = self._make_simple_model()
        # Mark a parameter as non-offloadable (like LoRA)
        for param in model.layers[0].parameters():
            param.offloadable = False
            break

        scheduler = ExecutionScheduler.from_model(
            model=model, config=config.offload, device=torch.device("cuda:0")
        )
        assert scheduler is not None
        # Group for layer 0 should have fewer params
        group = scheduler.get_group(0)
        if group is not None:
            # At least one param was excluded
            total_params = sum(1 for _ in model.layers[0].parameters())
            assert len(group.tiles) < total_params

    def test_get_group(self):
        from ironcore.offload.scheduler import ExecutionScheduler

        model, config = self._make_simple_model(num_layers=3)
        scheduler = ExecutionScheduler.from_model(
            model=model, config=config.offload, device=torch.device("cuda:0")
        )
        assert scheduler is not None
        for i in range(3):
            group = scheduler.get_group(i)
            assert group is not None
            assert group.layer_idx == i


# ---------------------------------------------------------------------------
# M2 Config Validation
# ---------------------------------------------------------------------------


class TestM2ConfigValidation:
    """Test M2-specific config validation rules."""

    def test_weight_offload_requires_enabled(self):
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.weight_offload = True
        with pytest.raises(ValueError, match="weight_offload requires offload.enabled"):
            _config_validation(config)

    def test_weight_offload_incompatible_with_fsdp(self):
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.enabled = True
        config.offload.weight_offload = True
        config.parallel.use_fsdp = True
        with pytest.raises(ValueError, match="incompatible with FSDP"):
            _config_validation(config)

    def test_invalid_prefetch_layers(self):
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.enabled = True
        config.offload.weight_offload = True
        config.offload.weight_prefetch_layers = 0
        with pytest.raises(ValueError, match="weight_prefetch_layers must be >= 1"):
            _config_validation(config)

    def test_valid_m2_config(self):
        from ironcore.config import _config_validation

        config = _make_minimal_main_config()
        config.offload.enabled = True
        config.offload.weight_offload = True
        config.offload.weight_prefetch_layers = 2
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
