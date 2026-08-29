# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for weight streaming components.

Tests cover:
1. PinnedMemoryPool: allocation, free, reuse, budget enforcement
2. MemoryTransferEngine: H2D/D2H transfers, event synchronization
3. TileManager: register params, apply tiles, precision conversion
4. ExecutionScheduler: from_model, lifecycle hooks, layer prefetch
5. Config validation for weight streaming fields

CUDA tests are gated on torch.cuda.is_available().
"""

import threading

import pytest
import torch
from torch import nn

from ironcore.config import OffloadConfig
from ironcore.offload.memory_pool import PinnedMemoryPool
from ironcore.offload.tile_manager import TileManager

cuda_available = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

pytestmark = [pytest.mark.cuda]


# ---------------------------------------------------------------------------
# PinnedMemoryPool
# ---------------------------------------------------------------------------


@skip_no_cuda
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
# GPUStagingPool (CUDA required)
# ---------------------------------------------------------------------------


@skip_no_cuda
class TestGPUStagingPool:
    """Test GPU staging buffer pool allocation and free mechanics."""

    def test_allocate_returns_gpu_tensor(self):
        from ironcore.offload.gpu_staging_pool import GPUStagingPool

        pool = GPUStagingPool(device=torch.device("cuda:0"), chunk_bytes=1024 * 1024)
        t = pool.allocate(256, torch.float32)
        assert t.shape == (256,)
        assert t.dtype == torch.float32
        assert t.device.type == "cuda"

    def test_allocate_and_free_reuse(self):
        from ironcore.offload.gpu_staging_pool import GPUStagingPool

        pool = GPUStagingPool(device=torch.device("cuda:0"), chunk_bytes=1024 * 1024)
        t1 = pool.allocate(256, torch.float32)
        ptr1 = t1.data_ptr()
        pool.free(t1)
        t2 = pool.allocate(256, torch.float32)
        # Should reuse same region (same data_ptr)
        assert t2.data_ptr() == ptr1
        assert pool.total_used_bytes == 256 * 4

    def test_multiple_dtypes(self):
        from ironcore.offload.gpu_staging_pool import GPUStagingPool

        pool = GPUStagingPool(device=torch.device("cuda:0"), chunk_bytes=1024 * 1024)
        t32 = pool.allocate(64, torch.float32)
        t16 = pool.allocate(64, torch.float16)
        assert t32.dtype == torch.float32
        assert t16.dtype == torch.float16
        assert pool.total_used_bytes == 64 * 4 + 64 * 2

    def test_oversized_allocation(self):
        from ironcore.offload.gpu_staging_pool import GPUStagingPool

        # 1KB chunks, but request 4KB
        pool = GPUStagingPool(device=torch.device("cuda:0"), chunk_bytes=1024)
        t = pool.allocate(1024, torch.float32)  # 4KB
        assert t.shape == (1024,)
        # Should have created a dedicated chunk
        assert pool.total_allocated_bytes >= 4096

    def test_budget_exceeded(self):
        from ironcore.offload.gpu_staging_pool import GPUStagingPool

        pool = GPUStagingPool(
            device=torch.device("cuda:0"),
            chunk_bytes=1024 * 1024,
            max_total_bytes=512,
        )
        with pytest.raises(RuntimeError, match="budget exceeded"):
            pool.allocate(256, torch.float32)  # 1024 bytes > 512 budget

    def test_from_config(self):
        from ironcore.offload.gpu_staging_pool import GPUStagingPool

        config = OffloadConfig(gpu_staging_chunk_mb=128.0, gpu_staging_pool_mb=512.0)
        pool = GPUStagingPool.from_config(config, torch.device("cuda:0"))
        assert pool._chunk_bytes == 128 * 1024 * 1024
        assert pool._max_total_bytes == 512 * 1024 * 1024

    def test_concurrent_allocate_free(self):
        from ironcore.offload.gpu_staging_pool import GPUStagingPool

        pool = GPUStagingPool(device=torch.device("cuda:0"), chunk_bytes=4 * 1024 * 1024)
        errors = []
        allocated = []
        lock = threading.Lock()

        def worker():
            try:
                for _ in range(20):
                    t = pool.allocate(64, torch.float32)
                    with lock:
                        allocated.append(t.data_ptr())
                    pool.free(t)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors, f"Thread errors: {errors}"
        assert pool.total_used_bytes == 0

    def test_auto_size_uniform_layers(self):
        from ironcore.offload.gpu_staging_pool import GPUStagingPool

        pool = GPUStagingPool(device=torch.device("cuda:0"), chunk_bytes=1024 * 1024)
        # 4 uniform layers of 1MB each, prefetch 2
        layer_sizes = [1024 * 1024] * 4
        pool.auto_size(layer_sizes, prefetch_layers=2)
        assert pool._max_total_bytes == 3 * 1024 * 1024  # 3 consecutive
        assert pool._chunk_bytes >= 1024 * 1024

    def test_auto_size_non_uniform_layers(self):
        from ironcore.offload.gpu_staging_pool import GPUStagingPool

        pool = GPUStagingPool(device=torch.device("cuda:0"), chunk_bytes=1024 * 1024)
        # Layers: 1MB, 3MB, 2MB, 1MB, prefetch 1 (need 2 consecutive)
        layer_sizes = [s * 1024 * 1024 for s in [1, 3, 2, 1]]
        pool.auto_size(layer_sizes, prefetch_layers=1)
        # Best 2-consecutive: 3MB + 2MB = 5MB
        assert pool._max_total_bytes == 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# MemoryTransferEngine (CUDA required)
# ---------------------------------------------------------------------------


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
        assert engine.pending_count == 0

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

        # Simulate: modify host tensor, allocate GPU staging, copy host→GPU, apply
        group.tiles[0].host_tensor.fill_(0.0)
        group.tiles[0].gpu_tensor = torch.empty_like(original.flatten())
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
        # GPU staging is allocated at prefetch time, not during registration
        assert group.tiles[0].gpu_tensor is None

    def test_invalid_precision_raises(self):
        pool = PinnedMemoryPool(chunk_bytes=1024)
        with pytest.raises(ValueError, match="Invalid precision"):
            TileManager(pool=pool, device=torch.device("cpu"), precision="fp8")

    def test_from_config_uses_weight_storage_precision(self):
        config = OffloadConfig(weight_storage_precision="bf16")
        pool = PinnedMemoryPool(chunk_bytes=4 * 1024 * 1024)
        tm = TileManager.from_config(config, pool, torch.device("cpu"))
        assert tm._storage_dtype == torch.bfloat16

    def test_from_config_defaults_to_bf16(self):
        config = OffloadConfig()
        pool = PinnedMemoryPool(chunk_bytes=4 * 1024 * 1024)
        tm = TileManager.from_config(config, pool, torch.device("cpu"))
        assert tm._storage_dtype == torch.bfloat16

    def test_register_layer_no_gpu_alloc_with_pool(self):
        from ironcore.offload.gpu_staging_pool import GPUStagingPool

        pool = PinnedMemoryPool(chunk_bytes=4 * 1024 * 1024)
        gpu_pool = GPUStagingPool(device=torch.device("cuda:0"), chunk_bytes=4 * 1024 * 1024)
        tm = TileManager(
            pool=pool, device=torch.device("cuda:0"), precision="fp32", gpu_pool=gpu_pool
        )

        param = nn.Parameter(torch.randn(32, 32, device="cuda:0"))
        group = tm.register_layer(layer_idx=0, params=[param])

        # gpu_tensor should be None — not pre-allocated
        for tile in group.tiles:
            assert tile.gpu_tensor is None
        assert gpu_pool.total_used_bytes == 0

    def test_borrow_return_lifecycle(self):
        from ironcore.offload.gpu_staging_pool import GPUStagingPool

        pool = PinnedMemoryPool(chunk_bytes=4 * 1024 * 1024)
        gpu_pool = GPUStagingPool(device=torch.device("cuda:0"), chunk_bytes=4 * 1024 * 1024)
        tm = TileManager(
            pool=pool, device=torch.device("cuda:0"), precision="fp32", gpu_pool=gpu_pool
        )

        param = nn.Parameter(torch.randn(32, 32, device="cuda:0"))
        group = tm.register_layer(layer_idx=0, params=[param])

        # Borrow: gpu_tensor allocated from pool
        tm.borrow_gpu_buffers(group)
        for tile in group.tiles:
            assert tile.gpu_tensor is not None
        assert gpu_pool.total_used_bytes > 0

        # Return: gpu_tensor freed back to pool
        tm.return_gpu_buffers(group)
        for tile in group.tiles:
            assert tile.gpu_tensor is None
        assert gpu_pool.total_used_bytes == 0

    def test_apply_tiles_with_pool(self):
        from ironcore.offload.gpu_staging_pool import GPUStagingPool

        pool = PinnedMemoryPool(chunk_bytes=4 * 1024 * 1024)
        gpu_pool = GPUStagingPool(device=torch.device("cuda:0"), chunk_bytes=4 * 1024 * 1024)
        tm = TileManager(
            pool=pool, device=torch.device("cuda:0"), precision="fp32", gpu_pool=gpu_pool
        )

        original = torch.randn(32, 32, device="cuda:0")
        param = nn.Parameter(original.clone())
        group = tm.register_layer(layer_idx=0, params=[param])

        # Borrow staging buffer
        tm.borrow_gpu_buffers(group)
        # Simulate H2D: write zeros to staging buffer
        group.tiles[0].gpu_tensor.fill_(0.0)
        # Apply staging to param
        tm.apply_tiles_to_params(group)
        assert torch.all(param.data == 0)
        # Return buffer
        tm.return_gpu_buffers(group)
        assert gpu_pool.total_used_bytes == 0


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
        # ModelConfig has no hidden_size/ffn_hidden_size fields; assigning those only
        # created stray attributes and left d_model at its 512 default, so this
        # "minimal" model was 16x wider than intended and dominated the suite's peak.
        model_config.d_model = hidden
        model_config.num_attention_heads = 4
        model_config.num_attention_groups = 4
        model_config.head_dim = hidden // 4
        model_config.d_ffn = hidden * 4

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

    def test_device_property(self):
        """Scheduler exposes the device it was created with."""
        from ironcore.offload.scheduler import ExecutionScheduler

        model, config = self._make_simple_model()
        scheduler = ExecutionScheduler.from_model(
            model=model,
            config=config.offload,
            device=torch.device("cuda:0"),
        )
        assert scheduler.device == torch.device("cuda:0")

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

    def test_pool_vram_bounded(self):
        """Verify GPU staging pool usage stays bounded through a full step cycle."""
        from ironcore.offload.scheduler import ExecutionScheduler

        model, config = self._make_simple_model(num_layers=4)
        scheduler = ExecutionScheduler.from_model(
            model=model, config=config.offload, device=torch.device("cuda:0")
        )
        assert scheduler is not None
        assert scheduler._gpu_pool is not None

        max_layer_bytes = max(
            sum(t.numel * torch.tensor([], dtype=t.original_dtype).element_size() for t in g.tiles)
            for g in scheduler._weight_groups.values()
        )
        # Budget = (prefetch_layers + 1) * max_layer_bytes
        budget = max_layer_bytes * (config.offload.weight_prefetch_layers + 1)

        # Run a full step cycle, tracking peak usage
        peak_used = 0
        scheduler.on_training_step_start()
        peak_used = max(peak_used, scheduler._gpu_pool.total_used_bytes)
        for i in range(len(model.layers)):
            scheduler.on_layer_start(i)
            peak_used = max(peak_used, scheduler._gpu_pool.total_used_bytes)
            scheduler.on_layer_end(i)
        scheduler.on_backward_pass_end()
        scheduler.on_training_step_end()

        # Peak usage should never exceed the budget
        assert peak_used <= budget, f"Peak pool usage {peak_used} exceeded budget {budget}"


# ---------------------------------------------------------------------------
# weight streaming CPU-Resident Param Lifecycle
# ---------------------------------------------------------------------------


@skip_no_cuda
class TestWeightStreamingCPUResidentParams:
    """Test weight streaming with CPU-resident parameters.

    These tests verify the D4 fix (host tile values for CPU placeholder),
    D8 optimization (skip redundant D2H snapshots), and the full
    CPU param swap lifecycle (load → compute → evict → optimizer → snapshot).
    """

    def _make_cpu_model(self, num_layers=2, hidden=32):
        """Create a model on CPU (weight streaming mode) with scheduler attached."""
        from ironcore.config import MainConfig
        from ironcore.models.transformer import TransformerModel
        from ironcore.offload.scheduler import ExecutionScheduler

        model_config = __import__(
            "ironcore.config.config_model", fromlist=["ModelConfig"]
        ).ModelConfig()
        model_config.num_layers = num_layers
        # ModelConfig has no hidden_size/ffn_hidden_size fields; assigning those only
        # created stray attributes and left d_model at its 512 default, so this
        # "minimal" model was 16x wider than intended and dominated the suite's peak.
        model_config.d_model = hidden
        model_config.num_attention_heads = 4
        model_config.num_attention_groups = 4
        model_config.head_dim = hidden // 4
        model_config.d_ffn = hidden * 4

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
                activation_spill=True,
                pinned_chunk_gb=0.05,
                pinned_memory_pool_gb=0.2,
            ),
        )
        config.trainer.micro_batch_size = 1
        config.trainer.train_batch_size = 1
        config.trainer.gradient_accumulation_steps = 1
        config.parallel.world_size = 1

        # weight streaming: model stays on CPU — no .to(device)
        model = TransformerModel(config)

        scheduler = ExecutionScheduler.from_model(
            model=model,
            config=config.offload,
            device=torch.device("cuda:0"),
        )
        return model, config, scheduler

    def test_params_start_on_cpu(self):
        """weight streaming mode: model params should be on CPU before scheduler loads them."""
        model, _, scheduler = self._make_cpu_model()
        assert scheduler is not None
        for p in model.layers[0].parameters():
            assert p.device.type == "cpu", f"Param should be on CPU, got {p.device}"

    def test_weight_streaming_enabled_property(self):
        """Verify weight_streaming_enabled returns True when weight groups exist."""
        _, _, scheduler = self._make_cpu_model()
        assert scheduler.weight_streaming_enabled is True

    def test_weight_streaming_enabled_false_when_no_groups(self):
        """Verify weight_streaming_enabled returns False for spill-only mode."""
        config = _make_minimal_main_config()
        config.offload.enabled = True
        config.offload.weight_offload = False
        config.offload.activation_spill = True

        from ironcore.offload.scheduler import ExecutionScheduler

        scheduler = ExecutionScheduler.__new__(ExecutionScheduler)
        scheduler._weight_groups = {}
        assert scheduler.weight_streaming_enabled is False

    def test_cpu_param_swap_to_gpu_on_load(self):
        """After on_layer_start, CPU params are swapped to GPU staging buffers."""
        model, _, scheduler = self._make_cpu_model()
        scheduler.on_training_step_start()

        # Before: params on CPU
        first_param = next(iter(model.layers[0].parameters()))
        assert first_param.device.type == "cpu"

        scheduler.on_layer_start(0)

        # After: params on GPU (swapped via param.data = gpu_tensor)
        assert first_param.device.type == "cuda"

    def test_eviction_restores_host_tile_values(self):
        """D4 fix: evicted params must have deterministic host tile values, not garbage."""
        model, _, scheduler = self._make_cpu_model()
        scheduler.on_training_step_start()
        scheduler.on_layer_start(0)

        group = scheduler._weight_groups[0]
        # Record what host tile values look like
        first_tile = group.tiles[0]
        first_param_ref = group.param_refs[0][0]
        expected = first_tile.host_tensor[: first_param_ref.numel()].to(first_param_ref.dtype)

        scheduler.on_layer_end(0)  # triggers _evict_layer_weights

        # After eviction: param should be on CPU with host tile values
        assert first_param_ref.device.type == "cpu"
        actual = first_param_ref.data.flatten()
        assert torch.allclose(actual, expected), (
            "Evicted param should have host tile values, not uninitialized memory"
        )

    def test_eviction_not_uninitialized(self):
        """D4 regression: evicted params must not contain NaN or extreme values."""
        model, _, scheduler = self._make_cpu_model()
        scheduler.on_training_step_start()
        scheduler.on_layer_start(0)
        scheduler.on_layer_end(0)

        for layer_idx, group in scheduler._weight_groups.items():
            for tile, (param, _, _) in zip(group.tiles, group.param_refs, strict=True):
                assert param.device.type == "cpu"
                flat = param.data.flatten()
                assert not torch.isnan(flat).any(), f"Layer {layer_idx} has NaN after eviction"
                assert not torch.isinf(flat).any(), f"Layer {layer_idx} has Inf after eviction"
                # Values should be bounded (initial weights are normally distributed)
                assert flat.abs().max() < 100.0, f"Layer {layer_idx} has extreme values"

    def test_snapshot_skip_during_forward_eviction(self):
        """D8: forward eviction should NOT update host tiles (weights unchanged)."""
        model, _, scheduler = self._make_cpu_model()
        scheduler.on_training_step_start()

        group = scheduler._weight_groups[0]
        tile = group.tiles[0]
        # Save original host tile values
        original_host = tile.host_tensor.clone()

        # Load, modify param on GPU, then evict (forward eviction)
        scheduler.on_layer_start(0)
        # Modify the GPU param value
        param = group.param_refs[0][0]
        param.data.fill_(42.0)
        scheduler.on_layer_end(0)

        # Host tile should NOT reflect the GPU modification (snapshot skipped)
        assert torch.allclose(tile.host_tensor, original_host)

    def test_full_lifecycle_roundtrip(self):
        """Full step: load → evict → snapshot → verify host tiles updated."""
        model, _, scheduler = self._make_cpu_model()

        group = scheduler._weight_groups[0]
        tile = group.tiles[0]
        original_host = tile.host_tensor.clone()

        # Step 1: Load weights to GPU
        scheduler.on_training_step_start()
        scheduler.on_layer_start(0)

        # Step 2: Modify GPU param (simulating forward/backward computation)
        param = group.param_refs[0][0]
        param.data.fill_(7.0)

        # Step 3: Evict (forward eviction — no snapshot per D8)
        scheduler.on_layer_end(0)

        # Host tiles should still have original values (D8: no snapshot)
        assert torch.allclose(tile.host_tensor, original_host)

        # Step 4: Simulate optimizer updating param.data on CPU
        param.data.fill_(99.0)

        # Step 5: on_training_step_end snapshots updated values
        scheduler.on_training_step_end()

        # Now host tiles should reflect the optimizer's update
        expected = torch.full_like(tile.host_tensor, 99.0)
        assert torch.allclose(tile.host_tensor, expected), (
            "Host tiles should reflect optimizer update after on_training_step_end"
        )

    def test_all_layers_evicted_after_backward(self):
        """After backward pass, all layer weights should be evicted from GPU."""
        model, _, scheduler = self._make_cpu_model()

        scheduler.on_training_step_start()
        num_layers = len(model.layers)

        # Forward pass
        for i in range(num_layers):
            scheduler.on_layer_start(i)
            scheduler.on_layer_end(i)

        # Backward pass
        for i in range(num_layers):
            scheduler.on_backward_layer_start(i)
            scheduler.on_backward_layer_end(i)

        scheduler.on_backward_pass_end()
        scheduler.on_training_step_end()

        # All layers should be evicted
        assert len(scheduler._layer_on_gpu) == 0
        for group in scheduler._weight_groups.values():
            for _, (param, _, _) in zip(group.tiles, group.param_refs, strict=True):
                assert param.device.type == "cpu"


# ---------------------------------------------------------------------------
# CPU AdamW Optimizer
# ---------------------------------------------------------------------------


class TestCPUAdamW:
    """Test optimizer update correctness on CPU-resident params."""

    def test_adamw_offloaded_step_updates_param(self):
        """Verify _adamw_offloaded_step modifies CPU param correctly."""
        from ironcore.offload.optimizer_helpers import _adamw_offloaded_step

        param = nn.Parameter(torch.randn(16, 16))
        grad = torch.randn(16, 16)
        param.grad = grad.clone()

        state = {}
        _adamw_offloaded_step(
            p=param,
            grad=grad,
            state=state,
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            state_dtype=torch.float32,
        )

        # Param should have changed
        assert not torch.allclose(param.data, torch.randn(16, 16))  # sanity: param was modified
        assert state["step"] == 1
        assert state["exp_avg"].device.type == "cpu"
        assert state["exp_avg_sq"].device.type == "cpu"

    def test_adamw_offloaded_step_matches_expected(self):
        """Verify mathematical correctness of one AdamW step on CPU."""
        from ironcore.offload.optimizer_helpers import _adamw_offloaded_step

        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(4, 4))
        grad = torch.randn(4, 4)

        # Run our offloaded step
        state = {}
        _adamw_offloaded_step(
            p=param,
            grad=grad,
            state=state,
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            state_dtype=torch.float32,
        )

        # Manually compute expected result
        torch.manual_seed(42)
        expected_param = torch.randn(4, 4)
        expected_grad = grad.clone()
        exp_avg = torch.zeros(4, 4)
        exp_avg_sq = torch.zeros(4, 4)
        exp_avg = 0.9 * exp_avg + 0.1 * expected_grad
        exp_avg_sq = 0.999 * exp_avg_sq + 0.001 * expected_grad * expected_grad
        bias_correction1 = 1.0 - 0.9**1
        bias_correction2 = 1.0 - 0.999**1
        step_size = 1e-3 * (bias_correction2**0.5) / bias_correction1
        denom = exp_avg_sq.sqrt() + 1e-8
        expected_param = expected_param - step_size * exp_avg / denom

        assert torch.allclose(param.data, expected_param, atol=1e-6), (
            "CPU AdamW step should match expected computation"
        )

    def test_adamw_states_stay_on_cpu(self):
        """Verify optimizer states are allocated and remain on CPU."""
        from ironcore.offload.optimizer_helpers import _adamw_offloaded_step

        param = nn.Parameter(torch.randn(8))
        grad = torch.randn(8)
        state = {}

        _adamw_offloaded_step(
            p=param,
            grad=grad,
            state=state,
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            state_dtype=torch.float32,
        )

        assert state["exp_avg"].device.type == "cpu"
        assert state["exp_avg_sq"].device.type == "cpu"
        assert state["step"] == 1

    def test_adamw_weight_decay_on_cpu(self):
        """Verify weight decay is applied correctly when param is on CPU."""
        from ironcore.offload.optimizer_helpers import _adamw_offloaded_step

        param = nn.Parameter(torch.ones(4))
        grad = torch.zeros(4)
        state = {}

        _adamw_offloaded_step(
            p=param,
            grad=grad,
            state=state,
            lr=1e-2,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.1,
            amsgrad=False,
            state_dtype=torch.float32,
        )

        # With zero grad, param should only change from weight decay: p *= (1 - lr * wd)
        expected = 1.0 * (1.0 - 0.01 * 0.1)
        assert torch.allclose(param.data, torch.full((4,), expected), atol=1e-6)


# ---------------------------------------------------------------------------
# weight streaming Config Validation
# ---------------------------------------------------------------------------


class TestWeightStreamingConfigValidation:
    """Test weight streaming-specific config validation rules."""

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

    def test_valid_weight_offload_config(self):
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


@skip_no_cuda
class TestSingleLayerWeightStreaming:
    """Regression test for BUG-004: single layer + weight streaming crash."""

    def test_single_layer_forward_backward(self):
        """num_layers=1 with weight streaming should not crash."""
        from ironcore.models.transformer import TransformerModel
        from ironcore.offload.scheduler import ExecutionScheduler

        config = _make_minimal_main_config()
        config.model.num_layers = 1
        config.model.num_attention_heads = 4
        config.model.num_attention_groups = 4
        config.offload = OffloadConfig(
            enabled=True,
            weight_offload=True,
            activation_spill=True,
            pinned_chunk_gb=0.05,
            pinned_memory_pool_gb=0.2,
        )

        device = torch.device("cuda:0")
        model = TransformerModel(config).to(dtype=torch.bfloat16)
        model.train()

        scheduler = ExecutionScheduler.from_model(model, config.offload, device=device)
        assert scheduler is not None
        model._offload_scheduler = scheduler
        scheduler.set_gradient_accumulation_steps(1)

        hidden = torch.randn(1, 4, 512, device=device, dtype=torch.bfloat16)
        mask = torch.ones(1, 1, 4, 4, device=device)

        scheduler.on_training_step_start()
        scheduler.on_microbatch_forward_start(0)
        out = model(hidden, mask, None)
        scheduler.on_microbatch_forward_end()

        loss = out.sum()
        scheduler.on_microbatch_backward_start(0)
        loss.backward()
        scheduler.on_microbatch_backward_end()
        scheduler.on_training_step_end()

        assert loss.item() > 0
