# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""CPU storage semantics without requiring a CUDA pinned-memory allocator."""

from __future__ import annotations

import torch

from ironcore.offload.tile_manager import TileManager


class _CPUPool:
    def allocate(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(numel, dtype=dtype)


def test_frozen_cpu_weights_share_host_storage_without_a_second_copy():
    parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.bfloat16), requires_grad=False)
    expected = parameter.detach().clone()
    manager = TileManager(_CPUPool(), device=torch.device("cpu"), precision="bf16")
    group = manager.register_layer(0, [parameter])
    assert parameter.data_ptr() == group.tiles[0].host_tensor.data_ptr()
    torch.testing.assert_close(parameter, expected, atol=0, rtol=0)


def test_fp32_adapter_updates_survive_bf16_weight_storage():
    parameter = torch.nn.Parameter(torch.tensor([0.123456789, 0.987654321]))
    parameter.preserve_offload_precision = True
    manager = TileManager(_CPUPool(), device=torch.device("cpu"), precision="bf16")
    group = manager.register_layer(0, [parameter])
    assert group.tiles[0].host_tensor.dtype == torch.float32
    with torch.no_grad():
        parameter.add_(0.000001)
    manager.snapshot_params_to_host(group)
    torch.testing.assert_close(parameter, group.tiles[0].host_tensor, atol=0, rtol=0)


def test_adapter_loaded_after_registration_reaches_first_prefetch():
    parameter = torch.nn.Parameter(torch.zeros(4, 2))
    parameter.preserve_offload_precision = True
    manager = TileManager(_CPUPool(), device=torch.device("cpu"), precision="bf16")
    group = manager.register_layer(0, [parameter])
    assert parameter.data_ptr() == group.tiles[0].host_tensor.data_ptr()
    loaded = torch.arange(8).reshape(4, 2).float() / 7
    with torch.no_grad():
        parameter.copy_(loaded)
    # No snapshot is required between registration and the first prefetch.
    torch.testing.assert_close(group.tiles[0].host_tensor.view_as(loaded), loaded, atol=0, rtol=0)
    group.tiles[0].gpu_tensor = group.tiles[0].host_tensor.clone()
    manager.apply_tiles_to_params(group)
    torch.testing.assert_close(parameter, loaded, atol=0, rtol=0)
