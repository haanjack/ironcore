# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor parallel communication utilities with buffer pooling optimization."""

import torch
import torch.distributed as dist

from ironcore.parallel import parallel_states
from ironcore.profiler import timed_comm


class BufferPool:
    """Thread-safe buffer pool for tensor parallel communication.

    Caches tensor buffers by shape, dtype, and device to avoid repeated
    allocation overhead during all-gather operations.

    Usage:
        pool = BufferPool()
        slices = pool.get_buffers(shape, dtype, device, count)
        # Use slices for all_gather...
        # No need to return - next call reuses or creates as needed
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._buffers = {}  # (shape, dtype, device) -> list of tensors
            cls._instance._max_pool_size = 32  # Max buffers per (shape, dtype, device)
        return cls._instance

    def get_buffers(
        self,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        count: int,
    ) -> list[torch.Tensor]:
        """Get or create buffer tensors for all-gather.

        Args:
            shape: Shape of each buffer tensor
            dtype: Data type
            device: Device to create tensors on
            count: Number of buffers needed

        Returns:
            List of buffer tensors
        """
        key = (tuple(shape), dtype, device)

        if key not in self._buffers:
            self._buffers[key] = []

        pool = self._buffers[key]

        # Create new buffers if needed
        while len(pool) < count:
            if len(pool) >= self._max_pool_size:
                # Pool full, create temporary buffer
                return [torch.empty(shape, dtype=dtype, device=device) for _ in range(count)]
            pool.append(torch.empty(shape, dtype=dtype, device=device))

        return pool[:count]

    def clear(self):
        """Clear all cached buffers (call before model changes)."""
        self._buffers.clear()


def get_buffer_pool() -> BufferPool:
    """Get the global buffer pool instance."""
    return BufferPool()


def _reduce(
    x: torch.Tensor, async_op: bool = False
) -> torch.Tensor | tuple[torch.Tensor, dist.Work | None]:
    if parallel_states.get_tensor_model_parallel_world_size() == 1:
        if async_op:
            return x, None
        return x

    if not x.is_contiguous():
        x = x.contiguous()

    if not async_op:
        with timed_comm("tp_all_reduce"):
            dist.all_reduce(
                x, group=parallel_states.get_tensor_model_parallel_group(), async_op=False
            )
        return x

    handle = dist.all_reduce(
        x, group=parallel_states.get_tensor_model_parallel_group(), async_op=True
    )
    return x, handle


def _split_tensor_along_last_dim(x: torch.Tensor):
    world_size = parallel_states.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return x

    # split along last dimension
    assert x.shape[-1] % world_size == 0
    partition_dim = x.shape[-1] // world_size
    partitions = torch.split(x, partition_dim, dim=-1)

    rank = parallel_states.get_tensor_model_parallel_rank()
    output = partitions[rank].contiguous()

    return output


def _split_concated_tensor_along_last_dim(x: torch.Tensor, num_types: int) -> torch.Tensor:
    world_size = parallel_states.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return x

    # split input tensor along the types
    last_dim = x.shape[-1] // num_types
    splited_weights = torch.split(x, last_dim, dim=-1)

    outputs = []
    for splited_weight in splited_weights:
        partition_dim = last_dim // world_size
        partition = torch.split(splited_weight, partition_dim, dim=-1)

        outputs.append(partition[parallel_states.get_tensor_model_parallel_rank()])
    output = torch.cat(outputs, dim=-1)

    # # split along last dimension
    # assert x.shape[-1] % world_size == 0
    # x = x.view(-1, world_size, x.shape[-1] // world_size)

    # rank = parallel_states.get_tensor_model_parallel_rank()
    # output = x[:, rank].contiguous()

    return output


def _split_tensor_along_first_dim(x: torch.Tensor):
    world_size = parallel_states.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return x

    # split along first dimension
    assert x.shape[0] % world_size == 0
    partition_dim = x.shape[0] // world_size
    partitions = torch.split(x, partition_dim, dim=0)

    rank = parallel_states.get_tensor_model_parallel_rank()
    output = partitions[rank].contiguous()

    return output


def _gather_tensor_along_last_dim(x: torch.Tensor):
    world_size = parallel_states.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return x

    # Use buffer pool to avoid repeated allocation
    pool = get_buffer_pool()
    slices = pool.get_buffers(x.shape, x.dtype, x.device, world_size)

    with timed_comm("tp_all_gather"):
        dist.all_gather(slices, x, group=parallel_states.get_tensor_model_parallel_group())

    # Concatenate slices along the last dimension
    output = torch.cat(slices, dim=-1)

    return output


def _gather_concated_tensor_along_last_dim(x: torch.Tensor, num_types: int) -> torch.Tensor:
    world_size = parallel_states.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return x

    # Use buffer pool to avoid repeated allocation
    pool = get_buffer_pool()

    # split input tensors along with the last_dim size
    last_dim = x.shape[-1] // num_types
    weight_splits = torch.split(x, last_dim, dim=-1)

    outputs = []
    for weight_split in weight_splits:
        slices = pool.get_buffers(
            weight_split.shape, weight_split.dtype, weight_split.device, world_size
        )
        with timed_comm("tp_all_gather_concat"):
            dist.all_gather(
                slices,
                weight_split.contiguous(),
                group=parallel_states.get_tensor_model_parallel_group(),
            )

        # Concatenate slices along the last dimension
        outputs.append(torch.cat(slices, dim=-1))
    output = torch.cat(outputs, dim=-1)

    return output


def _gather_tensor_along_first_dim(x: torch.Tensor):
    world_size = parallel_states.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return x

    # Use buffer pool to avoid repeated allocation
    pool = get_buffer_pool()
    slices = pool.get_buffers(x.shape, x.dtype, x.device, world_size)

    with timed_comm("tp_all_gather_first_dim"):
        dist.all_gather(slices, x, group=parallel_states.get_tensor_model_parallel_group())

    # Concatenate slices along the first dimension
    output = torch.cat(slices, dim=0)

    return output


class _CopyToModelParallelWorkers(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=unused-argument
        return x

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, dist.Work | None]:
        return _reduce(grad_output)


class _ReduceFromModelParallelWorkers(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return _reduce(x, async_op=False)  # type: ignore

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class _ScatterToModelParallelWorkers(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=unused-argument
        return _split_tensor_along_last_dim(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return _gather_tensor_along_last_dim(grad_output)


class _GatherFromModelParallelWorkers(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, column_parallel: bool, row_parallel: bool, concatenated_weights: int
    ):
        ctx.column_parallel = column_parallel
        ctx.row_parallel = row_parallel
        ctx.concatenated_weights = concatenated_weights

        if column_parallel:
            if concatenated_weights > 1:
                return _gather_concated_tensor_along_last_dim(x, concatenated_weights)
            else:
                return _gather_tensor_along_last_dim(x)
        elif row_parallel:
            return _gather_tensor_along_first_dim(x)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        column_parallel = ctx.column_parallel
        row_parallel = ctx.row_parallel
        concatenated_weights = ctx.concatenated_weights

        if column_parallel:
            if concatenated_weights > 1:
                return (
                    _split_concated_tensor_along_last_dim(grad_output, concatenated_weights),
                    None,
                    None,
                    None,
                )
            else:
                return _split_tensor_along_last_dim(grad_output), None, None, None
        elif row_parallel:
            return _split_tensor_along_first_dim(grad_output), None, None, None
        return grad_output, None, None, None


def copy_inputs_to_model_parallel_workers(x) -> torch.Tensor:
    return _CopyToModelParallelWorkers.apply(x)  # type: ignore


def reduce_inputs_from_model_parallel_workers(x) -> torch.Tensor:
    return _ReduceFromModelParallelWorkers.apply(x)  # type: ignore


def reduce_async(x: torch.Tensor) -> tuple[torch.Tensor, dist.Work | None]:
    """
    Asynchronous reduction. This is NOT tracked by autograd in the forward pass.
    Used for Async TP to overlap chunk reductions with next chunk computation.
    """
    return _reduce(x, async_op=True)  # type: ignore


def scatter_input_to_model_parallel_workers(x) -> torch.Tensor:
    return _ScatterToModelParallelWorkers.apply(x)  # type: ignore


def gather_from_model_parallel_workers(x, attrib) -> torch.Tensor:
    return _GatherFromModelParallelWorkers.apply(
        x,
        attrib.get("column_parallel", False),
        attrib.get("row_parallel", False),
        attrib.get("concatenated_weights", 1),
    )  # type: ignore


def split_to_model_parallel_workers(x, attrib):
    if attrib["column_parallel"]:
        if attrib["concatenated_weights"] > 1:
            return _split_concated_tensor_along_last_dim(x, attrib["concatenated_weights"])
        else:
            return _split_tensor_along_last_dim(x)
    elif attrib["row_parallel"]:
        return _split_tensor_along_first_dim(x)
