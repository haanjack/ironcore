# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Async DMA transfer engine for weight streaming.

Manages H2D (host-to-device) and D2H (device-to-host) transfers on a dedicated
CUDA stream. This keeps DMA transfers off the default stream, allowing compute
and transfer to overlap.

Weight streaming during training. The engine handles async memcpy
with CUDA event synchronization to ensure correctness with torch.compile and
gradient checkpointing.

Usage pattern for weight streaming:
  1. During forward pass setup: submit H2D for next N layers' weights
  2. Before each layer executes: wait for that layer's transfer to complete
  3. After layer executes: optionally submit D2H to evict weights
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ironcore.offload.config import OffloadConfig


@dataclass
class TransferHandle:
    """Tracks an in-flight async transfer."""

    event: torch.cuda.Event
    direction: str  # "h2d" or "d2h"
    src_shape: tuple[int, ...]
    dst_shape: tuple[int, ...]
    nbytes: int
    completed: bool = False


class MemoryTransferEngine:
    """
    Async DMA transfer engine on a dedicated CUDA stream.

    All transfers are non-blocking on the caller's CPU thread. Completion
    is tracked via CUDA events. The caller must call wait() or synchronize()
    before using transferred data.

    Thread-safe: internal lock protects the transfer queue.

    Args:
        device: Target CUDA device (e.g. torch.device("cuda:0"))
        prefetch_streams: Number of CUDA streams for concurrent transfers.
            Default 1 is sufficient for PCIe-bound transfers. Multiple streams
            help when transferring from multiple source buffers simultaneously.
    """

    def __init__(self, device: torch.device, prefetch_streams: int = 1):
        if not torch.cuda.is_available():
            raise RuntimeError("MemoryTransferEngine requires CUDA")

        self._device = device
        self._streams = [torch.cuda.Stream(device=device) for _ in range(prefetch_streams)]
        self._stream_idx = 0
        self._lock = threading.Lock()
        self._pending: list[TransferHandle] = []
        self._total_h2d_bytes = 0
        self._total_d2h_bytes = 0

    @classmethod
    def from_config(cls, config: OffloadConfig, device: torch.device) -> MemoryTransferEngine:
        """Create an engine from OffloadConfig."""
        return cls(device=device, prefetch_streams=config.prefetch_streams)

    def submit_h2d(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        stream_idx: int | None = None,
    ) -> TransferHandle:
        """
        Submit an async host-to-device transfer.

        Args:
            src: Source tensor on CPU (must be pinned for true async)
            dst: Destination tensor on GPU
            stream_idx: Optional specific stream index. Auto-rotates if None.

        Returns:
            TransferHandle for tracking completion
        """
        assert src.shape == dst.shape, f"Shape mismatch: {src.shape} vs {dst.shape}"
        assert dst.device.type == "cuda", f"Destination must be CUDA, got {dst.device}"

        stream = self._get_stream(stream_idx)
        event = torch.cuda.Event(interprocess=False, enable_timing=False)

        # Transfer stream must wait for the default stream to finish
        # before writing to the GPU destination buffer. The buffer may have
        # been freed from the pool and recycled — without this barrier the
        # H2D write races with the default stream's backward computation
        # still reading from the same memory.
        stream.wait_stream(torch.cuda.current_stream(self._device))

        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
            event.record(stream)

        handle = TransferHandle(
            event=event,
            direction="h2d",
            src_shape=tuple(src.shape),
            dst_shape=tuple(dst.shape),
            nbytes=src.numel() * src.element_size(),
        )

        with self._lock:
            self._pending.append(handle)
            self._total_h2d_bytes += handle.nbytes

        return handle

    def submit_d2h(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        stream_idx: int | None = None,
    ) -> TransferHandle:
        """
        Submit an async device-to-host transfer.

        Args:
            src: Source tensor on GPU
            dst: Destination tensor on CPU (must be pinned for true async)
            stream_idx: Optional specific stream index. Auto-rotates if None.

        Returns:
            TransferHandle for tracking completion
        """
        assert src.shape == dst.shape, f"Shape mismatch: {src.shape} vs {dst.shape}"
        assert src.device.type == "cuda", f"Source must be CUDA, got {src.device}"

        stream = self._get_stream(stream_idx)
        event = torch.cuda.Event(interprocess=False, enable_timing=False)

        # Transfer stream must wait for the default stream to finish
        # producing the GPU data before we copy it to host.
        stream.wait_stream(torch.cuda.current_stream(self._device))

        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
            event.record(stream)

        handle = TransferHandle(
            event=event,
            direction="d2h",
            src_shape=tuple(src.shape),
            dst_shape=tuple(dst.shape),
            nbytes=src.numel() * src.element_size(),
        )

        with self._lock:
            self._pending.append(handle)
            self._total_d2h_bytes += handle.nbytes

        return handle

    def wait(self, handle: TransferHandle) -> None:
        """Block until a specific transfer completes."""
        if handle.completed:
            return
        handle.event.synchronize()
        handle.completed = True

    def synchronize(self) -> None:
        """Block until all pending transfers complete."""
        with self._lock:
            pending = list(self._pending)
        for handle in pending:
            self.wait(handle)

    def synchronize_with_default_stream(self) -> None:
        """
        Add a dependency: default stream waits for all transfer streams.

        Call this before using transferred data in a torch.compile region
        or any computation on the default stream.
        """
        for stream in self._streams:
            torch.cuda.current_stream(self._device).wait_stream(stream)

    def drain_completed(self) -> int:
        """
        Remove completed transfers from the pending list.

        Returns the number of transfers still pending.
        """
        with self._lock:
            still_pending = []
            for handle in self._pending:
                if handle.completed:
                    continue
                # Query without blocking
                if handle.event.query():
                    handle.completed = True
                else:
                    still_pending.append(handle)
            self._pending = still_pending
            return len(self._pending)

    @property
    def pending_count(self) -> int:
        """Number of in-flight transfers."""
        with self._lock:
            return len(self._pending)

    @property
    def total_h2d_bytes(self) -> int:
        return self._total_h2d_bytes

    @property
    def total_d2h_bytes(self) -> int:
        return self._total_d2h_bytes

    def _get_stream(self, idx: int | None = None) -> torch.cuda.Stream:
        if idx is not None:
            return self._streams[idx % len(self._streams)]
        stream = self._streams[self._stream_idx]
        self._stream_idx = (self._stream_idx + 1) % len(self._streams)
        return stream

    def __repr__(self) -> str:
        return (
            f"MemoryTransferEngine("
            f"device={self._device}, "
            f"streams={len(self._streams)}, "
            f"pending={self.pending_count}, "
            f"h2d={self._total_h2d_bytes / 1024**3:.1f}GB, "
            f"d2h={self._total_d2h_bytes / 1024**3:.1f}GB)"
        )
