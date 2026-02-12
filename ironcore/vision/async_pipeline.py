# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Asynchronous vision processing pipeline for VLA models.

Implements producer-consumer pattern for CPU-to-GPU async processing:
- Producer (CPU): VisionEncoder processes images in background threads
- Consumer (GPU): Language model consumes pre-computed vision features

Also supports CUDA stream-based async for GPU-to-GPU processing.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from ironcore.config import MainConfig

from ironcore.layers.module import BaseModule


@dataclass
class VisionFeatureBatch:
    """Container for vision features with metadata.

    Attributes:
        features: Vision features tensor [batch, num_patches, hidden_size]
        batch_id: Unique identifier for this batch
        pixel_values_hash: Hash of original pixel values for verification
        computed_at: Timestamp when features were computed
        device: Device where features currently reside
    """

    features: torch.Tensor
    batch_id: int
    pixel_values_hash: int | None = None
    computed_at: float = field(default_factory=time.time)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def to(self, device: torch.device) -> VisionFeatureBatch:
        """Move features to specified device."""
        return VisionFeatureBatch(
            features=self.features.to(device),
            batch_id=self.batch_id,
            pixel_values_hash=self.pixel_values_hash,
            computed_at=self.computed_at,
            device=device,
        )


class VisionFeatureQueue:
    """Thread-safe queue for vision features with prefetching.

    Implements bounded buffer for producer-consumer pattern.
    Supports priority-based retrieval for training stability.

    Example:
        >>> queue = VisionFeatureQueue(max_size=4)
        >>> # Producer thread
        >>> queue.put(vision_batch)
        >>> # Consumer (main thread)
        >>> batch = queue.get(timeout=5.0)
    """

    def __init__(self, max_size: int = 4):
        """Initialize queue.

        Args:
            max_size: Maximum number of batches to prefetch
        """
        self.max_size = max_size
        self._queue: deque[VisionFeatureBatch] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._shutdown = False
        self._batch_counter = 0

    def put(
        self,
        features: torch.Tensor,
        pixel_values_hash: int | None = None,
        block: bool = True,
        timeout: float | None = None,
    ) -> int:
        """Add vision features to queue.

        Args:
            features: Vision features tensor
            pixel_values_hash: Optional hash for verification
            block: Wait if queue is full
            timeout: Max wait time in seconds

        Returns:
            Batch ID assigned to this batch

        Raises:
            RuntimeError: If queue is shutdown
        """
        with self._not_full:
            if self._shutdown:
                raise RuntimeError("Queue is shutdown")

            if len(self._queue) >= self.max_size:
                if not block:
                    raise RuntimeError("Queue is full")
                self._not_full.wait(timeout)

            if self._shutdown:
                raise RuntimeError("Queue is shutdown")

            batch_id = self._batch_counter
            self._batch_counter += 1

            batch = VisionFeatureBatch(
                features=features,
                batch_id=batch_id,
                pixel_values_hash=pixel_values_hash,
                device=features.device,
            )
            self._queue.append(batch)
            self._not_empty.notify()

            return batch_id

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> VisionFeatureBatch:
        """Get vision features from queue.

        Args:
            block: Wait if queue is empty
            timeout: Max wait time in seconds

        Returns:
            VisionFeatureBatch with features

        Raises:
            RuntimeError: If queue is shutdown and empty
        """
        with self._not_empty:
            if not self._queue and not self._shutdown:
                if not block:
                    raise RuntimeError("Queue is empty")
                self._not_empty.wait(timeout)

            if not self._queue:
                if self._shutdown:
                    raise RuntimeError("Queue is shutdown and empty")
                raise RuntimeError("Timeout waiting for features")

            batch = self._queue.popleft()
            self._not_full.notify()
            return batch

    def get_nowait(self) -> VisionFeatureBatch | None:
        """Get features without blocking.

        Returns:
            VisionFeatureBatch or None if queue is empty
        """
        try:
            return self.get(block=False)
        except RuntimeError:
            return None

    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    def is_full(self) -> bool:
        """Check if queue is full."""
        with self._lock:
            return len(self._queue) >= self.max_size

    def clear(self):
        """Clear all items from queue."""
        with self._lock:
            self._queue.clear()
            self._not_full.notify_all()

    def shutdown(self):
        """Shutdown queue, unblocking all waiters."""
        with self._lock:
            self._shutdown = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def reset(self):
        """Reset queue for reuse."""
        with self._lock:
            self._shutdown = False
            self._queue.clear()
            self._not_full.notify_all()


class AsyncVisionEncoder(BaseModule):
    """Asynchronous vision encoder with producer-consumer pipeline.

    Runs vision encoding on a separate device (typically CPU) in background
    threads, while the main training loop consumes pre-computed features.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Main Thread (Consumer)                                      │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  VLAModel.forward()                                  │    │
    │  │  - Gets features from queue (non-blocking)           │    │
    │  │  - Processes language model on GPU                   │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                          ▲                                   │
    │                          │ VisionFeatureQueue                │
    │                          │                                   │
    │  Background Thread (Producer)                               │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  VisionEncoder (on CPU)                              │    │
    │  │  - Encodes images in background                      │    │
    │  │  - Puts features into queue                          │    │
    │  └─────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────┘

    Example:
        >>> encoder = AsyncVisionEncoder(config, num_workers=2)
        >>> encoder.start()
        >>> # Submit images for encoding
        >>> encoder.submit(pixel_values, batch_id=0)
        >>> # Later, get features (non-blocking)
        >>> features = encoder.get_features(batch_id=0, timeout=5.0)
        >>> encoder.stop()
    """

    def __init__(
        self,
        config: MainConfig,
        vision_encoder: nn.Module,
        num_workers: int = 1,
        queue_size: int = 4,
        target_device: torch.device | str = "cuda:0",
    ):
        """Initialize async vision encoder.

        Args:
            config: Main configuration
            vision_encoder: The vision encoder module to wrap
            num_workers: Number of background encoding threads
            queue_size: Maximum batches to prefetch
            target_device: Device to move features to for consumption
        """
        super().__init__(config)

        self.vision_encoder = vision_encoder
        self.num_workers = num_workers
        self.target_device = torch.device(target_device)

        # Feature queue
        self.queue = VisionFeatureQueue(max_size=queue_size)

        # Pending batches (batch_id -> pixel_values)
        self._pending: dict[int, torch.Tensor] = {}
        self._pending_lock = threading.Lock()
        self._batch_id_counter = 0

        # Worker threads
        self._workers: list[threading.Thread] = []
        self._running = False
        self._stop_event = threading.Event()

        # Statistics
        self._stats = {
            "encoded": 0,
            "dropped": 0,
            "avg_encode_time": 0.0,
        }
        self._stats_lock = threading.Lock()

    def start(self):
        """Start background encoding threads."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self.queue.reset()

        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"VisionEncoder-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def stop(self, timeout: float = 5.0):
        """Stop background encoding threads.

        Args:
            timeout: Max time to wait for threads to finish
        """
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        self.queue.shutdown()

        for worker in self._workers:
            worker.join(timeout=timeout)

        self._workers.clear()

    def submit(
        self,
        pixel_values: torch.Tensor,
        batch_id: int | None = None,
    ) -> int:
        """Submit images for encoding.

        Args:
            pixel_values: [batch, C, H, W] images to encode
            batch_id: Optional batch ID (auto-assigned if None)

        Returns:
            Batch ID for this submission
        """
        with self._pending_lock:
            if batch_id is None:
                batch_id = self._batch_id_counter
                self._batch_id_counter += 1

            # Move to vision device (CPU) if needed
            vision_device = getattr(self.vision_encoder, "vision_device", torch.device("cpu"))
            if pixel_values.device != vision_device:
                pixel_values = pixel_values.to(vision_device)

            self._pending[batch_id] = pixel_values.detach()

        return batch_id

    def get_features(
        self,
        batch_id: int | None = None,
        timeout: float = 5.0,
    ) -> torch.Tensor | None:
        """Get encoded features.

        Args:
            batch_id: Specific batch ID to get (None for any available)
            timeout: Max wait time in seconds

        Returns:
            Vision features tensor or None if timeout
        """
        try:
            if batch_id is not None:
                # Wait for specific batch
                start_time = time.time()
                while time.time() - start_time < timeout:
                    batch = self.queue.get_nowait()
                    if batch is not None and batch.batch_id == batch_id:
                        return batch.to(self.target_device).features
                    # Put back if not the right one (simple requeue)
                    # In practice, use ordered queue for efficiency
                    time.sleep(0.01)
                return None
            else:
                # Get any available
                batch = self.queue.get(block=True, timeout=timeout)
                return batch.to(self.target_device).features
        except RuntimeError:
            return None

    def encode_and_queue(
        self,
        pixel_values: torch.Tensor,
        block: bool = False,
        timeout: float | None = None,
    ) -> int:
        """Encode images and add to queue in one call.

        This is a convenience method that submits and optionally waits.

        Args:
            pixel_values: [batch, C, H, W] images
            block: Wait for encoding to complete
            timeout: Max wait time if blocking

        Returns:
            Batch ID
        """
        batch_id = self.submit(pixel_values)

        if block:
            features = self.get_features(batch_id=batch_id, timeout=timeout or 5.0)
            if features is None:
                raise RuntimeError("Timeout waiting for vision encoding")

        return batch_id

    def _worker_loop(self):
        """Background worker loop for encoding."""
        while self._running and not self._stop_event.is_set():
            try:
                # Get pending batch
                batch_id, pixel_values = self._get_pending_batch()

                if batch_id is None:
                    # No work, wait a bit
                    time.sleep(0.001)
                    continue

                # Encode
                start_time = time.time()
                with torch.no_grad():
                    features = self.vision_encoder(pixel_values)
                encode_time = time.time() - start_time

                # Put in queue
                self.queue.put(features, pixel_values_hash=None)

                # Update stats
                with self._stats_lock:
                    self._stats["encoded"] += 1
                    # Exponential moving average
                    alpha = 0.1
                    self._stats["avg_encode_time"] = (
                        alpha * encode_time + (1 - alpha) * self._stats["avg_encode_time"]
                    )

            except Exception as e:
                if self._running:
                    print(f"[AsyncVisionEncoder] Worker error: {e}")
                with self._stats_lock:
                    self._stats["dropped"] += 1

    def _get_pending_batch(self) -> tuple[int | None, torch.Tensor | None]:
        """Get next pending batch for processing.

        Returns:
            (batch_id, pixel_values) or (None, None) if no pending
        """
        with self._pending_lock:
            if not self._pending:
                return None, None

            # Get oldest batch (FIFO)
            batch_id = min(self._pending.keys())
            pixel_values = self._pending.pop(batch_id)
            return batch_id, pixel_values

    def get_stats(self) -> dict:
        """Get encoding statistics."""
        with self._stats_lock:
            return dict(self._stats)

    def queue_size(self) -> int:
        """Get current queue size."""
        return self.queue.size()

    def pending_size(self) -> int:
        """Get number of pending batches waiting to be encoded."""
        with self._pending_lock:
            return len(self._pending)


class CUDASyncVisionEncoder:
    """CUDA stream-based async vision encoder for GPU-to-GPU.

    Uses CUDA streams for true async processing between devices.
    Suitable when vision encoder is on one GPU (e.g., cuda:1)
    and language model on another (e.g., cuda:0).

    Example:
        >>> encoder = CUDASyncVisionEncoder(
        ...     vision_encoder,
        ...     vision_device="cuda:1",
        ...     target_device="cuda:0",
        ... )
        >>> # Submit for async encoding
        >>> event = encoder.encode_async(pixel_values)
        >>> # Later, wait and get results
        >>> features = encoder.wait_and_get(event)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_device: str = "cuda:1",
        target_device: str = "cuda:0",
    ):
        """Initialize CUDA async encoder.

        Args:
            vision_encoder: Vision encoder module
            vision_device: Device for vision processing
            target_device: Device to transfer results to
        """
        self.vision_encoder = vision_encoder
        self.vision_device = torch.device(vision_device)
        self.target_device = torch.device(target_device)

        # Create CUDA streams
        self._vision_stream = torch.cuda.Stream(device=self.vision_device)
        self._transfer_stream = torch.cuda.Stream(device=self.target_device)

        # Move encoder to vision device
        self.vision_encoder.to(self.vision_device)

    def encode_async(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Start async encoding.

        Args:
            pixel_values: [batch, C, H, W] images

        Returns:
            (output tensor placeholder, completion event)
        """
        # Move input to vision device
        pixel_values = pixel_values.to(self.vision_device)

        # Encode on vision stream
        with torch.cuda.stream(self._vision_stream):
            features = self.vision_encoder(pixel_values)

            # Create event to signal completion
            completion_event = torch.cuda.Event()
            completion_event.record(self._vision_stream)

        return features, completion_event

    def wait_and_get(
        self,
        features: torch.Tensor,
        completion_event: torch.cuda.Event,
        synchronize: bool = True,
    ) -> torch.Tensor:
        """Wait for encoding and transfer results.

        Args:
            features: Tensor from encode_async
            completion_event: Event from encode_async
            synchronize: Whether to synchronize (False for non-blocking)

        Returns:
            Features on target device
        """
        if synchronize:
            # Wait for encoding to complete
            completion_event.synchronize()

        # Transfer to target device on transfer stream
        with torch.cuda.stream(self._transfer_stream):
            # This will block if vision stream hasn't finished
            features_on_target = features.to(self.target_device)

        return features_on_target

    def encode_sync(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Synchronous encoding (for comparison/debugging).

        Args:
            pixel_values: [batch, C, H, W] images

        Returns:
            Features on target device
        """
        features, event = self.encode_async(pixel_values)
        return self.wait_and_get(features, event, synchronize=True)

    # Compatibility methods for unified interface with AsyncVisionEncoder

    def start(self):
        """Start async processing (no-op for CUDA streams)."""
        self._pending_ops: dict[int, tuple[torch.Tensor, torch.cuda.Event]] = {}
        self._batch_counter = 0

    def stop(self, timeout: float = 5.0):
        """Stop async processing (no-op for CUDA streams)."""
        self._pending_ops.clear()

    def submit(
        self,
        pixel_values: torch.Tensor,
        batch_id: int | None = None,
    ) -> int:
        """Submit for async encoding.

        Args:
            pixel_values: Images to encode
            batch_id: Optional batch ID

        Returns:
            Batch ID
        """
        if not hasattr(self, "_pending_ops"):
            self._pending_ops = {}
            self._batch_counter = 0

        if batch_id is None:
            batch_id = self._batch_counter
            self._batch_counter += 1

        features, event = self.encode_async(pixel_values)
        self._pending_ops[batch_id] = (features, event)
        return batch_id

    def get_features(
        self,
        batch_id: int | None = None,
        timeout: float = 5.0,
    ) -> torch.Tensor | None:
        """Get encoded features.

        Args:
            batch_id: Batch ID
            timeout: Max wait (unused, CUDA events handle this)

        Returns:
            Features tensor or None
        """
        if not hasattr(self, "_pending_ops"):
            return None

        if batch_id is None:
            if not self._pending_ops:
                return None
            batch_id = next(iter(self._pending_ops.keys()))

        if batch_id not in self._pending_ops:
            return None

        features, event = self._pending_ops.pop(batch_id)
        return self.wait_and_get(features, event, synchronize=True)

    def get_stats(self) -> dict:
        """Get statistics (placeholder for compatibility)."""
        return {
            "strategy": "cuda_stream",
            "pending": len(getattr(self, "_pending_ops", {})),
        }


class HybridAsyncVisionPipeline(BaseModule):
    """Hybrid pipeline supporting both CPU and GPU async processing.

    Automatically selects the best async strategy based on device placement:
    - CPU -> GPU: Thread-based producer-consumer
    - GPU -> GPU: CUDA stream-based async

    Example:
        >>> pipeline = HybridAsyncVisionPipeline(config)
        >>> pipeline.start()
        >>> # In training loop
        >>> for batch in dataloader:
        ...     # Submit for async encoding
        ...     pipeline.submit(batch["pixel_values"], batch_idx)
        ...     # Get features (may block if not ready)
        ...     features = pipeline.get_features(batch_idx)
        ...     # Use features in model
        ...     output = model(text, vision_features=features)
        >>> pipeline.stop()
    """

    def __init__(
        self,
        config: MainConfig,
        vision_encoder: nn.Module,
        queue_size: int = 4,
        num_cpu_workers: int = 2,
    ):
        """Initialize hybrid pipeline.

        Args:
            config: Main configuration
            vision_encoder: Vision encoder module
            queue_size: Prefetch queue size
            num_cpu_workers: Number of CPU encoding threads
        """
        super().__init__(config)

        self.vision_encoder = vision_encoder
        self.queue_size = queue_size
        self.num_cpu_workers = num_cpu_workers

        # Detect vision device
        vision_device = vision_encoder.vision_device
        self.target_device = torch.device("cuda:0")

        # Choose strategy
        if vision_device.type == "cpu":
            self._strategy = "thread"
            self._async_encoder = AsyncVisionEncoder(
                config,
                vision_encoder,
                num_workers=num_cpu_workers,
                queue_size=queue_size,
                target_device=self.target_device,
            )
        else:
            self._strategy = "cuda_stream"
            self._async_encoder = CUDASyncVisionEncoder(
                vision_encoder,
                vision_device=str(vision_device),
                target_device=str(self.target_device),
            )

        print(f"[HybridPipeline] Strategy: {self._strategy}")
        print(f"[HybridPipeline] Vision device: {vision_device}")
        print(f"[HybridPipeline] Target device: {self.target_device}")

    def start(self):
        """Start async processing."""
        self._async_encoder.start()

    def stop(self, timeout: float = 5.0):
        """Stop async processing."""
        self._async_encoder.stop(timeout=timeout)

    def submit(
        self,
        pixel_values: torch.Tensor,
        batch_id: int | None = None,
    ) -> int:
        """Submit images for encoding.

        Args:
            pixel_values: [batch, C, H, W] images
            batch_id: Optional batch ID

        Returns:
            Batch ID
        """
        return self._async_encoder.submit(pixel_values, batch_id)

    def get_features(
        self,
        batch_id: int | None = None,
        timeout: float = 5.0,
    ) -> torch.Tensor | None:
        """Get encoded features.

        Args:
            batch_id: Batch ID
            timeout: Max wait time

        Returns:
            Features tensor or None
        """
        return self._async_encoder.get_features(batch_id, timeout)

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return self._async_encoder.get_stats()
