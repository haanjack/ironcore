# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Device management utilities for hybrid VLA training.

Supports:
- CPU inference with AVX-512 optimization
- Multi-GPU including e-GPU (cuda:2, etc.)
- Automatic device selection
- Vision feature caching
"""

import os

import torch


def check_cpu_capabilities() -> dict:
    """Check CPU capabilities for optimized inference."""
    caps = {
        "avx512": False,
        "avx2": False,
        "num_threads": os.cpu_count() or 1,
    }

    # Check PyTorch CPU capabilities
    if hasattr(torch.cpu, '_is_avx512_vnni_supported'):
        caps["avx512"] = torch.cpu._is_avx512_vnni_supported()
    elif hasattr(torch, '__config__'):
        config = torch.__config__.show()
        caps["avx512"] = "AVX512" in config
        caps["avx2"] = "AVX2" in config

    return caps


def optimize_cpu_inference():
    """Optimize PyTorch for CPU inference with AVX-512."""
    caps = check_cpu_capabilities()

    if caps["avx512"]:
        print("[VLA] AVX-512 detected - enabling optimized CPU inference")
        # Set optimal thread count for AVX-512
        torch.set_num_threads(caps["num_threads"])
        # Enable TF32 for faster matmul (if available)
        torch.set_float32_matmul_precision('high')

    return caps


class DeviceManager:
    """Manages device placement for VLA components.

    Supports:
    - Auto device selection based on available GPUs
    - Explicit device placement (cuda:0, cuda:1, cuda:2 for e-GPU, cpu)
    - CPU offloading with AVX-512 optimization
    - e-GPU support (external GPUs)
    """

    def __init__(
        self,
        vision_device: str = "auto",
        language_device: str = "auto",
        offload_threshold_mb: int = 0,
        optimize_cpu: bool = True,
    ):
        if optimize_cpu:
            self.cpu_caps = optimize_cpu_inference()
        else:
            self.cpu_caps = check_cpu_capabilities()

        self.vision_device = self._resolve_device(vision_device, "vision")
        self.language_device = self._resolve_device(language_device, "language")
        self.offload_threshold_mb = offload_threshold_mb

        # Print device summary
        self._print_device_summary()

    def _print_device_summary(self):
        """Print summary of device configuration."""
        print("[VLA] Device configuration:")
        print(f"  - Vision encoder: {self.vision_device}")
        print(f"  - Language model: {self.language_device}")
        if self.vision_device.type == "cpu":
            caps = self.cpu_caps
            print(f"  - CPU capabilities: AVX512={caps['avx512']}, AVX2={caps['avx2']}, threads={caps['num_threads']}")

    def _resolve_device(self, device: str, component: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            return self._auto_select_device(component)
        return torch.device(device)

    def _auto_select_device(self, component: str) -> torch.device:
        """Auto-select best device for component."""
        if not torch.cuda.is_available():
            print("[VLA] No CUDA available, using CPU")
            return torch.device("cpu")

        num_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0")  # Default

        # Print available GPUs
        print(f"[VLA] Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  - cuda:{i}: {props.name} ({props.total_memory // 1024**3}GB)")

        if component == "vision":
            # Vision encoder: prefer separate GPU or CPU to save GPU memory
            if num_gpus >= 3:
                # 3+ GPUs: use last GPU for vision (could be e-GPU)
                device = torch.device(f"cuda:{num_gpus - 1}")
            elif num_gpus >= 2 and self.cpu_caps["avx512"]:
                # 2 GPUs with TP=2: use CPU for vision (AVX-512 helps)
                device = torch.device("cpu")
            elif num_gpus >= 2:
                # Use second GPU for vision
                device = torch.device("cuda:1")
            elif num_gpus == 1:
                # Single GPU: check memory and decide
                free_memory = torch.cuda.mem_get_info(0)[0] / 1024**3  # GB
                if free_memory <= 20 and self.cpu_caps["avx512"]:
                    # CPU with AVX-512 is viable when GPU memory is limited
                    device = torch.device("cpu")

        return device

    def place_module(self, module: torch.nn.Module, component: str) -> torch.nn.Module:
        """Place module on appropriate device."""
        if component == "vision":
            device = self.vision_device
        else:
            device = self.language_device

        return module.to(device)

    def move_tensor(self, tensor: torch.Tensor, component: str) -> torch.Tensor:
        """Move tensor to component's device."""
        if component == "vision":
            return tensor.to(self.vision_device)
        return tensor.to(self.language_device)

    def get_memory_info(self) -> dict:
        """Get memory information for all GPUs."""
        info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                free = torch.cuda.mem_get_info(i)[0] / 1024**3
                used = total - free
                info[f"cuda:{i}"] = {
                    "name": torch.cuda.get_device_properties(i).name,
                    "total_gb": round(total, 2),
                    "used_gb": round(used, 2),
                    "free_gb": round(free, 2),
                }
        return info

    def should_offload_to_cpu(self, param_count: int) -> bool:
        """Check if parameters should be offloaded to CPU."""
        if self.offload_threshold_mb <= 0:
            return False

        # Estimate memory in MB (fp32 = 4 bytes per param)
        param_memory_mb = (param_count * 4) / 1024**2
        return param_memory_mb > self.offload_threshold_mb


class HybridVisionEncoder(torch.nn.Module):
    """Vision encoder with device-aware placement.

    Supports:
    - Running vision encoder on different device than language model
    - Automatic tensor transfer between devices
    - Optional vision feature caching for repeated images
    - CPU inference with AVX-512 optimization
    """

    def __init__(
        self,
        vision_encoder: torch.nn.Module,
        projector: torch.nn.Module,
        device_manager: DeviceManager,
        cache_features: bool = False,
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.projector = projector
        self.device_manager = device_manager

        # Place vision encoder on its designated device
        self.vision_encoder = device_manager.place_module(
            self.vision_encoder, "vision"
        )

        # Projector stays on language device (trainable)
        self.projector = device_manager.place_module(
            self.projector, "language"
        )

        # Feature caching
        self.cache_features = cache_features
        self._feature_cache: dict = {}

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_ids: list | None = None,
    ) -> torch.Tensor:
        """Encode images and project to language space.

        Args:
            pixel_values: [batch, C, H, W] images
            image_ids: Optional IDs for caching (e.g., file paths)

        Returns:
            [batch, num_patches, language_dim] on language device
        """
        # Check cache
        if self.cache_features and image_ids is not None:
            cached = [self._feature_cache.get(img_id) for img_id in image_ids]
            if all(c is not None for c in cached):
                # All cached, return from cache
                vision_features = torch.stack(cached)
                return self.projector(vision_features)

        # Move images to vision device
        pixel_values = self.device_manager.move_tensor(pixel_values, "vision")

        # Encode on vision device
        with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
            vision_features = self.vision_encoder(pixel_values)

        # Cache features
        if self.cache_features and image_ids is not None:
            for i, img_id in enumerate(image_ids):
                self._feature_cache[img_id] = vision_features[i].cpu()

        # Move features to language device
        vision_features = self.device_manager.move_tensor(vision_features, "language")

        # Project on language device
        projected = self.projector(vision_features)

        return projected

    def clear_cache(self):
        """Clear feature cache."""
        self._feature_cache.clear()

    def get_vision_device(self) -> torch.device:
        """Get the device where vision encoder runs."""
        return self.device_manager.vision_device

    def get_language_device(self) -> torch.device:
        """Get the device where language model runs."""
        return self.device_manager.language_device


def get_optimal_device_config(
    tensor_parallel_size: int = 1,
    prefer_cpu_for_vision: bool = False,
) -> dict:
    """Get optimal device configuration for given setup.

    Args:
        tensor_parallel_size: TP size for language model
        prefer_cpu_for_vision: Force CPU for vision (useful with AVX-512)

    Returns:
        Dictionary with recommended device placement
    """
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cpu_caps = check_cpu_capabilities()
    has_avx512 = cpu_caps["avx512"]

    # Default configuration
    config = {
        "vision_device": "cuda:0",
        "language_device": "cuda:0",
        "recommendation": "Default configuration",
        "cpu_avx512": has_avx512,
    }

    # No GPUs
    if num_gpus == 0:
        config.update({
            "vision_device": "cpu",
            "language_device": "cpu",
            "recommendation": "No GPUs available, using CPU only",
        })
        return config

    # Check for e-GPU info
    egpu_info = ""
    if num_gpus >= 3:
        last_gpu_name = torch.cuda.get_device_properties(num_gpus - 1).name
        egpu_info = f" (e-GPU: {last_gpu_name})"

    # Prefer CPU for vision with AVX-512
    if prefer_cpu_for_vision and has_avx512:
        config.update({
            "vision_device": "cpu",
            "recommendation": f"CPU (AVX-512) for vision, GPU0 for language{egpu_info}",
        })
        return config

    # Single GPU
    if num_gpus == 1:
        config["recommendation"] = "Single GPU - vision and language share GPU"
        return config

    # Dual GPU
    if num_gpus == 2:
        if tensor_parallel_size == 2:
            config.update({
                "vision_device": "cpu",
                "recommendation": (
                    "Dual GPU with TP=2 - vision on CPU (AVX-512), language across both GPUs"
                    if has_avx512
                    else "Dual GPU with TP=2 - vision on CPU to maximize language memory"
                ),
            })
        else:
            config.update({
                "vision_device": "cuda:1",
                "recommendation": "Dual GPU - vision on GPU1, language on GPU0",
            })
        return config

    # 3+ GPUs (including e-GPU)
    vision_gpu = num_gpus - 1
    if tensor_parallel_size >= 2:
        config.update({
            "vision_device": f"cuda:{vision_gpu}",
            "recommendation": f"Vision on e-GPU (cuda:{vision_gpu}), language TP on GPUs 0-{tensor_parallel_size-1}",
        })
    else:
        config.update({
            "vision_device": f"cuda:{vision_gpu}",
            "recommendation": f"Vision on cuda:{vision_gpu} (e-GPU), language on GPU0",
        })

    return config
