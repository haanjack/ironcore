# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Run-local compatibility shim for vLLM 0.30 prefetch offloading with LoRA.

vLLM allocates LoRA buffers on the base parameter's storage device. Prefetch
stores that parameter on CPU and executes it on CUDA. Keep adapter buffers
on the worker's CUDA device; leave base storage and all computation unchanged.
"""

import sys

from vllm.v1.worker.gpu_worker import Worker


class PrefetchLoRAWorker(Worker):
    def load_model(self, *args, **kwargs):
        from vllm.lora.layers import utils

        original = utils._get_lora_device
        changes = 0

        def execution_device(base_layer):
            nonlocal changes
            device = original(base_layer)
            if device.type == "cpu":
                changes += 1
                return self.device
            return device

        # Existing from-import aliases and later imports must use the same helper.
        for name, module in tuple(sys.modules.items()):
            if (
                name.startswith("vllm.lora.layers")
                and getattr(module, "_get_lora_device", None) is original
            ):
                module._get_lora_device = execution_device
        result = super().load_model(*args, **kwargs)
        print(f"PREFETCH_LORA_DEVICE_FIX rank={self.rank} buffers_on_cuda={changes}", flush=True)
        return result
