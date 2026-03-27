# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: MIT

"""RMSNorm with configurable backend: torch (default), auto, triton."""

import torch
import torch.nn as nn
from ironcore.layers.module import BaseModule


def _has_triton():
    try:
        import triton
        return True
    except ImportError:
        return False


class TritonRmsNorm(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.d_model = config.model.d_model
        self.eps = config.model.ln_eps
        self.weight = nn.Parameter(torch.ones(self.d_model))
        from ironcore.kernels.triton.rmsnorm import triton_rmsnorm
        self._triton_rmsnorm = triton_rmsnorm

    def forward(self, x):
        return self._triton_rmsnorm(x, self.weight, self.eps)


class TorchNativeRmsNorm(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.d_model = config.model.d_model
        self.eps = config.model.ln_eps
        self.weight = nn.Parameter(torch.ones(self.d_model))
        if not hasattr(nn, 'RMSNorm'):
            raise ImportError("nn.RMSNorm requires PyTorch 2.4+. Use kernel_backend='auto' or 'triton'.")
        try:
            self._rmsnorm = nn.RMSNorm(self.d_model, eps=self.eps, bias=False)
        except TypeError:
            self._rmsnorm = nn.RMSNorm(self.d_model, eps=self.eps)

    def forward(self, x):
        self._rmsnorm.weight = self.weight
        return self._rmsnorm(x)


def get_rmsnorm_layer(config):
    backend = config.model.kernel_backend
    has_triton = _has_triton()

    if backend == "triton":
        if not has_triton:
            raise ImportError("Triton not available. Use kernel_backend='auto' or 'torch'.")
        return TritonRmsNorm(config)
    if backend == "auto":
        return TritonRmsNorm(config) if has_triton else TorchNativeRmsNorm(config)
    if backend == "torch":
        return TorchNativeRmsNorm(config)
    raise ValueError(f"Invalid kernel_backend: {backend}. Use 'torch', 'auto', or 'triton'.")


RmsNorm = TorchNativeRmsNorm  # legacy alias
