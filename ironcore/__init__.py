# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import torch

if torch.version.cuda is not None and torch.__version__ < "2.15.0a0":
    raise ImportError(
        "IronCore requires PyTorch 2.15 or newer for CUDA, including NGC 2.15.0a0 builds. "
        f"Found {torch.__version__}. Rebuild the CUDA container with scripts/docker/build.sh."
    )

from .global_vars import (  # noqa: E402
    get_config,
    get_logger,
    get_timer,
    get_tokenizer,
    set_global_states,
)
from .utils.mfu import MFUCalculator, MFUResult, compute_tflops  # noqa: E402

__all__ = [
    "get_config",
    "get_tokenizer",
    "set_global_states",
    "get_logger",
    "get_timer",
    "MFUCalculator",
    "MFUResult",
    "compute_tflops",
    # Entrypoints
    "load_full_config",
    "train",
    "generate",
    "export",
    "preprocess",
    "evaluate",
]

_ENTRYPOINTS = {
    "load_full_config": (".train", "load_full_config"),
    "train": (".train", "train"),
    "generate": (".generate", "generate"),
    "export": (".export", "export"),
    "preprocess": (".preprocess", "preprocess"),
    "evaluate": (".evaluate", "evaluate"),
}


def __getattr__(name):
    if name in _ENTRYPOINTS:
        import importlib

        module_path, attr = _ENTRYPOINTS[name]
        return getattr(importlib.import_module(module_path, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
