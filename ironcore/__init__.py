# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

from .global_vars import get_config, get_logger, get_timer, get_tokenizer, set_global_states
from .utils.mfu import MFUCalculator, MFUResult, compute_tflops

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
