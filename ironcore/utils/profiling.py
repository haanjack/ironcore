# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import contextmanager

import torch


def profile_function(tag):
    """Decorator for profiling function"""
    enabled = os.environ.get("IRONCORE_PROFILE", "0") == "1"

    def decorator(func):
        if not enabled:
            return func

        def wrapper(*args, **kwargs):
            with torch.profiler.record_function(tag):
                if hasattr(torch.cuda, "nvtx"):
                    torch.cuda.nvtx.range_push(tag)
                result = func(*args, **kwargs)
                if hasattr(torch.cuda, "nvtx"):
                    torch.cuda.nvtx.range_pop()
                return result

        return wrapper

    return decorator


@contextmanager
def profile_context(tag):
    """Context manager for profiling"""
    if os.environ.get("IRONCORE_PROFILE", "0") != "1":
        yield
        return

    if hasattr(torch.cuda, "nvtx"):
        torch.cuda.nvtx.range_push(tag)
    with torch.profiler.record_function(tag):
        yield
    if hasattr(torch.cuda, "nvtx"):
        torch.cuda.nvtx.range_pop()
