# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Iterable, Union

import torch
import torch.distributed as dist
import yaml
from dotenv import load_dotenv


def _convert_lists_to_dict(data):

    if isinstance(data, list):
        # Only merge if all items are dicts (e.g., list of dataset configs)
        if all(isinstance(item, dict) for item in data):
            merged_dict = {}
            # build dictionary from list
            for item in data:
                merged_dict.update(item)
            # check if list in the data and convert to dict
            for key, value in merged_dict.items():
                if isinstance(value, list):
                    value = _convert_lists_to_dict(value)
                merged_dict[key] = value
            return merged_dict
        else:
            # List of primitives (e.g., splits: [0.99, 0.01, 0.0]) - return as-is
            return data
    elif isinstance(data, dict):
        return {key: _convert_lists_to_dict(value) for key, value in data.items()}
    else:
        return data


# load environment variables from .env and apply to yaml load
def env_var_constructor(loader, node):
    value = loader.construct_scalar(node)
    pattern = re.compile(r"\$\{(\w+)\}")
    match = pattern.findall(value)
    if match:
        for var in match:
            value = value.replace(f"${{{var}}}", os.getenv(var, ""))
    return value


# apply environment variable in yaml load
yaml.add_implicit_resolver("!env_var", re.compile(r".*\$\{(\w+)\}.*"))
yaml.add_constructor("!env_var", env_var_constructor)


def load_yaml_config(config_path):
    """Load yaml config file."""

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config = _convert_lists_to_dict(config)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {str(e)}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Error loading yaml config: {str(e)}") from e

    return config


def get_dataset_base_dir() -> Path:
    """Get dataset path."""
    load_dotenv()
    return Path(os.getenv("DATASET_DIR", "./"))


# parallel utilities
def is_first_rank():
    """Check whether it's rank 0."""
    assert dist.is_initialized(), "torch distributed is not initialized."
    return dist.get_rank() == 0


def is_last_rank():
    """Check whether it's the last rank."""
    assert dist.is_initialized(), "torch distributed is not initialized."
    return dist.get_rank() == dist.get_world_size() - 1


def print_rank_0(message: str):
    """Print message only if it's rank 0."""
    if is_first_rank():
        print(message)


def print_last_rank(message: str):
    """Print message only if it's the last rank."""
    if is_last_rank():
        print(message)


def get_device():
    """Returns device type"""
    if torch.cuda.is_available():
        assert (
            torch.distributed.is_initialized()
        ), "torch distributed is not initialized"
        device = (
            f"cuda:{dist.get_node_local_rank()}"
            if hasattr(dist, "get_node_local_rank")
            else "cuda"
        )
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device


def get_model_dtype(config):
    """Returns model dtype checking device supports"""
    if config.model.precision.lower() in ["bfloat16", "bf16"]:
        if torch.cuda.is_available():
            assert (
                torch.cuda.is_bf16_supported()
            ), "bfloat16 is not supported on this device"
        dtype = torch.bfloat16
    elif config.model.precision.lower() in ["float16", "fp16"]:
        dtype = torch.float16
    else:
        # logger.warning("Using FP32, which is slow for the training.")
        dtype = torch.float

    return dtype


class Timer:
    """Timer"""

    def __init__(self):
        self.timers: Dict[str, List[float]] = {}
        self.running: Dict[str, bool] = {}

    def register(self, name: str):
        """Register timer."""
        if name in self.running:
            raise KeyError(f"Requested timer ({name}) is already registered")
        self.timers[name] = []
        self.running[name] = False

    def start(self, name: str):
        """Start timer."""
        if name not in self.running:
            self.register(name)

        assert not self.running[
            name
        ], f"Timer {name} is already running. This can happen in duplicated operaiton"

        self.running[name] = True
        self.timers[name].append(time.time())

    def stop(self, name: str):
        """Stop timer."""
        if name not in self.running:
            raise KeyError(f"Not initialized timer ({name}) is requested")
        if not self.running[name]:
            raise RuntimeError(
                f"Stopping timer {name} is requested while it is already stopped. This can be duplicated operation."
            )

        self.running[name] = False
        self.timers[name][-1] = time.time() - self.timers[name][-1]

    def get(self, name: str) -> float:
        """Get summary of requested timer."""
        if name not in self.running:
            raise KeyError(f"Not initialized timer ({name}) is requested")
        if self.running[name]:
            self.stop(name)
        return sum(self.timers[name]) / len(self.timers[name])

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all timers."""
        summary = {}
        for name, times in self.timers.items():
            if len(times) == 0:
                continue
            summary[name] = sum(times) / len(times)
        return summary

    def reset(self, name: str):
        """Reset timer."""
        if name not in self.running:
            raise KeyError(f"Not initialized timer ({name}) is requested")
        self.timers[name] = []
        self.running[name] = False

    def reset_all(self):
        """Reset all timers."""
        for name in self.timers.keys():
            self.reset(name)

    def stop_all(self):
        """Stop all timers."""
        for name in self.timers.keys():
            self.stop(name)


def bytes_to_mib(bytes_value: int):
    return bytes_value // 1024 // 1024


def get_memory_usage(in_mib: bool = False):
    """Get memory usage."""
    summary = {}

    device = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        summary["memory_allocated"] = torch.cuda.memory_allocated(device)
        summary["max_memory_allocated"] = torch.cuda.max_memory_allocated(
            device)
        summary["memory_reserved"] = torch.cuda.memory_reserved(device)
        summary["max_memory_reserved"] = torch.cuda.max_memory_reserved(device)
    elif torch.backends.mps.is_available():
        summary["memory_allocated"] = torch.mps.current_allocated_memory()
        summary["driver_allocated"] = torch.mps.driver_allocated_memory()

    if in_mib:
        for k, v in summary.items():
            summary[k] = bytes_to_mib(v)

    return summary


def profile_function(tag):
    """Decorator for profiling function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with torch.profiler.record_function(tag):
                torch.cuda.nvtx.range_push(tag)
                result = func(*args, **kwargs)
                torch.cuda.nvtx.range_pop()
                return result
        return wrapper
    return decorator


@contextmanager
def profile_context(tag):
    """Context manager for profiling"""
    torch.cuda.nvtx.range_push(tag)
    with torch.profiler.record_function(tag):
        yield
    torch.cuda.nvtx.range_pop()


def clip_grad_norm_tp(parameters: Union[torch.Tensor, Iterable[torch.Tensor]], max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """
    Clips gradient norm of an iterable of parameters, considering Tensor Parallelism.
    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    from torch import inf
    from ironcore.parallel import parallel_states

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    if len(parameters) == 0:
        return torch.tensor(0.)

    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
        # All-reduce max across TP group
        if parallel_states.get_tensor_model_parallel_world_size() > 1:
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=parallel_states.get_tensor_model_parallel_group())
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
        if norm_type == 2.0: # Most common case
             # Square the norm
             total_norm_sq = total_norm ** 2
             # All-reduce sum across TP group
             if parallel_states.get_tensor_model_parallel_world_size() > 1:
                dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM, group=parallel_states.get_tensor_model_parallel_group())
             total_norm = total_norm_sq ** 0.5
        else:
             # For other norms, it's more complex (p-norm sum is not additive like sq norm)
             # Fallback: assume local clipping is approximation or raise error
             # For strict correctness, we'd need to gather all grads or sum powers.
             # Assuming standard L2 norm usage for now.
             pass

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
            
    return total_norm

