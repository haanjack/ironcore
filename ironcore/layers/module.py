# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import math
import torch
from torch.cuda import nvtx
from torch.profiler import record_function

from ironcore.config import MainConfig


class BaseModule(torch.nn.Module):
    """
    class BaseModule
    """

    def __init__(self, config: MainConfig):

        super().__init__()

        self.config = config

        self.init_std = self.config.init.init_std
        self.xavier_init = self.config.init.xavier_init
        self.use_reentrant = (
            True if self.config.operation.recompute_strategy == "optimized" else False
        )

        # profier
        self._record_function = None  # for pytorch profiler
        self._nvtx_range = None
        self._hooks_registered = False
        self._profile_torch = False
        self._profile_nsys = False

    def init_weights(self):
        """Initialize model weights with proper handling for tensor parallel layers."""
        from ironcore.parallel import parallel_states

        for name, param in self.named_parameters():
            if "bias" in name:
                # zero initialization to bias
                torch.nn.init.zeros_(param)
                continue

            if "norm" in name:
                # layer norm scale parameter initialization
                torch.nn.init.ones_(param)
                continue

            # Check if this parameter belongs to a tensor parallel layer
            # We need to initialize the full tensor and then extract the shard
            module = self._get_module_for_param(name)
            is_tp_layer = (hasattr(module, 'column_parallel') or hasattr(module, 'row_parallel')) and name.endswith("weight")

            if is_tp_layer and parallel_states.get_tensor_model_parallel_world_size() > 1:
                # For TP layers, initialize full tensor then extract shard
                self._init_tp_weight(name, param, module)
            else:
                # Non-TP layer: initialize directly
                if self.xavier_init:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.normal_(param, std=self.init_std, mean=0.0)

                # following nanoGPT's trick
                # apply special scaled init to the residual projections, per GPT-2 paper
                if name.endswith("output.weight") or name.endswith("down_proj.weight"):
                    torch.nn.init.normal_(param, std=self.init_std/math.sqrt(2 * self.config.model.num_layers), mean=0.0)

    def _get_module_for_param(self, param_name):
        """Get the module that owns a given parameter."""
        parts = param_name.split('.')
        module = self
        for part in parts[:-1]:  # Exclude the last part (parameter name)
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _init_tp_weight(self, name, param, module):
        """Initialize weights for tensor parallel layers by creating full tensor and extracting shard."""
        from ironcore.parallel import parallel_states

        tp_rank = parallel_states.get_tensor_model_parallel_rank()
        tp_size = parallel_states.get_tensor_model_parallel_world_size()

        # Determine full shape and shard dimension based on parallelism type
        if hasattr(module, 'column_parallel') and module.column_parallel:
            # ColumnParallelLinear: shard along dim=1 (columns)
            # Current param shape: [input_size, output_size // tp_size]
            # Full shape: [input_size, output_size]
            # Note: concatenated_weights is already accounted for in the output_size before division
            full_shape = (param.shape[0], param.shape[1] * tp_size)
            shard_dim = 1
        elif hasattr(module, 'row_parallel') and module.row_parallel:
            # RowParallelLinear or VocabParallelEmbedding: shard along dim=0 (rows)
            # Current param shape: [input_size // tp_size, output_size]
            # Full shape: [input_size, output_size]
            full_shape = (param.shape[0] * tp_size, param.shape[1])
            shard_dim = 0
        else:
            # Fallback: initialize directly
            torch.nn.init.normal_(param, std=self.init_std, mean=0.0)
            return

        # Initialize the full tensor (same seed on all ranks ensures consistency)
        full_tensor = torch.empty(full_shape, dtype=param.dtype, device=param.device)

        # Determine initialization std (special case for residual projections)
        init_std = self.init_std
        if name.endswith("output.weight") or name.endswith("down_proj.weight"):
            init_std = self.init_std / math.sqrt(2 * self.config.model.num_layers)

        if self.xavier_init:
            torch.nn.init.xavier_uniform_(full_tensor)
        else:
            torch.nn.init.normal_(full_tensor, std=init_std, mean=0.0)

        # Extract the shard for this rank
        shard_size = param.shape[shard_dim]
        
        # Check for concatenated weights (only for ColumnParallelLinear)
        concatenated_weights = getattr(module, 'concatenated_weights', 1)

        if shard_dim == 1 and concatenated_weights > 1:
             # Complex splitting for ColumnParallel with concatenation (e.g. QKV or KV)
             # full_tensor: [input, output_total]
             output_total = full_shape[1]
             per_part = output_total // concatenated_weights
             
             # Split into parts (e.g. K, V)
             parts = torch.split(full_tensor, per_part, dim=1)
             
             shards = []
             for part in parts:
                 # Split part into TP shards
                 part_shard_size = per_part // tp_size
                 start = tp_rank * part_shard_size
                 end = (tp_rank + 1) * part_shard_size
                 shards.append(part[:, start:end])
                 
             shard = torch.cat(shards, dim=1)
        else:
            start_idx = tp_rank * shard_size
            end_idx = (tp_rank + 1) * shard_size

            if shard_dim == 0:
                shard = full_tensor[start_idx:end_idx, :]
            else:  # shard_dim == 1
                shard = full_tensor[:, start_idx:end_idx]

        # Copy the shard to the parameter
        with torch.no_grad():
            param.copy_(shard)

    def save_to_state_dict(self, destination, prefix, keep_vars):
        """Save module state to the `destination` dictionary."""
        return super()._save_to_state_dict(destination, prefix, keep_vars)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _forward_pre_hook(self, module, input):
        name = f"{self.__class__.__name__}"
        if self._profile_torch:
            self._record_function = record_function(f"Forward {name}")
            self._record_function.__enter__()
        if self._profile_nsys:
            self._nvtx_range = nvtx.range_push(f"Forward {name}")

    def _forward_post_hook(self, module, input, output):
        if self._record_function is not None:
            self._record_function.__exit__(None, None, None)
            self._record_function = None
        if self._nvtx_range is not None:
            nvtx.range_pop()
            self._nvtx_range = None

    def _backward_pre_hook(self, module, grad_output):
        name = f"{self.__class__.__name__}"
        if self._profile_torch:
            self._record_function = record_function(f"Backward {name}")
            self._record_function.__enter__()
        if self._profile_nsys:
            self._nvtx_range = nvtx.range_push(f"Backward {name}")

    def _backward_post_hook(self, module, grad_input, grad_output):
        if self._record_function is not None:
            self._record_function.__exit__(None, None, None)
            self._record_function = None
        if self._nvtx_range is not None:
            nvtx.range_pop()
            self._nvtx_range = None

    def register_profile_hooks(self,
                               profile_torch: bool = False,
                               profile_nsys: bool = False,
                               ):
        if self._hooks_registered:
            return
        self._profile_torch = profile_torch
        self._profile_nsys = profile_nsys

        self.register_forward_pre_hook(self._forward_pre_hook)
        self.register_forward_hook(self._forward_post_hook)
        self.register_full_backward_pre_hook(self._backward_pre_hook)
        self.register_full_backward_hook(self._backward_post_hook)
        self._hooks_registered = True
        print(self._get_name())

        for child in self.children():
            if isinstance(child, BaseModule):
                child.register_profile_hooks(profile_torch, profile_nsys)
            if isinstance(child, torch.nn.ModuleList):
                for module in child:
                    if isinstance(module, BaseModule):
                        module.register_profile_hooks(
                            profile_torch, profile_nsys)
