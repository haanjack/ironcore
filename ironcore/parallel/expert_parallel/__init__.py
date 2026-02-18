# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Expert Parallelism module for Mixture of Experts models.

This module provides two communication approaches:

1. All-Reduce (Simple):
   - all_reduce_ep: Synchronous all-reduce
   - all_reduce_ep_with_grad: All-reduce with gradient support

2. All-to-All (Async-Optimized):
   - AllToAllDispatcher: Class for dispatch/gather with async support
   - dispatch_tokens: Dispatch tokens to expert ranks
   - gather_tokens: Gather expert outputs with optional async

The all-to-all approach allows overlapping communication with computation,
which is beneficial for large models with many experts.
"""

# All-Reduce Approach
# All-to-All Approach
from .comm import (
    AllToAllDispatcher,
    DispatchMetadata,
    DispatchOutput,
    all_reduce_ep,
    all_reduce_ep_with_grad,
    dispatch_tokens,
    gather_tokens,
)
from .parallel_states import (
    destroy_expert_parallel,
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    initialize_expert_parallel,
)

__all__ = [
    # Parallel states
    "initialize_expert_parallel",
    "destroy_expert_parallel",
    "get_expert_model_parallel_group",
    "get_expert_model_parallel_rank",
    "get_expert_model_parallel_world_size",
    # All-Reduce Approach
    "all_reduce_ep",
    "all_reduce_ep_with_grad",
    # All-to-All Approach
    "AllToAllDispatcher",
    "DispatchOutput",
    "DispatchMetadata",
    "dispatch_tokens",
    "gather_tokens",
]
