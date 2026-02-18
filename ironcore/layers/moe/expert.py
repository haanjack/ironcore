# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Expert MLP layer for Mixture of Experts.

Each expert is a feed-forward network with:
- Up projection: hidden_size -> intermediate_size (ColumnParallel)
- Activation function
- Down projection: intermediate_size -> hidden_size (RowParallel)

This inherits from ParallelMLP and adds expert-specific functionality.
"""

from ironcore.config import MainConfig
from ironcore.layers.parallel_mlp import ParallelMLP


class ExpertMLP(ParallelMLP):
    """Single Expert MLP layer for MoE.

    Each expert processes tokens routed to it independently.
    Supports tensor parallelism and async communication for overlapping.

    This is a specialized ParallelMLP with:
    - Expert ID tracking for debugging
    - gather_output=False (output stays parallel for TP compatibility)

    Async Communication Pattern:
    ----------------------------
    When async_communication=True in forward():
    1. down_proj returns (partial_output, handle) instead of final output
    2. Caller processes other experts while handle completes
    3. Caller calls finalize(output, handle) to get final result
    This allows overlapping expert computation with all-reduce communication.

    Args:
        config: Main configuration
        hidden_size: Input/output hidden dimension
        intermediate_size: Expert intermediate dimension (d_ffn for expert)
        expert_id: Index of this expert (for debugging/logging)
    """

    def __init__(
        self,
        config: MainConfig,
        hidden_size: int,
        intermediate_size: int,
        expert_id: int = 0,
    ):
        # Initialize base ParallelMLP with gather_output=False
        # (output stays parallel for RowParallel input)
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gather_output=False,
            name=f"expert_{expert_id}",
        )

        self.expert_id = expert_id

    def __repr__(self):
        return f"ExpertMLP(id={self.expert_id}, hidden={self.hidden_size}, ffn={self.intermediate_size})"
