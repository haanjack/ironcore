# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mixture of Experts (MoE) module for ironcore.

This module provides DeepSeek-MoE style MoE layers:
- MoEMLP: Main MoE layer combining shared and routed experts
- TopKRouter: Token routing with top-k selection
- ExpertMLP: Single expert feed-forward network
- Load balance loss functions

YAML Configuration Example:
---------------------------
Add a `moe` section to your model config YAML:

    # configs/model/my-moe-model.yaml
    d_model: 768
    d_ffn: 3072
    num_layers: 12

    moe:
      use_moe: true                    # Enable MoE
      num_shared_experts: 2            # Experts that process ALL tokens
      num_routed_experts: 64           # Experts selected by top-k routing
      num_experts_per_token: 6         # Top-k experts per token
      aux_loss_alpha: 0.01             # Load balancing loss weight
      expert_model_parallel_size: 1    # EP degree (1 = no parallelism)

Then use with the trainer:

    python -m ironcore.trainer --config configs/train.yaml

Key Parameters:
    - num_shared_experts: Always active, process all tokens (e.g., 2)
    - num_routed_experts: Pool of experts for routing (e.g., 64)
    - num_experts_per_token: How many routed experts per token (e.g., 6)
    - aux_loss_alpha: Higher = more balanced expert utilization

Python API Example:
-------------------
    from ironcore.layers.moe import MoEMLP
    from ironcore.config import MainConfig

    config = MainConfig.from_yaml("configs/model/my-moe-model.yaml")
    moe_layer = MoEMLP(config)
    output = moe_layer(hidden_states)

    # Get auxiliary loss for training
    aux_loss = moe_layer.get_aux_loss()
    total_loss = lm_loss + aux_loss
"""

from .expert import ExpertMLP
from .load_balance_loss import (
    LoadBalanceLoss,
    compute_load_balance_loss,
    compute_router_z_loss,
    get_expert_utilization,
)
from .moe_layer import CommunicationMode, MoEMLP
from .router import RouterOutput, TopKRouter

__all__ = [
    # Main MoE layer
    "MoEMLP",
    "CommunicationMode",
    # Components
    "ExpertMLP",
    "TopKRouter",
    "RouterOutput",
    # Loss functions
    "LoadBalanceLoss",
    "compute_load_balance_loss",
    "compute_router_z_loss",
    "get_expert_utilization",
]
