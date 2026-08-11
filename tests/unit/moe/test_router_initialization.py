# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import torch
from tests.fixtures.config_fixtures import create_moe_test_config

from ironcore.layers.moe.router import TopKRouter


def test_router_constructor_applies_scaled_initialization():
    config = create_moe_test_config(
        hidden_size=4096, num_routed_experts=16, num_experts_per_token=2
    )
    torch.manual_seed(7)
    router = TopKRouter(config, hidden_size=4096, num_experts=16, top_k=2)

    expected_std = config.init.init_std / 16**0.5
    assert abs(router.weight.std().item() - expected_std) < expected_std * 0.08
