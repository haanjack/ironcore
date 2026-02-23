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

from torch import nn
from torch.nn import functional as F


class GLUActivation(nn.Module):
    """GLU Activation."""

    def __init__(self, input_dim: int, variant="glu"):
        super().__init__()
        self.variant = variant.lower()
        if self.variant in ["geglu", "siglu"]:
            self.linear = nn.Linear(input_dim, input_dim)
        if self.variant in ["glu", "geglu", "siglue"]:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [seq_len, batch_size, hidden_dim * 2]
        x_feature, x_gate = x.chunk(2, dim=-1)

        output = None
        if self.variant == "glu":
            output = x_feature * self.sigmoid(x_gate)
        elif self.variant == "geglu":
            output = self.linear(x_feature) * self.sigmoid(x_gate)
        elif self.variant == "siglu":
            output = x_feature * self.sigmoid(self.linear(x_gate))
        elif self.variant == "swiglu":
            output = F.silu(x_feature) * x_gate
        else:
            raise ValueError("Unknown GLU variant: {self.variant}")

        return output


def get_activation(activation_name, input_dim: int = None):
    """get activation function"""
    activation_layer = None
    if activation_name == "relu":
        activation_layer = nn.ReLU()
    elif activation_name == "gelu":
        activation_layer = nn.GELU()
    elif activation_name == "silu":
        activation_layer = nn.SiLU()
    elif "glu" in activation_name:
        assert input_dim is not None, "input_dim is required for GLU activation"
        activation_layer = GLUActivation(input_dim=input_dim, variant=activation_name)
    else:
        raise NotImplementedError
    return activation_layer
