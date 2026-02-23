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

from ironcore.config import MainConfig

from .fused_layer_norm import LayerNorm
from .fused_rms_norm import RmsNorm


def get_norm(config: MainConfig):
    """Returns the normalization layer."""
    ln_type = config.model.ln_type.lower()

    if ln_type == "layernorm":
        return LayerNorm(config)
    if ln_type == "rmsnorm":
        return RmsNorm(config)

    raise NotImplementedError(f"{config.ln_type} is not supported")
