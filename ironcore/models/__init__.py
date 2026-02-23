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

from .dummy import DummyModel
from .transformer import TransformerModel

SUPPORTED_TRANSFORMER_PREFIXES = ["GPT", "LLAMA", "GEMMA1", "QWEN", "PHI1", "PHI2"]


def get_model_provider_func(config):
    model_name = config.model.name.upper()

    if model_name == "DUMMY":
        return DummyModel

    if any(model_name.startswith(prefix) for prefix in SUPPORTED_TRANSFORMER_PREFIXES):
        return TransformerModel

    raise NotImplementedError(f"Model architecture '{config.model.name}' is not supported")


__all__ = [
    "DummyModel",
    "TransformerModel",
]
