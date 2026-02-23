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

"""
Backward compatibility module for data configuration.

This module re-exports DataConfig, DatasetConfig, and UniversalDataConfig
from ironcore.config.config_data for backward compatibility.
"""

# Re-export from the canonical location
from ironcore.config.config_data import (
    DataConfig,
    DatasetConfig,
    UniversalDataConfig,
    load_data_config,
)

__all__ = ["DataConfig", "DatasetConfig", "UniversalDataConfig", "load_data_config"]
