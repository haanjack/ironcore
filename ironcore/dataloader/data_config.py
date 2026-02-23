# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

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
