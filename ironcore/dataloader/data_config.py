# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

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
