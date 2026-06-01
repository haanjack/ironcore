# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Backward compatibility re-export of OffloadConfig.

DEPRECATED: OffloadConfig has moved to ironcore.config.config_offload.
Import from ironcore.config instead:
    from ironcore.config import OffloadConfig
This file will be removed in a future version.
"""

import warnings

from ironcore.config.config_offload import OffloadConfig

__all__ = ["OffloadConfig"]


class _DeprecatedMeta(type):
    """Metaclass to emit warning when OffloadConfig is accessed via this module."""

    def __getattr__(cls, name):
        if name == "__origin__":  # For type hinting
            return OffloadConfig
        warnings.warn(
            "Importing OffloadConfig from ironcore.offload.config is deprecated. "
            "Use 'from ironcore.config import OffloadConfig' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(OffloadConfig, name)


# Trigger warning on module import, not just class access
warnings.warn(
    "Importing from ironcore.offload.config is deprecated. "
    "Use 'from ironcore.config import OffloadConfig' instead.",
    DeprecationWarning,
    stacklevel=2,
)
