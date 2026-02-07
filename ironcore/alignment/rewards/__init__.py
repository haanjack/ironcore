# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Reward managers for alignment algorithms.

This module contains reward computation components:
- BaseReward: Abstract base class for reward managers
- RuleBasedReward: Mechanical verification (math, code) [planned]
- RemoteReward: External API feedback (Claude, GLM) [planned]
"""

from ironcore.alignment.rewards.base import BaseReward

__all__ = ["BaseReward"]
