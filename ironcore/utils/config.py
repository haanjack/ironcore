# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import re
from pathlib import Path

import yaml


def _convert_lists_to_dict(data):

    if isinstance(data, list):
        # Only merge if all items are dicts AND none of the items share keys.
        # Lists where items share keys are distinct entries (e.g., reward functions),
        # not a composite config that should be merged.
        if all(isinstance(item, dict) for item in data):
            all_keys = [key for item in data for key in item]
            # Merge only when there are multiple items AND they have no overlapping keys.
            # A single-item list or items sharing keys are distinct entries (e.g., reward
            # functions, dataset splits) that must stay as a list.
            if len(data) > 1 and len(all_keys) == len(set(all_keys)):
                # Multiple dicts, all keys unique — safe to merge into a single dict
                merged_dict = {}
                for item in data:
                    merged_dict.update(item)
                for key, value in merged_dict.items():
                    if isinstance(value, list):
                        value = _convert_lists_to_dict(value)
                    merged_dict[key] = value
                return merged_dict
            else:
                # Single item or overlapping keys — treat as an ordered list of distinct entries
                return [_convert_lists_to_dict(item) for item in data]
        else:
            # List of primitives (e.g., splits: [0.99, 0.01, 0.0]) - return as-is
            return data
    elif isinstance(data, dict):
        return {key: _convert_lists_to_dict(value) for key, value in data.items()}
    else:
        return data


# load environment variables from .env and apply to yaml load
def env_var_constructor(loader, node):
    value = loader.construct_scalar(node)
    pattern = re.compile(r"\$\{(\w+)\}")
    match = pattern.findall(value)
    if match:
        for var in match:
            value = value.replace(f"${{{var}}}", os.getenv(var, ""))
    return value


# apply environment variable in yaml load
yaml.add_implicit_resolver("!env_var", re.compile(r".*\$\{(\w+)\}.*"))
yaml.add_constructor("!env_var", env_var_constructor)


def load_yaml_config(config_path):
    """Load yaml config file."""

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config = _convert_lists_to_dict(config)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {str(e)}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Error loading yaml config: {str(e)}") from e

    return config


def sanitize_path_component(path_component: str) -> str:
    """Sanitize a path component to prevent directory traversal attacks."""
    return os.path.basename(path_component)


def validate_path_within_dir(path: Path, base_dir: Path) -> bool:
    """Validate that a path resolves within the base directory."""
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        return resolved_path.is_relative_to(resolved_base)
    except (OSError, ValueError):
        return False


def deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge two dicts. Override values take precedence.

    Returns a new merged dictionary (does not mutate inputs).
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
