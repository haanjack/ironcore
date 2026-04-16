# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Training tracking setup — configure logging backends."""

import sys
from argparse import Namespace
from pathlib import Path

import yaml

from .utils import load_yaml_config

BACKEND_FIELDS = {
    "tensorboard": {"tensorboard_dir": "./runs/tensorboard"},
    "mlflow": {
        "mlflow_tracking_uri": "file:///tmp/mlflow",
        "mlflow_experiment_name": "ironcore",
    },
    "wandb": {
        "wandb_project": "ironcore",
        "wandb_entity": None,
        "wandb_name": None,
    },
}

BACKEND_PROMPTS = {
    "tensorboard": "Enable TensorBoard logging? [y/n]: ",
    "mlflow": "Enable MLflow logging? [y/n]: ",
    "wandb": "Enable WandB logging? [y/n]: ",
}


def run_track(args: Namespace) -> None:
    """Configure logging backends for a training config.

    Args:
        args: Command-line arguments.
            - config: Path to training config YAML
            - backends: Optional comma-separated backend list
            - --wandb-project, --wandb-entity, --wandb-name
            - --mlflow-uri, --mlflow-experiment
            - --tensorboard-dir
            - --output: Output path for patched config (default: stdout)
    """
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_yaml_config(config_path)

    # Determine which backends to enable
    if args.backends:
        backends = [b.strip().lower() for b in args.backends.split(",")]
    else:
        backends = _prompt_backends()

    if not backends:
        print("No backends selected. Exiting.")
        sys.exit(0)

    # Build utils patch
    utils_patch: dict = config.get("utils", {})

    for backend in backends:
        if backend not in BACKEND_FIELDS:
            print(f"Warning: Unknown backend '{backend}', skipping.")
            continue
        field_defaults = BACKEND_FIELDS[backend]
        for field, default in field_defaults.items():
            # Check CLI flag override first
            cli_value = _get_cli_flag(args, field)
            if cli_value is not None:
                utils_patch[field] = cli_value
            elif field not in utils_patch or utils_patch.get(field) is None:
                # Prompt for backend-specific fields if interactive
                if args.backends is None:
                    prompted = _prompt_backend_fields(backend, field, default)
                    if prompted is not None:
                        utils_patch[field] = prompted
                elif default is not None:
                    utils_patch[field] = default

    # Patch config
    config["utils"] = utils_patch

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Patched config written to: {output_path}")
    else:
        # Print just the utils snippet
        snippet = {"utils": utils_patch}
        print(yaml.dump(snippet, default_flow_style=False, sort_keys=False))

    print(f"\nEnabled backends: {', '.join(backends)}")


def _prompt_backends() -> list[str]:
    """Interactively prompt for backend selection.

    Returns:
        List of selected backend names.
    """
    selected = []
    for backend, prompt in BACKEND_PROMPTS.items():
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            selected.append(backend)
    return selected


def _prompt_backend_fields(backend: str, field: str, default: str | None) -> str | None:
    """Prompt for a backend-specific field value.

    Args:
        backend: Backend name (e.g., 'wandb') — used as prompt prefix.
        field: Field name (e.g., 'wandb_project').
        default: Default value.

    Returns:
        User-provided value or None to skip.
    """
    if default is not None:
        answer = input(f"  [{backend}] {field} [{default}]: ").strip()
    else:
        answer = input(f"  [{backend}] {field}: ").strip()

    if answer == "" and default is not None:
        return default
    return answer if answer else None


def _get_cli_flag(args: Namespace, field: str) -> str | None:
    """Map config field names to CLI argument names and get value.

    Args:
        args: CLI arguments namespace.
        field: Config field name (e.g., 'wandb_project').

    Returns:
        CLI flag value or None if not provided.
    """
    flag_map = {
        "wandb_project": getattr(args, "wandb_project", None),
        "wandb_entity": getattr(args, "wandb_entity", None),
        "wandb_name": getattr(args, "wandb_name", None),
        "mlflow_tracking_uri": getattr(args, "mlflow_uri", None),
        "mlflow_experiment_name": getattr(args, "mlflow_experiment", None),
        "tensorboard_dir": getattr(args, "tensorboard_dir", None),
    }
    return flag_map.get(field)
