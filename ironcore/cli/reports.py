# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Report template and generation utilities."""

from pathlib import Path
from typing import Any

from .utils import estimate_params, gather_metadata, load_yaml_config

REPORT_TEMPLATE = """\
# Experiment: {name}

## Metadata
- **Category:** {category}
- **Date:** {date}
- **Commit:** `{commit}`
- **Config:** `{config_path}`

## Objective
{objective}

## Methodology
- **Model:** {model_arch} ({model_params})
- **Hardware:** {hardware}
- **Software:** Python {python_version}, PyTorch {torch_version}, CUDA {cuda_version}
- **Parallelism:** TP={tp_size}, DP={dp_size}, FSDP={fsdp}
- **Hyperparameters:**
  - Optimizer: {optimizer}, LR: {max_lr}, Warmup: {warmup_steps}
  - Batch: micro={micro_batch}, global={global_batch}, accum={grad_accum}
  - Steps: {train_steps}

## Results
### Training Curves
{training_curves}

### Key Metrics
| Metric | Value |
|--------|-------|
| Final Loss | {final_loss} |
| MFU | {mfu} |
| Throughput | {throughput} tokens/s |
| Peak Memory | {peak_memory} GB |

### Comparisons
{comparisons}

## Analysis
{analysis}

## Conclusion
- **Status:** {status}
- **Criteria:** {criteria}
- **Next Steps:** {next_steps}

## Artifacts
- Config: `{config_path}`
- Checkpoints: `{checkpoint_path}`
- Logs: `{log_path}`
- Profiling: `{profile_path}`
"""

SCALING_REPORT_TEMPLATE = """\
# Scaling Analysis: {name}

## Metadata
- **Date:** {date}
- **Commit:** `{commit}`
- **Base Config:** `{config_path}`

## Objective
{objective}

## Methodology
- **Scale Dimension:** {scale_dimension}
- **Scale Points:** {scale_points}
- **Steps per point:** {steps_per_point}
- **Dataset:** {dataset}

## Results
{scaling_table}

### Scaling Law Fit
{scaling_law}

## Analysis
{analysis}

## Conclusion
- **Status:** {status}
- **Observations:** {observations}
"""


def format_report(template: str, **kwargs: Any) -> str:
    """Fill a report template with provided values.

    Missing keys are left as placeholders.

    Args:
        template: Template string with {key} placeholders.
        **kwargs: Key-value pairs to substitute.

    Returns:
        Formatted report string.
    """
    import re

    def replacer(match: re.Match) -> str:
        key = match.group(1)
        return str(kwargs.get(key, match.group(0)))

    return re.sub(r"\{(\w+)\}", replacer, template)


def write_report(content: str, output_path: str | Path) -> Path:
    """Write a report to file, creating directories as needed.

    Args:
        content: Report content as a string.
        output_path: Path to write the report file.

    Returns:
        Path to the written report file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def gather_experiment_metadata(config_path: str | Path | None = None) -> dict[str, Any]:
    """Gather metadata for a report, filling in defaults for missing values.

    Args:
        config_path: Optional path to the training config.

    Returns:
        Dict of metadata suitable for passing to format_report().
    """
    metadata = gather_metadata(config_path)

    result: dict[str, Any] = {
        "name": "unnamed",
        "category": "unspecified",
        "date": metadata["date"],
        "commit": metadata["commit"],
        "config_path": metadata.get("config_path", "[not specified]"),
        "objective": "[TODO]",
        "model_arch": "unknown",
        "model_params": "unknown",
        "hardware": "unknown",
        "python_version": _get_python_version(),
        "torch_version": _get_torch_version(),
        "cuda_version": _get_cuda_version(),
        "tp_size": 1,
        "dp_size": 1,
        "fsdp": False,
        "optimizer": "unknown",
        "max_lr": "unknown",
        "warmup_steps": "unknown",
        "micro_batch": "unknown",
        "global_batch": "unknown",
        "grad_accum": "unknown",
        "train_steps": "unknown",
        "training_curves": "[TODO]",
        "final_loss": "[TODO]",
        "mfu": "[TODO]",
        "throughput": "[TODO]",
        "peak_memory": "[TODO]",
        "comparisons": "[N/A]",
        "analysis": "[TODO]",
        "status": "PENDING",
        "criteria": "[TODO]",
        "next_steps": "[TODO]",
        "checkpoint_path": "[N/A]",
        "log_path": "[N/A]",
        "profile_path": "[N/A]",
    }

    if config_path and Path(config_path).exists():
        config = load_yaml_config(config_path)
        _fill_from_config(result, config)

    return result


def _fill_from_config(result: dict[str, Any], config: dict) -> None:
    """Fill report fields from a parsed config dict.

    Args:
        result: Dict to update with config-derived values.
        config: Config as a dict.
    """
    model = config.get("model", {})
    if isinstance(model, dict):
        result["model_arch"] = model.get("hf_model_type", "unknown")
        result["model_params"] = _estimate_params(model)

    trainer = config.get("trainer", {})
    result["tp_size"] = trainer.get("tensor_model_parallel_size", 1)
    result["micro_batch"] = trainer.get("micro_batch_size", "?")
    result["global_batch"] = trainer.get("train_batch_size", "?")
    result["grad_accum"] = trainer.get("gradient_accumulation_steps", "?")

    parallel = config.get("parallel", {})
    result["fsdp"] = parallel.get("use_fsdp", False)

    optim = config.get("optim", {})
    result["optimizer"] = optim.get("optimizer", "?")
    result["max_lr"] = optim.get("max_lr", "?")
    result["warmup_steps"] = optim.get("warmup_steps", "?")

    operation = config.get("operation", {})
    result["train_steps"] = operation.get("train_steps", "?")

    data = config.get("data", {})
    if isinstance(data, dict):
        result["dataset"] = data.get("train_datasets", [{}])[0].get("dataset_path", "unknown")


def _estimate_params(model_config: dict) -> str:
    """Rough parameter count estimate from model config.

    Args:
        model_config: Model section of config as a dict.

    Returns:
        Human-readable parameter count string.
    """
    try:
        heads = model_config.get("num_attention_heads", 0)
        total = estimate_params(
            d_model=model_config.get("d_model", 0),
            d_ffn=model_config.get("d_ffn", 0),
            layers=model_config.get("num_layers", 0),
            heads=heads,
            head_dim=model_config.get("head_dim", 64),
            groups=model_config.get("num_attention_groups", heads),
            vocab_size=model_config.get("vocab_size", 50257),
        )

        if total >= 1e9:
            return f"{total / 1e9:.1f}B"
        elif total >= 1e6:
            return f"{total / 1e6:.0f}M"
        else:
            return f"{total / 1e3:.0f}K"
    except Exception:
        return "unknown"


def _get_python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return "N/A"


def _get_cuda_version() -> str:
    try:
        import torch

        return torch.version.cuda or "N/A"
    except (ImportError, AttributeError):
        return "N/A"
