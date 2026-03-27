# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import torch


def bytes_to_mib(bytes_value: int):
    return bytes_value // 1024 // 1024


def get_memory_usage(in_mib: bool = False):
    """Get memory usage."""
    summary = {}

    device = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        summary["memory_allocated"] = torch.cuda.memory_allocated(device)
        summary["max_memory_allocated"] = torch.cuda.max_memory_allocated(device)
        summary["memory_reserved"] = torch.cuda.memory_reserved(device)
        summary["max_memory_reserved"] = torch.cuda.max_memory_reserved(device)
    elif torch.backends.mps.is_available():
        summary["memory_allocated"] = torch.mps.current_allocated_memory()
        summary["driver_allocated"] = torch.mps.driver_allocated_memory()

    if in_mib:
        for k, v in summary.items():
            summary[k] = bytes_to_mib(v)

    return summary


def get_detailed_memory_breakdown(model, optimizer=None, in_mib: bool = True) -> dict:
    """
    Get detailed memory breakdown by component type.

    Returns memory usage for:
    - Model parameters (trainable and frozen)
    - Gradients
    - Optimizer states (if optimizer provided)
    - Activations (current CUDA memory minus above)

    Args:
        model: The PyTorch model
        optimizer: Optional optimizer to measure states
        in_mib: If True, return values in MiB instead of bytes

    Returns:
        Dictionary with memory breakdown by component
    """
    breakdown = {}

    # Model parameters
    trainable_params = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )
    frozen_params = sum(
        p.numel() * p.element_size() for p in model.parameters() if not p.requires_grad
    )

    # Gradients (only for trainable params)
    gradients = sum(
        p.grad.numel() * p.grad.element_size()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )

    breakdown["params_trainable"] = trainable_params
    breakdown["params_frozen"] = frozen_params
    breakdown["gradients"] = gradients

    # Optimizer states
    if optimizer is not None:
        opt_state_mem = 0
        state_count = 0
        for state in optimizer.state.values():
            if isinstance(state, dict):
                for v in state.values():
                    if isinstance(v, torch.Tensor):
                        opt_state_mem += v.numel() * v.element_size()
                        state_count += 1
        breakdown["optimizer_states"] = opt_state_mem
        breakdown["optimizer_state_tensors"] = state_count

    # Current total and peak
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        breakdown["cuda_allocated"] = torch.cuda.memory_allocated(device)
        breakdown["cuda_reserved"] = torch.cuda.memory_reserved(device)
        breakdown["cuda_peak_allocated"] = torch.cuda.max_memory_allocated(device)

        # Estimated activations = current allocated - known components
        known_memory = trainable_params + frozen_params + gradients
        if optimizer is not None:
            known_memory += breakdown.get("optimizer_states", 0)
        breakdown["estimated_activations"] = max(0, breakdown["cuda_allocated"] - known_memory)

    if in_mib:
        bytes_keys = [k for k in breakdown.keys() if k != "optimizer_state_tensors"]
        for k in bytes_keys:
            breakdown[k] = bytes_to_mib(breakdown[k])

    return breakdown


def format_memory_report(breakdown: dict, title: str = "Memory Report") -> str:
    """Format memory breakdown as a readable report string."""
    lines = [f"\n{'=' * 50}", f"  {title}", f"{'=' * 50}"]

    # Parameters section
    lines.append("  Model Parameters:")
    if "params_trainable" in breakdown:
        lines.append(f"    Trainable:    {breakdown['params_trainable']:>10.1f} MiB")
    if "params_frozen" in breakdown:
        lines.append(f"    Frozen:       {breakdown['params_frozen']:>10.1f} MiB")

    # Gradients
    if "gradients" in breakdown:
        lines.append(f"  Gradients:       {breakdown['gradients']:>10.1f} MiB")

    # Optimizer states
    if "optimizer_states" in breakdown:
        count = breakdown.get("optimizer_state_tensors", "?")
        lines.append(
            f"  Optimizer States: {breakdown['optimizer_states']:>10.1f} MiB ({count} tensors)"
        )

    # Activations
    if "estimated_activations" in breakdown:
        lines.append(
            f"  Activations:     {breakdown['estimated_activations']:>10.1f} MiB (estimated)"
        )

    lines.append(f"  {'-' * 40}")

    # Totals
    if "cuda_allocated" in breakdown:
        lines.append(f"  Current Allocated: {breakdown['cuda_allocated']:>8.1f} MiB")
    if "cuda_peak_allocated" in breakdown:
        lines.append(f"  Peak Allocated:    {breakdown['cuda_peak_allocated']:>8.1f} MiB")
    if "cuda_reserved" in breakdown:
        lines.append(f"  Reserved:          {breakdown['cuda_reserved']:>8.1f} MiB")

    lines.append(f"{'=' * 50}\n")
    return "\n".join(lines)
