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

"""Test utility functions."""

from __future__ import annotations

import torch


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    msg: str = "",
) -> None:
    """Assert that two tensors are close within tolerance."""
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        max_diff = (actual - expected).abs().max().item()
        mean_diff = (actual - expected).abs().mean().item()
        raise AssertionError(
            f"{msg}\n"
            f"Tensors not close: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}\n"
            f"rtol={rtol}, atol={atol}"
        )


def assert_finite(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert that a tensor contains only finite values."""
    if not torch.isfinite(tensor).all():
        num_nan = torch.isnan(tensor).sum().item()
        num_inf = torch.isinf(tensor).sum().item()
        raise AssertionError(f"{name} contains {num_nan} NaN and {num_inf} Inf values")


def assert_shape(
    tensor: torch.Tensor, expected_shape: tuple[int, ...], name: str = "tensor"
) -> None:
    """Assert that a tensor has the expected shape."""
    if tensor.shape != expected_shape:
        raise AssertionError(f"{name} has shape {tensor.shape}, expected {expected_shape}")


def create_causal_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a causal attention mask."""
    return (
        torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, -1, -1, -1)
    )


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute total gradient norm for a model."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm().item() ** 2
    return total_norm**0.5


def compute_parameter_norm(model: torch.nn.Module) -> float:
    """Compute total parameter norm for a model."""
    total_norm = 0.0
    for param in model.parameters():
        total_norm += param.norm().item() ** 2
    return total_norm**0.5


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_memory_usage(device: torch.device) -> dict[str, float]:
    """Get memory usage in MB."""
    if device.type != "cuda":
        return {"allocated": 0.0, "reserved": 0.0, "peak": 0.0}

    return {
        "allocated": torch.cuda.memory_allocated(device) / (1024**2),
        "reserved": torch.cuda.memory_reserved(device) / (1024**2),
        "peak": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def reset_memory_stats(device: torch.device) -> None:
    """Reset CUDA memory stats."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TensorComparator:
    """Helper class for comparing tensors with detailed reporting."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-5):
        self.rtol = rtol
        self.atol = atol

    def compare(self, actual: torch.Tensor, expected: torch.Tensor, name: str = "") -> dict:
        """Compare two tensors and return detailed statistics."""
        diff = (actual - expected).abs()

        return {
            "name": name,
            "max_diff": diff.max().item(),
            "mean_diff": diff.mean().item(),
            "std_diff": diff.std().item() if diff.numel() > 1 else 0.0,
            "actual_norm": actual.norm().item(),
            "expected_norm": expected.norm().item(),
            "norm_diff": (actual.norm() - expected.norm()).abs().item(),
            "is_close": torch.allclose(actual, expected, rtol=self.rtol, atol=self.atol),
        }

    def assert_close(self, actual: torch.Tensor, expected: torch.Tensor, name: str = "") -> None:
        """Assert tensors are close with detailed error message."""
        stats = self.compare(actual, expected, name)
        if not stats["is_close"]:
            raise AssertionError(
                f"Tensors not close for '{name}':\n"
                f"  max_diff={stats['max_diff']:.2e}\n"
                f"  mean_diff={stats['mean_diff']:.2e}\n"
                f"  norm_diff={stats['norm_diff']:.2e}\n"
                f"  (rtol={self.rtol}, atol={self.atol})"
            )


# =============================================================================
# FIM (Fill-In-the-Middle) Utility Functions
# =============================================================================


def verify_psm_structure(token_ids: list[int], fp_id: int, fs_id: int, fm_id: int) -> bool:
    """
    Verify FIM sequence has correct PSM (Prefix-Suffix-Middle) structure.

    Args:
        token_ids: Token IDs to check
        fp_id: fim_prefix token ID
        fs_id: fim_suffix token ID
        fm_id: fim_middle token ID

    Returns:
        True if structure is valid, False otherwise
    """
    try:
        fp_pos = token_ids.index(fp_id)
        fs_pos = token_ids.index(fs_id)
        fm_pos = token_ids.index(fm_id)

        # Check order: prefix < suffix < middle
        if not (fp_pos < fs_pos < fm_pos):
            return False

        # Check each special token appears exactly once
        if (
            token_ids.count(fp_id) != 1
            or token_ids.count(fs_id) != 1
            or token_ids.count(fm_id) != 1
        ):
            return False

        return True
    except ValueError:
        return False


def has_fim_tokens(token_ids: list[int], fp_id: int, fs_id: int, fm_id: int) -> bool:
    """Check if sequence contains all three FIM special tokens."""
    return fp_id in token_ids and fs_id in token_ids and fm_id in token_ids


def count_fim_samples(token_sequences: list[list[int]], fp_id: int) -> int:
    """Count number of sequences containing FIM tokens."""
    return sum(1 for seq in token_sequences if fp_id in seq)


def calculate_fim_rate(token_sequences: list[list[int]], fp_id: int) -> float:
    """Calculate actual FIM transformation rate."""
    if not token_sequences:
        return 0.0
    return count_fim_samples(token_sequences, fp_id) / len(token_sequences)


def get_section_lengths(
    fim_tokens: list[int], fp_id: int, fs_id: int, fm_id: int
) -> tuple[int, int, int]:
    """Get lengths of prefix, suffix, and middle sections."""
    fp_pos = fim_tokens.index(fp_id)
    fs_pos = fim_tokens.index(fs_id)
    fm_pos = fim_tokens.index(fm_id)

    prefix_len = fs_pos - fp_pos - 1
    suffix_len = fm_pos - fs_pos - 1
    middle_len = len(fim_tokens) - fm_pos - 1

    return (prefix_len, suffix_len, middle_len)


def extract_split_positions(
    fim_tokens: list[int], fp_id: int, fs_id: int, fm_id: int
) -> tuple[int, int]:
    """Reverse engineer original split positions from FIM-transformed sequence."""
    prefix_len, _suffix_len, middle_len = get_section_lengths(fim_tokens, fp_id, fs_id, fm_id)
    return (prefix_len, prefix_len + middle_len)


def reconstruct_original_sequence(
    fim_tokens: list[int], fp_id: int, fs_id: int, fm_id: int
) -> list[int]:
    """Reconstruct original sequence from FIM-transformed tokens."""
    fp_pos = fim_tokens.index(fp_id)
    fs_pos = fim_tokens.index(fs_id)
    fm_pos = fim_tokens.index(fm_id)

    prefix = fim_tokens[fp_pos + 1 : fs_pos]
    suffix = fim_tokens[fs_pos + 1 : fm_pos]
    middle = fim_tokens[fm_pos + 1 :]

    return prefix + middle + suffix


def validate_token_conservation(
    original: list[int], transformed: list[int], fp_id: int, fs_id: int, fm_id: int
) -> bool:
    """Verify that all original tokens are present in transformed sequence."""
    transformed_no_special = [t for t in transformed if t not in [fp_id, fs_id, fm_id]]
    return sorted(original) == sorted(transformed_no_special)
