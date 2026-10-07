# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Checksum verification and coordinated failures before checkpoint collectives."""

import hashlib
import json

import torch
import torch.distributed as dist


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collective_checkpoint_action(action, device=None):
    error = None
    result = None
    try:
        result = action()
    except Exception as caught:
        error = caught
    if dist.is_initialized():
        if device is None:
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if dist.get_backend() == "nccl"
                else torch.device("cpu")
            )
        invalid = torch.tensor(int(error is not None), device=device)
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        if invalid.item():
            raise RuntimeError(
                "Checkpoint operation failed; all ranks stopped before commit/load"
            ) from error
    elif error is not None:
        raise error
    return result


def verify_manifest(directory, name):
    def verify():
        manifest = json.loads((directory / name).read_text())
        for path, expected in manifest["files"].items():
            if digest_file(directory / path) != expected:
                raise RuntimeError(f"Checkpoint integrity failure: {path}")
        return manifest

    return collective_checkpoint_action(verify)
