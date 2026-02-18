#!/usr/bin/env python3
"""Prepare DPO init checkpoint from a trained SFT checkpoint.

Copies only the model weights (no optimizer state, no LR scheduler) from
the SFT checkpoint to a new directory suitable for DPO initialization.
The new checkpoint is saved as step_0 so that DPO training starts fresh
at step 1 with a clean optimizer.

Usage:
    python scripts/prepare_dpo_init.py \
        --sft-path models/val_sft_gpt2_small \
        --dpo-path models/val_dpo_gpt2_small \
        [--sft-step 3000]   # defaults to latest step
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Prepare DPO init from SFT checkpoint")
    parser.add_argument(
        "--sft-path",
        type=str,
        default="models/val_sft_gpt2_small",
        help="Path to the SFT checkpoint directory",
    )
    parser.add_argument(
        "--dpo-path",
        type=str,
        default="models/val_dpo_gpt2_small",
        help="Path to write the DPO init checkpoint",
    )
    parser.add_argument(
        "--sft-step",
        type=int,
        default=None,
        help="SFT checkpoint step to load (default: latest)",
    )
    args = parser.parse_args()

    sft_dir = Path(args.sft_path)
    dpo_dir = Path(args.dpo_path)

    # Determine SFT step to load
    if args.sft_step is not None:
        sft_step = args.sft_step
    else:
        latest_file = sft_dir / "latest_step.txt"
        if not latest_file.exists():
            raise FileNotFoundError(f"latest_step.txt not found in {sft_dir}")
        sft_step = int(latest_file.read_text().strip())
        print(f"Using latest SFT step: {sft_step}")

    sft_ckpt_path = sft_dir / f"step_{sft_step}" / "pytorch_model.bin"
    if not sft_ckpt_path.exists():
        raise FileNotFoundError(f"SFT checkpoint not found: {sft_ckpt_path}")

    print(f"Loading SFT checkpoint from: {sft_ckpt_path}")
    checkpoint = torch.load(sft_ckpt_path, weights_only=False, map_location="cpu")

    model_state_dict = checkpoint["model_state_dict"]
    print(f"  Loaded {len(model_state_dict)} parameter tensors")

    # Build a minimal step_0 checkpoint with only model weights
    dpo_checkpoint = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "lr_scheduler": {},
        "step": 0,
        "config": checkpoint.get("config"),
        "hf_config": checkpoint.get("hf_config"),
    }

    # Write DPO init checkpoint
    dpo_step0_dir = dpo_dir / "step_0"
    dpo_step0_dir.mkdir(parents=True, exist_ok=True)
    dpo_ckpt_path = dpo_step0_dir / "pytorch_model.bin"

    print(f"Saving DPO init checkpoint to: {dpo_ckpt_path}")
    with open(dpo_ckpt_path, "wb") as f:
        torch.save(dpo_checkpoint, f)

    # Copy HF config.json if it exists
    hf_config_src = sft_dir / "config.json"
    if hf_config_src.exists():
        import shutil
        shutil.copy(hf_config_src, dpo_dir / "config.json")
        print("  Copied config.json")

    # Write latest_step.txt
    latest_step_path = dpo_dir / "latest_step.txt"
    latest_step_path.write_text("0\n")
    print("  Wrote latest_step.txt: 0")

    print(
        f"\nDPO init checkpoint ready at {dpo_dir}\n"
        f"  - Model weights from SFT step {sft_step}\n"
        f"  - Optimizer state: empty (fresh start)\n"
        f"  - LR scheduler: empty (fresh start)\n"
        f"  - Step: 0 (DPO will train from step 1)\n"
    )


if __name__ == "__main__":
    main()
