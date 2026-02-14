#!/usr/bin/env python
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""
Simple VLA Training Test Script.

A minimal training script to validate the VLA pipeline:
1. Load pre-trained vision encoder (frozen)
2. Train projector + action head
3. Monitor training loss and evaluation metrics
4. Validate the pipeline works end-to-end

Usage:
    # Quick test (CPU/small GPU)
    python train_vla_simple.py --steps 100

    # Full test on GPU
    python train_vla_simple.py --steps 1000

    # With evaluation
    python train_vla_simple.py --steps 1000 --eval-every 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple VLA Training Test")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vla/vla_tiny.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log metrics every N steps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/vla_test",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium"],
        help="Model size: tiny (2 layers, 512 dim), small (8 layers, 1024 dim), medium (16 layers, 2048 dim)",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model_config(size: str = "tiny"):
    """Create config for different model sizes."""
    from ironcore.config import MainConfig
    from ironcore.config.config_data import DataConfig
    from ironcore.config.config_model import ModelConfig
    from ironcore.config.config_optim import OptimConfig
    from ironcore.config.config_parallel import ParallelConfig
    from ironcore.config.config_trainer import InitConfig, OperationConfig, TrainerConfig
    from ironcore.config.config_utils import UtilsConfig
    from ironcore.config.config_vla import (
        ActionConfig,
        FusionConfig,
        ProjectorConfig,
        VisionConfig,
        VLAConfig,
    )

    size_configs = {
        "tiny": {
            "d_model": 512,
            "d_ffn": 1024,
            "num_layers": 2,
            "num_attention_heads": 8,
            "vision_layers": 4,
            "vision_hidden": 1152,
        },
        "small": {
            "d_model": 1024,
            "d_ffn": 4096,
            "num_layers": 8,
            "num_attention_heads": 16,
            "vision_layers": 12,
            "vision_hidden": 1152,
        },
        "medium": {
            "d_model": 2048,
            "d_ffn": 8192,
            "num_layers": 16,
            "num_attention_heads": 32,
            "vision_layers": 24,
            "vision_hidden": 1152,
        },
    }

    cfg = size_configs[size]

    vision_config = VisionConfig(
        encoder_type="siglip",
        model_name="google/siglip-so400m-patch14-384",
        image_size=384,
        patch_size=14,
        hidden_size=cfg["vision_hidden"],
        num_hidden_layers=cfg["vision_layers"],
        num_attention_heads=16,
        intermediate_size=2048,
        freeze_vision=True,
        device="cpu",
    )

    action_config = ActionConfig(
        action_dim=7,
        loss_type="mse",
        prediction_horizon=1,
        action_weight=1.0,
    )

    fusion_config = FusionConfig(
        fusion_type="gated_cross_attention",
        num_layers=2 if size != "tiny" else 1,
    )

    projector_config = ProjectorConfig(
        projector_type="mlp",
        num_layers=2,
        hidden_size=cfg["d_model"],
    )

    vla_config = VLAConfig(
        vision=vision_config,
        action=action_config,
        fusion=fusion_config,
        projector=projector_config,
        image_token_id=32000,
        num_image_tokens=729,
        language_weight=0.0,
    )

    model_config = ModelConfig(
        d_model=cfg["d_model"],
        d_ffn=cfg["d_ffn"],
        num_layers=cfg["num_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_attention_groups=cfg["num_attention_heads"],
        head_dim=cfg["d_model"] // cfg["num_attention_heads"],
    )

    trainer_config = TrainerConfig(
        micro_batch_size=2,
        tensor_model_parallel_size=1,
    )

    init_config = InitConfig(seed=42, init_std=0.02)
    optim_config = OptimConfig(max_lr=1e-4, weight_decay=0.01)
    data_config = DataConfig()
    parallel_config = ParallelConfig()
    operation_config = OperationConfig(
        train_steps=100,
        activation_recompute=False,
    )
    utils_config = UtilsConfig()

    config = MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=init_config,
        optim=optim_config,
        data=data_config,
        parallel=parallel_config,
        operation=operation_config,
        utils=utils_config,
        vla=vla_config,
    )

    return config


class SimpleVLATrainer:
    """Simple trainer for VLA model testing."""

    def __init__(self, config, device: str = "cpu", lr: float = 1e-4):
        self.config = config
        self.device = torch.device(device)
        self.lr = lr

        # Initialize parallel states
        self._init_parallel()

        # Create model
        self._create_model()

        # Create optimizer
        self._create_optimizer()

        # Metrics tracking
        self.train_metrics = {
            "loss": [],
            "action_loss": [],
            "action_mse": [],
        }
        self.eval_metrics = {
            "action_mse": [],
            "success_rate": [],
        }

    def _init_parallel(self):
        from ironcore import global_vars
        from ironcore.parallel import parallel_states

        if global_vars.GLOBAL_STATES is None:
            global_vars.set_global_states(self.config)

        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

    def _create_model(self):
        from ironcore.vla_model import VLAModel

        self.config.vla.vision.device = str(self.device)
        self.model = VLAModel(self.config)
        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen: {total_params - trainable_params:,}")

    def _create_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            print("WARNING: No trainable parameters found!")
            self.optimizer = None
        else:
            self.optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=0.01)

    def create_dummy_batch(self, batch_size: int):
        seq_len = 32
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "pixel_values": torch.randn(batch_size, 3, 384, 384),
            "actions": torch.randn(batch_size, 7),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
        }
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

    @torch.no_grad()
    def evaluate(self, num_batches: int = 10) -> dict:
        self.model.eval()
        total_mse = 0.0
        total_success = 0
        total_samples = 0

        for _ in range(num_batches):
            batch = self.create_dummy_batch(batch_size=2)
            pred_actions = self.model.predict_action(
                batch["input_ids"],
                batch["pixel_values"],
            )
            mse = F.mse_loss(pred_actions, batch["actions"])
            total_mse += mse.item() * pred_actions.shape[0]
            errors = (pred_actions - batch["actions"]).abs().max(dim=1)[0]
            total_success += (errors < 0.1).sum().item()
            total_samples += pred_actions.shape[0]

        return {
            "action_mse": total_mse / total_samples,
            "success_rate": total_success / total_samples * 100,
        }

    def train(self, num_steps: int, eval_every: int = 50, log_every: int = 10):
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Steps: {num_steps}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for step in range(1, num_steps + 1):
            batch = self.create_dummy_batch(batch_size=2)
            metrics = self.train_step(batch)

            self.train_metrics["loss"].append(metrics["loss"])
            self.train_metrics["action_mse"].append(metrics["action_mse"])

            if step % log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                print(
                    f"Step {step}/{num_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Action MSE: {metrics['action_mse']:.4f} | "
                    f"Speed: {steps_per_sec:.2f} steps/s"
                )

            if step % eval_every == 0:
                eval_metrics = self.evaluate(num_batches=5)
                self.eval_metrics["action_mse"].append(eval_metrics["action_mse"])
                self.eval_metrics["success_rate"].append(eval_metrics["success_rate"])
                print(
                    f"  [Eval] Action MSE: {eval_metrics['action_mse']:.4f} | "
                    f"Success Rate: {eval_metrics['success_rate']:.1f}%"
                )

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("Training Complete")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Final loss: {self.train_metrics['loss'][-1]:.4f}")
        print(f"Final action MSE: {self.train_metrics['action_mse'][-1]:.4f}")

        if self.eval_metrics["action_mse"]:
            print(f"Final eval MSE: {self.eval_metrics['action_mse'][-1]:.4f}")
            print(f"Final success rate: {self.eval_metrics['success_rate'][-1]:.1f}%")

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.model(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            labels=batch.get("labels"),
            actions=batch["actions"],
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            pred_actions = self.model.predict_action(
                batch["input_ids"],
                batch["pixel_values"],
            )
            action_mse = F.mse_loss(pred_actions, batch["actions"]).item()

        return {"loss": loss.item(), "action_mse": action_mse}

    def save_metrics(self, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metrics = {
            "train": self.train_metrics,
            "eval": self.eval_metrics,
        }

        metrics_file = output_path / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {metrics_file}")


def main():
    args = parse_args()

    print("="*60)
    print("VLA Training Test")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Model size: {args.model_size}")
    print(f"Device: {args.device}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*60)

    set_seed(args.seed)

    config = create_model_config(args.model_size)
    config.trainer.micro_batch_size = args.batch_size

    trainer = SimpleVLATrainer(
        config=config,
        device=args.device,
        lr=args.lr,
    )

    trainer.train(
        num_steps=args.steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
    )

    trainer.save_metrics(args.output_dir)

    final_mse = trainer.train_metrics["action_mse"][-1] if trainer.train_metrics["action_mse"] else float("inf")

    if final_mse < 2.0:
        print("\n✓ Training test PASSED")
        return 0
    else:
        print("\n✗ Training test FAILED (MSE too high)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
