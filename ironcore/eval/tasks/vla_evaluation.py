# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""VLA Evaluation tasks for action prediction and success detection.

This module provides evaluation infrastructure for VLA models:
1. Action MSE - Compare predicted actions vs ground truth
2. Text-Conditioned Success Detection - Validate task completion
3. Per-dimension error analysis (position, rotation, gripper)

Future: Integration with SimplerEnv for simulation-based evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

from ironcore.eval.tasks.base_task import Task

if TYPE_CHECKING:
    from ironcore.config import MainConfig


@dataclass
class VLAMetrics:
    """Container for VLA evaluation metrics.

    Attributes:
        action_mse: Mean squared error for all action dimensions
        action_l1: Mean absolute error for all action dimensions
        position_mse: MSE for xyz position (dims 0-2)
        rotation_mse: MSE for rotation (dims 3-5)
        gripper_mse: MSE for gripper state (dim 6)
        max_error: Maximum per-sample error
        success_rate: Percentage of predictions within tolerance
        num_samples: Total number of evaluated samples
    """

    action_mse: float = 0.0
    action_l1: float = 0.0
    position_mse: float = 0.0
    rotation_mse: float = 0.0
    gripper_mse: float = 0.0
    max_error: float = 0.0
    success_rate: float = 0.0
    num_samples: int = 0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "action_mse": self.action_mse,
            "action_l1": self.action_l1,
            "position_mse": self.position_mse,
            "rotation_mse": self.rotation_mse,
            "gripper_mse": self.gripper_mse,
            "max_error": self.max_error,
            "success_rate": self.success_rate,
            "num_samples": self.num_samples,
        }


@dataclass
class SuccessThresholds:
    """Thresholds for success detection.

    Default thresholds based on typical robot manipulation tasks.
    """

    position_tol: float = 0.05  # 5cm position tolerance
    rotation_tol: float = 0.1  # ~5.7 degree rotation tolerance
    gripper_tol: float = 0.1  # 10% gripper state tolerance

    # Combined success requires all dimensions within tolerance
    require_all: bool = True


class VLAEvaluator(Task):
    """Evaluator for VLA action prediction.

    Computes action prediction metrics by comparing model outputs
    against ground truth actions from the dataset.

    Metrics computed:
    - Action MSE (overall and per-component)
    - Action L1 error
    - Position/Rotation/Gripper-specific errors
    - Success rate (predictions within tolerance)

    Example:
        >>> evaluator = VLAEvaluator(
        ...     tokenizer=tokenizer,
        ...     batch_size=4,
        ...     num_samples=100,
        ...     cache_dir="/path/to/data",
        ... )
        >>> result = evaluator.process(model)
        >>> print(f"Action MSE: {result['score']:.4f}")
    """

    def __init__(
        self,
        tokenizer,
        batch_size: int,
        num_samples: int,
        cache_dir: str,
        action_dim: int = 7,
        horizon: int = 1,
        thresholds: SuccessThresholds | None = None,
        config: MainConfig | None = None,
    ):
        """Initialize VLA evaluator.

        Args:
            tokenizer: Tokenizer for text processing
            batch_size: Evaluation batch size
            num_samples: Number of samples to evaluate
            cache_dir: Directory for caching data
            action_dim: Dimension of action vector
            horizon: Prediction horizon (number of future actions)
            thresholds: Success detection thresholds
            config: Main configuration (for VLA dataset creation)
        """
        super().__init__(
            task_name="vla_action",
            split_name="validation",
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_samples=num_samples,
            cache_dir=cache_dir,
        )

        self.action_dim = action_dim
        self.horizon = horizon
        self.thresholds = thresholds or SuccessThresholds()
        self.config = config

        self._metrics = VLAMetrics()

    def _preprocess(self):
        """Prepare evaluation dataset."""
        from ironcore.dataloader.vla_collator import VLACollator
        from ironcore.dataloader.vla_dataset import VLADataset

        if self.config is None:
            raise ValueError("Config required for VLA evaluation")

        # Create validation dataset
        self.dataset = VLADataset(
            config=self.config,
            split="validation",
            num_samples=self.num_samples,
        )

        self.collator = VLACollator(
            tokenizer=self.tokenizer,
            max_length=self.config.model.max_seq_length,
        )

        from torch.utils.data import DataLoader

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=2,
            pin_memory=True,
        )

    def _get_batch(self) -> dict[str, torch.Tensor]:
        """Get next batch from data loader."""
        return next(self._batch_iter)

    def process(self, model) -> dict[str, Any]:
        """Run evaluation on model.

        Args:
            model: VLA model to evaluate

        Returns:
            Dictionary with metric name and score
        """
        self._preprocess()
        self._batch_iter = iter(self.data_loader)

        model.eval()
        device = next(model.parameters()).device

        all_preds = []
        all_targets = []

        num_batches = min(len(self.data_loader), self.num_samples // self.batch_size + 1)

        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="VLA Evaluation", disable=self.rank != 0):
                try:
                    batch = self._get_batch()
                except StopIteration:
                    break

                # Move to device
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values.to(device)

                # Get predictions
                pred_actions = model.predict_action(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                )

                # Get ground truth
                target_actions = batch["actions"].to(device)

                all_preds.append(pred_actions.cpu())
                all_targets.append(target_actions.cpu())

        # Concatenate all predictions
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        self._metrics = self._compute_metrics(all_preds, all_targets)
        self._metrics.num_samples = all_preds.shape[0]

        return self._get_score()

    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> VLAMetrics:
        """Compute evaluation metrics.

        Args:
            predictions: [N, action_dim * horizon] predicted actions
            targets: [N, action_dim * horizon] ground truth actions

        Returns:
            VLAMetrics with computed values
        """
        # Overall errors
        mse = (predictions - targets) ** 2
        l1 = (predictions - targets).abs()

        action_mse = mse.mean().item()
        action_l1 = l1.mean().item()
        max_error = l1.max(dim=1)[0].mean().item()

        # Per-component errors (assuming action_dim=7)
        # [x, y, z, rx, ry, rz, gripper]
        position_mse = mse[:, :3].mean().item() if predictions.shape[1] >= 3 else 0.0
        rotation_mse = mse[:, 3:6].mean().item() if predictions.shape[1] >= 6 else 0.0
        gripper_mse = mse[:, 6:7].mean().item() if predictions.shape[1] >= 7 else 0.0

        # Success rate (within tolerance)
        success = self._compute_success_rate(predictions, targets)

        return VLAMetrics(
            action_mse=action_mse,
            action_l1=action_l1,
            position_mse=position_mse,
            rotation_mse=rotation_mse,
            gripper_mse=gripper_mse,
            max_error=max_error,
            success_rate=success,
            num_samples=predictions.shape[0],
        )

    def _compute_success_rate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute success rate based on tolerances.

        Args:
            predictions: Predicted actions
            targets: Ground truth actions

        Returns:
            Percentage of predictions within tolerance
        """
        errors = (predictions - targets).abs()

        if predictions.shape[1] >= 3:
            # Position error (first 3 dims)
            pos_errors = errors[:, :3].max(dim=1)[0]
            pos_success = pos_errors < self.thresholds.position_tol
        else:
            pos_success = torch.ones(predictions.shape[0], dtype=torch.bool)

        if predictions.shape[1] >= 6:
            # Rotation error (dims 3-5)
            rot_errors = errors[:, 3:6].max(dim=1)[0]
            rot_success = rot_errors < self.thresholds.rotation_tol
        else:
            rot_success = torch.ones(predictions.shape[0], dtype=torch.bool)

        if predictions.shape[1] >= 7:
            # Gripper error (dim 6)
            grip_errors = errors[:, 6]
            grip_success = grip_errors < self.thresholds.gripper_tol
        else:
            grip_success = torch.ones(predictions.shape[0], dtype=torch.bool)

        if self.thresholds.require_all:
            success = pos_success & rot_success & grip_success
        else:
            success = pos_success | rot_success | grip_success

        return success.float().mean().item() * 100

    def _get_score(self) -> dict[str, Any]:
        """Return evaluation score."""
        return {
            "metric": "action_mse",
            "score": self._metrics.action_mse,
            "metrics": self._metrics.to_dict(),
        }


class TextConditionedSuccessEvaluator(Task):
    """Evaluator for text-conditioned success detection.

    Evaluates whether the model correctly predicts task success/failure
    based on text instructions and visual observations.

    This is a binary classification task:
    - Given instruction + image + action
    - Predict whether the action leads to successful task completion

    Future: Integrate with simulation (SimplerEnv) for ground truth success.
    """

    def __init__(
        self,
        tokenizer,
        batch_size: int,
        num_samples: int,
        cache_dir: str,
        success_threshold: float = 0.5,
        config: MainConfig | None = None,
    ):
        """Initialize success detection evaluator.

        Args:
            tokenizer: Tokenizer for text processing
            batch_size: Evaluation batch size
            num_samples: Number of samples to evaluate
            cache_dir: Directory for caching data
            success_threshold: Threshold for success probability
            config: Main configuration
        """
        super().__init__(
            task_name="vla_success",
            split_name="validation",
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_samples=num_samples,
            cache_dir=cache_dir,
        )

        self.success_threshold = success_threshold
        self.config = config

        # Metrics
        self._accuracy = 0.0
        self._precision = 0.0
        self._recall = 0.0
        self._f1 = 0.0

    def _preprocess(self):
        """Prepare evaluation dataset."""
        # Similar to VLAEvaluator but with success labels
        # For now, use action error as proxy for success
        pass

    def _get_batch(self) -> dict[str, torch.Tensor]:
        """Get next batch."""
        return next(self._batch_iter)

    def process(self, model) -> dict[str, Any]:
        """Run success detection evaluation.

        Args:
            model: VLA model

        Returns:
            Dictionary with success detection metrics
        """
        # This evaluator requires a dataset with success labels
        # For initial implementation, use action MSE threshold as proxy

        if self.config is None:
            return {"metric": "success_accuracy", "score": 0.0}

        self._preprocess()
        self._batch_iter = iter(self.data_loader)

        model.eval()
        device = next(model.parameters()).device

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Success Detection", disable=self.rank != 0):
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values.to(device)

                # Predict actions
                pred_actions = model.predict_action(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                )

                # Compare with ground truth
                target_actions = batch["actions"].to(device)
                errors = (pred_actions - target_actions).abs().mean(dim=1)

                # Convert to success prediction (low error = success)
                pred_success = (errors < self.success_threshold).float()

                # Ground truth (from dataset or derived)
                if "success" in batch:
                    true_success = batch["success"].to(device).float()
                else:
                    # Use same threshold for proxy labels
                    true_errors = torch.zeros_like(errors)  # Placeholder
                    true_success = (true_errors < self.success_threshold).float()

                all_preds.append(pred_success.cpu())
                all_labels.append(true_success.cpu())

        # Compute classification metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        self._compute_classification_metrics(all_preds, all_labels)

        return self._get_score()

    def _compute_classification_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute precision, recall, F1."""
        predictions = (predictions > 0.5).float()

        tp = ((predictions == 1) & (labels == 1)).sum().float()
        fp = ((predictions == 1) & (labels == 0)).sum().float()
        fn = ((predictions == 0) & (labels == 1)).sum().float()
        tn = ((predictions == 0) & (labels == 0)).sum().float()

        self._accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        self._precision = tp / (tp + fp + 1e-8)
        self._recall = tp / (tp + fn + 1e-8)
        self._f1 = 2 * self._precision * self._recall / (self._precision + self._recall + 1e-8)

    def _get_score(self) -> dict[str, Any]:
        """Return success detection score."""
        return {
            "metric": "success_accuracy",
            "score": self._accuracy.item(),
            "metrics": {
                "accuracy": self._accuracy.item(),
                "precision": self._precision.item(),
                "recall": self._recall.item(),
                "f1": self._f1.item(),
            },
        }


class VLAMetricLogger:
    """Logger for VLA metrics during training and evaluation.

    Collects and logs VLA-specific metrics:
    - Action prediction errors
    - Success rates
    - Per-dimension breakdowns
    - Temporal consistency (for action chunking)

    Example:
        >>> logger = VLAMetricLogger()
        >>> for batch in val_loader:
        ...     pred = model.predict_action(...)
        ...     logger.update(pred, batch["actions"])
        >>> logger.log_metrics(step=1000)
        >>> logger.reset()
    """

    def __init__(
        self,
        action_dim: int = 7,
        horizon: int = 1,
        thresholds: SuccessThresholds | None = None,
    ):
        """Initialize metric logger.

        Args:
            action_dim: Action vector dimension
            horizon: Prediction horizon
            thresholds: Success thresholds
        """
        self.action_dim = action_dim
        self.horizon = horizon
        self.thresholds = thresholds or SuccessThresholds()

        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self._mse_sum = 0.0
        self._l1_sum = 0.0
        self._position_mse_sum = 0.0
        self._rotation_mse_sum = 0.0
        self._gripper_mse_sum = 0.0
        self._success_count = 0
        self._num_samples = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """Update metrics with new predictions.

        Args:
            predictions: [B, action_dim * horizon] predictions
            targets: [B, action_dim * horizon] ground truth
        """
        batch_size = predictions.shape[0]

        mse = (predictions - targets) ** 2
        l1 = (predictions - targets).abs()

        self._mse_sum += mse.sum().item()
        self._l1_sum += l1.sum().item()

        if predictions.shape[1] >= 3:
            self._position_mse_sum += mse[:, :3].sum().item()
        if predictions.shape[1] >= 6:
            self._rotation_mse_sum += mse[:, 3:6].sum().item()
        if predictions.shape[1] >= 7:
            self._gripper_mse_sum += mse[:, 6:7].sum().item()

        # Count successes
        max_errors = l1.max(dim=1)[0]
        self._success_count += (max_errors < self.thresholds.position_tol).sum().item()

        self._num_samples += batch_size

    def compute(self) -> VLAMetrics:
        """Compute final metrics."""
        if self._num_samples == 0:
            return VLAMetrics()

        total_dims = self._num_samples * self.action_dim * self.horizon

        return VLAMetrics(
            action_mse=self._mse_sum / total_dims,
            action_l1=self._l1_sum / total_dims,
            position_mse=self._position_mse_sum / (self._num_samples * 3) if self.action_dim >= 3 else 0.0,
            rotation_mse=self._rotation_mse_sum / (self._num_samples * 3) if self.action_dim >= 6 else 0.0,
            gripper_mse=self._gripper_mse_sum / self._num_samples if self.action_dim >= 7 else 0.0,
            success_rate=self._success_count / self._num_samples * 100,
            num_samples=self._num_samples,
        )

    def log_metrics(self, step: int, prefix: str = "val"):
        """Log metrics to TensorBoard/MLFlow.

        Args:
            step: Current training step
            prefix: Metric name prefix (e.g., "train", "val")
        """
        from ironcore.global_vars import log_metric

        metrics = self.compute()

        log_metric(f"{prefix}/action_mse", metrics.action_mse, step)
        log_metric(f"{prefix}/action_l1", metrics.action_l1, step)
        log_metric(f"{prefix}/position_mse", metrics.position_mse, step)
        log_metric(f"{prefix}/rotation_mse", metrics.rotation_mse, step)
        log_metric(f"{prefix}/gripper_mse", metrics.gripper_mse, step)
        log_metric(f"{prefix}/success_rate", metrics.success_rate, step)


# Factory function for creating evaluators
def get_vla_evaluators(
    tokenizer,
    config: MainConfig,
) -> list[Task]:
    """Create VLA evaluators based on configuration.

    Args:
        tokenizer: Tokenizer instance
        config: Main configuration

    Returns:
        List of evaluator instances
    """
    evaluators = []

    # Action prediction evaluator
    action_evaluator = VLAEvaluator(
        tokenizer=tokenizer,
        batch_size=config.trainer.eval_batch_size or config.trainer.micro_batch_size,
        num_samples=config.operation.eval_samples,
        cache_dir=config.data.data_path,
        action_dim=config.vla.action.action_dim,
        horizon=config.vla.action.prediction_horizon,
        config=config,
    )
    evaluators.append(action_evaluator)

    # Success detection evaluator (optional)
    if hasattr(config.vla, "eval_success") and config.vla.eval_success:
        success_evaluator = TextConditionedSuccessEvaluator(
            tokenizer=tokenizer,
            batch_size=config.trainer.eval_batch_size or config.trainer.micro_batch_size,
            num_samples=config.operation.eval_samples,
            cache_dir=config.data.data_path,
            config=config,
        )
        evaluators.append(success_evaluator)

    return evaluators
