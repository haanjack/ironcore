# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import inspect

from torch import nn
from torch.optim import AdamW, Optimizer

from ironcore.config import MainConfig
from ironcore.global_vars import get_logger
from ironcore.optimizer.distributed_optimizer import DistributedOptimizer
from ironcore.optimizer.muon import MuonOptimizer, is_muon_param
from ironcore.optimizer.adamw import AdamWOptimizer
from ironcore.peft.utils import freeze_base_model


def get_optimizer(config: MainConfig, model, device_type: str | None = None) -> Optimizer:
    """Returns the optimizer."""

    logger = get_logger()

    # Freeze base model parameters if using PEFT
    if config.peft.method != "none":
        logger.info(f"Freezing base model parameters for PEFT method: {config.peft.method}")
        freeze_base_model(model, config.peft.method)

    # optimizer arguments
    max_lr = config.optim.max_lr
    weight_decay = config.optim.weight_decay
    no_decay_on_embedding = config.optim.no_decay_on_embedding

    # Use dict (not set) to maintain insertion order - deterministic across ranks
    decay = {}
    no_decay = {}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            # Skip frozen parameters
            if not p.requires_grad:
                continue

            fpn = f"{mn}.{pn}" if mn else pn

            # LoRA-specific weight decay rules
            if "lora_A" in pn:
                # No weight decay on LoRA A matrices (standard practice)
                no_decay[fpn] = True
            elif pn == "bias" or isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
                no_decay[fpn] = True
            elif pn == "weight" and isinstance(m, nn.Embedding):
                if no_decay_on_embedding:
                    no_decay[fpn] = True
                else:
                    decay[fpn] = True
            else:
                decay[fpn] = True

    param_dict = dict(model.named_parameters())
    decay_params = [param_dict[n] for n in decay if param_dict[n].requires_grad]
    no_decay_params = [param_dict[n] for n in no_decay if param_dict[n].requires_grad]

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100.0 * trainable_params / total_params:.2f}%)"
    )

    fused_available = "fused" in inspect.signature(AdamW).parameters
    use_fused = fused_available and "cuda" in device_type
    extra_args = dict(fused=False) if use_fused else dict()

    if config.optim.optimizer == "adam":
        optimizer = AdamWOptimizer(
            optimizer_grouped_parameters,
            lr=max_lr,
            weight_decay=weight_decay,
            betas=(config.optim.adam_beta1, config.optim.adam_beta2),
            eps=config.optim.adam_eps,
            **extra_args,
        )

    elif config.optim.optimizer == "muon":
        optimizer = get_muon_optimizer(config, model, device_type)

    else:
        message = f"optimizer {config.optim.optimizer} is not implemented"
        logger.error(message)
        raise NotImplementedError(message)

    return optimizer


def get_muon_optimizer(
    config: MainConfig, model, device_type: str | None = None
) -> MuonOptimizer:
    """
    Create Muon optimizer with proper parameter grouping.

    Muon uses a hybrid approach:
    - Newton-Schulz orthogonalization for 2D hidden layer weights
    - AdamW for embeddings, output layers, biases, and norms

    Args:
        config: MainConfig with optimizer settings
        model: The model to optimize
        device_type: Device type for fused operations

    Returns:
        MuonOptimizer instance
    """
    logger = get_logger()

    weight_decay = config.optim.weight_decay
    no_decay_on_embedding = config.optim.no_decay_on_embedding

    # Classify parameters into 4 groups:
    # 1. Muon params with weight decay
    # 2. Muon params without weight decay
    # 3. AdamW params with weight decay
    # 4. AdamW params without weight decay
    muon_decay = []
    muon_no_decay = []
    adamw_decay = []
    adamw_no_decay = []

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            # Skip frozen parameters
            if not p.requires_grad:
                continue

            fpn = f"{mn}.{pn}" if mn else pn

            # Determine if this param should have weight decay
            should_decay = True
            if "lora_A" in pn:
                should_decay = False
            elif pn == "bias" or isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
                should_decay = False
            elif pn == "weight" and isinstance(m, nn.Embedding):
                should_decay = not no_decay_on_embedding

            # Determine if this param should use Muon or AdamW
            use_muon = is_muon_param(fpn, p)

            if use_muon:
                if should_decay:
                    muon_decay.append(p)
                else:
                    muon_no_decay.append(p)
            else:
                if should_decay:
                    adamw_decay.append(p)
                else:
                    adamw_no_decay.append(p)

    # Log parameter distribution
    muon_decay_count = sum(p.numel() for p in muon_decay)
    muon_no_decay_count = sum(p.numel() for p in muon_no_decay)
    adamw_decay_count = sum(p.numel() for p in adamw_decay)
    adamw_no_decay_count = sum(p.numel() for p in adamw_no_decay)
    total = muon_decay_count + muon_no_decay_count + adamw_decay_count + adamw_no_decay_count

    if total > 0:
        logger.info(
            f"Muon optimizer parameter distribution:\n"
            f"  Muon (2D hidden weights): {muon_decay_count + muon_no_decay_count:,} "
            f"({100.0 * (muon_decay_count + muon_no_decay_count) / total:.1f}%)\n"
            f"  AdamW (other params):     {adamw_decay_count + adamw_no_decay_count:,} "
            f"({100.0 * (adamw_decay_count + adamw_no_decay_count) / total:.1f}%)"
        )

    # Calculate learning rates
    muon_lr = config.optim.max_lr * config.optim.muon_lr_scale
    adamw_lr = config.optim.max_lr * config.optim.adamw_lr_scale

    optimizer = MuonOptimizer(
        muon_params=[
            {"params": muon_decay, "weight_decay": weight_decay},
            {"params": muon_no_decay, "weight_decay": 0.0},
        ],
        adamw_params=[
            {"params": adamw_decay, "weight_decay": weight_decay},
            {"params": adamw_no_decay, "weight_decay": 0.0},
        ],
        lr=muon_lr,
        momentum=config.optim.muon_momentum,
        newton_schulz_steps=config.optim.muon_newton_schulz_steps,
        weight_decay=weight_decay,
        nesterov=True,
        adamw_lr=adamw_lr,
        adamw_betas=(config.optim.adam_beta1, config.optim.adam_beta2),
        adamw_eps=config.optim.adam_eps,
        adamw_weight_decay=weight_decay,
    )

    return optimizer
