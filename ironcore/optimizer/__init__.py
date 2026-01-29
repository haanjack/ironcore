# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import inspect
from typing import Optional

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ironcore.config import MainConfig
from ironcore.global_vars import get_logger
from ironcore.optimizer.optimizer import AdamWOptimizer


def get_optimizer(config: MainConfig, model, device_type: Optional[str] = None) -> Optimizer:
    """Returns the optimizer."""

    logger = get_logger()

    # optimizer arguments
    max_lr = config.optim.max_lr
    weight_decay = config.optim.weight_decay
    no_decay_on_embedding = config.optim.no_decay_on_embedding

    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn == 'bias' or isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
                no_decay.add(fpn)
            elif pn == 'weight' and isinstance(m, nn.Embedding):
                if no_decay_on_embedding:
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
            else:
                decay.add(fpn)

    param_dict = dict(model.named_parameters())
    optimizer_grouped_parameters = [
        {"params": [param_dict[n] for n in decay], "weight_decay": weight_decay},
        {"params": [param_dict[n] for n in no_decay], "weight_decay": 0.0},
    ]

    fused_available = 'fused' in inspect.signature(AdamW).parameters
    use_fused = fused_available and 'cuda' in device_type
    extra_args = dict(fused=False) if use_fused else dict()

    if config.optim.optimizer == "adam":
        # optimizer = AdamW(
        #     optimizer_grouped_parameters, lr=max_lr, weight_decay=weight_decay,
        #     betas=(config.optim.adam_beta1, config.optim.adam_beta2),
        #     eps=config.optim.adam_eps,
        #     **extra_args
        # )
        optimizer = AdamWOptimizer(
            optimizer_grouped_parameters, lr=max_lr, weight_decay=weight_decay,
            betas=(config.optim.adam_beta1, config.optim.adam_beta2),
            eps=config.optim.adam_eps,
            **extra_args
        )

    else:
        message = f"optimizer {config.optim.optimizer} is not implemented"
        logger.error(message)
        raise NotImplementedError(message)

    return optimizer
