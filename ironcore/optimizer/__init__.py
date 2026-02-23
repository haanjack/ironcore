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

import inspect

from torch import nn
from torch.optim import AdamW, Optimizer

from ironcore.config import MainConfig
from ironcore.global_vars import get_logger
from ironcore.optimizer.optimizer import AdamWOptimizer
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

    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            # Skip frozen parameters
            if not p.requires_grad:
                continue

            fpn = f"{mn}.{pn}" if mn else pn

            # LoRA-specific weight decay rules
            if "lora_A" in pn:
                # No weight decay on LoRA A matrices (standard practice)
                no_decay.add(fpn)
            elif pn == "bias" or isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
                no_decay.add(fpn)
            elif pn == "weight" and isinstance(m, nn.Embedding):
                if no_decay_on_embedding:
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
            else:
                decay.add(fpn)

    param_dict = dict(model.named_parameters())
    # Filter out frozen parameters
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
        # optimizer = AdamW(
        #     optimizer_grouped_parameters, lr=max_lr, weight_decay=weight_decay,
        #     betas=(config.optim.adam_beta1, config.optim.adam_beta2),
        #     eps=config.optim.adam_eps,
        #     **extra_args
        # )
        optimizer = AdamWOptimizer(
            optimizer_grouped_parameters,
            lr=max_lr,
            weight_decay=weight_decay,
            betas=(config.optim.adam_beta1, config.optim.adam_beta2),
            eps=config.optim.adam_eps,
            **extra_args,
        )

    else:
        message = f"optimizer {config.optim.optimizer} is not implemented"
        logger.error(message)
        raise NotImplementedError(message)

    return optimizer
