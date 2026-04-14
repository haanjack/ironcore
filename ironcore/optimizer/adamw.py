# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.optim import Optimizer

from ironcore.offload.optimizer_helpers import _adamw_offloaded_step, _should_offload_param


class AdamWOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-2,
        amsgrad=False,
        offload_enabled: bool = False,
        offload_min_param_elements: int = 65536,
        **kwargs,
    ):
        # hyperparameters
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        super().__init__(params, defaults)

        self.state_dtype = torch.float32
        self.offload_enabled = offload_enabled
        self.offload_min_param_elements = offload_min_param_elements

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # Offload path: optimizer states live on CPU
                if self.offload_enabled and _should_offload_param(
                    p, self.offload_min_param_elements
                ):
                    _adamw_offloaded_step(
                        p=p,
                        grad=grad,
                        state=state,
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad,
                        state_dtype=self.state_dtype,
                    )
                    continue

                # Standard in-VRAM path
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.state_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.state_dtype)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, dtype=self.state_dtype)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        p.grad.zero_()

    def state_dict(self):
        state_dict = super().state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(lr={self.defaults['lr']}, "
            f"betas={self.defaults['betas']}, "
            f"eps={self.defaults['eps']}, "
            f"weight_decay={self.defaults['weight_decay']}, "
            f"amsgrad={self.defaults['amsgrad']}, "
            f"state_dtype={self.state_dtype}"
        )
