# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import math
import torch
from torch.optim import Optimizer
from torch.optim.adamw import AdamW

class AdamWOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08,
                 weight_decay=1e-2, amsgrad=False, **kwargs):
        # hyperparameters
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        super().__init__(params, defaults)

        self.state_dtype = torch.float32

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, dtype=self.state_dtype)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=self.state_dtype)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, dtype=self.state_dtype)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                if weight_decay != 0:
                    # Apply decoupled weight decay (AdamW style)
                    p.data.mul_(1 - lr * weight_decay)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    p.grad.zero_()
        return self

    def state_dict(self):
        state_dict = super().state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        return self

    def __repr__(self):
        return (f'{self.__class__.__name__}(lr={self.defaults["lr"]}, '
                f'betas={self.defaults["betas"]}, '
                f'eps={self.defaults["eps"]}, '
                f'weight_decay={self.defaults["weight_decay"]}, '
                f'amsgrad={self.defaults["amsgrad"]}, '
                f'state_dtype={self.state_dtype}')
