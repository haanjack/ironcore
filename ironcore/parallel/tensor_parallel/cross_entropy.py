# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

#

import torch
import torch.distributed as dist
from typing import Optional

from ironcore.parallel import parallel_states


class _VocabParallelCrossEntropyWorkers(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                vocab_parallel_logits: torch.Tensor,
                labels: torch.Tensor,
                padding_start_idx: Optional[int] = None):
        """
        Compute cross entropy loss when logits are sharped across tensor parallel groups.
        """

        # vocab_parallel_logits: [b, s, v_p], where v_p is this rank's vocab shard (partition size = vocab_size / world_size)
        # labels: [b, s] with token ids in the global vocab range [0, vocab_size)

        # stablize softmax operation
        # find maximum value across all GPUs
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        if parallel_states.get_tensor_model_parallel_world_size() > 1:
            dist.all_reduce(
                logits_max,
                op=dist.ReduceOp.MAX,
                group=parallel_states.get_tensor_model_parallel_group(),
            )
        # then, substract the maximum value
        vocab_parallel_logits = vocab_parallel_logits - \
            logits_max.unsqueeze(dim=-1)

        # get each partition's vocab indices
        partition_vocab_size = vocab_parallel_logits.shape[-1]
        rank = parallel_states.get_tensor_model_parallel_rank()
        world_size = parallel_states.get_tensor_model_parallel_world_size()

        start_idx = partition_vocab_size * rank
        end_idx = partition_vocab_size * (rank + 1)

        # Create mask for tokens that belong to this rank
        target_mask = (labels >= start_idx) & (labels < end_idx)

        # Create labels for this partition
        labels_partition = (labels - start_idx).clone()

        # Mask for padding tokens (global across ranks)
        if padding_start_idx is not None:
            mask_padding = labels >= padding_start_idx
        else:
            mask_padding = None

        # For indexing, ensure labels_partition is in valid range [0, partition_vocab_size)
        # We will zero out the results for these tokens anyway
        labels_partition.clamp_(0, partition_vocab_size - 1)

        # build predicted logits from the mask: [b, s]
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        labels_1d = labels_partition.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)

        predicted_logits_1d = logits_2d[arange_1d, labels_1d].clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(labels)

        # Zero out predicted logits for tokens NOT in this rank or PADDING tokens
        predicted_logits[~target_mask] = 0.0
        if mask_padding is not None:
            predicted_logits[mask_padding] = 0.0

        # All-reduce to get the correct predicted logit on all ranks
        if world_size > 1:
            dist.all_reduce(
                predicted_logits,
                op=dist.ReduceOp.SUM,
                group=parallel_states.get_tensor_model_parallel_group(),
            )

        # sum of exp(logits) along vocab dimension across all GPUs.
        exp_logits = torch.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        if world_size > 1:
            dist.all_reduce(
                sum_exp_logits,
                op=dist.ReduceOp.SUM,
                group=parallel_states.get_tensor_model_parallel_group(),
            )

        # then calculate cross entropy loss: log(sum(exp(x_i))) - x_target
        loss = torch.log(sum_exp_logits) - predicted_logits

        # For padding tokens, set loss to 0
        if mask_padding is not None:
            loss[mask_padding] = 0.0
            # Also ensure sum_exp_logits is 1.0 for padding positions to avoid div by zero/inf in backward
            sum_exp_logits[mask_padding] = 1.0

        # normalize logits for backward pass (softmax)
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        # Save for backward pass
        # Use target_mask to know which rank should subtract 1.0 from softmax
        ctx.save_for_backward(exp_logits, target_mask, labels_partition)
        ctx.mask_padding = mask_padding

        return loss

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Compute the gradient of the cross entropy loss w.r.t. the predicted logits.
        """
        # Retrieve saved tensors
        softmax, target_mask, labels_partition = ctx.saved_tensors
        mask_padding = ctx.mask_padding

        # For cross entropy, gradient is (softmax_p - target)
        grad_input = softmax

        # Create a 2D view for easier indexing
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)
        labels_1d = labels_partition.view(-1)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        # Update gradient: subtract 1.0 from softmax where target matches
        # Only if it's NOT a padding token
        update_mask = target_mask.clone()
        if mask_padding is not None:
            update_mask &= ~mask_padding

        softmax_update = update_mask.view(-1).float()
        grad_2d[arange_1d, labels_1d] -= softmax_update

        # Zero out gradients for padding tokens across entire vocab shard
        if mask_padding is not None:
            mask_padding_2d = mask_padding.view(-1, 1)
            grad_2d.masked_fill_(mask_padding_2d, 0.0)

        # Finally elementwise multiplication with the output (loss) gradients.
        grad_input.mul_(grad_outputs.unsqueeze(dim=-1))

        return grad_input, None, None


def vocab_parallel_cross_entropy(
    vocab_parallel_logits, labels, padding_start_idx: Optional[int] = None
):
    """
    Performs cross entropy loss calculation when logits are sharded across tensor parallel ranks.
    """
    return _VocabParallelCrossEntropyWorkers.apply(
        vocab_parallel_logits, labels, padding_start_idx
    )
