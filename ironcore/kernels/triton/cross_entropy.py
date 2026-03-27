import torch
import triton
import triton.language as tl


@triton.jit
def forward_kernel(
    logits_ptr,
    labels_ptr,
    loss_ptr,
    n_rows,
    vocab_size,
    ignore_index,
    stride_logits,
    stride_labels,
    stride_loss,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= n_rows:
        return
    
    row_logits_ptr = logits_ptr + pid * stride_logits
    label = tl.load(labels_ptr + pid * stride_labels).to(tl.int32)
    
    if label == ignore_index:
        tl.store(loss_ptr + pid * stride_loss, 0.0)
        return
    
    # Find max for numerical stability
    max_val = -1e30
    for b in range(0, vocab_size, BLOCK_SIZE):
        offs = b + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        x = tl.load(row_logits_ptr + offs, mask=mask).to(tl.float32)
        x_for_max = tl.where(mask, x, -1e30)
        local_max = tl.max(x_for_max, 0)
        max_val = tl.maximum(max_val, local_max)
    
    # Compute sum of exp
    sum_exp = 0.0
    for b in range(0, vocab_size, BLOCK_SIZE):
        offs = b + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        x = tl.load(row_logits_ptr + offs, mask=mask).to(tl.float32)
        x_safe = tl.where(mask, x, max_val)
        exp_x = tl.exp(x_safe - max_val)
        exp_x = tl.where(mask, exp_x, 0.0)
        sum_exp = sum_exp + tl.sum(exp_x, 0)
    
    # Compute loss: log_sum_exp - x[label]
    x_label = tl.load(row_logits_ptr + label).to(tl.float32)
    log_sum_exp = max_val + tl.log(sum_exp)
    loss = log_sum_exp - x_label
    
    tl.store(loss_ptr + pid * stride_loss, loss)


@triton.jit
def backward_kernel(
    grad_output_ptr,
    logits_ptr,
    labels_ptr,
    grad_logits_ptr,
    n_rows,
    vocab_size,
    ignore_index,
    stride_grad_out,
    stride_logits,
    stride_labels,
    stride_grad_logits,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= n_rows:
        return
    
    grad_out = tl.load(grad_output_ptr + pid * stride_grad_out).to(tl.float32)
    label = tl.load(labels_ptr + pid * stride_labels).to(tl.int32)
    
    row_logits_ptr = logits_ptr + pid * stride_logits
    row_grad_ptr = grad_logits_ptr + pid * stride_grad_logits
    
    if label == ignore_index:
        for b in range(0, vocab_size, BLOCK_SIZE):
            offs = b + tl.arange(0, BLOCK_SIZE)
            mask = offs < vocab_size
            tl.store(row_grad_ptr + offs, 0.0, mask=mask)
        return
    
    # Find max for numerical stability
    max_val = -1e30
    for b in range(0, vocab_size, BLOCK_SIZE):
        offs = b + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        x = tl.load(row_logits_ptr + offs, mask=mask).to(tl.float32)
        x_for_max = tl.where(mask, x, -1e30)
        local_max = tl.max(x_for_max, 0)
        max_val = tl.maximum(max_val, local_max)
    
    # Compute sum of exp
    sum_exp = 0.0
    for b in range(0, vocab_size, BLOCK_SIZE):
        offs = b + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        x = tl.load(row_logits_ptr + offs, mask=mask).to(tl.float32)
        x_safe = tl.where(mask, x, max_val)
        exp_x = tl.exp(x_safe - max_val)
        exp_x = tl.where(mask, exp_x, 0.0)
        sum_exp = sum_exp + tl.sum(exp_x, 0)
    
    # Compute and store gradients: (softmax - one_hot) * grad_out
    for b in range(0, vocab_size, BLOCK_SIZE):
        offs = b + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        x = tl.load(row_logits_ptr + offs, mask=mask).to(tl.float32)
        x_safe = tl.where(mask, x, max_val)
        exp_x = tl.exp(x_safe - max_val)
        softmax = exp_x / sum_exp
        softmax = tl.where(mask, softmax, 0.0)
        one_hot = tl.where(offs == label, 1.0, 0.0)
        grad = (softmax - one_hot) * grad_out
        tl.store(row_grad_ptr + offs, grad, mask=mask)


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, ignore_index=-100):
        batch_size, vocab_size = logits.shape
        
        loss = torch.empty(batch_size, dtype=torch.float32, device=logits.device)
        
        BLOCK_SIZE = 1024
        grid = (batch_size,)
        
        forward_kernel[grid](
            logits,
            labels,
            loss,
            batch_size,
            vocab_size,
            ignore_index,
            logits.stride(0),
            labels.stride(0),
            loss.stride(0),
            BLOCK_SIZE,
        )
        
        ctx.save_for_backward(logits, labels)
        ctx.ignore_index = ignore_index
        ctx.vocab_size = vocab_size
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        
        batch_size, vocab_size = logits.shape
        grad_logits = torch.zeros_like(logits)
        
        BLOCK_SIZE = 1024
        grid = (batch_size,)
        
        backward_kernel[grid](
            grad_output,
            logits,
            labels,
            grad_logits,
            batch_size,
            vocab_size,
            ctx.ignore_index,
            grad_output.stride(0),
            logits.stride(0),
            labels.stride(0),
            grad_logits.stride(0),
            BLOCK_SIZE,
        )
        
        return grad_logits, None, None


def triton_cross_entropy(logits, labels, ignore_index=-100):
    return CrossEntropyFunction.apply(logits, labels, ignore_index)