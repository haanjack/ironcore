import torch
import triton
import triton.language as tl

@triton.jit
def forward_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, 
    M, N, eps,
    stride_xm, stride_xn, stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    # Compute Mean and Variance
    m = tl.zeros([], dtype=tl.float32)
    m2 = tl.zeros([], dtype=tl.float32)
    
    for off in range(0, N, BLOCK_SIZE):
        mask = off + tl.arange(0, BLOCK_SIZE) < N
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + pid * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        m += tl.sum(x)
        m2 += tl.sum(x * x)
        
    mean = m / N
    var = m2 / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean/rstd for backward
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)
    
    # Compute and store output
    for off in range(0, N, BLOCK_SIZE):
        mask = off + tl.arange(0, BLOCK_SIZE) < N
        cols = off + tl.arange(0, BLOCK_SIZE)
        
        x = tl.load(x_ptr + pid * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask).to(tl.float32)
        
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(y_ptr + pid * stride_ym + cols * stride_yn, y, mask=mask)

@triton.jit
def backward_kernel(
    x_ptr, dy_ptr, w_ptr, dx_ptr, dw_ptr, db_ptr,
    mean_ptr, rstd_ptr,
    M, N,
    stride_xm, stride_xn, stride_dym, stride_dyn,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    # Load mean/rstd for this row
    mean = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)
    
    # Reduction loop 1: compute dhat_sum and xhat_dhat_sum
    dhat_sum = tl.zeros([], dtype=tl.float32)
    xhat_dhat_sum = tl.zeros([], dtype=tl.float32)
    
    for off in range(0, N, BLOCK_SIZE):
        mask = off + tl.arange(0, BLOCK_SIZE) < N
        cols = off + tl.arange(0, BLOCK_SIZE)
        
        x = tl.load(x_ptr + pid * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + pid * stride_dym + cols * stride_dyn, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
        
        x_hat = (x - mean) * rstd
        dhat = dy * w
        
        dhat_sum += tl.sum(dhat)
        xhat_dhat_sum += tl.sum(dhat * x_hat)
    
    mean_dhat = dhat_sum / N
    mean_xhat_dhat = xhat_dhat_sum / N
    
    # Loop 2: compute dx, dw, db
    for off in range(0, N, BLOCK_SIZE):
        mask = off + tl.arange(0, BLOCK_SIZE) < N
        cols = off + tl.arange(0, BLOCK_SIZE)
        
        x = tl.load(x_ptr + pid * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + pid * stride_dym + cols * stride_dyn, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
        
        x_hat = (x - mean) * rstd
        dhat = dy * w
        
        dx = (dhat - mean_dhat - x_hat * mean_xhat_dhat) * rstd
        tl.store(dx_ptr + pid * stride_xm + cols * stride_xn, dx, mask=mask)
        
        # Accumulate gradients for weights
        tl.atomic_add(db_ptr + cols, dy, mask=mask)
        tl.atomic_add(dw_ptr + cols, dy * x_hat, mask=mask)

class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        original_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1]).contiguous()
        M, N = x_reshaped.shape
        
        y_reshaped = torch.empty_like(x_reshaped)
        
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        
        stride_xm = x_reshaped.stride(0)
        stride_xn = x_reshaped.stride(1)
        stride_ym = y_reshaped.stride(0)
        stride_yn = y_reshaped.stride(1)
        
        BLOCK_SIZE = 1024
        
        grid = lambda meta: (M,)
        
        forward_kernel[grid](
            x_reshaped, y_reshaped, weight, bias, mean, rstd,
            M, N, eps,
            stride_xm, stride_xn, stride_ym, stride_yn,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        ctx.save_for_backward(x_reshaped, weight, mean, rstd)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.original_shape = original_shape
        
        return y_reshaped.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        M, N = grad_output_reshaped.shape
        
        x_reshaped, weight, mean, rstd = ctx.saved_tensors
        
        grad_x = torch.empty_like(x_reshaped)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(weight)
        
        stride_xm = x_reshaped.stride(0)
        stride_xn = x_reshaped.stride(1)
        stride_dym = grad_output_reshaped.stride(0)
        stride_dyn = grad_output_reshaped.stride(1)
        
        grid = lambda meta: (M,)
        
        backward_kernel[grid](
            x_reshaped, grad_output_reshaped, weight, grad_x, grad_weight, grad_bias,
            mean, rstd,
            M, N,
            stride_xm, stride_xn, stride_dym, stride_dyn,
            BLOCK_SIZE=ctx.BLOCK_SIZE
        )
        
        return grad_x.reshape(ctx.original_shape), grad_weight, grad_bias, None

def triton_layernorm(x, weight, bias, eps=1e-5):
    return MyFunction.apply(x, weight, bias, eps)