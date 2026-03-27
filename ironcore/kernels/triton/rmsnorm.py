import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, y_ptr, w_ptr,
    stride, n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * stride
    y_ptr += row_idx * stride

    # Pass 1: Compute sum of squares
    acc = 0.0
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)

    mean_sq = acc / n_cols
    r_factor = tl.rsqrt(mean_sq + eps)

    # Pass 2: Normalize and Scale
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + cols, mask=mask, other=0.0)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        y = x * r_factor * w
        tl.store(y_ptr + cols, y, mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(
    x_ptr, w_ptr, dout_ptr,
    dx_ptr, dw_ptr,
    stride, n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * stride
    dout_ptr += row_idx * stride
    dx_ptr += row_idx * stride

    # Pass 1: Reductions
    # We need sum(x^2) to compute the r_factor again
    # We need sum(dout * x * w) for the chain rule term
    acc_x2 = 0.0
    acc_cross = 0.0

    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dout = tl.load(dout_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        acc_x2 += tl.sum(x * x)
        acc_cross += tl.sum(dout * x * w)

    mean_sq = acc_x2 / n_cols
    r_factor = tl.rsqrt(mean_sq + eps)
    r_factor_cubed = r_factor ** 3

    # Common factor for the derivative of the normalization term
    # d(1/rms)/dx_i = - (1/n) * (1/rms^3) * x_i
    # combined with sum(dout * x * w)
    c = (r_factor_cubed / n_cols) * acc_cross

    # Pass 2: Compute gradients and write
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dout = tl.load(dout_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # dx = dout * w * r_factor - x * c
        # Note: c already includes the cross sum and division by N
        dx = dout * w * r_factor - x * c
        
        dw = dout * x * r_factor

        tl.store(dx_ptr + cols, dx, mask=mask)
        tl.atomic_add(dw_ptr + cols, dw)


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        original_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1]).contiguous()
        M, N = x_reshaped.shape

        BLOCK_SIZE = triton.next_power_of_2(N)
        if BLOCK_SIZE > 8192:
            BLOCK_SIZE = 8192

        grid = (M,)

        output = torch.empty_like(x_reshaped)
        
        rmsnorm_fwd_kernel[grid](
            x_ptr=x_reshaped,
            y_ptr=output,
            w_ptr=weight,
            stride=N,
            n_rows=M,
            n_cols=N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(x_reshaped, weight)
        ctx.original_shape = original_shape
        ctx.eps = eps
        return output.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_reshaped, weight = ctx.saved_tensors
        original_shape = ctx.original_shape
        eps = ctx.eps

        grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        M, N = x_reshaped.shape

        BLOCK_SIZE = triton.next_power_of_2(N)
        if BLOCK_SIZE > 8192:
            BLOCK_SIZE = 8192

        grid = (M,)

        grad_x = torch.empty_like(x_reshaped)
        # Create float32 buffer for weight gradient accumulation
        dw = torch.zeros(N, dtype=torch.float32, device=weight.device)

        rmsnorm_bwd_kernel[grid](
            x_ptr=x_reshaped,
            w_ptr=weight,
            dout_ptr=grad_output_reshaped,
            dx_ptr=grad_x,
            dw_ptr=dw,
            stride=N,
            n_rows=M,
            n_cols=N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # Cast dw back to weight dtype if necessary
        if weight.dtype != torch.float32:
            dw = dw.to(weight.dtype)

        return grad_x.reshape(original_shape), dw, None

def triton_rmsnorm(x, weight, eps=1e-5):
    return RMSNorm.apply(x, weight, eps)