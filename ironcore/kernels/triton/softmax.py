import torch
import triton
import triton.language as tl

@triton.jit
def forward_kernel(
    x_ptr, output_ptr,
    stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_ptrs = x_ptr + row_start + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    # Online softmax algorithm: max -> subtract max -> exp -> sum -> normalize
    max_val = tl.max(x, axis=0)
    x = tl.exp(x - max_val)
    sum_val = tl.sum(x, axis=0)
    out = x / sum_val

    out_ptrs = output_ptr + row_start + col_offsets
    tl.store(out_ptrs, out, mask=mask)

@triton.jit
def backward_kernel(
    output_ptr, grad_output_ptr, grad_input_ptr,
    stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_ptrs = output_ptr + row_start + col_offsets
    dy_ptrs = grad_output_ptr + row_start + col_offsets

    y = tl.load(y_ptrs, mask=mask)
    dy = tl.load(dy_ptrs, mask=mask)

    # Softmax backward: y * (dy - sum(y * dy))
    s = tl.sum(dy * y, axis=0)
    dx = y * (dy - s)

    dx_ptrs = grad_input_ptr + row_start + col_offsets
    tl.store(dx_ptrs, dx, mask=mask)

class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        original_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1]).contiguous()
        n_rows, n_cols = x_reshaped.shape
        stride = x_reshaped.stride(0)

        output = torch.empty_like(x_reshaped)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)

        forward_kernel[grid](
            x_reshaped, output,
            stride, n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(output)
        ctx.stride = stride
        ctx.n_cols = n_cols
        ctx.original_shape = original_shape

        return output.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        original_shape = ctx.original_shape
        stride = ctx.stride
        n_cols = ctx.n_cols

        grad_reshaped = grad_output.reshape(-1, n_cols).contiguous()
        output_reshaped = output.reshape(-1, n_cols).contiguous()
        n_rows = grad_reshaped.shape[0]

        grad_input = torch.empty_like(grad_reshaped)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)

        backward_kernel[grid](
            output_reshaped, grad_reshaped, grad_input,
            stride, n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return grad_input.reshape(original_shape)

def triton_softmax(x):
    return MyFunction.apply(x)