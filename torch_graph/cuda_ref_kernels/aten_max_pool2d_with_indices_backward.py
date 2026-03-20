"""Reference CUDA kernel for aten.max_pool2d_with_indices_backward."""
import torch, numpy as np
KERNEL_SRC = r"""
extern "C" __global__ void aten_maxpool2d_bwd(
    const float *grad_output, const long *indices, float *grad_input,
    unsigned int total_out, unsigned int C, unsigned int H, unsigned int W,
    unsigned int outH, unsigned int outW
) {
    // Zero init + scatter
    unsigned int total_in = (total_out / (outH * outW)) * H * W;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_in) grad_input[idx] = 0.0f;
    __syncthreads();
    if (idx >= total_out) return;
    unsigned int ow = idx % outW, oh = (idx / outW) % outH;
    unsigned int c = (idx / (outW * outH)) % C;
    unsigned int n = idx / (outW * outH * C);
    long flat_idx = indices[idx];
    atomicAdd(&grad_input[n*C*H*W + c*H*W + flat_idx], grad_output[idx]);
}
"""
def init_once():
    x = torch.randn(1, 4, 8, 8, device="cuda")
    out, indices = torch.ops.aten.max_pool2d_with_indices.default(x, [2,2], [2,2])
    grad = torch.randn_like(out)
    result = torch.ops.aten.max_pool2d_with_indices_backward.default(grad, x, [2,2], [2,2], [0,0], [1,1], False, indices)
    total_out = grad.numel()
    total_in = x.numel()
    return {"kernel_source": KERNEL_SRC, "inputs": [grad.contiguous(), indices.contiguous()],
            "expected": [result.flatten()],
            "outputs": ["float32;n=%d" % total_in], "grid": ((max(total_in, total_out) + 255) // 256,), "atol": 1e-5}
def run(inputs, kernel):
    total_out = inputs[0].numel()
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(total_out), np.uint32(4), np.uint32(8), np.uint32(8),
        np.uint32(4), np.uint32(4)])]
