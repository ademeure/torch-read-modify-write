"""Reference CUDA kernel for aten._native_batch_norm_legit."""
import torch, numpy as np
KERNEL_SRC = r"""
extern "C" __global__ void aten_bn_legit(
    const float *input, const float *weight, const float *bias,
    const float *mean, const float *var, float *output,
    unsigned int N, unsigned int C, unsigned int HW, float eps
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * HW) return;
    unsigned int c = (idx / HW) % C;
    output[idx] = (input[idx] - mean[c]) * rsqrtf(var[c] + eps) * weight[c] + bias[c];
}
"""
def init_once():
    x = torch.randn(2, 8, 4, 4, device="cuda")
    w, b = torch.randn(8, device="cuda"), torch.randn(8, device="cuda")
    rm, rv = torch.randn(8, device="cuda"), torch.rand(8, device="cuda") + 0.1
    total = x.numel()
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous(), w, b, rm, rv],
            "expected": [torch.ops.aten._native_batch_norm_legit.default(x, w, b, rm, rv, False, 0.1, 1e-5)[0].flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-4}
def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.in_ptr(3), kernel.in_ptr(4),
        kernel.out_ptr(0), np.uint32(2), np.uint32(8), np.uint32(16), np.float32(1e-5)])]
