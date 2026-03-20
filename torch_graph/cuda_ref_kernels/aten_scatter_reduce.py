"""Reference CUDA kernel for aten.scatter_reduce (sum)."""
import torch, numpy as np
KERNEL_SRC = r"""
extern "C" __global__ void aten_scatter_reduce_k(
    const float *self, const long *index, const float *src, float *out,
    unsigned int rows, unsigned int self_cols, unsigned int src_cols, unsigned int total_self
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_self) out[idx] = self[idx];
    if (idx < rows * src_cols) {
        unsigned int r = idx / src_cols;
        atomicAdd(&out[r * self_cols + index[idx]], src[idx]);
    }
}
"""
def init_once():
    x = torch.zeros(8, 32, device="cuda")
    idx = torch.randint(0, 32, (8, 16), device="cuda")
    src = torch.randn(8, 16, device="cuda")
    total = x.numel()
    return {"kernel_source": KERNEL_SRC, "inputs": [x, idx, src],
            "expected": [torch.ops.aten.scatter_reduce.two(x, 1, idx, src, "sum").flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-4}
def run(inputs, kernel):
    total = inputs[0].numel()
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(8), np.uint32(32), np.uint32(16), np.uint32(total)])]
