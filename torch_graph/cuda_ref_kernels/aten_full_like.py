"""Reference CUDA kernel for aten.full_like."""
import torch, numpy as np
KERNEL_SRC = r"""
extern "C" __global__ void aten_full_like_kernel(float *out0, float value, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = value;
}
"""
def init_once():
    x = torch.randn(1024, device="cuda")
    return {"kernel_source": KERNEL_SRC, "inputs": [x],
            "expected": [torch.full_like(x, 3.14)],
            "outputs": ["float32;n=1024"], "grid": (4,)}
def run(inputs, kernel):
    return [kernel(inputs[0], params=[kernel.out_ptr(0), np.float32(3.14), np.uint32(1024)])]
