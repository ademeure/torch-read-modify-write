"""Reference CUDA kernel for aten.empty_strided."""
import torch, numpy as np
KERNEL_SRC = r"""
extern "C" __global__ void aten_empty_strided_kernel(float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = 0.0f;
}
"""
def init_once():
    return {"kernel_source": KERNEL_SRC, "inputs": [],
            "expected": [torch.zeros(1024, device="cuda")],
            "outputs": ["float32;n=1024"], "grid": (4,)}
def run(inputs, kernel):
    return [kernel(params=[kernel.out_ptr(0), np.uint32(1024)])]
