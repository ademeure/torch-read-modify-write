"""Reference CUDA kernel for aten.empty_strided — allocate with strides."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_empty_strided(float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = 0.0f;
}
"""

def init_once():
    n = 1024
    return {"kernel_source": KERNEL_SRC, "inputs": [],
            "expected": [torch.zeros(n, device="cuda")],
            "outputs": ["float32;n=%d" % n], "grid": ((n + 255) // 256,)}

def run(inputs, kernel):
    return [kernel(params=[kernel.out_ptr(0), np.uint32(1024)])]
