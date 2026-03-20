"""Reference CUDA kernel for aten.repeat — tile tensor along dimensions."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Repeat 2D: input[R,C] → output[R*rr, C*rc]
extern "C" __global__ void aten_repeat_2d(
    const float *input, float *output,
    unsigned int R, unsigned int C, unsigned int rr, unsigned int rc
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int out_rows = R * rr, out_cols = C * rc;
    if (idx >= out_rows * out_cols) return;
    unsigned int r = idx / out_cols, c = idx % out_cols;
    output[idx] = input[(r % R) * C + (c % C)];
}

torch::Tensor aten_repeat_2d_fwd(torch::Tensor input, int64_t rr, int64_t rc) {
    auto ci = input.contiguous();
    int R = ci.size(0), C = ci.size(1);
    auto output = torch::empty({R*(int)rr, C*(int)rc}, ci.options());
    int total = output.numel();
    aten_repeat_2d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), R, C, rr, rc);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_repeat_2d", KERNEL_SRC, ["aten_repeat_2d_fwd"])
    x = torch.randn(8, 16, device="cuda")
    result = ext.aten_repeat_2d_fwd(x, 3, 2)
    expected = aten.repeat.default(x, [3, 2]).contiguous()
    check("aten.repeat", result, expected)
    print("PASS aten.repeat")

if __name__ == "__main__":
    test()
