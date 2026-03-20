"""Reference CUDA kernel for aten.nll_loss_forward — negative log likelihood."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_nll_loss_kernel(
    const float *log_probs, const long *target, float *output,
    unsigned int N, unsigned int C
) {
    // Single-thread reduction for simplicity
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < N; i++) {
        long t = target[i];
        sum -= log_probs[i * C + t];
    }
    output[0] = sum / (float)N;
}

torch::Tensor aten_nll_loss_fwd(torch::Tensor log_probs, torch::Tensor target) {
    auto ci = log_probs.contiguous();
    int N = ci.size(0), C = ci.size(1);
    auto output = torch::zeros({}, ci.options());
    aten_nll_loss_kernel<<<1, 1>>>(
        ci.data_ptr<float>(), target.data_ptr<long>(), output.data_ptr<float>(), N, C);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_nll_loss_kernel", KERNEL_SRC, ["aten_nll_loss_fwd"])
    log_probs = torch.randn(16, 10, device="cuda").log_softmax(dim=-1)
    target = torch.randint(0, 10, (16,), device="cuda")
    result = ext.aten_nll_loss_fwd(log_probs, target)
    expected = aten.nll_loss_forward.default(log_probs, target, None, 1, -100)
    check("aten.nll_loss_forward", result, expected[0], atol=1e-4)
    print("PASS aten.nll_loss_forward")

if __name__ == "__main__":
    test()
