"""Reference CUDA kernel for scaled dot product attention — naive Q@K^T/sqrt(d) @ V."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

// Naive attention: Q @ K^T / sqrt(d), softmax, @ V
// One thread per output element in the final result
extern "C" __global__ void aten_sdpa_kernel(
    const float *Q, const float *K, const float *V, float *output,
    unsigned int B, unsigned int H, unsigned int S, unsigned int D
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = B * H * S * D;
    if (idx >= total) return;

    unsigned int d = idx % D;
    unsigned int s = (idx / D) % S;
    unsigned int h = (idx / (D * S)) % H;
    unsigned int b = idx / (D * S * H);

    float scale = rsqrtf((float)D);

    // Compute softmax(Q[b,h,s,:] @ K[b,h,:,:]^T / sqrt(D)) @ V[b,h,:,d]
    // First: attention weights for row s
    // QK[j] = sum_k Q[s,k] * K[j,k] / sqrt(D)

    // Pass 1: compute max of QK for numerical stability
    float max_qk = -1e38f;
    for (unsigned int j = 0; j < S; j++) {
        float qk = 0.0f;
        for (unsigned int k = 0; k < D; k++)
            qk += Q[b*H*S*D + h*S*D + s*D + k] * K[b*H*S*D + h*S*D + j*D + k];
        qk *= scale;
        if (qk > max_qk) max_qk = qk;
    }

    // Pass 2: exp and sum
    float sum_exp = 0.0f;
    float weighted_v = 0.0f;
    for (unsigned int j = 0; j < S; j++) {
        float qk = 0.0f;
        for (unsigned int k = 0; k < D; k++)
            qk += Q[b*H*S*D + h*S*D + s*D + k] * K[b*H*S*D + h*S*D + j*D + k];
        qk *= scale;
        float w = expf(qk - max_qk);
        sum_exp += w;
        weighted_v += w * V[b*H*S*D + h*S*D + j*D + d];
    }
    output[idx] = weighted_v / sum_exp;
}

torch::Tensor aten_sdpa_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto cq = Q.contiguous(), ck = K.contiguous(), cv = V.contiguous();
    int B = cq.size(0), H = cq.size(1), S = cq.size(2), D = cq.size(3);
    auto output = torch::empty({B, H, S, D}, cq.options());
    int total = B * H * S * D;
    aten_sdpa_kernel<<<(total+255)/256, 256>>>(
        cq.data_ptr<float>(), ck.data_ptr<float>(), cv.data_ptr<float>(),
        output.data_ptr<float>(), B, H, S, D);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_sdpa_kernel", KERNEL_SRC, ["aten_sdpa_fwd"])
    Q = torch.randn(1, 2, 8, 16, device="cuda")
    K = torch.randn(1, 2, 8, 16, device="cuda")
    V = torch.randn(1, 2, 8, 16, device="cuda")
    result = ext.aten_sdpa_fwd(Q, K, V)
    expected = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    check("aten.scaled_dot_product_attention", result, expected, atol=1e-3)
    print("PASS aten.scaled_dot_product_attention")

if __name__ == "__main__":
    test()
