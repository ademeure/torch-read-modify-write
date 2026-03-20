"""Reference CUDA kernel for scaled dot product attention.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_scaled_dot_product_attention.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
    float max_qk = -1e38f;
    for (unsigned int j = 0; j < S; j++) {
        float qk = 0.0f;
        for (unsigned int k = 0; k < D; k++)
            qk += Q[b*H*S*D + h*S*D + s*D + k] * K[b*H*S*D + h*S*D + j*D + k];
        qk *= scale;
        if (qk > max_qk) max_qk = qk;
    }
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
"""

BB, HH, SS, DD = 1, 2, 8, 16

def init_once():
    Q = torch.randn(BB, HH, SS, DD, device="cuda")
    K = torch.randn(BB, HH, SS, DD, device="cuda")
    V = torch.randn(BB, HH, SS, DD, device="cuda")
    total = BB * HH * SS * DD
    return {
        "kernel_source": KERNEL_SRC, "inputs": [Q, K, V],
        "expected": [torch.nn.functional.scaled_dot_product_attention(Q, K, V).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,), "atol": 1e-3,
    }

def run(inputs, kernel):
    total = BB * HH * SS * DD
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(BB), np.uint32(HH), np.uint32(SS), np.uint32(DD),
    ])]
