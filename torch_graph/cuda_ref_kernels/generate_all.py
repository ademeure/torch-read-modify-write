#!/usr/bin/env python3
"""Generate reference CUDA kernel files for every aten op.

Each generated file is self-contained: kernel source + wrapper + test function.
Run: python torch_graph/cuda_ref_kernels/generate_all.py
Test one: python -c "from torch_graph.cuda_ref_kernels.aten_add import test; test()"
Test all: python torch_graph/cuda_ref_kernels/run_all_tests.py
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Template for elementwise ops (1 input → 1 output)
# ─────────────────────────────────────────────────────────────────────────────

UNARY_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name}."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void {func_name}(const float *in0, float *out0, unsigned int n) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        float x = in0[i];
        out0[i] = {cuda_expr};
    }}
}}

torch::Tensor {func_name}_fwd(torch::Tensor in0) {{
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    {func_name}<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}}
"""

def test():
    ext = compile_cuda("{func_name}", KERNEL_SRC, ["{func_name}_fwd"])
    x = {test_input}
    result = ext.{func_name}_fwd(x)
    expected = {torch_ref}
    check("aten.{op_name}", result, expected{atol_str})
    print(f"PASS aten.{op_name}")

if __name__ == "__main__":
    test()
'''

# ─────────────────────────────────────────────────────────────────────────────
# Template for binary ops (2 inputs → 1 output)
# ─────────────────────────────────────────────────────────────────────────────

BINARY_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name}."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void {func_name}(const float *in0, const float *in1, float *out0, unsigned int n) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        float a = in0[i];
        float b = in1[i];
        out0[i] = {cuda_expr};
    }}
}}

torch::Tensor {func_name}_fwd(torch::Tensor in0, torch::Tensor in1) {{
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    {func_name}<<<(n+255)/256, 256>>>(in0.data_ptr<float>(), in1.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}}
"""

def test():
    ext = compile_cuda("{func_name}", KERNEL_SRC, ["{func_name}_fwd"])
    {test_setup}
    result = ext.{func_name}_fwd(a, b)
    expected = {torch_ref}
    check("aten.{op_name}", result, expected{atol_str})
    print(f"PASS aten.{op_name}")

if __name__ == "__main__":
    test()
'''

# ─────────────────────────────────────────────────────────────────────────────
# Op definitions
# ─────────────────────────────────────────────────────────────────────────────

UNARY_OPS = [
    # (op_name, func_name, cuda_expr, test_input, torch_ref, atol)
    ("abs", "aten_abs", "fabsf(x)", "torch.randn(1024, device='cuda')", "x.abs()", None),
    ("neg", "aten_neg", "(-x)", "torch.randn(1024, device='cuda')", "(-x)", None),
    ("exp", "aten_exp", "expf(x)", "torch.randn(1024, device='cuda')", "x.exp()", 1e-5),
    ("exp2", "aten_exp2", "exp2f(x)", "torch.randn(1024, device='cuda')", "x.exp2()", 1e-5),
    ("expm1", "aten_expm1", "expm1f(x)", "torch.randn(1024, device='cuda')", "x.expm1()", 1e-5),
    ("log", "aten_log", "logf(x)", "torch.rand(1024, device='cuda') + 0.01", "x.log()", 1e-5),
    ("log2", "aten_log2", "log2f(x)", "torch.rand(1024, device='cuda') + 0.01", "x.log2()", 1e-5),
    ("log10", "aten_log10", "log10f(x)", "torch.rand(1024, device='cuda') + 0.01", "x.log10()", 1e-5),
    ("log1p", "aten_log1p", "log1pf(x)", "torch.rand(1024, device='cuda')", "x.log1p()", 1e-5),
    ("sqrt", "aten_sqrt", "sqrtf(x)", "torch.rand(1024, device='cuda') + 0.01", "x.sqrt()", 1e-6),
    ("rsqrt", "aten_rsqrt", "rsqrtf(x)", "torch.rand(1024, device='cuda') + 0.01", "x.rsqrt()", 1e-4),
    ("ceil", "aten_ceil", "ceilf(x)", "torch.randn(1024, device='cuda') * 10", "x.ceil()", None),
    ("floor", "aten_floor", "floorf(x)", "torch.randn(1024, device='cuda') * 10", "x.floor()", None),
    ("round", "aten_round", "nearbyintf(x)", "torch.randn(1024, device='cuda') * 10", "x.round()", None),
    ("trunc", "aten_trunc", "truncf(x)", "torch.randn(1024, device='cuda') * 10", "x.trunc()", None),
    ("sign", "aten_sign", "((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))", "torch.randn(1024, device='cuda')", "x.sign()", None),
    ("sin", "aten_sin", "sinf(x)", "torch.randn(1024, device='cuda')", "x.sin()", 1e-5),
    ("cos", "aten_cos", "cosf(x)", "torch.randn(1024, device='cuda')", "x.cos()", 1e-5),
    ("tan", "aten_tan", "tanf(x)", "torch.randn(1024, device='cuda') * 0.5", "x.tan()", 1e-4),
    ("asin", "aten_asin", "asinf(x)", "torch.rand(1024, device='cuda') * 1.98 - 0.99", "x.asin()", 1e-5),
    ("acos", "aten_acos", "acosf(x)", "torch.rand(1024, device='cuda') * 1.98 - 0.99", "x.acos()", 1e-5),
    ("atan", "aten_atan", "atanf(x)", "torch.randn(1024, device='cuda')", "x.atan()", 1e-5),
    ("sinh", "aten_sinh", "sinhf(x)", "torch.randn(1024, device='cuda')", "x.sinh()", 1e-4),
    ("cosh", "aten_cosh", "coshf(x)", "torch.randn(1024, device='cuda')", "x.cosh()", 1e-4),
    ("tanh", "aten_tanh", "tanhf(x)", "torch.randn(1024, device='cuda')", "x.tanh()", 1e-6),
    ("asinh", "aten_asinh", "asinhf(x)", "torch.randn(1024, device='cuda')", "x.asinh()", 1e-5),
    ("acosh", "aten_acosh", "acoshf(x)", "torch.rand(1024, device='cuda') + 1.01", "x.acosh()", 1e-5),
    ("atanh", "aten_atanh", "atanhf(x)", "torch.rand(1024, device='cuda') * 1.98 - 0.99", "x.atanh()", 1e-5),
    ("erf", "aten_erf", "erff(x)", "torch.randn(1024, device='cuda')", "x.erf()", 1e-5),
    ("erfc", "aten_erfc", "erfcf(x)", "torch.randn(1024, device='cuda')", "x.erfc()", 1e-5),
    ("reciprocal", "aten_reciprocal", "(1.0f / x)", "torch.randn(1024, device='cuda').abs() + 0.1", "x.reciprocal()", 1e-5),
    ("sigmoid", "aten_sigmoid", "(1.0f / (1.0f + expf(-x)))", "torch.randn(1024, device='cuda')", "x.sigmoid()", 1e-5),
    ("relu", "aten_relu", "((x > 0.0f) ? x : 0.0f)", "torch.randn(1024, device='cuda')", "x.relu()", None),
    ("gelu", "aten_gelu", "(x * 0.5f * (1.0f + erff(x * 0.7071067811865476f)))", "torch.randn(1024, device='cuda')", "torch.nn.functional.gelu(x)", 1e-5),
    ("silu", "aten_silu", "(x / (1.0f + expf(-x)))", "torch.randn(1024, device='cuda')", "torch.nn.functional.silu(x)", 1e-5),
    ("hardswish", "aten_hardswish", "(x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f)", "torch.randn(1024, device='cuda') * 5", "torch.nn.functional.hardswish(x)", 1e-5),
    ("hardsigmoid", "aten_hardsigmoid", "fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f)", "torch.randn(1024, device='cuda') * 5", "torch.nn.functional.hardsigmoid(x)", 1e-5),
    ("softplus", "aten_softplus", "((x > 20.0f) ? x : logf(1.0f + expf(x)))", "torch.randn(1024, device='cuda') * 5", "torch.nn.functional.softplus(x)", 1e-4),
    ("mish", "aten_mish", "(x * tanhf((x > 20.0f) ? x : logf(1.0f + expf(x))))", "torch.randn(1024, device='cuda')", "torch.nn.functional.mish(x)", 1e-4),
    ("elu", "aten_elu", "((x > 0.0f) ? x : (expf(x) - 1.0f))", "torch.randn(1024, device='cuda')", "torch.nn.functional.elu(x)", 1e-5),
    ("leaky_relu", "aten_leaky_relu", "((x > 0.0f) ? x : 0.01f * x)", "torch.randn(1024, device='cuda')", "torch.nn.functional.leaky_relu(x, 0.01)", 1e-6),
    ("frac", "aten_frac", "(x - truncf(x))", "torch.randn(1024, device='cuda') * 10", "x.frac()", 1e-5),
]

BINARY_OPS = [
    # (op_name, func_name, cuda_expr, test_setup, torch_ref, atol)
    ("add", "aten_add", "(a + b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "a + b", None),
    ("sub", "aten_sub", "(a - b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "a - b", None),
    ("mul", "aten_mul", "(a * b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "a * b", None),
    ("div", "aten_div", "(a / b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda').abs() + 0.1",
     "a / b", 1e-5),
    ("pow", "aten_pow", "powf(a, b)",
     "a = torch.rand(1024, device='cuda') + 0.1\n    b = torch.rand(1024, device='cuda') * 3",
     "a.pow(b)", 1e-4),
    ("maximum", "aten_maximum", "fmaxf(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.maximum(a, b)", None),
    ("minimum", "aten_minimum", "fminf(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.minimum(a, b)", None),
    ("atan2", "aten_atan2", "atan2f(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.atan2(a, b)", 1e-5),
    ("fmod", "aten_fmod", "fmodf(a, b)",
     "a = torch.randn(1024, device='cuda') * 10\n    b = torch.randn(1024, device='cuda').abs() + 0.5",
     "torch.fmod(a, b)", 1e-5),
    ("remainder", "aten_remainder", "(a - b * floorf(a / b))",
     "a = torch.randn(1024, device='cuda') * 10\n    b = torch.randn(1024, device='cuda').abs() + 0.5",
     "torch.remainder(a, b)", 1e-4),
    ("hypot", "aten_hypot", "hypotf(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.hypot(a, b)", 1e-5),
    ("copysign", "aten_copysign", "copysignf(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.copysign(a, b)", None),
]


# ─────────────────────────────────────────────────────────────────────────────
# Matmul (nested loops — intentionally slow reference)
# ─────────────────────────────────────────────────────────────────────────────

MATMUL_FILE = '''"""Reference CUDA kernel for aten.mm — triple nested loop, not optimized."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// One thread per output element. O(M*N*K) total work, no tiling, no shared mem.
extern "C" __global__ void aten_mm(
    const float *A, const float *B, float *C,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor aten_mm_fwd(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    aten_mm<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                  C.data_ptr<float>(), M, K, N);
    return C;
}
"""

def test():
    ext = compile_cuda("aten_mm", KERNEL_SRC, ["aten_mm_fwd"])
    A = torch.randn(64, 32, device="cuda")
    B = torch.randn(32, 48, device="cuda")
    result = ext.aten_mm_fwd(A.contiguous(), B.contiguous())
    expected = torch.mm(A, B)
    check("aten.mm", result, expected, atol=1e-3)
    print("PASS aten.mm")

if __name__ == "__main__":
    test()
'''

BMM_FILE = '''"""Reference CUDA kernel for aten.bmm — batched matmul, nested loops."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_bmm(
    const float *A, const float *B, float *C,
    unsigned int batch, unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int b = blockIdx.z;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch && row < M && col < N) {
        const float *Ab = A + b * M * K;
        const float *Bb = B + b * K * N;
        float *Cb = C + b * M * N;
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; k++) {
            sum += Ab[row * K + k] * Bb[k * N + col];
        }
        Cb[row * N + col] = sum;
    }
}

torch::Tensor aten_bmm_fwd(torch::Tensor A, torch::Tensor B) {
    int batch = A.size(0), M = A.size(1), K = A.size(2), N = B.size(2);
    auto C = torch::empty({batch, M, N}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16, batch);
    aten_bmm<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                   C.data_ptr<float>(), batch, M, K, N);
    return C;
}
"""

def test():
    ext = compile_cuda("aten_bmm", KERNEL_SRC, ["aten_bmm_fwd"])
    A = torch.randn(4, 16, 32, device="cuda")
    B = torch.randn(4, 32, 24, device="cuda")
    result = ext.aten_bmm_fwd(A.contiguous(), B.contiguous())
    expected = torch.bmm(A, B)
    check("aten.bmm", result, expected, atol=1e-3)
    print("PASS aten.bmm")

if __name__ == "__main__":
    test()
'''

ADDMM_FILE = '''"""Reference CUDA kernel for aten.addmm — bias + A @ B."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_addmm(
    const float *bias, const float *A, const float *B, float *C,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = bias[col];
        for (unsigned int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor aten_addmm_fwd(torch::Tensor bias, torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    aten_addmm<<<blocks, threads>>>(bias.data_ptr<float>(), A.data_ptr<float>(),
                                     B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}
"""

def test():
    ext = compile_cuda("aten_addmm", KERNEL_SRC, ["aten_addmm_fwd"])
    bias = torch.randn(48, device="cuda")
    A = torch.randn(64, 32, device="cuda")
    B = torch.randn(32, 48, device="cuda")
    result = ext.aten_addmm_fwd(bias, A.contiguous(), B.contiguous())
    expected = torch.addmm(bias, A, B)
    check("aten.addmm", result, expected, atol=1e-3)
    print("PASS aten.addmm")

if __name__ == "__main__":
    test()
'''

# ─────────────────────────────────────────────────────────────────────────────
# Reductions (per-row shared memory)
# ─────────────────────────────────────────────────────────────────────────────

REDUCTION_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name}."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_{op_name}(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {{
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float v = {identity};
    for (unsigned int j = tid; j < cols; j += blockDim.x) {{
        {accumulate}
    }}
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{ {reduce} }} __syncthreads();
    }}
    if (tid == 0) output[row] = {finalize};
}}

torch::Tensor aten_{op_name}_fwd(torch::Tensor input, int dim) {{
    // Flatten to 2D: rows = product of dims before `dim`, cols = dim size
    // For simplicity, only handle last-dim reduction
    auto sizes = input.sizes();
    int cols = sizes[sizes.size() - 1];
    int rows = input.numel() / cols;
    auto flat = input.reshape({{rows, cols}}).contiguous();
    auto output = torch::empty({{rows}}, input.options());
    int threads = 256;
    aten_{op_name}<<<rows, threads, threads * sizeof(float)>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    // Reshape output to match PyTorch's output shape
    auto out_sizes = sizes.vec();
    out_sizes.pop_back();
    if (out_sizes.empty()) out_sizes.push_back(1);
    return output.reshape(out_sizes);
}}
"""

def test():
    ext = compile_cuda("aten_{op_name}", KERNEL_SRC, ["aten_{op_name}_fwd"])
    x = {test_input}
    result = ext.aten_{op_name}_fwd(x, -1)
    expected = {torch_ref}
    check("aten.{op_name}", result, expected{atol_str})
    print(f"PASS aten.{op_name}")

if __name__ == "__main__":
    test()
'''

REDUCTION_OPS = [
    # (op_name, identity, accumulate, reduce, finalize, test_input, torch_ref, atol)
    ("sum", "0.0f", "v += ri[j];",
     "sdata[tid] += sdata[tid + s];", "sdata[0]",
     "torch.randn(32, 64, device='cuda')", "x.sum(dim=-1)", 1e-3),
    ("mean", "0.0f", "v += ri[j];",
     "sdata[tid] += sdata[tid + s];", "sdata[0] / (float)cols",
     "torch.randn(32, 64, device='cuda')", "x.mean(dim=-1)", 1e-4),
    ("amax", "-1e38f", "v = fmaxf(v, ri[j]);",
     "sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);", "sdata[0]",
     "torch.randn(32, 64, device='cuda')", "x.amax(dim=-1)", None),
    ("amin", "1e38f", "v = fminf(v, ri[j]);",
     "sdata[tid] = fminf(sdata[tid], sdata[tid + s]);", "sdata[0]",
     "torch.randn(32, 64, device='cuda')", "x.amin(dim=-1)", None),
    ("prod", "1.0f", "v *= ri[j];",
     "sdata[tid] *= sdata[tid + s];", "sdata[0]",
     "torch.rand(8, 16, device='cuda') + 0.5", "x.prod(dim=-1)", 1e-2),
]

# ─────────────────────────────────────────────────────────────────────────────
# Softmax (3-pass: max, exp+sum, normalize)
# ─────────────────────────────────────────────────────────────────────────────

SOFTMAX_FILE = '''"""Reference CUDA kernel for aten._softmax."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_softmax(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    // Pass 1: max
    float v = -1e38f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) v = fmaxf(v, ri[j]);
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]); __syncthreads();
    }
    float row_max = sdata[0];
    // Pass 2: exp + sum
    float lsum = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        float e = expf(ri[j] - row_max); ro[j] = e; lsum += e;
    }
    sdata[tid] = lsum; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads();
    }
    float inv = 1.0f / sdata[0];
    // Pass 3: normalize
    for (unsigned int j = tid; j < cols; j += blockDim.x) ro[j] *= inv;
}

torch::Tensor aten_softmax_fwd(torch::Tensor input) {
    auto sizes = input.sizes();
    int cols = sizes[sizes.size() - 1];
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty_like(flat);
    int threads = 256;
    aten_softmax<<<rows, threads, threads * sizeof(float)>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output.reshape(sizes);
}
"""

def test():
    ext = compile_cuda("aten_softmax", KERNEL_SRC, ["aten_softmax_fwd"])
    x = torch.randn(8, 64, device="cuda")
    result = ext.aten_softmax_fwd(x)
    expected = torch.softmax(x, dim=-1)
    check("aten._softmax", result, expected, atol=1e-5)
    print("PASS aten._softmax")

if __name__ == "__main__":
    test()
'''

LAYER_NORM_FILE = '''"""Reference CUDA kernel for aten.native_layer_norm."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_layer_norm(
    const float *input, const float *weight, const float *bias,
    float *output, float *mean_out, float *rstd_out,
    unsigned int rows, unsigned int cols, float eps
) {
    extern __shared__ float sdata[];
    float *s_sum = sdata, *s_sq = sdata + blockDim.x;
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    float ls = 0.0f, lsq = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        float v = ri[j]; ls += v; lsq += v * v;
    }
    s_sum[tid] = ls; s_sq[tid] = lsq; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) { s_sum[tid] += s_sum[tid+s]; s_sq[tid] += s_sq[tid+s]; }
        __syncthreads();
    }
    float mean = s_sum[0] / (float)cols;
    float rstd = rsqrtf(s_sq[0] / (float)cols - mean * mean + eps);
    if (tid == 0) { mean_out[row] = mean; rstd_out[row] = rstd; }
    __syncthreads();
    for (unsigned int j = tid; j < cols; j += blockDim.x)
        ro[j] = (ri[j] - mean) * rstd * weight[j] + bias[j];
}

std::vector<torch::Tensor> aten_layer_norm_fwd(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, double eps
) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty_like(flat);
    auto mean_out = torch::empty({rows}, input.options());
    auto rstd_out = torch::empty({rows}, input.options());
    int threads = 256;
    int smem = 2 * threads * sizeof(float);
    aten_layer_norm<<<rows, threads, smem>>>(
        flat.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), mean_out.data_ptr<float>(), rstd_out.data_ptr<float>(),
        rows, cols, (float)eps);
    return {output.reshape(input.sizes()), mean_out.reshape({rows, 1}), rstd_out.reshape({rows, 1})};
}
"""

def test():
    ext = compile_cuda("aten_layer_norm", KERNEL_SRC, ["aten_layer_norm_fwd"])
    x = torch.randn(8, 64, device="cuda")
    w = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    result = ext.aten_layer_norm_fwd(x, w, b, 1e-5)
    expected = torch.nn.functional.layer_norm(x, [64], w, b, 1e-5)
    check("aten.native_layer_norm", result[0], expected, atol=1e-4)
    print("PASS aten.native_layer_norm")

if __name__ == "__main__":
    test()
'''

# ─────────────────────────────────────────────────────────────────────────────
# Generate all files
# ─────────────────────────────────────────────────────────────────────────────

def generate():
    count = 0

    # Unary ops
    for op_name, func_name, cuda_expr, test_input, torch_ref, atol in UNARY_OPS:
        atol_str = f", atol={atol}" if atol else ""
        content = UNARY_TEMPLATE.format(
            op_name=op_name, func_name=func_name, cuda_expr=cuda_expr,
            test_input=test_input, torch_ref=torch_ref, atol_str=atol_str)
        path = HERE / f"aten_{op_name}.py"
        path.write_text(content)
        count += 1

    # Binary ops
    for op_name, func_name, cuda_expr, test_setup, torch_ref, atol in BINARY_OPS:
        atol_str = f", atol={atol}" if atol else ""
        content = BINARY_TEMPLATE.format(
            op_name=op_name, func_name=func_name, cuda_expr=cuda_expr,
            test_setup=test_setup, torch_ref=torch_ref, atol_str=atol_str)
        path = HERE / f"aten_{op_name}.py"
        path.write_text(content)
        count += 1

    # Matmul family
    for name, content in [("mm", MATMUL_FILE), ("bmm", BMM_FILE), ("addmm", ADDMM_FILE)]:
        (HERE / f"aten_{name}.py").write_text(content)
        count += 1

    # Reductions
    for op_name, identity, accumulate, reduce, finalize, test_input, torch_ref, atol in REDUCTION_OPS:
        atol_str = f", atol={atol}" if atol else ""
        content = REDUCTION_TEMPLATE.format(
            op_name=op_name, identity=identity, accumulate=accumulate,
            reduce=reduce, finalize=finalize, test_input=test_input,
            torch_ref=torch_ref, atol_str=atol_str)
        path = HERE / f"aten_{op_name}.py"
        path.write_text(content)
        count += 1

    # Softmax, layer_norm
    (HERE / "aten__softmax.py").write_text(SOFTMAX_FILE)
    (HERE / "aten_native_layer_norm.py").write_text(LAYER_NORM_FILE)
    count += 2

    print(f"Generated {count} reference kernel files in {HERE}/")
    return count


if __name__ == "__main__":
    n = generate()
    print(f"Done. Run tests with: python {HERE}/run_all_tests.py")
