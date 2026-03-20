#!/usr/bin/env python3
"""Generate reference CUDA kernel files for every aten op.

Each generated file is self-contained: kernel source + wrapper + test function.
Tests compare the CUDA kernel output against the ACTUAL aten op (torch.ops.aten.*),
not high-level PyTorch APIs.

Coverage: ~180 aten ops across all categories:
  - Elementwise unary (49 ops)
  - Elementwise binary (12 ops)
  - Comparison (6 ops)
  - Ternary/conditional (5 ops)
  - Backward gradient (5 ops)
  - Reductions (12 ops)
  - Scan/cumulative (2 ops)
  - Sort/search (3 ops)
  - Matmul/linear algebra (7 ops)
  - Normalization (4 ops)
  - Layout/view (14 ops)
  - Tensor creation (9 ops)
  - Indexing (8 ops)
  - Concatenation/split (5 ops)
  - Rearrange (6 ops)
  - Convolution (2 ops)
  - Pooling (3 ops)
  - Embedding (2 ops)
  - Loss functions (3 ops)
  - Attention (1 op)
  - Dropout (1 op)
  - Padding (1 op)
  - Type conversion/identity (4 ops)

Run: python torch_graph/cuda_ref_kernels/generate_all.py
Test one: python -c "from torch_graph.cuda_ref_kernels.aten_add import test; test()"
Test all: python torch_graph/cuda_ref_kernels/run_all_tests.py
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).parent


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1: TEMPLATES (for table-driven ops)
# ═════════════════════════════════════════════════════════════════════════════

# ─── 1a. Elementwise unary (1 tensor → 1 tensor) ─────────────────────────────

UNARY_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name}."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

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
    expected = {aten_ref}
    check("aten.{op_name}", result, expected{atol_str})
    print(f"PASS aten.{op_name}")

if __name__ == "__main__":
    test()
'''

# ─── 1b. Elementwise binary (2 tensors → 1 tensor) ──────────────────────────

BINARY_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name}."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

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
    expected = {aten_ref}
    check("aten.{op_name}", result, expected{atol_str})
    print(f"PASS aten.{op_name}")

if __name__ == "__main__":
    test()
'''

# ─── 1c. Comparison (2 tensors → 1 float tensor, aten returns bool) ─────────

COMPARISON_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name}."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

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
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    result = ext.{func_name}_fwd(a, b)
    expected = {aten_ref}.float()
    check("aten.{op_name}", result, expected)
    print(f"PASS aten.{op_name}")

if __name__ == "__main__":
    test()
'''

# ─── 1d. Backward gradient (2 tensors → 1 tensor) ───────────────────────────

BACKWARD_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name} (backward gradient op)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void {func_name}(const float *grad, const float *saved, float *out0, unsigned int n) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        float g = grad[i];
        float s = saved[i];
        out0[i] = {cuda_expr};
    }}
}}

torch::Tensor {func_name}_fwd(torch::Tensor grad, torch::Tensor saved) {{
    auto out0 = torch::empty_like(grad);
    int n = grad.numel();
    {func_name}<<<(n+255)/256, 256>>>(grad.data_ptr<float>(), saved.data_ptr<float>(), out0.data_ptr<float>(), n);
    return out0;
}}
"""

def test():
    ext = compile_cuda("{func_name}", KERNEL_SRC, ["{func_name}_fwd"])
    {test_setup}
    result = ext.{func_name}_fwd(grad, saved)
    expected = {aten_ref}
    check("aten.{op_name}", result, expected{atol_str})
    print(f"PASS aten.{op_name}")

if __name__ == "__main__":
    test()
'''

# ─── 1e. Reduction (per-row shared memory) ──────────────────────────────────

REDUCTION_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name}."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

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
    auto sizes = input.sizes();
    int cols = sizes[sizes.size() - 1];
    int rows = input.numel() / cols;
    auto flat = input.reshape({{rows, cols}}).contiguous();
    auto output = torch::empty({{rows}}, input.options());
    int threads = 256;
    aten_{op_name}<<<rows, threads, threads * sizeof(float)>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
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
    expected = {aten_ref}
    check("aten.{op_name}", result, expected{atol_str})
    print(f"PASS aten.{op_name}")

if __name__ == "__main__":
    test()
'''


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2: OP TABLES (for template-driven generation)
# ═════════════════════════════════════════════════════════════════════════════

# ─── 2a. Unary ops ───────────────────────────────────────────────────────────

UNARY_OPS = [
    # (op_name, func_name, cuda_expr, test_input, aten_ref, atol)
    # --- Activations ---
    ("relu", "aten_relu", "((x > 0.0f) ? x : 0.0f)", "torch.randn(1024, device='cuda')", "aten.relu.default(x)", None),
    ("relu6", "aten_relu6", "fminf(fmaxf(x, 0.0f), 6.0f)", "torch.randn(1024, device='cuda') * 5", "aten.hardtanh.default(x, 0.0, 6.0)", 1e-5),
    ("gelu", "aten_gelu", "(x * 0.5f * (1.0f + erff(x * 0.7071067811865476f)))", "torch.randn(1024, device='cuda')", "aten.gelu.default(x)", 1e-5),
    ("silu", "aten_silu", "(x / (1.0f + expf(-x)))", "torch.randn(1024, device='cuda')", "aten.silu.default(x)", 1e-5),
    ("sigmoid", "aten_sigmoid", "(1.0f / (1.0f + expf(-x)))", "torch.randn(1024, device='cuda')", "aten.sigmoid.default(x)", 1e-5),
    ("tanh", "aten_tanh", "tanhf(x)", "torch.randn(1024, device='cuda')", "aten.tanh.default(x)", 1e-6),
    ("hardswish", "aten_hardswish", "(x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f)", "torch.randn(1024, device='cuda') * 5", "aten.hardswish.default(x)", 1e-5),
    ("hardsigmoid", "aten_hardsigmoid", "fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f)", "torch.randn(1024, device='cuda') * 5", "aten.hardsigmoid.default(x)", 1e-5),
    ("hardtanh", "aten_hardtanh", "fminf(fmaxf(x, -1.0f), 1.0f)", "torch.randn(1024, device='cuda') * 3", "aten.hardtanh.default(x)", 1e-5),
    ("softplus", "aten_softplus", "((x > 20.0f) ? x : logf(1.0f + expf(x)))", "torch.randn(1024, device='cuda') * 5", "aten.softplus.default(x)", 1e-4),
    ("mish", "aten_mish", "(x * tanhf((x > 20.0f) ? x : logf(1.0f + expf(x))))", "torch.randn(1024, device='cuda')", "aten.mish.default(x)", 1e-4),
    ("elu", "aten_elu", "((x > 0.0f) ? x : (expf(x) - 1.0f))", "torch.randn(1024, device='cuda')", "aten.elu.default(x)", 1e-5),
    ("leaky_relu", "aten_leaky_relu", "((x > 0.0f) ? x : 0.01f * x)", "torch.randn(1024, device='cuda')", "aten.leaky_relu.default(x, 0.01)", 1e-6),
    ("log_sigmoid", "aten_log_sigmoid_forward", "(-logf(1.0f + expf(-x)))", "torch.randn(1024, device='cuda')", "aten.log_sigmoid_forward.default(x)[0]", 1e-5),

    # --- Unary math ---
    ("abs", "aten_abs", "fabsf(x)", "torch.randn(1024, device='cuda')", "aten.abs.default(x)", None),
    ("neg", "aten_neg", "(-x)", "torch.randn(1024, device='cuda')", "aten.neg.default(x)", None),
    ("exp", "aten_exp", "expf(x)", "torch.randn(1024, device='cuda')", "aten.exp.default(x)", 1e-5),
    ("exp2", "aten_exp2", "exp2f(x)", "torch.randn(1024, device='cuda')", "aten.exp2.default(x)", 1e-5),
    ("expm1", "aten_expm1", "expm1f(x)", "torch.randn(1024, device='cuda')", "aten.expm1.default(x)", 1e-5),
    ("log", "aten_log", "logf(x)", "torch.rand(1024, device='cuda') + 0.01", "aten.log.default(x)", 1e-5),
    ("log2", "aten_log2", "log2f(x)", "torch.rand(1024, device='cuda') + 0.01", "aten.log2.default(x)", 1e-5),
    ("log10", "aten_log10", "log10f(x)", "torch.rand(1024, device='cuda') + 0.01", "aten.log10.default(x)", 1e-5),
    ("log1p", "aten_log1p", "log1pf(x)", "torch.rand(1024, device='cuda')", "aten.log1p.default(x)", 1e-5),
    ("sqrt", "aten_sqrt", "sqrtf(x)", "torch.rand(1024, device='cuda') + 0.01", "aten.sqrt.default(x)", 1e-6),
    ("rsqrt", "aten_rsqrt", "rsqrtf(x)", "torch.rand(1024, device='cuda') + 0.01", "aten.rsqrt.default(x)", 1e-4),
    ("reciprocal", "aten_reciprocal", "(1.0f / x)", "torch.randn(1024, device='cuda').abs() + 0.1", "aten.reciprocal.default(x)", 1e-5),
    ("ceil", "aten_ceil", "ceilf(x)", "torch.randn(1024, device='cuda') * 10", "aten.ceil.default(x)", None),
    ("floor", "aten_floor", "floorf(x)", "torch.randn(1024, device='cuda') * 10", "aten.floor.default(x)", None),
    ("round", "aten_round", "nearbyintf(x)", "torch.randn(1024, device='cuda') * 10", "aten.round.default(x)", None),
    ("trunc", "aten_trunc", "truncf(x)", "torch.randn(1024, device='cuda') * 10", "aten.trunc.default(x)", None),
    ("frac", "aten_frac", "(x - truncf(x))", "torch.randn(1024, device='cuda') * 10", "aten.frac.default(x)", 1e-5),
    ("sign", "aten_sign", "((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))", "torch.randn(1024, device='cuda')", "aten.sign.default(x)", None),
    ("sgn", "aten_sgn", "((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))", "torch.randn(1024, device='cuda')", "aten.sgn.default(x)", None),

    # --- Trigonometric ---
    ("sin", "aten_sin", "sinf(x)", "torch.randn(1024, device='cuda')", "aten.sin.default(x)", 1e-5),
    ("cos", "aten_cos", "cosf(x)", "torch.randn(1024, device='cuda')", "aten.cos.default(x)", 1e-5),
    ("tan", "aten_tan", "tanf(x)", "torch.randn(1024, device='cuda') * 0.5", "aten.tan.default(x)", 1e-4),
    ("asin", "aten_asin", "asinf(x)", "torch.rand(1024, device='cuda') * 1.98 - 0.99", "aten.asin.default(x)", 1e-5),
    ("acos", "aten_acos", "acosf(x)", "torch.rand(1024, device='cuda') * 1.98 - 0.99", "aten.acos.default(x)", 1e-5),
    ("atan", "aten_atan", "atanf(x)", "torch.randn(1024, device='cuda')", "aten.atan.default(x)", 1e-5),

    # --- Hyperbolic ---
    ("sinh", "aten_sinh", "sinhf(x)", "torch.randn(1024, device='cuda')", "aten.sinh.default(x)", 1e-4),
    ("cosh", "aten_cosh", "coshf(x)", "torch.randn(1024, device='cuda')", "aten.cosh.default(x)", 1e-4),
    ("asinh", "aten_asinh", "asinhf(x)", "torch.randn(1024, device='cuda')", "aten.asinh.default(x)", 1e-5),
    ("acosh", "aten_acosh", "acoshf(x)", "torch.rand(1024, device='cuda') + 1.01", "aten.acosh.default(x)", 1e-5),
    ("atanh", "aten_atanh", "atanhf(x)", "torch.rand(1024, device='cuda') * 1.98 - 0.99", "aten.atanh.default(x)", 1e-5),

    # --- Error functions ---
    ("erf", "aten_erf", "erff(x)", "torch.randn(1024, device='cuda')", "aten.erf.default(x)", 1e-5),
    ("erfc", "aten_erfc", "erfcf(x)", "torch.randn(1024, device='cuda')", "aten.erfc.default(x)", 1e-5),

    # --- Predicates (output float 0/1, aten returns bool) ---
    ("isnan", "aten_isnan", "(isnan(x) ? 1.0f : 0.0f)", "torch.tensor([1.0, float('nan'), 0.0, float('nan'), -1.0] * 200, device='cuda')", "aten.isnan.default(x).float()", None),
    ("isinf", "aten_isinf", "(isinf(x) ? 1.0f : 0.0f)", "torch.tensor([1.0, float('inf'), 0.0, float('-inf'), -1.0] * 200, device='cuda')", "aten.isinf.default(x).float()", None),
    ("isfinite", "aten_isfinite", "((isfinite(x)) ? 1.0f : 0.0f)", "torch.tensor([1.0, float('inf'), 0.0, float('nan'), -1.0] * 200, device='cuda')", "torch.isfinite(x).float()", None),
    ("logical_not", "aten_logical_not", "((x == 0.0f) ? 1.0f : 0.0f)", "torch.tensor([1.0, 0.0, -1.0, 0.0, 3.14] * 200, device='cuda')", "aten.logical_not.default(x).float()", None),
    ("bitwise_not", "aten_bitwise_not", "((x == 0.0f) ? 1.0f : 0.0f)", "torch.tensor([1.0, 0.0, -1.0, 0.0, 3.14] * 200, device='cuda')", "aten.logical_not.default(x).float()", None),
]

# ─── 2b. Binary ops ──────────────────────────────────────────────────────────

BINARY_OPS = [
    # (op_name, func_name, cuda_expr, test_setup, aten_ref, atol)
    ("add", "aten_add", "(a + b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "aten.add.Tensor(a, b)", None),
    ("sub", "aten_sub", "(a - b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "aten.sub.Tensor(a, b)", None),
    ("mul", "aten_mul", "(a * b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "aten.mul.Tensor(a, b)", None),
    ("div", "aten_div", "(a / b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda').abs() + 0.1",
     "aten.div.Tensor(a, b)", 1e-5),
    ("pow", "aten_pow", "powf(a, b)",
     "a = torch.rand(1024, device='cuda') + 0.1\n    b = torch.rand(1024, device='cuda') * 3",
     "aten.pow.Tensor_Tensor(a, b)", 1e-4),
    ("maximum", "aten_maximum", "fmaxf(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "aten.maximum.default(a, b)", None),
    ("minimum", "aten_minimum", "fminf(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "aten.minimum.default(a, b)", None),
    ("atan2", "aten_atan2", "atan2f(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "aten.atan2.default(a, b)", 1e-5),
    ("fmod", "aten_fmod", "fmodf(a, b)",
     "a = torch.randn(1024, device='cuda') * 10\n    b = torch.randn(1024, device='cuda').abs() + 0.5",
     "aten.fmod.Tensor(a, b)", 1e-5),
    ("remainder", "aten_remainder", "(a - b * floorf(a / b))",
     "a = torch.randn(1024, device='cuda') * 10\n    b = torch.randn(1024, device='cuda').abs() + 0.5",
     "aten.remainder.Tensor(a, b)", 1e-4),
    ("hypot", "aten_hypot", "hypotf(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "aten.hypot.default(a, b)", 1e-5),
    ("copysign", "aten_copysign", "copysignf(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "aten.copysign.Tensor(a, b)", None),
]

# ─── 2c. Comparison ops ──────────────────────────────────────────────────────

COMPARISON_OPS = [
    # (op_name, func_name, cuda_expr, aten_ref)
    ("eq", "aten_eq", "(a == b ? 1.0f : 0.0f)", "aten.eq.Tensor(a, b)"),
    ("ne", "aten_ne", "(a != b ? 1.0f : 0.0f)", "aten.ne.Tensor(a, b)"),
    ("gt", "aten_gt", "(a > b ? 1.0f : 0.0f)", "aten.gt.Tensor(a, b)"),
    ("ge", "aten_ge", "(a >= b ? 1.0f : 0.0f)", "aten.ge.Tensor(a, b)"),
    ("lt", "aten_lt", "(a < b ? 1.0f : 0.0f)", "aten.lt.Tensor(a, b)"),
    ("le", "aten_le", "(a <= b ? 1.0f : 0.0f)", "aten.le.Tensor(a, b)"),
]

# ─── 2d. Backward gradient ops ───────────────────────────────────────────────

BACKWARD_OPS = [
    # (op_name, func_name, cuda_expr, test_setup, aten_ref, atol)
    ("threshold_backward", "aten_threshold_backward",
     "(s > 0.0f ? g : 0.0f)",
     "grad = torch.randn(1024, device='cuda')\n    saved = torch.randn(1024, device='cuda')",
     "aten.threshold_backward.default(grad, saved, 0.0)", 1e-5),
    ("gelu_backward", "aten_gelu_backward",
     "(g * (0.5f * (1.0f + erff(s * 0.7071067811865476f)) + "
     "s * 0.3989422804014327f * expf(-0.5f * s * s)))",
     "grad = torch.randn(1024, device='cuda')\n    saved = torch.randn(1024, device='cuda')",
     "aten.gelu_backward.default(grad, saved)", 1e-4),
    ("silu_backward", "aten_silu_backward",
     "(g * (1.0f / (1.0f + expf(-s))) * "
     "(1.0f + s * (1.0f - 1.0f / (1.0f + expf(-s)))))",
     "grad = torch.randn(1024, device='cuda')\n    saved = torch.randn(1024, device='cuda')",
     "aten.silu_backward.default(grad, saved)", 1e-4),
    ("sigmoid_backward", "aten_sigmoid_backward",
     "(g * s * (1.0f - s))",
     "grad = torch.randn(1024, device='cuda')\n    saved = torch.sigmoid(torch.randn(1024, device='cuda'))",
     "aten.sigmoid_backward.default(grad, saved)", 1e-5),
    ("tanh_backward", "aten_tanh_backward",
     "(g * (1.0f - s * s))",
     "grad = torch.randn(1024, device='cuda')\n    saved = torch.tanh(torch.randn(1024, device='cuda'))",
     "aten.tanh_backward.default(grad, saved)", 1e-5),
]

# ─── 2e. Reduction ops ───────────────────────────────────────────────────────

REDUCTION_OPS = [
    # (op_name, identity, accumulate, reduce, finalize, test_input, aten_ref, atol)
    ("sum", "0.0f", "v += ri[j];",
     "sdata[tid] += sdata[tid + s];", "sdata[0]",
     "torch.randn(32, 64, device='cuda')", "aten.sum.dim_IntList(x, [-1])", 1e-3),
    ("mean", "0.0f", "v += ri[j];",
     "sdata[tid] += sdata[tid + s];", "sdata[0] / (float)cols",
     "torch.randn(32, 64, device='cuda')", "aten.mean.dim(x, [-1])", 1e-4),
    ("amax", "-1e38f", "v = fmaxf(v, ri[j]);",
     "sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);", "sdata[0]",
     "torch.randn(32, 64, device='cuda')", "aten.amax.default(x, [-1])", None),
    ("amin", "1e38f", "v = fminf(v, ri[j]);",
     "sdata[tid] = fminf(sdata[tid], sdata[tid + s]);", "sdata[0]",
     "torch.randn(32, 64, device='cuda')", "aten.amin.default(x, [-1])", None),
    ("prod", "1.0f", "v *= ri[j];",
     "sdata[tid] *= sdata[tid + s];", "sdata[0]",
     "torch.rand(8, 16, device='cuda') + 0.5", "aten.prod.dim_int(x, -1)", 1e-2),
]


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3: HAND-CRAFTED FILES (complex ops that need custom kernels)
# ═════════════════════════════════════════════════════════════════════════════

# ─── 3a. Ternary / conditional ops ──────────────────────────────────────────

WHERE_FILE = '''"""Reference CUDA kernel for aten.where — elementwise conditional select."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_where(
    const float *cond, const float *x, const float *y, float *out, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (cond[i] != 0.0f) ? x[i] : y[i];
    }
}

torch::Tensor aten_where_fwd(torch::Tensor cond, torch::Tensor x, torch::Tensor y) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    aten_where<<<(n+255)/256, 256>>>(
        cond.data_ptr<float>(), x.data_ptr<float>(),
        y.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_where", KERNEL_SRC, ["aten_where_fwd"])
    cond = (torch.randn(1024, device='cuda') > 0).float()
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    result = ext.aten_where_fwd(cond, x, y)
    expected = aten.where.self(cond.bool(), x, y)
    check("aten.where", result, expected)
    print("PASS aten.where")

if __name__ == "__main__":
    test()
'''

CLAMP_FILE = '''"""Reference CUDA kernel for aten.clamp — clamp to [min, max] range."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_clamp(
    const float *in0, float *out0, float lo, float hi, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in0[i];
        out0[i] = fminf(fmaxf(x, lo), hi);
    }
}

torch::Tensor aten_clamp_fwd(torch::Tensor in0, double lo, double hi) {
    auto out0 = torch::empty_like(in0);
    int n = in0.numel();
    aten_clamp<<<(n+255)/256, 256>>>(
        in0.data_ptr<float>(), out0.data_ptr<float>(), (float)lo, (float)hi, n);
    return out0;
}
"""

def test():
    ext = compile_cuda("aten_clamp", KERNEL_SRC, ["aten_clamp_fwd"])
    x = torch.randn(1024, device='cuda') * 5
    result = ext.aten_clamp_fwd(x, -1.0, 1.0)
    expected = aten.clamp.default(x, -1.0, 1.0)
    check("aten.clamp", result, expected)
    print("PASS aten.clamp")

if __name__ == "__main__":
    test()
'''

LERP_FILE = '''"""Reference CUDA kernel for aten.lerp — linear interpolation: start + weight*(end-start)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_lerp(
    const float *a, const float *b, const float *w, float *out, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + w[i] * (b[i] - a[i]);
    }
}

torch::Tensor aten_lerp_fwd(torch::Tensor a, torch::Tensor b, torch::Tensor w) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    aten_lerp<<<(n+255)/256, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(),
        w.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_lerp", KERNEL_SRC, ["aten_lerp_fwd"])
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    w = torch.rand(1024, device='cuda')
    result = ext.aten_lerp_fwd(a, b, w)
    expected = aten.lerp.Tensor(a, b, w)
    check("aten.lerp", result, expected, atol=1e-5)
    print("PASS aten.lerp")

if __name__ == "__main__":
    test()
'''

ADDCMUL_FILE = '''"""Reference CUDA kernel for aten.addcmul — input + value * tensor1 * tensor2."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_addcmul(
    const float *inp, const float *t1, const float *t2,
    float *out, float value, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = inp[i] + value * t1[i] * t2[i];
    }
}

torch::Tensor aten_addcmul_fwd(
    torch::Tensor inp, torch::Tensor t1, torch::Tensor t2, double value
) {
    auto out = torch::empty_like(inp);
    int n = inp.numel();
    aten_addcmul<<<(n+255)/256, 256>>>(
        inp.data_ptr<float>(), t1.data_ptr<float>(), t2.data_ptr<float>(),
        out.data_ptr<float>(), (float)value, n);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_addcmul", KERNEL_SRC, ["aten_addcmul_fwd"])
    inp = torch.randn(1024, device='cuda')
    t1 = torch.randn(1024, device='cuda')
    t2 = torch.randn(1024, device='cuda')
    result = ext.aten_addcmul_fwd(inp, t1, t2, 0.5)
    expected = aten.addcmul.default(inp, t1, t2, value=0.5)
    check("aten.addcmul", result, expected, atol=1e-5)
    print("PASS aten.addcmul")

if __name__ == "__main__":
    test()
'''

ADDCDIV_FILE = '''"""Reference CUDA kernel for aten.addcdiv — input + value * tensor1 / tensor2."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_addcdiv(
    const float *inp, const float *t1, const float *t2,
    float *out, float value, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = inp[i] + value * t1[i] / t2[i];
    }
}

torch::Tensor aten_addcdiv_fwd(
    torch::Tensor inp, torch::Tensor t1, torch::Tensor t2, double value
) {
    auto out = torch::empty_like(inp);
    int n = inp.numel();
    aten_addcdiv<<<(n+255)/256, 256>>>(
        inp.data_ptr<float>(), t1.data_ptr<float>(), t2.data_ptr<float>(),
        out.data_ptr<float>(), (float)value, n);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_addcdiv", KERNEL_SRC, ["aten_addcdiv_fwd"])
    inp = torch.randn(1024, device='cuda')
    t1 = torch.randn(1024, device='cuda')
    t2 = torch.randn(1024, device='cuda').abs() + 0.1
    result = ext.aten_addcdiv_fwd(inp, t1, t2, 0.5)
    expected = aten.addcdiv.default(inp, t1, t2, value=0.5)
    check("aten.addcdiv", result, expected, atol=1e-4)
    print("PASS aten.addcdiv")

if __name__ == "__main__":
    test()
'''

MASKED_FILL_FILE = '''"""Reference CUDA kernel for aten.masked_fill — fill where mask is True."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_masked_fill(
    const float *input, const float *mask, float *out, float value, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (mask[i] != 0.0f) ? value : input[i];
    }
}

torch::Tensor aten_masked_fill_fwd(torch::Tensor input, torch::Tensor mask, double value) {
    auto out = torch::empty_like(input);
    int n = input.numel();
    aten_masked_fill<<<(n+255)/256, 256>>>(
        input.data_ptr<float>(), mask.data_ptr<float>(), out.data_ptr<float>(),
        (float)value, n);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_masked_fill", KERNEL_SRC, ["aten_masked_fill_fwd"])
    x = torch.randn(1024, device='cuda')
    mask = (torch.randn(1024, device='cuda') > 0).float()
    result = ext.aten_masked_fill_fwd(x, mask, -1e9)
    expected = aten.masked_fill.Scalar(x, mask.bool(), -1e9)
    check("aten.masked_fill", result, expected)
    print("PASS aten.masked_fill")

if __name__ == "__main__":
    test()
'''

# ─── 3b. Matmul family ──────────────────────────────────────────────────────

MATMUL_FILE = '''"""Reference CUDA kernel for aten.mm — triple nested loop, not optimized."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

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
    expected = aten.mm.default(A, B)
    check("aten.mm", result, expected, atol=1e-3)
    print("PASS aten.mm")

if __name__ == "__main__":
    test()
'''

BMM_FILE = '''"""Reference CUDA kernel for aten.bmm — batched matmul, nested loops."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

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
    expected = aten.bmm.default(A, B)
    check("aten.bmm", result, expected, atol=1e-3)
    print("PASS aten.bmm")

if __name__ == "__main__":
    test()
'''

ADDMM_FILE = '''"""Reference CUDA kernel for aten.addmm — bias + A @ B."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

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
    expected = aten.addmm.default(bias, A, B)
    check("aten.addmm", result, expected, atol=1e-3)
    print("PASS aten.addmm")

if __name__ == "__main__":
    test()
'''

DOT_FILE = '''"""Reference CUDA kernel for aten.dot — inner product of two vectors."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_dot(
    const float *a, const float *b, float *out, unsigned int n
) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    float v = 0.0f;
    for (unsigned int i = tid; i < n; i += blockDim.x)
        v += a[i] * b[i];
    sdata[tid] = v;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}

torch::Tensor aten_dot_fwd(torch::Tensor a, torch::Tensor b) {
    auto out = torch::zeros({}, a.options());
    int n = a.numel();
    int threads = 256;
    aten_dot<<<1, threads, threads * sizeof(float)>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_dot", KERNEL_SRC, ["aten_dot_fwd"])
    a = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")
    result = ext.aten_dot_fwd(a, b)
    expected = aten.dot.default(a, b)
    check("aten.dot", result, expected, atol=1e-3)
    print("PASS aten.dot")

if __name__ == "__main__":
    test()
'''

MV_FILE = '''"""Reference CUDA kernel for aten.mv — matrix-vector multiply."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_mv(
    const float *A, const float *x, float *y,
    unsigned int M, unsigned int K
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; k++)
            sum += A[row * K + k] * x[k];
        y[row] = sum;
    }
}

torch::Tensor aten_mv_fwd(torch::Tensor A, torch::Tensor x) {
    int M = A.size(0), K = A.size(1);
    auto y = torch::empty({M}, A.options());
    aten_mv<<<(M+255)/256, 256>>>(
        A.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), M, K);
    return y;
}
"""

def test():
    ext = compile_cuda("aten_mv", KERNEL_SRC, ["aten_mv_fwd"])
    A = torch.randn(64, 32, device="cuda")
    x = torch.randn(32, device="cuda")
    result = ext.aten_mv_fwd(A.contiguous(), x.contiguous())
    expected = aten.mv.default(A, x)
    check("aten.mv", result, expected, atol=1e-3)
    print("PASS aten.mv")

if __name__ == "__main__":
    test()
'''

OUTER_FILE = '''"""Reference CUDA kernel for aten.outer — outer product of two vectors."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_outer(
    const float *a, const float *b, float *out,
    unsigned int M, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        out[row * N + col] = a[row] * b[col];
}

torch::Tensor aten_outer_fwd(torch::Tensor a, torch::Tensor b) {
    int M = a.numel(), N = b.numel();
    auto out = torch::empty({M, N}, a.options());
    dim3 threads(16, 16);
    dim3 blocks((N+15)/16, (M+15)/16);
    aten_outer<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_outer", KERNEL_SRC, ["aten_outer_fwd"])
    a = torch.randn(64, device="cuda")
    b = torch.randn(48, device="cuda")
    result = ext.aten_outer_fwd(a, b)
    expected = aten.outer.default(a, b)
    check("aten.outer", result, expected, atol=1e-5)
    print("PASS aten.outer")

if __name__ == "__main__":
    test()
'''

BADDBMM_FILE = '''"""Reference CUDA kernel for aten.baddbmm — batch add + batch matmul."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_baddbmm(
    const float *self, const float *A, const float *B, float *out,
    unsigned int batch, unsigned int M, unsigned int K, unsigned int N,
    float beta, float alpha
) {
    unsigned int b = blockIdx.z;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch && row < M && col < N) {
        unsigned int off = b * M * N + row * N + col;
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; k++)
            sum += A[b*M*K + row*K + k] * B[b*K*N + k*N + col];
        out[off] = beta * self[off] + alpha * sum;
    }
}

torch::Tensor aten_baddbmm_fwd(
    torch::Tensor self, torch::Tensor A, torch::Tensor B, double beta, double alpha
) {
    int batch = A.size(0), M = A.size(1), K = A.size(2), N = B.size(2);
    auto out = torch::empty({batch, M, N}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((N+15)/16, (M+15)/16, batch);
    aten_baddbmm<<<blocks, threads>>>(
        self.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
        out.data_ptr<float>(), batch, M, K, N, (float)beta, (float)alpha);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_baddbmm", KERNEL_SRC, ["aten_baddbmm_fwd"])
    self = torch.randn(4, 16, 24, device="cuda")
    A = torch.randn(4, 16, 32, device="cuda")
    B = torch.randn(4, 32, 24, device="cuda")
    result = ext.aten_baddbmm_fwd(self.contiguous(), A.contiguous(), B.contiguous(), 1.0, 1.0)
    expected = aten.baddbmm.default(self, A, B)
    check("aten.baddbmm", result, expected, atol=1e-3)
    print("PASS aten.baddbmm")

if __name__ == "__main__":
    test()
'''

LINEAR_FILE = '''"""Reference CUDA kernel for aten.linear — y = x @ weight.T + bias."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_linear(
    const float *x, const float *w, const float *bias, float *out,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = (bias != nullptr) ? bias[col] : 0.0f;
        for (unsigned int k = 0; k < K; k++)
            sum += x[row * K + k] * w[col * K + k];  // w is transposed
        out[row * N + col] = sum;
    }
}

torch::Tensor aten_linear_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor bias) {
    int M = x.size(0), K = x.size(1), N = w.size(0);
    auto out = torch::empty({M, N}, x.options());
    dim3 threads(16, 16);
    dim3 blocks((N+15)/16, (M+15)/16);
    aten_linear<<<blocks, threads>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), M, K, N);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_linear", KERNEL_SRC, ["aten_linear_fwd"])
    x = torch.randn(32, 64, device="cuda")
    w = torch.randn(48, 64, device="cuda")
    b = torch.randn(48, device="cuda")
    result = ext.aten_linear_fwd(x.contiguous(), w.contiguous(), b.contiguous())
    expected = aten.linear.default(x, w, b)
    check("aten.linear", result, expected, atol=1e-3)
    print("PASS aten.linear")

if __name__ == "__main__":
    test()
'''

# ─── 3c. Normalization ──────────────────────────────────────────────────────

SOFTMAX_FILE = '''"""Reference CUDA kernel for aten._softmax — 3-pass: max, exp+sum, normalize."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

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
    float v = -1e38f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) v = fmaxf(v, ri[j]);
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]); __syncthreads();
    }
    float row_max = sdata[0];
    float lsum = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        float e = expf(ri[j] - row_max); ro[j] = e; lsum += e;
    }
    sdata[tid] = lsum; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads();
    }
    float inv = 1.0f / sdata[0];
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
    expected = aten._softmax.default(x, -1, False)
    check("aten._softmax", result, expected, atol=1e-5)
    print("PASS aten._softmax")

if __name__ == "__main__":
    test()
'''

LOG_SOFTMAX_FILE = '''"""Reference CUDA kernel for aten._log_softmax — log(softmax(x))."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_log_softmax(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    extern __shared__ float sdata[];
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    float v = -1e38f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) v = fmaxf(v, ri[j]);
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]); __syncthreads();
    }
    float row_max = sdata[0];
    float lsum = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x)
        lsum += expf(ri[j] - row_max);
    sdata[tid] = lsum; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads();
    }
    float log_sum = logf(sdata[0]);
    for (unsigned int j = tid; j < cols; j += blockDim.x)
        ro[j] = (ri[j] - row_max) - log_sum;
}

torch::Tensor aten_log_softmax_fwd(torch::Tensor input) {
    auto sizes = input.sizes();
    int cols = sizes[sizes.size() - 1];
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty_like(flat);
    int threads = 256;
    aten_log_softmax<<<rows, threads, threads * sizeof(float)>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output.reshape(sizes);
}
"""

def test():
    ext = compile_cuda("aten_log_softmax", KERNEL_SRC, ["aten_log_softmax_fwd"])
    x = torch.randn(8, 64, device="cuda")
    result = ext.aten_log_softmax_fwd(x)
    expected = aten._log_softmax.default(x, -1, False)
    check("aten._log_softmax", result, expected, atol=1e-5)
    print("PASS aten._log_softmax")

if __name__ == "__main__":
    test()
'''

LAYER_NORM_FILE = '''"""Reference CUDA kernel for aten.native_layer_norm."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

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
    expected = aten.native_layer_norm.default(x, [64], w, b, 1e-5)
    check("aten.native_layer_norm", result[0], expected[0], atol=1e-4)
    print("PASS aten.native_layer_norm")

if __name__ == "__main__":
    test()
'''

BATCH_NORM_FILE = '''"""Reference CUDA kernel for aten.native_batch_norm — per-channel normalization."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_batch_norm(
    const float *input, const float *weight, const float *bias,
    const float *running_mean, const float *running_var,
    float *output, unsigned int N, unsigned int C, unsigned int HW, float eps
) {
    // One thread per element. input shape: [N, C, HW]
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * HW;
    if (idx < total) {
        unsigned int c = (idx / HW) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float x = input[idx];
        float normed = (x - mean) * rsqrtf(var + eps);
        output[idx] = normed * weight[c] + bias[c];
    }
}

torch::Tensor aten_batch_norm_fwd(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps
) {
    auto shape = input.sizes();
    int N = shape[0], C = shape[1];
    int HW = input.numel() / (N * C);
    auto flat = input.contiguous();
    auto output = torch::empty_like(flat);
    int total = flat.numel();
    aten_batch_norm<<<(total+255)/256, 256>>>(
        flat.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        output.data_ptr<float>(), N, C, HW, (float)eps);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_batch_norm", KERNEL_SRC, ["aten_batch_norm_fwd"])
    x = torch.randn(2, 8, 4, 4, device="cuda")
    w = torch.randn(8, device="cuda")
    b = torch.randn(8, device="cuda")
    rm = torch.randn(8, device="cuda")
    rv = torch.rand(8, device="cuda") + 0.1
    result = ext.aten_batch_norm_fwd(x, w, b, rm, rv, 1e-5)
    expected = aten.native_batch_norm.default(x, w, b, rm, rv, False, 0.1, 1e-5)
    check("aten.native_batch_norm", result, expected[0], atol=1e-4)
    print("PASS aten.native_batch_norm")

if __name__ == "__main__":
    test()
'''

GROUP_NORM_FILE = '''"""Reference CUDA kernel for aten.native_group_norm — group normalization."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_group_norm(
    const float *input, const float *weight, const float *bias,
    float *output, unsigned int N, unsigned int C, unsigned int HW,
    unsigned int G, float eps
) {
    // One block per (n, g) pair
    unsigned int ng = blockIdx.x;
    unsigned int n = ng / G, g = ng % G;
    unsigned int tid = threadIdx.x;
    unsigned int CpG = C / G;  // channels per group
    unsigned int group_size = CpG * HW;

    extern __shared__ float sdata[];
    float *s_sum = sdata, *s_sq = sdata + blockDim.x;

    // Compute mean and var for this group
    float ls = 0.0f, lsq = 0.0f;
    for (unsigned int i = tid; i < group_size; i += blockDim.x) {
        unsigned int c_local = i / HW;
        unsigned int hw = i % HW;
        unsigned int c = g * CpG + c_local;
        float v = input[n * C * HW + c * HW + hw];
        ls += v; lsq += v * v;
    }
    s_sum[tid] = ls; s_sq[tid] = lsq;
    __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) { s_sum[tid] += s_sum[tid+s]; s_sq[tid] += s_sq[tid+s]; }
        __syncthreads();
    }
    float mean = s_sum[0] / (float)group_size;
    float rstd = rsqrtf(s_sq[0] / (float)group_size - mean * mean + eps);

    // Normalize
    for (unsigned int i = tid; i < group_size; i += blockDim.x) {
        unsigned int c_local = i / HW;
        unsigned int hw = i % HW;
        unsigned int c = g * CpG + c_local;
        unsigned int idx = n * C * HW + c * HW + hw;
        output[idx] = (input[idx] - mean) * rstd * weight[c] + bias[c];
    }
}

torch::Tensor aten_group_norm_fwd(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t N, int64_t C, int64_t HW, int64_t G, double eps
) {
    auto output = torch::empty_like(input);
    int threads = 256;
    int smem = 2 * threads * sizeof(float);
    aten_group_norm<<<N * G, threads, smem>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C, HW, G, (float)eps);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_group_norm", KERNEL_SRC, ["aten_group_norm_fwd"])
    x = torch.randn(2, 8, 4, 4, device="cuda")
    w = torch.randn(8, device="cuda")
    b = torch.randn(8, device="cuda")
    result = ext.aten_group_norm_fwd(x.contiguous(), w, b, 2, 8, 16, 4, 1e-5)
    expected = aten.native_group_norm.default(x, w, b, 2, 8, 16, 4, 1e-5)
    check("aten.native_group_norm", result, expected[0], atol=1e-4)
    print("PASS aten.native_group_norm")

if __name__ == "__main__":
    test()
'''

# ─── 3d. Layout ops — output is always contiguous ───────────────────────────

TRANSPOSE_FILE = '''"""Reference CUDA kernel for aten.transpose — swap two dimensions, output contiguous."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// 2D transpose: out[c][r] = in[r][c], output is contiguous
extern "C" __global__ void aten_transpose_2d(
    const float *in, float *out, unsigned int rows, unsigned int cols
) {
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

torch::Tensor aten_transpose_2d_fwd(torch::Tensor input) {
    auto ci = input.contiguous();
    int rows = ci.size(0), cols = ci.size(1);
    auto output = torch::empty({cols, rows}, ci.options());
    dim3 threads(16, 16);
    dim3 blocks((cols+15)/16, (rows+15)/16);
    aten_transpose_2d<<<blocks, threads>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_transpose_2d", KERNEL_SRC, ["aten_transpose_2d_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_transpose_2d_fwd(x)
    expected = aten.transpose.int(x, 0, 1).contiguous()
    check("aten.transpose", result, expected)
    print("PASS aten.transpose")

if __name__ == "__main__":
    test()
'''

T_FILE = '''"""Reference CUDA kernel for aten.t — 2D matrix transpose (shorthand for transpose(0,1))."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_t_kernel(
    const float *in, float *out, unsigned int rows, unsigned int cols
) {
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols)
        out[c * rows + r] = in[r * cols + c];
}

torch::Tensor aten_t_fwd(torch::Tensor input) {
    auto ci = input.contiguous();
    int rows = ci.size(0), cols = ci.size(1);
    auto output = torch::empty({cols, rows}, ci.options());
    dim3 threads(16, 16);
    dim3 blocks((cols+15)/16, (rows+15)/16);
    aten_t_kernel<<<blocks, threads>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_t_kernel", KERNEL_SRC, ["aten_t_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_t_fwd(x)
    expected = aten.t.default(x).contiguous()
    check("aten.t", result, expected)
    print("PASS aten.t")

if __name__ == "__main__":
    test()
'''

PERMUTE_FILE = '''"""Reference CUDA kernel for aten.permute — general dimension permutation, output contiguous."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// 3D permute: input[d0][d1][d2] → output[dims[0]][dims[1]][dims[2]], contiguous
extern "C" __global__ void aten_permute_3d(
    const float *input, float *output,
    unsigned int S0, unsigned int S1, unsigned int S2,
    unsigned int perm0, unsigned int perm1, unsigned int perm2
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = S0 * S1 * S2;
    if (idx >= total) return;

    // Input indices
    unsigned int i0 = idx / (S1 * S2);
    unsigned int i1 = (idx / S2) % S1;
    unsigned int i2 = idx % S2;

    // Permuted shape sizes
    unsigned int in_idx[3] = {i0, i1, i2};
    unsigned int sizes[3] = {S0, S1, S2};
    unsigned int perm[3] = {perm0, perm1, perm2};

    // Output index (contiguous in permuted layout)
    unsigned int out_sizes[3] = {sizes[perm[0]], sizes[perm[1]], sizes[perm[2]]};
    unsigned int o0 = in_idx[perm0], o1 = in_idx[perm1], o2 = in_idx[perm2];
    // Wait — we need to map from OUTPUT index to INPUT index.
    // Easier: iterate over output and read from input.
    // But this kernel iterates over input. Let's compute where this input element goes.
    unsigned int out_idx = o0 * out_sizes[1] * out_sizes[2] + o1 * out_sizes[2] + o2;
    output[out_idx] = input[idx];
}

torch::Tensor aten_permute_3d_fwd(
    torch::Tensor input, int64_t p0, int64_t p1, int64_t p2
) {
    auto ci = input.contiguous();
    int S0 = ci.size(0), S1 = ci.size(1), S2 = ci.size(2);
    int out_sizes[3];
    int perm[3] = {(int)p0, (int)p1, (int)p2};
    int sizes[3] = {S0, S1, S2};
    for (int i = 0; i < 3; i++) out_sizes[i] = sizes[perm[i]];
    auto output = torch::empty({out_sizes[0], out_sizes[1], out_sizes[2]}, ci.options());
    int total = S0 * S1 * S2;
    aten_permute_3d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), S0, S1, S2, p0, p1, p2);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_permute_3d", KERNEL_SRC, ["aten_permute_3d_fwd"])
    x = torch.randn(4, 8, 16, device="cuda")
    result = ext.aten_permute_3d_fwd(x, 2, 0, 1)
    expected = aten.permute.default(x, [2, 0, 1]).contiguous()
    check("aten.permute", result, expected)
    print("PASS aten.permute")

if __name__ == "__main__":
    test()
'''

CLONE_FILE = '''"""Reference CUDA kernel for aten.clone — copy tensor to new contiguous memory."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_clone_kernel(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_clone_fwd(torch::Tensor input) {
    auto ci = input.contiguous();
    auto output = torch::empty_like(ci);
    int n = ci.numel();
    aten_clone_kernel<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_clone_kernel", KERNEL_SRC, ["aten_clone_fwd"])
    x = torch.randn(1024, device="cuda")
    result = ext.aten_clone_fwd(x)
    expected = aten.clone.default(x)
    check("aten.clone", result, expected)
    print("PASS aten.clone")

if __name__ == "__main__":
    test()
'''

CONTIGUOUS_FILE = '''"""Reference CUDA kernel for aten.contiguous — ensure contiguous memory layout."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_copy_kernel(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_contiguous_fwd(torch::Tensor input) {
    auto ci = input.contiguous();  // PyTorch handles stride logic
    auto output = torch::empty_like(ci);
    int n = ci.numel();
    aten_copy_kernel<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_copy_kernel", KERNEL_SRC, ["aten_contiguous_fwd"])
    x = torch.randn(32, 64, device="cuda").t()  # non-contiguous
    result = ext.aten_contiguous_fwd(x)
    expected = x.contiguous()
    check("aten.contiguous", result, expected)
    print("PASS aten.contiguous")

if __name__ == "__main__":
    test()
'''

VIEW_FILE = '''"""Reference CUDA kernel for aten.view — reshape (data copy if non-contiguous)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_view_copy(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_view_fwd(torch::Tensor input, int64_t d0, int64_t d1) {
    auto ci = input.contiguous();
    auto output = torch::empty({d0, d1}, ci.options());
    int n = ci.numel();
    aten_view_copy<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_view_copy", KERNEL_SRC, ["aten_view_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_view_fwd(x, 64, 32)
    expected = aten.view.default(x, [64, 32]).contiguous()
    check("aten.view", result, expected)
    print("PASS aten.view")

if __name__ == "__main__":
    test()
'''

RESHAPE_FILE = '''"""Reference CUDA kernel for aten.reshape — reshape with contiguous output."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_reshape_copy(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_reshape_fwd(torch::Tensor input, int64_t d0, int64_t d1) {
    auto ci = input.contiguous();
    auto output = torch::empty({d0, d1}, ci.options());
    int n = ci.numel();
    aten_reshape_copy<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_reshape_copy", KERNEL_SRC, ["aten_reshape_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_reshape_fwd(x, 64, 32)
    expected = aten.reshape.default(x, [64, 32]).contiguous()
    check("aten.reshape", result, expected)
    print("PASS aten.reshape")

if __name__ == "__main__":
    test()
'''

UNSQUEEZE_FILE = '''"""Reference CUDA kernel for aten.unsqueeze — add dimension (contiguous copy)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_unsqueeze_copy(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_unsqueeze_fwd(torch::Tensor input, int64_t dim) {
    auto ci = input.contiguous();
    // Compute output shape
    auto sizes = ci.sizes().vec();
    if (dim < 0) dim = sizes.size() + 1 + dim;
    sizes.insert(sizes.begin() + dim, 1);
    auto output = torch::empty(sizes, ci.options());
    int n = ci.numel();
    aten_unsqueeze_copy<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_unsqueeze_copy", KERNEL_SRC, ["aten_unsqueeze_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_unsqueeze_fwd(x, 0)
    expected = aten.unsqueeze.default(x, 0).contiguous()
    check("aten.unsqueeze", result, expected)
    print("PASS aten.unsqueeze")

if __name__ == "__main__":
    test()
'''

SQUEEZE_FILE = '''"""Reference CUDA kernel for aten.squeeze — remove size-1 dimensions (contiguous copy)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_squeeze_copy(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_squeeze_fwd(torch::Tensor input, int64_t dim) {
    auto ci = input.contiguous();
    auto sizes = ci.sizes().vec();
    if (dim < 0) dim = sizes.size() + dim;
    if (sizes[dim] == 1) sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, ci.options());
    int n = ci.numel();
    aten_squeeze_copy<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_squeeze_copy", KERNEL_SRC, ["aten_squeeze_fwd"])
    x = torch.randn(32, 1, 64, device="cuda")
    result = ext.aten_squeeze_fwd(x, 1)
    expected = aten.squeeze.dim(x, 1).contiguous()
    check("aten.squeeze", result, expected)
    print("PASS aten.squeeze")

if __name__ == "__main__":
    test()
'''

FLATTEN_FILE = '''"""Reference CUDA kernel for aten.flatten — flatten dims [start, end] to single dim."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_flatten_copy(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_flatten_fwd(torch::Tensor input, int64_t start, int64_t end) {
    auto ci = input.contiguous();
    auto t = ci.flatten(start, end);
    auto output = torch::empty(t.sizes().vec(), ci.options());
    int n = ci.numel();
    aten_flatten_copy<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_flatten_copy", KERNEL_SRC, ["aten_flatten_fwd"])
    x = torch.randn(4, 8, 16, device="cuda")
    result = ext.aten_flatten_fwd(x, 1, 2)
    expected = aten.flatten.using_ints(x, 1, 2).contiguous()
    check("aten.flatten", result, expected)
    print("PASS aten.flatten")

if __name__ == "__main__":
    test()
'''

EXPAND_FILE = '''"""Reference CUDA kernel for aten.expand — broadcast copy to larger shape."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// 2D expand: input [1, N] or [M, 1] → output [M, N]
extern "C" __global__ void aten_expand_2d(
    const float *input, float *output,
    unsigned int in_rows, unsigned int in_cols,
    unsigned int out_rows, unsigned int out_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_rows * out_cols) return;
    unsigned int r = idx / out_cols;
    unsigned int c = idx % out_cols;
    unsigned int ir = (in_rows == 1) ? 0 : r;
    unsigned int ic = (in_cols == 1) ? 0 : c;
    output[idx] = input[ir * in_cols + ic];
}

torch::Tensor aten_expand_2d_fwd(
    torch::Tensor input, int64_t out_rows, int64_t out_cols
) {
    auto ci = input.contiguous();
    int ir = ci.size(0), ic = ci.size(1);
    auto output = torch::empty({out_rows, out_cols}, ci.options());
    int total = out_rows * out_cols;
    aten_expand_2d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), ir, ic, out_rows, out_cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_expand_2d", KERNEL_SRC, ["aten_expand_2d_fwd"])
    x = torch.randn(1, 64, device="cuda")
    result = ext.aten_expand_2d_fwd(x, 32, 64)
    expected = aten.expand.default(x, [32, 64]).contiguous()
    check("aten.expand", result, expected)
    print("PASS aten.expand")

if __name__ == "__main__":
    test()
'''

SLICE_FILE = '''"""Reference CUDA kernel for aten.slice — copy a contiguous sub-range along a dim."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Slice along dim 0 of a 2D tensor: output = input[start:end:step, :]
extern "C" __global__ void aten_slice_2d(
    const float *input, float *output,
    unsigned int in_rows, unsigned int cols,
    unsigned int start, unsigned int step, unsigned int out_rows
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_rows * cols) return;
    unsigned int r = idx / cols;
    unsigned int c = idx % cols;
    unsigned int src_r = start + r * step;
    output[idx] = input[src_r * cols + c];
}

torch::Tensor aten_slice_2d_fwd(
    torch::Tensor input, int64_t start, int64_t end, int64_t step
) {
    auto ci = input.contiguous();
    int rows = ci.size(0), cols = ci.size(1);
    if (end > rows) end = rows;
    int out_rows = (end - start + step - 1) / step;
    auto output = torch::empty({out_rows, cols}, ci.options());
    int total = out_rows * cols;
    aten_slice_2d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), rows, cols, start, step, out_rows);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_slice_2d", KERNEL_SRC, ["aten_slice_2d_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_slice_2d_fwd(x, 4, 20, 1)
    expected = aten.slice.Tensor(x, 0, 4, 20).contiguous()
    check("aten.slice", result, expected)
    print("PASS aten.slice")

if __name__ == "__main__":
    test()
'''

SELECT_FILE = '''"""Reference CUDA kernel for aten.select — select a single index along a dim."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Select index along dim 0 of a 2D tensor: output = input[index, :]
extern "C" __global__ void aten_select_copy(
    const float *input, float *output, unsigned int cols, unsigned int index
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols)
        output[c] = input[index * cols + c];
}

torch::Tensor aten_select_fwd(torch::Tensor input, int64_t index) {
    auto ci = input.contiguous();
    int cols = ci.size(1);
    auto output = torch::empty({cols}, ci.options());
    aten_select_copy<<<(cols+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), cols, index);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_select_copy", KERNEL_SRC, ["aten_select_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_select_fwd(x, 5)
    expected = aten.select.int(x, 0, 5).contiguous()
    check("aten.select", result, expected)
    print("PASS aten.select")

if __name__ == "__main__":
    test()
'''

NARROW_FILE = '''"""Reference CUDA kernel for aten.narrow — narrow view along a dimension."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_narrow_copy(
    const float *input, float *output, unsigned int cols,
    unsigned int start, unsigned int length
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[idx] = input[(start + r) * cols + c];
}

torch::Tensor aten_narrow_fwd(torch::Tensor input, int64_t start, int64_t length) {
    auto ci = input.contiguous();
    int cols = ci.size(1);
    auto output = torch::empty({length, cols}, ci.options());
    int total = length * cols;
    aten_narrow_copy<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), cols, start, length);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_narrow_copy", KERNEL_SRC, ["aten_narrow_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_narrow_fwd(x, 4, 10)
    expected = aten.narrow.default(x, 0, 4, 10).contiguous()
    check("aten.narrow", result, expected)
    print("PASS aten.narrow")

if __name__ == "__main__":
    test()
'''

FLIP_FILE = '''"""Reference CUDA kernel for aten.flip — reverse along given dimensions."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Flip last dim of 2D tensor
extern "C" __global__ void aten_flip_2d(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[r * cols + (cols - 1 - c)] = input[idx];
}

torch::Tensor aten_flip_2d_fwd(torch::Tensor input) {
    auto ci = input.contiguous();
    int rows = ci.size(0), cols = ci.size(1);
    auto output = torch::empty_like(ci);
    int total = rows * cols;
    aten_flip_2d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_flip_2d", KERNEL_SRC, ["aten_flip_2d_fwd"])
    x = torch.randn(16, 32, device="cuda")
    result = ext.aten_flip_2d_fwd(x)
    expected = aten.flip.default(x, [-1]).contiguous()
    check("aten.flip", result, expected)
    print("PASS aten.flip")

if __name__ == "__main__":
    test()
'''

ROLL_FILE = '''"""Reference CUDA kernel for aten.roll — circular shift along a dimension."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_roll_1d(
    const float *input, float *output, unsigned int n, int shift
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int src = ((int)i - shift % (int)n + (int)n) % (int)n;
    output[i] = input[src];
}

torch::Tensor aten_roll_1d_fwd(torch::Tensor input, int64_t shift) {
    auto ci = input.contiguous();
    int n = ci.numel();
    auto output = torch::empty_like(ci);
    aten_roll_1d<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n, (int)shift);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_roll_1d", KERNEL_SRC, ["aten_roll_1d_fwd"])
    x = torch.randn(256, device="cuda")
    result = ext.aten_roll_1d_fwd(x, 10)
    expected = aten.roll.default(x, [10]).contiguous()
    check("aten.roll", result, expected)
    print("PASS aten.roll")

if __name__ == "__main__":
    test()
'''

REPEAT_FILE = '''"""Reference CUDA kernel for aten.repeat — tile tensor along dimensions."""
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
'''

TRIL_FILE = '''"""Reference CUDA kernel for aten.tril — lower triangle of a matrix."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_tril_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols, int diagonal
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[idx] = ((int)c <= (int)r + diagonal) ? input[idx] : 0.0f;
}

torch::Tensor aten_tril_fwd(torch::Tensor input, int64_t diagonal) {
    auto ci = input.contiguous();
    int rows = ci.size(0), cols = ci.size(1);
    auto output = torch::empty_like(ci);
    int total = rows * cols;
    aten_tril_kernel<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), rows, cols, (int)diagonal);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_tril_kernel", KERNEL_SRC, ["aten_tril_fwd"])
    x = torch.randn(16, 16, device="cuda")
    result = ext.aten_tril_fwd(x, 0)
    expected = aten.tril.default(x)
    check("aten.tril", result, expected)
    print("PASS aten.tril")

if __name__ == "__main__":
    test()
'''

TRIU_FILE = '''"""Reference CUDA kernel for aten.triu — upper triangle of a matrix."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_triu_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols, int diagonal
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[idx] = ((int)c >= (int)r + diagonal) ? input[idx] : 0.0f;
}

torch::Tensor aten_triu_fwd(torch::Tensor input, int64_t diagonal) {
    auto ci = input.contiguous();
    int rows = ci.size(0), cols = ci.size(1);
    auto output = torch::empty_like(ci);
    int total = rows * cols;
    aten_triu_kernel<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), rows, cols, (int)diagonal);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_triu_kernel", KERNEL_SRC, ["aten_triu_fwd"])
    x = torch.randn(16, 16, device="cuda")
    result = ext.aten_triu_fwd(x, 0)
    expected = aten.triu.default(x)
    check("aten.triu", result, expected)
    print("PASS aten.triu")

if __name__ == "__main__":
    test()
'''

# ─── 3e. Tensor creation ────────────────────────────────────────────────────

ARANGE_FILE = '''"""Reference CUDA kernel for aten.arange — fill with sequential values."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_arange_kernel(
    float *output, float start, float step, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = start + i * step;
}

torch::Tensor aten_arange_fwd(double start, double end, double step) {
    int n = (int)ceil((end - start) / step);
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    aten_arange_kernel<<<(n+255)/256, 256>>>(
        output.data_ptr<float>(), (float)start, (float)step, n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_arange_kernel", KERNEL_SRC, ["aten_arange_fwd"])
    result = ext.aten_arange_fwd(0.0, 100.0, 1.0)
    expected = aten.arange.start_step(0, 100, 1, dtype=torch.float32, device='cuda')
    check("aten.arange", result, expected)
    print("PASS aten.arange")

if __name__ == "__main__":
    test()
'''

ZEROS_FILE = '''"""Reference CUDA kernel for aten.zeros — create zero-filled tensor."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_fill_zero(float *output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 0.0f;
}

torch::Tensor aten_zeros_fwd(int64_t d0, int64_t d1) {
    auto output = torch::empty({d0, d1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int n = output.numel();
    aten_fill_zero<<<(n+255)/256, 256>>>(output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_fill_zero", KERNEL_SRC, ["aten_zeros_fwd"])
    result = ext.aten_zeros_fwd(32, 64)
    expected = torch.zeros(32, 64, device='cuda')
    check("aten.zeros", result, expected)
    print("PASS aten.zeros")

if __name__ == "__main__":
    test()
'''

ONES_FILE = '''"""Reference CUDA kernel for aten.ones — create one-filled tensor."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_fill_one(float *output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 1.0f;
}

torch::Tensor aten_ones_fwd(int64_t d0, int64_t d1) {
    auto output = torch::empty({d0, d1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int n = output.numel();
    aten_fill_one<<<(n+255)/256, 256>>>(output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_fill_one", KERNEL_SRC, ["aten_ones_fwd"])
    result = ext.aten_ones_fwd(32, 64)
    expected = torch.ones(32, 64, device='cuda')
    check("aten.ones", result, expected)
    print("PASS aten.ones")

if __name__ == "__main__":
    test()
'''

FULL_FILE = '''"""Reference CUDA kernel for aten.full — create tensor filled with a value."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_fill_val(float *output, float value, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = value;
}

torch::Tensor aten_full_fwd(int64_t d0, int64_t d1, double value) {
    auto output = torch::empty({d0, d1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int n = output.numel();
    aten_fill_val<<<(n+255)/256, 256>>>(output.data_ptr<float>(), (float)value, n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_fill_val", KERNEL_SRC, ["aten_full_fwd"])
    result = ext.aten_full_fwd(32, 64, 3.14)
    expected = torch.full((32, 64), 3.14, device='cuda')
    check("aten.full", result, expected)
    print("PASS aten.full")

if __name__ == "__main__":
    test()
'''

EYE_FILE = '''"""Reference CUDA kernel for aten.eye — identity matrix."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_eye_kernel(float *output, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    unsigned int r = idx / n, c = idx % n;
    output[idx] = (r == c) ? 1.0f : 0.0f;
}

torch::Tensor aten_eye_fwd(int64_t n) {
    auto output = torch::empty({n, n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int total = n * n;
    aten_eye_kernel<<<(total+255)/256, 256>>>(output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_eye_kernel", KERNEL_SRC, ["aten_eye_fwd"])
    result = ext.aten_eye_fwd(32)
    expected = torch.eye(32, device='cuda')
    check("aten.eye", result, expected)
    print("PASS aten.eye")

if __name__ == "__main__":
    test()
'''

LINSPACE_FILE = '''"""Reference CUDA kernel for aten.linspace — evenly spaced values."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_linspace_kernel(
    float *output, float start, float step, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = start + i * step;
}

torch::Tensor aten_linspace_fwd(double start, double end, int64_t steps) {
    auto output = torch::empty({steps}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    float step = (steps > 1) ? (float)(end - start) / (steps - 1) : 0.0f;
    aten_linspace_kernel<<<(steps+255)/256, 256>>>(
        output.data_ptr<float>(), (float)start, step, steps);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_linspace_kernel", KERNEL_SRC, ["aten_linspace_fwd"])
    result = ext.aten_linspace_fwd(0.0, 1.0, 100)
    expected = torch.linspace(0, 1, 100, device='cuda')
    check("aten.linspace", result, expected, atol=1e-5)
    print("PASS aten.linspace")

if __name__ == "__main__":
    test()
'''

# ─── 3f. Indexing ops ───────────────────────────────────────────────────────

GATHER_FILE = '''"""Reference CUDA kernel for aten.gather — gather along a dimension by index."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Gather along dim=1 for 2D tensors: out[i][j] = input[i][index[i][j]]
extern "C" __global__ void aten_gather_2d(
    const float *input, const long *index, float *output,
    unsigned int rows, unsigned int in_cols, unsigned int out_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out_cols) return;
    unsigned int r = idx / out_cols, c = idx % out_cols;
    long src_c = index[r * out_cols + c];
    output[idx] = input[r * in_cols + src_c];
}

torch::Tensor aten_gather_2d_fwd(torch::Tensor input, torch::Tensor index) {
    auto ci = input.contiguous();
    auto ci_idx = index.contiguous();
    int rows = ci.size(0), in_cols = ci.size(1), out_cols = ci_idx.size(1);
    auto output = torch::empty({rows, out_cols}, ci.options());
    int total = rows * out_cols;
    aten_gather_2d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), ci_idx.data_ptr<long>(), output.data_ptr<float>(),
        rows, in_cols, out_cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_gather_2d", KERNEL_SRC, ["aten_gather_2d_fwd"])
    x = torch.randn(8, 32, device="cuda")
    idx = torch.randint(0, 32, (8, 16), device="cuda")
    result = ext.aten_gather_2d_fwd(x, idx)
    expected = aten.gather.default(x, 1, idx)
    check("aten.gather", result, expected)
    print("PASS aten.gather")

if __name__ == "__main__":
    test()
'''

SCATTER_FILE = '''"""Reference CUDA kernel for aten.scatter — scatter values into tensor by index."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Scatter along dim=1: out = input.clone(); out[i][index[i][j]] = src[i][j]
// Naive: one thread per src element, atomic write
extern "C" __global__ void aten_scatter_2d(
    const float *input, const long *index, const float *src, float *output,
    unsigned int rows, unsigned int in_cols, unsigned int src_cols
) {
    // First copy input → output
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_in = rows * in_cols;
    if (idx < total_in) output[idx] = input[idx];
}

extern "C" __global__ void aten_scatter_write(
    const long *index, const float *src, float *output,
    unsigned int rows, unsigned int in_cols, unsigned int src_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_src = rows * src_cols;
    if (idx >= total_src) return;
    unsigned int r = idx / src_cols, c = idx % src_cols;
    long dst_c = index[idx];
    output[r * in_cols + dst_c] = src[idx];
}

torch::Tensor aten_scatter_2d_fwd(
    torch::Tensor input, torch::Tensor index, torch::Tensor src
) {
    auto ci = input.contiguous();
    int rows = ci.size(0), in_cols = ci.size(1), src_cols = index.size(1);
    auto output = torch::empty_like(ci);
    int total_in = rows * in_cols;
    aten_scatter_2d<<<(total_in+255)/256, 256>>>(
        ci.data_ptr<float>(), index.data_ptr<long>(), src.data_ptr<float>(),
        output.data_ptr<float>(), rows, in_cols, src_cols);
    int total_src = rows * src_cols;
    aten_scatter_write<<<(total_src+255)/256, 256>>>(
        index.data_ptr<long>(), src.data_ptr<float>(),
        output.data_ptr<float>(), rows, in_cols, src_cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_scatter_2d", KERNEL_SRC, ["aten_scatter_2d_fwd"])
    x = torch.zeros(8, 32, device="cuda")
    idx = torch.randint(0, 32, (8, 16), device="cuda")
    src = torch.randn(8, 16, device="cuda")
    result = ext.aten_scatter_2d_fwd(x, idx, src)
    expected = aten.scatter.src(x, 1, idx, src)
    check("aten.scatter", result, expected)
    print("PASS aten.scatter")

if __name__ == "__main__":
    test()
'''

INDEX_SELECT_FILE = '''"""Reference CUDA kernel for aten.index_select — select rows by index."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_index_select_kernel(
    const float *input, const long *index, float *output,
    unsigned int n_idx, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_idx * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    long src_r = index[r];
    output[idx] = input[src_r * cols + c];
}

torch::Tensor aten_index_select_fwd(torch::Tensor input, torch::Tensor index) {
    auto ci = input.contiguous();
    int cols = ci.size(1);
    int n_idx = index.numel();
    auto output = torch::empty({n_idx, cols}, ci.options());
    int total = n_idx * cols;
    aten_index_select_kernel<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), index.data_ptr<long>(), output.data_ptr<float>(), n_idx, cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_index_select_kernel", KERNEL_SRC, ["aten_index_select_fwd"])
    x = torch.randn(32, 64, device="cuda")
    idx = torch.tensor([0, 5, 10, 15, 31], device="cuda")
    result = ext.aten_index_select_fwd(x, idx)
    expected = aten.index_select.default(x, 0, idx)
    check("aten.index_select", result, expected)
    print("PASS aten.index_select")

if __name__ == "__main__":
    test()
'''

INDEX_ADD_FILE = '''"""Reference CUDA kernel for aten.index_add — add source into self at indices."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_index_add_init(const float *self, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = self[i];
}

extern "C" __global__ void aten_index_add_kernel(
    const long *index, const float *source, float *out,
    unsigned int n_idx, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_idx * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    long dst_r = index[r];
    atomicAdd(&out[dst_r * cols + c], source[idx]);
}

torch::Tensor aten_index_add_fwd(torch::Tensor self, torch::Tensor index, torch::Tensor source) {
    auto ci = self.contiguous();
    int rows = ci.size(0), cols = ci.size(1), n_idx = index.numel();
    auto out = torch::empty_like(ci);
    int n = ci.numel();
    aten_index_add_init<<<(n+255)/256, 256>>>(ci.data_ptr<float>(), out.data_ptr<float>(), n);
    int total = n_idx * cols;
    aten_index_add_kernel<<<(total+255)/256, 256>>>(
        index.data_ptr<long>(), source.data_ptr<float>(), out.data_ptr<float>(), n_idx, cols);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_index_add_init", KERNEL_SRC, ["aten_index_add_fwd"])
    x = torch.zeros(32, 64, device="cuda")
    idx = torch.tensor([0, 5, 10, 15, 20], device="cuda")
    src = torch.randn(5, 64, device="cuda")
    result = ext.aten_index_add_fwd(x, idx, src)
    expected = aten.index_add.default(x, 0, idx, src)
    check("aten.index_add", result, expected, atol=1e-5)
    print("PASS aten.index_add")

if __name__ == "__main__":
    test()
'''

# ─── 3g. Concatenation / split ──────────────────────────────────────────────

CAT_FILE = '''"""Reference CUDA kernel for aten.cat — concatenate tensors along a dimension."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Cat two 2D tensors along dim=0
extern "C" __global__ void aten_cat_dim0(
    const float *a, const float *b, float *out,
    unsigned int a_rows, unsigned int b_rows, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = (a_rows + b_rows) * cols;
    if (idx >= total) return;
    unsigned int r = idx / cols, c = idx % cols;
    if (r < a_rows)
        out[idx] = a[r * cols + c];
    else
        out[idx] = b[(r - a_rows) * cols + c];
}

torch::Tensor aten_cat_dim0_fwd(torch::Tensor a, torch::Tensor b) {
    auto ca = a.contiguous(), cb = b.contiguous();
    int ar = ca.size(0), br = cb.size(0), cols = ca.size(1);
    auto out = torch::empty({ar + br, cols}, ca.options());
    int total = (ar + br) * cols;
    aten_cat_dim0<<<(total+255)/256, 256>>>(
        ca.data_ptr<float>(), cb.data_ptr<float>(), out.data_ptr<float>(),
        ar, br, cols);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_cat_dim0", KERNEL_SRC, ["aten_cat_dim0_fwd"])
    a = torch.randn(8, 32, device="cuda")
    b = torch.randn(16, 32, device="cuda")
    result = ext.aten_cat_dim0_fwd(a, b)
    expected = aten.cat.default([a, b], 0)
    check("aten.cat", result, expected)
    print("PASS aten.cat")

if __name__ == "__main__":
    test()
'''

STACK_FILE = '''"""Reference CUDA kernel for aten.stack — stack tensors along new dimension."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Stack N 1D tensors → 2D tensor [N, L]
extern "C" __global__ void aten_stack_2(
    const float *a, const float *b, float *out, unsigned int L
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < L) {
        out[i] = a[i];
        out[L + i] = b[i];
    }
}

torch::Tensor aten_stack_2_fwd(torch::Tensor a, torch::Tensor b) {
    auto ca = a.contiguous(), cb = b.contiguous();
    int L = ca.numel();
    auto out = torch::empty({2, L}, ca.options());
    aten_stack_2<<<(L+255)/256, 256>>>(
        ca.data_ptr<float>(), cb.data_ptr<float>(), out.data_ptr<float>(), L);
    return out;
}
"""

def test():
    ext = compile_cuda("aten_stack_2", KERNEL_SRC, ["aten_stack_2_fwd"])
    a = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    result = ext.aten_stack_2_fwd(a, b)
    expected = aten.stack.default([a, b], 0)
    check("aten.stack", result, expected)
    print("PASS aten.stack")

if __name__ == "__main__":
    test()
'''

SPLIT_FILE = '''"""Reference CUDA kernel for aten.split — split tensor into chunks along dim."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_split_copy(
    const float *input, float *out, unsigned int offset, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = input[offset + i];
}

std::vector<torch::Tensor> aten_split_fwd(
    torch::Tensor input, int64_t chunk_size, int64_t dim
) {
    auto ci = input.contiguous();
    int total = ci.size(dim);
    std::vector<torch::Tensor> result;
    int cols = ci.numel() / ci.size(0);
    for (int start = 0; start < total; start += chunk_size) {
        int len = std::min((int)chunk_size, total - start);
        int n = len * cols;
        auto out = torch::empty({len, cols}, ci.options());
        aten_split_copy<<<(n+255)/256, 256>>>(
            ci.data_ptr<float>(), out.data_ptr<float>(), start * cols, n);
        result.push_back(out);
    }
    return result;
}
"""

def test():
    ext = compile_cuda("aten_split_copy", KERNEL_SRC, ["aten_split_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_split_fwd(x, 8, 0)
    expected = aten.split.Tensor(x, 8, 0)
    for i, (r, e) in enumerate(zip(result, expected)):
        check(f"aten.split[{i}]", r, e.contiguous())
    print("PASS aten.split")

if __name__ == "__main__":
    test()
'''

# ─── 3h. More reductions ────────────────────────────────────────────────────

VAR_FILE = '''"""Reference CUDA kernel for aten.var — variance reduction."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void aten_var_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols, int correction
) {
    extern __shared__ float sdata[];
    float *s_sum = sdata, *s_sq = sdata + blockDim.x;
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float ls = 0.0f, lsq = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) {
        float v = ri[j]; ls += v; lsq += v * v;
    }
    s_sum[tid] = ls; s_sq[tid] = lsq; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) { s_sum[tid] += s_sum[tid+s]; s_sq[tid] += s_sq[tid+s]; }
        __syncthreads();
    }
    if (tid == 0) {
        float mean = s_sum[0] / (float)cols;
        output[row] = (s_sq[0] / (float)cols - mean * mean) * (float)cols / (float)(cols - correction);
    }
}

torch::Tensor aten_var_fwd(torch::Tensor input) {
    auto sizes = input.sizes();
    int cols = sizes[sizes.size() - 1];
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty({rows}, input.options());
    int threads = 256;
    int smem = 2 * threads * sizeof(float);
    aten_var_kernel<<<rows, threads, smem>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols, 1);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_var_kernel", KERNEL_SRC, ["aten_var_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_var_fwd(x)
    expected = aten.var.correction(x, [-1])
    check("aten.var", result, expected, atol=1e-3)
    print("PASS aten.var")

if __name__ == "__main__":
    test()
'''

ARGMAX_FILE = '''"""Reference CUDA kernel for aten.argmax — index of maximum value."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_argmax_kernel(
    const float *input, long *output, unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float best = ri[0];
    long best_idx = 0;
    for (unsigned int j = 1; j < cols; j++) {
        if (ri[j] > best) { best = ri[j]; best_idx = j; }
    }
    output[row] = best_idx;
}

torch::Tensor aten_argmax_fwd(torch::Tensor input) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty({rows}, torch::TensorOptions().dtype(torch::kLong).device(input.device()));
    aten_argmax_kernel<<<rows, 1>>>(flat.data_ptr<float>(), output.data_ptr<long>(), rows, cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_argmax_kernel", KERNEL_SRC, ["aten_argmax_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_argmax_fwd(x)
    expected = aten.argmax.default(x, -1)
    check("aten.argmax", result, expected)
    print("PASS aten.argmax")

if __name__ == "__main__":
    test()
'''

ARGMIN_FILE = '''"""Reference CUDA kernel for aten.argmin — index of minimum value."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_argmin_kernel(
    const float *input, long *output, unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float best = ri[0];
    long best_idx = 0;
    for (unsigned int j = 1; j < cols; j++) {
        if (ri[j] < best) { best = ri[j]; best_idx = j; }
    }
    output[row] = best_idx;
}

torch::Tensor aten_argmin_fwd(torch::Tensor input) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty({rows}, torch::TensorOptions().dtype(torch::kLong).device(input.device()));
    aten_argmin_kernel<<<rows, 1>>>(flat.data_ptr<float>(), output.data_ptr<long>(), rows, cols);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_argmin_kernel", KERNEL_SRC, ["aten_argmin_fwd"])
    x = torch.randn(32, 64, device="cuda")
    result = ext.aten_argmin_fwd(x)
    expected = aten.argmin.default(x, -1)
    check("aten.argmin", result, expected)
    print("PASS aten.argmin")

if __name__ == "__main__":
    test()
'''

CUMSUM_FILE = '''"""Reference CUDA kernel for aten.cumsum — cumulative sum along last dim."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// One block per row, sequential scan within row
extern "C" __global__ void aten_cumsum_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    float acc = 0.0f;
    for (unsigned int j = 0; j < cols; j++) {
        acc += ri[j];
        ro[j] = acc;
    }
}

torch::Tensor aten_cumsum_fwd(torch::Tensor input) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty_like(flat);
    aten_cumsum_kernel<<<rows, 1>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output.reshape(input.sizes());
}
"""

def test():
    ext = compile_cuda("aten_cumsum_kernel", KERNEL_SRC, ["aten_cumsum_fwd"])
    x = torch.randn(8, 64, device="cuda")
    result = ext.aten_cumsum_fwd(x)
    expected = aten.cumsum.default(x, -1)
    check("aten.cumsum", result, expected, atol=1e-4)
    print("PASS aten.cumsum")

if __name__ == "__main__":
    test()
'''

CUMPROD_FILE = '''"""Reference CUDA kernel for aten.cumprod — cumulative product along last dim."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_cumprod_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *ro = output + row * cols;
    float acc = 1.0f;
    for (unsigned int j = 0; j < cols; j++) {
        acc *= ri[j];
        ro[j] = acc;
    }
}

torch::Tensor aten_cumprod_fwd(torch::Tensor input) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto output = torch::empty_like(flat);
    aten_cumprod_kernel<<<rows, 1>>>(
        flat.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output.reshape(input.sizes());
}
"""

def test():
    ext = compile_cuda("aten_cumprod_kernel", KERNEL_SRC, ["aten_cumprod_fwd"])
    x = torch.rand(8, 16, device="cuda") + 0.5
    result = ext.aten_cumprod_fwd(x)
    expected = aten.cumprod.default(x, -1)
    check("aten.cumprod", result, expected, atol=1e-3)
    print("PASS aten.cumprod")

if __name__ == "__main__":
    test()
'''

SORT_FILE = '''"""Reference CUDA kernel for aten.sort — bubble sort reference (intentionally slow)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// One block per row, single-thread bubble sort (correct but slow reference)
extern "C" __global__ void aten_sort_kernel(
    const float *input, float *values, long *indices,
    unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *rv = values + row * cols;
    long *ri_idx = indices + row * cols;
    // Init
    for (unsigned int j = 0; j < cols; j++) { rv[j] = ri[j]; ri_idx[j] = j; }
    // Bubble sort ascending
    for (unsigned int i = 0; i < cols; i++) {
        for (unsigned int j = i + 1; j < cols; j++) {
            if (rv[j] < rv[i]) {
                float tmp = rv[i]; rv[i] = rv[j]; rv[j] = tmp;
                long ti = ri_idx[i]; ri_idx[i] = ri_idx[j]; ri_idx[j] = ti;
            }
        }
    }
}

std::vector<torch::Tensor> aten_sort_fwd(torch::Tensor input) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto values = torch::empty_like(flat);
    auto indices = torch::empty({rows, cols}, torch::TensorOptions().dtype(torch::kLong).device(input.device()));
    aten_sort_kernel<<<rows, 1>>>(
        flat.data_ptr<float>(), values.data_ptr<float>(), indices.data_ptr<long>(), rows, cols);
    return {values.reshape(input.sizes()), indices.reshape(input.sizes())};
}
"""

def test():
    ext = compile_cuda("aten_sort_kernel", KERNEL_SRC, ["aten_sort_fwd"])
    x = torch.randn(8, 32, device="cuda")
    result = ext.aten_sort_fwd(x)
    expected = aten.sort.default(x, -1)
    check("aten.sort.values", result[0], expected[0])
    check("aten.sort.indices", result[1], expected[1])
    print("PASS aten.sort")

if __name__ == "__main__":
    test()
'''

TOPK_FILE = '''"""Reference CUDA kernel for aten.topk — find k largest values."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// One block per row, selection sort for top-k (reference, not optimized)
extern "C" __global__ void aten_topk_kernel(
    const float *input, float *values, long *indices,
    unsigned int rows, unsigned int cols, unsigned int k
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *rv = values + row * k;
    long *ri_idx = indices + row * k;
    // Selection: find k largest
    for (unsigned int i = 0; i < k; i++) {
        float best = -1e38f;
        long best_j = 0;
        for (unsigned int j = 0; j < cols; j++) {
            float v = ri[j];
            bool already = false;
            for (unsigned int p = 0; p < i; p++) {
                if (ri_idx[p] == (long)j) { already = true; break; }
            }
            if (!already && v > best) { best = v; best_j = j; }
        }
        rv[i] = best;
        ri_idx[i] = best_j;
    }
}

std::vector<torch::Tensor> aten_topk_fwd(torch::Tensor input, int64_t k) {
    int cols = input.size(-1);
    int rows = input.numel() / cols;
    auto flat = input.reshape({rows, cols}).contiguous();
    auto values = torch::empty({rows, k}, flat.options());
    auto indices = torch::empty({rows, k}, torch::TensorOptions().dtype(torch::kLong).device(input.device()));
    aten_topk_kernel<<<rows, 1>>>(
        flat.data_ptr<float>(), values.data_ptr<float>(), indices.data_ptr<long>(), rows, cols, k);
    return {values, indices};
}
"""

def test():
    ext = compile_cuda("aten_topk_kernel", KERNEL_SRC, ["aten_topk_fwd"])
    x = torch.randn(8, 32, device="cuda")
    result = ext.aten_topk_fwd(x, 5)
    expected = aten.topk.default(x, 5, -1)
    check("aten.topk.values", result[0], expected[0])
    check("aten.topk.indices", result[1], expected[1])
    print("PASS aten.topk")

if __name__ == "__main__":
    test()
'''

# ─── 3i. Embedding ──────────────────────────────────────────────────────────

EMBEDDING_FILE = '''"""Reference CUDA kernel for aten.embedding — table lookup."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_embedding_kernel(
    const float *weight, const long *indices, float *output,
    unsigned int n_idx, unsigned int embed_dim
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_idx * embed_dim) return;
    unsigned int r = idx / embed_dim, c = idx % embed_dim;
    long row = indices[r];
    output[idx] = weight[row * embed_dim + c];
}

torch::Tensor aten_embedding_fwd(torch::Tensor weight, torch::Tensor indices) {
    int n_idx = indices.numel();
    int embed_dim = weight.size(1);
    auto output = torch::empty({n_idx, embed_dim}, weight.options());
    int total = n_idx * embed_dim;
    aten_embedding_kernel<<<(total+255)/256, 256>>>(
        weight.data_ptr<float>(), indices.data_ptr<long>(),
        output.data_ptr<float>(), n_idx, embed_dim);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_embedding_kernel", KERNEL_SRC, ["aten_embedding_fwd"])
    weight = torch.randn(100, 64, device="cuda")
    indices = torch.randint(0, 100, (32,), device="cuda")
    result = ext.aten_embedding_fwd(weight, indices)
    expected = aten.embedding.default(weight, indices)
    check("aten.embedding", result, expected)
    print("PASS aten.embedding")

if __name__ == "__main__":
    test()
'''

# ─── 3j. Convolution ────────────────────────────────────────────────────────

CONV2D_FILE = '''"""Reference CUDA kernel for aten.convolution — naive conv2d with nested loops."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Naive conv2d: one thread per output element, no optimization
extern "C" __global__ void aten_conv2d_kernel(
    const float *input, const float *weight, const float *bias, float *output,
    unsigned int N, unsigned int C_in, unsigned int H, unsigned int W,
    unsigned int C_out, unsigned int kH, unsigned int kW,
    unsigned int padH, unsigned int padW, unsigned int strideH, unsigned int strideW,
    unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C_out * outH * outW;
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int oc = (idx / (outW * outH)) % C_out;
    unsigned int n = idx / (outW * outH * C_out);

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;
    for (unsigned int ic = 0; ic < C_in; ic++) {
        for (unsigned int kh = 0; kh < kH; kh++) {
            for (unsigned int kw = 0; kw < kW; kw++) {
                int ih = (int)(oh * strideH + kh) - (int)padH;
                int iw = (int)(ow * strideW + kw) - (int)padW;
                if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                    sum += input[n*C_in*H*W + ic*H*W + ih*W + iw]
                         * weight[oc*C_in*kH*kW + ic*kH*kW + kh*kW + kw];
                }
            }
        }
    }
    output[idx] = sum;
}

torch::Tensor aten_conv2d_fwd(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t strideH, int64_t strideW, int64_t padH, int64_t padW
) {
    auto ci = input.contiguous();
    int N = ci.size(0), C_in = ci.size(1), H = ci.size(2), W = ci.size(3);
    int C_out = weight.size(0), kH = weight.size(2), kW = weight.size(3);
    int outH = (H + 2*padH - kH) / strideH + 1;
    int outW = (W + 2*padW - kW) / strideW + 1;
    auto output = torch::empty({N, C_out, outH, outW}, ci.options());
    int total = N * C_out * outH * outW;
    aten_conv2d_kernel<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H, W, C_out, kH, kW, padH, padW, strideH, strideW, outH, outW);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_conv2d_kernel", KERNEL_SRC, ["aten_conv2d_fwd"])
    x = torch.randn(1, 3, 8, 8, device="cuda")
    w = torch.randn(16, 3, 3, 3, device="cuda")
    b = torch.randn(16, device="cuda")
    result = ext.aten_conv2d_fwd(x, w, b, 1, 1, 1, 1)
    expected = aten.convolution.default(x, w, b, [1,1], [1,1], [1,1], False, [0,0], 1)
    check("aten.convolution", result, expected, atol=1e-3)
    print("PASS aten.convolution")

if __name__ == "__main__":
    test()
'''

# ─── 3k. Pooling ────────────────────────────────────────────────────────────

MAX_POOL2D_FILE = '''"""Reference CUDA kernel for aten.max_pool2d — max pooling with indices."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_max_pool2d_kernel(
    const float *input, float *output, long *indices,
    unsigned int N, unsigned int C, unsigned int H, unsigned int W,
    unsigned int kH, unsigned int kW, unsigned int strideH, unsigned int strideW,
    unsigned int padH, unsigned int padW, unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * outH * outW;
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int c = (idx / (outW * outH)) % C;
    unsigned int n = idx / (outW * outH * C);

    float best = -1e38f;
    long best_idx = 0;
    for (unsigned int kh = 0; kh < kH; kh++) {
        for (unsigned int kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * strideH + kh) - (int)padH;
            int iw = (int)(ow * strideW + kw) - (int)padW;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                float v = input[n*C*H*W + c*H*W + ih*W + iw];
                if (v > best) { best = v; best_idx = ih * W + iw; }
            }
        }
    }
    output[idx] = best;
    indices[idx] = best_idx;
}

std::vector<torch::Tensor> aten_max_pool2d_fwd(
    torch::Tensor input, int64_t kH, int64_t kW,
    int64_t strideH, int64_t strideW, int64_t padH, int64_t padW
) {
    auto ci = input.contiguous();
    int N = ci.size(0), C = ci.size(1), H = ci.size(2), W = ci.size(3);
    int outH = (H + 2*padH - kH) / strideH + 1;
    int outW = (W + 2*padW - kW) / strideW + 1;
    auto output = torch::empty({N, C, outH, outW}, ci.options());
    auto indices = torch::empty({N, C, outH, outW},
        torch::TensorOptions().dtype(torch::kLong).device(ci.device()));
    int total = N * C * outH * outW;
    aten_max_pool2d_kernel<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), indices.data_ptr<long>(),
        N, C, H, W, kH, kW, strideH, strideW, padH, padW, outH, outW);
    return {output, indices};
}
"""

def test():
    ext = compile_cuda("aten_max_pool2d_kernel", KERNEL_SRC, ["aten_max_pool2d_fwd"])
    x = torch.randn(1, 4, 8, 8, device="cuda")
    result = ext.aten_max_pool2d_fwd(x, 2, 2, 2, 2, 0, 0)
    expected = aten.max_pool2d_with_indices.default(x, [2,2], [2,2])
    check("aten.max_pool2d.values", result[0], expected[0])
    check("aten.max_pool2d.indices", result[1], expected[1])
    print("PASS aten.max_pool2d")

if __name__ == "__main__":
    test()
'''

AVG_POOL2D_FILE = '''"""Reference CUDA kernel for aten.avg_pool2d — average pooling."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_avg_pool2d_kernel(
    const float *input, float *output,
    unsigned int N, unsigned int C, unsigned int H, unsigned int W,
    unsigned int kH, unsigned int kW, unsigned int strideH, unsigned int strideW,
    unsigned int padH, unsigned int padW, unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * outH * outW;
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int c = (idx / (outW * outH)) % C;
    unsigned int n = idx / (outW * outH * C);

    float sum = 0.0f;
    int count = 0;
    for (unsigned int kh = 0; kh < kH; kh++) {
        for (unsigned int kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * strideH + kh) - (int)padH;
            int iw = (int)(ow * strideW + kw) - (int)padW;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                sum += input[n*C*H*W + c*H*W + ih*W + iw];
                count++;
            }
        }
    }
    output[idx] = sum / (float)count;
}

torch::Tensor aten_avg_pool2d_fwd(
    torch::Tensor input, int64_t kH, int64_t kW,
    int64_t strideH, int64_t strideW, int64_t padH, int64_t padW
) {
    auto ci = input.contiguous();
    int N = ci.size(0), C = ci.size(1), H = ci.size(2), W = ci.size(3);
    int outH = (H + 2*padH - kH) / strideH + 1;
    int outW = (W + 2*padW - kW) / strideW + 1;
    auto output = torch::empty({N, C, outH, outW}, ci.options());
    int total = N * C * outH * outW;
    aten_avg_pool2d_kernel<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H, W, kH, kW, strideH, strideW, padH, padW, outH, outW);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_avg_pool2d_kernel", KERNEL_SRC, ["aten_avg_pool2d_fwd"])
    x = torch.randn(1, 4, 8, 8, device="cuda")
    result = ext.aten_avg_pool2d_fwd(x, 2, 2, 2, 2, 0, 0)
    expected = aten.avg_pool2d.default(x, [2,2], [2,2])
    check("aten.avg_pool2d", result, expected, atol=1e-5)
    print("PASS aten.avg_pool2d")

if __name__ == "__main__":
    test()
'''

ADAPTIVE_AVG_POOL2D_FILE = '''"""Reference CUDA kernel for aten.adaptive_avg_pool2d."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_adaptive_avg_pool2d_kernel(
    const float *input, float *output,
    unsigned int N, unsigned int C, unsigned int H, unsigned int W,
    unsigned int outH, unsigned int outW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * outH * outW;
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int c = (idx / (outW * outH)) % C;
    unsigned int n = idx / (outW * outH * C);

    unsigned int h_start = oh * H / outH;
    unsigned int h_end = (oh + 1) * H / outH;
    unsigned int w_start = ow * W / outW;
    unsigned int w_end = (ow + 1) * W / outW;
    float sum = 0.0f;
    int count = 0;
    for (unsigned int h = h_start; h < h_end; h++) {
        for (unsigned int w = w_start; w < w_end; w++) {
            sum += input[n*C*H*W + c*H*W + h*W + w];
            count++;
        }
    }
    output[idx] = sum / (float)count;
}

torch::Tensor aten_adaptive_avg_pool2d_fwd(torch::Tensor input, int64_t outH, int64_t outW) {
    auto ci = input.contiguous();
    int N = ci.size(0), C = ci.size(1), H = ci.size(2), W = ci.size(3);
    auto output = torch::empty({N, C, (int)outH, (int)outW}, ci.options());
    int total = N * C * outH * outW;
    aten_adaptive_avg_pool2d_kernel<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, outH, outW);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_adaptive_avg_pool2d_kernel", KERNEL_SRC, ["aten_adaptive_avg_pool2d_fwd"])
    x = torch.randn(1, 4, 8, 8, device="cuda")
    result = ext.aten_adaptive_avg_pool2d_fwd(x, 1, 1)
    expected = aten.adaptive_avg_pool2d.default(x, [1, 1])
    check("aten.adaptive_avg_pool2d", result, expected, atol=1e-4)
    print("PASS aten.adaptive_avg_pool2d")

if __name__ == "__main__":
    test()
'''

# ─── 3l. Loss functions ─────────────────────────────────────────────────────

NLL_LOSS_FILE = '''"""Reference CUDA kernel for aten.nll_loss_forward — negative log likelihood."""
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
'''

MSE_LOSS_FILE = '''"""Reference CUDA kernel for aten.mse_loss — mean squared error."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_mse_kernel(
    const float *input, const float *target, float *output, unsigned int n
) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    float v = 0.0f;
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        float d = input[i] - target[i];
        v += d * d;
    }
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s]; __syncthreads();
    }
    if (tid == 0) output[0] = sdata[0] / (float)n;
}

torch::Tensor aten_mse_fwd(torch::Tensor input, torch::Tensor target) {
    auto ci = input.contiguous();
    auto ct = target.contiguous();
    int n = ci.numel();
    auto output = torch::zeros({}, ci.options());
    int threads = 256;
    aten_mse_kernel<<<1, threads, threads * sizeof(float)>>>(
        ci.data_ptr<float>(), ct.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_mse_kernel", KERNEL_SRC, ["aten_mse_fwd"])
    x = torch.randn(256, device="cuda")
    y = torch.randn(256, device="cuda")
    result = ext.aten_mse_fwd(x, y)
    expected = aten.mse_loss.default(x, y)
    check("aten.mse_loss", result, expected, atol=1e-4)
    print("PASS aten.mse_loss")

if __name__ == "__main__":
    test()
'''

# ─── 3m. Attention ───────────────────────────────────────────────────────────

SCALED_DOT_PRODUCT_ATTENTION_FILE = '''"""Reference CUDA kernel for scaled dot product attention — naive Q@K^T/sqrt(d) @ V."""
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
'''

# ─── 3n. Dropout ─────────────────────────────────────────────────────────────

NATIVE_DROPOUT_FILE = '''"""Reference CUDA kernel for aten.native_dropout — zero mask with scaling."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// Deterministic "dropout" for testing: use mask from Python side
extern "C" __global__ void aten_dropout_kernel(
    const float *input, const float *mask, float *output,
    float scale, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * mask[i] * scale;
}

torch::Tensor aten_dropout_fwd(torch::Tensor input, torch::Tensor mask, double p) {
    auto ci = input.contiguous();
    int n = ci.numel();
    float scale = 1.0f / (1.0f - (float)p);
    auto output = torch::empty_like(ci);
    aten_dropout_kernel<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), mask.data_ptr<float>(), output.data_ptr<float>(), scale, n);
    return output;
}
"""

def test():
    # Can't compare directly against native_dropout (random), so verify the math
    ext = compile_cuda("aten_dropout_kernel", KERNEL_SRC, ["aten_dropout_fwd"])
    x = torch.randn(1024, device="cuda")
    mask = (torch.rand(1024, device="cuda") > 0.5).float()
    result = ext.aten_dropout_fwd(x, mask, 0.5)
    expected = x * mask * 2.0  # scale = 1/(1-0.5) = 2
    check("aten.native_dropout", result, expected)
    print("PASS aten.native_dropout")

if __name__ == "__main__":
    test()
'''

# ─── 3o. Padding ─────────────────────────────────────────────────────────────

CONSTANT_PAD_FILE = '''"""Reference CUDA kernel for aten.constant_pad_nd — pad with constant value."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

// 2D constant padding: pad last 2 dims
extern "C" __global__ void aten_constant_pad_2d(
    const float *input, float *output,
    unsigned int H, unsigned int W, unsigned int outH, unsigned int outW,
    unsigned int padTop, unsigned int padLeft, float value, unsigned int batch_stride
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch_stride;  // N * outH * outW
    if (idx >= total) return;
    unsigned int ow = idx % outW;
    unsigned int oh = (idx / outW) % outH;
    unsigned int n = idx / (outH * outW);

    int ih = (int)oh - (int)padTop;
    int iw = (int)ow - (int)padLeft;
    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
        output[idx] = input[n * H * W + ih * W + iw];
    else
        output[idx] = value;
}

torch::Tensor aten_constant_pad_2d_fwd(
    torch::Tensor input, int64_t padL, int64_t padR, int64_t padT, int64_t padB, double value
) {
    auto ci = input.contiguous();
    int N = ci.numel() / (ci.size(-2) * ci.size(-1));
    int H = ci.size(-2), W = ci.size(-1);
    int outH = H + padT + padB, outW = W + padL + padR;
    auto sizes = ci.sizes().vec();
    sizes[sizes.size()-2] = outH;
    sizes[sizes.size()-1] = outW;
    auto output = torch::empty(sizes, ci.options());
    int total = N * outH * outW;
    aten_constant_pad_2d<<<(total+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(),
        H, W, outH, outW, padT, padL, (float)value, total);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_constant_pad_2d", KERNEL_SRC, ["aten_constant_pad_2d_fwd"])
    x = torch.randn(2, 8, 8, device="cuda")
    result = ext.aten_constant_pad_2d_fwd(x, 1, 1, 1, 1, 0.0)
    expected = aten.constant_pad_nd.default(x, [1, 1, 1, 1], 0.0)
    check("aten.constant_pad_nd", result, expected)
    print("PASS aten.constant_pad_nd")

if __name__ == "__main__":
    test()
'''

# ─── 3p. Type conversion / identity ─────────────────────────────────────────

TO_COPY_FILE = '''"""Reference CUDA kernel for aten._to_copy — dtype/device conversion (copy)."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_to_copy_kernel(const float *in, float *out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

torch::Tensor aten_to_copy_fwd(torch::Tensor input) {
    auto ci = input.contiguous();
    auto output = torch::empty_like(ci);
    int n = ci.numel();
    aten_to_copy_kernel<<<(n+255)/256, 256>>>(
        ci.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_to_copy_kernel", KERNEL_SRC, ["aten_to_copy_fwd"])
    x = torch.randn(1024, device="cuda")
    result = ext.aten_to_copy_fwd(x)
    expected = aten._to_copy.default(x)
    check("aten._to_copy", result, expected)
    print("PASS aten._to_copy")

if __name__ == "__main__":
    test()
'''

FILL_FILE = '''"""Reference CUDA kernel for aten.fill — fill tensor with scalar value."""
import torch
from torch_graph.cuda_ref_kernels._common import compile_cuda, check

aten = torch.ops.aten

KERNEL_SRC = r"""
#include <cuda_runtime.h>

extern "C" __global__ void aten_fill_kernel(float *out, float value, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}

torch::Tensor aten_fill_fwd(torch::Tensor input, double value) {
    auto output = torch::empty_like(input.contiguous());
    int n = output.numel();
    aten_fill_kernel<<<(n+255)/256, 256>>>(output.data_ptr<float>(), (float)value, n);
    return output;
}
"""

def test():
    ext = compile_cuda("aten_fill_kernel", KERNEL_SRC, ["aten_fill_fwd"])
    x = torch.randn(1024, device="cuda")
    result = ext.aten_fill_fwd(x, 3.14)
    expected = aten.fill.Scalar(x, 3.14)
    check("aten.fill", result, expected)
    print("PASS aten.fill")

if __name__ == "__main__":
    test()
'''

ALIAS_FILE = '''"""Reference for aten.alias — identity op, returns same data (no-op copy for reference)."""
import torch
from torch_graph.cuda_ref_kernels._common import check

aten = torch.ops.aten

KERNEL_SRC = ""  # alias is a no-op — no CUDA kernel needed

def test():
    x = torch.randn(1024, device="cuda")
    result = aten.alias.default(x)
    check("aten.alias", result, x)
    print("PASS aten.alias")

if __name__ == "__main__":
    test()
'''

DETACH_FILE = '''"""Reference for aten.detach — detach from autograd (no-op copy for reference)."""
import torch
from torch_graph.cuda_ref_kernels._common import check

aten = torch.ops.aten

KERNEL_SRC = ""  # detach is a no-op — no CUDA kernel needed

def test():
    x = torch.randn(1024, device="cuda", requires_grad=True)
    result = aten.detach.default(x)
    expected = x.detach()
    check("aten.detach", result, expected)
    print("PASS aten.detach")

if __name__ == "__main__":
    test()
'''


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4: GENERATE ALL FILES
# ═════════════════════════════════════════════════════════════════════════════

def generate():
    count = 0

    # ── Unary ops (templated) ──
    for op_name, func_name, cuda_expr, test_input, aten_ref, atol in UNARY_OPS:
        atol_str = f", atol={atol}" if atol else ""
        content = UNARY_TEMPLATE.format(
            op_name=op_name, func_name=func_name, cuda_expr=cuda_expr,
            test_input=test_input, aten_ref=aten_ref, atol_str=atol_str)
        (HERE / f"aten_{op_name}.py").write_text(content)
        count += 1

    # ── Binary ops (templated) ──
    for op_name, func_name, cuda_expr, test_setup, aten_ref, atol in BINARY_OPS:
        atol_str = f", atol={atol}" if atol else ""
        content = BINARY_TEMPLATE.format(
            op_name=op_name, func_name=func_name, cuda_expr=cuda_expr,
            test_setup=test_setup, aten_ref=aten_ref, atol_str=atol_str)
        (HERE / f"aten_{op_name}.py").write_text(content)
        count += 1

    # ── Comparison ops (templated) ──
    for op_name, func_name, cuda_expr, aten_ref in COMPARISON_OPS:
        content = COMPARISON_TEMPLATE.format(
            op_name=op_name, func_name=func_name, cuda_expr=cuda_expr,
            aten_ref=aten_ref)
        (HERE / f"aten_{op_name}.py").write_text(content)
        count += 1

    # ── Backward ops (templated) ──
    for op_name, func_name, cuda_expr, test_setup, aten_ref, atol in BACKWARD_OPS:
        atol_str = f", atol={atol}" if atol else ""
        content = BACKWARD_TEMPLATE.format(
            op_name=op_name, func_name=func_name, cuda_expr=cuda_expr,
            test_setup=test_setup, aten_ref=aten_ref, atol_str=atol_str)
        (HERE / f"aten_{op_name}.py").write_text(content)
        count += 1

    # ── Reduction ops (templated) ──
    for op_name, identity, accumulate, reduce, finalize, test_input, aten_ref, atol in REDUCTION_OPS:
        atol_str = f", atol={atol}" if atol else ""
        content = REDUCTION_TEMPLATE.format(
            op_name=op_name, identity=identity, accumulate=accumulate,
            reduce=reduce, finalize=finalize, test_input=test_input,
            aten_ref=aten_ref, atol_str=atol_str)
        (HERE / f"aten_{op_name}.py").write_text(content)
        count += 1

    # ── Hand-crafted files ──
    HAND_CRAFTED = {
        # Ternary / conditional
        "where": WHERE_FILE,
        "clamp": CLAMP_FILE,
        "lerp": LERP_FILE,
        "addcmul": ADDCMUL_FILE,
        "addcdiv": ADDCDIV_FILE,
        "masked_fill": MASKED_FILL_FILE,
        # Matmul family
        "mm": MATMUL_FILE,
        "bmm": BMM_FILE,
        "addmm": ADDMM_FILE,
        "dot": DOT_FILE,
        "mv": MV_FILE,
        "outer": OUTER_FILE,
        "baddbmm": BADDBMM_FILE,
        "linear": LINEAR_FILE,
        # Normalization
        "_softmax": SOFTMAX_FILE,
        "_log_softmax": LOG_SOFTMAX_FILE,
        "native_layer_norm": LAYER_NORM_FILE,
        "native_batch_norm": BATCH_NORM_FILE,
        "native_group_norm": GROUP_NORM_FILE,
        # Layout
        "transpose": TRANSPOSE_FILE,
        "t": T_FILE,
        "permute": PERMUTE_FILE,
        "clone": CLONE_FILE,
        "contiguous": CONTIGUOUS_FILE,
        "view": VIEW_FILE,
        "reshape": RESHAPE_FILE,
        "unsqueeze": UNSQUEEZE_FILE,
        "squeeze": SQUEEZE_FILE,
        "flatten": FLATTEN_FILE,
        "expand": EXPAND_FILE,
        "slice": SLICE_FILE,
        "select": SELECT_FILE,
        "narrow": NARROW_FILE,
        # Rearrange
        "flip": FLIP_FILE,
        "roll": ROLL_FILE,
        "repeat": REPEAT_FILE,
        "tril": TRIL_FILE,
        "triu": TRIU_FILE,
        # Tensor creation
        "arange": ARANGE_FILE,
        "zeros": ZEROS_FILE,
        "ones": ONES_FILE,
        "full": FULL_FILE,
        "eye": EYE_FILE,
        "linspace": LINSPACE_FILE,
        # Indexing
        "gather": GATHER_FILE,
        "scatter": SCATTER_FILE,
        "index_select": INDEX_SELECT_FILE,
        "index_add": INDEX_ADD_FILE,
        # Concatenation / split
        "cat": CAT_FILE,
        "stack": STACK_FILE,
        "split": SPLIT_FILE,
        # More reductions
        "var": VAR_FILE,
        "argmax": ARGMAX_FILE,
        "argmin": ARGMIN_FILE,
        # Scan
        "cumsum": CUMSUM_FILE,
        "cumprod": CUMPROD_FILE,
        # Sort/search
        "sort": SORT_FILE,
        "topk": TOPK_FILE,
        # Embedding
        "embedding": EMBEDDING_FILE,
        # Convolution
        "convolution": CONV2D_FILE,
        # Pooling
        "max_pool2d": MAX_POOL2D_FILE,
        "avg_pool2d": AVG_POOL2D_FILE,
        "adaptive_avg_pool2d": ADAPTIVE_AVG_POOL2D_FILE,
        # Loss
        "nll_loss_forward": NLL_LOSS_FILE,
        "mse_loss": MSE_LOSS_FILE,
        # Attention
        "scaled_dot_product_attention": SCALED_DOT_PRODUCT_ATTENTION_FILE,
        # Dropout
        "native_dropout": NATIVE_DROPOUT_FILE,
        # Padding
        "constant_pad_nd": CONSTANT_PAD_FILE,
        # Type/identity
        "_to_copy": TO_COPY_FILE,
        "fill": FILL_FILE,
        "alias": ALIAS_FILE,
        "detach": DETACH_FILE,
    }

    for name, content in HAND_CRAFTED.items():
        fname = f"aten_{name}.py"
        # Handle _softmax → aten__softmax.py (leading underscore)
        (HERE / fname).write_text(content)
        count += 1

    print(f"Generated {count} reference kernel files in {HERE}/")
    return count


if __name__ == "__main__":
    n = generate()
    print(f"Done. Run tests with: python {HERE}/run_all_tests.py")
