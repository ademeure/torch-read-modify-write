#!/usr/bin/env python3
"""Generate reference CUDA kernel files for every aten op (kernelbox format).

Each generated file is self-contained with:
  KERNEL_SRC: raw CUDA kernel string (extern "C" __global__ only, no C++ wrappers)
  init_once(): create inputs, compute expected outputs via torch.ops.aten.*, return state dict
  run(inputs, kernel): execute the kernel and return results

Uses kernelbox for compilation (no load_inline / ninja needed).
Run with: kbox iterate torch_graph/cuda_ref_kernels/aten_add.py --once
Test all:  python torch_graph/cuda_ref_kernels/run_all_tests.py

Coverage: 151 aten ops across all categories. Every op has a real CUDA kernel
except alias and detach (true no-ops).

Run: python torch_graph/cuda_ref_kernels/generate_all.py
"""

from pathlib import Path

HERE = Path(__file__).parent


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1: TEMPLATES
# ═════════════════════════════════════════════════════════════════════════════

# ─── 1a. Elementwise unary (in0, out0, n) — standard kbox pattern ────────────

UNARY_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name}."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void {func_name}(const float *in0, float *out0, unsigned int n) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{ float x = in0[i]; out0[i] = {cuda_expr}; }}
}}
"""

ATOL = {atol_val}

def make_inputs(n=1024, seed=1):
    """seed=0 → special values (nan/inf/0/1/etc), seed>0 → seeded random."""
    if seed == 0:
        special = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 100.0, -100.0, 1e-7, 1e7, float("nan"), float("inf"), float("-inf")], device="cuda")
        return [special.repeat((n + len(special) - 1) // len(special))[:n]]
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [{test_input_seeded}]

def expected(inputs):
    x = inputs[0]
    return [{aten_ref_fn}]

def init_once():
    inputs = make_inputs()
    return {{"kernel_source": KERNEL_SRC, "inputs": inputs, "expected": expected(inputs), "atol": ATOL}}

def run(inputs, kernel):
    return [kernel(*inputs)]
'''

# ─── 1b. Elementwise binary (in0, in1, out0, n) ─────────────────────────────

BINARY_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name}."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void {func_name}(const float *in0, const float *in1, float *out0, unsigned int n) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{ float a = in0[i], b = in1[i]; out0[i] = {cuda_expr}; }}
}}
"""

ATOL = {atol_val}

def make_inputs(n=1024, seed=1):
    if seed == 0:
        special = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 100.0, -100.0, 1e-7, 1e7, float("nan"), float("inf"), float("-inf")], device="cuda")
        s = special.repeat((n + len(special) - 1) // len(special))[:n]
        return [s, s.flip(0)]
    g = torch.Generator(device="cuda").manual_seed(seed)
    {test_setup_seeded}
    return [a, b]

def expected(inputs):
    a, b = inputs
    return [{aten_ref_fn}]

def init_once():
    inputs = make_inputs()
    return {{"kernel_source": KERNEL_SRC, "inputs": inputs, "expected": expected(inputs), "atol": ATOL}}

def run(inputs, kernel):
    return [kernel(*inputs)]
'''

# ─── 1c. Comparison (in0, in1, out0, n) ─────────────────────────────────────

COMPARISON_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name}."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void {func_name}(const float *in0, const float *in1, float *out0, unsigned int n) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{ float a = in0[i], b = in1[i]; out0[i] = {cuda_expr}; }}
}}
"""

def make_inputs(n=1024, seed=1):
    if seed == 0:
        special = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 100.0, -100.0, float("nan"), float("inf"), float("-inf")], device="cuda")
        s = special.repeat((n + len(special) - 1) // len(special))[:n]
        return [s, s.flip(0)]
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(n, device="cuda", generator=g), torch.randn(n, device="cuda", generator=g)]

def expected(inputs):
    a, b = inputs
    return [{aten_ref_fn}.float()]

def init_once():
    inputs = make_inputs()
    return {{"kernel_source": KERNEL_SRC, "inputs": inputs, "expected": expected(inputs)}}

def run(inputs, kernel):
    return [kernel(*inputs)]
'''

# ─── 1d. Backward gradient (grad, saved, out0, n) ───────────────────────────

BACKWARD_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name} (backward gradient op)."""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void {func_name}(const float *grad, const float *saved, float *out0, unsigned int n) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{ float g = grad[i], s = saved[i]; out0[i] = {cuda_expr}; }}
}}
"""

ATOL = {atol_val}

def make_inputs(n=1024, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    {test_setup_seeded}
    return [grad, saved]

def expected(inputs):
    grad, saved = inputs
    return [{aten_ref_fn}]

def init_once():
    inputs = make_inputs()
    return {{"kernel_source": KERNEL_SRC, "inputs": inputs, "expected": expected(inputs), "atol": ATOL}}

def run(inputs, kernel):
    return [kernel(*inputs)]
'''

# ─── 1e. Reduction (input, output, rows, cols) — custom params, static smem ──

REDUCTION_TEMPLATE = '''"""Reference CUDA kernel for aten.{op_name}.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_{op_name}.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_{op_name}(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {{
    __shared__ float sdata[256];
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
"""

def init_once():
    x = {test_input}
    cols = x.size(-1)
    rows = x.numel() // cols
    return {{
        "kernel_source": KERNEL_SRC,
        "inputs": [x.reshape(rows, cols).contiguous()],
        "expected": [{aten_ref}],
        "outputs": ["float32;n=%d" % rows],
        "grid": (rows,),
        "block": (256,),{atol_str}
    }}

def run(inputs, kernel):
    x = inputs[0]
    rows, cols = x.shape
    return [kernel(x, params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(rows), np.uint32(cols),
    ])]
'''


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2: OP TABLES
# ═════════════════════════════════════════════════════════════════════════════

UNARY_OPS = [
    # (op_name, func_name, cuda_expr, test_input, aten_ref, atol)
    ("relu", "aten_relu", "((x > 0.0f) ? x : 0.0f)", "torch.randn(1024, device='cuda')", "torch.ops.aten.relu.default(x)", None),
    ("relu6", "aten_relu6", "fminf(fmaxf(x, 0.0f), 6.0f)", "torch.randn(1024, device='cuda') * 5", "torch.ops.aten.hardtanh.default(x, 0.0, 6.0)", 1e-5),
    ("gelu", "aten_gelu", "(x * 0.5f * (1.0f + erff(x * 0.7071067811865476f)))", "torch.randn(1024, device='cuda')", "torch.ops.aten.gelu.default(x)", 1e-5),
    ("silu", "aten_silu", "(x / (1.0f + expf(-x)))", "torch.randn(1024, device='cuda')", "torch.ops.aten.silu.default(x)", 1e-5),
    ("sigmoid", "aten_sigmoid", "(1.0f / (1.0f + expf(-x)))", "torch.randn(1024, device='cuda')", "torch.ops.aten.sigmoid.default(x)", 1e-5),
    ("tanh", "aten_tanh", "tanhf(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.tanh.default(x)", 1e-6),
    ("hardswish", "aten_hardswish", "(x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f)", "torch.randn(1024, device='cuda') * 5", "torch.ops.aten.hardswish.default(x)", 1e-5),
    ("hardsigmoid", "aten_hardsigmoid", "fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f)", "torch.randn(1024, device='cuda') * 5", "torch.ops.aten.hardsigmoid.default(x)", 1e-5),
    ("hardtanh", "aten_hardtanh", "fminf(fmaxf(x, -1.0f), 1.0f)", "torch.randn(1024, device='cuda') * 3", "torch.ops.aten.hardtanh.default(x)", 1e-5),
    ("softplus", "aten_softplus", "((x > 20.0f) ? x : logf(1.0f + expf(x)))", "torch.randn(1024, device='cuda') * 5", "torch.ops.aten.softplus.default(x)", 1e-4),
    ("mish", "aten_mish", "(x * tanhf((x > 20.0f) ? x : logf(1.0f + expf(x))))", "torch.randn(1024, device='cuda')", "torch.ops.aten.mish.default(x)", 1e-4),
    ("elu", "aten_elu", "((x > 0.0f) ? x : (expf(x) - 1.0f))", "torch.randn(1024, device='cuda')", "torch.ops.aten.elu.default(x)", 1e-5),
    ("leaky_relu", "aten_leaky_relu", "((x > 0.0f) ? x : 0.01f * x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.leaky_relu.default(x, 0.01)", 1e-6),
    ("log_sigmoid", "aten_log_sigmoid_forward", "(-logf(1.0f + expf(-x)))", "torch.randn(1024, device='cuda')", "torch.ops.aten.log_sigmoid_forward.default(x)[0]", 1e-5),
    ("abs", "aten_abs", "fabsf(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.abs.default(x)", None),
    ("neg", "aten_neg", "(-x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.neg.default(x)", None),
    ("exp", "aten_exp", "expf(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.exp.default(x)", 1e-5),
    ("exp2", "aten_exp2", "exp2f(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.exp2.default(x)", 1e-5),
    ("expm1", "aten_expm1", "expm1f(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.expm1.default(x)", 1e-5),
    ("log", "aten_log", "logf(x)", "torch.rand(1024, device='cuda') + 0.01", "torch.ops.aten.log.default(x)", 1e-5),
    ("log2", "aten_log2", "log2f(x)", "torch.rand(1024, device='cuda') + 0.01", "torch.ops.aten.log2.default(x)", 1e-5),
    ("log10", "aten_log10", "log10f(x)", "torch.rand(1024, device='cuda') + 0.01", "torch.ops.aten.log10.default(x)", 1e-5),
    ("log1p", "aten_log1p", "log1pf(x)", "torch.rand(1024, device='cuda')", "torch.ops.aten.log1p.default(x)", 1e-5),
    ("sqrt", "aten_sqrt", "sqrtf(x)", "torch.rand(1024, device='cuda') + 0.01", "torch.ops.aten.sqrt.default(x)", 1e-6),
    ("rsqrt", "aten_rsqrt", "rsqrtf(x)", "torch.rand(1024, device='cuda') + 0.01", "torch.ops.aten.rsqrt.default(x)", 1e-4),
    ("reciprocal", "aten_reciprocal", "(1.0f / x)", "torch.randn(1024, device='cuda').abs() + 0.1", "torch.ops.aten.reciprocal.default(x)", 1e-5),
    ("ceil", "aten_ceil", "ceilf(x)", "torch.randn(1024, device='cuda') * 10", "torch.ops.aten.ceil.default(x)", None),
    ("floor", "aten_floor", "floorf(x)", "torch.randn(1024, device='cuda') * 10", "torch.ops.aten.floor.default(x)", None),
    ("round", "aten_round", "nearbyintf(x)", "torch.randn(1024, device='cuda') * 10", "torch.ops.aten.round.default(x)", None),
    ("trunc", "aten_trunc", "truncf(x)", "torch.randn(1024, device='cuda') * 10", "torch.ops.aten.trunc.default(x)", None),
    ("frac", "aten_frac", "(x - truncf(x))", "torch.randn(1024, device='cuda') * 10", "torch.ops.aten.frac.default(x)", 1e-5),
    ("sign", "aten_sign", "((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))", "torch.randn(1024, device='cuda')", "torch.ops.aten.sign.default(x)", None),
    ("sgn", "aten_sgn", "((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))", "torch.randn(1024, device='cuda')", "torch.ops.aten.sgn.default(x)", None),
    ("sin", "aten_sin", "sinf(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.sin.default(x)", 1e-5),
    ("cos", "aten_cos", "cosf(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.cos.default(x)", 1e-5),
    ("tan", "aten_tan", "tanf(x)", "torch.randn(1024, device='cuda') * 0.5", "torch.ops.aten.tan.default(x)", 1e-4),
    ("asin", "aten_asin", "asinf(x)", "torch.rand(1024, device='cuda') * 1.98 - 0.99", "torch.ops.aten.asin.default(x)", 1e-5),
    ("acos", "aten_acos", "acosf(x)", "torch.rand(1024, device='cuda') * 1.98 - 0.99", "torch.ops.aten.acos.default(x)", 1e-5),
    ("atan", "aten_atan", "atanf(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.atan.default(x)", 1e-5),
    ("sinh", "aten_sinh", "sinhf(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.sinh.default(x)", 1e-4),
    ("cosh", "aten_cosh", "coshf(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.cosh.default(x)", 1e-4),
    ("asinh", "aten_asinh", "asinhf(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.asinh.default(x)", 1e-5),
    ("acosh", "aten_acosh", "acoshf(x)", "torch.rand(1024, device='cuda') + 1.01", "torch.ops.aten.acosh.default(x)", 1e-5),
    ("atanh", "aten_atanh", "atanhf(x)", "torch.rand(1024, device='cuda') * 1.98 - 0.99", "torch.ops.aten.atanh.default(x)", 1e-5),
    ("erf", "aten_erf", "erff(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.erf.default(x)", 1e-5),
    ("erfc", "aten_erfc", "erfcf(x)", "torch.randn(1024, device='cuda')", "torch.ops.aten.erfc.default(x)", 1e-5),
    ("isnan", "aten_isnan", "(isnan(x) ? 1.0f : 0.0f)", "torch.tensor([1.0, float('nan'), 0.0, float('nan'), -1.0] * 200, device='cuda')", "torch.ops.aten.isnan.default(x).float()", None),
    ("isinf", "aten_isinf", "(isinf(x) ? 1.0f : 0.0f)", "torch.tensor([1.0, float('inf'), 0.0, float('-inf'), -1.0] * 200, device='cuda')", "torch.ops.aten.isinf.default(x).float()", None),
    ("isfinite", "aten_isfinite", "((isfinite(x)) ? 1.0f : 0.0f)", "torch.tensor([1.0, float('inf'), 0.0, float('nan'), -1.0] * 200, device='cuda')", "torch.isfinite(x).float()", None),
    ("logical_not", "aten_logical_not", "((x == 0.0f) ? 1.0f : 0.0f)", "torch.tensor([1.0, 0.0, -1.0, 0.0, 3.14] * 200, device='cuda')", "torch.ops.aten.logical_not.default(x).float()", None),
    ("bitwise_not", "aten_bitwise_not", "((x == 0.0f) ? 1.0f : 0.0f)", "torch.tensor([1.0, 0.0, -1.0, 0.0, 3.14] * 200, device='cuda')", "torch.ops.aten.logical_not.default(x).float()", None),
]

BINARY_OPS = [
    ("add", "aten_add", "(a + b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.ops.aten.add.Tensor(a, b)", None),
    ("sub", "aten_sub", "(a - b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.ops.aten.sub.Tensor(a, b)", None),
    ("mul", "aten_mul", "(a * b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.ops.aten.mul.Tensor(a, b)", None),
    ("div", "aten_div", "(a / b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda').abs() + 0.1",
     "torch.ops.aten.div.Tensor(a, b)", 1e-5),
    ("pow", "aten_pow", "powf(a, b)",
     "a = torch.rand(1024, device='cuda') + 0.1\n    b = torch.rand(1024, device='cuda') * 3",
     "torch.ops.aten.pow.Tensor_Tensor(a, b)", 1e-4),
    ("maximum", "aten_maximum", "fmaxf(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.ops.aten.maximum.default(a, b)", None),
    ("minimum", "aten_minimum", "fminf(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.ops.aten.minimum.default(a, b)", None),
    ("atan2", "aten_atan2", "atan2f(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.ops.aten.atan2.default(a, b)", 1e-5),
    ("fmod", "aten_fmod", "fmodf(a, b)",
     "a = torch.randn(1024, device='cuda') * 10\n    b = torch.randn(1024, device='cuda').abs() + 0.5",
     "torch.ops.aten.fmod.Tensor(a, b)", 1e-5),
    ("remainder", "aten_remainder", "(a - b * floorf(a / b))",
     "a = torch.randn(1024, device='cuda') * 10\n    b = torch.randn(1024, device='cuda').abs() + 0.5",
     "torch.ops.aten.remainder.Tensor(a, b)", 1e-4),
    ("hypot", "aten_hypot", "hypotf(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.ops.aten.hypot.default(a, b)", 1e-5),
    ("copysign", "aten_copysign", "copysignf(a, b)",
     "a = torch.randn(1024, device='cuda')\n    b = torch.randn(1024, device='cuda')",
     "torch.ops.aten.copysign.Tensor(a, b)", None),
]

COMPARISON_OPS = [
    ("eq", "aten_eq", "(a == b ? 1.0f : 0.0f)", "torch.ops.aten.eq.Tensor(a, b)"),
    ("ne", "aten_ne", "(a != b ? 1.0f : 0.0f)", "torch.ops.aten.ne.Tensor(a, b)"),
    ("gt", "aten_gt", "(a > b ? 1.0f : 0.0f)", "torch.ops.aten.gt.Tensor(a, b)"),
    ("ge", "aten_ge", "(a >= b ? 1.0f : 0.0f)", "torch.ops.aten.ge.Tensor(a, b)"),
    ("lt", "aten_lt", "(a < b ? 1.0f : 0.0f)", "torch.ops.aten.lt.Tensor(a, b)"),
    ("le", "aten_le", "(a <= b ? 1.0f : 0.0f)", "torch.ops.aten.le.Tensor(a, b)"),
]

BACKWARD_OPS = [
    ("threshold_backward", "aten_threshold_backward",
     "(s > 0.0f ? g : 0.0f)",
     "grad = torch.randn(1024, device='cuda')\n    saved = torch.randn(1024, device='cuda')",
     "torch.ops.aten.threshold_backward.default(grad, saved, 0.0)", 1e-5),
    ("gelu_backward", "aten_gelu_backward",
     "(g * (0.5f * (1.0f + erff(s * 0.7071067811865476f)) + "
     "s * 0.3989422804014327f * expf(-0.5f * s * s)))",
     "grad = torch.randn(1024, device='cuda')\n    saved = torch.randn(1024, device='cuda')",
     "torch.ops.aten.gelu_backward.default(grad, saved)", 1e-4),
    ("silu_backward", "aten_silu_backward",
     "(g * (1.0f / (1.0f + expf(-s))) * "
     "(1.0f + s * (1.0f - 1.0f / (1.0f + expf(-s)))))",
     "grad = torch.randn(1024, device='cuda')\n    saved = torch.randn(1024, device='cuda')",
     "torch.ops.aten.silu_backward.default(grad, saved)", 1e-4),
    ("sigmoid_backward", "aten_sigmoid_backward",
     "(g * s * (1.0f - s))",
     "grad = torch.randn(1024, device='cuda')\n    saved = torch.sigmoid(torch.randn(1024, device='cuda'))",
     "torch.ops.aten.sigmoid_backward.default(grad, saved)", 1e-5),
    ("tanh_backward", "aten_tanh_backward",
     "(g * (1.0f - s * s))",
     "grad = torch.randn(1024, device='cuda')\n    saved = torch.tanh(torch.randn(1024, device='cuda'))",
     "torch.ops.aten.tanh_backward.default(grad, saved)", 1e-5),
]

REDUCTION_OPS = [
    ("sum", "0.0f", "v += ri[j];",
     "sdata[tid] += sdata[tid + s];", "sdata[0]",
     "torch.randn(32, 64, device='cuda')", "torch.ops.aten.sum.dim_IntList(x, [-1])", 1e-3),
    ("mean", "0.0f", "v += ri[j];",
     "sdata[tid] += sdata[tid + s];", "sdata[0] / (float)cols",
     "torch.randn(32, 64, device='cuda')", "torch.ops.aten.mean.dim(x, [-1])", 1e-4),
    ("amax", "-1e38f", "v = fmaxf(v, ri[j]);",
     "sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);", "sdata[0]",
     "torch.randn(32, 64, device='cuda')", "torch.ops.aten.amax.default(x, [-1])", None),
    ("amin", "1e38f", "v = fminf(v, ri[j]);",
     "sdata[tid] = fminf(sdata[tid], sdata[tid + s]);", "sdata[0]",
     "torch.randn(32, 64, device='cuda')", "torch.ops.aten.amin.default(x, [-1])", None),
    ("prod", "1.0f", "v *= ri[j];",
     "sdata[tid] *= sdata[tid + s];", "sdata[0]",
     "torch.rand(8, 16, device='cuda') + 0.5", "torch.ops.aten.prod.dim_int(x, -1)", 1e-2),
]


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3: HAND-CRAFTED FILES (complex ops)
#  Each uses the kernelbox init_once/run pattern.
#  No aten alias — uses torch.ops.aten.X.Y() directly.
#  No #include lines — NVRTC has builtins.
#  No C++ wrappers — only extern "C" __global__ kernels.
# ═════════════════════════════════════════════════════════════════════════════

HAND_CRAFTED = {}

# ─── Ternary / conditional ──────────────────────────────────────────────────

HAND_CRAFTED["where"] = '''"""Reference CUDA kernel for aten.where — elementwise conditional select.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_where.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_where(
    const float *cond, const float *x, const float *y, float *out, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (cond[i] != 0.0f) ? x[i] : y[i];
}
"""

def init_once():
    cond = (torch.randn(1024, device='cuda') > 0).float()
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC, "inputs": [cond, x, y],
        "expected": [torch.ops.aten.where.self(cond.bool(), x, y)],
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
'''

HAND_CRAFTED["clamp"] = '''"""Reference CUDA kernel for aten.clamp — clamp to [min, max] range.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_clamp.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_clamp(
    const float *in0, float *out0, float lo, float hi, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = fminf(fmaxf(in0[i], lo), hi);
}
"""

def init_once():
    x = torch.randn(1024, device='cuda') * 5
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.clamp.default(x, -1.0, 1.0)],
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.float32(-1.0), np.float32(1.0), np.uint32(inputs[0].numel()),
    ])]
'''

HAND_CRAFTED["lerp"] = '''"""Reference CUDA kernel for aten.lerp — linear interpolation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_lerp.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_lerp(
    const float *a, const float *b, const float *w, float *out, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + w[i] * (b[i] - a[i]);
}
"""

def init_once():
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    w = torch.rand(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC, "inputs": [a, b, w],
        "expected": [torch.ops.aten.lerp.Tensor(a, b, w)], "atol": 1e-5,
    }

def run(inputs, kernel):
    return [kernel(*inputs)]
'''

HAND_CRAFTED["addcmul"] = '''"""Reference CUDA kernel for aten.addcmul — input + value * t1 * t2.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_addcmul.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_addcmul(
    const float *inp, const float *t1, const float *t2,
    float *out, float value, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = inp[i] + value * t1[i] * t2[i];
}
"""

def init_once():
    inp = torch.randn(1024, device='cuda')
    t1 = torch.randn(1024, device='cuda')
    t2 = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC, "inputs": [inp, t1, t2],
        "expected": [torch.ops.aten.addcmul.default(inp, t1, t2, value=0.5)], "atol": 1e-5,
    }

def run(inputs, kernel):
    n = inputs[0].numel()
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2),
        kernel.out_ptr(0), np.float32(0.5), np.uint32(n),
    ])]
'''

HAND_CRAFTED["addcdiv"] = '''"""Reference CUDA kernel for aten.addcdiv — input + value * t1 / t2.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_addcdiv.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_addcdiv(
    const float *inp, const float *t1, const float *t2,
    float *out, float value, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = inp[i] + value * t1[i] / t2[i];
}
"""

def init_once():
    inp = torch.randn(1024, device='cuda')
    t1 = torch.randn(1024, device='cuda')
    t2 = torch.randn(1024, device='cuda').abs() + 0.1
    return {
        "kernel_source": KERNEL_SRC, "inputs": [inp, t1, t2],
        "expected": [torch.ops.aten.addcdiv.default(inp, t1, t2, value=0.5)], "atol": 1e-4,
    }

def run(inputs, kernel):
    n = inputs[0].numel()
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2),
        kernel.out_ptr(0), np.float32(0.5), np.uint32(n),
    ])]
'''

HAND_CRAFTED["masked_fill"] = '''"""Reference CUDA kernel for aten.masked_fill — fill where mask is True.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_masked_fill.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_masked_fill(
    const float *input, const float *mask, float *out, float value, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (mask[i] != 0.0f) ? value : input[i];
}
"""

def init_once():
    x = torch.randn(1024, device='cuda')
    mask = (torch.randn(1024, device='cuda') > 0).float()
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, mask],
        "expected": [torch.ops.aten.masked_fill.Scalar(x, mask.bool(), -1e9)],
    }

def run(inputs, kernel):
    n = inputs[0].numel()
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.float32(-1e9), np.uint32(n),
    ])]
'''

# ─── Matmul family — all need custom params for M, K, N ─────────────────────

HAND_CRAFTED["mm"] = '''"""Reference CUDA kernel for aten.mm — naive nested loop matmul.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_mm.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_mm(
    const float *A, const float *B, float *C,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
"""

M, K, N = 64, 32, 48

def init_once():
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [A, B],
        "expected": [torch.ops.aten.mm.default(A, B).flatten()],
        "outputs": ["float32;n=%d" % (M * N)],
        "grid": ((N + 15) // 16, (M + 15) // 16),
        "block": (16, 16), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(M), np.uint32(K), np.uint32(N),
    ])]
'''

HAND_CRAFTED["bmm"] = '''"""Reference CUDA kernel for aten.bmm — batched matmul.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_bmm.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_bmm(
    const float *A, const float *B, float *C,
    unsigned int batch, unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int b = blockIdx.z;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch && row < M && col < N) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; k++)
            sum += A[b*M*K + row*K + k] * B[b*K*N + k*N + col];
        C[b*M*N + row*N + col] = sum;
    }
}
"""

B, M, K, N = 4, 16, 32, 24

def init_once():
    A = torch.randn(B, M, K, device="cuda")
    Bt = torch.randn(B, K, N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [A, Bt],
        "expected": [torch.ops.aten.bmm.default(A, Bt).flatten()],
        "outputs": ["float32;n=%d" % (B * M * N)],
        "grid": ((N + 15) // 16, (M + 15) // 16, B),
        "block": (16, 16), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(B), np.uint32(M), np.uint32(K), np.uint32(N),
    ])]
'''

HAND_CRAFTED["addmm"] = '''"""Reference CUDA kernel for aten.addmm — bias + A @ B.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_addmm.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_addmm(
    const float *bias, const float *A, const float *B, float *C,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = bias[col];
        for (unsigned int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
"""

M, K, N = 64, 32, 48

def init_once():
    bias = torch.randn(N, device="cuda")
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [bias, A, B],
        "expected": [torch.ops.aten.addmm.default(bias, A, B).flatten()],
        "outputs": ["float32;n=%d" % (M * N)],
        "grid": ((N + 15) // 16, (M + 15) // 16),
        "block": (16, 16), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(M), np.uint32(K), np.uint32(N),
    ])]
'''

HAND_CRAFTED["dot"] = '''"""Reference CUDA kernel for aten.dot — inner product of two vectors.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_dot.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_dot(
    const float *a, const float *b, float *out, unsigned int n
) {
    __shared__ float sdata[256];
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
"""

def init_once():
    a = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [a, b],
        "expected": [torch.ops.aten.dot.default(a, b).reshape(1)],
        "outputs": ["float32;n=1"], "grid": ((1 + 255) // 256,),
        "grid": (1,),
        "block": (256,), "atol": 1e-3,
    }

def run(inputs, kernel):
    a, b = inputs
    return [kernel(a, b, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(a.numel()),
    ])]
'''

HAND_CRAFTED["mv"] = '''"""Reference CUDA kernel for aten.mv — matrix-vector multiply.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_mv.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

M, K = 64, 32

def init_once():
    A = torch.randn(M, K, device="cuda")
    x = torch.randn(K, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [A, x],
        "expected": [torch.ops.aten.mv.default(A, x)],
        "outputs": ["float32;n=%d" % M], "atol": 1e-3,
        "grid": ((M + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(M), np.uint32(K),
    ])]
'''

HAND_CRAFTED["outer"] = '''"""Reference CUDA kernel for aten.outer — outer product of two vectors.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_outer.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_outer(
    const float *a, const float *b, float *out,
    unsigned int M, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        out[row * N + col] = a[row] * b[col];
}
"""

M, N = 64, 48

def init_once():
    a = torch.randn(M, device="cuda")
    b = torch.randn(N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [a, b],
        "expected": [torch.ops.aten.outer.default(a, b).flatten()],
        "outputs": ["float32;n=%d" % (M * N)],
        "grid": ((N + 15) // 16, (M + 15) // 16),
        "block": (16, 16), "atol": 1e-5,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(M), np.uint32(N),
    ])]
'''

HAND_CRAFTED["baddbmm"] = '''"""Reference CUDA kernel for aten.baddbmm — batch add + batch matmul.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_baddbmm.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

BATCH, M, K, N = 4, 16, 32, 24

def init_once():
    s = torch.randn(BATCH, M, N, device="cuda")
    A = torch.randn(BATCH, M, K, device="cuda")
    B = torch.randn(BATCH, K, N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [s, A, B],
        "expected": [torch.ops.aten.baddbmm.default(s, A, B).flatten()],
        "outputs": ["float32;n=%d" % (BATCH * M * N)],
        "grid": ((N + 15) // 16, (M + 15) // 16, BATCH),
        "block": (16, 16), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(BATCH), np.uint32(M), np.uint32(K), np.uint32(N),
        np.float32(1.0), np.float32(1.0),
    ])]
'''

HAND_CRAFTED["linear"] = '''"""Reference CUDA kernel for aten.linear — y = x @ weight.T + bias.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_linear.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_linear(
    const float *x, const float *w, const float *bias, float *out,
    unsigned int M, unsigned int K, unsigned int N
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = bias[col];
        for (unsigned int k = 0; k < K; k++)
            sum += x[row * K + k] * w[col * K + k];
        out[row * N + col] = sum;
    }
}
"""

M, K, N = 32, 64, 48

def init_once():
    x = torch.randn(M, K, device="cuda")
    w = torch.randn(N, K, device="cuda")
    b = torch.randn(N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, w, b],
        "expected": [torch.ops.aten.linear.default(x, w, b).flatten()],
        "outputs": ["float32;n=%d" % (M * N)],
        "grid": ((N + 15) // 16, (M + 15) // 16),
        "block": (16, 16), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(M), np.uint32(K), np.uint32(N),
    ])]
'''

# ─── Normalization ──────────────────────────────────────────────────────────

HAND_CRAFTED["_softmax"] = '''"""Reference CUDA kernel for aten._softmax — 3-pass: max, exp+sum, normalize.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten__softmax.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_softmax(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    __shared__ float sdata[256];
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
"""

ROWS, COLS = 8, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten._softmax.default(x, -1, False).flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": (ROWS,),
        "block": (256,), "atol": 1e-5,
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
'''

HAND_CRAFTED["_log_softmax"] = '''"""Reference CUDA kernel for aten._log_softmax — log(softmax(x)).
Run: kbox iterate torch_graph/cuda_ref_kernels/aten__log_softmax.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_log_softmax(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    __shared__ float sdata[256];
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
"""

ROWS, COLS = 8, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten._log_softmax.default(x, -1, False).flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": (ROWS,),
        "block": (256,), "atol": 1e-5,
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
'''

HAND_CRAFTED["native_layer_norm"] = '''"""Reference CUDA kernel for aten.native_layer_norm.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_native_layer_norm.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_layer_norm(
    const float *input, const float *weight, const float *bias,
    float *output, unsigned int rows, unsigned int cols, float eps
) {
    __shared__ float s_sum[256];
    __shared__ float s_sq[256];
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
    __syncthreads();
    for (unsigned int j = tid; j < cols; j += blockDim.x)
        ro[j] = (ri[j] - mean) * rstd * weight[j] + bias[j];
}
"""

ROWS, COLS = 8, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    w = torch.randn(COLS, device="cuda")
    b = torch.randn(COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, w, b],
        "expected": [torch.ops.aten.native_layer_norm.default(x, [COLS], w, b, 1e-5)[0].flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": (ROWS,),
        "block": (256,), "atol": 1e-4,
    }

def run(inputs, kernel):
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS), np.float32(1e-5),
    ])]
'''

HAND_CRAFTED["native_batch_norm"] = '''"""Reference CUDA kernel for aten.native_batch_norm — per-channel normalization.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_native_batch_norm.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_batch_norm(
    const float *input, const float *weight, const float *bias,
    const float *running_mean, const float *running_var,
    float *output, unsigned int N, unsigned int C, unsigned int HW, float eps
) {
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
"""

NN, CC, HH, WW = 2, 8, 4, 4

def init_once():
    x = torch.randn(NN, CC, HH, WW, device="cuda")
    w = torch.randn(CC, device="cuda")
    b = torch.randn(CC, device="cuda")
    rm = torch.randn(CC, device="cuda")
    rv = torch.rand(CC, device="cuda") + 0.1
    total = NN * CC * HH * WW
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x.contiguous(), w, b, rm, rv],
        "expected": [torch.ops.aten.native_batch_norm.default(x, w, b, rm, rv, False, 0.1, 1e-5)[0].flatten()],
        "outputs": ["float32;n=%d" % total], "atol": 1e-4,
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    total = inputs[0].numel()
    HW = HH * WW
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2),
        kernel.in_ptr(3), kernel.in_ptr(4), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(HW), np.float32(1e-5),
    ])]
'''

HAND_CRAFTED["native_group_norm"] = '''"""Reference CUDA kernel for aten.native_group_norm — group normalization.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_native_group_norm.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_group_norm(
    const float *input, const float *weight, const float *bias,
    float *output, unsigned int N, unsigned int C, unsigned int HW,
    unsigned int G, float eps
) {
    unsigned int ng = blockIdx.x;
    unsigned int n = ng / G, g = ng % G;
    unsigned int tid = threadIdx.x;
    unsigned int CpG = C / G;
    unsigned int group_size = CpG * HW;

    __shared__ float s_sum[256];
    __shared__ float s_sq[256];

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

    for (unsigned int i = tid; i < group_size; i += blockDim.x) {
        unsigned int c_local = i / HW;
        unsigned int hw = i % HW;
        unsigned int c = g * CpG + c_local;
        unsigned int idx = n * C * HW + c * HW + hw;
        output[idx] = (input[idx] - mean) * rstd * weight[c] + bias[c];
    }
}
"""

NN, CC, HH, WW, GG = 2, 8, 4, 4, 4

def init_once():
    HW = HH * WW
    x = torch.randn(NN, CC, HH, WW, device="cuda")
    w = torch.randn(CC, device="cuda")
    b = torch.randn(CC, device="cuda")
    total = NN * CC * HW
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x.contiguous(), w, b],
        "expected": [torch.ops.aten.native_group_norm.default(x, w, b, NN, CC, HW, GG, 1e-5)[0].flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": (NN * GG,),
        "block": (256,), "atol": 1e-4,
    }

def run(inputs, kernel):
    HW = HH * WW
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(HW),
        np.uint32(GG), np.float32(1e-5),
    ])]
'''

# ─── Layout ops with CUDA kernels ───────────────────────────────────────────

HAND_CRAFTED["transpose"] = '''"""Reference CUDA kernel for aten.transpose — 2D transpose, contiguous output.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_transpose.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_transpose_2d(
    const float *in0, float *out0, unsigned int rows, unsigned int cols
) {
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) out0[c * rows + r] = in0[r * cols + c];
}
"""

ROWS, COLS = 32, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.transpose.int(x, 0, 1).contiguous().flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": ((COLS + 15) // 16, (ROWS + 15) // 16),
        "block": (16, 16),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
'''

HAND_CRAFTED["t"] = '''"""Reference CUDA kernel for aten.t — 2D matrix transpose.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_t.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_t_kernel(
    const float *in0, float *out0, unsigned int rows, unsigned int cols
) {
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) out0[c * rows + r] = in0[r * cols + c];
}
"""

ROWS, COLS = 32, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.t.default(x).contiguous().flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": ((COLS + 15) // 16, (ROWS + 15) // 16),
        "block": (16, 16),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
'''

HAND_CRAFTED["permute"] = '''"""Reference CUDA kernel for aten.permute — 3D dimension permutation.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_permute.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_permute_3d(
    const float *input, float *output,
    unsigned int S0, unsigned int S1, unsigned int S2,
    unsigned int perm0, unsigned int perm1, unsigned int perm2
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = S0 * S1 * S2;
    if (idx >= total) return;
    unsigned int i0 = idx / (S1 * S2);
    unsigned int i1 = (idx / S2) % S1;
    unsigned int i2 = idx % S2;
    unsigned int in_idx[3];
    in_idx[0] = i0; in_idx[1] = i1; in_idx[2] = i2;
    unsigned int sizes[3];
    sizes[0] = S0; sizes[1] = S1; sizes[2] = S2;
    unsigned int perm[3];
    perm[0] = perm0; perm[1] = perm1; perm[2] = perm2;
    unsigned int out_sizes[3];
    out_sizes[0] = sizes[perm[0]]; out_sizes[1] = sizes[perm[1]]; out_sizes[2] = sizes[perm[2]];
    unsigned int o0 = in_idx[perm0], o1 = in_idx[perm1], o2 = in_idx[perm2];
    unsigned int out_idx = o0 * out_sizes[1] * out_sizes[2] + o1 * out_sizes[2] + o2;
    output[out_idx] = input[idx];
}
"""

S0, S1, S2 = 4, 8, 16

def init_once():
    x = torch.randn(S0, S1, S2, device="cuda")
    total = S0 * S1 * S2
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.permute.default(x, [2, 0, 1]).contiguous().flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    total = S0 * S1 * S2
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(S0), np.uint32(S1), np.uint32(S2),
        np.uint32(2), np.uint32(0), np.uint32(1),
    ])]
'''

# ─── Copy-based layout ops (in0, out0, n) ───────────────────────────────────

_COPY_OPS = {
    "clone": ("aten.clone — copy tensor to new contiguous memory.",
              "x = torch.randn(1024, device='cuda')",
              "torch.ops.aten.clone.default(x)"),
    "contiguous": ("aten.contiguous — ensure contiguous memory layout.",
                   "x = torch.randn(32, 64, device='cuda').t().contiguous()",
                   "x.clone()"),
    "view": ("aten.view — reshape (contiguous data copy).",
             "x = torch.randn(32, 64, device='cuda')",
             "torch.ops.aten.view.default(x, [64, 32]).contiguous()"),
    "reshape": ("aten.reshape — reshape with contiguous output.",
                "x = torch.randn(32, 64, device='cuda')",
                "torch.ops.aten.reshape.default(x, [64, 32]).contiguous()"),
    "unsqueeze": ("aten.unsqueeze — add dimension (contiguous copy).",
                  "x = torch.randn(32, 64, device='cuda')",
                  "torch.ops.aten.unsqueeze.default(x, 0).contiguous()"),
    "squeeze": ("aten.squeeze — remove size-1 dimensions.",
                "x = torch.randn(32, 1, 64, device='cuda')",
                "torch.ops.aten.squeeze.dim(x, 1).contiguous()"),
    "flatten": ("aten.flatten — flatten dimensions.",
                "x = torch.randn(4, 8, 16, device='cuda')",
                "torch.ops.aten.flatten.using_ints(x, 1, 2).contiguous()"),
    "_to_copy": ("aten._to_copy — dtype/device conversion (copy).",
                 "x = torch.randn(1024, device='cuda')",
                 "torch.ops.aten._to_copy.default(x)"),
}

for _copy_name, (_copy_doc, _copy_setup, _copy_expected) in _COPY_OPS.items():
    HAND_CRAFTED[_copy_name] = f'''"""Reference CUDA kernel for {_copy_doc}
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_{_copy_name}.py --once
"""
import torch

KERNEL_SRC = r"""
extern "C" __global__ void aten_{_copy_name}_copy(const float *in0, float *out0, unsigned int n) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i];
}}
"""

def init_once():
    {_copy_setup}
    return {{
        "kernel_source": KERNEL_SRC,
        "inputs": [x.contiguous()],
        "expected": [{_copy_expected}.flatten()],
    }}

def run(inputs, kernel):
    return [kernel(*inputs)]
'''

# ─── Expand (broadcast copy) ────────────────────────────────────────────────

HAND_CRAFTED["expand"] = '''"""Reference CUDA kernel for aten.expand — broadcast copy to larger shape.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_expand.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

def init_once():
    x = torch.randn(1, 64, device="cuda")
    total = 32 * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.expand.default(x, [32, 64]).contiguous().flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(1), np.uint32(64),
        np.uint32(32), np.uint32(64),
    ])]
'''

# ─── Slice / select / narrow ────────────────────────────────────────────────

HAND_CRAFTED["slice"] = '''"""Reference CUDA kernel for aten.slice — copy sub-range along dim 0.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_slice.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

def init_once():
    x = torch.randn(32, 64, device="cuda")
    out_rows = 16
    total = out_rows * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.slice.Tensor(x, 0, 4, 20).contiguous().flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(32), np.uint32(64),
        np.uint32(4), np.uint32(1), np.uint32(16),
    ])]
'''

HAND_CRAFTED["select"] = '''"""Reference CUDA kernel for aten.select — select single index along dim 0.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_select.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_select_copy(
    const float *input, float *output, unsigned int cols, unsigned int index
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) output[c] = input[index * cols + c];
}
"""

def init_once():
    x = torch.randn(32, 64, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.select.int(x, 0, 5).contiguous().flatten()],
        "outputs": ["float32;n=64"], "grid": ((64 + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(64), np.uint32(5),
    ])]
'''

HAND_CRAFTED["narrow"] = '''"""Reference CUDA kernel for aten.narrow — narrow view along a dimension.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_narrow.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_narrow_copy(
    const float *input, float *output, unsigned int cols,
    unsigned int start, unsigned int length
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[idx] = input[(start + r) * cols + c];
}
"""

def init_once():
    x = torch.randn(32, 64, device="cuda")
    total = 10 * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.narrow.default(x, 0, 4, 10).contiguous().flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(64), np.uint32(4), np.uint32(10),
    ])]
'''

# ─── Rearrange ops ──────────────────────────────────────────────────────────

HAND_CRAFTED["flip"] = '''"""Reference CUDA kernel for aten.flip — reverse along last dimension.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_flip.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_flip_2d(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[r * cols + (cols - 1 - c)] = input[idx];
}
"""

ROWS, COLS = 16, 32

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.flip.default(x, [-1]).contiguous().flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": (((ROWS * COLS) + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
'''

HAND_CRAFTED["roll"] = '''"""Reference CUDA kernel for aten.roll — circular shift.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_roll.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_roll_1d(
    const float *input, float *output, unsigned int n, int shift
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int src = ((int)i - shift % (int)n + (int)n) % (int)n;
    output[i] = input[src];
}
"""

def init_once():
    x = torch.randn(256, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.roll.default(x, [10]).contiguous()],
    }

def run(inputs, kernel):
    n = inputs[0].numel()
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(n), np.int32(10),
    ])]
'''

HAND_CRAFTED["repeat"] = '''"""Reference CUDA kernel for aten.repeat — tile tensor along dimensions.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_repeat.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

R, C, RR, RC = 8, 16, 3, 2

def init_once():
    x = torch.randn(R, C, device="cuda")
    total = R * RR * C * RC
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.repeat.default(x, [RR, RC]).contiguous().flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(R), np.uint32(C), np.uint32(RR), np.uint32(RC),
    ])]
'''

HAND_CRAFTED["tril"] = '''"""Reference CUDA kernel for aten.tril — lower triangle of a matrix.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_tril.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_tril_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols, int diagonal
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[idx] = ((int)c <= (int)r + diagonal) ? input[idx] : 0.0f;
}
"""

N = 16

def init_once():
    x = torch.randn(N, N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.tril.default(x)],
    }

def run(inputs, kernel):
    n = inputs[0].size(0)
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(n), np.uint32(n), np.int32(0),
    ])]
'''

HAND_CRAFTED["triu"] = '''"""Reference CUDA kernel for aten.triu — upper triangle of a matrix.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_triu.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_triu_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols, int diagonal
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    unsigned int r = idx / cols, c = idx % cols;
    output[idx] = ((int)c >= (int)r + diagonal) ? input[idx] : 0.0f;
}
"""

N = 16

def init_once():
    x = torch.randn(N, N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.triu.default(x)],
    }

def run(inputs, kernel):
    n = inputs[0].size(0)
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(n), np.uint32(n), np.int32(0),
    ])]
'''

# ─── Tensor creation ────────────────────────────────────────────────────────

HAND_CRAFTED["arange"] = '''"""Reference CUDA kernel for aten.arange — fill with sequential values.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_arange.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_arange_kernel(
    float *output, float start, float step, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = start + i * step;
}
"""

def init_once():
    return {
        "kernel_source": KERNEL_SRC, "inputs": [],
        "expected": [torch.ops.aten.arange.start_step(0, 100, 1, dtype=torch.float32, device='cuda')],
        "outputs": ["float32;n=100"], "grid": ((100 + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(params=[
        kernel.out_ptr(0),
        np.float32(0.0), np.float32(1.0), np.uint32(100),
    ])]
'''

HAND_CRAFTED["zeros"] = '''"""Reference CUDA kernel for aten.zeros — create zero-filled tensor.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_zeros.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_fill_zero(float *output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 0.0f;
}
"""

def init_once():
    return {
        "kernel_source": KERNEL_SRC, "inputs": [],
        "expected": [torch.zeros(32, 64, device='cuda').flatten()],
        "outputs": ["float32;n=%d" % (32 * 64)],
        "grid": (((32 * 64) + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(params=[
        kernel.out_ptr(0), np.uint32(32 * 64),
    ])]
'''

HAND_CRAFTED["ones"] = '''"""Reference CUDA kernel for aten.ones — create one-filled tensor.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_ones.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_fill_one(float *output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = 1.0f;
}
"""

def init_once():
    return {
        "kernel_source": KERNEL_SRC, "inputs": [],
        "expected": [torch.ones(32, 64, device='cuda').flatten()],
        "outputs": ["float32;n=%d" % (32 * 64)],
        "grid": (((32 * 64) + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(params=[
        kernel.out_ptr(0), np.uint32(32 * 64),
    ])]
'''

HAND_CRAFTED["full"] = '''"""Reference CUDA kernel for aten.full — create tensor filled with a value.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_full.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_fill_val(float *output, float value, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = value;
}
"""

def init_once():
    return {
        "kernel_source": KERNEL_SRC, "inputs": [],
        "expected": [torch.full((32, 64), 3.14, device='cuda').flatten()],
        "outputs": ["float32;n=%d" % (32 * 64)],
        "grid": (((32 * 64) + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(params=[
        kernel.out_ptr(0), np.float32(3.14), np.uint32(32 * 64),
    ])]
'''

HAND_CRAFTED["eye"] = '''"""Reference CUDA kernel for aten.eye — identity matrix.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_eye.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_eye_kernel(float *output, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    unsigned int r = idx / n, c = idx % n;
    output[idx] = (r == c) ? 1.0f : 0.0f;
}
"""

N = 32

def init_once():
    return {
        "kernel_source": KERNEL_SRC, "inputs": [],
        "expected": [torch.eye(N, device='cuda').flatten()],
        "outputs": ["float32;n=%d" % (N * N)],
        "grid": (((N * N) + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(params=[
        kernel.out_ptr(0), np.uint32(N),
    ])]
'''

HAND_CRAFTED["linspace"] = '''"""Reference CUDA kernel for aten.linspace — evenly spaced values.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_linspace.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_linspace_kernel(
    float *output, float start, float step, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = start + i * step;
}
"""

def init_once():
    return {
        "kernel_source": KERNEL_SRC, "inputs": [],
        "expected": [torch.linspace(0, 1, 100, device='cuda')],
        "outputs": ["float32;n=100"], "grid": ((100 + 255) // 256,), "atol": 1e-5,
    }

def run(inputs, kernel):
    step = 1.0 / 99.0
    return [kernel(params=[
        kernel.out_ptr(0), np.float32(0.0), np.float32(step), np.uint32(100),
    ])]
'''

HAND_CRAFTED["fill"] = '''"""Reference CUDA kernel for aten.fill — fill tensor with scalar value.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_fill.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_fill_kernel(const float *in0, float *out0, float value, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = value;
}
"""

def init_once():
    x = torch.randn(1024, device='cuda')
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.fill.Scalar(x, 3.14)],
    }

def run(inputs, kernel):
    n = inputs[0].numel()
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.float32(3.14), np.uint32(n),
    ])]
'''

# ─── Indexing ops ────────────────────────────────────────────────────────────

HAND_CRAFTED["gather"] = '''"""Reference CUDA kernel for aten.gather — gather along dim by index.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_gather.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

def init_once():
    x = torch.randn(8, 32, device="cuda")
    idx = torch.randint(0, 32, (8, 16), device="cuda")
    total = 8 * 16
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, idx],
        "expected": [torch.ops.aten.gather.default(x, 1, idx).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    x, idx = inputs
    rows, in_cols = x.shape
    out_cols = idx.shape[1]
    return [kernel(x, idx, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(rows), np.uint32(in_cols), np.uint32(out_cols),
    ])]
'''

HAND_CRAFTED["scatter"] = '''"""Reference CUDA kernel for aten.scatter — scatter values into tensor by index.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_scatter.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_scatter_2d(
    const float *input, const long *index, const float *src, float *output,
    unsigned int rows, unsigned int in_cols, unsigned int src_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_in = rows * in_cols;
    if (idx < total_in) output[idx] = input[idx];
    __syncthreads();
    unsigned int total_src = rows * src_cols;
    if (idx < total_src) {
        unsigned int r = idx / src_cols, c = idx % src_cols;
        long dst_c = index[idx];
        output[r * in_cols + dst_c] = src[idx];
    }
}
"""

def init_once():
    x = torch.zeros(8, 32, device="cuda")
    idx = torch.randint(0, 32, (8, 16), device="cuda")
    src = torch.randn(8, 16, device="cuda")
    total = 8 * 32
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, idx, src],
        "expected": [torch.ops.aten.scatter.src(x, 1, idx, src).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    x, idx, src = inputs
    rows, in_cols = x.shape
    src_cols = idx.shape[1]
    return [kernel(x, idx, src, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(rows), np.uint32(in_cols), np.uint32(src_cols),
    ])]
'''

HAND_CRAFTED["index_select"] = '''"""Reference CUDA kernel for aten.index_select — select rows by index.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_index_select.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

def init_once():
    x = torch.randn(32, 64, device="cuda")
    idx = torch.tensor([0, 5, 10, 15, 31], device="cuda")
    total = 5 * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, idx],
        "expected": [torch.ops.aten.index_select.default(x, 0, idx).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    x, idx = inputs
    n_idx = idx.numel()
    cols = x.shape[1]
    return [kernel(x, idx, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(n_idx), np.uint32(cols),
    ])]
'''

HAND_CRAFTED["index_add"] = '''"""Reference CUDA kernel for aten.index_add — add source into self at indices.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_index_add.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_index_add_kernel(
    const float *self, const long *index, const float *source, float *out,
    unsigned int rows, unsigned int cols, unsigned int n_idx
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * cols;
    if (idx < total) out[idx] = self[idx];
    __syncthreads();
    unsigned int total_src = n_idx * cols;
    if (idx < total_src) {
        unsigned int r = idx / cols, c = idx % cols;
        long dst_r = index[r];
        atomicAdd(&out[dst_r * cols + c], source[idx]);
    }
}
"""

def init_once():
    x = torch.zeros(32, 64, device="cuda")
    idx = torch.tensor([0, 5, 10, 15, 20], device="cuda")
    src = torch.randn(5, 64, device="cuda")
    total = 32 * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, idx, src],
        "expected": [torch.ops.aten.index_add.default(x, 0, idx, src).flatten()],
        "outputs": ["float32;n=%d" % total], "atol": 1e-5,
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    x, idx, src = inputs
    rows, cols = x.shape
    n_idx = idx.numel()
    return [kernel(x, idx, src, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(rows), np.uint32(cols), np.uint32(n_idx),
    ])]
'''

# ─── Concatenation / split ──────────────────────────────────────────────────

HAND_CRAFTED["cat"] = '''"""Reference CUDA kernel for aten.cat — concatenate tensors along dim 0.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_cat.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

def init_once():
    a = torch.randn(8, 32, device="cuda")
    b = torch.randn(16, 32, device="cuda")
    total = (8 + 16) * 32
    return {
        "kernel_source": KERNEL_SRC, "inputs": [a, b],
        "expected": [torch.ops.aten.cat.default([a, b], 0).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    a, b = inputs
    return [kernel(a, b, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(a.shape[0]), np.uint32(b.shape[0]), np.uint32(a.shape[1]),
    ])]
'''

HAND_CRAFTED["stack"] = '''"""Reference CUDA kernel for aten.stack — stack tensors along new dim 0.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_stack.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_stack_2(
    const float *a, const float *b, float *out, unsigned int L
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < L) {
        out[i] = a[i];
        out[L + i] = b[i];
    }
}
"""

def init_once():
    a = torch.randn(64, device="cuda")
    b = torch.randn(64, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [a, b],
        "expected": [torch.ops.aten.stack.default([a, b], 0).flatten()],
        "outputs": ["float32;n=128"], "grid": ((128 + 255) // 256,),
    }

def run(inputs, kernel):
    a, b = inputs
    L = a.numel()
    return [kernel(a, b, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(L),
    ])]
'''

HAND_CRAFTED["split"] = '''"""Reference CUDA kernel for aten.split — extract first chunk along dim 0.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_split.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_split_copy(
    const float *input, float *out, unsigned int offset, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = input[offset + i];
}
"""

def init_once():
    x = torch.randn(32, 64, device="cuda")
    chunk = list(torch.ops.aten.split.Tensor(x, 8, 0))[0].contiguous()
    total = 8 * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [chunk.flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(0), np.uint32(8 * 64),
    ])]
'''

# ─── More reductions ────────────────────────────────────────────────────────

HAND_CRAFTED["var"] = '''"""Reference CUDA kernel for aten.var — variance reduction.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_var.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_var_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols, int correction
) {
    __shared__ float s_sum[256];
    __shared__ float s_sq[256];
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
"""

ROWS, COLS = 32, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.var.correction(x, [-1])],
        "outputs": ["float32;n=%d" % ROWS],
        "grid": (ROWS,),
        "block": (256,), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS), np.int32(1),
    ])]
'''

HAND_CRAFTED["argmax"] = '''"""Reference CUDA kernel for aten.argmax — index of maximum value.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_argmax.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_argmax_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float best = ri[0];
    float best_idx = 0.0f;
    for (unsigned int j = 1; j < cols; j++) {
        if (ri[j] > best) { best = ri[j]; best_idx = (float)j; }
    }
    output[row] = best_idx;
}
"""

ROWS, COLS = 32, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.argmax.default(x, -1).float()],
        "outputs": ["float32;n=%d" % ROWS],
        "grid": (ROWS,),
        "block": (1,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
'''

HAND_CRAFTED["argmin"] = '''"""Reference CUDA kernel for aten.argmin — index of minimum value.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_argmin.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_argmin_kernel(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float best = ri[0];
    float best_idx = 0.0f;
    for (unsigned int j = 1; j < cols; j++) {
        if (ri[j] < best) { best = ri[j]; best_idx = (float)j; }
    }
    output[row] = best_idx;
}
"""

ROWS, COLS = 32, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.argmin.default(x, -1).float()],
        "outputs": ["float32;n=%d" % ROWS],
        "grid": (ROWS,),
        "block": (1,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
'''

# ─── Scan / cumulative ──────────────────────────────────────────────────────

HAND_CRAFTED["cumsum"] = '''"""Reference CUDA kernel for aten.cumsum — cumulative sum along last dim.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_cumsum.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

ROWS, COLS = 8, 64

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.cumsum.default(x, -1).flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": (ROWS,),
        "block": (1,), "atol": 1e-4,
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
'''

HAND_CRAFTED["cumprod"] = '''"""Reference CUDA kernel for aten.cumprod — cumulative product along last dim.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_cumprod.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

ROWS, COLS = 8, 16

def init_once():
    x = torch.rand(ROWS, COLS, device="cuda") + 0.5
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.cumprod.default(x, -1).flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": (ROWS,),
        "block": (1,), "atol": 1e-3,
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
'''

# ─── Sort / search ──────────────────────────────────────────────────────────

HAND_CRAFTED["sort"] = '''"""Reference CUDA kernel for aten.sort — bubble sort reference.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_sort.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_sort_kernel(
    const float *input, float *values,
    unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *rv = values + row * cols;
    for (unsigned int j = 0; j < cols; j++) rv[j] = ri[j];
    for (unsigned int i = 0; i < cols; i++) {
        for (unsigned int j = i + 1; j < cols; j++) {
            if (rv[j] < rv[i]) {
                float tmp = rv[i]; rv[i] = rv[j]; rv[j] = tmp;
            }
        }
    }
}
"""

ROWS, COLS = 8, 32

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.sort.default(x, -1)[0].flatten()],
        "outputs": ["float32;n=%d" % (ROWS * COLS)],
        "grid": (ROWS,),
        "block": (1,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS),
    ])]
'''

HAND_CRAFTED["topk"] = '''"""Reference CUDA kernel for aten.topk — find k largest values.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_topk.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_topk_kernel(
    const float *input, float *values,
    unsigned int rows, unsigned int cols, unsigned int k
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float *rv = values + row * k;
    // Track selected indices locally
    int selected[32];  // max k=32
    for (unsigned int i = 0; i < k; i++) {
        float best = -1e38f;
        int best_j = 0;
        for (unsigned int j = 0; j < cols; j++) {
            float v = ri[j];
            int already = 0;
            for (unsigned int p = 0; p < i; p++) {
                if (selected[p] == (int)j) { already = 1; break; }
            }
            if (!already && v > best) { best = v; best_j = j; }
        }
        rv[i] = best;
        selected[i] = best_j;
    }
}
"""

ROWS, COLS, K = 8, 32, 5

def init_once():
    x = torch.randn(ROWS, COLS, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.topk.default(x, K, -1)[0].flatten()],
        "outputs": ["float32;n=%d" % (ROWS * K)],
        "grid": (ROWS,),
        "block": (1,),
    }

def run(inputs, kernel):
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(ROWS), np.uint32(COLS), np.uint32(K),
    ])]
'''

# ─── Embedding ──────────────────────────────────────────────────────────────

HAND_CRAFTED["embedding"] = '''"""Reference CUDA kernel for aten.embedding — table lookup.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_embedding.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

def init_once():
    weight = torch.randn(100, 64, device="cuda")
    indices = torch.randint(0, 100, (32,), device="cuda")
    total = 32 * 64
    return {
        "kernel_source": KERNEL_SRC, "inputs": [weight, indices],
        "expected": [torch.ops.aten.embedding.default(weight, indices).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    weight, indices = inputs
    n_idx = indices.numel()
    embed_dim = weight.shape[1]
    return [kernel(weight, indices, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(n_idx), np.uint32(embed_dim),
    ])]
'''

# ─── Convolution ────────────────────────────────────────────────────────────

HAND_CRAFTED["convolution"] = '''"""Reference CUDA kernel for aten.convolution — naive conv2d.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_convolution.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
    float sum = bias[oc];
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
"""

NN, C_IN, H, W, C_OUT, KH, KW = 1, 3, 8, 8, 16, 3, 3
PAD, STRIDE = 1, 1
OUT_H = (H + 2 * PAD - KH) // STRIDE + 1
OUT_W = (W + 2 * PAD - KW) // STRIDE + 1

def init_once():
    x = torch.randn(NN, C_IN, H, W, device="cuda")
    w = torch.randn(C_OUT, C_IN, KH, KW, device="cuda")
    b = torch.randn(C_OUT, device="cuda")
    total = NN * C_OUT * OUT_H * OUT_W
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x.contiguous(), w.contiguous(), b],
        "expected": [torch.ops.aten.convolution.default(x, w, b, [STRIDE,STRIDE], [PAD,PAD], [1,1], False, [0,0], 1).flatten()],
        "outputs": ["float32;n=%d" % total], "atol": 1e-3,
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    total = NN * C_OUT * OUT_H * OUT_W
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(C_IN), np.uint32(H), np.uint32(W),
        np.uint32(C_OUT), np.uint32(KH), np.uint32(KW),
        np.uint32(PAD), np.uint32(PAD), np.uint32(STRIDE), np.uint32(STRIDE),
        np.uint32(OUT_H), np.uint32(OUT_W),
    ])]
'''

# ─── Pooling ────────────────────────────────────────────────────────────────

HAND_CRAFTED["max_pool2d"] = '''"""Reference CUDA kernel for aten.max_pool2d — max pooling (values only).
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_max_pool2d.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_max_pool2d_kernel(
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
    float best = -1e38f;
    for (unsigned int kh = 0; kh < kH; kh++) {
        for (unsigned int kw = 0; kw < kW; kw++) {
            int ih = (int)(oh * strideH + kh) - (int)padH;
            int iw = (int)(ow * strideW + kw) - (int)padW;
            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                float v = input[n*C*H*W + c*H*W + ih*W + iw];
                if (v > best) best = v;
            }
        }
    }
    output[idx] = best;
}
"""

NN, CC, H, W = 1, 4, 8, 8
KH, KW, SH, SW, PH, PW = 2, 2, 2, 2, 0, 0
OUT_H = (H + 2*PH - KH) // SH + 1
OUT_W = (W + 2*PW - KW) // SW + 1

def init_once():
    x = torch.randn(NN, CC, H, W, device="cuda")
    total = NN * CC * OUT_H * OUT_W
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.max_pool2d_with_indices.default(x, [KH,KW], [SH,SW])[0].flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    total = NN * CC * OUT_H * OUT_W
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(H), np.uint32(W),
        np.uint32(KH), np.uint32(KW), np.uint32(SH), np.uint32(SW),
        np.uint32(PH), np.uint32(PW), np.uint32(OUT_H), np.uint32(OUT_W),
    ])]
'''

HAND_CRAFTED["avg_pool2d"] = '''"""Reference CUDA kernel for aten.avg_pool2d — average pooling.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_avg_pool2d.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

NN, CC, H, W = 1, 4, 8, 8
KH, KW, SH, SW, PH, PW = 2, 2, 2, 2, 0, 0
OUT_H = (H + 2*PH - KH) // SH + 1
OUT_W = (W + 2*PW - KW) // SW + 1

def init_once():
    x = torch.randn(NN, CC, H, W, device="cuda")
    total = NN * CC * OUT_H * OUT_W
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.avg_pool2d.default(x, [KH,KW], [SH,SW]).flatten()],
        "outputs": ["float32;n=%d" % total], "atol": 1e-5,
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    total = NN * CC * OUT_H * OUT_W
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(H), np.uint32(W),
        np.uint32(KH), np.uint32(KW), np.uint32(SH), np.uint32(SW),
        np.uint32(PH), np.uint32(PW), np.uint32(OUT_H), np.uint32(OUT_W),
    ])]
'''

HAND_CRAFTED["adaptive_avg_pool2d"] = '''"""Reference CUDA kernel for aten.adaptive_avg_pool2d.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_adaptive_avg_pool2d.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
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
"""

NN, CC, H, W = 1, 4, 8, 8
OUT_H, OUT_W = 1, 1

def init_once():
    x = torch.randn(NN, CC, H, W, device="cuda")
    total = NN * CC * OUT_H * OUT_W
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.adaptive_avg_pool2d.default(x, [OUT_H, OUT_W]).flatten()],
        "outputs": ["float32;n=%d" % total], "atol": 1e-4,
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    total = NN * CC * OUT_H * OUT_W
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(H), np.uint32(W),
        np.uint32(OUT_H), np.uint32(OUT_W),
    ])]
'''

# ─── Loss functions ─────────────────────────────────────────────────────────

HAND_CRAFTED["nll_loss_forward"] = '''"""Reference CUDA kernel for aten.nll_loss_forward — negative log likelihood.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_nll_loss_forward.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_nll_loss_kernel(
    const float *log_probs, const long *target, float *output,
    unsigned int N, unsigned int C
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < N; i++) {
        long t = target[i];
        sum -= log_probs[i * C + t];
    }
    output[0] = sum / (float)N;
}
"""

NN, CC = 16, 10

def init_once():
    log_probs = torch.randn(NN, CC, device="cuda").log_softmax(dim=-1)
    target = torch.randint(0, CC, (NN,), device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [log_probs, target],
        "expected": [torch.ops.aten.nll_loss_forward.default(log_probs, target, None, 1, -100)[0].reshape(1)],
        "outputs": ["float32;n=1"], "grid": ((1 + 255) // 256,),
        "grid": (1,),
        "block": (1,), "atol": 1e-4,
    }

def run(inputs, kernel):
    log_probs, target = inputs
    return [kernel(log_probs, target, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC),
    ])]
'''

HAND_CRAFTED["mse_loss"] = '''"""Reference CUDA kernel for aten.mse_loss — mean squared error.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_mse_loss.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_mse_kernel(
    const float *input, const float *target, float *output, unsigned int n
) {
    __shared__ float sdata[256];
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
"""

N = 256

def init_once():
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, y],
        "expected": [torch.ops.aten.mse_loss.default(x, y).reshape(1)],
        "outputs": ["float32;n=1"], "grid": ((1 + 255) // 256,),
        "grid": (1,),
        "block": (256,), "atol": 1e-4,
    }

def run(inputs, kernel):
    x, y = inputs
    return [kernel(x, y, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(x.numel()),
    ])]
'''

# ─── Attention ──────────────────────────────────────────────────────────────

HAND_CRAFTED["scaled_dot_product_attention"] = '''"""Reference CUDA kernel for scaled dot product attention.
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
        "outputs": ["float32;n=%d" % total], "atol": 1e-3,
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    total = BB * HH * SS * DD
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.in_ptr(2), kernel.out_ptr(0),
        np.uint32(BB), np.uint32(HH), np.uint32(SS), np.uint32(DD),
    ])]
'''

# ─── Dropout ────────────────────────────────────────────────────────────────

HAND_CRAFTED["native_dropout"] = '''"""Reference CUDA kernel for aten.native_dropout — deterministic dropout with mask.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_native_dropout.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_dropout_kernel(
    const float *input, const float *mask, float *output,
    float scale, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * mask[i] * scale;
}
"""

def init_once():
    x = torch.randn(1024, device="cuda")
    mask = (torch.rand(1024, device="cuda") > 0.5).float()
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x, mask],
        "expected": [x * mask * 2.0],
    }

def run(inputs, kernel):
    x, mask = inputs
    n = x.numel()
    return [kernel(x, mask, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.float32(2.0), np.uint32(n),
    ])]
'''

# ─── Padding ────────────────────────────────────────────────────────────────

HAND_CRAFTED["constant_pad_nd"] = '''"""Reference CUDA kernel for aten.constant_pad_nd — 2D constant padding.
Run: kbox iterate torch_graph/cuda_ref_kernels/aten_constant_pad_nd.py --once
"""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_constant_pad_2d(
    const float *input, float *output,
    unsigned int H, unsigned int W, unsigned int outH, unsigned int outW,
    unsigned int padTop, unsigned int padLeft, float value, unsigned int total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
"""

NN, H, W = 2, 8, 8
PAD = 1
OUT_H = H + 2 * PAD
OUT_W = W + 2 * PAD

def init_once():
    x = torch.randn(NN, H, W, device="cuda")
    total = NN * OUT_H * OUT_W
    return {
        "kernel_source": KERNEL_SRC, "inputs": [x],
        "expected": [torch.ops.aten.constant_pad_nd.default(x, [PAD, PAD, PAD, PAD], 0.0).flatten()],
        "outputs": ["float32;n=%d" % total],
        "grid": ((total + 255) // 256,),
    }

def run(inputs, kernel):
    total = NN * OUT_H * OUT_W
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(H), np.uint32(W), np.uint32(OUT_H), np.uint32(OUT_W),
        np.uint32(PAD), np.uint32(PAD), np.float32(0.0), np.uint32(total),
    ])]
'''

# ─── No-op identity ops ─────────────────────────────────────────────────────

HAND_CRAFTED["alias"] = '''"""Reference for aten.alias — identity op."""
import torch

def init_once():
    x = torch.randn(1024, device="cuda")
    return {"inputs": [x], "expected": [torch.ops.aten.alias.default(x)]}

def run(inputs):
    return [inputs[0]]
'''

HAND_CRAFTED["detach"] = '''"""Reference for aten.detach — detach from autograd."""
import torch

def init_once():
    x = torch.randn(1024, device="cuda", requires_grad=True)
    return {"inputs": [x], "expected": [x.detach()]}

def run(inputs):
    return [inputs[0].detach()]
'''


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4: GENERATE
# ═════════════════════════════════════════════════════════════════════════════

def generate():
    count = 0

    # Templated ops — auto-convert old format to new
    def _to_seeded_input(test_input):
        """Convert 'torch.randn(1024, device='cuda')' to seeded version."""
        s = test_input.replace("1024", "n")
        if "torch.randn(n" in s:
            s = s.replace("torch.randn(n, device='cuda')", 'torch.randn(n, device="cuda", generator=g)')
        if "torch.rand(n" in s:
            s = s.replace("torch.rand(n, device='cuda')", 'torch.rand(n, device="cuda", generator=g)')
        if "torch.tensor" in s:
            return test_input  # keep as-is for fixed tensors
        return s

    def _to_ref_fn(aten_ref, var="x"):
        """Convert 'torch.ops.aten.abs.default(x)' to use inputs[0]."""
        return aten_ref.replace(f"({var})", "(inputs[0])").replace(f"({var},", "(inputs[0],")

    def _to_ref_fn2(aten_ref):
        """Convert binary ref to use a, b from inputs."""
        return aten_ref.replace("(a, b)", "(a, b)").replace("(a,", "(a,")  # already uses a, b

    for op_name, func_name, cuda_expr, test_input, aten_ref, atol in UNARY_OPS:
        content = UNARY_TEMPLATE.format(
            op_name=op_name, func_name=func_name, cuda_expr=cuda_expr,
            test_input_seeded=_to_seeded_input(test_input),
            aten_ref_fn=_to_ref_fn(aten_ref),
            atol_val=atol if atol else "1e-5")
        (HERE / f"aten_{op_name}.py").write_text(content)
        count += 1

    def _to_seeded_setup(test_setup):
        """Convert binary test_setup to use generator g."""
        return test_setup.replace("device='cuda')", 'device="cuda", generator=g)')

    for op_name, func_name, cuda_expr, test_setup, aten_ref, atol in BINARY_OPS:
        content = BINARY_TEMPLATE.format(
            op_name=op_name, func_name=func_name, cuda_expr=cuda_expr,
            test_setup_seeded=_to_seeded_setup(test_setup),
            aten_ref_fn=aten_ref,  # already uses a, b
            atol_val=atol if atol else "1e-5")
        (HERE / f"aten_{op_name}.py").write_text(content)
        count += 1

    for op_name, func_name, cuda_expr, aten_ref in COMPARISON_OPS:
        content = COMPARISON_TEMPLATE.format(
            op_name=op_name, func_name=func_name, cuda_expr=cuda_expr,
            aten_ref_fn=aten_ref)  # already uses a, b
        (HERE / f"aten_{op_name}.py").write_text(content)
        count += 1

    for op_name, func_name, cuda_expr, test_setup, aten_ref, atol in BACKWARD_OPS:
        content = BACKWARD_TEMPLATE.format(
            op_name=op_name, func_name=func_name, cuda_expr=cuda_expr,
            test_setup_seeded=_to_seeded_setup(test_setup),
            aten_ref_fn=aten_ref,  # already uses grad, saved
            atol_val=atol if atol else "1e-5")
        (HERE / f"aten_{op_name}.py").write_text(content)
        count += 1

    for op_name, identity, accumulate, reduce, finalize, test_input, aten_ref, atol in REDUCTION_OPS:
        atol_str = f'\n        "atol": {atol},' if atol else ""
        content = REDUCTION_TEMPLATE.format(
            op_name=op_name, identity=identity, accumulate=accumulate,
            reduce=reduce, finalize=finalize, test_input=test_input,
            aten_ref=aten_ref, atol_str=atol_str)
        (HERE / f"aten_{op_name}.py").write_text(content)
        count += 1

    # Hand-crafted ops (all with CUDA kernels, except alias/detach)
    for name, content in HAND_CRAFTED.items():
        (HERE / f"aten_{name}.py").write_text(content)
        count += 1

    print(f"Generated {count} reference kernel files in {HERE}/")
    return count


if __name__ == "__main__":
    n = generate()
    print(f"Done. Test with: kbox iterate torch_graph/cuda_ref_kernels/aten_add.py --once")
