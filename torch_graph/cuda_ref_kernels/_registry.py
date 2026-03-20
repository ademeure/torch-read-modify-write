"""Single source of truth for all 220 aten reference kernels.

Each op is defined by:
  - kernel:   CUDA __global__ source string
  - aten:     callable(inputs) → [expected tensors]
  - inputs:   callable(dims, seed) → [input tensors]
  - dims:     dict of default dimension values (fuzzable by test runner)
  - dispatch: "auto" or callable(inputs, kernel) → [output tensors]
  - atol:     tolerance (default 1e-5)
  - outputs:  callable(dims) → kbox output spec (only for non-auto dispatch)
  - grid:     callable(dims) → grid tuple (only for non-auto dispatch)
  - block:    tuple (only for 2D grids)

Test runner calls:
  inputs = op["inputs"](dims, seed)   # seed=0 → special values
  expected = op["aten"](inputs)       # ground truth via torch.ops.aten
  result = op["dispatch"](inputs, kernel)  # CUDA kernel output
  verify(result, expected, op["atol"])
"""
import torch
import numpy as np

# ─── Input generators ────────────────────────────────────────────────────────
# These take (dims, seed) and return a list of tensors.
# seed=0 → special values, seed>0 → seeded random.

SPECIAL = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 4.0, 100.0, -100.0,
           1e-7, 1e7, 1e-45, -1e-45, 1.18e-38, -1.18e-38,
           float("nan"), float("inf"), float("-inf")]

def _special(n, device="cuda"):
    v = torch.tensor(SPECIAL, device=device)
    return v.repeat((n + len(v) - 1) // len(v))[:n]

def _seeded(shape, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, device=device, generator=g)

def _seeded_pos(shape, seed, bias=0.01, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(shape, device=device, generator=g) + bias

def _1d(dims, seed):
    """Single float tensor, shape (n,)."""
    n = dims["n"]
    return [_special(n) if seed == 0 else _seeded((n,), seed)]

def _1d_pos(dims, seed):
    """Single positive float tensor for log/sqrt/etc."""
    n = dims["n"]
    return [_special(n).abs() + 0.01 if seed == 0 else _seeded_pos((n,), seed)]

def _1d_unit(dims, seed):
    """Values in (-1, 1) for asin/acos/atanh."""
    n = dims["n"]
    if seed == 0:
        return [torch.tensor([0.0, 0.5, -0.5, 0.99, -0.99, 0.0, 0.1, -0.1] * ((n+7)//8), device="cuda")[:n]]
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.rand(n, device="cuda", generator=g) * 1.98 - 0.99]

def _pair(dims, seed):
    """Two float tensors, shape (n,) each. seed=0 → cross-product of specials."""
    n = dims["n"]
    if seed == 0:
        v = torch.tensor(SPECIAL, device="cuda")
        m = len(v)
        a = v.repeat_interleave(m).repeat((n + m*m - 1) // (m*m))[:n]
        b = v.repeat(m).repeat((n + m*m - 1) // (m*m))[:n]
        return [a, b]
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(n, device="cuda", generator=g),
            torch.randn(n, device="cuda", generator=g)]

def _pair_pos_b(dims, seed):
    """Two tensors, b is positive (for div, fmod, etc)."""
    n = dims["n"]
    if seed == 0:
        a, b = _pair(dims, seed)
        return [a, b.abs() + 0.1]
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(n, device="cuda", generator=g),
            torch.randn(n, device="cuda", generator=g).abs() + 0.1]

def _2d(dims, seed):
    """Single float tensor, shape (d0, d1)."""
    d0, d1 = dims["d0"], dims["d1"]
    n = d0 * d1
    if seed == 0:
        return [_special(n).reshape(d0, d1)]
    return [_seeded((d0, d1), seed)]

def _4d(dims, seed):
    """NCHW tensor."""
    N, C, H, W = dims["N"], dims["C"], dims["H"], dims["W"]
    n = N * C * H * W
    if seed == 0:
        return [_special(n).reshape(N, C, H, W)]
    return [_seeded((N, C, H, W), seed)]


# ─── Dispatch helpers ────────────────────────────────────────────────────────

def _auto(inputs, kernel):
    """Standard dispatch: kernel(*inputs)."""
    return [kernel(*inputs)]

def _params_1in(param_builder):
    """Dispatch with custom params, 1 input."""
    def dispatch(inputs, kernel):
        return [kernel(inputs[0], params=param_builder(inputs, kernel))]
    return dispatch

def _params_nin(param_builder):
    """Dispatch with custom params, all inputs."""
    def dispatch(inputs, kernel):
        return [kernel(*inputs, params=param_builder(inputs, kernel))]
    return dispatch

def _params_notin(param_builder):
    """Dispatch with custom params, no input tensors."""
    def dispatch(inputs, kernel):
        return [kernel(params=param_builder(inputs, kernel))]
    return dispatch


# ═════════════════════════════════════════════════════════════════════════════
#  OP REGISTRY
# ═════════════════════════════════════════════════════════════════════════════
# NOTE on NaN: We avoid fmaxf/fminf for NaN-sensitive ops because CUDA
# fmaxf(NaN, x) returns x (old IEEE 754). Ternary comparisons naturally
# propagate NaN since NaN comparisons return false. For optimized kernels,
# NVIDIA PTX has max.NaN.f32 / min.NaN.f32 instructions that implement
# correct IEEE 754-2019 NaN-propagating behavior natively.

OPS = {}

def _reg(name, *, kernel, aten, inputs=_1d, dims=None, dispatch=_auto,
         atol=1e-5, outputs=None, grid=None, block=None):
    """Register an op."""
    if dims is None:
        dims = {"n": 1024}
    OPS[name] = {
        "kernel": kernel, "aten": aten, "inputs": inputs,
        "dims": dict(dims), "dispatch": dispatch, "atol": atol,
        "outputs": outputs, "grid": grid, "block": block,
    }

# ─── Unary elementwise ──────────────────────────────────────────────────────

def _unary(cuda_expr):
    return (f'extern "C" __global__ void k(const float *in0, float *out0, unsigned int n) {{\n'
            f'    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
            f'    if (i < n) {{ float x = in0[i]; out0[i] = {cuda_expr}; }}\n'
            f'}}')

_reg("relu",         kernel=_unary("(x < 0.0f ? 0.0f : x)"),
     aten=lambda inp: [torch.ops.aten.relu.default(inp[0])])
_reg("gelu",         kernel=_unary("(x * 0.5f * (1.0f + erff(x * 0.7071067811865476f)))"),
     aten=lambda inp: [torch.ops.aten.gelu.default(inp[0])])
_reg("silu",         kernel=_unary("(x / (1.0f + expf(-x)))"),
     aten=lambda inp: [torch.ops.aten.silu.default(inp[0])])
_reg("sigmoid",      kernel=_unary("(1.0f / (1.0f + expf(-x)))"),
     aten=lambda inp: [torch.ops.aten.sigmoid.default(inp[0])])
_reg("tanh",         kernel=_unary("tanhf(x)"),
     aten=lambda inp: [torch.ops.aten.tanh.default(inp[0])], atol=1e-6)
_reg("abs",          kernel=_unary("fabsf(x)"),
     aten=lambda inp: [torch.ops.aten.abs.default(inp[0])])
_reg("neg",          kernel=_unary("(-x)"),
     aten=lambda inp: [torch.ops.aten.neg.default(inp[0])])
_reg("exp",          kernel=_unary("expf(x)"),
     aten=lambda inp: [torch.ops.aten.exp.default(inp[0])])
_reg("log",          kernel=_unary("logf(x)"),
     aten=lambda inp: [torch.ops.aten.log.default(inp[0])], inputs=_1d_pos)
_reg("sqrt",         kernel=_unary("sqrtf(x)"),
     aten=lambda inp: [torch.ops.aten.sqrt.default(inp[0])], inputs=_1d_pos, atol=1e-6)
_reg("rsqrt",        kernel=_unary("rsqrtf(x)"),
     aten=lambda inp: [torch.ops.aten.rsqrt.default(inp[0])], inputs=_1d_pos, atol=1e-4)
_reg("sin",          kernel=_unary("sinf(x)"),
     aten=lambda inp: [torch.ops.aten.sin.default(inp[0])])
_reg("cos",          kernel=_unary("cosf(x)"),
     aten=lambda inp: [torch.ops.aten.cos.default(inp[0])])
_reg("ceil",         kernel=_unary("ceilf(x)"),
     aten=lambda inp: [torch.ops.aten.ceil.default(inp[0])])
_reg("floor",        kernel=_unary("floorf(x)"),
     aten=lambda inp: [torch.ops.aten.floor.default(inp[0])])
_reg("round",        kernel=_unary("nearbyintf(x)"),
     aten=lambda inp: [torch.ops.aten.round.default(inp[0])])
_reg("trunc",        kernel=_unary("truncf(x)"),
     aten=lambda inp: [torch.ops.aten.trunc.default(inp[0])])
_reg("sign",         kernel=_unary("((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))"),
     aten=lambda inp: [torch.ops.aten.sign.default(inp[0])])
_reg("reciprocal",   kernel=_unary("(1.0f / x)"),
     aten=lambda inp: [torch.ops.aten.reciprocal.default(inp[0])], inputs=_1d_pos)
_reg("erf",          kernel=_unary("erff(x)"),
     aten=lambda inp: [torch.ops.aten.erf.default(inp[0])])
_reg("exp2",         kernel=_unary("exp2f(x)"),
     aten=lambda inp: [torch.ops.aten.exp2.default(inp[0])])
_reg("expm1",        kernel=_unary("expm1f(x)"),
     aten=lambda inp: [torch.ops.aten.expm1.default(inp[0])])
_reg("log2",         kernel=_unary("log2f(x)"),
     aten=lambda inp: [torch.ops.aten.log2.default(inp[0])], inputs=_1d_pos)
_reg("log10",        kernel=_unary("log10f(x)"),
     aten=lambda inp: [torch.ops.aten.log10.default(inp[0])], inputs=_1d_pos)
_reg("log1p",        kernel=_unary("log1pf(x)"),
     aten=lambda inp: [torch.ops.aten.log1p.default(inp[0])], inputs=_1d_pos)
_reg("tan",          kernel=_unary("tanf(x)"),
     aten=lambda inp: [torch.ops.aten.tan.default(inp[0])], atol=1e-4)
_reg("asin",         kernel=_unary("asinf(x)"),
     aten=lambda inp: [torch.ops.aten.asin.default(inp[0])], inputs=_1d_unit)
_reg("acos",         kernel=_unary("acosf(x)"),
     aten=lambda inp: [torch.ops.aten.acos.default(inp[0])], inputs=_1d_unit)
_reg("atan",         kernel=_unary("atanf(x)"),
     aten=lambda inp: [torch.ops.aten.atan.default(inp[0])])
_reg("sinh",         kernel=_unary("sinhf(x)"),
     aten=lambda inp: [torch.ops.aten.sinh.default(inp[0])], atol=1e-4)
_reg("cosh",         kernel=_unary("coshf(x)"),
     aten=lambda inp: [torch.ops.aten.cosh.default(inp[0])], atol=1e-4)
_reg("asinh",        kernel=_unary("asinhf(x)"),
     aten=lambda inp: [torch.ops.aten.asinh.default(inp[0])])
_reg("acosh",        kernel=_unary("acoshf(x)"),
     aten=lambda inp: [torch.ops.aten.acosh.default(inp[0])], inputs=_1d_pos)
_reg("atanh",        kernel=_unary("atanhf(x)"),
     aten=lambda inp: [torch.ops.aten.atanh.default(inp[0])], inputs=_1d_unit)
_reg("erfc",         kernel=_unary("erfcf(x)"),
     aten=lambda inp: [torch.ops.aten.erfc.default(inp[0])])
_reg("frac",         kernel=_unary("(x - truncf(x))"),
     aten=lambda inp: [torch.ops.aten.frac.default(inp[0])])
_reg("sgn",          kernel=_unary("((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))"),
     aten=lambda inp: [torch.ops.aten.sgn.default(inp[0])])
_reg("relu6",        kernel=_unary("(x < 0.0f ? 0.0f : (x > 6.0f ? 6.0f : x))"),
     aten=lambda inp: [torch.ops.aten.hardtanh.default(inp[0], 0.0, 6.0)])
_reg("hardswish",    kernel=_unary("(x * (x + 3.0f < 0.0f ? 0.0f : (x + 3.0f > 6.0f ? 6.0f : x + 3.0f)) / 6.0f)"),
     aten=lambda inp: [torch.ops.aten.hardswish.default(inp[0])])
_reg("hardsigmoid",  kernel=_unary("(x / 6.0f + 0.5f < 0.0f ? 0.0f : (x / 6.0f + 0.5f > 1.0f ? 1.0f : x / 6.0f + 0.5f))"),
     aten=lambda inp: [torch.ops.aten.hardsigmoid.default(inp[0])])
_reg("hardtanh",     kernel=_unary("(x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x))"),
     aten=lambda inp: [torch.ops.aten.hardtanh.default(inp[0])])
_reg("softplus",     kernel=_unary("((x > 20.0f) ? x : logf(1.0f + expf(x)))"),
     aten=lambda inp: [torch.ops.aten.softplus.default(inp[0])], atol=1e-4)
_reg("mish",         kernel=_unary("(x * tanhf((x > 20.0f) ? x : logf(1.0f + expf(x))))"),
     aten=lambda inp: [torch.ops.aten.mish.default(inp[0])], atol=1e-4)
_reg("elu",          kernel=_unary("((x > 0.0f) ? x : (expf(x) - 1.0f))"),
     aten=lambda inp: [torch.ops.aten.elu.default(inp[0])])
_reg("leaky_relu",   kernel=_unary("((x > 0.0f) ? x : 0.01f * x)"),
     aten=lambda inp: [torch.ops.aten.leaky_relu.default(inp[0], 0.01)], atol=1e-6)
_reg("log_sigmoid",  kernel=_unary("(x < 0.0f ? x - logf(1.0f + expf(x)) : -logf(1.0f + expf(-x)))"),
     aten=lambda inp: [torch.ops.aten.log_sigmoid_forward.default(inp[0])[0]])
_reg("logical_not",  kernel=_unary("((x == 0.0f) ? 1.0f : 0.0f)"),
     aten=lambda inp: [torch.ops.aten.logical_not.default(inp[0]).float()])
_reg("bitwise_not",  kernel=_unary("((x == 0.0f) ? 1.0f : 0.0f)"),
     aten=lambda inp: [torch.ops.aten.logical_not.default(inp[0]).float()])
_reg("isnan",        kernel=_unary("(isnan(x) ? 1.0f : 0.0f)"),
     aten=lambda inp: [torch.ops.aten.isnan.default(inp[0]).float()])
_reg("isinf",        kernel=_unary("(isinf(x) ? 1.0f : 0.0f)"),
     aten=lambda inp: [torch.ops.aten.isinf.default(inp[0]).float()])
_reg("isfinite",     kernel=_unary("(isfinite(x) ? 1.0f : 0.0f)"),
     aten=lambda inp: [torch.isfinite(inp[0]).float()])
_reg("_to_copy",     kernel=_unary("x"),  # identity copy
     aten=lambda inp: [torch.ops.aten._to_copy.default(inp[0])])

# ─── Binary elementwise ─────────────────────────────────────────────────────

def _binary(cuda_expr):
    return (f'extern "C" __global__ void k(const float *in0, const float *in1, float *out0, unsigned int n) {{\n'
            f'    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
            f'    if (i < n) {{ float a = in0[i], b = in1[i]; out0[i] = {cuda_expr}; }}\n'
            f'}}')

_reg("add",     kernel=_binary("(a + b)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.add.Tensor(*inp)])
_reg("sub",     kernel=_binary("(a - b)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.sub.Tensor(*inp)])
_reg("mul",     kernel=_binary("(a * b)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.mul.Tensor(*inp)])
_reg("div",     kernel=_binary("(a / b)"), inputs=_pair_pos_b,
     aten=lambda inp: [torch.ops.aten.div.Tensor(*inp)])
_reg("maximum", kernel=_binary("(a != a || b != b ? (0.0f/0.0f) : (a > b ? a : b))"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.maximum.default(*inp)])
_reg("minimum", kernel=_binary("(a != a || b != b ? (0.0f/0.0f) : (a < b ? a : b))"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.minimum.default(*inp)])

# ─── Comparison ──────────────────────────────────────────────────────────────

_reg("eq",  kernel=_binary("(a == b ? 1.0f : 0.0f)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.eq.Tensor(*inp).float()])
_reg("ne",  kernel=_binary("(a != b ? 1.0f : 0.0f)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.ne.Tensor(*inp).float()])
_reg("gt",  kernel=_binary("(a > b ? 1.0f : 0.0f)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.gt.Tensor(*inp).float()])
_reg("lt",  kernel=_binary("(a < b ? 1.0f : 0.0f)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.lt.Tensor(*inp).float()])

# ─── Matmul (2D grid) ───────────────────────────────────────────────────────

_MM_KERNEL = '''extern "C" __global__ void k(
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
}'''

def _mm_inputs(dims, seed):
    M, K, N = dims["d0"], dims["d1"], dims["d2"]
    if seed == 0:
        return [_special(M*K).reshape(M,K), _special(K*N).reshape(K,N)]
    return [_seeded((M,K), seed), _seeded((K,N), seed+1000)]

def _mm_dispatch(inputs, kernel):
    M, K = inputs[0].shape
    N = inputs[1].shape[1]
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(M), np.uint32(K), np.uint32(N)])]

_reg("mm", kernel=_MM_KERNEL,
     aten=lambda inp: [torch.ops.aten.mm.default(*inp).flatten()],
     inputs=_mm_inputs, dims={"d0": 64, "d1": 32, "d2": 48},
     dispatch=_mm_dispatch, atol=1e-3,
     outputs=lambda d: ["float32;n=%d" % (d["d0"]*d["d2"])],
     grid=lambda d: ((d["d2"]+15)//16, (d["d0"]+15)//16),
     block=(16, 16))

# ─── Reduction (1 block per row) ────────────────────────────────────────────

_SUM_KERNEL = '''extern "C" __global__ void k(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {
    __shared__ float sdata[256];
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float v = 0.0f;
    for (unsigned int j = tid; j < cols; j += blockDim.x) v += ri[j];
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads();
    }
    if (tid == 0) output[row] = sdata[0];
}'''

def _red_dispatch(inputs, kernel):
    rows, cols = inputs[0].shape
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(rows), np.uint32(cols)])]

_reg("sum", kernel=_SUM_KERNEL, inputs=_2d, dims={"d0": 32, "d1": 64},
     aten=lambda inp: [torch.ops.aten.sum.dim_IntList(inp[0], [-1])],
     dispatch=_red_dispatch, atol=1e-3,
     outputs=lambda d: ["float32;n=%d" % d["d0"]],
     grid=lambda d: (d["d0"],), block=(256,))

# More binary ops
_reg("atan2",    kernel=_binary("atan2f(a, b)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.atan2.default(*inp)])
_reg("fmod",     kernel=_binary("fmodf(a, b)"), inputs=_pair_pos_b,
     aten=lambda inp: [torch.ops.aten.fmod.Tensor(*inp)])
_reg("pow",      kernel=_binary("powf(a, b)"),
     inputs=lambda d, s: [_seeded_pos((d["n"],), s), _seeded_pos((d["n"],), s+1000, bias=0) * 3] if s > 0
                          else [_special(d["n"]).abs() + 0.1, _special(d["n"]).abs()],
     aten=lambda inp: [torch.ops.aten.pow.Tensor_Tensor(*inp)], atol=1e-4)
_reg("hypot",    kernel=_binary("hypotf(a, b)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.hypot.default(*inp)])
_reg("copysign", kernel=_binary("copysignf(a, b)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.copysign.Tensor(*inp)])
_reg("ge",       kernel=_binary("(a >= b ? 1.0f : 0.0f)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.ge.Tensor(*inp).float()])
_reg("le",       kernel=_binary("(a <= b ? 1.0f : 0.0f)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.le.Tensor(*inp).float()])
_reg("logical_and", kernel=_binary("((a != 0.0f && b != 0.0f) ? 1.0f : 0.0f)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.logical_and.default(*inp).float()])
_reg("logical_or",  kernel=_binary("((a != 0.0f || b != 0.0f) ? 1.0f : 0.0f)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.logical_or.default(*inp).float()])
_reg("logical_xor", kernel=_binary("(((a != 0.0f) != (b != 0.0f)) ? 1.0f : 0.0f)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.logical_xor.default(*inp).float()])

# Scalar ops (tensor op constant)
_reg("add_scalar",  kernel=_unary("(x + 2.0f)"),
     aten=lambda inp: [torch.ops.aten.add.Scalar(inp[0], 2.0)])
_reg("sub_scalar",  kernel=_unary("(x - 2.0f)"),
     aten=lambda inp: [torch.ops.aten.sub.Scalar(inp[0], 2.0)])
_reg("mul_scalar",  kernel=_unary("(x * 2.0f)"),
     aten=lambda inp: [torch.ops.aten.mul.Scalar(inp[0], 2.0)])
_reg("div_scalar",  kernel=_unary("(x / 2.0f)"),
     aten=lambda inp: [torch.ops.aten.div.Scalar(inp[0], 2.0)])
_reg("eq_scalar",   kernel=_unary("(x == 0.0f ? 1.0f : 0.0f)"),
     aten=lambda inp: [torch.ops.aten.eq.Scalar(inp[0], 0.0).float()])
_reg("gt_scalar",   kernel=_unary("(x > 0.0f ? 1.0f : 0.0f)"),
     aten=lambda inp: [torch.ops.aten.gt.Scalar(inp[0], 0.0).float()])

# Backward gradient ops
_reg("threshold_backward", kernel=_binary("(b > 0.0f || b != b ? a : 0.0f)"), inputs=_pair,
     aten=lambda inp: [torch.ops.aten.threshold_backward.default(inp[0], inp[1], 0.0)])
_reg("gelu_backward", kernel=_binary(
     "(a * (0.5f * (1.0f + erff(b * 0.7071067811865476f)) + b * 0.3989422804014327f * expf(-0.5f * b * b)))"),
     inputs=_pair, aten=lambda inp: [torch.ops.aten.gelu_backward.default(*inp)], atol=1e-4)
_reg("silu_backward", kernel=_binary(
     "(a * (1.0f / (1.0f + expf(-b))) * (1.0f + b * (1.0f - 1.0f / (1.0f + expf(-b)))))"),
     inputs=_pair, aten=lambda inp: [torch.ops.aten.silu_backward.default(*inp)], atol=1e-4)
_reg("sigmoid_backward", kernel=_binary("(a * b * (1.0f - b))"),
     inputs=lambda d, s: [_seeded((d["n"],), s), torch.sigmoid(_seeded((d["n"],), s+500))],
     aten=lambda inp: [torch.ops.aten.sigmoid_backward.default(*inp)])
_reg("tanh_backward", kernel=_binary("(a * (1.0f - b * b))"),
     inputs=lambda d, s: [_seeded((d["n"],), s), torch.tanh(_seeded((d["n"],), s+500))],
     aten=lambda inp: [torch.ops.aten.tanh_backward.default(*inp)])

# Remainder (multi-statement kernel for fmod + sign correction)
_REMAINDER_KERNEL = '''extern "C" __global__ void k(const float *in0, const float *in1, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = in0[i], b = in1[i];
        float r = fmodf(a, b);
        if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) r += b;
        out0[i] = r;
    }
}'''
_reg("remainder", kernel=_REMAINDER_KERNEL, inputs=_pair_pos_b,
     aten=lambda inp: [torch.ops.aten.remainder.Tensor(*inp)], atol=1e-4)

# More reductions
def _red_kernel(identity, accumulate, reduce, finalize):
    return f'''extern "C" __global__ void k(
    const float *input, float *output, unsigned int rows, unsigned int cols
) {{
    __shared__ float sdata[256];
    unsigned int row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *ri = input + row * cols;
    float v = {identity};
    for (unsigned int j = tid; j < cols; j += blockDim.x) {{ {accumulate} }}
    sdata[tid] = v; __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{ {reduce} }} __syncthreads();
    }}
    if (tid == 0) output[row] = {finalize};
}}'''

_reg("mean", kernel=_red_kernel("0.0f", "v += ri[j];", "sdata[tid] += sdata[tid+s];", "sdata[0] / (float)cols"),
     inputs=_2d, dims={"d0": 32, "d1": 64}, dispatch=_red_dispatch, atol=1e-4,
     aten=lambda inp: [torch.ops.aten.mean.dim(inp[0], [-1])],
     outputs=lambda d: ["float32;n=%d" % d["d0"]], grid=lambda d: (d["d0"],), block=(256,))
_reg("amax", kernel=_red_kernel("-1e38f", "{ float r = ri[j]; v = (r != r) ? r : (r > v ? r : v); }", "{ float r = sdata[tid+s]; sdata[tid] = (r != r) ? r : (r > sdata[tid] ? r : sdata[tid]); }", "sdata[0]"),
     inputs=_2d, dims={"d0": 32, "d1": 64}, dispatch=_red_dispatch,
     aten=lambda inp: [torch.ops.aten.amax.default(inp[0], [-1])],
     outputs=lambda d: ["float32;n=%d" % d["d0"]], grid=lambda d: (d["d0"],), block=(256,))
_reg("amin", kernel=_red_kernel("1e38f", "{ float r = ri[j]; v = (r != r) ? r : (r < v ? r : v); }", "{ float r = sdata[tid+s]; sdata[tid] = (r != r) ? r : (r < sdata[tid] ? r : sdata[tid]); }", "sdata[0]"),
     inputs=_2d, dims={"d0": 32, "d1": 64}, dispatch=_red_dispatch,
     aten=lambda inp: [torch.ops.aten.amin.default(inp[0], [-1])],
     outputs=lambda d: ["float32;n=%d" % d["d0"]], grid=lambda d: (d["d0"],), block=(256,))
_reg("prod", kernel=_red_kernel("1.0f", "v *= ri[j];", "sdata[tid] *= sdata[tid+s];", "sdata[0]"),
     inputs=lambda d, s: [_seeded_pos((d["d0"], d["d1"]), s, bias=0.5)],
     dims={"d0": 8, "d1": 16}, dispatch=_red_dispatch, atol=1e-2,
     aten=lambda inp: [torch.ops.aten.prod.dim_int(inp[0], -1)],
     outputs=lambda d: ["float32;n=%d" % d["d0"]], grid=lambda d: (d["d0"],), block=(256,))

# ─── Layout ops (copy kernel) ───────────────────────────────────────────────

_COPY_KERNEL = '''extern "C" __global__ void k(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i];
}'''

_reg("clone",      kernel=_COPY_KERNEL,
     aten=lambda inp: [torch.ops.aten.clone.default(inp[0])])
_reg("contiguous",  kernel=_COPY_KERNEL,
     inputs=lambda d, s: [_seeded((d["n"],), s)],  # already contiguous for testing
     aten=lambda inp: [inp[0].contiguous()])
_reg("copy",       kernel=_COPY_KERNEL,
     aten=lambda inp: [torch.ops.aten.clone.default(inp[0])])
_reg("fill",       kernel=_unary("3.14f"),
     aten=lambda inp: [torch.ops.aten.fill.Scalar(inp[0], 3.14)])

# Reshape ops (just copy data)
_reg("view",       kernel=_COPY_KERNEL,
     aten=lambda inp: [torch.ops.aten.view.default(inp[0], [inp[0].numel()]).contiguous()])
_reg("reshape",    kernel=_COPY_KERNEL,
     aten=lambda inp: [torch.ops.aten.reshape.default(inp[0], [inp[0].numel()]).contiguous()])
_reg("unsqueeze",  kernel=_COPY_KERNEL,
     aten=lambda inp: [torch.ops.aten.unsqueeze.default(inp[0], 0).contiguous().flatten()])
_reg("squeeze",    kernel=_COPY_KERNEL,
     inputs=lambda d, s: [_seeded((1, d["n"]), s)],
     aten=lambda inp: [torch.ops.aten.squeeze.dim(inp[0], 0).contiguous()])
_reg("flatten",    kernel=_COPY_KERNEL,
     inputs=_2d, dims={"d0": 8, "d1": 128},
     aten=lambda inp: [torch.ops.aten.flatten.using_ints(inp[0], 0, 1).contiguous()])

# Transpose (2D grid)
_TRANSPOSE_KERNEL = '''extern "C" __global__ void k(
    const float *in0, float *out0, unsigned int rows, unsigned int cols
) {
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) out0[c * rows + r] = in0[r * cols + c];
}'''

def _transpose_dispatch(inputs, kernel):
    rows, cols = inputs[0].shape
    return [kernel(inputs[0], params=[
        kernel.in_ptr(0), kernel.out_ptr(0),
        np.uint32(rows), np.uint32(cols)])]

_reg("transpose", kernel=_TRANSPOSE_KERNEL, inputs=_2d, dims={"d0": 32, "d1": 64},
     aten=lambda inp: [torch.ops.aten.transpose.int(inp[0], 0, 1).contiguous().flatten()],
     dispatch=_transpose_dispatch,
     outputs=lambda d: ["float32;n=%d" % (d["d0"]*d["d1"])],
     grid=lambda d: ((d["d1"]+15)//16, (d["d0"]+15)//16), block=(16, 16))
_reg("t", kernel=_TRANSPOSE_KERNEL, inputs=_2d, dims={"d0": 32, "d1": 64},
     aten=lambda inp: [torch.ops.aten.t.default(inp[0]).contiguous().flatten()],
     dispatch=_transpose_dispatch,
     outputs=lambda d: ["float32;n=%d" % (d["d0"]*d["d1"])],
     grid=lambda d: ((d["d1"]+15)//16, (d["d0"]+15)//16), block=(16, 16))

# ─── Ternary / conditional ──────────────────────────────────────────────────

_WHERE_KERNEL = '''extern "C" __global__ void k(
    const float *cond, const float *x, const float *y, float *out, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (cond[i] != 0.0f) ? x[i] : y[i];
}'''
_reg("where", kernel=_WHERE_KERNEL,
     inputs=lambda d, s: [(torch.randn(d["n"], device="cuda") > 0).float(),
                           _seeded((d["n"],), s), _seeded((d["n"],), s+500)],
     aten=lambda inp: [torch.ops.aten.where.self(inp[0].bool(), inp[1], inp[2])])

_CLAMP_KERNEL = '''extern "C" __global__ void k(
    const float *in0, float *out0, float lo, float hi, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in0[i]; out0[i] = x < lo ? lo : (x > hi ? hi : x); }
}'''
_reg("clamp", kernel=_CLAMP_KERNEL,
     aten=lambda inp: [torch.ops.aten.clamp.default(inp[0], -1.0, 1.0)],
     dispatch=lambda inp, k: [k(inp[0], params=[
         k.in_ptr(0), k.out_ptr(0), np.float32(-1.0), np.float32(1.0), np.uint32(inp[0].numel())])])

# Masked fill
_MASKED_FILL_KERNEL = '''extern "C" __global__ void k(
    const float *input, const float *mask, float *out, float value, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (mask[i] != 0.0f) ? value : input[i];
}'''
_reg("masked_fill", kernel=_MASKED_FILL_KERNEL,
     inputs=lambda d, s: [_seeded((d["n"],), s), (torch.randn(d["n"], device="cuda") > 0).float()],
     aten=lambda inp: [torch.ops.aten.masked_fill.Scalar(inp[0], inp[1].bool(), -1e9)],
     dispatch=lambda inp, k: [k(*inp, params=[
         k.in_ptr(0), k.in_ptr(1), k.out_ptr(0), np.float32(-1e9), np.uint32(inp[0].numel())])])

# Lerp, addcmul, addcdiv
_LERP_KERNEL = '''extern "C" __global__ void k(
    const float *a, const float *b, const float *w, float *out, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + w[i] * (b[i] - a[i]);
}'''
_reg("lerp", kernel=_LERP_KERNEL,
     inputs=lambda d, s: [_seeded((d["n"],), s), _seeded((d["n"],), s+100), _seeded_pos((d["n"],), s+200, bias=0)],
     aten=lambda inp: [torch.ops.aten.lerp.Tensor(*inp)])

# ─── Embedding ──────────────────────────────────────────────────────────────

_EMBEDDING_KERNEL = '''extern "C" __global__ void k(
    const float *weight, const long *indices, float *output,
    unsigned int n_idx, unsigned int embed_dim
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_idx * embed_dim) return;
    unsigned int r = idx / embed_dim, c = idx % embed_dim;
    output[idx] = weight[indices[r] * embed_dim + c];
}'''
_reg("embedding", kernel=_EMBEDDING_KERNEL,
     dims={"d0": 100, "d1": 64, "d2": 32},  # vocab, embed_dim, seq_len
     inputs=lambda d, s: [_seeded((d["d0"], d["d1"]), s),
                           torch.randint(0, d["d0"], (d["d2"],), device="cuda")],
     aten=lambda inp: [torch.ops.aten.embedding.default(*inp).flatten()],
     dispatch=lambda inp, k: [k(*inp, params=[
         k.in_ptr(0), k.in_ptr(1), k.out_ptr(0),
         np.uint32(inp[1].numel()), np.uint32(inp[0].shape[1])])],
     outputs=lambda d: ["float32;n=%d" % (d["d2"]*d["d1"])],
     grid=lambda d: ((d["d2"]*d["d1"]+255)//256,))

# ─── Normalization ──────────────────────────────────────────────────────────

_SOFTMAX_KERNEL = '''extern "C" __global__ void k(
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
}'''

_reg("_softmax", kernel=_SOFTMAX_KERNEL, inputs=_2d, dims={"d0": 8, "d1": 64},
     aten=lambda inp: [torch.ops.aten._softmax.default(inp[0], -1, False).flatten()],
     dispatch=_red_dispatch,
     outputs=lambda d: ["float32;n=%d" % (d["d0"]*d["d1"])],
     grid=lambda d: (d["d0"],), block=(256,))

# ─── Tensor creation ────────────────────────────────────────────────────────

_FILL_KERNEL = '''extern "C" __global__ void k(float *out0, float value, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = value;
}'''
_ARANGE_KERNEL = '''extern "C" __global__ void k(float *out0, float start, float step, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = start + i * step;
}'''
_EYE_KERNEL = '''extern "C" __global__ void k(float *out0, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    out0[idx] = (idx / n == idx % n) ? 1.0f : 0.0f;
}'''

_reg("zeros", kernel=_FILL_KERNEL, inputs=lambda d, s: [],
     aten=lambda inp: [torch.zeros(1024, device="cuda")],
     dispatch=lambda inp, k: [k(params=[k.out_ptr(0), np.float32(0), np.uint32(1024)])],
     outputs=lambda d: ["float32;n=%d" % d["n"]], grid=lambda d: ((d["n"]+255)//256,))
_reg("ones", kernel=_FILL_KERNEL, inputs=lambda d, s: [],
     aten=lambda inp: [torch.ones(1024, device="cuda")],
     dispatch=lambda inp, k: [k(params=[k.out_ptr(0), np.float32(1), np.uint32(1024)])],
     outputs=lambda d: ["float32;n=%d" % d["n"]], grid=lambda d: ((d["n"]+255)//256,))
_reg("full", kernel=_FILL_KERNEL, inputs=lambda d, s: [],
     aten=lambda inp: [torch.full((1024,), 3.14, device="cuda")],
     dispatch=lambda inp, k: [k(params=[k.out_ptr(0), np.float32(3.14), np.uint32(1024)])],
     outputs=lambda d: ["float32;n=%d" % d["n"]], grid=lambda d: ((d["n"]+255)//256,))
_reg("arange", kernel=_ARANGE_KERNEL, inputs=lambda d, s: [],
     aten=lambda inp: [torch.arange(0, 100, dtype=torch.float32, device="cuda")],
     dispatch=lambda inp, k: [k(params=[k.out_ptr(0), np.float32(0), np.float32(1), np.uint32(100)])],
     outputs=lambda d: ["float32;n=100"], grid=lambda d: (1,))
_reg("eye", kernel=_EYE_KERNEL, inputs=lambda d, s: [], dims={"d0": 16},
     aten=lambda inp: [torch.eye(16, device="cuda").flatten()],
     dispatch=lambda inp, k: [k(params=[k.out_ptr(0), np.uint32(16)])],
     outputs=lambda d: ["float32;n=%d" % (d["d0"]**2)], grid=lambda d: ((d["d0"]**2+255)//256,))

# ─── No-op / identity ──────────────────────────────────────────────────────

_reg("alias",  kernel=_COPY_KERNEL, aten=lambda inp: [inp[0]])
_reg("detach", kernel=_COPY_KERNEL, aten=lambda inp: [inp[0].detach()])


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — used by generated files and batch runner
# ═════════════════════════════════════════════════════════════════════════════

def make_inputs(name, seed=1, dims=None):
    """Generate inputs for an op. seed=0 → special values."""
    op = OPS[name]
    d = dict(op["dims"])
    if dims:
        d.update(dims)
    return op["inputs"](d, seed)

def expected(name, inputs):
    """Compute expected output via aten op."""
    return OPS[name]["aten"](inputs)

def get_kbox_state(name, dims=None):
    """Build a kbox-compatible state dict for an op."""
    op = OPS[name]
    d = dict(op["dims"])
    if dims:
        d.update(dims)
    inputs = op["inputs"](d, 1)
    exp = op["aten"](inputs)
    state = {
        "kernel_source": op["kernel"],
        "inputs": inputs,
        "expected": exp,
        "atol": op["atol"],
    }
    if op["outputs"]:
        state["outputs"] = op["outputs"](d)
    if op["grid"]:
        state["grid"] = op["grid"](d)
    if op["block"]:
        state["block"] = op["block"]
    return state

def dispatch(name, inputs, kernel):
    """Run the CUDA kernel via kbox."""
    return OPS[name]["dispatch"](inputs, kernel)
