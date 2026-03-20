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
_reg("isnan",        kernel=_unary("(isnan(x) ? 1.0f : 0.0f)"),
     aten=lambda inp: [torch.ops.aten.isnan.default(inp[0]).float()])
_reg("isinf",        kernel=_unary("(isinf(x) ? 1.0f : 0.0f)"),
     aten=lambda inp: [torch.ops.aten.isinf.default(inp[0]).float()])

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
