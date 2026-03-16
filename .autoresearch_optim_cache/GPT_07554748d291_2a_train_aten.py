"""Auto-generated aten-level PyTorch program.

This file contains the COMPLETE computation graph decomposed into
low-level aten ops. You can edit ANY operation and rerun this file.

Operations use torch.ops.aten.* - the lowest-level PyTorch ops.
The backward pass (autograd) is also expressed as explicit aten ops.

Parameter mapping:
  input [32, 2048]                           [32, 2048]
  self.cos                                   [1, 20480, 1, 64]
  self.sin                                   [1, 20480, 1, 64]
  self.transformer.wte.weight                [8192, 512]
  self.resid_lambdas                         [8]
  self.x0_lambdas                            [8]
  self.transformer.h.0.attn.c_q.weight       [512, 512]
  self.transformer.h.0.attn.c_k.weight       [512, 512]
  self.transformer.h.0.attn.c_v.weight       [512, 512]
  self.transformer.h.0.attn.c_proj.weight    [512, 512]
  self.transformer.h.0.mlp.c_fc.weight       [2048, 512]
  self.transformer.h.0.mlp.c_proj.weight     [512, 2048]
  self.value_embeds.1.weight                 [8192, 512]
  self.transformer.h.1.attn.c_q.weight       [512, 512]
  self.transformer.h.1.attn.c_k.weight       [512, 512]
  self.transformer.h.1.attn.c_v.weight       [512, 512]
  self.transformer.h.1.attn.ve_gate.weight   [4, 32]
  self.transformer.h.1.attn.c_proj.weight    [512, 512]
  self.transformer.h.1.mlp.c_fc.weight       [2048, 512]
  self.transformer.h.1.mlp.c_proj.weight     [512, 2048]
  self.transformer.h.2.attn.c_q.weight       [512, 512]
  self.transformer.h.2.attn.c_k.weight       [512, 512]
  self.transformer.h.2.attn.c_v.weight       [512, 512]
  self.transformer.h.2.attn.c_proj.weight    [512, 512]
  self.transformer.h.2.mlp.c_fc.weight       [2048, 512]
  self.transformer.h.2.mlp.c_proj.weight     [512, 2048]
  self.value_embeds.3.weight                 [8192, 512]
  self.transformer.h.3.attn.c_q.weight       [512, 512]
  self.transformer.h.3.attn.c_k.weight       [512, 512]
  self.transformer.h.3.attn.c_v.weight       [512, 512]
  self.transformer.h.3.attn.ve_gate.weight   [4, 32]
  self.transformer.h.3.attn.c_proj.weight    [512, 512]
  self.transformer.h.3.mlp.c_fc.weight       [2048, 512]
  self.transformer.h.3.mlp.c_proj.weight     [512, 2048]
  self.transformer.h.4.attn.c_q.weight       [512, 512]
  self.transformer.h.4.attn.c_k.weight       [512, 512]
  self.transformer.h.4.attn.c_v.weight       [512, 512]
  self.transformer.h.4.attn.c_proj.weight    [512, 512]
  self.transformer.h.4.mlp.c_fc.weight       [2048, 512]
  self.transformer.h.4.mlp.c_proj.weight     [512, 2048]
  self.value_embeds.5.weight                 [8192, 512]
  self.transformer.h.5.attn.c_q.weight       [512, 512]
  self.transformer.h.5.attn.c_k.weight       [512, 512]
  self.transformer.h.5.attn.c_v.weight       [512, 512]
  self.transformer.h.5.attn.ve_gate.weight   [4, 32]
  self.transformer.h.5.attn.c_proj.weight    [512, 512]
  self.transformer.h.5.mlp.c_fc.weight       [2048, 512]
  self.transformer.h.5.mlp.c_proj.weight     [512, 2048]
  self.transformer.h.6.attn.c_q.weight       [512, 512]
  self.transformer.h.6.attn.c_k.weight       [512, 512]
  self.transformer.h.6.attn.c_v.weight       [512, 512]
  self.transformer.h.6.attn.c_proj.weight    [512, 512]
  self.transformer.h.6.mlp.c_fc.weight       [2048, 512]
  self.transformer.h.6.mlp.c_proj.weight     [512, 2048]
  self.value_embeds.7.weight                 [8192, 512]
  self.transformer.h.7.attn.c_q.weight       [512, 512]
  self.transformer.h.7.attn.c_k.weight       [512, 512]
  self.transformer.h.7.attn.c_v.weight       [512, 512]
  self.transformer.h.7.attn.ve_gate.weight   [4, 32]
  self.transformer.h.7.attn.c_proj.weight    [512, 512]
  self.transformer.h.7.mlp.c_fc.weight       [2048, 512]
  self.transformer.h.7.mlp.c_proj.weight     [512, 2048]
  self.lm_head.weight                        [8192, 512]
  input [32, 2048]                           [32, 2048]
"""

import operator
import os
import torch
aten = torch.ops.aten  # shorthand for low-level ops

# Device: defaults to CUDA if available, use --cpu to force CPU
import argparse as _argparse
_dev_parser = _argparse.ArgumentParser(add_help=False)
_dev_parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
_dev_parser.add_argument("--atol", type=float, default=None, help="Absolute tolerance for verification")
_dev_parser.add_argument("--rtol", type=float, default=None, help="Relative tolerance for verification")
_dev_args, _ = _dev_parser.parse_known_args()
_device = "cpu" if _dev_args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")


# ── Triton squared ReLU kernel ───────────────────────────────────────
import triton
import triton.language as tl

@triton.jit
def _squared_relu_kernel(
    x_ptr, out_ptr, n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # relu then square
    r = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offs, r * r, mask=mask)

def triton_squared_relu(x):
    """Fused relu().square() → single kernel."""
    out = torch.empty_like(x)
    n = x.numel()
    _squared_relu_kernel[(n + 1023) // 1024](x, out, n, BLOCK=1024)
    return out
# ── End Triton squared ReLU kernel ──────────────────────────────────


# ── Inline CUDA kernel: vectorized bf16 add ──────────────────────────
import torch.utils.cpp_extension

_cuda_add_cpp = "torch::Tensor cuda_bf16_add(torch::Tensor a, torch::Tensor b);"

_cuda_add_cu = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>

__global__ void bf16_add_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int64_t n
) {
    // Process 4 elements per thread for better throughput
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        // Vectorized load/store (2x bf16 = 1x int32)
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a + idx);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b + idx);
        __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out + idx);
        out2[0] = __hadd2(a2[0], b2[0]);
        out2[1] = __hadd2(a2[1], b2[1]);
    } else {
        for (int64_t i = idx; i < min(idx + 4, n); i++) {
            out[i] = __hadd(a[i], b[i]);
        }
    }
}

torch::Tensor cuda_bf16_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dtype() == torch::kBFloat16, "expected bf16");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
    auto out = torch::empty_like(a);
    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    bf16_add_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        n
    );
    return out;
}
"""

_cuda_add_mod = torch.utils.cpp_extension.load_inline(
    name="cuda_bf16_add_v2",
    cpp_sources=_cuda_add_cpp,
    cuda_sources=_cuda_add_cu,
    functions=["cuda_bf16_add"],
    verbose=False,
)

def cuda_add(a, b):
    """Drop-in replacement for aten.add.Tensor using vectorized CUDA kernel."""
    return _cuda_add_mod.cuda_bf16_add(a.contiguous(), b.contiguous())
# ── End inline CUDA kernel ───────────────────────────────────────────


# ── Triton squared ReLU kernel ───────────────────────────────────────
import triton
import triton.language as tl

@triton.jit
def _squared_relu_kernel(
    x_ptr, out_ptr, n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # relu then square
    r = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offs, r * r, mask=mask)

def triton_squared_relu(x):
    """Fused relu().square() → single kernel."""
    out = torch.empty_like(x)
    n = x.numel()
    _squared_relu_kernel[(n + 1023) // 1024](x, out, n, BLOCK=1024)
    return out
# ── End Triton squared ReLU kernel ──────────────────────────────────


# ── Inline CUDA kernel: vectorized bf16 add ──────────────────────────
import torch.utils.cpp_extension

_cuda_add_cpp = "torch::Tensor cuda_bf16_add(torch::Tensor a, torch::Tensor b);"

_cuda_add_cu = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>

__global__ void bf16_add_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int64_t n
) {
    // Process 4 elements per thread for better throughput
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        // Vectorized load/store (2x bf16 = 1x int32)
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a + idx);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b + idx);
        __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out + idx);
        out2[0] = __hadd2(a2[0], b2[0]);
        out2[1] = __hadd2(a2[1], b2[1]);
    } else {
        for (int64_t i = idx; i < min(idx + 4, n); i++) {
            out[i] = __hadd(a[i], b[i]);
        }
    }
}

torch::Tensor cuda_bf16_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dtype() == torch::kBFloat16, "expected bf16");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
    auto out = torch::empty_like(a);
    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    bf16_add_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        n
    );
    return out;
}
"""

_cuda_add_mod = torch.utils.cpp_extension.load_inline(
    name="cuda_bf16_add_v2",
    cpp_sources=_cuda_add_cpp,
    cuda_sources=_cuda_add_cu,
    functions=["cuda_bf16_add"],
    verbose=False,
)

def cuda_add(a, b):
    """Drop-in replacement for aten.add.Tensor using vectorized CUDA kernel."""
    return _cuda_add_mod.cuda_bf16_add(a.contiguous(), b.contiguous())
# ── End inline CUDA kernel ───────────────────────────────────────────

# ── Triton fused lambda scale kernel ─────────────────────────────────
# Fuses: select(resid_lambdas) * x + select(x0_lambdas) * x0 → single kernel
# Replaces 2 scalar-tensor muls + 1 add per layer (8 layers = 24 ops → 8 ops)
@triton.jit
def _lambda_scale_kernel(
    x_ptr, x0_ptr, out_ptr, a, b, n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    x0 = tl.load(x0_ptr + offs, mask=mask).to(tl.float32)
    result = a * x + b * x0
    tl.store(out_ptr + offs, result.to(tl.bfloat16), mask=mask)

def triton_lambda_scale(x, x0, a_scalar, b_scalar):
    """Fused a*x + b*x0 for lambda scaling."""
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    _lambda_scale_kernel[grid](x, x0, out, a_scalar.item(), b_scalar.item(), n, BLOCK=BLOCK)
    return out
# ── End Triton fused lambda scale kernel ────────────────────────────

# ── Triton fused softcap forward kernel ──────────────────────────────
# Fuses: _to_copy(bf16→fp32) + div(softcap) + tanh + mul(softcap)
# Produces both the softcapped logits AND the tanh output (for backward)
from triton.language.extra.cuda import libdevice as _libdevice

@triton.jit
def _softcap_fwd_kernel(
    x_ptr, out_ptr, tanh_ptr, n,
    SOFTCAP: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    t = _libdevice.tanh(x / SOFTCAP)
    tl.store(tanh_ptr + offs, t, mask=mask)
    tl.store(out_ptr + offs, t * SOFTCAP, mask=mask)

def triton_softcap_fwd(x_bf16, softcap=15):
    """Fused softcap forward: bf16→fp32, div, tanh, mul in one kernel."""
    n = x_bf16.numel()
    shape = x_bf16.shape
    out = torch.empty(shape, dtype=torch.float32, device=x_bf16.device)
    tanh_out = torch.empty(shape, dtype=torch.float32, device=x_bf16.device)
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    _softcap_fwd_kernel[grid](x_bf16, out, tanh_out, n, SOFTCAP=softcap, BLOCK=BLOCK)
    return out, tanh_out
# ── End Triton fused softcap forward kernel ─────────────────────────

# ── Triton fused softcap backward kernel ────────────────────────────
# Fuses: mul(softcap) + tanh_backward + div(softcap) + _to_copy(fp32→bf16)
@triton.jit
def _softcap_bwd_kernel(
    grad_ptr, tanh_ptr, out_ptr, n,
    SOFTCAP: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    grad = tl.load(grad_ptr + offs, mask=mask)  # fp32
    t = tl.load(tanh_ptr + offs, mask=mask)      # fp32 tanh output
    # grad * softcap * (1 - tanh^2) / softcap = grad * (1 - tanh^2)
    result = grad * (1.0 - t * t)
    tl.store(out_ptr + offs, result.to(tl.bfloat16), mask=mask)

def triton_softcap_bwd(grad_fp32, tanh_out_fp32, softcap=15):
    """Fused softcap backward: mul(sc) + tanh_bwd + div(sc) + to_bf16 in one kernel."""
    n = grad_fp32.numel()
    shape = grad_fp32.shape
    out = torch.empty(shape, dtype=torch.bfloat16, device=grad_fp32.device)
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    _softcap_bwd_kernel[grid](grad_fp32, tanh_out_fp32, out, n, SOFTCAP=softcap, BLOCK=BLOCK)
    return out
# ── End Triton fused softcap backward kernel ────────────────────────

# ── Triton fused squared_relu backward kernel ───────────────────────
# Fuses: pow(x,1) + mul(2) + mul(grad,result) + _to_copy(bf16) + threshold_backward
# = grad * 2 * relu_fp32 * (relu > 0) → bf16
@triton.jit
def _squared_relu_bwd_kernel(
    grad_ptr, relu_fp32_ptr, relu_bf16_ptr, out_ptr, n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    grad = tl.load(grad_ptr + offs, mask=mask)           # fp32
    relu_fp32 = tl.load(relu_fp32_ptr + offs, mask=mask)  # fp32
    relu_bf16 = tl.load(relu_bf16_ptr + offs, mask=mask).to(tl.float32)  # bf16 → fp32
    # d/dx [relu(x)^2] = 2*relu(x) * (x > 0)
    # threshold_backward: output 0 where relu was 0
    result = tl.where(relu_bf16 > 0, grad * 2.0 * relu_fp32, 0.0)
    tl.store(out_ptr + offs, result.to(tl.bfloat16), mask=mask)

def triton_squared_relu_bwd(grad_fp32, relu_fp32, relu_bf16):
    """Fused squared relu backward: grad * 2 * relu * (relu > 0) → bf16."""
    n = grad_fp32.numel()
    out = torch.empty(grad_fp32.shape, dtype=torch.bfloat16, device=grad_fp32.device)
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    _squared_relu_bwd_kernel[grid](grad_fp32, relu_fp32, relu_bf16, out, n, BLOCK=BLOCK)
    return out
# ── End Triton fused squared_relu backward kernel ──────────────────

# ── Triton fused RoPE forward kernel ────────────────────────────────
# Fuses: slice(x, low/high) + mul(cos/sin) + neg + add + cat
# Input x: [B, T, H, D] where D=128, HALF_D=64
# cos, sin: [1, T, 1, HALF_D]
# Output: [B, T, H, D] with rotary embedding applied
@triton.jit
def _rope_fwd_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    B, T, H, HALF_D: tl.constexpr,
    stride_xb, stride_xt, stride_xh, stride_xd,
    stride_cb, stride_ct, stride_ch, stride_cd,
    BLOCK_D: tl.constexpr,
):
    # Each program handles one (b, t, h) combination
    pid = tl.program_id(0)
    b = pid // (T * H)
    rem = pid % (T * H)
    t = rem // H
    h = rem % H

    # Load cos and sin for this position
    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < HALF_D

    cos_base = t * stride_ct + d_offs * stride_cd
    sin_base = t * stride_ct + d_offs * stride_cd
    cos_val = tl.load(cos_ptr + cos_base, mask=mask)
    sin_val = tl.load(sin_ptr + sin_base, mask=mask)

    # Load x_low (first HALF_D) and x_high (second HALF_D)
    x_base = b * stride_xb + t * stride_xt + h * stride_xh
    x_low = tl.load(x_ptr + x_base + d_offs * stride_xd, mask=mask)
    x_high = tl.load(x_ptr + x_base + (d_offs + HALF_D) * stride_xd, mask=mask)

    # RoPE: out_low = x_low * cos + x_high * sin
    #        out_high = x_low * (-sin) + x_high * cos
    out_low = x_low * cos_val + x_high * sin_val
    out_high = x_low * (-sin_val) + x_high * cos_val

    tl.store(out_ptr + x_base + d_offs * stride_xd, out_low, mask=mask)
    tl.store(out_ptr + x_base + (d_offs + HALF_D) * stride_xd, out_high, mask=mask)

def triton_rope_fwd(x, cos, sin):
    """Fused RoPE forward: replaces slice+mul+neg+mul+add+cat (10 ops → 1 kernel).
    x: [B, T, H, D], cos/sin: [1, T, 1, HALF_D]
    """
    B, T, H, D = x.shape
    HALF_D = D // 2
    out = torch.empty_like(x)
    grid = (B * T * H,)
    _rope_fwd_kernel[grid](
        x, cos, sin, out,
        B, T, H, HALF_D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1), cos.stride(2), cos.stride(3),
        BLOCK_D=64,  # HALF_D=64 fits in one block
    )
    return out
# ── End Triton fused RoPE forward kernel ───────────────────────────


# ======================================================================
# WEIGHTS / PARAMETERS
# ======================================================================

input__32__2048 = torch.zeros([32, 2048], dtype=torch.int64)  # input [32, 2048]
cos = torch.randn([1, 20480, 1, 64], dtype=torch.bfloat16)  # self.cos
sin = torch.randn([1, 20480, 1, 64], dtype=torch.bfloat16)  # self.sin
transformer_wte_weight = torch.randn([8192, 512], dtype=torch.bfloat16)  # self.transformer.wte.weight
resid_lambdas = torch.randn([8], dtype=torch.float32)  # self.resid_lambdas
x0_lambdas = torch.randn([8], dtype=torch.float32)  # self.x0_lambdas
transformer_h_0_attn_c_q_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.0.attn.c_q.weight
transformer_h_0_attn_c_k_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.0.attn.c_k.weight
transformer_h_0_attn_c_v_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.0.attn.c_v.weight
transformer_h_0_attn_c_proj_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.0.attn.c_proj.weight
transformer_h_0_mlp_c_fc_weight = torch.randn([2048, 512], dtype=torch.float32)  # self.transformer.h.0.mlp.c_fc.weight
transformer_h_0_mlp_c_proj_weight = torch.randn([512, 2048], dtype=torch.float32)  # self.transformer.h.0.mlp.c_proj.weight
value_embeds_1_weight = torch.randn([8192, 512], dtype=torch.bfloat16)  # self.value_embeds.1.weight
transformer_h_1_attn_c_q_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.1.attn.c_q.weight
transformer_h_1_attn_c_k_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.1.attn.c_k.weight
transformer_h_1_attn_c_v_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.1.attn.c_v.weight
transformer_h_1_attn_ve_gate_weight = torch.randn([4, 32], dtype=torch.float32)  # self.transformer.h.1.attn.ve_gate.weight
transformer_h_1_attn_c_proj_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.1.attn.c_proj.weight
transformer_h_1_mlp_c_fc_weight = torch.randn([2048, 512], dtype=torch.float32)  # self.transformer.h.1.mlp.c_fc.weight
transformer_h_1_mlp_c_proj_weight = torch.randn([512, 2048], dtype=torch.float32)  # self.transformer.h.1.mlp.c_proj.weight
transformer_h_2_attn_c_q_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.2.attn.c_q.weight
transformer_h_2_attn_c_k_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.2.attn.c_k.weight
transformer_h_2_attn_c_v_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.2.attn.c_v.weight
transformer_h_2_attn_c_proj_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.2.attn.c_proj.weight
transformer_h_2_mlp_c_fc_weight = torch.randn([2048, 512], dtype=torch.float32)  # self.transformer.h.2.mlp.c_fc.weight
transformer_h_2_mlp_c_proj_weight = torch.randn([512, 2048], dtype=torch.float32)  # self.transformer.h.2.mlp.c_proj.weight
value_embeds_3_weight = torch.randn([8192, 512], dtype=torch.bfloat16)  # self.value_embeds.3.weight
transformer_h_3_attn_c_q_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.3.attn.c_q.weight
transformer_h_3_attn_c_k_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.3.attn.c_k.weight
transformer_h_3_attn_c_v_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.3.attn.c_v.weight
transformer_h_3_attn_ve_gate_weight = torch.randn([4, 32], dtype=torch.float32)  # self.transformer.h.3.attn.ve_gate.weight
transformer_h_3_attn_c_proj_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.3.attn.c_proj.weight
transformer_h_3_mlp_c_fc_weight = torch.randn([2048, 512], dtype=torch.float32)  # self.transformer.h.3.mlp.c_fc.weight
transformer_h_3_mlp_c_proj_weight = torch.randn([512, 2048], dtype=torch.float32)  # self.transformer.h.3.mlp.c_proj.weight
transformer_h_4_attn_c_q_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.4.attn.c_q.weight
transformer_h_4_attn_c_k_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.4.attn.c_k.weight
transformer_h_4_attn_c_v_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.4.attn.c_v.weight
transformer_h_4_attn_c_proj_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.4.attn.c_proj.weight
transformer_h_4_mlp_c_fc_weight = torch.randn([2048, 512], dtype=torch.float32)  # self.transformer.h.4.mlp.c_fc.weight
transformer_h_4_mlp_c_proj_weight = torch.randn([512, 2048], dtype=torch.float32)  # self.transformer.h.4.mlp.c_proj.weight
value_embeds_5_weight = torch.randn([8192, 512], dtype=torch.bfloat16)  # self.value_embeds.5.weight
transformer_h_5_attn_c_q_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.5.attn.c_q.weight
transformer_h_5_attn_c_k_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.5.attn.c_k.weight
transformer_h_5_attn_c_v_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.5.attn.c_v.weight
transformer_h_5_attn_ve_gate_weight = torch.randn([4, 32], dtype=torch.float32)  # self.transformer.h.5.attn.ve_gate.weight
transformer_h_5_attn_c_proj_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.5.attn.c_proj.weight
transformer_h_5_mlp_c_fc_weight = torch.randn([2048, 512], dtype=torch.float32)  # self.transformer.h.5.mlp.c_fc.weight
transformer_h_5_mlp_c_proj_weight = torch.randn([512, 2048], dtype=torch.float32)  # self.transformer.h.5.mlp.c_proj.weight
transformer_h_6_attn_c_q_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.6.attn.c_q.weight
transformer_h_6_attn_c_k_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.6.attn.c_k.weight
transformer_h_6_attn_c_v_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.6.attn.c_v.weight
transformer_h_6_attn_c_proj_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.6.attn.c_proj.weight
transformer_h_6_mlp_c_fc_weight = torch.randn([2048, 512], dtype=torch.float32)  # self.transformer.h.6.mlp.c_fc.weight
transformer_h_6_mlp_c_proj_weight = torch.randn([512, 2048], dtype=torch.float32)  # self.transformer.h.6.mlp.c_proj.weight
value_embeds_7_weight = torch.randn([8192, 512], dtype=torch.bfloat16)  # self.value_embeds.7.weight
transformer_h_7_attn_c_q_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.7.attn.c_q.weight
transformer_h_7_attn_c_k_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.7.attn.c_k.weight
transformer_h_7_attn_c_v_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.7.attn.c_v.weight
transformer_h_7_attn_ve_gate_weight = torch.randn([4, 32], dtype=torch.float32)  # self.transformer.h.7.attn.ve_gate.weight
transformer_h_7_attn_c_proj_weight = torch.randn([512, 512], dtype=torch.float32)  # self.transformer.h.7.attn.c_proj.weight
transformer_h_7_mlp_c_fc_weight = torch.randn([2048, 512], dtype=torch.float32)  # self.transformer.h.7.mlp.c_fc.weight
transformer_h_7_mlp_c_proj_weight = torch.randn([512, 2048], dtype=torch.float32)  # self.transformer.h.7.mlp.c_proj.weight
lm_head_weight = torch.randn([8192, 512], dtype=torch.float32)  # self.lm_head.weight
input__32__2048_1 = torch.zeros([32, 2048], dtype=torch.int64)  # input [32, 2048]

# Move to target device
input__32__2048 = input__32__2048.to(_device)
cos = cos.to(_device)
sin = sin.to(_device)
transformer_wte_weight = transformer_wte_weight.to(_device)
resid_lambdas = resid_lambdas.to(_device)
x0_lambdas = x0_lambdas.to(_device)
transformer_h_0_attn_c_q_weight = transformer_h_0_attn_c_q_weight.to(_device)
transformer_h_0_attn_c_k_weight = transformer_h_0_attn_c_k_weight.to(_device)
transformer_h_0_attn_c_v_weight = transformer_h_0_attn_c_v_weight.to(_device)
transformer_h_0_attn_c_proj_weight = transformer_h_0_attn_c_proj_weight.to(_device)
transformer_h_0_mlp_c_fc_weight = transformer_h_0_mlp_c_fc_weight.to(_device)
transformer_h_0_mlp_c_proj_weight = transformer_h_0_mlp_c_proj_weight.to(_device)
value_embeds_1_weight = value_embeds_1_weight.to(_device)
transformer_h_1_attn_c_q_weight = transformer_h_1_attn_c_q_weight.to(_device)
transformer_h_1_attn_c_k_weight = transformer_h_1_attn_c_k_weight.to(_device)
transformer_h_1_attn_c_v_weight = transformer_h_1_attn_c_v_weight.to(_device)
transformer_h_1_attn_ve_gate_weight = transformer_h_1_attn_ve_gate_weight.to(_device)
transformer_h_1_attn_c_proj_weight = transformer_h_1_attn_c_proj_weight.to(_device)
transformer_h_1_mlp_c_fc_weight = transformer_h_1_mlp_c_fc_weight.to(_device)
transformer_h_1_mlp_c_proj_weight = transformer_h_1_mlp_c_proj_weight.to(_device)
transformer_h_2_attn_c_q_weight = transformer_h_2_attn_c_q_weight.to(_device)
transformer_h_2_attn_c_k_weight = transformer_h_2_attn_c_k_weight.to(_device)
transformer_h_2_attn_c_v_weight = transformer_h_2_attn_c_v_weight.to(_device)
transformer_h_2_attn_c_proj_weight = transformer_h_2_attn_c_proj_weight.to(_device)
transformer_h_2_mlp_c_fc_weight = transformer_h_2_mlp_c_fc_weight.to(_device)
transformer_h_2_mlp_c_proj_weight = transformer_h_2_mlp_c_proj_weight.to(_device)
value_embeds_3_weight = value_embeds_3_weight.to(_device)
transformer_h_3_attn_c_q_weight = transformer_h_3_attn_c_q_weight.to(_device)
transformer_h_3_attn_c_k_weight = transformer_h_3_attn_c_k_weight.to(_device)
transformer_h_3_attn_c_v_weight = transformer_h_3_attn_c_v_weight.to(_device)
transformer_h_3_attn_ve_gate_weight = transformer_h_3_attn_ve_gate_weight.to(_device)
transformer_h_3_attn_c_proj_weight = transformer_h_3_attn_c_proj_weight.to(_device)
transformer_h_3_mlp_c_fc_weight = transformer_h_3_mlp_c_fc_weight.to(_device)
transformer_h_3_mlp_c_proj_weight = transformer_h_3_mlp_c_proj_weight.to(_device)
transformer_h_4_attn_c_q_weight = transformer_h_4_attn_c_q_weight.to(_device)
transformer_h_4_attn_c_k_weight = transformer_h_4_attn_c_k_weight.to(_device)
transformer_h_4_attn_c_v_weight = transformer_h_4_attn_c_v_weight.to(_device)
transformer_h_4_attn_c_proj_weight = transformer_h_4_attn_c_proj_weight.to(_device)
transformer_h_4_mlp_c_fc_weight = transformer_h_4_mlp_c_fc_weight.to(_device)
transformer_h_4_mlp_c_proj_weight = transformer_h_4_mlp_c_proj_weight.to(_device)
value_embeds_5_weight = value_embeds_5_weight.to(_device)
transformer_h_5_attn_c_q_weight = transformer_h_5_attn_c_q_weight.to(_device)
transformer_h_5_attn_c_k_weight = transformer_h_5_attn_c_k_weight.to(_device)
transformer_h_5_attn_c_v_weight = transformer_h_5_attn_c_v_weight.to(_device)
transformer_h_5_attn_ve_gate_weight = transformer_h_5_attn_ve_gate_weight.to(_device)
transformer_h_5_attn_c_proj_weight = transformer_h_5_attn_c_proj_weight.to(_device)
transformer_h_5_mlp_c_fc_weight = transformer_h_5_mlp_c_fc_weight.to(_device)
transformer_h_5_mlp_c_proj_weight = transformer_h_5_mlp_c_proj_weight.to(_device)
transformer_h_6_attn_c_q_weight = transformer_h_6_attn_c_q_weight.to(_device)
transformer_h_6_attn_c_k_weight = transformer_h_6_attn_c_k_weight.to(_device)
transformer_h_6_attn_c_v_weight = transformer_h_6_attn_c_v_weight.to(_device)
transformer_h_6_attn_c_proj_weight = transformer_h_6_attn_c_proj_weight.to(_device)
transformer_h_6_mlp_c_fc_weight = transformer_h_6_mlp_c_fc_weight.to(_device)
transformer_h_6_mlp_c_proj_weight = transformer_h_6_mlp_c_proj_weight.to(_device)
value_embeds_7_weight = value_embeds_7_weight.to(_device)
transformer_h_7_attn_c_q_weight = transformer_h_7_attn_c_q_weight.to(_device)
transformer_h_7_attn_c_k_weight = transformer_h_7_attn_c_k_weight.to(_device)
transformer_h_7_attn_c_v_weight = transformer_h_7_attn_c_v_weight.to(_device)
transformer_h_7_attn_ve_gate_weight = transformer_h_7_attn_ve_gate_weight.to(_device)
transformer_h_7_attn_c_proj_weight = transformer_h_7_attn_c_proj_weight.to(_device)
transformer_h_7_mlp_c_fc_weight = transformer_h_7_mlp_c_fc_weight.to(_device)
transformer_h_7_mlp_c_proj_weight = transformer_h_7_mlp_c_proj_weight.to(_device)
lm_head_weight = lm_head_weight.to(_device)
input__32__2048_1 = input__32__2048_1.to(_device)


# ======================================================================
# SHARED LAYER FUNCTIONS
# ======================================================================
# Repeated module groups extracted into reusable functions.
# Edit once — changes apply to all instances.
# ======================================================================

def embedding(
    weight: 'bfloat16[8192, 512]',
    input__32__2048: 'int64[32, 2048]',
) -> tuple['bfloat16[32, 2048, 512]']:
    value_embeds1_embedding: 'bfloat16[32, 2048, 512]' = aten.embedding(weight, input__32__2048)
    return (value_embeds1_embedding,)

# ======================================================================
# FORWARD PASS (aten ops)
# ======================================================================
# Edit any operation below. The ops are pure aten - no autograd,
# no module abstractions, just raw tensor operations.
# 
# Source annotations show which part of the original PyTorch model
# or function each op/group came from, with compact file:line notes
# for source-only ops and section headers for module groups.
# 
# The return value includes saved tensors needed for backward.
# ======================================================================

def forward(
    input__32__2048: 'int64[32, 2048]',
    cos: 'bfloat16[1, 20480, 1, 64]',
    sin: 'bfloat16[1, 20480, 1, 64]',
    transformer_wte_weight: 'bfloat16[8192, 512]',
    resid_lambdas: 'float32[8]',
    x0_lambdas: 'float32[8]',
    transformer_h_0_attn_c_q_weight: 'float32[512, 512]',
    transformer_h_0_attn_c_k_weight: 'float32[512, 512]',
    transformer_h_0_attn_c_v_weight: 'float32[512, 512]',
    transformer_h_0_attn_c_proj_weight: 'float32[512, 512]',
    transformer_h_0_mlp_c_fc_weight: 'float32[2048, 512]',
    transformer_h_0_mlp_c_proj_weight: 'float32[512, 2048]',
    value_embeds_1_weight: 'bfloat16[8192, 512]',
    transformer_h_1_attn_c_q_weight: 'float32[512, 512]',
    transformer_h_1_attn_c_k_weight: 'float32[512, 512]',
    transformer_h_1_attn_c_v_weight: 'float32[512, 512]',
    transformer_h_1_attn_ve_gate_weight: 'float32[4, 32]',
    transformer_h_1_attn_c_proj_weight: 'float32[512, 512]',
    transformer_h_1_mlp_c_fc_weight: 'float32[2048, 512]',
    transformer_h_1_mlp_c_proj_weight: 'float32[512, 2048]',
    transformer_h_2_attn_c_q_weight: 'float32[512, 512]',
    transformer_h_2_attn_c_k_weight: 'float32[512, 512]',
    transformer_h_2_attn_c_v_weight: 'float32[512, 512]',
    transformer_h_2_attn_c_proj_weight: 'float32[512, 512]',
    transformer_h_2_mlp_c_fc_weight: 'float32[2048, 512]',
    transformer_h_2_mlp_c_proj_weight: 'float32[512, 2048]',
    value_embeds_3_weight: 'bfloat16[8192, 512]',
    transformer_h_3_attn_c_q_weight: 'float32[512, 512]',
    transformer_h_3_attn_c_k_weight: 'float32[512, 512]',
    transformer_h_3_attn_c_v_weight: 'float32[512, 512]',
    transformer_h_3_attn_ve_gate_weight: 'float32[4, 32]',
    transformer_h_3_attn_c_proj_weight: 'float32[512, 512]',
    transformer_h_3_mlp_c_fc_weight: 'float32[2048, 512]',
    transformer_h_3_mlp_c_proj_weight: 'float32[512, 2048]',
    transformer_h_4_attn_c_q_weight: 'float32[512, 512]',
    transformer_h_4_attn_c_k_weight: 'float32[512, 512]',
    transformer_h_4_attn_c_v_weight: 'float32[512, 512]',
    transformer_h_4_attn_c_proj_weight: 'float32[512, 512]',
    transformer_h_4_mlp_c_fc_weight: 'float32[2048, 512]',
    transformer_h_4_mlp_c_proj_weight: 'float32[512, 2048]',
    value_embeds_5_weight: 'bfloat16[8192, 512]',
    transformer_h_5_attn_c_q_weight: 'float32[512, 512]',
    transformer_h_5_attn_c_k_weight: 'float32[512, 512]',
    transformer_h_5_attn_c_v_weight: 'float32[512, 512]',
    transformer_h_5_attn_ve_gate_weight: 'float32[4, 32]',
    transformer_h_5_attn_c_proj_weight: 'float32[512, 512]',
    transformer_h_5_mlp_c_fc_weight: 'float32[2048, 512]',
    transformer_h_5_mlp_c_proj_weight: 'float32[512, 2048]',
    transformer_h_6_attn_c_q_weight: 'float32[512, 512]',
    transformer_h_6_attn_c_k_weight: 'float32[512, 512]',
    transformer_h_6_attn_c_v_weight: 'float32[512, 512]',
    transformer_h_6_attn_c_proj_weight: 'float32[512, 512]',
    transformer_h_6_mlp_c_fc_weight: 'float32[2048, 512]',
    transformer_h_6_mlp_c_proj_weight: 'float32[512, 2048]',
    value_embeds_7_weight: 'bfloat16[8192, 512]',
    transformer_h_7_attn_c_q_weight: 'float32[512, 512]',
    transformer_h_7_attn_c_k_weight: 'float32[512, 512]',
    transformer_h_7_attn_c_v_weight: 'float32[512, 512]',
    transformer_h_7_attn_ve_gate_weight: 'float32[4, 32]',
    transformer_h_7_attn_c_proj_weight: 'float32[512, 512]',
    transformer_h_7_mlp_c_fc_weight: 'float32[2048, 512]',
    transformer_h_7_mlp_c_proj_weight: 'float32[512, 2048]',
    lm_head_weight: 'float32[8192, 512]',
    input__32__2048_1: 'int64[32, 2048]',
):

    # /.autoresearch_repo/train.py:266
    # cos_sin = self.cos[:, :T], self.sin[:, :T]
    getitem_slice: 'bfloat16[1, 2048, 1, 64]' = aten.slice.Tensor(cos, 1, 0, 2048)  # strides=(1310720, 64, 64, 1), contiguous=True, view=True
    getitem_1_slice: 'bfloat16[1, 2048, 1, 64]' = aten.slice.Tensor(sin, 1, 0, 2048)  # strides=(1310720, 64, 64, 1), contiguous=True, view=True
    # Precompute -sin once (reused for all 16 RoPE neg computations)
    _neg_sin: 'bfloat16[1, 2048, 1, 64]' = aten.neg(getitem_1_slice)
    # Precompute attention mask once (reused for 6 windowed attention layers)
    _arange = aten.arange(2048, device=torch.device('cuda:0'), pin_memory=False)
    _row_idx = aten.unsqueeze(_arange, 1)
    _row_idx_shifted = aten.add.Tensor(_row_idx, 0)
    _col_idx = aten.unsqueeze(_arange, 0)
    _causal = aten.le.Tensor(_col_idx, _row_idx_shifted)
    _window_dist = aten.sub.Tensor(_row_idx_shifted, _col_idx)
    _in_window = aten.le.Scalar(_window_dist, 1024)
    _mask_bool = aten.bitwise_and.Tensor(_causal, _in_window)
    _neg_inf = aten.scalar_tensor(-65504.0, dtype=torch.bfloat16, device=torch.device('cuda:0'))
    _zero = aten.scalar_tensor(0.0, dtype=torch.bfloat16, layout=torch.strided, device=torch.device('cuda:0'))
    _attn_mask: 'bfloat16[2048, 2048]' = aten.where.self(_mask_bool, _zero, _neg_inf)

    # ════════════════════════════════════════════════════════════════
    # self.transformer.wte
    # ════════════════════════════════════════════════════════════════

    # self.transformer.wte (Embedding)
    # /.autoresearch_repo/train.py:268
    # x = self.transformer.wte(idx)
    wte_embedding: 'bfloat16[32, 2048, 512]' = aten.embedding(transformer_wte_weight, input__32__2048)  # strides=(1048576, 512, 1), contiguous=True, view=False

    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    rms_norm__fused_rms_norm = aten._fused_rms_norm(wte_embedding, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    rms_norm_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(rms_norm__fused_rms_norm, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    rms_norm_getitem_1: 'float32[32, 2048, 1]' = operator.getitem(rms_norm__fused_rms_norm, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    rms_norm_detach: 'float32[32, 2048, 1]' = aten.detach(rms_norm_getitem_1)  # strides=(2048, 1, 1), contiguous=True, view=True

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    getitem_2_select: 'float32[]' = aten.select.int(resid_lambdas, 0, 0)  # strides=(), contiguous=True, view=True
    getitem_3_select: 'float32[]' = aten.select.int(x0_lambdas, 0, 0)  # strides=(), contiguous=True, view=True
    add_add: 'bfloat16[32, 2048, 512]' = triton_lambda_scale(rms_norm_getitem, rms_norm_getitem, getitem_2_select, getitem_3_select)  # FUSED: a*x + b*x0

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.0
    # ════════════════════════════════════════════════════════════════

    # self.transformer.h.0 (Block)
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    h0__fused_rms_norm = aten._fused_rms_norm(add_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h0_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(h0__fused_rms_norm, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h0_getitem_1: 'float32[32, 2048, 1]' = operator.getitem(h0__fused_rms_norm, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h0_detach: 'float32[32, 2048, 1]' = aten.detach(h0_getitem_1)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.0.attn.c_q (Linear)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h0_attn_c_q__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_0_attn_c_q_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h0_attn_c_q_t: 'bfloat16[512, 512]' = aten.t(h0_attn_c_q__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h0_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(h0_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h0_attn_c_q_mm: 'bfloat16[65536, 512]' = aten.mm(h0_attn_c_q_view, h0_attn_c_q_t)  # strides=(512, 1), contiguous=True, view=False
    h0_attn_c_q__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h0_attn_c_q_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.0.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h0_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(h0_attn_c_q__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.0.attn.c_k (Linear)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h0_attn_c_k__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_0_attn_c_k_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h0_attn_c_k_t: 'bfloat16[512, 512]' = aten.t(h0_attn_c_k__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h0_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(h0_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h0_attn_c_k_mm: 'bfloat16[65536, 512]' = aten.mm(h0_attn_c_k_view, h0_attn_c_k_t)  # strides=(512, 1), contiguous=True, view=False
    h0_attn_c_k__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h0_attn_c_k_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.0.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h0_attn_view_1: 'bfloat16[32, 2048, 4, 128]' = aten.view(h0_attn_c_k__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.0.attn.c_v (Linear)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h0_attn_c_v__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_0_attn_c_v_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h0_attn_c_v_t: 'bfloat16[512, 512]' = aten.t(h0_attn_c_v__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h0_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(h0_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h0_attn_c_v_mm: 'bfloat16[65536, 512]' = aten.mm(h0_attn_c_v_view, h0_attn_c_v_t)  # strides=(512, 1), contiguous=True, view=False
    h0_attn_c_v__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h0_attn_c_v_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.0.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h0_attn_view_2: 'bfloat16[32, 2048, 4, 128]' = aten.view(h0_attn_c_v__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    # FUSED: RoPE for Q (slice+mul+neg+mul+add+cat → single Triton kernel)
    h0_attn_neg: 'bfloat16[1, 2048, 1, 64]' = _neg_sin  # saved for backward
    h0_attn_cat: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h0_attn_view, getitem_slice, getitem_1_slice)
    # FUSED: RoPE for K
    h0_attn_neg_1: 'bfloat16[1, 2048, 1, 64]' = _neg_sin  # saved for backward
    h0_attn_cat_1: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h0_attn_view_1, getitem_slice, getitem_1_slice)
    h0_attn__fused_rms_norm = aten._fused_rms_norm(h0_attn_cat, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h0_attn_getitem: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h0_attn__fused_rms_norm, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h0_attn_getitem_1: 'float32[32, 2048, 4, 1]' = operator.getitem(h0_attn__fused_rms_norm, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h0_attn_detach: 'float32[32, 2048, 4, 1]' = aten.detach(h0_attn_getitem_1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h0_attn__fused_rms_norm_1 = aten._fused_rms_norm(h0_attn_cat_1, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h0_attn_getitem_2: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h0_attn__fused_rms_norm_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h0_attn_getitem_3: 'float32[32, 2048, 4, 1]' = operator.getitem(h0_attn__fused_rms_norm_1, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h0_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(h0_attn_getitem_3)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h0_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h0_attn_getitem, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h0_attn_transpose_1: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h0_attn_getitem_2, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h0_attn_transpose_2: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h0_attn_view_2, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h0_attn_where = _attn_mask  # reuse precomputed mask
    h0_attn__scaled_dot_product_cudnn_attention = aten._scaled_dot_product_cudnn_attention(h0_attn_transpose, h0_attn_transpose_1, h0_attn_transpose_2, h0_attn_where, True)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(8192, 2048, 1, 1), contiguous=True; out6: strides=(), contiguous=True; out7: strides=(), contiguous=True, view=False
    h0_attn_getitem_4: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(h0_attn__scaled_dot_product_cudnn_attention, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h0_attn_getitem_5: 'float32[32, 4, 2048, 1]' = operator.getitem(h0_attn__scaled_dot_product_cudnn_attention, 1)  # strides=(8192, 2048, 1, 1), contiguous=True, view=True
    h0_attn_getitem_6: 'int64[]' = operator.getitem(h0_attn__scaled_dot_product_cudnn_attention, 6)  # strides=(), contiguous=True, view=True
    h0_attn_getitem_7: 'int64[]' = operator.getitem(h0_attn__scaled_dot_product_cudnn_attention, 7)  # strides=(), contiguous=True, view=True
    h0_attn_detach_2: 'bfloat16[32, 4, 2048, 128]' = aten.detach(h0_attn_getitem_4)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h0_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(h0_attn_getitem_4, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h0_attn_view_3: 'bfloat16[32, 2048, 512]' = aten.view(h0_attn_transpose_3, [32, 2048, -1])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # self.transformer.h.0.attn.c_proj (Linear)
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    h0_attn_c_proj__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_0_attn_c_proj_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h0_attn_c_proj_t: 'bfloat16[512, 512]' = aten.t(h0_attn_c_proj__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h0_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(h0_attn_view_3, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h0_attn_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h0_attn_c_proj_view, h0_attn_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h0_attn_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h0_attn_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.0 (Block)
    # /.autoresearch_repo/train.py:114
    # x = x + self.attn(norm(x), ve, cos_sin, window_size)
    h0_add: 'bfloat16[32, 2048, 512]' = cuda_add(add_add, h0_attn_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add
    h0__fused_rms_norm_1 = aten._fused_rms_norm(h0_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h0_getitem_2: 'bfloat16[32, 2048, 512]' = operator.getitem(h0__fused_rms_norm_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h0_getitem_3: 'float32[32, 2048, 1]' = operator.getitem(h0__fused_rms_norm_1, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h0_detach_1: 'float32[32, 2048, 1]' = aten.detach(h0_getitem_3)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.0.mlp.c_fc (Linear)
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    h0_mlp_c_fc__to_copy: 'bfloat16[2048, 512]' = aten._to_copy(transformer_h_0_mlp_c_fc_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h0_mlp_c_fc_t: 'bfloat16[512, 2048]' = aten.t(h0_mlp_c_fc__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h0_mlp_c_fc_view: 'bfloat16[65536, 512]' = aten.view(h0_getitem_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h0_mlp_c_fc_mm: 'bfloat16[65536, 2048]' = aten.mm(h0_mlp_c_fc_view, h0_mlp_c_fc_t)  # strides=(2048, 1), contiguous=True, view=False
    h0_mlp_c_fc__unsafe_view: 'bfloat16[32, 2048, 2048]' = aten._unsafe_view(h0_mlp_c_fc_mm, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.0.mlp (MLP)
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    h0_mlp_relu: 'bfloat16[32, 2048, 2048]' = aten.relu(h0_mlp_c_fc__unsafe_view)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h0_mlp_detach: 'bfloat16[32, 2048, 2048]' = aten.detach(h0_mlp_relu)  # strides=(4194304, 2048, 1), contiguous=True, view=True
    h0_mlp__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(h0_mlp_relu, dtype=torch.float32)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h0_mlp_pow: 'float32[32, 2048, 2048]' = aten.pow.Tensor_Scalar(h0_mlp__to_copy, 2)  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.0.mlp.c_proj (Linear)
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    h0_mlp_c_proj__to_copy: 'bfloat16[512, 2048]' = aten._to_copy(transformer_h_0_mlp_c_proj_weight, dtype=torch.bfloat16)  # strides=(2048, 1), contiguous=True, view=False
    h0_mlp_c_proj__to_copy_1: 'bfloat16[32, 2048, 2048]' = aten._to_copy(h0_mlp_pow, dtype=torch.bfloat16)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h0_mlp_c_proj_t: 'bfloat16[2048, 512]' = aten.t(h0_mlp_c_proj__to_copy)  # strides=(1, 2048), contiguous=False, view=True
    h0_mlp_c_proj_view: 'bfloat16[65536, 2048]' = aten.view(h0_mlp_c_proj__to_copy_1, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    h0_mlp_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h0_mlp_c_proj_view, h0_mlp_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h0_mlp_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h0_mlp_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.0 (Block)
    # /.autoresearch_repo/train.py:115
    # x = x + self.mlp(norm(x))
    h0_add_1: 'bfloat16[32, 2048, 512]' = cuda_add(h0_add, h0_mlp_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    getitem_8_select: 'float32[]' = aten.select.int(resid_lambdas, 0, 1)  # strides=(), contiguous=True, view=True
    getitem_9_select: 'float32[]' = aten.select.int(x0_lambdas, 0, 1)  # strides=(), contiguous=True, view=True
    add_8_add: 'bfloat16[32, 2048, 512]' = triton_lambda_scale(h0_add_1, rms_norm_getitem, getitem_8_select, getitem_9_select)  # FUSED: a*x + b*x0

    # ════════════════════════════════════════════════════════════════
    # self.value_embeds.1 (Embedding)
    # ════════════════════════════════════════════════════════════════
    (value_embeds1_embedding,) = embedding(
        weight=value_embeds_1_weight,
        input__32__2048=input__32__2048,
    )

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.1
    # ════════════════════════════════════════════════════════════════

    # self.transformer.h.1 (Block)
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    h1__fused_rms_norm = aten._fused_rms_norm(add_8_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h1_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(h1__fused_rms_norm, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h1_getitem_1: 'float32[32, 2048, 1]' = operator.getitem(h1__fused_rms_norm, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h1_detach: 'float32[32, 2048, 1]' = aten.detach(h1_getitem_1)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.1.attn.c_q (Linear)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h1_attn_c_q__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_1_attn_c_q_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h1_attn_c_q_t: 'bfloat16[512, 512]' = aten.t(h1_attn_c_q__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h1_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(h1_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h1_attn_c_q_mm: 'bfloat16[65536, 512]' = aten.mm(h1_attn_c_q_view, h1_attn_c_q_t)  # strides=(512, 1), contiguous=True, view=False
    h1_attn_c_q__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h1_attn_c_q_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.1.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h1_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(h1_attn_c_q__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.1.attn.c_k (Linear)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h1_attn_c_k__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_1_attn_c_k_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h1_attn_c_k_t: 'bfloat16[512, 512]' = aten.t(h1_attn_c_k__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h1_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(h1_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h1_attn_c_k_mm: 'bfloat16[65536, 512]' = aten.mm(h1_attn_c_k_view, h1_attn_c_k_t)  # strides=(512, 1), contiguous=True, view=False
    h1_attn_c_k__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h1_attn_c_k_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.1.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h1_attn_view_1: 'bfloat16[32, 2048, 4, 128]' = aten.view(h1_attn_c_k__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.1.attn.c_v (Linear)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h1_attn_c_v__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_1_attn_c_v_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h1_attn_c_v_t: 'bfloat16[512, 512]' = aten.t(h1_attn_c_v__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h1_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(h1_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h1_attn_c_v_mm: 'bfloat16[65536, 512]' = aten.mm(h1_attn_c_v_view, h1_attn_c_v_t)  # strides=(512, 1), contiguous=True, view=False
    h1_attn_c_v__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h1_attn_c_v_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.1.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h1_attn_view_2: 'bfloat16[32, 2048, 4, 128]' = aten.view(h1_attn_c_v__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h1_attn_view_3: 'bfloat16[32, 2048, 4, 128]' = aten.view(value_embeds1_embedding, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h1_attn_slice: 'bfloat16[32, 2048, 32]' = aten.slice.Tensor(h1_getitem, 2, 0, 32)  # strides=(1048576, 512, 1), contiguous=False, view=True

    # self.transformer.h.1.attn.ve_gate (Linear)
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    h1_attn_ve_gate__to_copy: 'bfloat16[4, 32]' = aten._to_copy(transformer_h_1_attn_ve_gate_weight, dtype=torch.bfloat16)  # strides=(32, 1), contiguous=True, view=False
    h1_attn_ve_gate_t: 'bfloat16[32, 4]' = aten.t(h1_attn_ve_gate__to_copy)  # strides=(1, 32), contiguous=False, view=True
    h1_attn_ve_gate_view: 'bfloat16[65536, 32]' = aten.view(h1_attn_slice, [65536, 32])  # strides=(512, 1), contiguous=False, view=True
    h1_attn_ve_gate_mm: 'bfloat16[65536, 4]' = aten.mm(h1_attn_ve_gate_view, h1_attn_ve_gate_t)  # strides=(4, 1), contiguous=True, view=False
    h1_attn_ve_gate__unsafe_view: 'bfloat16[32, 2048, 4]' = aten._unsafe_view(h1_attn_ve_gate_mm, [32, 2048, 4])  # strides=(8192, 4, 1), contiguous=True, view=False

    # self.transformer.h.1.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    h1_attn_sigmoid: 'bfloat16[32, 2048, 4]' = aten.sigmoid(h1_attn_ve_gate__unsafe_view)  # strides=(8192, 4, 1), contiguous=True, view=False
    h1_attn_detach: 'bfloat16[32, 2048, 4]' = aten.detach(h1_attn_sigmoid)  # strides=(8192, 4, 1), contiguous=True, view=True
    h1_attn_mul: 'bfloat16[32, 2048, 4]' = aten.mul.Tensor(h1_attn_sigmoid, 2)  # strides=(8192, 4, 1), contiguous=True, view=False
    h1_attn_unsqueeze: 'bfloat16[32, 2048, 4, 1]' = aten.unsqueeze(h1_attn_mul, -1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h1_attn_mul_1: 'bfloat16[32, 2048, 4, 128]' = aten.mul.Tensor(h1_attn_unsqueeze, h1_attn_view_3)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    h1_attn_add: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(h1_attn_view_2, h1_attn_mul_1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    # FUSED: RoPE for Q
    h1_attn_neg: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h1_attn_cat: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h1_attn_view, getitem_slice, getitem_1_slice)
    # FUSED: RoPE for K
    h1_attn_neg_1: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h1_attn_cat_1: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h1_attn_view_1, getitem_slice, getitem_1_slice)
    h1_attn__fused_rms_norm = aten._fused_rms_norm(h1_attn_cat, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h1_attn_getitem: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h1_attn__fused_rms_norm, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h1_attn_getitem_1: 'float32[32, 2048, 4, 1]' = operator.getitem(h1_attn__fused_rms_norm, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h1_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(h1_attn_getitem_1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h1_attn__fused_rms_norm_1 = aten._fused_rms_norm(h1_attn_cat_1, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h1_attn_getitem_2: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h1_attn__fused_rms_norm_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h1_attn_getitem_3: 'float32[32, 2048, 4, 1]' = operator.getitem(h1_attn__fused_rms_norm_1, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h1_attn_detach_2: 'float32[32, 2048, 4, 1]' = aten.detach(h1_attn_getitem_3)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h1_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h1_attn_getitem, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h1_attn_transpose_1: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h1_attn_getitem_2, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h1_attn_transpose_2: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h1_attn_add, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h1_attn_where = _attn_mask  # reuse precomputed mask
    h1_attn__scaled_dot_product_cudnn_attention = aten._scaled_dot_product_cudnn_attention(h1_attn_transpose, h1_attn_transpose_1, h1_attn_transpose_2, h1_attn_where, True)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(8192, 2048, 1, 1), contiguous=True; out6: strides=(), contiguous=True; out7: strides=(), contiguous=True, view=False
    h1_attn_getitem_4: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(h1_attn__scaled_dot_product_cudnn_attention, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h1_attn_getitem_5: 'float32[32, 4, 2048, 1]' = operator.getitem(h1_attn__scaled_dot_product_cudnn_attention, 1)  # strides=(8192, 2048, 1, 1), contiguous=True, view=True
    h1_attn_getitem_6: 'int64[]' = operator.getitem(h1_attn__scaled_dot_product_cudnn_attention, 6)  # strides=(), contiguous=True, view=True
    h1_attn_getitem_7: 'int64[]' = operator.getitem(h1_attn__scaled_dot_product_cudnn_attention, 7)  # strides=(), contiguous=True, view=True
    h1_attn_detach_3: 'bfloat16[32, 4, 2048, 128]' = aten.detach(h1_attn_getitem_4)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h1_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(h1_attn_getitem_4, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h1_attn_view_4: 'bfloat16[32, 2048, 512]' = aten.view(h1_attn_transpose_3, [32, 2048, -1])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # self.transformer.h.1.attn.c_proj (Linear)
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    h1_attn_c_proj__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_1_attn_c_proj_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h1_attn_c_proj_t: 'bfloat16[512, 512]' = aten.t(h1_attn_c_proj__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h1_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(h1_attn_view_4, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h1_attn_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h1_attn_c_proj_view, h1_attn_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h1_attn_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h1_attn_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.1 (Block)
    # /.autoresearch_repo/train.py:114
    # x = x + self.attn(norm(x), ve, cos_sin, window_size)
    h1_add: 'bfloat16[32, 2048, 512]' = cuda_add(add_8_add, h1_attn_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add
    h1__fused_rms_norm_1 = aten._fused_rms_norm(h1_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h1_getitem_2: 'bfloat16[32, 2048, 512]' = operator.getitem(h1__fused_rms_norm_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h1_getitem_3: 'float32[32, 2048, 1]' = operator.getitem(h1__fused_rms_norm_1, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h1_detach_1: 'float32[32, 2048, 1]' = aten.detach(h1_getitem_3)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.1.mlp.c_fc (Linear)
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    h1_mlp_c_fc__to_copy: 'bfloat16[2048, 512]' = aten._to_copy(transformer_h_1_mlp_c_fc_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h1_mlp_c_fc_t: 'bfloat16[512, 2048]' = aten.t(h1_mlp_c_fc__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h1_mlp_c_fc_view: 'bfloat16[65536, 512]' = aten.view(h1_getitem_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h1_mlp_c_fc_mm: 'bfloat16[65536, 2048]' = aten.mm(h1_mlp_c_fc_view, h1_mlp_c_fc_t)  # strides=(2048, 1), contiguous=True, view=False
    h1_mlp_c_fc__unsafe_view: 'bfloat16[32, 2048, 2048]' = aten._unsafe_view(h1_mlp_c_fc_mm, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.1.mlp (MLP)
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    h1_mlp_relu: 'bfloat16[32, 2048, 2048]' = aten.relu(h1_mlp_c_fc__unsafe_view)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h1_mlp_detach: 'bfloat16[32, 2048, 2048]' = aten.detach(h1_mlp_relu)  # strides=(4194304, 2048, 1), contiguous=True, view=True
    h1_mlp__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(h1_mlp_relu, dtype=torch.float32)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h1_mlp_pow: 'float32[32, 2048, 2048]' = aten.pow.Tensor_Scalar(h1_mlp__to_copy, 2)  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.1.mlp.c_proj (Linear)
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    h1_mlp_c_proj__to_copy: 'bfloat16[512, 2048]' = aten._to_copy(transformer_h_1_mlp_c_proj_weight, dtype=torch.bfloat16)  # strides=(2048, 1), contiguous=True, view=False
    h1_mlp_c_proj__to_copy_1: 'bfloat16[32, 2048, 2048]' = aten._to_copy(h1_mlp_pow, dtype=torch.bfloat16)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h1_mlp_c_proj_t: 'bfloat16[2048, 512]' = aten.t(h1_mlp_c_proj__to_copy)  # strides=(1, 2048), contiguous=False, view=True
    h1_mlp_c_proj_view: 'bfloat16[65536, 2048]' = aten.view(h1_mlp_c_proj__to_copy_1, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    h1_mlp_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h1_mlp_c_proj_view, h1_mlp_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h1_mlp_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h1_mlp_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.1 (Block)
    # /.autoresearch_repo/train.py:115
    # x = x + self.mlp(norm(x))
    h1_add_1: 'bfloat16[32, 2048, 512]' = cuda_add(h1_add, h1_mlp_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    getitem_15_select: 'float32[]' = aten.select.int(resid_lambdas, 0, 2)  # strides=(), contiguous=True, view=True
    getitem_16_select: 'float32[]' = aten.select.int(x0_lambdas, 0, 2)  # strides=(), contiguous=True, view=True
    add_17_add: 'bfloat16[32, 2048, 512]' = triton_lambda_scale(h1_add_1, rms_norm_getitem, getitem_15_select, getitem_16_select)  # FUSED: a*x + b*x0

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.2
    # ════════════════════════════════════════════════════════════════

    # self.transformer.h.2 (Block)
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    h2__fused_rms_norm = aten._fused_rms_norm(add_17_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h2_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(h2__fused_rms_norm, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h2_getitem_1: 'float32[32, 2048, 1]' = operator.getitem(h2__fused_rms_norm, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h2_detach: 'float32[32, 2048, 1]' = aten.detach(h2_getitem_1)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.2.attn.c_q (Linear)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h2_attn_c_q__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_2_attn_c_q_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h2_attn_c_q_t: 'bfloat16[512, 512]' = aten.t(h2_attn_c_q__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h2_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(h2_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h2_attn_c_q_mm: 'bfloat16[65536, 512]' = aten.mm(h2_attn_c_q_view, h2_attn_c_q_t)  # strides=(512, 1), contiguous=True, view=False
    h2_attn_c_q__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h2_attn_c_q_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.2.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h2_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(h2_attn_c_q__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.2.attn.c_k (Linear)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h2_attn_c_k__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_2_attn_c_k_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h2_attn_c_k_t: 'bfloat16[512, 512]' = aten.t(h2_attn_c_k__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h2_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(h2_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h2_attn_c_k_mm: 'bfloat16[65536, 512]' = aten.mm(h2_attn_c_k_view, h2_attn_c_k_t)  # strides=(512, 1), contiguous=True, view=False
    h2_attn_c_k__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h2_attn_c_k_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.2.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h2_attn_view_1: 'bfloat16[32, 2048, 4, 128]' = aten.view(h2_attn_c_k__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.2.attn.c_v (Linear)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h2_attn_c_v__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_2_attn_c_v_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h2_attn_c_v_t: 'bfloat16[512, 512]' = aten.t(h2_attn_c_v__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h2_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(h2_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h2_attn_c_v_mm: 'bfloat16[65536, 512]' = aten.mm(h2_attn_c_v_view, h2_attn_c_v_t)  # strides=(512, 1), contiguous=True, view=False
    h2_attn_c_v__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h2_attn_c_v_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.2.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h2_attn_view_2: 'bfloat16[32, 2048, 4, 128]' = aten.view(h2_attn_c_v__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    # FUSED: RoPE for Q
    h2_attn_neg: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h2_attn_cat: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h2_attn_view, getitem_slice, getitem_1_slice)
    # FUSED: RoPE for K
    h2_attn_neg_1: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h2_attn_cat_1: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h2_attn_view_1, getitem_slice, getitem_1_slice)
    h2_attn__fused_rms_norm = aten._fused_rms_norm(h2_attn_cat, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h2_attn_getitem: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h2_attn__fused_rms_norm, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h2_attn_getitem_1: 'float32[32, 2048, 4, 1]' = operator.getitem(h2_attn__fused_rms_norm, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h2_attn_detach: 'float32[32, 2048, 4, 1]' = aten.detach(h2_attn_getitem_1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h2_attn__fused_rms_norm_1 = aten._fused_rms_norm(h2_attn_cat_1, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h2_attn_getitem_2: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h2_attn__fused_rms_norm_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h2_attn_getitem_3: 'float32[32, 2048, 4, 1]' = operator.getitem(h2_attn__fused_rms_norm_1, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h2_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(h2_attn_getitem_3)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h2_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h2_attn_getitem, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h2_attn_transpose_1: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h2_attn_getitem_2, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h2_attn_transpose_2: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h2_attn_view_2, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h2_attn_where = _attn_mask  # reuse precomputed mask
    h2_attn__scaled_dot_product_cudnn_attention = aten._scaled_dot_product_cudnn_attention(h2_attn_transpose, h2_attn_transpose_1, h2_attn_transpose_2, h2_attn_where, True)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(8192, 2048, 1, 1), contiguous=True; out6: strides=(), contiguous=True; out7: strides=(), contiguous=True, view=False
    h2_attn_getitem_4: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(h2_attn__scaled_dot_product_cudnn_attention, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h2_attn_getitem_5: 'float32[32, 4, 2048, 1]' = operator.getitem(h2_attn__scaled_dot_product_cudnn_attention, 1)  # strides=(8192, 2048, 1, 1), contiguous=True, view=True
    h2_attn_getitem_6: 'int64[]' = operator.getitem(h2_attn__scaled_dot_product_cudnn_attention, 6)  # strides=(), contiguous=True, view=True
    h2_attn_getitem_7: 'int64[]' = operator.getitem(h2_attn__scaled_dot_product_cudnn_attention, 7)  # strides=(), contiguous=True, view=True
    h2_attn_detach_2: 'bfloat16[32, 4, 2048, 128]' = aten.detach(h2_attn_getitem_4)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h2_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(h2_attn_getitem_4, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h2_attn_view_3: 'bfloat16[32, 2048, 512]' = aten.view(h2_attn_transpose_3, [32, 2048, -1])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # self.transformer.h.2.attn.c_proj (Linear)
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    h2_attn_c_proj__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_2_attn_c_proj_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h2_attn_c_proj_t: 'bfloat16[512, 512]' = aten.t(h2_attn_c_proj__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h2_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(h2_attn_view_3, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h2_attn_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h2_attn_c_proj_view, h2_attn_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h2_attn_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h2_attn_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.2 (Block)
    # /.autoresearch_repo/train.py:114
    # x = x + self.attn(norm(x), ve, cos_sin, window_size)
    h2_add: 'bfloat16[32, 2048, 512]' = cuda_add(add_17_add, h2_attn_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add
    h2__fused_rms_norm_1 = aten._fused_rms_norm(h2_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h2_getitem_2: 'bfloat16[32, 2048, 512]' = operator.getitem(h2__fused_rms_norm_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h2_getitem_3: 'float32[32, 2048, 1]' = operator.getitem(h2__fused_rms_norm_1, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h2_detach_1: 'float32[32, 2048, 1]' = aten.detach(h2_getitem_3)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.2.mlp.c_fc (Linear)
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    h2_mlp_c_fc__to_copy: 'bfloat16[2048, 512]' = aten._to_copy(transformer_h_2_mlp_c_fc_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h2_mlp_c_fc_t: 'bfloat16[512, 2048]' = aten.t(h2_mlp_c_fc__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h2_mlp_c_fc_view: 'bfloat16[65536, 512]' = aten.view(h2_getitem_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h2_mlp_c_fc_mm: 'bfloat16[65536, 2048]' = aten.mm(h2_mlp_c_fc_view, h2_mlp_c_fc_t)  # strides=(2048, 1), contiguous=True, view=False
    h2_mlp_c_fc__unsafe_view: 'bfloat16[32, 2048, 2048]' = aten._unsafe_view(h2_mlp_c_fc_mm, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.2.mlp (MLP)
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    h2_mlp_relu: 'bfloat16[32, 2048, 2048]' = aten.relu(h2_mlp_c_fc__unsafe_view)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h2_mlp_detach: 'bfloat16[32, 2048, 2048]' = aten.detach(h2_mlp_relu)  # strides=(4194304, 2048, 1), contiguous=True, view=True
    h2_mlp__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(h2_mlp_relu, dtype=torch.float32)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h2_mlp_pow: 'float32[32, 2048, 2048]' = aten.pow.Tensor_Scalar(h2_mlp__to_copy, 2)  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.2.mlp.c_proj (Linear)
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    h2_mlp_c_proj__to_copy: 'bfloat16[512, 2048]' = aten._to_copy(transformer_h_2_mlp_c_proj_weight, dtype=torch.bfloat16)  # strides=(2048, 1), contiguous=True, view=False
    h2_mlp_c_proj__to_copy_1: 'bfloat16[32, 2048, 2048]' = aten._to_copy(h2_mlp_pow, dtype=torch.bfloat16)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h2_mlp_c_proj_t: 'bfloat16[2048, 512]' = aten.t(h2_mlp_c_proj__to_copy)  # strides=(1, 2048), contiguous=False, view=True
    h2_mlp_c_proj_view: 'bfloat16[65536, 2048]' = aten.view(h2_mlp_c_proj__to_copy_1, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    h2_mlp_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h2_mlp_c_proj_view, h2_mlp_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h2_mlp_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h2_mlp_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.2 (Block)
    # /.autoresearch_repo/train.py:115
    # x = x + self.mlp(norm(x))
    h2_add_1: 'bfloat16[32, 2048, 512]' = cuda_add(h2_add, h2_mlp_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    getitem_21_select: 'float32[]' = aten.select.int(resid_lambdas, 0, 3)  # strides=(), contiguous=True, view=True
    getitem_22_select: 'float32[]' = aten.select.int(x0_lambdas, 0, 3)  # strides=(), contiguous=True, view=True
    add_25_add: 'bfloat16[32, 2048, 512]' = triton_lambda_scale(h2_add_1, rms_norm_getitem, getitem_21_select, getitem_22_select)  # FUSED: a*x + b*x0

    # ════════════════════════════════════════════════════════════════
    # self.value_embeds.3 (Embedding)
    # ════════════════════════════════════════════════════════════════
    (value_embeds3_embedding,) = embedding(
        weight=value_embeds_3_weight,
        input__32__2048=input__32__2048,
    )

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.3
    # ════════════════════════════════════════════════════════════════

    # self.transformer.h.3 (Block)
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    h3__fused_rms_norm = aten._fused_rms_norm(add_25_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h3_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(h3__fused_rms_norm, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h3_getitem_1: 'float32[32, 2048, 1]' = operator.getitem(h3__fused_rms_norm, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h3_detach: 'float32[32, 2048, 1]' = aten.detach(h3_getitem_1)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.3.attn.c_q (Linear)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h3_attn_c_q__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_3_attn_c_q_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h3_attn_c_q_t: 'bfloat16[512, 512]' = aten.t(h3_attn_c_q__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h3_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(h3_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h3_attn_c_q_mm: 'bfloat16[65536, 512]' = aten.mm(h3_attn_c_q_view, h3_attn_c_q_t)  # strides=(512, 1), contiguous=True, view=False
    h3_attn_c_q__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h3_attn_c_q_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.3.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h3_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(h3_attn_c_q__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.3.attn.c_k (Linear)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h3_attn_c_k__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_3_attn_c_k_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h3_attn_c_k_t: 'bfloat16[512, 512]' = aten.t(h3_attn_c_k__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h3_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(h3_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h3_attn_c_k_mm: 'bfloat16[65536, 512]' = aten.mm(h3_attn_c_k_view, h3_attn_c_k_t)  # strides=(512, 1), contiguous=True, view=False
    h3_attn_c_k__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h3_attn_c_k_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.3.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h3_attn_view_1: 'bfloat16[32, 2048, 4, 128]' = aten.view(h3_attn_c_k__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.3.attn.c_v (Linear)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h3_attn_c_v__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_3_attn_c_v_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h3_attn_c_v_t: 'bfloat16[512, 512]' = aten.t(h3_attn_c_v__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h3_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(h3_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h3_attn_c_v_mm: 'bfloat16[65536, 512]' = aten.mm(h3_attn_c_v_view, h3_attn_c_v_t)  # strides=(512, 1), contiguous=True, view=False
    h3_attn_c_v__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h3_attn_c_v_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.3.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h3_attn_view_2: 'bfloat16[32, 2048, 4, 128]' = aten.view(h3_attn_c_v__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h3_attn_view_3: 'bfloat16[32, 2048, 4, 128]' = aten.view(value_embeds3_embedding, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h3_attn_slice: 'bfloat16[32, 2048, 32]' = aten.slice.Tensor(h3_getitem, 2, 0, 32)  # strides=(1048576, 512, 1), contiguous=False, view=True

    # self.transformer.h.3.attn.ve_gate (Linear)
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    h3_attn_ve_gate__to_copy: 'bfloat16[4, 32]' = aten._to_copy(transformer_h_3_attn_ve_gate_weight, dtype=torch.bfloat16)  # strides=(32, 1), contiguous=True, view=False
    h3_attn_ve_gate_t: 'bfloat16[32, 4]' = aten.t(h3_attn_ve_gate__to_copy)  # strides=(1, 32), contiguous=False, view=True
    h3_attn_ve_gate_view: 'bfloat16[65536, 32]' = aten.view(h3_attn_slice, [65536, 32])  # strides=(512, 1), contiguous=False, view=True
    h3_attn_ve_gate_mm: 'bfloat16[65536, 4]' = aten.mm(h3_attn_ve_gate_view, h3_attn_ve_gate_t)  # strides=(4, 1), contiguous=True, view=False
    h3_attn_ve_gate__unsafe_view: 'bfloat16[32, 2048, 4]' = aten._unsafe_view(h3_attn_ve_gate_mm, [32, 2048, 4])  # strides=(8192, 4, 1), contiguous=True, view=False

    # self.transformer.h.3.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    h3_attn_sigmoid: 'bfloat16[32, 2048, 4]' = aten.sigmoid(h3_attn_ve_gate__unsafe_view)  # strides=(8192, 4, 1), contiguous=True, view=False
    h3_attn_detach: 'bfloat16[32, 2048, 4]' = aten.detach(h3_attn_sigmoid)  # strides=(8192, 4, 1), contiguous=True, view=True
    h3_attn_mul: 'bfloat16[32, 2048, 4]' = aten.mul.Tensor(h3_attn_sigmoid, 2)  # strides=(8192, 4, 1), contiguous=True, view=False
    h3_attn_unsqueeze: 'bfloat16[32, 2048, 4, 1]' = aten.unsqueeze(h3_attn_mul, -1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h3_attn_mul_1: 'bfloat16[32, 2048, 4, 128]' = aten.mul.Tensor(h3_attn_unsqueeze, h3_attn_view_3)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    h3_attn_add: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(h3_attn_view_2, h3_attn_mul_1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    # FUSED: RoPE for Q
    h3_attn_neg: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h3_attn_cat: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h3_attn_view, getitem_slice, getitem_1_slice)
    # FUSED: RoPE for K
    h3_attn_neg_1: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h3_attn_cat_1: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h3_attn_view_1, getitem_slice, getitem_1_slice)
    h3_attn__fused_rms_norm = aten._fused_rms_norm(h3_attn_cat, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h3_attn_getitem: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h3_attn__fused_rms_norm, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h3_attn_getitem_1: 'float32[32, 2048, 4, 1]' = operator.getitem(h3_attn__fused_rms_norm, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h3_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(h3_attn_getitem_1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h3_attn__fused_rms_norm_1 = aten._fused_rms_norm(h3_attn_cat_1, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h3_attn_getitem_2: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h3_attn__fused_rms_norm_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h3_attn_getitem_3: 'float32[32, 2048, 4, 1]' = operator.getitem(h3_attn__fused_rms_norm_1, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h3_attn_detach_2: 'float32[32, 2048, 4, 1]' = aten.detach(h3_attn_getitem_3)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h3_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h3_attn_getitem, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h3_attn_transpose_1: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h3_attn_getitem_2, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h3_attn_transpose_2: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h3_attn_add, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h3_attn__scaled_dot_product_cudnn_attention = aten._scaled_dot_product_cudnn_attention(h3_attn_transpose, h3_attn_transpose_1, h3_attn_transpose_2, None, True, 0.0, True)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(8192, 2048, 1, 1), contiguous=True; out6: strides=(), contiguous=True; out7: strides=(), contiguous=True, view=False
    h3_attn_getitem_4: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(h3_attn__scaled_dot_product_cudnn_attention, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h3_attn_getitem_5: 'float32[32, 4, 2048, 1]' = operator.getitem(h3_attn__scaled_dot_product_cudnn_attention, 1)  # strides=(8192, 2048, 1, 1), contiguous=True, view=True
    h3_attn_getitem_6: 'int64[]' = operator.getitem(h3_attn__scaled_dot_product_cudnn_attention, 6)  # strides=(), contiguous=True, view=True
    h3_attn_getitem_7: 'int64[]' = operator.getitem(h3_attn__scaled_dot_product_cudnn_attention, 7)  # strides=(), contiguous=True, view=True
    h3_attn_detach_3: 'bfloat16[32, 4, 2048, 128]' = aten.detach(h3_attn_getitem_4)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h3_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(h3_attn_getitem_4, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h3_attn_view_4: 'bfloat16[32, 2048, 512]' = aten.view(h3_attn_transpose_3, [32, 2048, -1])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # self.transformer.h.3.attn.c_proj (Linear)
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    h3_attn_c_proj__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_3_attn_c_proj_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h3_attn_c_proj_t: 'bfloat16[512, 512]' = aten.t(h3_attn_c_proj__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h3_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(h3_attn_view_4, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h3_attn_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h3_attn_c_proj_view, h3_attn_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h3_attn_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h3_attn_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.3 (Block)
    # /.autoresearch_repo/train.py:114
    # x = x + self.attn(norm(x), ve, cos_sin, window_size)
    h3_add: 'bfloat16[32, 2048, 512]' = cuda_add(add_25_add, h3_attn_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add
    h3__fused_rms_norm_1 = aten._fused_rms_norm(h3_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h3_getitem_2: 'bfloat16[32, 2048, 512]' = operator.getitem(h3__fused_rms_norm_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h3_getitem_3: 'float32[32, 2048, 1]' = operator.getitem(h3__fused_rms_norm_1, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h3_detach_1: 'float32[32, 2048, 1]' = aten.detach(h3_getitem_3)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.3.mlp.c_fc (Linear)
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    h3_mlp_c_fc__to_copy: 'bfloat16[2048, 512]' = aten._to_copy(transformer_h_3_mlp_c_fc_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h3_mlp_c_fc_t: 'bfloat16[512, 2048]' = aten.t(h3_mlp_c_fc__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h3_mlp_c_fc_view: 'bfloat16[65536, 512]' = aten.view(h3_getitem_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h3_mlp_c_fc_mm: 'bfloat16[65536, 2048]' = aten.mm(h3_mlp_c_fc_view, h3_mlp_c_fc_t)  # strides=(2048, 1), contiguous=True, view=False
    h3_mlp_c_fc__unsafe_view: 'bfloat16[32, 2048, 2048]' = aten._unsafe_view(h3_mlp_c_fc_mm, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.3.mlp (MLP)
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    h3_mlp_relu: 'bfloat16[32, 2048, 2048]' = aten.relu(h3_mlp_c_fc__unsafe_view)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h3_mlp_detach: 'bfloat16[32, 2048, 2048]' = aten.detach(h3_mlp_relu)  # strides=(4194304, 2048, 1), contiguous=True, view=True
    h3_mlp__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(h3_mlp_relu, dtype=torch.float32)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h3_mlp_pow: 'float32[32, 2048, 2048]' = aten.pow.Tensor_Scalar(h3_mlp__to_copy, 2)  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.3.mlp.c_proj (Linear)
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    h3_mlp_c_proj__to_copy: 'bfloat16[512, 2048]' = aten._to_copy(transformer_h_3_mlp_c_proj_weight, dtype=torch.bfloat16)  # strides=(2048, 1), contiguous=True, view=False
    h3_mlp_c_proj__to_copy_1: 'bfloat16[32, 2048, 2048]' = aten._to_copy(h3_mlp_pow, dtype=torch.bfloat16)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h3_mlp_c_proj_t: 'bfloat16[2048, 512]' = aten.t(h3_mlp_c_proj__to_copy)  # strides=(1, 2048), contiguous=False, view=True
    h3_mlp_c_proj_view: 'bfloat16[65536, 2048]' = aten.view(h3_mlp_c_proj__to_copy_1, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    h3_mlp_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h3_mlp_c_proj_view, h3_mlp_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h3_mlp_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h3_mlp_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.3 (Block)
    # /.autoresearch_repo/train.py:115
    # x = x + self.mlp(norm(x))
    h3_add_1: 'bfloat16[32, 2048, 512]' = cuda_add(h3_add, h3_mlp_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    getitem_28_select: 'float32[]' = aten.select.int(resid_lambdas, 0, 4)  # strides=(), contiguous=True, view=True
    getitem_29_select: 'float32[]' = aten.select.int(x0_lambdas, 0, 4)  # strides=(), contiguous=True, view=True
    add_33_add: 'bfloat16[32, 2048, 512]' = triton_lambda_scale(h3_add_1, rms_norm_getitem, getitem_28_select, getitem_29_select)  # FUSED: a*x + b*x0

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.4
    # ════════════════════════════════════════════════════════════════

    # self.transformer.h.4 (Block)
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    h4__fused_rms_norm = aten._fused_rms_norm(add_33_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h4_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(h4__fused_rms_norm, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h4_getitem_1: 'float32[32, 2048, 1]' = operator.getitem(h4__fused_rms_norm, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h4_detach: 'float32[32, 2048, 1]' = aten.detach(h4_getitem_1)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.4.attn.c_q (Linear)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h4_attn_c_q__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_4_attn_c_q_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h4_attn_c_q_t: 'bfloat16[512, 512]' = aten.t(h4_attn_c_q__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h4_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(h4_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h4_attn_c_q_mm: 'bfloat16[65536, 512]' = aten.mm(h4_attn_c_q_view, h4_attn_c_q_t)  # strides=(512, 1), contiguous=True, view=False
    h4_attn_c_q__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h4_attn_c_q_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.4.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h4_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(h4_attn_c_q__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.4.attn.c_k (Linear)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h4_attn_c_k__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_4_attn_c_k_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h4_attn_c_k_t: 'bfloat16[512, 512]' = aten.t(h4_attn_c_k__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h4_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(h4_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h4_attn_c_k_mm: 'bfloat16[65536, 512]' = aten.mm(h4_attn_c_k_view, h4_attn_c_k_t)  # strides=(512, 1), contiguous=True, view=False
    h4_attn_c_k__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h4_attn_c_k_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.4.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h4_attn_view_1: 'bfloat16[32, 2048, 4, 128]' = aten.view(h4_attn_c_k__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.4.attn.c_v (Linear)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h4_attn_c_v__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_4_attn_c_v_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h4_attn_c_v_t: 'bfloat16[512, 512]' = aten.t(h4_attn_c_v__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h4_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(h4_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h4_attn_c_v_mm: 'bfloat16[65536, 512]' = aten.mm(h4_attn_c_v_view, h4_attn_c_v_t)  # strides=(512, 1), contiguous=True, view=False
    h4_attn_c_v__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h4_attn_c_v_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.4.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h4_attn_view_2: 'bfloat16[32, 2048, 4, 128]' = aten.view(h4_attn_c_v__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    # FUSED: RoPE for Q
    h4_attn_neg: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h4_attn_cat: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h4_attn_view, getitem_slice, getitem_1_slice)
    # FUSED: RoPE for K
    h4_attn_neg_1: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h4_attn_cat_1: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h4_attn_view_1, getitem_slice, getitem_1_slice)
    h4_attn__fused_rms_norm = aten._fused_rms_norm(h4_attn_cat, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h4_attn_getitem: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h4_attn__fused_rms_norm, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h4_attn_getitem_1: 'float32[32, 2048, 4, 1]' = operator.getitem(h4_attn__fused_rms_norm, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h4_attn_detach: 'float32[32, 2048, 4, 1]' = aten.detach(h4_attn_getitem_1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h4_attn__fused_rms_norm_1 = aten._fused_rms_norm(h4_attn_cat_1, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h4_attn_getitem_2: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h4_attn__fused_rms_norm_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h4_attn_getitem_3: 'float32[32, 2048, 4, 1]' = operator.getitem(h4_attn__fused_rms_norm_1, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h4_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(h4_attn_getitem_3)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h4_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h4_attn_getitem, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h4_attn_transpose_1: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h4_attn_getitem_2, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h4_attn_transpose_2: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h4_attn_view_2, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h4_attn_where = _attn_mask  # reuse precomputed mask
    h4_attn__scaled_dot_product_cudnn_attention = aten._scaled_dot_product_cudnn_attention(h4_attn_transpose, h4_attn_transpose_1, h4_attn_transpose_2, h4_attn_where, True)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(8192, 2048, 1, 1), contiguous=True; out6: strides=(), contiguous=True; out7: strides=(), contiguous=True, view=False
    h4_attn_getitem_4: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(h4_attn__scaled_dot_product_cudnn_attention, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h4_attn_getitem_5: 'float32[32, 4, 2048, 1]' = operator.getitem(h4_attn__scaled_dot_product_cudnn_attention, 1)  # strides=(8192, 2048, 1, 1), contiguous=True, view=True
    h4_attn_getitem_6: 'int64[]' = operator.getitem(h4_attn__scaled_dot_product_cudnn_attention, 6)  # strides=(), contiguous=True, view=True
    h4_attn_getitem_7: 'int64[]' = operator.getitem(h4_attn__scaled_dot_product_cudnn_attention, 7)  # strides=(), contiguous=True, view=True
    h4_attn_detach_2: 'bfloat16[32, 4, 2048, 128]' = aten.detach(h4_attn_getitem_4)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h4_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(h4_attn_getitem_4, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h4_attn_view_3: 'bfloat16[32, 2048, 512]' = aten.view(h4_attn_transpose_3, [32, 2048, -1])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # self.transformer.h.4.attn.c_proj (Linear)
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    h4_attn_c_proj__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_4_attn_c_proj_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h4_attn_c_proj_t: 'bfloat16[512, 512]' = aten.t(h4_attn_c_proj__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h4_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(h4_attn_view_3, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h4_attn_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h4_attn_c_proj_view, h4_attn_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h4_attn_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h4_attn_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.4 (Block)
    # /.autoresearch_repo/train.py:114
    # x = x + self.attn(norm(x), ve, cos_sin, window_size)
    h4_add: 'bfloat16[32, 2048, 512]' = cuda_add(add_33_add, h4_attn_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add
    h4__fused_rms_norm_1 = aten._fused_rms_norm(h4_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h4_getitem_2: 'bfloat16[32, 2048, 512]' = operator.getitem(h4__fused_rms_norm_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h4_getitem_3: 'float32[32, 2048, 1]' = operator.getitem(h4__fused_rms_norm_1, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h4_detach_1: 'float32[32, 2048, 1]' = aten.detach(h4_getitem_3)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.4.mlp.c_fc (Linear)
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    h4_mlp_c_fc__to_copy: 'bfloat16[2048, 512]' = aten._to_copy(transformer_h_4_mlp_c_fc_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h4_mlp_c_fc_t: 'bfloat16[512, 2048]' = aten.t(h4_mlp_c_fc__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h4_mlp_c_fc_view: 'bfloat16[65536, 512]' = aten.view(h4_getitem_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h4_mlp_c_fc_mm: 'bfloat16[65536, 2048]' = aten.mm(h4_mlp_c_fc_view, h4_mlp_c_fc_t)  # strides=(2048, 1), contiguous=True, view=False
    h4_mlp_c_fc__unsafe_view: 'bfloat16[32, 2048, 2048]' = aten._unsafe_view(h4_mlp_c_fc_mm, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.4.mlp (MLP)
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    h4_mlp_relu: 'bfloat16[32, 2048, 2048]' = aten.relu(h4_mlp_c_fc__unsafe_view)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h4_mlp_detach: 'bfloat16[32, 2048, 2048]' = aten.detach(h4_mlp_relu)  # strides=(4194304, 2048, 1), contiguous=True, view=True
    h4_mlp__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(h4_mlp_relu, dtype=torch.float32)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h4_mlp_pow: 'float32[32, 2048, 2048]' = aten.pow.Tensor_Scalar(h4_mlp__to_copy, 2)  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.4.mlp.c_proj (Linear)
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    h4_mlp_c_proj__to_copy: 'bfloat16[512, 2048]' = aten._to_copy(transformer_h_4_mlp_c_proj_weight, dtype=torch.bfloat16)  # strides=(2048, 1), contiguous=True, view=False
    h4_mlp_c_proj__to_copy_1: 'bfloat16[32, 2048, 2048]' = aten._to_copy(h4_mlp_pow, dtype=torch.bfloat16)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h4_mlp_c_proj_t: 'bfloat16[2048, 512]' = aten.t(h4_mlp_c_proj__to_copy)  # strides=(1, 2048), contiguous=False, view=True
    h4_mlp_c_proj_view: 'bfloat16[65536, 2048]' = aten.view(h4_mlp_c_proj__to_copy_1, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    h4_mlp_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h4_mlp_c_proj_view, h4_mlp_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h4_mlp_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h4_mlp_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.4 (Block)
    # /.autoresearch_repo/train.py:115
    # x = x + self.mlp(norm(x))
    h4_add_1: 'bfloat16[32, 2048, 512]' = cuda_add(h4_add, h4_mlp_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    getitem_34_select: 'float32[]' = aten.select.int(resid_lambdas, 0, 5)  # strides=(), contiguous=True, view=True
    getitem_35_select: 'float32[]' = aten.select.int(x0_lambdas, 0, 5)  # strides=(), contiguous=True, view=True
    add_41_add: 'bfloat16[32, 2048, 512]' = triton_lambda_scale(h4_add_1, rms_norm_getitem, getitem_34_select, getitem_35_select)  # FUSED: a*x + b*x0

    # ════════════════════════════════════════════════════════════════
    # self.value_embeds.5 (Embedding)
    # ════════════════════════════════════════════════════════════════
    (value_embeds5_embedding,) = embedding(
        weight=value_embeds_5_weight,
        input__32__2048=input__32__2048,
    )

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.5
    # ════════════════════════════════════════════════════════════════

    # self.transformer.h.5 (Block)
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    h5__fused_rms_norm = aten._fused_rms_norm(add_41_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h5_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(h5__fused_rms_norm, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h5_getitem_1: 'float32[32, 2048, 1]' = operator.getitem(h5__fused_rms_norm, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h5_detach: 'float32[32, 2048, 1]' = aten.detach(h5_getitem_1)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.5.attn.c_q (Linear)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h5_attn_c_q__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_5_attn_c_q_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h5_attn_c_q_t: 'bfloat16[512, 512]' = aten.t(h5_attn_c_q__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h5_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(h5_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h5_attn_c_q_mm: 'bfloat16[65536, 512]' = aten.mm(h5_attn_c_q_view, h5_attn_c_q_t)  # strides=(512, 1), contiguous=True, view=False
    h5_attn_c_q__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h5_attn_c_q_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.5.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h5_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(h5_attn_c_q__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.5.attn.c_k (Linear)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h5_attn_c_k__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_5_attn_c_k_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h5_attn_c_k_t: 'bfloat16[512, 512]' = aten.t(h5_attn_c_k__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h5_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(h5_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h5_attn_c_k_mm: 'bfloat16[65536, 512]' = aten.mm(h5_attn_c_k_view, h5_attn_c_k_t)  # strides=(512, 1), contiguous=True, view=False
    h5_attn_c_k__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h5_attn_c_k_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.5.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h5_attn_view_1: 'bfloat16[32, 2048, 4, 128]' = aten.view(h5_attn_c_k__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.5.attn.c_v (Linear)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h5_attn_c_v__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_5_attn_c_v_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h5_attn_c_v_t: 'bfloat16[512, 512]' = aten.t(h5_attn_c_v__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h5_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(h5_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h5_attn_c_v_mm: 'bfloat16[65536, 512]' = aten.mm(h5_attn_c_v_view, h5_attn_c_v_t)  # strides=(512, 1), contiguous=True, view=False
    h5_attn_c_v__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h5_attn_c_v_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.5.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h5_attn_view_2: 'bfloat16[32, 2048, 4, 128]' = aten.view(h5_attn_c_v__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h5_attn_view_3: 'bfloat16[32, 2048, 4, 128]' = aten.view(value_embeds5_embedding, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h5_attn_slice: 'bfloat16[32, 2048, 32]' = aten.slice.Tensor(h5_getitem, 2, 0, 32)  # strides=(1048576, 512, 1), contiguous=False, view=True

    # self.transformer.h.5.attn.ve_gate (Linear)
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    h5_attn_ve_gate__to_copy: 'bfloat16[4, 32]' = aten._to_copy(transformer_h_5_attn_ve_gate_weight, dtype=torch.bfloat16)  # strides=(32, 1), contiguous=True, view=False
    h5_attn_ve_gate_t: 'bfloat16[32, 4]' = aten.t(h5_attn_ve_gate__to_copy)  # strides=(1, 32), contiguous=False, view=True
    h5_attn_ve_gate_view: 'bfloat16[65536, 32]' = aten.view(h5_attn_slice, [65536, 32])  # strides=(512, 1), contiguous=False, view=True
    h5_attn_ve_gate_mm: 'bfloat16[65536, 4]' = aten.mm(h5_attn_ve_gate_view, h5_attn_ve_gate_t)  # strides=(4, 1), contiguous=True, view=False
    h5_attn_ve_gate__unsafe_view: 'bfloat16[32, 2048, 4]' = aten._unsafe_view(h5_attn_ve_gate_mm, [32, 2048, 4])  # strides=(8192, 4, 1), contiguous=True, view=False

    # self.transformer.h.5.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    h5_attn_sigmoid: 'bfloat16[32, 2048, 4]' = aten.sigmoid(h5_attn_ve_gate__unsafe_view)  # strides=(8192, 4, 1), contiguous=True, view=False
    h5_attn_detach: 'bfloat16[32, 2048, 4]' = aten.detach(h5_attn_sigmoid)  # strides=(8192, 4, 1), contiguous=True, view=True
    h5_attn_mul: 'bfloat16[32, 2048, 4]' = aten.mul.Tensor(h5_attn_sigmoid, 2)  # strides=(8192, 4, 1), contiguous=True, view=False
    h5_attn_unsqueeze: 'bfloat16[32, 2048, 4, 1]' = aten.unsqueeze(h5_attn_mul, -1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h5_attn_mul_1: 'bfloat16[32, 2048, 4, 128]' = aten.mul.Tensor(h5_attn_unsqueeze, h5_attn_view_3)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    h5_attn_add: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(h5_attn_view_2, h5_attn_mul_1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    # FUSED: RoPE for Q
    h5_attn_neg: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h5_attn_cat: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h5_attn_view, getitem_slice, getitem_1_slice)
    # FUSED: RoPE for K
    h5_attn_neg_1: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h5_attn_cat_1: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h5_attn_view_1, getitem_slice, getitem_1_slice)
    h5_attn__fused_rms_norm = aten._fused_rms_norm(h5_attn_cat, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h5_attn_getitem: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h5_attn__fused_rms_norm, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h5_attn_getitem_1: 'float32[32, 2048, 4, 1]' = operator.getitem(h5_attn__fused_rms_norm, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h5_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(h5_attn_getitem_1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h5_attn__fused_rms_norm_1 = aten._fused_rms_norm(h5_attn_cat_1, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h5_attn_getitem_2: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h5_attn__fused_rms_norm_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h5_attn_getitem_3: 'float32[32, 2048, 4, 1]' = operator.getitem(h5_attn__fused_rms_norm_1, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h5_attn_detach_2: 'float32[32, 2048, 4, 1]' = aten.detach(h5_attn_getitem_3)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h5_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h5_attn_getitem, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h5_attn_transpose_1: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h5_attn_getitem_2, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h5_attn_transpose_2: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h5_attn_add, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h5_attn_where = _attn_mask  # reuse precomputed mask
    h5_attn__scaled_dot_product_cudnn_attention = aten._scaled_dot_product_cudnn_attention(h5_attn_transpose, h5_attn_transpose_1, h5_attn_transpose_2, h5_attn_where, True)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(8192, 2048, 1, 1), contiguous=True; out6: strides=(), contiguous=True; out7: strides=(), contiguous=True, view=False
    h5_attn_getitem_4: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(h5_attn__scaled_dot_product_cudnn_attention, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h5_attn_getitem_5: 'float32[32, 4, 2048, 1]' = operator.getitem(h5_attn__scaled_dot_product_cudnn_attention, 1)  # strides=(8192, 2048, 1, 1), contiguous=True, view=True
    h5_attn_getitem_6: 'int64[]' = operator.getitem(h5_attn__scaled_dot_product_cudnn_attention, 6)  # strides=(), contiguous=True, view=True
    h5_attn_getitem_7: 'int64[]' = operator.getitem(h5_attn__scaled_dot_product_cudnn_attention, 7)  # strides=(), contiguous=True, view=True
    h5_attn_detach_3: 'bfloat16[32, 4, 2048, 128]' = aten.detach(h5_attn_getitem_4)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h5_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(h5_attn_getitem_4, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h5_attn_view_4: 'bfloat16[32, 2048, 512]' = aten.view(h5_attn_transpose_3, [32, 2048, -1])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # self.transformer.h.5.attn.c_proj (Linear)
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    h5_attn_c_proj__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_5_attn_c_proj_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h5_attn_c_proj_t: 'bfloat16[512, 512]' = aten.t(h5_attn_c_proj__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h5_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(h5_attn_view_4, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h5_attn_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h5_attn_c_proj_view, h5_attn_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h5_attn_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h5_attn_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.5 (Block)
    # /.autoresearch_repo/train.py:114
    # x = x + self.attn(norm(x), ve, cos_sin, window_size)
    h5_add: 'bfloat16[32, 2048, 512]' = cuda_add(add_41_add, h5_attn_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add
    h5__fused_rms_norm_1 = aten._fused_rms_norm(h5_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h5_getitem_2: 'bfloat16[32, 2048, 512]' = operator.getitem(h5__fused_rms_norm_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h5_getitem_3: 'float32[32, 2048, 1]' = operator.getitem(h5__fused_rms_norm_1, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h5_detach_1: 'float32[32, 2048, 1]' = aten.detach(h5_getitem_3)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.5.mlp.c_fc (Linear)
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    h5_mlp_c_fc__to_copy: 'bfloat16[2048, 512]' = aten._to_copy(transformer_h_5_mlp_c_fc_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h5_mlp_c_fc_t: 'bfloat16[512, 2048]' = aten.t(h5_mlp_c_fc__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h5_mlp_c_fc_view: 'bfloat16[65536, 512]' = aten.view(h5_getitem_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h5_mlp_c_fc_mm: 'bfloat16[65536, 2048]' = aten.mm(h5_mlp_c_fc_view, h5_mlp_c_fc_t)  # strides=(2048, 1), contiguous=True, view=False
    h5_mlp_c_fc__unsafe_view: 'bfloat16[32, 2048, 2048]' = aten._unsafe_view(h5_mlp_c_fc_mm, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.5.mlp (MLP)
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    h5_mlp_relu: 'bfloat16[32, 2048, 2048]' = aten.relu(h5_mlp_c_fc__unsafe_view)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h5_mlp_detach: 'bfloat16[32, 2048, 2048]' = aten.detach(h5_mlp_relu)  # strides=(4194304, 2048, 1), contiguous=True, view=True
    h5_mlp__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(h5_mlp_relu, dtype=torch.float32)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h5_mlp_pow: 'float32[32, 2048, 2048]' = aten.pow.Tensor_Scalar(h5_mlp__to_copy, 2)  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.5.mlp.c_proj (Linear)
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    h5_mlp_c_proj__to_copy: 'bfloat16[512, 2048]' = aten._to_copy(transformer_h_5_mlp_c_proj_weight, dtype=torch.bfloat16)  # strides=(2048, 1), contiguous=True, view=False
    h5_mlp_c_proj__to_copy_1: 'bfloat16[32, 2048, 2048]' = aten._to_copy(h5_mlp_pow, dtype=torch.bfloat16)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h5_mlp_c_proj_t: 'bfloat16[2048, 512]' = aten.t(h5_mlp_c_proj__to_copy)  # strides=(1, 2048), contiguous=False, view=True
    h5_mlp_c_proj_view: 'bfloat16[65536, 2048]' = aten.view(h5_mlp_c_proj__to_copy_1, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    h5_mlp_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h5_mlp_c_proj_view, h5_mlp_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h5_mlp_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h5_mlp_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.5 (Block)
    # /.autoresearch_repo/train.py:115
    # x = x + self.mlp(norm(x))
    h5_add_1: 'bfloat16[32, 2048, 512]' = cuda_add(h5_add, h5_mlp_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    getitem_41_select: 'float32[]' = aten.select.int(resid_lambdas, 0, 6)  # strides=(), contiguous=True, view=True
    getitem_42_select: 'float32[]' = aten.select.int(x0_lambdas, 0, 6)  # strides=(), contiguous=True, view=True
    add_50_add: 'bfloat16[32, 2048, 512]' = triton_lambda_scale(h5_add_1, rms_norm_getitem, getitem_41_select, getitem_42_select)  # FUSED: a*x + b*x0

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.6
    # ════════════════════════════════════════════════════════════════

    # self.transformer.h.6 (Block)
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    h6__fused_rms_norm = aten._fused_rms_norm(add_50_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h6_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(h6__fused_rms_norm, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h6_getitem_1: 'float32[32, 2048, 1]' = operator.getitem(h6__fused_rms_norm, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h6_detach: 'float32[32, 2048, 1]' = aten.detach(h6_getitem_1)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.6.attn.c_q (Linear)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h6_attn_c_q__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_6_attn_c_q_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h6_attn_c_q_t: 'bfloat16[512, 512]' = aten.t(h6_attn_c_q__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h6_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(h6_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h6_attn_c_q_mm: 'bfloat16[65536, 512]' = aten.mm(h6_attn_c_q_view, h6_attn_c_q_t)  # strides=(512, 1), contiguous=True, view=False
    h6_attn_c_q__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h6_attn_c_q_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.6.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h6_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(h6_attn_c_q__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.6.attn.c_k (Linear)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h6_attn_c_k__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_6_attn_c_k_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h6_attn_c_k_t: 'bfloat16[512, 512]' = aten.t(h6_attn_c_k__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h6_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(h6_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h6_attn_c_k_mm: 'bfloat16[65536, 512]' = aten.mm(h6_attn_c_k_view, h6_attn_c_k_t)  # strides=(512, 1), contiguous=True, view=False
    h6_attn_c_k__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h6_attn_c_k_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.6.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h6_attn_view_1: 'bfloat16[32, 2048, 4, 128]' = aten.view(h6_attn_c_k__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.6.attn.c_v (Linear)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h6_attn_c_v__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_6_attn_c_v_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h6_attn_c_v_t: 'bfloat16[512, 512]' = aten.t(h6_attn_c_v__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h6_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(h6_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h6_attn_c_v_mm: 'bfloat16[65536, 512]' = aten.mm(h6_attn_c_v_view, h6_attn_c_v_t)  # strides=(512, 1), contiguous=True, view=False
    h6_attn_c_v__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h6_attn_c_v_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.6.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h6_attn_view_2: 'bfloat16[32, 2048, 4, 128]' = aten.view(h6_attn_c_v__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    # FUSED: RoPE for Q
    h6_attn_neg: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h6_attn_cat: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h6_attn_view, getitem_slice, getitem_1_slice)
    # FUSED: RoPE for K
    h6_attn_neg_1: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h6_attn_cat_1: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h6_attn_view_1, getitem_slice, getitem_1_slice)
    h6_attn__fused_rms_norm = aten._fused_rms_norm(h6_attn_cat, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h6_attn_getitem: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h6_attn__fused_rms_norm, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h6_attn_getitem_1: 'float32[32, 2048, 4, 1]' = operator.getitem(h6_attn__fused_rms_norm, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h6_attn_detach: 'float32[32, 2048, 4, 1]' = aten.detach(h6_attn_getitem_1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h6_attn__fused_rms_norm_1 = aten._fused_rms_norm(h6_attn_cat_1, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h6_attn_getitem_2: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h6_attn__fused_rms_norm_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h6_attn_getitem_3: 'float32[32, 2048, 4, 1]' = operator.getitem(h6_attn__fused_rms_norm_1, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h6_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(h6_attn_getitem_3)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h6_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h6_attn_getitem, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h6_attn_transpose_1: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h6_attn_getitem_2, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h6_attn_transpose_2: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h6_attn_view_2, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h6_attn_where = _attn_mask  # reuse precomputed mask
    h6_attn__scaled_dot_product_cudnn_attention = aten._scaled_dot_product_cudnn_attention(h6_attn_transpose, h6_attn_transpose_1, h6_attn_transpose_2, h6_attn_where, True)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(8192, 2048, 1, 1), contiguous=True; out6: strides=(), contiguous=True; out7: strides=(), contiguous=True, view=False
    h6_attn_getitem_4: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(h6_attn__scaled_dot_product_cudnn_attention, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h6_attn_getitem_5: 'float32[32, 4, 2048, 1]' = operator.getitem(h6_attn__scaled_dot_product_cudnn_attention, 1)  # strides=(8192, 2048, 1, 1), contiguous=True, view=True
    h6_attn_getitem_6: 'int64[]' = operator.getitem(h6_attn__scaled_dot_product_cudnn_attention, 6)  # strides=(), contiguous=True, view=True
    h6_attn_getitem_7: 'int64[]' = operator.getitem(h6_attn__scaled_dot_product_cudnn_attention, 7)  # strides=(), contiguous=True, view=True
    h6_attn_detach_2: 'bfloat16[32, 4, 2048, 128]' = aten.detach(h6_attn_getitem_4)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h6_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(h6_attn_getitem_4, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h6_attn_view_3: 'bfloat16[32, 2048, 512]' = aten.view(h6_attn_transpose_3, [32, 2048, -1])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # self.transformer.h.6.attn.c_proj (Linear)
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    h6_attn_c_proj__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_6_attn_c_proj_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h6_attn_c_proj_t: 'bfloat16[512, 512]' = aten.t(h6_attn_c_proj__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h6_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(h6_attn_view_3, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h6_attn_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h6_attn_c_proj_view, h6_attn_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h6_attn_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h6_attn_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.6 (Block)
    # /.autoresearch_repo/train.py:114
    # x = x + self.attn(norm(x), ve, cos_sin, window_size)
    h6_add: 'bfloat16[32, 2048, 512]' = cuda_add(add_50_add, h6_attn_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add
    h6__fused_rms_norm_1 = aten._fused_rms_norm(h6_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h6_getitem_2: 'bfloat16[32, 2048, 512]' = operator.getitem(h6__fused_rms_norm_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h6_getitem_3: 'float32[32, 2048, 1]' = operator.getitem(h6__fused_rms_norm_1, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h6_detach_1: 'float32[32, 2048, 1]' = aten.detach(h6_getitem_3)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.6.mlp.c_fc (Linear)
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    h6_mlp_c_fc__to_copy: 'bfloat16[2048, 512]' = aten._to_copy(transformer_h_6_mlp_c_fc_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h6_mlp_c_fc_t: 'bfloat16[512, 2048]' = aten.t(h6_mlp_c_fc__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h6_mlp_c_fc_view: 'bfloat16[65536, 512]' = aten.view(h6_getitem_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h6_mlp_c_fc_mm: 'bfloat16[65536, 2048]' = aten.mm(h6_mlp_c_fc_view, h6_mlp_c_fc_t)  # strides=(2048, 1), contiguous=True, view=False
    h6_mlp_c_fc__unsafe_view: 'bfloat16[32, 2048, 2048]' = aten._unsafe_view(h6_mlp_c_fc_mm, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.6.mlp (MLP)
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    h6_mlp_relu: 'bfloat16[32, 2048, 2048]' = aten.relu(h6_mlp_c_fc__unsafe_view)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h6_mlp_detach: 'bfloat16[32, 2048, 2048]' = aten.detach(h6_mlp_relu)  # strides=(4194304, 2048, 1), contiguous=True, view=True
    h6_mlp__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(h6_mlp_relu, dtype=torch.float32)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h6_mlp_pow: 'float32[32, 2048, 2048]' = aten.pow.Tensor_Scalar(h6_mlp__to_copy, 2)  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.6.mlp.c_proj (Linear)
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    h6_mlp_c_proj__to_copy: 'bfloat16[512, 2048]' = aten._to_copy(transformer_h_6_mlp_c_proj_weight, dtype=torch.bfloat16)  # strides=(2048, 1), contiguous=True, view=False
    h6_mlp_c_proj__to_copy_1: 'bfloat16[32, 2048, 2048]' = aten._to_copy(h6_mlp_pow, dtype=torch.bfloat16)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h6_mlp_c_proj_t: 'bfloat16[2048, 512]' = aten.t(h6_mlp_c_proj__to_copy)  # strides=(1, 2048), contiguous=False, view=True
    h6_mlp_c_proj_view: 'bfloat16[65536, 2048]' = aten.view(h6_mlp_c_proj__to_copy_1, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    h6_mlp_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h6_mlp_c_proj_view, h6_mlp_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h6_mlp_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h6_mlp_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.6 (Block)
    # /.autoresearch_repo/train.py:115
    # x = x + self.mlp(norm(x))
    h6_add_1: 'bfloat16[32, 2048, 512]' = cuda_add(h6_add, h6_mlp_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    getitem_47_select: 'float32[]' = aten.select.int(resid_lambdas, 0, 7)  # strides=(), contiguous=True, view=True
    getitem_48_select: 'float32[]' = aten.select.int(x0_lambdas, 0, 7)  # strides=(), contiguous=True, view=True
    add_58_add: 'bfloat16[32, 2048, 512]' = triton_lambda_scale(h6_add_1, rms_norm_getitem, getitem_47_select, getitem_48_select)  # FUSED: a*x + b*x0

    # ════════════════════════════════════════════════════════════════
    # self.value_embeds.7 (Embedding)
    # ════════════════════════════════════════════════════════════════
    (value_embeds7_embedding,) = embedding(
        weight=value_embeds_7_weight,
        input__32__2048=input__32__2048,
    )

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.7
    # ════════════════════════════════════════════════════════════════

    # self.transformer.h.7 (Block)
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    h7__fused_rms_norm = aten._fused_rms_norm(add_58_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h7_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(h7__fused_rms_norm, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h7_getitem_1: 'float32[32, 2048, 1]' = operator.getitem(h7__fused_rms_norm, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h7_detach: 'float32[32, 2048, 1]' = aten.detach(h7_getitem_1)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.7.attn.c_q (Linear)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h7_attn_c_q__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_7_attn_c_q_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h7_attn_c_q_t: 'bfloat16[512, 512]' = aten.t(h7_attn_c_q__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h7_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(h7_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h7_attn_c_q_mm: 'bfloat16[65536, 512]' = aten.mm(h7_attn_c_q_view, h7_attn_c_q_t)  # strides=(512, 1), contiguous=True, view=False
    h7_attn_c_q__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h7_attn_c_q_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.7.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    h7_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(h7_attn_c_q__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.7.attn.c_k (Linear)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h7_attn_c_k__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_7_attn_c_k_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h7_attn_c_k_t: 'bfloat16[512, 512]' = aten.t(h7_attn_c_k__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h7_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(h7_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h7_attn_c_k_mm: 'bfloat16[65536, 512]' = aten.mm(h7_attn_c_k_view, h7_attn_c_k_t)  # strides=(512, 1), contiguous=True, view=False
    h7_attn_c_k__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h7_attn_c_k_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.7.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    h7_attn_view_1: 'bfloat16[32, 2048, 4, 128]' = aten.view(h7_attn_c_k__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True

    # self.transformer.h.7.attn.c_v (Linear)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h7_attn_c_v__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_7_attn_c_v_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h7_attn_c_v_t: 'bfloat16[512, 512]' = aten.t(h7_attn_c_v__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h7_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(h7_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h7_attn_c_v_mm: 'bfloat16[65536, 512]' = aten.mm(h7_attn_c_v_view, h7_attn_c_v_t)  # strides=(512, 1), contiguous=True, view=False
    h7_attn_c_v__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h7_attn_c_v_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.7.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    h7_attn_view_2: 'bfloat16[32, 2048, 4, 128]' = aten.view(h7_attn_c_v__unsafe_view, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h7_attn_view_3: 'bfloat16[32, 2048, 4, 128]' = aten.view(value_embeds7_embedding, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h7_attn_slice: 'bfloat16[32, 2048, 32]' = aten.slice.Tensor(h7_getitem, 2, 0, 32)  # strides=(1048576, 512, 1), contiguous=False, view=True

    # self.transformer.h.7.attn.ve_gate (Linear)
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    h7_attn_ve_gate__to_copy: 'bfloat16[4, 32]' = aten._to_copy(transformer_h_7_attn_ve_gate_weight, dtype=torch.bfloat16)  # strides=(32, 1), contiguous=True, view=False
    h7_attn_ve_gate_t: 'bfloat16[32, 4]' = aten.t(h7_attn_ve_gate__to_copy)  # strides=(1, 32), contiguous=False, view=True
    h7_attn_ve_gate_view: 'bfloat16[65536, 32]' = aten.view(h7_attn_slice, [65536, 32])  # strides=(512, 1), contiguous=False, view=True
    h7_attn_ve_gate_mm: 'bfloat16[65536, 4]' = aten.mm(h7_attn_ve_gate_view, h7_attn_ve_gate_t)  # strides=(4, 1), contiguous=True, view=False
    h7_attn_ve_gate__unsafe_view: 'bfloat16[32, 2048, 4]' = aten._unsafe_view(h7_attn_ve_gate_mm, [32, 2048, 4])  # strides=(8192, 4, 1), contiguous=True, view=False

    # self.transformer.h.7.attn (CausalSelfAttention)
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    h7_attn_sigmoid: 'bfloat16[32, 2048, 4]' = aten.sigmoid(h7_attn_ve_gate__unsafe_view)  # strides=(8192, 4, 1), contiguous=True, view=False
    h7_attn_detach: 'bfloat16[32, 2048, 4]' = aten.detach(h7_attn_sigmoid)  # strides=(8192, 4, 1), contiguous=True, view=True
    h7_attn_mul: 'bfloat16[32, 2048, 4]' = aten.mul.Tensor(h7_attn_sigmoid, 2)  # strides=(8192, 4, 1), contiguous=True, view=False
    h7_attn_unsqueeze: 'bfloat16[32, 2048, 4, 1]' = aten.unsqueeze(h7_attn_mul, -1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h7_attn_mul_1: 'bfloat16[32, 2048, 4, 128]' = aten.mul.Tensor(h7_attn_unsqueeze, h7_attn_view_3)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    h7_attn_add: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(h7_attn_view_2, h7_attn_mul_1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    # FUSED: RoPE for Q
    h7_attn_neg: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h7_attn_cat: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h7_attn_view, getitem_slice, getitem_1_slice)
    # FUSED: RoPE for K
    h7_attn_neg_1: 'bfloat16[1, 2048, 1, 64]' = _neg_sin
    h7_attn_cat_1: 'bfloat16[32, 2048, 4, 128]' = triton_rope_fwd(h7_attn_view_1, getitem_slice, getitem_1_slice)
    h7_attn__fused_rms_norm = aten._fused_rms_norm(h7_attn_cat, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h7_attn_getitem: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h7_attn__fused_rms_norm, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h7_attn_getitem_1: 'float32[32, 2048, 4, 1]' = operator.getitem(h7_attn__fused_rms_norm, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h7_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(h7_attn_getitem_1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h7_attn__fused_rms_norm_1 = aten._fused_rms_norm(h7_attn_cat_1, [128], None, None)  # out0: strides=(1048576, 512, 128, 1), contiguous=True; out1: strides=(8192, 4, 1, 1), contiguous=True, view=False
    h7_attn_getitem_2: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(h7_attn__fused_rms_norm_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h7_attn_getitem_3: 'float32[32, 2048, 4, 1]' = operator.getitem(h7_attn__fused_rms_norm_1, 1)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h7_attn_detach_2: 'float32[32, 2048, 4, 1]' = aten.detach(h7_attn_getitem_3)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    h7_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h7_attn_getitem, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h7_attn_transpose_1: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h7_attn_getitem_2, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h7_attn_transpose_2: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(h7_attn_add, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h7_attn__scaled_dot_product_cudnn_attention = aten._scaled_dot_product_cudnn_attention(h7_attn_transpose, h7_attn_transpose_1, h7_attn_transpose_2, None, True, 0.0, True)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(8192, 2048, 1, 1), contiguous=True; out6: strides=(), contiguous=True; out7: strides=(), contiguous=True, view=False
    h7_attn_getitem_4: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(h7_attn__scaled_dot_product_cudnn_attention, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h7_attn_getitem_5: 'float32[32, 4, 2048, 1]' = operator.getitem(h7_attn__scaled_dot_product_cudnn_attention, 1)  # strides=(8192, 2048, 1, 1), contiguous=True, view=True
    h7_attn_getitem_6: 'int64[]' = operator.getitem(h7_attn__scaled_dot_product_cudnn_attention, 6)  # strides=(), contiguous=True, view=True
    h7_attn_getitem_7: 'int64[]' = operator.getitem(h7_attn__scaled_dot_product_cudnn_attention, 7)  # strides=(), contiguous=True, view=True
    h7_attn_detach_3: 'bfloat16[32, 4, 2048, 128]' = aten.detach(h7_attn_getitem_4)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    h7_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(h7_attn_getitem_4, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    h7_attn_view_4: 'bfloat16[32, 2048, 512]' = aten.view(h7_attn_transpose_3, [32, 2048, -1])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # self.transformer.h.7.attn.c_proj (Linear)
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    h7_attn_c_proj__to_copy: 'bfloat16[512, 512]' = aten._to_copy(transformer_h_7_attn_c_proj_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h7_attn_c_proj_t: 'bfloat16[512, 512]' = aten.t(h7_attn_c_proj__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h7_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(h7_attn_view_4, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h7_attn_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h7_attn_c_proj_view, h7_attn_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h7_attn_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h7_attn_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.7 (Block)
    # /.autoresearch_repo/train.py:114
    # x = x + self.attn(norm(x), ve, cos_sin, window_size)
    h7_add: 'bfloat16[32, 2048, 512]' = cuda_add(add_58_add, h7_attn_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add
    h7__fused_rms_norm_1 = aten._fused_rms_norm(h7_add, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    h7_getitem_2: 'bfloat16[32, 2048, 512]' = operator.getitem(h7__fused_rms_norm_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    h7_getitem_3: 'float32[32, 2048, 1]' = operator.getitem(h7__fused_rms_norm_1, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    h7_detach_1: 'float32[32, 2048, 1]' = aten.detach(h7_getitem_3)  # strides=(2048, 1, 1), contiguous=True, view=True

    # self.transformer.h.7.mlp.c_fc (Linear)
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    h7_mlp_c_fc__to_copy: 'bfloat16[2048, 512]' = aten._to_copy(transformer_h_7_mlp_c_fc_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    h7_mlp_c_fc_t: 'bfloat16[512, 2048]' = aten.t(h7_mlp_c_fc__to_copy)  # strides=(1, 512), contiguous=False, view=True
    h7_mlp_c_fc_view: 'bfloat16[65536, 512]' = aten.view(h7_getitem_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    h7_mlp_c_fc_mm: 'bfloat16[65536, 2048]' = aten.mm(h7_mlp_c_fc_view, h7_mlp_c_fc_t)  # strides=(2048, 1), contiguous=True, view=False
    h7_mlp_c_fc__unsafe_view: 'bfloat16[32, 2048, 2048]' = aten._unsafe_view(h7_mlp_c_fc_mm, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.7.mlp (MLP)
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    h7_mlp_relu: 'bfloat16[32, 2048, 2048]' = aten.relu(h7_mlp_c_fc__unsafe_view)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h7_mlp_detach: 'bfloat16[32, 2048, 2048]' = aten.detach(h7_mlp_relu)  # strides=(4194304, 2048, 1), contiguous=True, view=True
    h7_mlp__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(h7_mlp_relu, dtype=torch.float32)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h7_mlp_pow: 'float32[32, 2048, 2048]' = aten.pow.Tensor_Scalar(h7_mlp__to_copy, 2)  # strides=(4194304, 2048, 1), contiguous=True, view=False

    # self.transformer.h.7.mlp.c_proj (Linear)
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    h7_mlp_c_proj__to_copy: 'bfloat16[512, 2048]' = aten._to_copy(transformer_h_7_mlp_c_proj_weight, dtype=torch.bfloat16)  # strides=(2048, 1), contiguous=True, view=False
    h7_mlp_c_proj__to_copy_1: 'bfloat16[32, 2048, 2048]' = aten._to_copy(h7_mlp_pow, dtype=torch.bfloat16)  # strides=(4194304, 2048, 1), contiguous=True, view=False
    h7_mlp_c_proj_t: 'bfloat16[2048, 512]' = aten.t(h7_mlp_c_proj__to_copy)  # strides=(1, 2048), contiguous=False, view=True
    h7_mlp_c_proj_view: 'bfloat16[65536, 2048]' = aten.view(h7_mlp_c_proj__to_copy_1, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    h7_mlp_c_proj_mm: 'bfloat16[65536, 512]' = aten.mm(h7_mlp_c_proj_view, h7_mlp_c_proj_t)  # strides=(512, 1), contiguous=True, view=False
    h7_mlp_c_proj__unsafe_view: 'bfloat16[32, 2048, 512]' = aten._unsafe_view(h7_mlp_c_proj_mm, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=False

    # self.transformer.h.7 (Block)
    # /.autoresearch_repo/train.py:115
    # x = x + self.mlp(norm(x))
    h7_add_1: 'bfloat16[32, 2048, 512]' = cuda_add(h7_add, h7_mlp_c_proj__unsafe_view)  # FUSED: vectorized CUDA bf16 add

    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    rms_norm_33__fused_rms_norm = aten._fused_rms_norm(h7_add_1, [512], None, None)  # out0: strides=(1048576, 512, 1), contiguous=True; out1: strides=(2048, 1, 1), contiguous=True, view=False
    rms_norm_33_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(rms_norm_33__fused_rms_norm, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    rms_norm_33_getitem_1: 'float32[32, 2048, 1]' = operator.getitem(rms_norm_33__fused_rms_norm, 1)  # strides=(2048, 1, 1), contiguous=True, view=True
    rms_norm_33_detach: 'float32[32, 2048, 1]' = aten.detach(rms_norm_33_getitem_1)  # strides=(2048, 1, 1), contiguous=True, view=True

    # ════════════════════════════════════════════════════════════════
    # self.lm_head
    # ════════════════════════════════════════════════════════════════

    # self.lm_head (Linear)
    # /.autoresearch_repo/train.py:278
    # logits = self.lm_head(x)
    lm_head__to_copy: 'bfloat16[8192, 512]' = aten._to_copy(lm_head_weight, dtype=torch.bfloat16)  # strides=(512, 1), contiguous=True, view=False
    lm_head_t: 'bfloat16[512, 8192]' = aten.t(lm_head__to_copy)  # strides=(1, 512), contiguous=False, view=True
    lm_head_view: 'bfloat16[65536, 512]' = aten.view(rms_norm_33_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    lm_head_mm: 'bfloat16[65536, 8192]' = aten.mm(lm_head_view, lm_head_t)  # strides=(8192, 1), contiguous=True, view=False
    lm_head__unsafe_view: 'bfloat16[32, 2048, 8192]' = aten._unsafe_view(lm_head_mm, [32, 2048, 8192])  # strides=(16777216, 8192, 1), contiguous=True, view=False

    # /.autoresearch_repo/train.py:279-280
    # logits = softcap * torch.tanh(logits.float() / softcap)
    # FUSED: bf16→fp32 + div(15) + tanh + mul(15) in single Triton kernel
    mul_88_mul, tanh_detach = triton_softcap_fwd(lm_head__unsafe_view, softcap=15)

    # /.autoresearch_repo/train.py:283
    # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
    view_36_view: 'float32[65536, 8192]' = aten.view(mul_88_mul, [-1, 8192])  # strides=(8192, 1), contiguous=True, view=True
    view_37_view: 'int64[65536]' = aten.view(input__32__2048_1, [-1])  # strides=(1,), contiguous=True, view=True
    cross_entropy__log_softmax: 'float32[65536, 8192]' = aten._log_softmax(view_36_view, 1, False)  # strides=(8192, 1), contiguous=True, view=False
    cross_entropy_detach: 'float32[65536, 8192]' = aten.detach(cross_entropy__log_softmax)  # strides=(8192, 1), contiguous=True, view=True
    cross_entropy_nll_loss_forward = aten.nll_loss_forward(cross_entropy__log_softmax, view_37_view, None, 1, -1)  # out0: strides=(), contiguous=True; out1: strides=(), contiguous=True, view=False
    cross_entropy_getitem: 'float32[]' = operator.getitem(cross_entropy_nll_loss_forward, 0)  # strides=(), contiguous=True, view=True
    cross_entropy_getitem_1: 'float32[]' = operator.getitem(cross_entropy_nll_loss_forward, 1)  # strides=(), contiguous=True, view=True
    return (cross_entropy_getitem, input__32__2048, getitem_slice, getitem_1_slice, wte_embedding, rms_norm_getitem, rms_norm_detach, getitem_2_select, getitem_3_select, add_add, h0_detach, h0_attn_c_q_t, h0_attn_c_q_view, h0_attn_c_k_t, h0_attn_c_k_view, h0_attn_c_v_t, h0_attn_c_v_view, h0_attn_neg, h0_attn_cat, h0_attn_neg_1, h0_attn_cat_1, h0_attn_detach, h0_attn_detach_1, h0_attn_transpose, h0_attn_transpose_1, h0_attn_transpose_2, h0_attn_where, h0_attn_getitem_5, h0_attn_getitem_6, h0_attn_getitem_7, h0_attn_detach_2, h0_attn_c_proj_t, h0_attn_c_proj_view, h0_add, h0_detach_1, h0_mlp_c_fc_t, h0_mlp_c_fc_view, h0_mlp_detach, h0_mlp__to_copy, h0_mlp_c_proj_t, h0_mlp_c_proj_view, h0_add_1, getitem_8_select, getitem_9_select, add_8_add, h1_detach, h1_attn_c_q_t, h1_attn_c_q_view, h1_attn_c_k_t, h1_attn_c_k_view, h1_attn_c_v_t, h1_attn_c_v_view, h1_attn_view_3, h1_attn_ve_gate_t, h1_attn_ve_gate_view, h1_attn_detach, h1_attn_unsqueeze, h1_attn_neg, h1_attn_cat, h1_attn_neg_1, h1_attn_cat_1, h1_attn_detach_1, h1_attn_detach_2, h1_attn_transpose, h1_attn_transpose_1, h1_attn_transpose_2, h1_attn_where, h1_attn_getitem_5, h1_attn_getitem_6, h1_attn_getitem_7, h1_attn_detach_3, h1_attn_c_proj_t, h1_attn_c_proj_view, h1_add, h1_detach_1, h1_mlp_c_fc_t, h1_mlp_c_fc_view, h1_mlp_detach, h1_mlp__to_copy, h1_mlp_c_proj_t, h1_mlp_c_proj_view, h1_add_1, getitem_15_select, getitem_16_select, add_17_add, h2_detach, h2_attn_c_q_t, h2_attn_c_q_view, h2_attn_c_k_t, h2_attn_c_k_view, h2_attn_c_v_t, h2_attn_c_v_view, h2_attn_neg, h2_attn_cat, h2_attn_neg_1, h2_attn_cat_1, h2_attn_detach, h2_attn_detach_1, h2_attn_transpose, h2_attn_transpose_1, h2_attn_transpose_2, h2_attn_where, h2_attn_getitem_5, h2_attn_getitem_6, h2_attn_getitem_7, h2_attn_detach_2, h2_attn_c_proj_t, h2_attn_c_proj_view, h2_add, h2_detach_1, h2_mlp_c_fc_t, h2_mlp_c_fc_view, h2_mlp_detach, h2_mlp__to_copy, h2_mlp_c_proj_t, h2_mlp_c_proj_view, h2_add_1, getitem_21_select, getitem_22_select, add_25_add, h3_detach, h3_attn_c_q_t, h3_attn_c_q_view, h3_attn_c_k_t, h3_attn_c_k_view, h3_attn_c_v_t, h3_attn_c_v_view, h3_attn_view_3, h3_attn_ve_gate_t, h3_attn_ve_gate_view, h3_attn_detach, h3_attn_unsqueeze, h3_attn_neg, h3_attn_cat, h3_attn_neg_1, h3_attn_cat_1, h3_attn_detach_1, h3_attn_detach_2, h3_attn_transpose, h3_attn_transpose_1, h3_attn_transpose_2, h3_attn_getitem_5, h3_attn_getitem_6, h3_attn_getitem_7, h3_attn_detach_3, h3_attn_c_proj_t, h3_attn_c_proj_view, h3_add, h3_detach_1, h3_mlp_c_fc_t, h3_mlp_c_fc_view, h3_mlp_detach, h3_mlp__to_copy, h3_mlp_c_proj_t, h3_mlp_c_proj_view, h3_add_1, getitem_28_select, getitem_29_select, add_33_add, h4_detach, h4_attn_c_q_t, h4_attn_c_q_view, h4_attn_c_k_t, h4_attn_c_k_view, h4_attn_c_v_t, h4_attn_c_v_view, h4_attn_neg, h4_attn_cat, h4_attn_neg_1, h4_attn_cat_1, h4_attn_detach, h4_attn_detach_1, h4_attn_transpose, h4_attn_transpose_1, h4_attn_transpose_2, h4_attn_where, h4_attn_getitem_5, h4_attn_getitem_6, h4_attn_getitem_7, h4_attn_detach_2, h4_attn_c_proj_t, h4_attn_c_proj_view, h4_add, h4_detach_1, h4_mlp_c_fc_t, h4_mlp_c_fc_view, h4_mlp_detach, h4_mlp__to_copy, h4_mlp_c_proj_t, h4_mlp_c_proj_view, h4_add_1, getitem_34_select, getitem_35_select, add_41_add, h5_detach, h5_attn_c_q_t, h5_attn_c_q_view, h5_attn_c_k_t, h5_attn_c_k_view, h5_attn_c_v_t, h5_attn_c_v_view, h5_attn_view_3, h5_attn_ve_gate_t, h5_attn_ve_gate_view, h5_attn_detach, h5_attn_unsqueeze, h5_attn_neg, h5_attn_cat, h5_attn_neg_1, h5_attn_cat_1, h5_attn_detach_1, h5_attn_detach_2, h5_attn_transpose, h5_attn_transpose_1, h5_attn_transpose_2, h5_attn_where, h5_attn_getitem_5, h5_attn_getitem_6, h5_attn_getitem_7, h5_attn_detach_3, h5_attn_c_proj_t, h5_attn_c_proj_view, h5_add, h5_detach_1, h5_mlp_c_fc_t, h5_mlp_c_fc_view, h5_mlp_detach, h5_mlp__to_copy, h5_mlp_c_proj_t, h5_mlp_c_proj_view, h5_add_1, getitem_41_select, getitem_42_select, add_50_add, h6_detach, h6_attn_c_q_t, h6_attn_c_q_view, h6_attn_c_k_t, h6_attn_c_k_view, h6_attn_c_v_t, h6_attn_c_v_view, h6_attn_neg, h6_attn_cat, h6_attn_neg_1, h6_attn_cat_1, h6_attn_detach, h6_attn_detach_1, h6_attn_transpose, h6_attn_transpose_1, h6_attn_transpose_2, h6_attn_where, h6_attn_getitem_5, h6_attn_getitem_6, h6_attn_getitem_7, h6_attn_detach_2, h6_attn_c_proj_t, h6_attn_c_proj_view, h6_add, h6_detach_1, h6_mlp_c_fc_t, h6_mlp_c_fc_view, h6_mlp_detach, h6_mlp__to_copy, h6_mlp_c_proj_t, h6_mlp_c_proj_view, h6_add_1, getitem_47_select, getitem_48_select, add_58_add, h7_detach, h7_attn_c_q_t, h7_attn_c_q_view, h7_attn_c_k_t, h7_attn_c_k_view, h7_attn_c_v_t, h7_attn_c_v_view, h7_attn_view_3, h7_attn_ve_gate_t, h7_attn_ve_gate_view, h7_attn_detach, h7_attn_unsqueeze, h7_attn_neg, h7_attn_cat, h7_attn_neg_1, h7_attn_cat_1, h7_attn_detach_1, h7_attn_detach_2, h7_attn_transpose, h7_attn_transpose_1, h7_attn_transpose_2, h7_attn_getitem_5, h7_attn_getitem_6, h7_attn_getitem_7, h7_attn_detach_3, h7_attn_c_proj_t, h7_attn_c_proj_view, h7_add, h7_detach_1, h7_mlp_c_fc_t, h7_mlp_c_fc_view, h7_mlp_detach, h7_mlp__to_copy, h7_mlp_c_proj_t, h7_mlp_c_proj_view, h7_add_1, rms_norm_33_detach, lm_head_t, lm_head_view, tanh_detach, view_37_view, cross_entropy__log_softmax, cross_entropy_detach, cross_entropy_getitem_1,)


# ======================================================================
# BACKWARD PASS (aten ops - the autograd graph!)
# ======================================================================
# This IS the autograd. Every gradient computation is an explicit
# aten op. You can edit the backward pass just like the forward:
#   - Modify gradient computations
#   - Add gradient clipping as raw ops
#   - Implement custom gradient scaling
#   - Skip gradient computation for specific parameters
# 
# Source annotations show which FORWARD op each gradient group
# corresponds to, using compact file:line notes or module headers.
# ======================================================================

def backward(
    input__32__2048: 'int64[32, 2048]',
    slice_1: 'bfloat16[1, 2048, 1, 64]',
    slice_2: 'bfloat16[1, 2048, 1, 64]',
    embedding: 'bfloat16[32, 2048, 512]',
    getitem: 'bfloat16[32, 2048, 512]',
    detach: 'float32[32, 2048, 1]',
    select: 'float32[]',
    select_1: 'float32[]',
    add: 'bfloat16[32, 2048, 512]',
    detach_1: 'float32[32, 2048, 1]',
    t: 'bfloat16[512, 512]',
    view: 'bfloat16[65536, 512]',
    t_1: 'bfloat16[512, 512]',
    view_2: 'bfloat16[65536, 512]',
    t_2: 'bfloat16[512, 512]',
    view_4: 'bfloat16[65536, 512]',
    neg: 'bfloat16[1, 2048, 1, 64]',
    cat: 'bfloat16[32, 2048, 4, 128]',
    neg_1: 'bfloat16[1, 2048, 1, 64]',
    cat_1: 'bfloat16[32, 2048, 4, 128]',
    detach_2: 'float32[32, 2048, 4, 1]',
    detach_3: 'float32[32, 2048, 4, 1]',
    transpose: 'bfloat16[32, 4, 2048, 128]',
    transpose_1: 'bfloat16[32, 4, 2048, 128]',
    transpose_2: 'bfloat16[32, 4, 2048, 128]',
    where: 'bfloat16[2048, 2048]',
    getitem_9: 'float32[32, 4, 2048, 1]',
    getitem_14: 'int64[]',
    getitem_15: 'int64[]',
    detach_4: 'bfloat16[32, 4, 2048, 128]',
    t_3: 'bfloat16[512, 512]',
    view_7: 'bfloat16[65536, 512]',
    add_6: 'bfloat16[32, 2048, 512]',
    detach_5: 'float32[32, 2048, 1]',
    t_4: 'bfloat16[512, 2048]',
    view_8: 'bfloat16[65536, 512]',
    detach_6: 'bfloat16[32, 2048, 2048]',
    _to_copy_5: 'float32[32, 2048, 2048]',
    t_5: 'bfloat16[2048, 512]',
    view_9: 'bfloat16[65536, 2048]',
    add_7: 'bfloat16[32, 2048, 512]',
    select_2: 'float32[]',
    select_3: 'float32[]',
    add_8: 'bfloat16[32, 2048, 512]',
    detach_7: 'float32[32, 2048, 1]',
    t_6: 'bfloat16[512, 512]',
    view_10: 'bfloat16[65536, 512]',
    t_7: 'bfloat16[512, 512]',
    view_12: 'bfloat16[65536, 512]',
    t_8: 'bfloat16[512, 512]',
    view_14: 'bfloat16[65536, 512]',
    view_16: 'bfloat16[32, 2048, 4, 128]',
    t_9: 'bfloat16[32, 4]',
    view_17: 'bfloat16[65536, 32]',
    detach_8: 'bfloat16[32, 2048, 4]',
    unsqueeze_2: 'bfloat16[32, 2048, 4, 1]',
    neg_2: 'bfloat16[1, 2048, 1, 64]',
    cat_2: 'bfloat16[32, 2048, 4, 128]',
    neg_3: 'bfloat16[1, 2048, 1, 64]',
    cat_3: 'bfloat16[32, 2048, 4, 128]',
    detach_9: 'float32[32, 2048, 4, 1]',
    detach_10: 'float32[32, 2048, 4, 1]',
    transpose_4: 'bfloat16[32, 4, 2048, 128]',
    transpose_5: 'bfloat16[32, 4, 2048, 128]',
    transpose_6: 'bfloat16[32, 4, 2048, 128]',
    where_1: 'bfloat16[2048, 2048]',
    getitem_26: 'float32[32, 4, 2048, 1]',
    getitem_31: 'int64[]',
    getitem_32: 'int64[]',
    detach_11: 'bfloat16[32, 4, 2048, 128]',
    t_10: 'bfloat16[512, 512]',
    view_19: 'bfloat16[65536, 512]',
    add_15: 'bfloat16[32, 2048, 512]',
    detach_12: 'float32[32, 2048, 1]',
    t_11: 'bfloat16[512, 2048]',
    view_20: 'bfloat16[65536, 512]',
    detach_13: 'bfloat16[32, 2048, 2048]',
    _to_copy_14: 'float32[32, 2048, 2048]',
    t_12: 'bfloat16[2048, 512]',
    view_21: 'bfloat16[65536, 2048]',
    add_16: 'bfloat16[32, 2048, 512]',
    select_4: 'float32[]',
    select_5: 'float32[]',
    add_17: 'bfloat16[32, 2048, 512]',
    detach_14: 'float32[32, 2048, 1]',
    t_13: 'bfloat16[512, 512]',
    view_22: 'bfloat16[65536, 512]',
    t_14: 'bfloat16[512, 512]',
    view_24: 'bfloat16[65536, 512]',
    t_15: 'bfloat16[512, 512]',
    view_26: 'bfloat16[65536, 512]',
    neg_4: 'bfloat16[1, 2048, 1, 64]',
    cat_4: 'bfloat16[32, 2048, 4, 128]',
    neg_5: 'bfloat16[1, 2048, 1, 64]',
    cat_5: 'bfloat16[32, 2048, 4, 128]',
    detach_15: 'float32[32, 2048, 4, 1]',
    detach_16: 'float32[32, 2048, 4, 1]',
    transpose_8: 'bfloat16[32, 4, 2048, 128]',
    transpose_9: 'bfloat16[32, 4, 2048, 128]',
    transpose_10: 'bfloat16[32, 4, 2048, 128]',
    where_2: 'bfloat16[2048, 2048]',
    getitem_43: 'float32[32, 4, 2048, 1]',
    getitem_48: 'int64[]',
    getitem_49: 'int64[]',
    detach_17: 'bfloat16[32, 4, 2048, 128]',
    t_16: 'bfloat16[512, 512]',
    view_29: 'bfloat16[65536, 512]',
    add_23: 'bfloat16[32, 2048, 512]',
    detach_18: 'float32[32, 2048, 1]',
    t_17: 'bfloat16[512, 2048]',
    view_30: 'bfloat16[65536, 512]',
    detach_19: 'bfloat16[32, 2048, 2048]',
    _to_copy_22: 'float32[32, 2048, 2048]',
    t_18: 'bfloat16[2048, 512]',
    view_31: 'bfloat16[65536, 2048]',
    add_24: 'bfloat16[32, 2048, 512]',
    select_6: 'float32[]',
    select_7: 'float32[]',
    add_25: 'bfloat16[32, 2048, 512]',
    detach_20: 'float32[32, 2048, 1]',
    t_19: 'bfloat16[512, 512]',
    view_32: 'bfloat16[65536, 512]',
    t_20: 'bfloat16[512, 512]',
    view_34: 'bfloat16[65536, 512]',
    t_21: 'bfloat16[512, 512]',
    view_36: 'bfloat16[65536, 512]',
    view_38: 'bfloat16[32, 2048, 4, 128]',
    t_22: 'bfloat16[32, 4]',
    view_39: 'bfloat16[65536, 32]',
    detach_21: 'bfloat16[32, 2048, 4]',
    unsqueeze_7: 'bfloat16[32, 2048, 4, 1]',
    neg_6: 'bfloat16[1, 2048, 1, 64]',
    cat_6: 'bfloat16[32, 2048, 4, 128]',
    neg_7: 'bfloat16[1, 2048, 1, 64]',
    cat_7: 'bfloat16[32, 2048, 4, 128]',
    detach_22: 'float32[32, 2048, 4, 1]',
    detach_23: 'float32[32, 2048, 4, 1]',
    transpose_12: 'bfloat16[32, 4, 2048, 128]',
    transpose_13: 'bfloat16[32, 4, 2048, 128]',
    transpose_14: 'bfloat16[32, 4, 2048, 128]',
    getitem_60: 'float32[32, 4, 2048, 1]',
    getitem_65: 'int64[]',
    getitem_66: 'int64[]',
    detach_24: 'bfloat16[32, 4, 2048, 128]',
    t_23: 'bfloat16[512, 512]',
    view_41: 'bfloat16[65536, 512]',
    add_31: 'bfloat16[32, 2048, 512]',
    detach_25: 'float32[32, 2048, 1]',
    t_24: 'bfloat16[512, 2048]',
    view_42: 'bfloat16[65536, 512]',
    detach_26: 'bfloat16[32, 2048, 2048]',
    _to_copy_31: 'float32[32, 2048, 2048]',
    t_25: 'bfloat16[2048, 512]',
    view_43: 'bfloat16[65536, 2048]',
    add_32: 'bfloat16[32, 2048, 512]',
    select_8: 'float32[]',
    select_9: 'float32[]',
    add_33: 'bfloat16[32, 2048, 512]',
    detach_27: 'float32[32, 2048, 1]',
    t_26: 'bfloat16[512, 512]',
    view_44: 'bfloat16[65536, 512]',
    t_27: 'bfloat16[512, 512]',
    view_46: 'bfloat16[65536, 512]',
    t_28: 'bfloat16[512, 512]',
    view_48: 'bfloat16[65536, 512]',
    neg_8: 'bfloat16[1, 2048, 1, 64]',
    cat_8: 'bfloat16[32, 2048, 4, 128]',
    neg_9: 'bfloat16[1, 2048, 1, 64]',
    cat_9: 'bfloat16[32, 2048, 4, 128]',
    detach_28: 'float32[32, 2048, 4, 1]',
    detach_29: 'float32[32, 2048, 4, 1]',
    transpose_16: 'bfloat16[32, 4, 2048, 128]',
    transpose_17: 'bfloat16[32, 4, 2048, 128]',
    transpose_18: 'bfloat16[32, 4, 2048, 128]',
    where_3: 'bfloat16[2048, 2048]',
    getitem_77: 'float32[32, 4, 2048, 1]',
    getitem_82: 'int64[]',
    getitem_83: 'int64[]',
    detach_30: 'bfloat16[32, 4, 2048, 128]',
    t_29: 'bfloat16[512, 512]',
    view_51: 'bfloat16[65536, 512]',
    add_39: 'bfloat16[32, 2048, 512]',
    detach_31: 'float32[32, 2048, 1]',
    t_30: 'bfloat16[512, 2048]',
    view_52: 'bfloat16[65536, 512]',
    detach_32: 'bfloat16[32, 2048, 2048]',
    _to_copy_39: 'float32[32, 2048, 2048]',
    t_31: 'bfloat16[2048, 512]',
    view_53: 'bfloat16[65536, 2048]',
    add_40: 'bfloat16[32, 2048, 512]',
    select_10: 'float32[]',
    select_11: 'float32[]',
    add_41: 'bfloat16[32, 2048, 512]',
    detach_33: 'float32[32, 2048, 1]',
    t_32: 'bfloat16[512, 512]',
    view_54: 'bfloat16[65536, 512]',
    t_33: 'bfloat16[512, 512]',
    view_56: 'bfloat16[65536, 512]',
    t_34: 'bfloat16[512, 512]',
    view_58: 'bfloat16[65536, 512]',
    view_60: 'bfloat16[32, 2048, 4, 128]',
    t_35: 'bfloat16[32, 4]',
    view_61: 'bfloat16[65536, 32]',
    detach_34: 'bfloat16[32, 2048, 4]',
    unsqueeze_10: 'bfloat16[32, 2048, 4, 1]',
    neg_10: 'bfloat16[1, 2048, 1, 64]',
    cat_10: 'bfloat16[32, 2048, 4, 128]',
    neg_11: 'bfloat16[1, 2048, 1, 64]',
    cat_11: 'bfloat16[32, 2048, 4, 128]',
    detach_35: 'float32[32, 2048, 4, 1]',
    detach_36: 'float32[32, 2048, 4, 1]',
    transpose_20: 'bfloat16[32, 4, 2048, 128]',
    transpose_21: 'bfloat16[32, 4, 2048, 128]',
    transpose_22: 'bfloat16[32, 4, 2048, 128]',
    where_4: 'bfloat16[2048, 2048]',
    getitem_94: 'float32[32, 4, 2048, 1]',
    getitem_99: 'int64[]',
    getitem_100: 'int64[]',
    detach_37: 'bfloat16[32, 4, 2048, 128]',
    t_36: 'bfloat16[512, 512]',
    view_63: 'bfloat16[65536, 512]',
    add_48: 'bfloat16[32, 2048, 512]',
    detach_38: 'float32[32, 2048, 1]',
    t_37: 'bfloat16[512, 2048]',
    view_64: 'bfloat16[65536, 512]',
    detach_39: 'bfloat16[32, 2048, 2048]',
    _to_copy_48: 'float32[32, 2048, 2048]',
    t_38: 'bfloat16[2048, 512]',
    view_65: 'bfloat16[65536, 2048]',
    add_49: 'bfloat16[32, 2048, 512]',
    select_12: 'float32[]',
    select_13: 'float32[]',
    add_50: 'bfloat16[32, 2048, 512]',
    detach_40: 'float32[32, 2048, 1]',
    t_39: 'bfloat16[512, 512]',
    view_66: 'bfloat16[65536, 512]',
    t_40: 'bfloat16[512, 512]',
    view_68: 'bfloat16[65536, 512]',
    t_41: 'bfloat16[512, 512]',
    view_70: 'bfloat16[65536, 512]',
    neg_12: 'bfloat16[1, 2048, 1, 64]',
    cat_12: 'bfloat16[32, 2048, 4, 128]',
    neg_13: 'bfloat16[1, 2048, 1, 64]',
    cat_13: 'bfloat16[32, 2048, 4, 128]',
    detach_41: 'float32[32, 2048, 4, 1]',
    detach_42: 'float32[32, 2048, 4, 1]',
    transpose_24: 'bfloat16[32, 4, 2048, 128]',
    transpose_25: 'bfloat16[32, 4, 2048, 128]',
    transpose_26: 'bfloat16[32, 4, 2048, 128]',
    where_5: 'bfloat16[2048, 2048]',
    getitem_111: 'float32[32, 4, 2048, 1]',
    getitem_116: 'int64[]',
    getitem_117: 'int64[]',
    detach_43: 'bfloat16[32, 4, 2048, 128]',
    t_42: 'bfloat16[512, 512]',
    view_73: 'bfloat16[65536, 512]',
    add_56: 'bfloat16[32, 2048, 512]',
    detach_44: 'float32[32, 2048, 1]',
    t_43: 'bfloat16[512, 2048]',
    view_74: 'bfloat16[65536, 512]',
    detach_45: 'bfloat16[32, 2048, 2048]',
    _to_copy_56: 'float32[32, 2048, 2048]',
    t_44: 'bfloat16[2048, 512]',
    view_75: 'bfloat16[65536, 2048]',
    add_57: 'bfloat16[32, 2048, 512]',
    select_14: 'float32[]',
    select_15: 'float32[]',
    add_58: 'bfloat16[32, 2048, 512]',
    detach_46: 'float32[32, 2048, 1]',
    t_45: 'bfloat16[512, 512]',
    view_76: 'bfloat16[65536, 512]',
    t_46: 'bfloat16[512, 512]',
    view_78: 'bfloat16[65536, 512]',
    t_47: 'bfloat16[512, 512]',
    view_80: 'bfloat16[65536, 512]',
    view_82: 'bfloat16[32, 2048, 4, 128]',
    t_48: 'bfloat16[32, 4]',
    view_83: 'bfloat16[65536, 32]',
    detach_47: 'bfloat16[32, 2048, 4]',
    unsqueeze_15: 'bfloat16[32, 2048, 4, 1]',
    neg_14: 'bfloat16[1, 2048, 1, 64]',
    cat_14: 'bfloat16[32, 2048, 4, 128]',
    neg_15: 'bfloat16[1, 2048, 1, 64]',
    cat_15: 'bfloat16[32, 2048, 4, 128]',
    detach_48: 'float32[32, 2048, 4, 1]',
    detach_49: 'float32[32, 2048, 4, 1]',
    transpose_28: 'bfloat16[32, 4, 2048, 128]',
    transpose_29: 'bfloat16[32, 4, 2048, 128]',
    transpose_30: 'bfloat16[32, 4, 2048, 128]',
    getitem_128: 'float32[32, 4, 2048, 1]',
    getitem_133: 'int64[]',
    getitem_134: 'int64[]',
    detach_50: 'bfloat16[32, 4, 2048, 128]',
    t_49: 'bfloat16[512, 512]',
    view_85: 'bfloat16[65536, 512]',
    add_64: 'bfloat16[32, 2048, 512]',
    detach_51: 'float32[32, 2048, 1]',
    t_50: 'bfloat16[512, 2048]',
    view_86: 'bfloat16[65536, 512]',
    detach_52: 'bfloat16[32, 2048, 2048]',
    _to_copy_65: 'float32[32, 2048, 2048]',
    t_51: 'bfloat16[2048, 512]',
    view_87: 'bfloat16[65536, 2048]',
    add_65: 'bfloat16[32, 2048, 512]',
    detach_53: 'float32[32, 2048, 1]',
    t_52: 'bfloat16[512, 8192]',
    view_88: 'bfloat16[65536, 512]',
    detach_54: 'float32[32, 2048, 8192]',
    view_90: 'int64[65536]',
    _log_softmax: 'float32[65536, 8192]',
    detach_55: 'float32[65536, 8192]',
    getitem_141: 'float32[]',
    tangents_1: 'float32[]',
):

    # /.autoresearch_repo/train.py:283
    # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
    grad_cross_entropy_nll_loss_backward: 'float32[65536, 8192]' = aten.nll_loss_backward(tangents_1, _log_softmax, view_90, None, 1, -1, getitem_141)  # strides=(8192, 1), contiguous=True, view=False
    grad_cross_entropy_detach: 'float32[65536, 8192]' = aten.detach(detach_55)  # strides=(8192, 1), contiguous=True, view=True
    grad_cross_entropy__log_softmax_backward_data: 'float32[65536, 8192]' = aten._log_softmax_backward_data(grad_cross_entropy_nll_loss_backward, grad_cross_entropy_detach, 1, torch.float32)  # strides=(8192, 1), contiguous=True, view=False
    grad_view_36_view: 'float32[32, 2048, 8192]' = aten.view(grad_cross_entropy__log_softmax_backward_data, [32, 2048, 8192])  # strides=(16777216, 8192, 1), contiguous=True, view=True

    # /.autoresearch_repo/train.py:279-280
    # FUSED: softcap backward: mul(15) + tanh_backward + div(15) + to_bf16
    grad_float_1__to_copy: 'bfloat16[32, 2048, 8192]' = triton_softcap_bwd(grad_view_36_view, detach_54, softcap=15)

    # ════════════════════════════════════════════════════════════════
    # self.lm_head
    # ════════════════════════════════════════════════════════════════

    # grad of self.lm_head (Linear) → d_loss/d_lm_head
    # /.autoresearch_repo/train.py:278
    # logits = self.lm_head(x)
    grad_lm_head_view: 'bfloat16[65536, 8192]' = aten.view(grad_float_1__to_copy, [65536, 8192])  # strides=(8192, 1), contiguous=True, view=True
    grad_lm_head_t: 'bfloat16[8192, 65536]' = aten.t(grad_lm_head_view)  # strides=(1, 8192), contiguous=False, view=True
    grad_lm_head_mm: 'bfloat16[8192, 512]' = aten.mm(grad_lm_head_t, view_88)  # strides=(512, 1), contiguous=True, view=False
    grad_lm_head_t_1: 'bfloat16[512, 8192]' = aten.t(grad_lm_head_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_lm_head_t_2: 'bfloat16[8192, 512]' = aten.t(t_52)  # strides=(512, 1), contiguous=True, view=True
    grad_lm_head_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_lm_head_view, grad_lm_head_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_lm_head_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_lm_head_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_lm_head_t_3: 'bfloat16[8192, 512]' = aten.t(grad_lm_head_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_lm_head__to_copy: 'float32[8192, 512]' = aten._to_copy(grad_lm_head_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_rms_norm_33_detach: 'float32[32, 2048, 1]' = aten.detach(detach_53)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_rms_norm_33__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_lm_head_view_1, add_65, [512], grad_rms_norm_33_detach, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_rms_norm_33_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_rms_norm_33__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.7
    # ════════════════════════════════════════════════════════════════

    # grad of self.transformer.h.7.mlp.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    grad_h7_mlp_c_proj_view: 'bfloat16[65536, 512]' = aten.view(grad_rms_norm_33_getitem, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h7_mlp_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h7_mlp_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h7_mlp_c_proj_mm: 'bfloat16[512, 2048]' = aten.mm(grad_h7_mlp_c_proj_t, view_87)  # strides=(2048, 1), contiguous=True, view=False
    grad_h7_mlp_c_proj_t_1: 'bfloat16[2048, 512]' = aten.t(grad_h7_mlp_c_proj_mm)  # strides=(1, 2048), contiguous=False, view=True
    grad_h7_mlp_c_proj_t_2: 'bfloat16[512, 2048]' = aten.t(t_51)  # strides=(2048, 1), contiguous=True, view=True
    grad_h7_mlp_c_proj_mm_1: 'bfloat16[65536, 2048]' = aten.mm(grad_h7_mlp_c_proj_view, grad_h7_mlp_c_proj_t_2)  # strides=(2048, 1), contiguous=True, view=False
    grad_h7_mlp_c_proj_view_1: 'bfloat16[32, 2048, 2048]' = aten.view(grad_h7_mlp_c_proj_mm_1, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=True
    grad_h7_mlp_c_proj_t_3: 'bfloat16[512, 2048]' = aten.t(grad_h7_mlp_c_proj_t_1)  # strides=(2048, 1), contiguous=True, view=True
    grad_h7_mlp_c_proj__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(grad_h7_mlp_c_proj_view_1, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(4194304, 2048, 1), contiguous=True, view=False
    grad_h7_mlp_c_proj__to_copy_1: 'float32[512, 2048]' = aten._to_copy(grad_h7_mlp_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(2048, 1), contiguous=True, view=False

    # grad of self.transformer.h.7.mlp (MLP) → d_loss/d_mlp
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    # FUSED: squared_relu backward (pow+mul+mul+to_copy+threshold_backward → single Triton kernel)
    grad_h7_mlp_threshold_backward: 'bfloat16[32, 2048, 2048]' = triton_squared_relu_bwd(grad_h7_mlp_c_proj__to_copy, _to_copy_65, detach_52)

    # grad of self.transformer.h.7.mlp.c_fc (Linear) → d_loss/d_c_fc
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    grad_h7_mlp_c_fc_view: 'bfloat16[65536, 2048]' = aten.view(grad_h7_mlp_threshold_backward, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    grad_h7_mlp_c_fc_t: 'bfloat16[2048, 65536]' = aten.t(grad_h7_mlp_c_fc_view)  # strides=(1, 2048), contiguous=False, view=True
    grad_h7_mlp_c_fc_mm: 'bfloat16[2048, 512]' = aten.mm(grad_h7_mlp_c_fc_t, view_86)  # strides=(512, 1), contiguous=True, view=False
    grad_h7_mlp_c_fc_t_1: 'bfloat16[512, 2048]' = aten.t(grad_h7_mlp_c_fc_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h7_mlp_c_fc_t_2: 'bfloat16[2048, 512]' = aten.t(t_50)  # strides=(512, 1), contiguous=True, view=True
    grad_h7_mlp_c_fc_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h7_mlp_c_fc_view, grad_h7_mlp_c_fc_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h7_mlp_c_fc_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h7_mlp_c_fc_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h7_mlp_c_fc_t_3: 'bfloat16[2048, 512]' = aten.t(grad_h7_mlp_c_fc_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h7_mlp_c_fc__to_copy: 'float32[2048, 512]' = aten._to_copy(grad_h7_mlp_c_fc_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.7 (Block) → d_loss/d_7
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h7_detach: 'float32[32, 2048, 1]' = aten.detach(detach_51)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h7__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h7_mlp_c_fc_view_1, add_64, [512], grad_h7_detach, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h7_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h7__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_66: 'bfloat16[32, 2048, 512]' = cuda_add(grad_rms_norm_33_getitem, grad_h7_getitem)  # FUSED: vectorized CUDA bf16 add

    # grad of self.transformer.h.7.attn.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    grad_h7_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(add_66, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h7_attn_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h7_attn_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h7_attn_c_proj_mm: 'bfloat16[512, 512]' = aten.mm(grad_h7_attn_c_proj_t, view_85)  # strides=(512, 1), contiguous=True, view=False
    grad_h7_attn_c_proj_t_1: 'bfloat16[512, 512]' = aten.t(grad_h7_attn_c_proj_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h7_attn_c_proj_t_2: 'bfloat16[512, 512]' = aten.t(t_49)  # strides=(512, 1), contiguous=True, view=True
    grad_h7_attn_c_proj_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h7_attn_c_proj_view, grad_h7_attn_c_proj_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h7_attn_c_proj_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h7_attn_c_proj_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h7_attn_c_proj_t_3: 'bfloat16[512, 512]' = aten.t(grad_h7_attn_c_proj_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h7_attn_c_proj__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h7_attn_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.7.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:89
    # y = y.contiguous().view(B, T, -1)
    grad_h7_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(grad_h7_attn_c_proj_view_1, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h7_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(grad_h7_attn_view, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h7_attn_detach: 'bfloat16[32, 4, 2048, 128]' = aten.detach(detach_50)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h7_attn__scaled_dot_product_cudnn_attention_backward = aten._scaled_dot_product_cudnn_attention_backward(grad_h7_attn_transpose, transpose_28, transpose_29, transpose_30, grad_h7_attn_detach, getitem_128, getitem_133, getitem_134, None, None, None, 2048, 2048, 0.0, True)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(1048576, 128, 512, 1), contiguous=False; out2: strides=(1048576, 128, 512, 1), contiguous=False, view=False
    grad_h7_attn_getitem: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h7_attn__scaled_dot_product_cudnn_attention_backward, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h7_attn_getitem_1: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h7_attn__scaled_dot_product_cudnn_attention_backward, 1)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h7_attn_getitem_2: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h7_attn__scaled_dot_product_cudnn_attention_backward, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h7_attn_transpose_1: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h7_attn_getitem_2, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h7_attn_transpose_2: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h7_attn_getitem_1, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h7_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h7_attn_getitem, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h7_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(detach_49)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h7_attn__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h7_attn_transpose_2, cat_15, [128], grad_h7_attn_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h7_attn_getitem_3: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h7_attn__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h7_attn_detach_2: 'float32[32, 2048, 4, 1]' = aten.detach(detach_48)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h7_attn__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(grad_h7_attn_transpose_3, cat_14, [128], grad_h7_attn_detach_2, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h7_attn_getitem_4: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h7_attn__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h7_attn_slice: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h7_attn_getitem_3, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h7_attn_slice_1: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h7_attn_getitem_3, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h7_attn_mul: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h7_attn_slice_1, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h7_attn_mul_1: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h7_attn_slice_1, neg_15)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h7_attn_mul_2: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h7_attn_slice, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_67: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h7_attn_mul, grad_h7_attn_mul_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h7_attn_mul_3: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h7_attn_slice, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_68: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h7_attn_mul_1, grad_h7_attn_mul_3)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h7_attn_slice_backward: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_67, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h7_attn_slice_backward_1: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_68, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_69: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h7_attn_slice_backward, grad_h7_attn_slice_backward_1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h7_attn_slice_2: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h7_attn_getitem_4, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h7_attn_slice_3: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h7_attn_getitem_4, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h7_attn_mul_4: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h7_attn_slice_3, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h7_attn_mul_5: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h7_attn_slice_3, neg_14)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h7_attn_mul_6: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h7_attn_slice_2, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_70: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h7_attn_mul_4, grad_h7_attn_mul_6)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h7_attn_mul_7: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h7_attn_slice_2, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_71: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h7_attn_mul_5, grad_h7_attn_mul_7)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h7_attn_slice_backward_2: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_70, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h7_attn_slice_backward_3: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_71, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_72: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h7_attn_slice_backward_2, grad_h7_attn_slice_backward_3)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h7_attn_mul_8: 'bfloat16[32, 2048, 4, 128]' = aten.mul.Tensor(grad_h7_attn_transpose_1, unsqueeze_15)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h7_attn_mul_9: 'bfloat16[32, 2048, 4, 128]' = aten.mul.Tensor(grad_h7_attn_transpose_1, view_82)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h7_attn_sum: 'float32[32, 2048, 4, 1]' = aten.sum.dim_IntList(grad_h7_attn_mul_9, [3], True, dtype=torch.float32)  # strides=(8192, 4, 1, 1), contiguous=True, view=False
    grad_h7_attn__to_copy: 'bfloat16[32, 2048, 4, 1]' = aten._to_copy(grad_h7_attn_sum, dtype=torch.bfloat16)  # strides=(8192, 4, 1, 1), contiguous=True, view=False
    grad_h7_attn_squeeze: 'bfloat16[32, 2048, 4]' = aten.squeeze.dim(grad_h7_attn__to_copy, -1)  # strides=(8192, 4, 1), contiguous=True, view=True
    grad_h7_attn_mul_10: 'bfloat16[32, 2048, 4]' = aten.mul.Tensor(grad_h7_attn_squeeze, 2)  # strides=(8192, 4, 1), contiguous=True, view=False
    grad_h7_attn_detach_3: 'bfloat16[32, 2048, 4]' = aten.detach(detach_47)  # strides=(8192, 4, 1), contiguous=True, view=True
    grad_h7_attn_sigmoid_backward: 'bfloat16[32, 2048, 4]' = aten.sigmoid_backward(grad_h7_attn_mul_10, grad_h7_attn_detach_3)  # strides=(8192, 4, 1), contiguous=True, view=False

    # grad of self.transformer.h.7.attn.ve_gate (Linear) → d_loss/d_ve_gate
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    grad_h7_attn_ve_gate_view: 'bfloat16[65536, 4]' = aten.view(grad_h7_attn_sigmoid_backward, [65536, 4])  # strides=(4, 1), contiguous=True, view=True
    grad_h7_attn_ve_gate_t: 'bfloat16[4, 65536]' = aten.t(grad_h7_attn_ve_gate_view)  # strides=(1, 4), contiguous=False, view=True
    grad_h7_attn_ve_gate_mm: 'bfloat16[4, 32]' = aten.mm(grad_h7_attn_ve_gate_t, view_83)  # strides=(32, 1), contiguous=True, view=False
    grad_h7_attn_ve_gate_t_1: 'bfloat16[32, 4]' = aten.t(grad_h7_attn_ve_gate_mm)  # strides=(1, 32), contiguous=False, view=True
    grad_h7_attn_ve_gate_t_2: 'bfloat16[4, 32]' = aten.t(t_48)  # strides=(32, 1), contiguous=True, view=True
    grad_h7_attn_ve_gate_mm_1: 'bfloat16[65536, 32]' = aten.mm(grad_h7_attn_ve_gate_view, grad_h7_attn_ve_gate_t_2)  # strides=(32, 1), contiguous=True, view=False
    grad_h7_attn_ve_gate_view_1: 'bfloat16[32, 2048, 32]' = aten.view(grad_h7_attn_ve_gate_mm_1, [32, 2048, 32])  # strides=(65536, 32, 1), contiguous=True, view=True
    grad_h7_attn_ve_gate_t_3: 'bfloat16[4, 32]' = aten.t(grad_h7_attn_ve_gate_t_1)  # strides=(32, 1), contiguous=True, view=True
    grad_h7_attn_ve_gate__to_copy: 'float32[4, 32]' = aten._to_copy(grad_h7_attn_ve_gate_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(32, 1), contiguous=True, view=False

    # grad of self.transformer.h.7.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    grad_h7_attn_slice_backward_4: 'bfloat16[32, 2048, 512]' = aten.slice_backward(grad_h7_attn_ve_gate_view_1, [32, 2048, 512], 2, 0, 32, 1)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h7_attn_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h7_attn_mul_8, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h7_attn_view_2: 'bfloat16[32, 2048, 512]' = aten.view(grad_h7_attn_transpose_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.7.attn.c_v (Linear) → d_loss/d_c_v
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h7_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(grad_h7_attn_view_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h7_attn_c_v_t: 'bfloat16[512, 65536]' = aten.t(grad_h7_attn_c_v_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h7_attn_c_v_mm: 'bfloat16[512, 512]' = aten.mm(grad_h7_attn_c_v_t, view_80)  # strides=(512, 1), contiguous=True, view=False
    grad_h7_attn_c_v_t_1: 'bfloat16[512, 512]' = aten.t(grad_h7_attn_c_v_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h7_attn_c_v_t_2: 'bfloat16[512, 512]' = aten.t(t_47)  # strides=(512, 1), contiguous=True, view=True
    grad_h7_attn_c_v_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h7_attn_c_v_view, grad_h7_attn_c_v_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h7_attn_c_v_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h7_attn_c_v_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_73: 'bfloat16[32, 2048, 512]' = cuda_add(grad_h7_attn_slice_backward_4, grad_h7_attn_c_v_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h7_attn_c_v_t_3: 'bfloat16[512, 512]' = aten.t(grad_h7_attn_c_v_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h7_attn_c_v__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h7_attn_c_v_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.7.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h7_attn_view_3: 'bfloat16[32, 2048, 512]' = aten.view(add_69, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.7.attn.c_k (Linear) → d_loss/d_c_k
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h7_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(grad_h7_attn_view_3, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h7_attn_c_k_t: 'bfloat16[512, 65536]' = aten.t(grad_h7_attn_c_k_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h7_attn_c_k_mm: 'bfloat16[512, 512]' = aten.mm(grad_h7_attn_c_k_t, view_78)  # strides=(512, 1), contiguous=True, view=False
    grad_h7_attn_c_k_t_1: 'bfloat16[512, 512]' = aten.t(grad_h7_attn_c_k_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h7_attn_c_k_t_2: 'bfloat16[512, 512]' = aten.t(t_46)  # strides=(512, 1), contiguous=True, view=True
    grad_h7_attn_c_k_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h7_attn_c_k_view, grad_h7_attn_c_k_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h7_attn_c_k_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h7_attn_c_k_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_74: 'bfloat16[32, 2048, 512]' = cuda_add(add_73, grad_h7_attn_c_k_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h7_attn_c_k_t_3: 'bfloat16[512, 512]' = aten.t(grad_h7_attn_c_k_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h7_attn_c_k__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h7_attn_c_k_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.7.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h7_attn_view_4: 'bfloat16[32, 2048, 512]' = aten.view(add_72, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.7.attn.c_q (Linear) → d_loss/d_c_q
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h7_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(grad_h7_attn_view_4, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h7_attn_c_q_t: 'bfloat16[512, 65536]' = aten.t(grad_h7_attn_c_q_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h7_attn_c_q_mm: 'bfloat16[512, 512]' = aten.mm(grad_h7_attn_c_q_t, view_76)  # strides=(512, 1), contiguous=True, view=False
    grad_h7_attn_c_q_t_1: 'bfloat16[512, 512]' = aten.t(grad_h7_attn_c_q_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h7_attn_c_q_t_2: 'bfloat16[512, 512]' = aten.t(t_45)  # strides=(512, 1), contiguous=True, view=True
    grad_h7_attn_c_q_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h7_attn_c_q_view, grad_h7_attn_c_q_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h7_attn_c_q_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h7_attn_c_q_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_75: 'bfloat16[32, 2048, 512]' = cuda_add(add_74, grad_h7_attn_c_q_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h7_attn_c_q_t_3: 'bfloat16[512, 512]' = aten.t(grad_h7_attn_c_q_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h7_attn_c_q__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h7_attn_c_q_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.7 (Block) → d_loss/d_7
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h7_detach_1: 'float32[32, 2048, 1]' = aten.detach(detach_46)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h7__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(add_75, add_58, [512], grad_h7_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h7_getitem_1: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h7__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_76: 'bfloat16[32, 2048, 512]' = cuda_add(add_66, grad_h7_getitem_1)  # FUSED: vectorized CUDA bf16 add

    # ════════════════════════════════════════════════════════════════
    # self.value_embeds.7
    # ════════════════════════════════════════════════════════════════

    # grad of self.value_embeds.7 (Embedding) → d_loss/d_7
    # /.autoresearch_repo/train.py:273
    # ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
    grad_value_embeds7_embedding_dense_backward: 'bfloat16[8192, 512]' = aten.embedding_dense_backward(grad_h7_attn_view_1, input__32__2048, 8192, -1, False)  # strides=(512, 1), contiguous=True, view=False

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    grad_mul_77_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_76, select_15)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_77_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_76, getitem)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_77_sum: 'float32[]' = aten.sum(grad_mul_77_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    grad_getitem_48_select_backward: 'float32[8]' = aten.select_backward(grad_mul_77_sum, [8], 0, 7)  # strides=(1,), contiguous=True, view=False
    grad_mul_76_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_76, select_14)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_76_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_76, add_57)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_76_sum: 'float32[]' = aten.sum(grad_mul_76_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    grad_getitem_47_select_backward: 'float32[8]' = aten.select_backward(grad_mul_76_sum, [8], 0, 7)  # strides=(1,), contiguous=True, view=False

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.6
    # ════════════════════════════════════════════════════════════════

    # grad of self.transformer.h.6.mlp.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    grad_h6_mlp_c_proj_view: 'bfloat16[65536, 512]' = aten.view(grad_mul_76_mul, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h6_mlp_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h6_mlp_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h6_mlp_c_proj_mm: 'bfloat16[512, 2048]' = aten.mm(grad_h6_mlp_c_proj_t, view_75)  # strides=(2048, 1), contiguous=True, view=False
    grad_h6_mlp_c_proj_t_1: 'bfloat16[2048, 512]' = aten.t(grad_h6_mlp_c_proj_mm)  # strides=(1, 2048), contiguous=False, view=True
    grad_h6_mlp_c_proj_t_2: 'bfloat16[512, 2048]' = aten.t(t_44)  # strides=(2048, 1), contiguous=True, view=True
    grad_h6_mlp_c_proj_mm_1: 'bfloat16[65536, 2048]' = aten.mm(grad_h6_mlp_c_proj_view, grad_h6_mlp_c_proj_t_2)  # strides=(2048, 1), contiguous=True, view=False
    grad_h6_mlp_c_proj_view_1: 'bfloat16[32, 2048, 2048]' = aten.view(grad_h6_mlp_c_proj_mm_1, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=True
    grad_h6_mlp_c_proj_t_3: 'bfloat16[512, 2048]' = aten.t(grad_h6_mlp_c_proj_t_1)  # strides=(2048, 1), contiguous=True, view=True
    grad_h6_mlp_c_proj__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(grad_h6_mlp_c_proj_view_1, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(4194304, 2048, 1), contiguous=True, view=False
    grad_h6_mlp_c_proj__to_copy_1: 'float32[512, 2048]' = aten._to_copy(grad_h6_mlp_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(2048, 1), contiguous=True, view=False

    # grad of self.transformer.h.6.mlp (MLP) → d_loss/d_mlp
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    # FUSED: squared_relu backward
    grad_h6_mlp_threshold_backward: 'bfloat16[32, 2048, 2048]' = triton_squared_relu_bwd(grad_h6_mlp_c_proj__to_copy, _to_copy_56, detach_45)

    # grad of self.transformer.h.6.mlp.c_fc (Linear) → d_loss/d_c_fc
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    grad_h6_mlp_c_fc_view: 'bfloat16[65536, 2048]' = aten.view(grad_h6_mlp_threshold_backward, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    grad_h6_mlp_c_fc_t: 'bfloat16[2048, 65536]' = aten.t(grad_h6_mlp_c_fc_view)  # strides=(1, 2048), contiguous=False, view=True
    grad_h6_mlp_c_fc_mm: 'bfloat16[2048, 512]' = aten.mm(grad_h6_mlp_c_fc_t, view_74)  # strides=(512, 1), contiguous=True, view=False
    grad_h6_mlp_c_fc_t_1: 'bfloat16[512, 2048]' = aten.t(grad_h6_mlp_c_fc_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h6_mlp_c_fc_t_2: 'bfloat16[2048, 512]' = aten.t(t_43)  # strides=(512, 1), contiguous=True, view=True
    grad_h6_mlp_c_fc_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h6_mlp_c_fc_view, grad_h6_mlp_c_fc_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h6_mlp_c_fc_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h6_mlp_c_fc_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h6_mlp_c_fc_t_3: 'bfloat16[2048, 512]' = aten.t(grad_h6_mlp_c_fc_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h6_mlp_c_fc__to_copy: 'float32[2048, 512]' = aten._to_copy(grad_h6_mlp_c_fc_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.6 (Block) → d_loss/d_6
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h6_detach: 'float32[32, 2048, 1]' = aten.detach(detach_44)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h6__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h6_mlp_c_fc_view_1, add_56, [512], grad_h6_detach, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h6_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h6__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_77: 'bfloat16[32, 2048, 512]' = cuda_add(grad_mul_76_mul, grad_h6_getitem)  # FUSED: vectorized CUDA bf16 add

    # grad of self.transformer.h.6.attn.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    grad_h6_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(add_77, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h6_attn_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h6_attn_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h6_attn_c_proj_mm: 'bfloat16[512, 512]' = aten.mm(grad_h6_attn_c_proj_t, view_73)  # strides=(512, 1), contiguous=True, view=False
    grad_h6_attn_c_proj_t_1: 'bfloat16[512, 512]' = aten.t(grad_h6_attn_c_proj_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h6_attn_c_proj_t_2: 'bfloat16[512, 512]' = aten.t(t_42)  # strides=(512, 1), contiguous=True, view=True
    grad_h6_attn_c_proj_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h6_attn_c_proj_view, grad_h6_attn_c_proj_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h6_attn_c_proj_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h6_attn_c_proj_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h6_attn_c_proj_t_3: 'bfloat16[512, 512]' = aten.t(grad_h6_attn_c_proj_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h6_attn_c_proj__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h6_attn_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.6.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:89
    # y = y.contiguous().view(B, T, -1)
    grad_h6_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(grad_h6_attn_c_proj_view_1, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h6_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(grad_h6_attn_view, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h6_attn_detach: 'bfloat16[32, 4, 2048, 128]' = aten.detach(detach_43)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h6_attn__scaled_dot_product_cudnn_attention_backward = aten._scaled_dot_product_cudnn_attention_backward(grad_h6_attn_transpose, transpose_24, transpose_25, transpose_26, grad_h6_attn_detach, getitem_111, getitem_116, getitem_117, where_5, None, None, 2048, 2048, 0.0, False)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(1048576, 128, 512, 1), contiguous=False; out2: strides=(1048576, 128, 512, 1), contiguous=False, view=False
    grad_h6_attn_getitem: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h6_attn__scaled_dot_product_cudnn_attention_backward, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h6_attn_getitem_1: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h6_attn__scaled_dot_product_cudnn_attention_backward, 1)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h6_attn_getitem_2: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h6_attn__scaled_dot_product_cudnn_attention_backward, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h6_attn_transpose_1: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h6_attn_getitem_2, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h6_attn_transpose_2: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h6_attn_getitem_1, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h6_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h6_attn_getitem, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h6_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(detach_42)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h6_attn__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h6_attn_transpose_2, cat_13, [128], grad_h6_attn_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h6_attn_getitem_3: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h6_attn__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h6_attn_detach_2: 'float32[32, 2048, 4, 1]' = aten.detach(detach_41)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h6_attn__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(grad_h6_attn_transpose_3, cat_12, [128], grad_h6_attn_detach_2, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h6_attn_getitem_4: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h6_attn__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h6_attn_slice: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h6_attn_getitem_3, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h6_attn_slice_1: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h6_attn_getitem_3, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h6_attn_mul: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h6_attn_slice_1, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h6_attn_mul_1: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h6_attn_slice_1, neg_13)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h6_attn_mul_2: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h6_attn_slice, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_78: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h6_attn_mul, grad_h6_attn_mul_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h6_attn_mul_3: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h6_attn_slice, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_79: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h6_attn_mul_1, grad_h6_attn_mul_3)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h6_attn_slice_backward: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_78, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h6_attn_slice_backward_1: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_79, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_80: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h6_attn_slice_backward, grad_h6_attn_slice_backward_1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h6_attn_slice_2: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h6_attn_getitem_4, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h6_attn_slice_3: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h6_attn_getitem_4, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h6_attn_mul_4: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h6_attn_slice_3, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h6_attn_mul_5: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h6_attn_slice_3, neg_12)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h6_attn_mul_6: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h6_attn_slice_2, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_81: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h6_attn_mul_4, grad_h6_attn_mul_6)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h6_attn_mul_7: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h6_attn_slice_2, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_82: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h6_attn_mul_5, grad_h6_attn_mul_7)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h6_attn_slice_backward_2: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_81, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h6_attn_slice_backward_3: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_82, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_83: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h6_attn_slice_backward_2, grad_h6_attn_slice_backward_3)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h6_attn_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h6_attn_transpose_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.6.attn.c_v (Linear) → d_loss/d_c_v
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h6_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(grad_h6_attn_view_1, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h6_attn_c_v_t: 'bfloat16[512, 65536]' = aten.t(grad_h6_attn_c_v_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h6_attn_c_v_mm: 'bfloat16[512, 512]' = aten.mm(grad_h6_attn_c_v_t, view_70)  # strides=(512, 1), contiguous=True, view=False
    grad_h6_attn_c_v_t_1: 'bfloat16[512, 512]' = aten.t(grad_h6_attn_c_v_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h6_attn_c_v_t_2: 'bfloat16[512, 512]' = aten.t(t_41)  # strides=(512, 1), contiguous=True, view=True
    grad_h6_attn_c_v_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h6_attn_c_v_view, grad_h6_attn_c_v_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h6_attn_c_v_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h6_attn_c_v_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h6_attn_c_v_t_3: 'bfloat16[512, 512]' = aten.t(grad_h6_attn_c_v_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h6_attn_c_v__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h6_attn_c_v_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.6.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h6_attn_view_2: 'bfloat16[32, 2048, 512]' = aten.view(add_80, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.6.attn.c_k (Linear) → d_loss/d_c_k
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h6_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(grad_h6_attn_view_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h6_attn_c_k_t: 'bfloat16[512, 65536]' = aten.t(grad_h6_attn_c_k_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h6_attn_c_k_mm: 'bfloat16[512, 512]' = aten.mm(grad_h6_attn_c_k_t, view_68)  # strides=(512, 1), contiguous=True, view=False
    grad_h6_attn_c_k_t_1: 'bfloat16[512, 512]' = aten.t(grad_h6_attn_c_k_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h6_attn_c_k_t_2: 'bfloat16[512, 512]' = aten.t(t_40)  # strides=(512, 1), contiguous=True, view=True
    grad_h6_attn_c_k_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h6_attn_c_k_view, grad_h6_attn_c_k_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h6_attn_c_k_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h6_attn_c_k_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_84: 'bfloat16[32, 2048, 512]' = cuda_add(grad_h6_attn_c_v_view_1, grad_h6_attn_c_k_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h6_attn_c_k_t_3: 'bfloat16[512, 512]' = aten.t(grad_h6_attn_c_k_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h6_attn_c_k__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h6_attn_c_k_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.6.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h6_attn_view_3: 'bfloat16[32, 2048, 512]' = aten.view(add_83, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.6.attn.c_q (Linear) → d_loss/d_c_q
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h6_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(grad_h6_attn_view_3, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h6_attn_c_q_t: 'bfloat16[512, 65536]' = aten.t(grad_h6_attn_c_q_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h6_attn_c_q_mm: 'bfloat16[512, 512]' = aten.mm(grad_h6_attn_c_q_t, view_66)  # strides=(512, 1), contiguous=True, view=False
    grad_h6_attn_c_q_t_1: 'bfloat16[512, 512]' = aten.t(grad_h6_attn_c_q_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h6_attn_c_q_t_2: 'bfloat16[512, 512]' = aten.t(t_39)  # strides=(512, 1), contiguous=True, view=True
    grad_h6_attn_c_q_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h6_attn_c_q_view, grad_h6_attn_c_q_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h6_attn_c_q_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h6_attn_c_q_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_85: 'bfloat16[32, 2048, 512]' = cuda_add(add_84, grad_h6_attn_c_q_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h6_attn_c_q_t_3: 'bfloat16[512, 512]' = aten.t(grad_h6_attn_c_q_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h6_attn_c_q__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h6_attn_c_q_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.6 (Block) → d_loss/d_6
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h6_detach_1: 'float32[32, 2048, 1]' = aten.detach(detach_40)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h6__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(add_85, add_50, [512], grad_h6_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h6_getitem_1: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h6__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_86: 'bfloat16[32, 2048, 512]' = cuda_add(add_77, grad_h6_getitem_1)  # FUSED: vectorized CUDA bf16 add

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    grad_mul_67_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_86, select_13)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_67_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_86, getitem)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_67_sum: 'float32[]' = aten.sum(grad_mul_67_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    add_87: 'bfloat16[32, 2048, 512]' = cuda_add(grad_mul_77_mul, grad_mul_67_mul)  # FUSED: vectorized CUDA bf16 add
    grad_getitem_42_select_backward: 'float32[8]' = aten.select_backward(grad_mul_67_sum, [8], 0, 6)  # strides=(1,), contiguous=True, view=False
    add_88: 'float32[8]' = aten.add.Tensor(grad_getitem_48_select_backward, grad_getitem_42_select_backward)  # strides=(1,), contiguous=True, view=False
    grad_mul_66_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_86, select_12)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_66_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_86, add_49)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_66_sum: 'float32[]' = aten.sum(grad_mul_66_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    grad_getitem_41_select_backward: 'float32[8]' = aten.select_backward(grad_mul_66_sum, [8], 0, 6)  # strides=(1,), contiguous=True, view=False
    add_89: 'float32[8]' = aten.add.Tensor(grad_getitem_47_select_backward, grad_getitem_41_select_backward)  # strides=(1,), contiguous=True, view=False

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.5
    # ════════════════════════════════════════════════════════════════

    # grad of self.transformer.h.5.mlp.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    grad_h5_mlp_c_proj_view: 'bfloat16[65536, 512]' = aten.view(grad_mul_66_mul, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h5_mlp_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h5_mlp_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h5_mlp_c_proj_mm: 'bfloat16[512, 2048]' = aten.mm(grad_h5_mlp_c_proj_t, view_65)  # strides=(2048, 1), contiguous=True, view=False
    grad_h5_mlp_c_proj_t_1: 'bfloat16[2048, 512]' = aten.t(grad_h5_mlp_c_proj_mm)  # strides=(1, 2048), contiguous=False, view=True
    grad_h5_mlp_c_proj_t_2: 'bfloat16[512, 2048]' = aten.t(t_38)  # strides=(2048, 1), contiguous=True, view=True
    grad_h5_mlp_c_proj_mm_1: 'bfloat16[65536, 2048]' = aten.mm(grad_h5_mlp_c_proj_view, grad_h5_mlp_c_proj_t_2)  # strides=(2048, 1), contiguous=True, view=False
    grad_h5_mlp_c_proj_view_1: 'bfloat16[32, 2048, 2048]' = aten.view(grad_h5_mlp_c_proj_mm_1, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=True
    grad_h5_mlp_c_proj_t_3: 'bfloat16[512, 2048]' = aten.t(grad_h5_mlp_c_proj_t_1)  # strides=(2048, 1), contiguous=True, view=True
    grad_h5_mlp_c_proj__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(grad_h5_mlp_c_proj_view_1, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(4194304, 2048, 1), contiguous=True, view=False
    grad_h5_mlp_c_proj__to_copy_1: 'float32[512, 2048]' = aten._to_copy(grad_h5_mlp_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(2048, 1), contiguous=True, view=False

    # grad of self.transformer.h.5.mlp (MLP) → d_loss/d_mlp
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    # FUSED: squared_relu backward
    grad_h5_mlp_threshold_backward: 'bfloat16[32, 2048, 2048]' = triton_squared_relu_bwd(grad_h5_mlp_c_proj__to_copy, _to_copy_48, detach_39)

    # grad of self.transformer.h.5.mlp.c_fc (Linear) → d_loss/d_c_fc
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    grad_h5_mlp_c_fc_view: 'bfloat16[65536, 2048]' = aten.view(grad_h5_mlp_threshold_backward, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    grad_h5_mlp_c_fc_t: 'bfloat16[2048, 65536]' = aten.t(grad_h5_mlp_c_fc_view)  # strides=(1, 2048), contiguous=False, view=True
    grad_h5_mlp_c_fc_mm: 'bfloat16[2048, 512]' = aten.mm(grad_h5_mlp_c_fc_t, view_64)  # strides=(512, 1), contiguous=True, view=False
    grad_h5_mlp_c_fc_t_1: 'bfloat16[512, 2048]' = aten.t(grad_h5_mlp_c_fc_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h5_mlp_c_fc_t_2: 'bfloat16[2048, 512]' = aten.t(t_37)  # strides=(512, 1), contiguous=True, view=True
    grad_h5_mlp_c_fc_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h5_mlp_c_fc_view, grad_h5_mlp_c_fc_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h5_mlp_c_fc_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h5_mlp_c_fc_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h5_mlp_c_fc_t_3: 'bfloat16[2048, 512]' = aten.t(grad_h5_mlp_c_fc_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h5_mlp_c_fc__to_copy: 'float32[2048, 512]' = aten._to_copy(grad_h5_mlp_c_fc_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.5 (Block) → d_loss/d_5
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h5_detach: 'float32[32, 2048, 1]' = aten.detach(detach_38)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h5__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h5_mlp_c_fc_view_1, add_48, [512], grad_h5_detach, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h5_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h5__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_90: 'bfloat16[32, 2048, 512]' = cuda_add(grad_mul_66_mul, grad_h5_getitem)  # FUSED: vectorized CUDA bf16 add

    # grad of self.transformer.h.5.attn.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    grad_h5_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(add_90, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h5_attn_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h5_attn_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h5_attn_c_proj_mm: 'bfloat16[512, 512]' = aten.mm(grad_h5_attn_c_proj_t, view_63)  # strides=(512, 1), contiguous=True, view=False
    grad_h5_attn_c_proj_t_1: 'bfloat16[512, 512]' = aten.t(grad_h5_attn_c_proj_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h5_attn_c_proj_t_2: 'bfloat16[512, 512]' = aten.t(t_36)  # strides=(512, 1), contiguous=True, view=True
    grad_h5_attn_c_proj_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h5_attn_c_proj_view, grad_h5_attn_c_proj_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h5_attn_c_proj_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h5_attn_c_proj_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h5_attn_c_proj_t_3: 'bfloat16[512, 512]' = aten.t(grad_h5_attn_c_proj_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h5_attn_c_proj__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h5_attn_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.5.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:89
    # y = y.contiguous().view(B, T, -1)
    grad_h5_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(grad_h5_attn_c_proj_view_1, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h5_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(grad_h5_attn_view, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h5_attn_detach: 'bfloat16[32, 4, 2048, 128]' = aten.detach(detach_37)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h5_attn__scaled_dot_product_cudnn_attention_backward = aten._scaled_dot_product_cudnn_attention_backward(grad_h5_attn_transpose, transpose_20, transpose_21, transpose_22, grad_h5_attn_detach, getitem_94, getitem_99, getitem_100, where_4, None, None, 2048, 2048, 0.0, False)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(1048576, 128, 512, 1), contiguous=False; out2: strides=(1048576, 128, 512, 1), contiguous=False, view=False
    grad_h5_attn_getitem: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h5_attn__scaled_dot_product_cudnn_attention_backward, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h5_attn_getitem_1: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h5_attn__scaled_dot_product_cudnn_attention_backward, 1)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h5_attn_getitem_2: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h5_attn__scaled_dot_product_cudnn_attention_backward, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h5_attn_transpose_1: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h5_attn_getitem_2, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h5_attn_transpose_2: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h5_attn_getitem_1, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h5_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h5_attn_getitem, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h5_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(detach_36)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h5_attn__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h5_attn_transpose_2, cat_11, [128], grad_h5_attn_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h5_attn_getitem_3: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h5_attn__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h5_attn_detach_2: 'float32[32, 2048, 4, 1]' = aten.detach(detach_35)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h5_attn__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(grad_h5_attn_transpose_3, cat_10, [128], grad_h5_attn_detach_2, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h5_attn_getitem_4: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h5_attn__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h5_attn_slice: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h5_attn_getitem_3, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h5_attn_slice_1: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h5_attn_getitem_3, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h5_attn_mul: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h5_attn_slice_1, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h5_attn_mul_1: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h5_attn_slice_1, neg_11)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h5_attn_mul_2: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h5_attn_slice, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_91: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h5_attn_mul, grad_h5_attn_mul_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h5_attn_mul_3: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h5_attn_slice, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_92: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h5_attn_mul_1, grad_h5_attn_mul_3)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h5_attn_slice_backward: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_91, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h5_attn_slice_backward_1: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_92, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_93: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h5_attn_slice_backward, grad_h5_attn_slice_backward_1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h5_attn_slice_2: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h5_attn_getitem_4, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h5_attn_slice_3: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h5_attn_getitem_4, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h5_attn_mul_4: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h5_attn_slice_3, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h5_attn_mul_5: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h5_attn_slice_3, neg_10)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h5_attn_mul_6: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h5_attn_slice_2, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_94: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h5_attn_mul_4, grad_h5_attn_mul_6)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h5_attn_mul_7: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h5_attn_slice_2, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_95: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h5_attn_mul_5, grad_h5_attn_mul_7)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h5_attn_slice_backward_2: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_94, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h5_attn_slice_backward_3: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_95, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_96: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h5_attn_slice_backward_2, grad_h5_attn_slice_backward_3)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h5_attn_mul_8: 'bfloat16[32, 2048, 4, 128]' = aten.mul.Tensor(grad_h5_attn_transpose_1, unsqueeze_10)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h5_attn_mul_9: 'bfloat16[32, 2048, 4, 128]' = aten.mul.Tensor(grad_h5_attn_transpose_1, view_60)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h5_attn_sum: 'float32[32, 2048, 4, 1]' = aten.sum.dim_IntList(grad_h5_attn_mul_9, [3], True, dtype=torch.float32)  # strides=(8192, 4, 1, 1), contiguous=True, view=False
    grad_h5_attn__to_copy: 'bfloat16[32, 2048, 4, 1]' = aten._to_copy(grad_h5_attn_sum, dtype=torch.bfloat16)  # strides=(8192, 4, 1, 1), contiguous=True, view=False
    grad_h5_attn_squeeze: 'bfloat16[32, 2048, 4]' = aten.squeeze.dim(grad_h5_attn__to_copy, -1)  # strides=(8192, 4, 1), contiguous=True, view=True
    grad_h5_attn_mul_10: 'bfloat16[32, 2048, 4]' = aten.mul.Tensor(grad_h5_attn_squeeze, 2)  # strides=(8192, 4, 1), contiguous=True, view=False
    grad_h5_attn_detach_3: 'bfloat16[32, 2048, 4]' = aten.detach(detach_34)  # strides=(8192, 4, 1), contiguous=True, view=True
    grad_h5_attn_sigmoid_backward: 'bfloat16[32, 2048, 4]' = aten.sigmoid_backward(grad_h5_attn_mul_10, grad_h5_attn_detach_3)  # strides=(8192, 4, 1), contiguous=True, view=False

    # grad of self.transformer.h.5.attn.ve_gate (Linear) → d_loss/d_ve_gate
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    grad_h5_attn_ve_gate_view: 'bfloat16[65536, 4]' = aten.view(grad_h5_attn_sigmoid_backward, [65536, 4])  # strides=(4, 1), contiguous=True, view=True
    grad_h5_attn_ve_gate_t: 'bfloat16[4, 65536]' = aten.t(grad_h5_attn_ve_gate_view)  # strides=(1, 4), contiguous=False, view=True
    grad_h5_attn_ve_gate_mm: 'bfloat16[4, 32]' = aten.mm(grad_h5_attn_ve_gate_t, view_61)  # strides=(32, 1), contiguous=True, view=False
    grad_h5_attn_ve_gate_t_1: 'bfloat16[32, 4]' = aten.t(grad_h5_attn_ve_gate_mm)  # strides=(1, 32), contiguous=False, view=True
    grad_h5_attn_ve_gate_t_2: 'bfloat16[4, 32]' = aten.t(t_35)  # strides=(32, 1), contiguous=True, view=True
    grad_h5_attn_ve_gate_mm_1: 'bfloat16[65536, 32]' = aten.mm(grad_h5_attn_ve_gate_view, grad_h5_attn_ve_gate_t_2)  # strides=(32, 1), contiguous=True, view=False
    grad_h5_attn_ve_gate_view_1: 'bfloat16[32, 2048, 32]' = aten.view(grad_h5_attn_ve_gate_mm_1, [32, 2048, 32])  # strides=(65536, 32, 1), contiguous=True, view=True
    grad_h5_attn_ve_gate_t_3: 'bfloat16[4, 32]' = aten.t(grad_h5_attn_ve_gate_t_1)  # strides=(32, 1), contiguous=True, view=True
    grad_h5_attn_ve_gate__to_copy: 'float32[4, 32]' = aten._to_copy(grad_h5_attn_ve_gate_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(32, 1), contiguous=True, view=False

    # grad of self.transformer.h.5.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    grad_h5_attn_slice_backward_4: 'bfloat16[32, 2048, 512]' = aten.slice_backward(grad_h5_attn_ve_gate_view_1, [32, 2048, 512], 2, 0, 32, 1)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h5_attn_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h5_attn_mul_8, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h5_attn_view_2: 'bfloat16[32, 2048, 512]' = aten.view(grad_h5_attn_transpose_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.5.attn.c_v (Linear) → d_loss/d_c_v
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h5_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(grad_h5_attn_view_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h5_attn_c_v_t: 'bfloat16[512, 65536]' = aten.t(grad_h5_attn_c_v_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h5_attn_c_v_mm: 'bfloat16[512, 512]' = aten.mm(grad_h5_attn_c_v_t, view_58)  # strides=(512, 1), contiguous=True, view=False
    grad_h5_attn_c_v_t_1: 'bfloat16[512, 512]' = aten.t(grad_h5_attn_c_v_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h5_attn_c_v_t_2: 'bfloat16[512, 512]' = aten.t(t_34)  # strides=(512, 1), contiguous=True, view=True
    grad_h5_attn_c_v_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h5_attn_c_v_view, grad_h5_attn_c_v_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h5_attn_c_v_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h5_attn_c_v_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_97: 'bfloat16[32, 2048, 512]' = cuda_add(grad_h5_attn_slice_backward_4, grad_h5_attn_c_v_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h5_attn_c_v_t_3: 'bfloat16[512, 512]' = aten.t(grad_h5_attn_c_v_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h5_attn_c_v__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h5_attn_c_v_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.5.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h5_attn_view_3: 'bfloat16[32, 2048, 512]' = aten.view(add_93, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.5.attn.c_k (Linear) → d_loss/d_c_k
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h5_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(grad_h5_attn_view_3, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h5_attn_c_k_t: 'bfloat16[512, 65536]' = aten.t(grad_h5_attn_c_k_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h5_attn_c_k_mm: 'bfloat16[512, 512]' = aten.mm(grad_h5_attn_c_k_t, view_56)  # strides=(512, 1), contiguous=True, view=False
    grad_h5_attn_c_k_t_1: 'bfloat16[512, 512]' = aten.t(grad_h5_attn_c_k_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h5_attn_c_k_t_2: 'bfloat16[512, 512]' = aten.t(t_33)  # strides=(512, 1), contiguous=True, view=True
    grad_h5_attn_c_k_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h5_attn_c_k_view, grad_h5_attn_c_k_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h5_attn_c_k_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h5_attn_c_k_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_98: 'bfloat16[32, 2048, 512]' = cuda_add(add_97, grad_h5_attn_c_k_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h5_attn_c_k_t_3: 'bfloat16[512, 512]' = aten.t(grad_h5_attn_c_k_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h5_attn_c_k__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h5_attn_c_k_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.5.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h5_attn_view_4: 'bfloat16[32, 2048, 512]' = aten.view(add_96, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.5.attn.c_q (Linear) → d_loss/d_c_q
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h5_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(grad_h5_attn_view_4, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h5_attn_c_q_t: 'bfloat16[512, 65536]' = aten.t(grad_h5_attn_c_q_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h5_attn_c_q_mm: 'bfloat16[512, 512]' = aten.mm(grad_h5_attn_c_q_t, view_54)  # strides=(512, 1), contiguous=True, view=False
    grad_h5_attn_c_q_t_1: 'bfloat16[512, 512]' = aten.t(grad_h5_attn_c_q_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h5_attn_c_q_t_2: 'bfloat16[512, 512]' = aten.t(t_32)  # strides=(512, 1), contiguous=True, view=True
    grad_h5_attn_c_q_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h5_attn_c_q_view, grad_h5_attn_c_q_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h5_attn_c_q_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h5_attn_c_q_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_99: 'bfloat16[32, 2048, 512]' = cuda_add(add_98, grad_h5_attn_c_q_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h5_attn_c_q_t_3: 'bfloat16[512, 512]' = aten.t(grad_h5_attn_c_q_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h5_attn_c_q__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h5_attn_c_q_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.5 (Block) → d_loss/d_5
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h5_detach_1: 'float32[32, 2048, 1]' = aten.detach(detach_33)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h5__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(add_99, add_41, [512], grad_h5_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h5_getitem_1: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h5__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_100: 'bfloat16[32, 2048, 512]' = cuda_add(add_90, grad_h5_getitem_1)  # FUSED: vectorized CUDA bf16 add

    # ════════════════════════════════════════════════════════════════
    # self.value_embeds.5
    # ════════════════════════════════════════════════════════════════

    # grad of self.value_embeds.5 (Embedding) → d_loss/d_5
    # /.autoresearch_repo/train.py:273
    # ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
    grad_value_embeds5_embedding_dense_backward: 'bfloat16[8192, 512]' = aten.embedding_dense_backward(grad_h5_attn_view_1, input__32__2048, 8192, -1, False)  # strides=(512, 1), contiguous=True, view=False

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    grad_mul_55_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_100, select_11)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_55_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_100, getitem)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_55_sum: 'float32[]' = aten.sum(grad_mul_55_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    add_101: 'bfloat16[32, 2048, 512]' = cuda_add(add_87, grad_mul_55_mul)  # FUSED: vectorized CUDA bf16 add
    grad_getitem_35_select_backward: 'float32[8]' = aten.select_backward(grad_mul_55_sum, [8], 0, 5)  # strides=(1,), contiguous=True, view=False
    add_102: 'float32[8]' = aten.add.Tensor(add_88, grad_getitem_35_select_backward)  # strides=(1,), contiguous=True, view=False
    grad_mul_54_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_100, select_10)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_54_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_100, add_40)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_54_sum: 'float32[]' = aten.sum(grad_mul_54_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    grad_getitem_34_select_backward: 'float32[8]' = aten.select_backward(grad_mul_54_sum, [8], 0, 5)  # strides=(1,), contiguous=True, view=False
    add_103: 'float32[8]' = aten.add.Tensor(add_89, grad_getitem_34_select_backward)  # strides=(1,), contiguous=True, view=False

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.4
    # ════════════════════════════════════════════════════════════════

    # grad of self.transformer.h.4.mlp.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    grad_h4_mlp_c_proj_view: 'bfloat16[65536, 512]' = aten.view(grad_mul_54_mul, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h4_mlp_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h4_mlp_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h4_mlp_c_proj_mm: 'bfloat16[512, 2048]' = aten.mm(grad_h4_mlp_c_proj_t, view_53)  # strides=(2048, 1), contiguous=True, view=False
    grad_h4_mlp_c_proj_t_1: 'bfloat16[2048, 512]' = aten.t(grad_h4_mlp_c_proj_mm)  # strides=(1, 2048), contiguous=False, view=True
    grad_h4_mlp_c_proj_t_2: 'bfloat16[512, 2048]' = aten.t(t_31)  # strides=(2048, 1), contiguous=True, view=True
    grad_h4_mlp_c_proj_mm_1: 'bfloat16[65536, 2048]' = aten.mm(grad_h4_mlp_c_proj_view, grad_h4_mlp_c_proj_t_2)  # strides=(2048, 1), contiguous=True, view=False
    grad_h4_mlp_c_proj_view_1: 'bfloat16[32, 2048, 2048]' = aten.view(grad_h4_mlp_c_proj_mm_1, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=True
    grad_h4_mlp_c_proj_t_3: 'bfloat16[512, 2048]' = aten.t(grad_h4_mlp_c_proj_t_1)  # strides=(2048, 1), contiguous=True, view=True
    grad_h4_mlp_c_proj__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(grad_h4_mlp_c_proj_view_1, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(4194304, 2048, 1), contiguous=True, view=False
    grad_h4_mlp_c_proj__to_copy_1: 'float32[512, 2048]' = aten._to_copy(grad_h4_mlp_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(2048, 1), contiguous=True, view=False

    # grad of self.transformer.h.4.mlp (MLP) → d_loss/d_mlp
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    # FUSED: squared_relu backward
    grad_h4_mlp_threshold_backward: 'bfloat16[32, 2048, 2048]' = triton_squared_relu_bwd(grad_h4_mlp_c_proj__to_copy, _to_copy_39, detach_32)

    # grad of self.transformer.h.4.mlp.c_fc (Linear) → d_loss/d_c_fc
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    grad_h4_mlp_c_fc_view: 'bfloat16[65536, 2048]' = aten.view(grad_h4_mlp_threshold_backward, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    grad_h4_mlp_c_fc_t: 'bfloat16[2048, 65536]' = aten.t(grad_h4_mlp_c_fc_view)  # strides=(1, 2048), contiguous=False, view=True
    grad_h4_mlp_c_fc_mm: 'bfloat16[2048, 512]' = aten.mm(grad_h4_mlp_c_fc_t, view_52)  # strides=(512, 1), contiguous=True, view=False
    grad_h4_mlp_c_fc_t_1: 'bfloat16[512, 2048]' = aten.t(grad_h4_mlp_c_fc_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h4_mlp_c_fc_t_2: 'bfloat16[2048, 512]' = aten.t(t_30)  # strides=(512, 1), contiguous=True, view=True
    grad_h4_mlp_c_fc_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h4_mlp_c_fc_view, grad_h4_mlp_c_fc_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h4_mlp_c_fc_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h4_mlp_c_fc_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h4_mlp_c_fc_t_3: 'bfloat16[2048, 512]' = aten.t(grad_h4_mlp_c_fc_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h4_mlp_c_fc__to_copy: 'float32[2048, 512]' = aten._to_copy(grad_h4_mlp_c_fc_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.4 (Block) → d_loss/d_4
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h4_detach: 'float32[32, 2048, 1]' = aten.detach(detach_31)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h4__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h4_mlp_c_fc_view_1, add_39, [512], grad_h4_detach, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h4_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h4__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_104: 'bfloat16[32, 2048, 512]' = cuda_add(grad_mul_54_mul, grad_h4_getitem)  # FUSED: vectorized CUDA bf16 add

    # grad of self.transformer.h.4.attn.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    grad_h4_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(add_104, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h4_attn_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h4_attn_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h4_attn_c_proj_mm: 'bfloat16[512, 512]' = aten.mm(grad_h4_attn_c_proj_t, view_51)  # strides=(512, 1), contiguous=True, view=False
    grad_h4_attn_c_proj_t_1: 'bfloat16[512, 512]' = aten.t(grad_h4_attn_c_proj_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h4_attn_c_proj_t_2: 'bfloat16[512, 512]' = aten.t(t_29)  # strides=(512, 1), contiguous=True, view=True
    grad_h4_attn_c_proj_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h4_attn_c_proj_view, grad_h4_attn_c_proj_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h4_attn_c_proj_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h4_attn_c_proj_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h4_attn_c_proj_t_3: 'bfloat16[512, 512]' = aten.t(grad_h4_attn_c_proj_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h4_attn_c_proj__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h4_attn_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.4.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:89
    # y = y.contiguous().view(B, T, -1)
    grad_h4_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(grad_h4_attn_c_proj_view_1, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h4_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(grad_h4_attn_view, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h4_attn_detach: 'bfloat16[32, 4, 2048, 128]' = aten.detach(detach_30)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h4_attn__scaled_dot_product_cudnn_attention_backward = aten._scaled_dot_product_cudnn_attention_backward(grad_h4_attn_transpose, transpose_16, transpose_17, transpose_18, grad_h4_attn_detach, getitem_77, getitem_82, getitem_83, where_3, None, None, 2048, 2048, 0.0, False)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(1048576, 128, 512, 1), contiguous=False; out2: strides=(1048576, 128, 512, 1), contiguous=False, view=False
    grad_h4_attn_getitem: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h4_attn__scaled_dot_product_cudnn_attention_backward, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h4_attn_getitem_1: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h4_attn__scaled_dot_product_cudnn_attention_backward, 1)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h4_attn_getitem_2: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h4_attn__scaled_dot_product_cudnn_attention_backward, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h4_attn_transpose_1: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h4_attn_getitem_2, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h4_attn_transpose_2: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h4_attn_getitem_1, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h4_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h4_attn_getitem, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h4_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(detach_29)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h4_attn__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h4_attn_transpose_2, cat_9, [128], grad_h4_attn_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h4_attn_getitem_3: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h4_attn__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h4_attn_detach_2: 'float32[32, 2048, 4, 1]' = aten.detach(detach_28)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h4_attn__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(grad_h4_attn_transpose_3, cat_8, [128], grad_h4_attn_detach_2, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h4_attn_getitem_4: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h4_attn__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h4_attn_slice: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h4_attn_getitem_3, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h4_attn_slice_1: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h4_attn_getitem_3, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h4_attn_mul: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h4_attn_slice_1, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h4_attn_mul_1: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h4_attn_slice_1, neg_9)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h4_attn_mul_2: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h4_attn_slice, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_105: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h4_attn_mul, grad_h4_attn_mul_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h4_attn_mul_3: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h4_attn_slice, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_106: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h4_attn_mul_1, grad_h4_attn_mul_3)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h4_attn_slice_backward: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_105, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h4_attn_slice_backward_1: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_106, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_107: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h4_attn_slice_backward, grad_h4_attn_slice_backward_1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h4_attn_slice_2: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h4_attn_getitem_4, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h4_attn_slice_3: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h4_attn_getitem_4, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h4_attn_mul_4: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h4_attn_slice_3, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h4_attn_mul_5: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h4_attn_slice_3, neg_8)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h4_attn_mul_6: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h4_attn_slice_2, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_108: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h4_attn_mul_4, grad_h4_attn_mul_6)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h4_attn_mul_7: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h4_attn_slice_2, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_109: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h4_attn_mul_5, grad_h4_attn_mul_7)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h4_attn_slice_backward_2: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_108, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h4_attn_slice_backward_3: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_109, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_110: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h4_attn_slice_backward_2, grad_h4_attn_slice_backward_3)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h4_attn_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h4_attn_transpose_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.4.attn.c_v (Linear) → d_loss/d_c_v
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h4_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(grad_h4_attn_view_1, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h4_attn_c_v_t: 'bfloat16[512, 65536]' = aten.t(grad_h4_attn_c_v_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h4_attn_c_v_mm: 'bfloat16[512, 512]' = aten.mm(grad_h4_attn_c_v_t, view_48)  # strides=(512, 1), contiguous=True, view=False
    grad_h4_attn_c_v_t_1: 'bfloat16[512, 512]' = aten.t(grad_h4_attn_c_v_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h4_attn_c_v_t_2: 'bfloat16[512, 512]' = aten.t(t_28)  # strides=(512, 1), contiguous=True, view=True
    grad_h4_attn_c_v_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h4_attn_c_v_view, grad_h4_attn_c_v_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h4_attn_c_v_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h4_attn_c_v_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h4_attn_c_v_t_3: 'bfloat16[512, 512]' = aten.t(grad_h4_attn_c_v_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h4_attn_c_v__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h4_attn_c_v_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.4.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h4_attn_view_2: 'bfloat16[32, 2048, 512]' = aten.view(add_107, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.4.attn.c_k (Linear) → d_loss/d_c_k
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h4_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(grad_h4_attn_view_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h4_attn_c_k_t: 'bfloat16[512, 65536]' = aten.t(grad_h4_attn_c_k_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h4_attn_c_k_mm: 'bfloat16[512, 512]' = aten.mm(grad_h4_attn_c_k_t, view_46)  # strides=(512, 1), contiguous=True, view=False
    grad_h4_attn_c_k_t_1: 'bfloat16[512, 512]' = aten.t(grad_h4_attn_c_k_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h4_attn_c_k_t_2: 'bfloat16[512, 512]' = aten.t(t_27)  # strides=(512, 1), contiguous=True, view=True
    grad_h4_attn_c_k_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h4_attn_c_k_view, grad_h4_attn_c_k_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h4_attn_c_k_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h4_attn_c_k_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_111: 'bfloat16[32, 2048, 512]' = cuda_add(grad_h4_attn_c_v_view_1, grad_h4_attn_c_k_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h4_attn_c_k_t_3: 'bfloat16[512, 512]' = aten.t(grad_h4_attn_c_k_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h4_attn_c_k__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h4_attn_c_k_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.4.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h4_attn_view_3: 'bfloat16[32, 2048, 512]' = aten.view(add_110, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.4.attn.c_q (Linear) → d_loss/d_c_q
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h4_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(grad_h4_attn_view_3, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h4_attn_c_q_t: 'bfloat16[512, 65536]' = aten.t(grad_h4_attn_c_q_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h4_attn_c_q_mm: 'bfloat16[512, 512]' = aten.mm(grad_h4_attn_c_q_t, view_44)  # strides=(512, 1), contiguous=True, view=False
    grad_h4_attn_c_q_t_1: 'bfloat16[512, 512]' = aten.t(grad_h4_attn_c_q_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h4_attn_c_q_t_2: 'bfloat16[512, 512]' = aten.t(t_26)  # strides=(512, 1), contiguous=True, view=True
    grad_h4_attn_c_q_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h4_attn_c_q_view, grad_h4_attn_c_q_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h4_attn_c_q_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h4_attn_c_q_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_112: 'bfloat16[32, 2048, 512]' = cuda_add(add_111, grad_h4_attn_c_q_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h4_attn_c_q_t_3: 'bfloat16[512, 512]' = aten.t(grad_h4_attn_c_q_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h4_attn_c_q__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h4_attn_c_q_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.4 (Block) → d_loss/d_4
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h4_detach_1: 'float32[32, 2048, 1]' = aten.detach(detach_27)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h4__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(add_112, add_33, [512], grad_h4_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h4_getitem_1: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h4__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_113: 'bfloat16[32, 2048, 512]' = cuda_add(add_104, grad_h4_getitem_1)  # FUSED: vectorized CUDA bf16 add

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    grad_mul_45_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_113, select_9)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_45_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_113, getitem)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_45_sum: 'float32[]' = aten.sum(grad_mul_45_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    add_114: 'bfloat16[32, 2048, 512]' = cuda_add(add_101, grad_mul_45_mul)  # FUSED: vectorized CUDA bf16 add
    grad_getitem_29_select_backward: 'float32[8]' = aten.select_backward(grad_mul_45_sum, [8], 0, 4)  # strides=(1,), contiguous=True, view=False
    add_115: 'float32[8]' = aten.add.Tensor(add_102, grad_getitem_29_select_backward)  # strides=(1,), contiguous=True, view=False
    grad_mul_44_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_113, select_8)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_44_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_113, add_32)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_44_sum: 'float32[]' = aten.sum(grad_mul_44_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    grad_getitem_28_select_backward: 'float32[8]' = aten.select_backward(grad_mul_44_sum, [8], 0, 4)  # strides=(1,), contiguous=True, view=False
    add_116: 'float32[8]' = aten.add.Tensor(add_103, grad_getitem_28_select_backward)  # strides=(1,), contiguous=True, view=False

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.3
    # ════════════════════════════════════════════════════════════════

    # grad of self.transformer.h.3.mlp.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    grad_h3_mlp_c_proj_view: 'bfloat16[65536, 512]' = aten.view(grad_mul_44_mul, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h3_mlp_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h3_mlp_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h3_mlp_c_proj_mm: 'bfloat16[512, 2048]' = aten.mm(grad_h3_mlp_c_proj_t, view_43)  # strides=(2048, 1), contiguous=True, view=False
    grad_h3_mlp_c_proj_t_1: 'bfloat16[2048, 512]' = aten.t(grad_h3_mlp_c_proj_mm)  # strides=(1, 2048), contiguous=False, view=True
    grad_h3_mlp_c_proj_t_2: 'bfloat16[512, 2048]' = aten.t(t_25)  # strides=(2048, 1), contiguous=True, view=True
    grad_h3_mlp_c_proj_mm_1: 'bfloat16[65536, 2048]' = aten.mm(grad_h3_mlp_c_proj_view, grad_h3_mlp_c_proj_t_2)  # strides=(2048, 1), contiguous=True, view=False
    grad_h3_mlp_c_proj_view_1: 'bfloat16[32, 2048, 2048]' = aten.view(grad_h3_mlp_c_proj_mm_1, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=True
    grad_h3_mlp_c_proj_t_3: 'bfloat16[512, 2048]' = aten.t(grad_h3_mlp_c_proj_t_1)  # strides=(2048, 1), contiguous=True, view=True
    grad_h3_mlp_c_proj__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(grad_h3_mlp_c_proj_view_1, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(4194304, 2048, 1), contiguous=True, view=False
    grad_h3_mlp_c_proj__to_copy_1: 'float32[512, 2048]' = aten._to_copy(grad_h3_mlp_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(2048, 1), contiguous=True, view=False

    # grad of self.transformer.h.3.mlp (MLP) → d_loss/d_mlp
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    # FUSED: squared_relu backward
    grad_h3_mlp_threshold_backward: 'bfloat16[32, 2048, 2048]' = triton_squared_relu_bwd(grad_h3_mlp_c_proj__to_copy, _to_copy_31, detach_26)

    # grad of self.transformer.h.3.mlp.c_fc (Linear) → d_loss/d_c_fc
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    grad_h3_mlp_c_fc_view: 'bfloat16[65536, 2048]' = aten.view(grad_h3_mlp_threshold_backward, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    grad_h3_mlp_c_fc_t: 'bfloat16[2048, 65536]' = aten.t(grad_h3_mlp_c_fc_view)  # strides=(1, 2048), contiguous=False, view=True
    grad_h3_mlp_c_fc_mm: 'bfloat16[2048, 512]' = aten.mm(grad_h3_mlp_c_fc_t, view_42)  # strides=(512, 1), contiguous=True, view=False
    grad_h3_mlp_c_fc_t_1: 'bfloat16[512, 2048]' = aten.t(grad_h3_mlp_c_fc_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h3_mlp_c_fc_t_2: 'bfloat16[2048, 512]' = aten.t(t_24)  # strides=(512, 1), contiguous=True, view=True
    grad_h3_mlp_c_fc_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h3_mlp_c_fc_view, grad_h3_mlp_c_fc_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h3_mlp_c_fc_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h3_mlp_c_fc_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h3_mlp_c_fc_t_3: 'bfloat16[2048, 512]' = aten.t(grad_h3_mlp_c_fc_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h3_mlp_c_fc__to_copy: 'float32[2048, 512]' = aten._to_copy(grad_h3_mlp_c_fc_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.3 (Block) → d_loss/d_3
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h3_detach: 'float32[32, 2048, 1]' = aten.detach(detach_25)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h3__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h3_mlp_c_fc_view_1, add_31, [512], grad_h3_detach, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h3_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h3__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_117: 'bfloat16[32, 2048, 512]' = cuda_add(grad_mul_44_mul, grad_h3_getitem)  # FUSED: vectorized CUDA bf16 add

    # grad of self.transformer.h.3.attn.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    grad_h3_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(add_117, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h3_attn_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h3_attn_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h3_attn_c_proj_mm: 'bfloat16[512, 512]' = aten.mm(grad_h3_attn_c_proj_t, view_41)  # strides=(512, 1), contiguous=True, view=False
    grad_h3_attn_c_proj_t_1: 'bfloat16[512, 512]' = aten.t(grad_h3_attn_c_proj_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h3_attn_c_proj_t_2: 'bfloat16[512, 512]' = aten.t(t_23)  # strides=(512, 1), contiguous=True, view=True
    grad_h3_attn_c_proj_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h3_attn_c_proj_view, grad_h3_attn_c_proj_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h3_attn_c_proj_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h3_attn_c_proj_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h3_attn_c_proj_t_3: 'bfloat16[512, 512]' = aten.t(grad_h3_attn_c_proj_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h3_attn_c_proj__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h3_attn_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.3.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:89
    # y = y.contiguous().view(B, T, -1)
    grad_h3_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(grad_h3_attn_c_proj_view_1, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h3_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(grad_h3_attn_view, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h3_attn_detach: 'bfloat16[32, 4, 2048, 128]' = aten.detach(detach_24)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h3_attn__scaled_dot_product_cudnn_attention_backward = aten._scaled_dot_product_cudnn_attention_backward(grad_h3_attn_transpose, transpose_12, transpose_13, transpose_14, grad_h3_attn_detach, getitem_60, getitem_65, getitem_66, None, None, None, 2048, 2048, 0.0, True)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(1048576, 128, 512, 1), contiguous=False; out2: strides=(1048576, 128, 512, 1), contiguous=False, view=False
    grad_h3_attn_getitem: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h3_attn__scaled_dot_product_cudnn_attention_backward, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h3_attn_getitem_1: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h3_attn__scaled_dot_product_cudnn_attention_backward, 1)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h3_attn_getitem_2: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h3_attn__scaled_dot_product_cudnn_attention_backward, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h3_attn_transpose_1: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h3_attn_getitem_2, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h3_attn_transpose_2: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h3_attn_getitem_1, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h3_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h3_attn_getitem, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h3_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(detach_23)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h3_attn__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h3_attn_transpose_2, cat_7, [128], grad_h3_attn_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h3_attn_getitem_3: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h3_attn__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h3_attn_detach_2: 'float32[32, 2048, 4, 1]' = aten.detach(detach_22)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h3_attn__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(grad_h3_attn_transpose_3, cat_6, [128], grad_h3_attn_detach_2, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h3_attn_getitem_4: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h3_attn__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h3_attn_slice: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h3_attn_getitem_3, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h3_attn_slice_1: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h3_attn_getitem_3, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h3_attn_mul: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h3_attn_slice_1, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h3_attn_mul_1: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h3_attn_slice_1, neg_7)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h3_attn_mul_2: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h3_attn_slice, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_118: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h3_attn_mul, grad_h3_attn_mul_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h3_attn_mul_3: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h3_attn_slice, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_119: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h3_attn_mul_1, grad_h3_attn_mul_3)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h3_attn_slice_backward: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_118, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h3_attn_slice_backward_1: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_119, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_120: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h3_attn_slice_backward, grad_h3_attn_slice_backward_1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h3_attn_slice_2: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h3_attn_getitem_4, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h3_attn_slice_3: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h3_attn_getitem_4, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h3_attn_mul_4: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h3_attn_slice_3, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h3_attn_mul_5: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h3_attn_slice_3, neg_6)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h3_attn_mul_6: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h3_attn_slice_2, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_121: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h3_attn_mul_4, grad_h3_attn_mul_6)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h3_attn_mul_7: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h3_attn_slice_2, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_122: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h3_attn_mul_5, grad_h3_attn_mul_7)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h3_attn_slice_backward_2: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_121, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h3_attn_slice_backward_3: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_122, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_123: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h3_attn_slice_backward_2, grad_h3_attn_slice_backward_3)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h3_attn_mul_8: 'bfloat16[32, 2048, 4, 128]' = aten.mul.Tensor(grad_h3_attn_transpose_1, unsqueeze_7)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h3_attn_mul_9: 'bfloat16[32, 2048, 4, 128]' = aten.mul.Tensor(grad_h3_attn_transpose_1, view_38)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h3_attn_sum: 'float32[32, 2048, 4, 1]' = aten.sum.dim_IntList(grad_h3_attn_mul_9, [3], True, dtype=torch.float32)  # strides=(8192, 4, 1, 1), contiguous=True, view=False
    grad_h3_attn__to_copy: 'bfloat16[32, 2048, 4, 1]' = aten._to_copy(grad_h3_attn_sum, dtype=torch.bfloat16)  # strides=(8192, 4, 1, 1), contiguous=True, view=False
    grad_h3_attn_squeeze: 'bfloat16[32, 2048, 4]' = aten.squeeze.dim(grad_h3_attn__to_copy, -1)  # strides=(8192, 4, 1), contiguous=True, view=True
    grad_h3_attn_mul_10: 'bfloat16[32, 2048, 4]' = aten.mul.Tensor(grad_h3_attn_squeeze, 2)  # strides=(8192, 4, 1), contiguous=True, view=False
    grad_h3_attn_detach_3: 'bfloat16[32, 2048, 4]' = aten.detach(detach_21)  # strides=(8192, 4, 1), contiguous=True, view=True
    grad_h3_attn_sigmoid_backward: 'bfloat16[32, 2048, 4]' = aten.sigmoid_backward(grad_h3_attn_mul_10, grad_h3_attn_detach_3)  # strides=(8192, 4, 1), contiguous=True, view=False

    # grad of self.transformer.h.3.attn.ve_gate (Linear) → d_loss/d_ve_gate
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    grad_h3_attn_ve_gate_view: 'bfloat16[65536, 4]' = aten.view(grad_h3_attn_sigmoid_backward, [65536, 4])  # strides=(4, 1), contiguous=True, view=True
    grad_h3_attn_ve_gate_t: 'bfloat16[4, 65536]' = aten.t(grad_h3_attn_ve_gate_view)  # strides=(1, 4), contiguous=False, view=True
    grad_h3_attn_ve_gate_mm: 'bfloat16[4, 32]' = aten.mm(grad_h3_attn_ve_gate_t, view_39)  # strides=(32, 1), contiguous=True, view=False
    grad_h3_attn_ve_gate_t_1: 'bfloat16[32, 4]' = aten.t(grad_h3_attn_ve_gate_mm)  # strides=(1, 32), contiguous=False, view=True
    grad_h3_attn_ve_gate_t_2: 'bfloat16[4, 32]' = aten.t(t_22)  # strides=(32, 1), contiguous=True, view=True
    grad_h3_attn_ve_gate_mm_1: 'bfloat16[65536, 32]' = aten.mm(grad_h3_attn_ve_gate_view, grad_h3_attn_ve_gate_t_2)  # strides=(32, 1), contiguous=True, view=False
    grad_h3_attn_ve_gate_view_1: 'bfloat16[32, 2048, 32]' = aten.view(grad_h3_attn_ve_gate_mm_1, [32, 2048, 32])  # strides=(65536, 32, 1), contiguous=True, view=True
    grad_h3_attn_ve_gate_t_3: 'bfloat16[4, 32]' = aten.t(grad_h3_attn_ve_gate_t_1)  # strides=(32, 1), contiguous=True, view=True
    grad_h3_attn_ve_gate__to_copy: 'float32[4, 32]' = aten._to_copy(grad_h3_attn_ve_gate_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(32, 1), contiguous=True, view=False

    # grad of self.transformer.h.3.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    grad_h3_attn_slice_backward_4: 'bfloat16[32, 2048, 512]' = aten.slice_backward(grad_h3_attn_ve_gate_view_1, [32, 2048, 512], 2, 0, 32, 1)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h3_attn_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h3_attn_mul_8, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h3_attn_view_2: 'bfloat16[32, 2048, 512]' = aten.view(grad_h3_attn_transpose_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.3.attn.c_v (Linear) → d_loss/d_c_v
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h3_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(grad_h3_attn_view_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h3_attn_c_v_t: 'bfloat16[512, 65536]' = aten.t(grad_h3_attn_c_v_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h3_attn_c_v_mm: 'bfloat16[512, 512]' = aten.mm(grad_h3_attn_c_v_t, view_36)  # strides=(512, 1), contiguous=True, view=False
    grad_h3_attn_c_v_t_1: 'bfloat16[512, 512]' = aten.t(grad_h3_attn_c_v_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h3_attn_c_v_t_2: 'bfloat16[512, 512]' = aten.t(t_21)  # strides=(512, 1), contiguous=True, view=True
    grad_h3_attn_c_v_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h3_attn_c_v_view, grad_h3_attn_c_v_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h3_attn_c_v_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h3_attn_c_v_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_124: 'bfloat16[32, 2048, 512]' = cuda_add(grad_h3_attn_slice_backward_4, grad_h3_attn_c_v_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h3_attn_c_v_t_3: 'bfloat16[512, 512]' = aten.t(grad_h3_attn_c_v_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h3_attn_c_v__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h3_attn_c_v_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.3.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h3_attn_view_3: 'bfloat16[32, 2048, 512]' = aten.view(add_120, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.3.attn.c_k (Linear) → d_loss/d_c_k
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h3_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(grad_h3_attn_view_3, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h3_attn_c_k_t: 'bfloat16[512, 65536]' = aten.t(grad_h3_attn_c_k_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h3_attn_c_k_mm: 'bfloat16[512, 512]' = aten.mm(grad_h3_attn_c_k_t, view_34)  # strides=(512, 1), contiguous=True, view=False
    grad_h3_attn_c_k_t_1: 'bfloat16[512, 512]' = aten.t(grad_h3_attn_c_k_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h3_attn_c_k_t_2: 'bfloat16[512, 512]' = aten.t(t_20)  # strides=(512, 1), contiguous=True, view=True
    grad_h3_attn_c_k_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h3_attn_c_k_view, grad_h3_attn_c_k_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h3_attn_c_k_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h3_attn_c_k_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_125: 'bfloat16[32, 2048, 512]' = cuda_add(add_124, grad_h3_attn_c_k_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h3_attn_c_k_t_3: 'bfloat16[512, 512]' = aten.t(grad_h3_attn_c_k_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h3_attn_c_k__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h3_attn_c_k_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.3.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h3_attn_view_4: 'bfloat16[32, 2048, 512]' = aten.view(add_123, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.3.attn.c_q (Linear) → d_loss/d_c_q
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h3_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(grad_h3_attn_view_4, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h3_attn_c_q_t: 'bfloat16[512, 65536]' = aten.t(grad_h3_attn_c_q_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h3_attn_c_q_mm: 'bfloat16[512, 512]' = aten.mm(grad_h3_attn_c_q_t, view_32)  # strides=(512, 1), contiguous=True, view=False
    grad_h3_attn_c_q_t_1: 'bfloat16[512, 512]' = aten.t(grad_h3_attn_c_q_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h3_attn_c_q_t_2: 'bfloat16[512, 512]' = aten.t(t_19)  # strides=(512, 1), contiguous=True, view=True
    grad_h3_attn_c_q_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h3_attn_c_q_view, grad_h3_attn_c_q_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h3_attn_c_q_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h3_attn_c_q_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_126: 'bfloat16[32, 2048, 512]' = cuda_add(add_125, grad_h3_attn_c_q_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h3_attn_c_q_t_3: 'bfloat16[512, 512]' = aten.t(grad_h3_attn_c_q_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h3_attn_c_q__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h3_attn_c_q_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.3 (Block) → d_loss/d_3
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h3_detach_1: 'float32[32, 2048, 1]' = aten.detach(detach_20)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h3__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(add_126, add_25, [512], grad_h3_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h3_getitem_1: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h3__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_127: 'bfloat16[32, 2048, 512]' = cuda_add(add_117, grad_h3_getitem_1)  # FUSED: vectorized CUDA bf16 add

    # ════════════════════════════════════════════════════════════════
    # self.value_embeds.3
    # ════════════════════════════════════════════════════════════════

    # grad of self.value_embeds.3 (Embedding) → d_loss/d_3
    # /.autoresearch_repo/train.py:273
    # ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
    grad_value_embeds3_embedding_dense_backward: 'bfloat16[8192, 512]' = aten.embedding_dense_backward(grad_h3_attn_view_1, input__32__2048, 8192, -1, False)  # strides=(512, 1), contiguous=True, view=False

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    grad_mul_33_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_127, select_7)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_33_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_127, getitem)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_33_sum: 'float32[]' = aten.sum(grad_mul_33_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    add_128: 'bfloat16[32, 2048, 512]' = cuda_add(add_114, grad_mul_33_mul)  # FUSED: vectorized CUDA bf16 add
    grad_getitem_22_select_backward: 'float32[8]' = aten.select_backward(grad_mul_33_sum, [8], 0, 3)  # strides=(1,), contiguous=True, view=False
    add_129: 'float32[8]' = aten.add.Tensor(add_115, grad_getitem_22_select_backward)  # strides=(1,), contiguous=True, view=False
    grad_mul_32_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_127, select_6)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_32_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_127, add_24)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_32_sum: 'float32[]' = aten.sum(grad_mul_32_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    grad_getitem_21_select_backward: 'float32[8]' = aten.select_backward(grad_mul_32_sum, [8], 0, 3)  # strides=(1,), contiguous=True, view=False
    add_130: 'float32[8]' = aten.add.Tensor(add_116, grad_getitem_21_select_backward)  # strides=(1,), contiguous=True, view=False

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.2
    # ════════════════════════════════════════════════════════════════

    # grad of self.transformer.h.2.mlp.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    grad_h2_mlp_c_proj_view: 'bfloat16[65536, 512]' = aten.view(grad_mul_32_mul, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h2_mlp_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h2_mlp_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h2_mlp_c_proj_mm: 'bfloat16[512, 2048]' = aten.mm(grad_h2_mlp_c_proj_t, view_31)  # strides=(2048, 1), contiguous=True, view=False
    grad_h2_mlp_c_proj_t_1: 'bfloat16[2048, 512]' = aten.t(grad_h2_mlp_c_proj_mm)  # strides=(1, 2048), contiguous=False, view=True
    grad_h2_mlp_c_proj_t_2: 'bfloat16[512, 2048]' = aten.t(t_18)  # strides=(2048, 1), contiguous=True, view=True
    grad_h2_mlp_c_proj_mm_1: 'bfloat16[65536, 2048]' = aten.mm(grad_h2_mlp_c_proj_view, grad_h2_mlp_c_proj_t_2)  # strides=(2048, 1), contiguous=True, view=False
    grad_h2_mlp_c_proj_view_1: 'bfloat16[32, 2048, 2048]' = aten.view(grad_h2_mlp_c_proj_mm_1, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=True
    grad_h2_mlp_c_proj_t_3: 'bfloat16[512, 2048]' = aten.t(grad_h2_mlp_c_proj_t_1)  # strides=(2048, 1), contiguous=True, view=True
    grad_h2_mlp_c_proj__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(grad_h2_mlp_c_proj_view_1, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(4194304, 2048, 1), contiguous=True, view=False
    grad_h2_mlp_c_proj__to_copy_1: 'float32[512, 2048]' = aten._to_copy(grad_h2_mlp_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(2048, 1), contiguous=True, view=False

    # grad of self.transformer.h.2.mlp (MLP) → d_loss/d_mlp
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    # FUSED: squared_relu backward
    grad_h2_mlp_threshold_backward: 'bfloat16[32, 2048, 2048]' = triton_squared_relu_bwd(grad_h2_mlp_c_proj__to_copy, _to_copy_22, detach_19)

    # grad of self.transformer.h.2.mlp.c_fc (Linear) → d_loss/d_c_fc
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    grad_h2_mlp_c_fc_view: 'bfloat16[65536, 2048]' = aten.view(grad_h2_mlp_threshold_backward, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    grad_h2_mlp_c_fc_t: 'bfloat16[2048, 65536]' = aten.t(grad_h2_mlp_c_fc_view)  # strides=(1, 2048), contiguous=False, view=True
    grad_h2_mlp_c_fc_mm: 'bfloat16[2048, 512]' = aten.mm(grad_h2_mlp_c_fc_t, view_30)  # strides=(512, 1), contiguous=True, view=False
    grad_h2_mlp_c_fc_t_1: 'bfloat16[512, 2048]' = aten.t(grad_h2_mlp_c_fc_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h2_mlp_c_fc_t_2: 'bfloat16[2048, 512]' = aten.t(t_17)  # strides=(512, 1), contiguous=True, view=True
    grad_h2_mlp_c_fc_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h2_mlp_c_fc_view, grad_h2_mlp_c_fc_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h2_mlp_c_fc_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h2_mlp_c_fc_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h2_mlp_c_fc_t_3: 'bfloat16[2048, 512]' = aten.t(grad_h2_mlp_c_fc_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h2_mlp_c_fc__to_copy: 'float32[2048, 512]' = aten._to_copy(grad_h2_mlp_c_fc_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.2 (Block) → d_loss/d_2
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h2_detach: 'float32[32, 2048, 1]' = aten.detach(detach_18)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h2__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h2_mlp_c_fc_view_1, add_23, [512], grad_h2_detach, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h2_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h2__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_131: 'bfloat16[32, 2048, 512]' = cuda_add(grad_mul_32_mul, grad_h2_getitem)  # FUSED: vectorized CUDA bf16 add

    # grad of self.transformer.h.2.attn.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    grad_h2_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(add_131, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h2_attn_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h2_attn_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h2_attn_c_proj_mm: 'bfloat16[512, 512]' = aten.mm(grad_h2_attn_c_proj_t, view_29)  # strides=(512, 1), contiguous=True, view=False
    grad_h2_attn_c_proj_t_1: 'bfloat16[512, 512]' = aten.t(grad_h2_attn_c_proj_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h2_attn_c_proj_t_2: 'bfloat16[512, 512]' = aten.t(t_16)  # strides=(512, 1), contiguous=True, view=True
    grad_h2_attn_c_proj_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h2_attn_c_proj_view, grad_h2_attn_c_proj_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h2_attn_c_proj_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h2_attn_c_proj_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h2_attn_c_proj_t_3: 'bfloat16[512, 512]' = aten.t(grad_h2_attn_c_proj_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h2_attn_c_proj__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h2_attn_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.2.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:89
    # y = y.contiguous().view(B, T, -1)
    grad_h2_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(grad_h2_attn_c_proj_view_1, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h2_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(grad_h2_attn_view, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h2_attn_detach: 'bfloat16[32, 4, 2048, 128]' = aten.detach(detach_17)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h2_attn__scaled_dot_product_cudnn_attention_backward = aten._scaled_dot_product_cudnn_attention_backward(grad_h2_attn_transpose, transpose_8, transpose_9, transpose_10, grad_h2_attn_detach, getitem_43, getitem_48, getitem_49, where_2, None, None, 2048, 2048, 0.0, False)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(1048576, 128, 512, 1), contiguous=False; out2: strides=(1048576, 128, 512, 1), contiguous=False, view=False
    grad_h2_attn_getitem: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h2_attn__scaled_dot_product_cudnn_attention_backward, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h2_attn_getitem_1: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h2_attn__scaled_dot_product_cudnn_attention_backward, 1)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h2_attn_getitem_2: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h2_attn__scaled_dot_product_cudnn_attention_backward, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h2_attn_transpose_1: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h2_attn_getitem_2, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h2_attn_transpose_2: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h2_attn_getitem_1, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h2_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h2_attn_getitem, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h2_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(detach_16)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h2_attn__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h2_attn_transpose_2, cat_5, [128], grad_h2_attn_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h2_attn_getitem_3: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h2_attn__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h2_attn_detach_2: 'float32[32, 2048, 4, 1]' = aten.detach(detach_15)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h2_attn__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(grad_h2_attn_transpose_3, cat_4, [128], grad_h2_attn_detach_2, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h2_attn_getitem_4: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h2_attn__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h2_attn_slice: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h2_attn_getitem_3, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h2_attn_slice_1: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h2_attn_getitem_3, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h2_attn_mul: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h2_attn_slice_1, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h2_attn_mul_1: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h2_attn_slice_1, neg_5)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h2_attn_mul_2: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h2_attn_slice, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_132: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h2_attn_mul, grad_h2_attn_mul_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h2_attn_mul_3: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h2_attn_slice, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_133: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h2_attn_mul_1, grad_h2_attn_mul_3)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h2_attn_slice_backward: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_132, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h2_attn_slice_backward_1: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_133, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_134: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h2_attn_slice_backward, grad_h2_attn_slice_backward_1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h2_attn_slice_2: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h2_attn_getitem_4, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h2_attn_slice_3: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h2_attn_getitem_4, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h2_attn_mul_4: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h2_attn_slice_3, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h2_attn_mul_5: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h2_attn_slice_3, neg_4)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h2_attn_mul_6: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h2_attn_slice_2, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_135: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h2_attn_mul_4, grad_h2_attn_mul_6)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h2_attn_mul_7: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h2_attn_slice_2, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_136: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h2_attn_mul_5, grad_h2_attn_mul_7)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h2_attn_slice_backward_2: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_135, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h2_attn_slice_backward_3: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_136, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_137: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h2_attn_slice_backward_2, grad_h2_attn_slice_backward_3)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h2_attn_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h2_attn_transpose_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.2.attn.c_v (Linear) → d_loss/d_c_v
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h2_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(grad_h2_attn_view_1, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h2_attn_c_v_t: 'bfloat16[512, 65536]' = aten.t(grad_h2_attn_c_v_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h2_attn_c_v_mm: 'bfloat16[512, 512]' = aten.mm(grad_h2_attn_c_v_t, view_26)  # strides=(512, 1), contiguous=True, view=False
    grad_h2_attn_c_v_t_1: 'bfloat16[512, 512]' = aten.t(grad_h2_attn_c_v_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h2_attn_c_v_t_2: 'bfloat16[512, 512]' = aten.t(t_15)  # strides=(512, 1), contiguous=True, view=True
    grad_h2_attn_c_v_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h2_attn_c_v_view, grad_h2_attn_c_v_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h2_attn_c_v_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h2_attn_c_v_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h2_attn_c_v_t_3: 'bfloat16[512, 512]' = aten.t(grad_h2_attn_c_v_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h2_attn_c_v__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h2_attn_c_v_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.2.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h2_attn_view_2: 'bfloat16[32, 2048, 512]' = aten.view(add_134, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.2.attn.c_k (Linear) → d_loss/d_c_k
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h2_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(grad_h2_attn_view_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h2_attn_c_k_t: 'bfloat16[512, 65536]' = aten.t(grad_h2_attn_c_k_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h2_attn_c_k_mm: 'bfloat16[512, 512]' = aten.mm(grad_h2_attn_c_k_t, view_24)  # strides=(512, 1), contiguous=True, view=False
    grad_h2_attn_c_k_t_1: 'bfloat16[512, 512]' = aten.t(grad_h2_attn_c_k_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h2_attn_c_k_t_2: 'bfloat16[512, 512]' = aten.t(t_14)  # strides=(512, 1), contiguous=True, view=True
    grad_h2_attn_c_k_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h2_attn_c_k_view, grad_h2_attn_c_k_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h2_attn_c_k_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h2_attn_c_k_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_138: 'bfloat16[32, 2048, 512]' = cuda_add(grad_h2_attn_c_v_view_1, grad_h2_attn_c_k_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h2_attn_c_k_t_3: 'bfloat16[512, 512]' = aten.t(grad_h2_attn_c_k_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h2_attn_c_k__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h2_attn_c_k_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.2.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h2_attn_view_3: 'bfloat16[32, 2048, 512]' = aten.view(add_137, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.2.attn.c_q (Linear) → d_loss/d_c_q
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h2_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(grad_h2_attn_view_3, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h2_attn_c_q_t: 'bfloat16[512, 65536]' = aten.t(grad_h2_attn_c_q_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h2_attn_c_q_mm: 'bfloat16[512, 512]' = aten.mm(grad_h2_attn_c_q_t, view_22)  # strides=(512, 1), contiguous=True, view=False
    grad_h2_attn_c_q_t_1: 'bfloat16[512, 512]' = aten.t(grad_h2_attn_c_q_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h2_attn_c_q_t_2: 'bfloat16[512, 512]' = aten.t(t_13)  # strides=(512, 1), contiguous=True, view=True
    grad_h2_attn_c_q_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h2_attn_c_q_view, grad_h2_attn_c_q_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h2_attn_c_q_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h2_attn_c_q_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_139: 'bfloat16[32, 2048, 512]' = cuda_add(add_138, grad_h2_attn_c_q_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h2_attn_c_q_t_3: 'bfloat16[512, 512]' = aten.t(grad_h2_attn_c_q_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h2_attn_c_q__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h2_attn_c_q_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.2 (Block) → d_loss/d_2
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h2_detach_1: 'float32[32, 2048, 1]' = aten.detach(detach_14)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h2__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(add_139, add_17, [512], grad_h2_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h2_getitem_1: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h2__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_140: 'bfloat16[32, 2048, 512]' = cuda_add(add_131, grad_h2_getitem_1)  # FUSED: vectorized CUDA bf16 add

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    grad_mul_23_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_140, select_5)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_23_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_140, getitem)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_23_sum: 'float32[]' = aten.sum(grad_mul_23_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    add_141: 'bfloat16[32, 2048, 512]' = cuda_add(add_128, grad_mul_23_mul)  # FUSED: vectorized CUDA bf16 add
    grad_getitem_16_select_backward: 'float32[8]' = aten.select_backward(grad_mul_23_sum, [8], 0, 2)  # strides=(1,), contiguous=True, view=False
    add_142: 'float32[8]' = aten.add.Tensor(add_129, grad_getitem_16_select_backward)  # strides=(1,), contiguous=True, view=False
    grad_mul_22_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_140, select_4)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_22_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_140, add_16)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_22_sum: 'float32[]' = aten.sum(grad_mul_22_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    grad_getitem_15_select_backward: 'float32[8]' = aten.select_backward(grad_mul_22_sum, [8], 0, 2)  # strides=(1,), contiguous=True, view=False
    add_143: 'float32[8]' = aten.add.Tensor(add_130, grad_getitem_15_select_backward)  # strides=(1,), contiguous=True, view=False

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.1
    # ════════════════════════════════════════════════════════════════

    # grad of self.transformer.h.1.mlp.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    grad_h1_mlp_c_proj_view: 'bfloat16[65536, 512]' = aten.view(grad_mul_22_mul, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h1_mlp_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h1_mlp_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h1_mlp_c_proj_mm: 'bfloat16[512, 2048]' = aten.mm(grad_h1_mlp_c_proj_t, view_21)  # strides=(2048, 1), contiguous=True, view=False
    grad_h1_mlp_c_proj_t_1: 'bfloat16[2048, 512]' = aten.t(grad_h1_mlp_c_proj_mm)  # strides=(1, 2048), contiguous=False, view=True
    grad_h1_mlp_c_proj_t_2: 'bfloat16[512, 2048]' = aten.t(t_12)  # strides=(2048, 1), contiguous=True, view=True
    grad_h1_mlp_c_proj_mm_1: 'bfloat16[65536, 2048]' = aten.mm(grad_h1_mlp_c_proj_view, grad_h1_mlp_c_proj_t_2)  # strides=(2048, 1), contiguous=True, view=False
    grad_h1_mlp_c_proj_view_1: 'bfloat16[32, 2048, 2048]' = aten.view(grad_h1_mlp_c_proj_mm_1, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=True
    grad_h1_mlp_c_proj_t_3: 'bfloat16[512, 2048]' = aten.t(grad_h1_mlp_c_proj_t_1)  # strides=(2048, 1), contiguous=True, view=True
    grad_h1_mlp_c_proj__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(grad_h1_mlp_c_proj_view_1, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(4194304, 2048, 1), contiguous=True, view=False
    grad_h1_mlp_c_proj__to_copy_1: 'float32[512, 2048]' = aten._to_copy(grad_h1_mlp_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(2048, 1), contiguous=True, view=False

    # grad of self.transformer.h.1.mlp (MLP) → d_loss/d_mlp
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    # FUSED: squared_relu backward
    grad_h1_mlp_threshold_backward: 'bfloat16[32, 2048, 2048]' = triton_squared_relu_bwd(grad_h1_mlp_c_proj__to_copy, _to_copy_14, detach_13)

    # grad of self.transformer.h.1.mlp.c_fc (Linear) → d_loss/d_c_fc
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    grad_h1_mlp_c_fc_view: 'bfloat16[65536, 2048]' = aten.view(grad_h1_mlp_threshold_backward, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    grad_h1_mlp_c_fc_t: 'bfloat16[2048, 65536]' = aten.t(grad_h1_mlp_c_fc_view)  # strides=(1, 2048), contiguous=False, view=True
    grad_h1_mlp_c_fc_mm: 'bfloat16[2048, 512]' = aten.mm(grad_h1_mlp_c_fc_t, view_20)  # strides=(512, 1), contiguous=True, view=False
    grad_h1_mlp_c_fc_t_1: 'bfloat16[512, 2048]' = aten.t(grad_h1_mlp_c_fc_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h1_mlp_c_fc_t_2: 'bfloat16[2048, 512]' = aten.t(t_11)  # strides=(512, 1), contiguous=True, view=True
    grad_h1_mlp_c_fc_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h1_mlp_c_fc_view, grad_h1_mlp_c_fc_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h1_mlp_c_fc_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h1_mlp_c_fc_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h1_mlp_c_fc_t_3: 'bfloat16[2048, 512]' = aten.t(grad_h1_mlp_c_fc_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h1_mlp_c_fc__to_copy: 'float32[2048, 512]' = aten._to_copy(grad_h1_mlp_c_fc_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.1 (Block) → d_loss/d_1
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h1_detach: 'float32[32, 2048, 1]' = aten.detach(detach_12)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h1__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h1_mlp_c_fc_view_1, add_15, [512], grad_h1_detach, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h1_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h1__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_144: 'bfloat16[32, 2048, 512]' = cuda_add(grad_mul_22_mul, grad_h1_getitem)  # FUSED: vectorized CUDA bf16 add

    # grad of self.transformer.h.1.attn.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    grad_h1_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(add_144, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h1_attn_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h1_attn_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h1_attn_c_proj_mm: 'bfloat16[512, 512]' = aten.mm(grad_h1_attn_c_proj_t, view_19)  # strides=(512, 1), contiguous=True, view=False
    grad_h1_attn_c_proj_t_1: 'bfloat16[512, 512]' = aten.t(grad_h1_attn_c_proj_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h1_attn_c_proj_t_2: 'bfloat16[512, 512]' = aten.t(t_10)  # strides=(512, 1), contiguous=True, view=True
    grad_h1_attn_c_proj_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h1_attn_c_proj_view, grad_h1_attn_c_proj_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h1_attn_c_proj_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h1_attn_c_proj_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h1_attn_c_proj_t_3: 'bfloat16[512, 512]' = aten.t(grad_h1_attn_c_proj_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h1_attn_c_proj__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h1_attn_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.1.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:89
    # y = y.contiguous().view(B, T, -1)
    grad_h1_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(grad_h1_attn_c_proj_view_1, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h1_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(grad_h1_attn_view, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h1_attn_detach: 'bfloat16[32, 4, 2048, 128]' = aten.detach(detach_11)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h1_attn__scaled_dot_product_cudnn_attention_backward = aten._scaled_dot_product_cudnn_attention_backward(grad_h1_attn_transpose, transpose_4, transpose_5, transpose_6, grad_h1_attn_detach, getitem_26, getitem_31, getitem_32, where_1, None, None, 2048, 2048, 0.0, False)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(1048576, 128, 512, 1), contiguous=False; out2: strides=(1048576, 128, 512, 1), contiguous=False, view=False
    grad_h1_attn_getitem: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h1_attn__scaled_dot_product_cudnn_attention_backward, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h1_attn_getitem_1: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h1_attn__scaled_dot_product_cudnn_attention_backward, 1)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h1_attn_getitem_2: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h1_attn__scaled_dot_product_cudnn_attention_backward, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h1_attn_transpose_1: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h1_attn_getitem_2, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h1_attn_transpose_2: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h1_attn_getitem_1, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h1_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h1_attn_getitem, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h1_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(detach_10)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h1_attn__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h1_attn_transpose_2, cat_3, [128], grad_h1_attn_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h1_attn_getitem_3: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h1_attn__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h1_attn_detach_2: 'float32[32, 2048, 4, 1]' = aten.detach(detach_9)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h1_attn__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(grad_h1_attn_transpose_3, cat_2, [128], grad_h1_attn_detach_2, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h1_attn_getitem_4: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h1_attn__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h1_attn_slice: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h1_attn_getitem_3, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h1_attn_slice_1: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h1_attn_getitem_3, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h1_attn_mul: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h1_attn_slice_1, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h1_attn_mul_1: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h1_attn_slice_1, neg_3)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h1_attn_mul_2: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h1_attn_slice, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_145: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h1_attn_mul, grad_h1_attn_mul_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h1_attn_mul_3: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h1_attn_slice, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_146: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h1_attn_mul_1, grad_h1_attn_mul_3)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h1_attn_slice_backward: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_145, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h1_attn_slice_backward_1: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_146, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_147: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h1_attn_slice_backward, grad_h1_attn_slice_backward_1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h1_attn_slice_2: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h1_attn_getitem_4, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h1_attn_slice_3: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h1_attn_getitem_4, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h1_attn_mul_4: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h1_attn_slice_3, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h1_attn_mul_5: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h1_attn_slice_3, neg_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h1_attn_mul_6: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h1_attn_slice_2, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_148: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h1_attn_mul_4, grad_h1_attn_mul_6)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h1_attn_mul_7: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h1_attn_slice_2, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_149: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h1_attn_mul_5, grad_h1_attn_mul_7)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h1_attn_slice_backward_2: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_148, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h1_attn_slice_backward_3: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_149, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_150: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h1_attn_slice_backward_2, grad_h1_attn_slice_backward_3)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h1_attn_mul_8: 'bfloat16[32, 2048, 4, 128]' = aten.mul.Tensor(grad_h1_attn_transpose_1, unsqueeze_2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h1_attn_mul_9: 'bfloat16[32, 2048, 4, 128]' = aten.mul.Tensor(grad_h1_attn_transpose_1, view_16)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h1_attn_sum: 'float32[32, 2048, 4, 1]' = aten.sum.dim_IntList(grad_h1_attn_mul_9, [3], True, dtype=torch.float32)  # strides=(8192, 4, 1, 1), contiguous=True, view=False
    grad_h1_attn__to_copy: 'bfloat16[32, 2048, 4, 1]' = aten._to_copy(grad_h1_attn_sum, dtype=torch.bfloat16)  # strides=(8192, 4, 1, 1), contiguous=True, view=False
    grad_h1_attn_squeeze: 'bfloat16[32, 2048, 4]' = aten.squeeze.dim(grad_h1_attn__to_copy, -1)  # strides=(8192, 4, 1), contiguous=True, view=True
    grad_h1_attn_mul_10: 'bfloat16[32, 2048, 4]' = aten.mul.Tensor(grad_h1_attn_squeeze, 2)  # strides=(8192, 4, 1), contiguous=True, view=False
    grad_h1_attn_detach_3: 'bfloat16[32, 2048, 4]' = aten.detach(detach_8)  # strides=(8192, 4, 1), contiguous=True, view=True
    grad_h1_attn_sigmoid_backward: 'bfloat16[32, 2048, 4]' = aten.sigmoid_backward(grad_h1_attn_mul_10, grad_h1_attn_detach_3)  # strides=(8192, 4, 1), contiguous=True, view=False

    # grad of self.transformer.h.1.attn.ve_gate (Linear) → d_loss/d_ve_gate
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    grad_h1_attn_ve_gate_view: 'bfloat16[65536, 4]' = aten.view(grad_h1_attn_sigmoid_backward, [65536, 4])  # strides=(4, 1), contiguous=True, view=True
    grad_h1_attn_ve_gate_t: 'bfloat16[4, 65536]' = aten.t(grad_h1_attn_ve_gate_view)  # strides=(1, 4), contiguous=False, view=True
    grad_h1_attn_ve_gate_mm: 'bfloat16[4, 32]' = aten.mm(grad_h1_attn_ve_gate_t, view_17)  # strides=(32, 1), contiguous=True, view=False
    grad_h1_attn_ve_gate_t_1: 'bfloat16[32, 4]' = aten.t(grad_h1_attn_ve_gate_mm)  # strides=(1, 32), contiguous=False, view=True
    grad_h1_attn_ve_gate_t_2: 'bfloat16[4, 32]' = aten.t(t_9)  # strides=(32, 1), contiguous=True, view=True
    grad_h1_attn_ve_gate_mm_1: 'bfloat16[65536, 32]' = aten.mm(grad_h1_attn_ve_gate_view, grad_h1_attn_ve_gate_t_2)  # strides=(32, 1), contiguous=True, view=False
    grad_h1_attn_ve_gate_view_1: 'bfloat16[32, 2048, 32]' = aten.view(grad_h1_attn_ve_gate_mm_1, [32, 2048, 32])  # strides=(65536, 32, 1), contiguous=True, view=True
    grad_h1_attn_ve_gate_t_3: 'bfloat16[4, 32]' = aten.t(grad_h1_attn_ve_gate_t_1)  # strides=(32, 1), contiguous=True, view=True
    grad_h1_attn_ve_gate__to_copy: 'float32[4, 32]' = aten._to_copy(grad_h1_attn_ve_gate_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(32, 1), contiguous=True, view=False

    # grad of self.transformer.h.1.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:81
    # gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    grad_h1_attn_slice_backward_4: 'bfloat16[32, 2048, 512]' = aten.slice_backward(grad_h1_attn_ve_gate_view_1, [32, 2048, 512], 2, 0, 32, 1)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h1_attn_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h1_attn_mul_8, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h1_attn_view_2: 'bfloat16[32, 2048, 512]' = aten.view(grad_h1_attn_transpose_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.1.attn.c_v (Linear) → d_loss/d_c_v
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h1_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(grad_h1_attn_view_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h1_attn_c_v_t: 'bfloat16[512, 65536]' = aten.t(grad_h1_attn_c_v_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h1_attn_c_v_mm: 'bfloat16[512, 512]' = aten.mm(grad_h1_attn_c_v_t, view_14)  # strides=(512, 1), contiguous=True, view=False
    grad_h1_attn_c_v_t_1: 'bfloat16[512, 512]' = aten.t(grad_h1_attn_c_v_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h1_attn_c_v_t_2: 'bfloat16[512, 512]' = aten.t(t_8)  # strides=(512, 1), contiguous=True, view=True
    grad_h1_attn_c_v_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h1_attn_c_v_view, grad_h1_attn_c_v_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h1_attn_c_v_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h1_attn_c_v_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_151: 'bfloat16[32, 2048, 512]' = cuda_add(grad_h1_attn_slice_backward_4, grad_h1_attn_c_v_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h1_attn_c_v_t_3: 'bfloat16[512, 512]' = aten.t(grad_h1_attn_c_v_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h1_attn_c_v__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h1_attn_c_v_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.1.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h1_attn_view_3: 'bfloat16[32, 2048, 512]' = aten.view(add_147, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.1.attn.c_k (Linear) → d_loss/d_c_k
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h1_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(grad_h1_attn_view_3, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h1_attn_c_k_t: 'bfloat16[512, 65536]' = aten.t(grad_h1_attn_c_k_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h1_attn_c_k_mm: 'bfloat16[512, 512]' = aten.mm(grad_h1_attn_c_k_t, view_12)  # strides=(512, 1), contiguous=True, view=False
    grad_h1_attn_c_k_t_1: 'bfloat16[512, 512]' = aten.t(grad_h1_attn_c_k_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h1_attn_c_k_t_2: 'bfloat16[512, 512]' = aten.t(t_7)  # strides=(512, 1), contiguous=True, view=True
    grad_h1_attn_c_k_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h1_attn_c_k_view, grad_h1_attn_c_k_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h1_attn_c_k_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h1_attn_c_k_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_152: 'bfloat16[32, 2048, 512]' = cuda_add(add_151, grad_h1_attn_c_k_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h1_attn_c_k_t_3: 'bfloat16[512, 512]' = aten.t(grad_h1_attn_c_k_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h1_attn_c_k__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h1_attn_c_k_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.1.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h1_attn_view_4: 'bfloat16[32, 2048, 512]' = aten.view(add_150, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.1.attn.c_q (Linear) → d_loss/d_c_q
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h1_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(grad_h1_attn_view_4, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h1_attn_c_q_t: 'bfloat16[512, 65536]' = aten.t(grad_h1_attn_c_q_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h1_attn_c_q_mm: 'bfloat16[512, 512]' = aten.mm(grad_h1_attn_c_q_t, view_10)  # strides=(512, 1), contiguous=True, view=False
    grad_h1_attn_c_q_t_1: 'bfloat16[512, 512]' = aten.t(grad_h1_attn_c_q_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h1_attn_c_q_t_2: 'bfloat16[512, 512]' = aten.t(t_6)  # strides=(512, 1), contiguous=True, view=True
    grad_h1_attn_c_q_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h1_attn_c_q_view, grad_h1_attn_c_q_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h1_attn_c_q_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h1_attn_c_q_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_153: 'bfloat16[32, 2048, 512]' = cuda_add(add_152, grad_h1_attn_c_q_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h1_attn_c_q_t_3: 'bfloat16[512, 512]' = aten.t(grad_h1_attn_c_q_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h1_attn_c_q__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h1_attn_c_q_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.1 (Block) → d_loss/d_1
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h1_detach_1: 'float32[32, 2048, 1]' = aten.detach(detach_7)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h1__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(add_153, add_8, [512], grad_h1_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h1_getitem_1: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h1__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_154: 'bfloat16[32, 2048, 512]' = cuda_add(add_144, grad_h1_getitem_1)  # FUSED: vectorized CUDA bf16 add

    # ════════════════════════════════════════════════════════════════
    # self.value_embeds.1
    # ════════════════════════════════════════════════════════════════

    # grad of self.value_embeds.1 (Embedding) → d_loss/d_1
    # /.autoresearch_repo/train.py:273
    # ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
    grad_value_embeds1_embedding_dense_backward: 'bfloat16[8192, 512]' = aten.embedding_dense_backward(grad_h1_attn_view_1, input__32__2048, 8192, -1, False)  # strides=(512, 1), contiguous=True, view=False

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    grad_mul_11_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_154, select_3)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_11_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_154, getitem)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_11_sum: 'float32[]' = aten.sum(grad_mul_11_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    add_155: 'bfloat16[32, 2048, 512]' = cuda_add(add_141, grad_mul_11_mul)  # FUSED: vectorized CUDA bf16 add
    grad_getitem_9_select_backward: 'float32[8]' = aten.select_backward(grad_mul_11_sum, [8], 0, 1)  # strides=(1,), contiguous=True, view=False
    add_156: 'float32[8]' = aten.add.Tensor(add_142, grad_getitem_9_select_backward)  # strides=(1,), contiguous=True, view=False
    grad_mul_10_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_154, select_2)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_10_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_154, add_7)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_10_sum: 'float32[]' = aten.sum(grad_mul_10_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    grad_getitem_8_select_backward: 'float32[8]' = aten.select_backward(grad_mul_10_sum, [8], 0, 1)  # strides=(1,), contiguous=True, view=False
    add_157: 'float32[8]' = aten.add.Tensor(add_143, grad_getitem_8_select_backward)  # strides=(1,), contiguous=True, view=False

    # ════════════════════════════════════════════════════════════════
    # self.transformer.h.0
    # ════════════════════════════════════════════════════════════════

    # grad of self.transformer.h.0.mlp.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:103
    # x = self.c_proj(x)
    grad_h0_mlp_c_proj_view: 'bfloat16[65536, 512]' = aten.view(grad_mul_10_mul, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h0_mlp_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h0_mlp_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h0_mlp_c_proj_mm: 'bfloat16[512, 2048]' = aten.mm(grad_h0_mlp_c_proj_t, view_9)  # strides=(2048, 1), contiguous=True, view=False
    grad_h0_mlp_c_proj_t_1: 'bfloat16[2048, 512]' = aten.t(grad_h0_mlp_c_proj_mm)  # strides=(1, 2048), contiguous=False, view=True
    grad_h0_mlp_c_proj_t_2: 'bfloat16[512, 2048]' = aten.t(t_5)  # strides=(2048, 1), contiguous=True, view=True
    grad_h0_mlp_c_proj_mm_1: 'bfloat16[65536, 2048]' = aten.mm(grad_h0_mlp_c_proj_view, grad_h0_mlp_c_proj_t_2)  # strides=(2048, 1), contiguous=True, view=False
    grad_h0_mlp_c_proj_view_1: 'bfloat16[32, 2048, 2048]' = aten.view(grad_h0_mlp_c_proj_mm_1, [32, 2048, 2048])  # strides=(4194304, 2048, 1), contiguous=True, view=True
    grad_h0_mlp_c_proj_t_3: 'bfloat16[512, 2048]' = aten.t(grad_h0_mlp_c_proj_t_1)  # strides=(2048, 1), contiguous=True, view=True
    grad_h0_mlp_c_proj__to_copy: 'float32[32, 2048, 2048]' = aten._to_copy(grad_h0_mlp_c_proj_view_1, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(4194304, 2048, 1), contiguous=True, view=False
    grad_h0_mlp_c_proj__to_copy_1: 'float32[512, 2048]' = aten._to_copy(grad_h0_mlp_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(2048, 1), contiguous=True, view=False

    # grad of self.transformer.h.0.mlp (MLP) → d_loss/d_mlp
    # /.autoresearch_repo/train.py:102
    # x = F.relu(x).square()
    # FUSED: squared_relu backward
    grad_h0_mlp_threshold_backward: 'bfloat16[32, 2048, 2048]' = triton_squared_relu_bwd(grad_h0_mlp_c_proj__to_copy, _to_copy_5, detach_6)

    # grad of self.transformer.h.0.mlp.c_fc (Linear) → d_loss/d_c_fc
    # /.autoresearch_repo/train.py:101
    # x = self.c_fc(x)
    grad_h0_mlp_c_fc_view: 'bfloat16[65536, 2048]' = aten.view(grad_h0_mlp_threshold_backward, [65536, 2048])  # strides=(2048, 1), contiguous=True, view=True
    grad_h0_mlp_c_fc_t: 'bfloat16[2048, 65536]' = aten.t(grad_h0_mlp_c_fc_view)  # strides=(1, 2048), contiguous=False, view=True
    grad_h0_mlp_c_fc_mm: 'bfloat16[2048, 512]' = aten.mm(grad_h0_mlp_c_fc_t, view_8)  # strides=(512, 1), contiguous=True, view=False
    grad_h0_mlp_c_fc_t_1: 'bfloat16[512, 2048]' = aten.t(grad_h0_mlp_c_fc_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h0_mlp_c_fc_t_2: 'bfloat16[2048, 512]' = aten.t(t_4)  # strides=(512, 1), contiguous=True, view=True
    grad_h0_mlp_c_fc_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h0_mlp_c_fc_view, grad_h0_mlp_c_fc_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h0_mlp_c_fc_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h0_mlp_c_fc_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h0_mlp_c_fc_t_3: 'bfloat16[2048, 512]' = aten.t(grad_h0_mlp_c_fc_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h0_mlp_c_fc__to_copy: 'float32[2048, 512]' = aten._to_copy(grad_h0_mlp_c_fc_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.0 (Block) → d_loss/d_0
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h0_detach: 'float32[32, 2048, 1]' = aten.detach(detach_5)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h0__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h0_mlp_c_fc_view_1, add_6, [512], grad_h0_detach, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h0_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h0__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_158: 'bfloat16[32, 2048, 512]' = cuda_add(grad_mul_10_mul, grad_h0_getitem)  # FUSED: vectorized CUDA bf16 add

    # grad of self.transformer.h.0.attn.c_proj (Linear) → d_loss/d_c_proj
    # /.autoresearch_repo/train.py:90
    # y = self.c_proj(y)
    grad_h0_attn_c_proj_view: 'bfloat16[65536, 512]' = aten.view(add_158, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h0_attn_c_proj_t: 'bfloat16[512, 65536]' = aten.t(grad_h0_attn_c_proj_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h0_attn_c_proj_mm: 'bfloat16[512, 512]' = aten.mm(grad_h0_attn_c_proj_t, view_7)  # strides=(512, 1), contiguous=True, view=False
    grad_h0_attn_c_proj_t_1: 'bfloat16[512, 512]' = aten.t(grad_h0_attn_c_proj_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h0_attn_c_proj_t_2: 'bfloat16[512, 512]' = aten.t(t_3)  # strides=(512, 1), contiguous=True, view=True
    grad_h0_attn_c_proj_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h0_attn_c_proj_view, grad_h0_attn_c_proj_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h0_attn_c_proj_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h0_attn_c_proj_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h0_attn_c_proj_t_3: 'bfloat16[512, 512]' = aten.t(grad_h0_attn_c_proj_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h0_attn_c_proj__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h0_attn_c_proj_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.0.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:89
    # y = y.contiguous().view(B, T, -1)
    grad_h0_attn_view: 'bfloat16[32, 2048, 4, 128]' = aten.view(grad_h0_attn_c_proj_view_1, [32, 2048, 4, 128])  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h0_attn_transpose: 'bfloat16[32, 4, 2048, 128]' = aten.transpose.int(grad_h0_attn_view, 1, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h0_attn_detach: 'bfloat16[32, 4, 2048, 128]' = aten.detach(detach_4)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h0_attn__scaled_dot_product_cudnn_attention_backward = aten._scaled_dot_product_cudnn_attention_backward(grad_h0_attn_transpose, transpose, transpose_1, transpose_2, grad_h0_attn_detach, getitem_9, getitem_14, getitem_15, where, None, None, 2048, 2048, 0.0, False)  # out0: strides=(1048576, 128, 512, 1), contiguous=False; out1: strides=(1048576, 128, 512, 1), contiguous=False; out2: strides=(1048576, 128, 512, 1), contiguous=False, view=False
    grad_h0_attn_getitem: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h0_attn__scaled_dot_product_cudnn_attention_backward, 0)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h0_attn_getitem_1: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h0_attn__scaled_dot_product_cudnn_attention_backward, 1)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h0_attn_getitem_2: 'bfloat16[32, 4, 2048, 128]' = operator.getitem(grad_h0_attn__scaled_dot_product_cudnn_attention_backward, 2)  # strides=(1048576, 128, 512, 1), contiguous=False, view=True
    grad_h0_attn_transpose_1: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h0_attn_getitem_2, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h0_attn_transpose_2: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h0_attn_getitem_1, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h0_attn_transpose_3: 'bfloat16[32, 2048, 4, 128]' = aten.transpose.int(grad_h0_attn_getitem, 1, 2)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h0_attn_detach_1: 'float32[32, 2048, 4, 1]' = aten.detach(detach_3)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h0_attn__fused_rms_norm_backward = aten._fused_rms_norm_backward(grad_h0_attn_transpose_2, cat_1, [128], grad_h0_attn_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h0_attn_getitem_3: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h0_attn__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h0_attn_detach_2: 'float32[32, 2048, 4, 1]' = aten.detach(detach_2)  # strides=(8192, 4, 1, 1), contiguous=True, view=True
    grad_h0_attn__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(grad_h0_attn_transpose_3, cat, [128], grad_h0_attn_detach_2, None, [True, False])  # out0: strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h0_attn_getitem_4: 'bfloat16[32, 2048, 4, 128]' = operator.getitem(grad_h0_attn__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 128, 1), contiguous=True, view=True
    grad_h0_attn_slice: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h0_attn_getitem_3, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h0_attn_slice_1: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h0_attn_getitem_3, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h0_attn_mul: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h0_attn_slice_1, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h0_attn_mul_1: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h0_attn_slice_1, neg_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h0_attn_mul_2: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h0_attn_slice, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_159: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h0_attn_mul, grad_h0_attn_mul_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h0_attn_mul_3: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h0_attn_slice, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_160: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h0_attn_mul_1, grad_h0_attn_mul_3)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h0_attn_slice_backward: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_159, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h0_attn_slice_backward_1: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_160, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_161: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h0_attn_slice_backward, grad_h0_attn_slice_backward_1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h0_attn_slice_2: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h0_attn_getitem_4, 3, 0, 64)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h0_attn_slice_3: 'bfloat16[32, 2048, 4, 64]' = aten.slice.Tensor(grad_h0_attn_getitem_4, 3, 64, 128)  # strides=(1048576, 512, 128, 1), contiguous=False, view=True
    grad_h0_attn_mul_4: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h0_attn_slice_3, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h0_attn_mul_5: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h0_attn_slice_3, neg)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h0_attn_mul_6: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h0_attn_slice_2, slice_2)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_162: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h0_attn_mul_4, grad_h0_attn_mul_6)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h0_attn_mul_7: 'bfloat16[32, 2048, 4, 64]' = aten.mul.Tensor(grad_h0_attn_slice_2, slice_1)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    add_163: 'bfloat16[32, 2048, 4, 64]' = aten.add.Tensor(grad_h0_attn_mul_5, grad_h0_attn_mul_7)  # strides=(524288, 256, 64, 1), contiguous=True, view=False
    grad_h0_attn_slice_backward_2: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_162, [32, 2048, 4, 128], 3, 64, 9223372036854775807, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h0_attn_slice_backward_3: 'bfloat16[32, 2048, 4, 128]' = aten.slice_backward(add_163, [32, 2048, 4, 128], 3, 0, 64, 1)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    add_164: 'bfloat16[32, 2048, 4, 128]' = aten.add.Tensor(grad_h0_attn_slice_backward_2, grad_h0_attn_slice_backward_3)  # strides=(1048576, 512, 128, 1), contiguous=True, view=False
    grad_h0_attn_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h0_attn_transpose_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.0.attn.c_v (Linear) → d_loss/d_c_v
    # /.autoresearch_repo/train.py:76
    # v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h0_attn_c_v_view: 'bfloat16[65536, 512]' = aten.view(grad_h0_attn_view_1, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h0_attn_c_v_t: 'bfloat16[512, 65536]' = aten.t(grad_h0_attn_c_v_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h0_attn_c_v_mm: 'bfloat16[512, 512]' = aten.mm(grad_h0_attn_c_v_t, view_4)  # strides=(512, 1), contiguous=True, view=False
    grad_h0_attn_c_v_t_1: 'bfloat16[512, 512]' = aten.t(grad_h0_attn_c_v_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h0_attn_c_v_t_2: 'bfloat16[512, 512]' = aten.t(t_2)  # strides=(512, 1), contiguous=True, view=True
    grad_h0_attn_c_v_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h0_attn_c_v_view, grad_h0_attn_c_v_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h0_attn_c_v_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h0_attn_c_v_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    grad_h0_attn_c_v_t_3: 'bfloat16[512, 512]' = aten.t(grad_h0_attn_c_v_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h0_attn_c_v__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h0_attn_c_v_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.0.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h0_attn_view_2: 'bfloat16[32, 2048, 512]' = aten.view(add_161, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.0.attn.c_k (Linear) → d_loss/d_c_k
    # /.autoresearch_repo/train.py:75
    # k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    grad_h0_attn_c_k_view: 'bfloat16[65536, 512]' = aten.view(grad_h0_attn_view_2, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h0_attn_c_k_t: 'bfloat16[512, 65536]' = aten.t(grad_h0_attn_c_k_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h0_attn_c_k_mm: 'bfloat16[512, 512]' = aten.mm(grad_h0_attn_c_k_t, view_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h0_attn_c_k_t_1: 'bfloat16[512, 512]' = aten.t(grad_h0_attn_c_k_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h0_attn_c_k_t_2: 'bfloat16[512, 512]' = aten.t(t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h0_attn_c_k_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h0_attn_c_k_view, grad_h0_attn_c_k_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h0_attn_c_k_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h0_attn_c_k_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_165: 'bfloat16[32, 2048, 512]' = cuda_add(grad_h0_attn_c_v_view_1, grad_h0_attn_c_k_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h0_attn_c_k_t_3: 'bfloat16[512, 512]' = aten.t(grad_h0_attn_c_k_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h0_attn_c_k__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h0_attn_c_k_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.0.attn (CausalSelfAttention) → d_loss/d_attn
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h0_attn_view_3: 'bfloat16[32, 2048, 512]' = aten.view(add_164, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True

    # grad of self.transformer.h.0.attn.c_q (Linear) → d_loss/d_c_q
    # /.autoresearch_repo/train.py:74
    # q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    grad_h0_attn_c_q_view: 'bfloat16[65536, 512]' = aten.view(grad_h0_attn_view_3, [65536, 512])  # strides=(512, 1), contiguous=True, view=True
    grad_h0_attn_c_q_t: 'bfloat16[512, 65536]' = aten.t(grad_h0_attn_c_q_view)  # strides=(1, 512), contiguous=False, view=True
    grad_h0_attn_c_q_mm: 'bfloat16[512, 512]' = aten.mm(grad_h0_attn_c_q_t, view)  # strides=(512, 1), contiguous=True, view=False
    grad_h0_attn_c_q_t_1: 'bfloat16[512, 512]' = aten.t(grad_h0_attn_c_q_mm)  # strides=(1, 512), contiguous=False, view=True
    grad_h0_attn_c_q_t_2: 'bfloat16[512, 512]' = aten.t(t)  # strides=(512, 1), contiguous=True, view=True
    grad_h0_attn_c_q_mm_1: 'bfloat16[65536, 512]' = aten.mm(grad_h0_attn_c_q_view, grad_h0_attn_c_q_t_2)  # strides=(512, 1), contiguous=True, view=False
    grad_h0_attn_c_q_view_1: 'bfloat16[32, 2048, 512]' = aten.view(grad_h0_attn_c_q_mm_1, [32, 2048, 512])  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_166: 'bfloat16[32, 2048, 512]' = cuda_add(add_165, grad_h0_attn_c_q_view_1)  # FUSED: vectorized CUDA bf16 add
    grad_h0_attn_c_q_t_3: 'bfloat16[512, 512]' = aten.t(grad_h0_attn_c_q_t_1)  # strides=(512, 1), contiguous=True, view=True
    grad_h0_attn_c_q__to_copy: 'float32[512, 512]' = aten._to_copy(grad_h0_attn_c_q_t_3, dtype=torch.float32, layout=torch.strided, device=torch.device('cuda:0'))  # strides=(512, 1), contiguous=True, view=False

    # grad of self.transformer.h.0 (Block) → d_loss/d_0
    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_h0_detach_1: 'float32[32, 2048, 1]' = aten.detach(detach_1)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_h0__fused_rms_norm_backward_1 = aten._fused_rms_norm_backward(add_166, add, [512], grad_h0_detach_1, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_h0_getitem_1: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_h0__fused_rms_norm_backward_1, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True
    add_167: 'bfloat16[32, 2048, 512]' = cuda_add(add_158, grad_h0_getitem_1)  # FUSED: vectorized CUDA bf16 add

    # /.autoresearch_repo/train.py:272
    # x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    grad_mul_1_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_167, select_1)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_1_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_167, getitem)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_1_sum: 'float32[]' = aten.sum(grad_mul_1_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    add_168: 'bfloat16[32, 2048, 512]' = cuda_add(add_155, grad_mul_1_mul)  # FUSED: vectorized CUDA bf16 add
    grad_getitem_3_select_backward: 'float32[8]' = aten.select_backward(grad_mul_1_sum, [8], 0, 0)  # strides=(1,), contiguous=True, view=False
    add_169: 'float32[8]' = aten.add.Tensor(add_156, grad_getitem_3_select_backward)  # strides=(1,), contiguous=True, view=False
    grad_mul_mul: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_167, select)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_mul_1: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(add_167, getitem)  # strides=(1048576, 512, 1), contiguous=True, view=False
    grad_mul_sum: 'float32[]' = aten.sum(grad_mul_mul_1, dtype=torch.float32)  # strides=(), contiguous=True, view=False
    add_170: 'bfloat16[32, 2048, 512]' = cuda_add(add_168, grad_mul_mul)  # FUSED: vectorized CUDA bf16 add
    grad_getitem_2_select_backward: 'float32[8]' = aten.select_backward(grad_mul_sum, [8], 0, 0)  # strides=(1,), contiguous=True, view=False
    add_171: 'float32[8]' = aten.add.Tensor(add_157, grad_getitem_2_select_backward)  # strides=(1,), contiguous=True, view=False

    # /.venv/lib/python3.12/site-packages/torch/nn/functional.py:2954
    # return torch.rms_norm(input, normalized_shape, weight, eps)
    grad_rms_norm_detach: 'float32[32, 2048, 1]' = aten.detach(detach)  # strides=(2048, 1, 1), contiguous=True, view=True
    grad_rms_norm__fused_rms_norm_backward = aten._fused_rms_norm_backward(add_170, embedding, [512], grad_rms_norm_detach, None, [True, False])  # out0: strides=(1048576, 512, 1), contiguous=True, view=False
    grad_rms_norm_getitem: 'bfloat16[32, 2048, 512]' = operator.getitem(grad_rms_norm__fused_rms_norm_backward, 0)  # strides=(1048576, 512, 1), contiguous=True, view=True

    # ════════════════════════════════════════════════════════════════
    # self.transformer.wte
    # ════════════════════════════════════════════════════════════════

    # grad of self.transformer.wte (Embedding) → d_loss/d_wte
    # /.autoresearch_repo/train.py:268
    # x = self.transformer.wte(idx)
    grad_wte_embedding_dense_backward: 'bfloat16[8192, 512]' = aten.embedding_dense_backward(grad_rms_norm_getitem, input__32__2048, 8192, -1, False)  # strides=(512, 1), contiguous=True, view=False
    return (None, None, None, grad_wte_embedding_dense_backward, add_171, add_169, grad_h0_attn_c_q__to_copy, grad_h0_attn_c_k__to_copy, grad_h0_attn_c_v__to_copy, grad_h0_attn_c_proj__to_copy, grad_h0_mlp_c_fc__to_copy, grad_h0_mlp_c_proj__to_copy_1, grad_value_embeds1_embedding_dense_backward, grad_h1_attn_c_q__to_copy, grad_h1_attn_c_k__to_copy, grad_h1_attn_c_v__to_copy, grad_h1_attn_ve_gate__to_copy, grad_h1_attn_c_proj__to_copy, grad_h1_mlp_c_fc__to_copy, grad_h1_mlp_c_proj__to_copy_1, grad_h2_attn_c_q__to_copy, grad_h2_attn_c_k__to_copy, grad_h2_attn_c_v__to_copy, grad_h2_attn_c_proj__to_copy, grad_h2_mlp_c_fc__to_copy, grad_h2_mlp_c_proj__to_copy_1, grad_value_embeds3_embedding_dense_backward, grad_h3_attn_c_q__to_copy, grad_h3_attn_c_k__to_copy, grad_h3_attn_c_v__to_copy, grad_h3_attn_ve_gate__to_copy, grad_h3_attn_c_proj__to_copy, grad_h3_mlp_c_fc__to_copy, grad_h3_mlp_c_proj__to_copy_1, grad_h4_attn_c_q__to_copy, grad_h4_attn_c_k__to_copy, grad_h4_attn_c_v__to_copy, grad_h4_attn_c_proj__to_copy, grad_h4_mlp_c_fc__to_copy, grad_h4_mlp_c_proj__to_copy_1, grad_value_embeds5_embedding_dense_backward, grad_h5_attn_c_q__to_copy, grad_h5_attn_c_k__to_copy, grad_h5_attn_c_v__to_copy, grad_h5_attn_ve_gate__to_copy, grad_h5_attn_c_proj__to_copy, grad_h5_mlp_c_fc__to_copy, grad_h5_mlp_c_proj__to_copy_1, grad_h6_attn_c_q__to_copy, grad_h6_attn_c_k__to_copy, grad_h6_attn_c_v__to_copy, grad_h6_attn_c_proj__to_copy, grad_h6_mlp_c_fc__to_copy, grad_h6_mlp_c_proj__to_copy_1, grad_value_embeds7_embedding_dense_backward, grad_h7_attn_c_q__to_copy, grad_h7_attn_c_k__to_copy, grad_h7_attn_c_v__to_copy, grad_h7_attn_ve_gate__to_copy, grad_h7_attn_c_proj__to_copy, grad_h7_mlp_c_fc__to_copy, grad_h7_mlp_c_proj__to_copy_1, grad_lm_head__to_copy, None,)


