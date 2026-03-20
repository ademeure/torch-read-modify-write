#!/usr/bin/env python3
"""Apply kernel fusion optimizations to an aten file, one at a time.

Each optimization is a function that takes the source code string and returns
modified source code. They are designed to be applied IN ORDER, each building
on the previous.

Usage:
    python scripts/apply_optimizations.py                    # Apply all
    python scripts/apply_optimizations.py softcap_fwd        # Apply one
    python scripts/apply_optimizations.py --list              # List available
    python scripts/apply_optimizations.py --dry-run all       # Preview changes
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ATEN_FILE = (
    Path(__file__).parent.parent
    / ".bench_cache/ar_aten_replay/GPT_07554748d291_2a_train_aten.py"
)

# ======================================================================
# Triton kernel source code
# ======================================================================

TRITON_IMPORTS = """\
import triton
import triton.language as tl
"""

TRITON_ROPE_FWD = '''\
# ── Triton fused RoPE kernel ─────────────────────────────────────────
import triton
import triton.language as tl

@triton.jit
def _rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_xb, stride_xt, stride_xh, stride_xd,
    stride_cb, stride_ct, stride_ch, stride_cd,
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    D: tl.constexpr, HALF_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (T * H)
    rem = pid % (T * H)
    t = rem // H
    h = rem % H

    base_x = b * stride_xb + t * stride_xt + h * stride_xh
    base_c = 0 * stride_cb + t * stride_ct + 0 * stride_ch

    for d in range(HALF_D):
        x1 = tl.load(x_ptr + base_x + d * stride_xd)
        x2 = tl.load(x_ptr + base_x + (d + HALF_D) * stride_xd)
        c = tl.load(cos_ptr + base_c + d * stride_cd)
        s = tl.load(sin_ptr + base_c + d * stride_cd)
        tl.store(out_ptr + base_x + d * stride_xd, x1 * c + x2 * s)
        tl.store(out_ptr + base_x + (d + HALF_D) * stride_xd, x1 * (-s) + x2 * c)

def triton_rope_fwd(x, cos, sin):
    """Fused RoPE: slice + mul + neg + mul + add + cat -> single kernel."""
    B, T, H, D = x.shape
    out = torch.empty_like(x)
    grid = (B * T * H,)
    _rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1), cos.stride(2), cos.stride(3),
        B=B, T=T, H=H, D=D, HALF_D=D // 2,
    )
    return out
# ── End Triton fused RoPE kernel ─────────────────────────────────────
'''

TRITON_SOFTCAP_FWD = '''\
# ── Triton softcap forward kernel ────────────────────────────────────
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as _libdevice

@triton.jit
def _softcap_fwd_kernel(
    x_ptr, out_ptr, tanh_ptr, n,
    softcap: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    t = _libdevice.tanh(x / softcap)
    tl.store(out_ptr + offs, t * softcap, mask=mask)
    tl.store(tanh_ptr + offs, t, mask=mask)

def triton_softcap_fwd(x_bf16, softcap=15):
    """Fused softcap: bf16->fp32, div, tanh, mul. Returns (softcapped_fp32, tanh_fp32)."""
    n = x_bf16.numel()
    out = torch.empty(x_bf16.shape, dtype=torch.float32, device=x_bf16.device)
    tanh_out = torch.empty(x_bf16.shape, dtype=torch.float32, device=x_bf16.device)
    _softcap_fwd_kernel[((n + 1023) // 1024,)](x_bf16, out, tanh_out, n, softcap=softcap, BLOCK=1024)
    return out, tanh_out
# ── End Triton softcap forward kernel ────────────────────────────────
'''

TRITON_SOFTCAP_BWD = '''\
# ── Triton softcap backward kernel ───────────────────────────────────
import triton
import triton.language as tl

@triton.jit
def _softcap_bwd_kernel(
    grad_ptr, tanh_ptr, out_ptr, n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    t = tl.load(tanh_ptr + offs, mask=mask).to(tl.float32)
    # d/dx [softcap * tanh(x/softcap)] = 1 - tanh^2(x/softcap)
    out = g * (1.0 - t * t)
    tl.store(out_ptr + offs, out.to(tl.bfloat16), mask=mask)

def triton_softcap_bwd(grad_fp32, tanh_fp32):
    """Fused softcap backward: grad * (1 - tanh^2) -> bf16."""
    n = grad_fp32.numel()
    out = torch.empty(grad_fp32.shape, dtype=torch.bfloat16, device=grad_fp32.device)
    _softcap_bwd_kernel[((n + 1023) // 1024,)](grad_fp32, tanh_fp32, out, n, BLOCK=1024)
    return out
# ── End Triton softcap backward kernel ───────────────────────────────
'''

TRITON_LAMBDA_SCALE = '''\
# ── Triton lambda scaling kernel ─────────────────────────────────────
import triton
import triton.language as tl

@triton.jit
def _lambda_scale_kernel(
    x_ptr, x0_ptr, out_ptr, a_scalar, b_scalar, n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    x0 = tl.load(x0_ptr + offs, mask=mask).to(tl.float32)
    out = a_scalar * x + b_scalar * x0
    tl.store(out_ptr + offs, out.to(tl.bfloat16), mask=mask)

def triton_lambda_scale(x, x0, a_scalar, b_scalar):
    """Fused: a*x + b*x0 in one kernel."""
    n = x.numel()
    out = torch.empty_like(x)
    a_val = a_scalar.item() if hasattr(a_scalar, 'item') else float(a_scalar)
    b_val = b_scalar.item() if hasattr(b_scalar, 'item') else float(b_scalar)
    _lambda_scale_kernel[((n + 1023) // 1024,)](x, x0, out, a_val, b_val, n, BLOCK=1024)
    return out
# ── End Triton lambda scaling kernel ─────────────────────────────────
'''

TRITON_SQRELU_BWD = '''\
# ── Triton squared ReLU backward kernel ──────────────────────────────
import triton
import triton.language as tl

@triton.jit
def _sqrelu_bwd_kernel(
    grad_ptr, relu_fp32_ptr, relu_bf16_ptr, out_ptr, n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    r = tl.load(relu_fp32_ptr + offs, mask=mask).to(tl.float32)
    rb = tl.load(relu_bf16_ptr + offs, mask=mask)
    # d/dx [relu(x)^2] = 2*relu(x) * (x > 0); threshold_backward handles (x > 0)
    # Combined: grad * 2 * relu * (relu_bf16 > 0) -> bf16
    deriv = g * 2.0 * r
    out = tl.where(rb > 0, deriv, 0.0)
    tl.store(out_ptr + offs, out.to(tl.bfloat16), mask=mask)

def triton_squared_relu_bwd(grad_fp32, relu_fp32, relu_bf16):
    """Fused squared ReLU backward: grad * 2 * relu * (relu > 0) -> bf16."""
    n = grad_fp32.numel()
    out = torch.empty(grad_fp32.shape, dtype=torch.bfloat16, device=grad_fp32.device)
    _sqrelu_bwd_kernel[((n + 1023) // 1024,)](grad_fp32, relu_fp32, relu_bf16, out, n, BLOCK=1024)
    return out
# ── End Triton squared ReLU backward kernel ──────────────────────────
'''

TRITON_SQRELU_FWD = '''\
# ── Triton squared ReLU forward kernel ───────────────────────────────
import triton
import triton.language as tl

@triton.jit
def _sqrelu_fwd_kernel(
    x_ptr, relu_bf16_ptr, relu_fp32_ptr, sq_bf16_ptr, n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # relu in bf16
    r_bf16 = tl.where(x > 0, x, 0.0)
    tl.store(relu_bf16_ptr + offs, r_bf16, mask=mask)
    # relu in fp32 for backward
    r_fp32 = r_bf16.to(tl.float32)
    tl.store(relu_fp32_ptr + offs, r_fp32, mask=mask)
    # squared in bf16 for c_proj input
    sq = (r_fp32 * r_fp32).to(tl.bfloat16)
    tl.store(sq_bf16_ptr + offs, sq, mask=mask)

def triton_squared_relu_fwd(x_bf16):
    """Fused relu + square: returns (relu_bf16, relu_fp32, squared_bf16)."""
    n = x_bf16.numel()
    relu_bf16 = torch.empty_like(x_bf16)
    relu_fp32 = torch.empty(x_bf16.shape, dtype=torch.float32, device=x_bf16.device)
    sq_bf16 = torch.empty_like(x_bf16)
    _sqrelu_fwd_kernel[((n + 1023) // 1024,)](x_bf16, relu_bf16, relu_fp32, sq_bf16, n, BLOCK=1024)
    return relu_bf16, relu_fp32, sq_bf16
# ── End Triton squared ReLU forward kernel ───────────────────────────
'''

# ======================================================================
# Kernel insertion helper
# ======================================================================

WEIGHTS_MARKER = (
    "# ======================================================================\n"
    "# WEIGHTS / PARAMETERS"
)


def insert_kernel(source: str, kernel_code: str) -> str:
    """Insert kernel code before WEIGHTS section."""
    return source.replace(WEIGHTS_MARKER, kernel_code + "\n" + WEIGHTS_MARKER, 1)


# ======================================================================
# Optimization 1: Deduplicate neg(sin) and attention mask
# ======================================================================

def apply_dedup_neg_sin_and_mask(source: str) -> str:
    """Compute neg(sin) and attention mask ONCE, reuse across all layers."""
    count = 0

    # --- neg(sin) deduplication ---
    # Add _neg_sin computation right after getitem_1_slice
    source = source.replace(
        "    getitem_1_slice: 'bfloat16[1, 2048, 1, 64]' = aten.slice.Tensor(sin, 1, 0, 2048)",
        "    getitem_1_slice: 'bfloat16[1, 2048, 1, 64]' = aten.slice.Tensor(sin, 1, 0, 2048)"
        "  # strides=(1310720, 64, 64, 1), contiguous=True, view=True\n"
        "    _neg_sin: 'bfloat16[1, 2048, 1, 64]' = aten.neg(getitem_1_slice)"
        "  # DEDUP: compute neg(sin) once",
        1,
    )
    # Hmm, that won't work because of the trailing comment. Let me use regex.
    # Revert and use a cleaner approach.
    source = source  # no-op, the replace above won't match due to comment

    # Actually insert _neg_sin after the getitem_1_slice line (match the full line with comment)
    neg_sin_insert = re.compile(
        r"(    getitem_1_slice: 'bfloat16\[1, 2048, 1, 64\]' = aten\.slice\.Tensor\(sin, 1, 0, 2048\).*\n)"
    )
    m = neg_sin_insert.search(source)
    if m:
        insert_pos = m.end()
        insert_line = "    _neg_sin: 'bfloat16[1, 2048, 1, 64]' = aten.neg(getitem_1_slice)  # DEDUP: compute neg(sin) once\n"
        source = source[:insert_pos] + insert_line + source[insert_pos:]
        count += 1

    # Replace all per-layer neg(getitem_1_slice) lines with alias to _neg_sin
    # Pattern: hN_attn_neg: ... = aten.neg(getitem_1_slice) ...
    # Also: hN_attn_neg_1: ... = aten.neg(getitem_1_slice) ...
    neg_pattern = re.compile(
        r"    (h\d+_attn_neg(?:_1)?): '[^']*' = aten\.neg\(getitem_1_slice\).*\n"
    )

    def _replace_neg(m):
        nonlocal count
        count += 1
        varname = m.group(1)
        return f"    {varname} = _neg_sin  # DEDUP: reuse precomputed neg(sin)\n"

    source = neg_pattern.sub(_replace_neg, source)

    # --- Attention mask deduplication ---
    # Layers 0,1,2,4,5,6 use window_size=1024; layers 3,7 use full attention (None mask)
    # The mask pattern is 9 lines from arange to where.
    # We compute it once after the neg_sin line as _attn_mask.

    # First, find and keep the h0 mask computation, rename output to _attn_mask
    mask_pattern = re.compile(
        r"(    (h\d+)_attn_arange: 'int64\[2048\]' = aten\.arange\(2048, device=torch\.device\('cuda:0'\), pin_memory=False\).*\n)"
        r"(    \2_attn_unsqueeze(?:_\d+)?: 'int64\[2048, 1\]' = aten\.unsqueeze\(\2_attn_arange, 1\).*\n)"
        r"(    \2_attn_add(?:_\d+)?: 'int64\[2048, 1\]' = aten\.add\.Tensor\(\2_attn_unsqueeze(?:_\d+)?, 0\).*\n)"
        r"(    \2_attn_arange_1: 'int64\[2048\]' = aten\.arange\(2048, device=torch\.device\('cuda:0'\), pin_memory=False\).*\n)"
        r"(    \2_attn_unsqueeze_\d+: 'int64\[1, 2048\]' = aten\.unsqueeze\(\2_attn_arange_1, 0\).*\n)"
        r"(    \2_attn_le: 'bool\[2048, 2048\]' = aten\.le\.Tensor\(.*\n)"
        r"(    \2_attn_sub: 'int64\[2048, 2048\]' = aten\.sub\.Tensor\(.*\n)"
        r"(    \2_attn_le_1: 'bool\[2048, 2048\]' = aten\.le\.Scalar\(\2_attn_sub, 1024\).*\n)"
        r"(    \2_attn_bitwise_and: 'bool\[2048, 2048\]' = aten\.bitwise_and\.Tensor\(.*\n)"
        r"(    \2_attn_scalar_tensor: 'bfloat16\[\]' = aten\.scalar_tensor\(-65504\.0.*\n)"
        r"(    \2_attn_scalar_tensor_1: 'bfloat16\[\]' = aten\.scalar_tensor\(0\.0.*\n)"
        r"(    (\2_attn_where): 'bfloat16\[2048, 2048\]' = aten\.where\.self\(.*\n)"
    )

    mask_count = [0]
    first_mask_kept = [False]

    def _replace_mask(m):
        prefix = m.group(2)
        where_var = m.group(14)
        mask_count[0] += 1

        if not first_mask_kept[0]:
            # Keep the first mask block, add _attn_mask alias
            first_mask_kept[0] = True
            return m.group(0) + f"    _attn_mask = {where_var}  # DEDUP: reuse this mask for all windowed layers\n"
        else:
            # Replace subsequent mask blocks with alias
            return f"    {where_var} = _attn_mask  # DEDUP: reuse precomputed attention mask\n"

    source = mask_pattern.sub(_replace_mask, source)
    count += mask_count[0]

    print(f"  [dedup_neg_sin_mask] {count} replacements (neg_sin + mask dedup)")
    return source


# ======================================================================
# Optimization 2: Fuse softcap forward
# ======================================================================

def apply_softcap_fwd(source: str) -> str:
    """Replace softcap forward: _to_copy(bf16->fp32) + div/15 + tanh + mul*15."""
    source = insert_kernel(source, TRITON_SOFTCAP_FWD)

    # Pattern (with possible blank/comment lines between _to_copy and div):
    #   float_1__to_copy: fp32 = aten._to_copy(lm_head__unsafe_view, dtype=torch.float32)
    #   <blank line>
    #   <comment lines>
    #   truediv_div: fp32 = aten.div.Tensor(float_1__to_copy, 15)
    #   tanh_tanh: fp32 = aten.tanh(truediv_div)
    #   tanh_detach: fp32 = aten.detach(tanh_tanh)
    #   mul_88_mul: fp32 = aten.mul.Tensor(tanh_tanh, 15)
    pat = re.compile(
        r"    (float_1__to_copy): '[^']*' = aten\._to_copy\((\w+), dtype=torch\.float32\).*\n"
        r"(?:\n|    #[^\n]*\n)*"  # blank lines and comments
        r"    (truediv_div): '[^']*' = aten\.div\.Tensor\(\1, 15\).*\n"
        r"    (tanh_tanh): '[^']*' = aten\.tanh\(\3\).*\n"
        r"    (tanh_detach): '[^']*' = aten\.detach\(\4\).*\n"
        r"    (\w+): '[^']*' = aten\.mul\.Tensor\(\4, 15\).*\n"
    )

    def _replace(m):
        input_var = m.group(2)
        tanh_detach_var = m.group(5)
        output_var = m.group(6)
        return (
            f"    {output_var}, {tanh_detach_var} = triton_softcap_fwd({input_var}, softcap=15)"
            f"  # FUSED: softcap forward via Triton\n"
        )

    new_source, n = pat.subn(_replace, source)
    print(f"  [softcap_fwd] {n} replacements")
    return new_source


# ======================================================================
# Optimization 3: Fuse softcap backward
# ======================================================================

def apply_softcap_bwd(source: str) -> str:
    """Replace softcap backward sequence with fused Triton kernel."""
    source = insert_kernel(source, TRITON_SOFTCAP_BWD)

    # Pattern in backward:
    #   grad_mul_88_mul: fp32 = aten.mul.Tensor(grad_view_36_view, 15)
    #   grad_tanh_detach: fp32 = aten.detach(detach_54)
    #   grad_tanh_tanh_backward: fp32 = aten.tanh_backward(grad_mul_88_mul, grad_tanh_detach)
    #   grad_truediv_div: fp32 = aten.div.Tensor(grad_tanh_tanh_backward, 15)
    #   <blank line or comment>
    #   grad_float_1__to_copy: bf16 = aten._to_copy(grad_truediv_div, dtype=torch.bfloat16, ...)
    pat = re.compile(
        r"    (grad_mul_88_mul): '[^']*' = aten\.mul\.Tensor\((\w+), 15\).*\n"
        r"    (grad_tanh_detach): '[^']*' = aten\.detach\((\w+)\).*\n"
        r"    (\w+): '[^']*' = aten\.tanh_backward\(\1, \3\).*\n"
        r"    (\w+): '[^']*' = aten\.div\.Tensor\(\5, 15\).*\n"
        r"\n"
        r"(?:    #[^\n]*\n)*"  # comment lines
        r"    (grad_float_1__to_copy): '[^']*' = aten\._to_copy\(\6, dtype=torch\.bfloat16.*\n"
    )

    def _replace(m):
        grad_in = m.group(2)  # grad_view_36_view
        tanh_saved = m.group(4)  # detach_54 (saved tanh from forward)
        output_var = m.group(7)  # grad_float_1__to_copy
        return (
            f"    {output_var}: 'bfloat16[32, 2048, 8192]' = triton_softcap_bwd({grad_in}, {tanh_saved})"
            f"  # FUSED: softcap backward via Triton\n"
        )

    new_source, n = pat.subn(_replace, source)
    print(f"  [softcap_bwd] {n} replacements")
    return new_source


# ======================================================================
# Optimization 4: Fuse lambda scaling
# ======================================================================

def apply_lambda_scale(source: str) -> str:
    """Replace per-layer lambda scaling: select a -> mul(a,x) -> select b -> mul(b,x0) -> add."""
    source = insert_kernel(source, TRITON_LAMBDA_SCALE)

    # Pattern:
    #   getitem_N_select: fp32[] = aten.select.int(resid_lambdas, 0, I)
    #   mul_N_mul: bf16[...] = aten.mul.Tensor(getitem_N_select, X)
    #   getitem_M_select: fp32[] = aten.select.int(x0_lambdas, 0, I)
    #   mul_M_mul: bf16[...] = aten.mul.Tensor(getitem_M_select, X0)
    #   add_N_add: bf16[...] = aten.add.Tensor(mul_N_mul, mul_M_mul)
    pat = re.compile(
        r"    (\w+_select): '[^']*' = aten\.select\.int\(resid_lambdas, 0, (\d+)\).*\n"
        r"    (\w+_mul): '[^']*' = aten\.mul\.Tensor\(\1, (\w+)\).*\n"
        r"    (\w+_select): '[^']*' = aten\.select\.int\(x0_lambdas, 0, \2\).*\n"
        r"    (\w+_mul): '[^']*' = aten\.mul\.Tensor\(\5, (\w+)\).*\n"
        r"    (\w+_add): '[^']*' = aten\.add\.Tensor\(\3, \6\).*\n"
    )

    count = [0]

    def _replace(m):
        count[0] += 1
        resid_select = m.group(1)
        idx = m.group(2)
        x_var = m.group(4)
        x0_select = m.group(5)
        x0_var = m.group(7)
        output_var = m.group(8)
        return (
            f"    {resid_select}: 'float32[]' = aten.select.int(resid_lambdas, 0, {idx})\n"
            f"    {x0_select}: 'float32[]' = aten.select.int(x0_lambdas, 0, {idx})\n"
            f"    {output_var} = triton_lambda_scale({x_var}, {x0_var}, {resid_select}, {x0_select})"
            f"  # FUSED: lambda scaling via Triton\n"
        )

    new_source = pat.sub(_replace, source)
    print(f"  [lambda_scale] {count[0]} replacements")
    return new_source


# ======================================================================
# Optimization 5: Fuse squared ReLU backward
# ======================================================================

def apply_sqrelu_bwd(source: str) -> str:
    """Replace squared ReLU backward: pow(x,1) -> mul(2,...) -> mul(grad,...) -> _to_copy(bf16) -> threshold_backward."""
    source = insert_kernel(source, TRITON_SQRELU_BWD)

    # Pattern per layer (h0..h7):
    #   grad_hN_mlp_pow: fp32 = aten.pow.Tensor_Scalar(RELU_FP32, 1.0)
    #   grad_hN_mlp_mul: fp32 = aten.mul.Scalar(grad_hN_mlp_pow, 2.0)
    #   grad_hN_mlp_mul_1: fp32 = aten.mul.Tensor(GRAD_FP32, grad_hN_mlp_mul)
    #   grad_hN_mlp__to_copy: bf16 = aten._to_copy(grad_hN_mlp_mul_1, dtype=torch.bfloat16, ...)
    #   grad_hN_mlp_detach: bf16 = aten.detach(RELU_BF16_SAVED)
    #   grad_hN_mlp_threshold_backward: bf16 = aten.threshold_backward(grad_hN_mlp__to_copy, grad_hN_mlp_detach, 0)
    pat = re.compile(
        r"    (grad_h\d+_mlp_pow): '[^']*' = aten\.pow\.Tensor_Scalar\((\w+), 1\.0\).*\n"
        r"    (grad_h\d+_mlp_mul)\b: '[^']*' = aten\.mul\.Scalar\(\1, 2\.0\).*\n"
        r"    (grad_h\d+_mlp_mul_1): '[^']*' = aten\.mul\.Tensor\((\w+), \3\).*\n"
        r"    (grad_h\d+_mlp__to_copy): '[^']*' = aten\._to_copy\(\4, dtype=torch\.bfloat16.*\n"
        r"    (grad_h\d+_mlp_detach): '[^']*' = aten\.detach\((\w+)\).*\n"
        r"    (grad_h\d+_mlp_threshold_backward): '[^']*' = aten\.threshold_backward\(\6, \7, 0\).*\n"
    )

    count = [0]

    def _replace(m):
        count[0] += 1
        relu_fp32 = m.group(2)   # e.g. _to_copy_65
        grad_fp32 = m.group(5)   # e.g. grad_h7_mlp_c_proj__to_copy
        relu_bf16_saved = m.group(8)  # e.g. detach_52
        output_var = m.group(9)  # e.g. grad_h7_mlp_threshold_backward
        return (
            f"    {output_var} = triton_squared_relu_bwd({grad_fp32}, {relu_fp32}, {relu_bf16_saved})"
            f"  # FUSED: squared ReLU backward via Triton\n"
        )

    new_source = pat.sub(_replace, source)
    print(f"  [sqrelu_bwd] {count[0]} replacements")
    return new_source


# ======================================================================
# Optimization 6: Fuse RoPE forward
# ======================================================================

def apply_rope_fwd(source: str) -> str:
    """Replace 10-op RoPE pattern with triton_rope_fwd call (16 instances)."""
    source = insert_kernel(source, TRITON_ROPE_FWD)

    # Use line-by-line approach for reliability.
    # The pattern per instance is exactly 10 lines:
    #   hN_attn_slice[_M]: ... = aten.slice.Tensor(INPUT, 3, 0, 64)
    #   hN_attn_slice[_M+1]: ... = aten.slice.Tensor(INPUT, 3, 64, 922...)
    #   hN_attn_mul[_M]: ... = aten.mul.Tensor(slice1, COS)
    #   hN_attn_mul[_M+1]: ... = aten.mul.Tensor(slice2, SIN)
    #   hN_attn_add[_M]: ... = aten.add.Tensor(mul1, mul2)
    #   hN_attn_neg[_M]: ... = aten.neg(SIN) OR _neg_sin alias
    #   hN_attn_mul[_M+2]: ... = aten.mul.Tensor(slice1, neg)
    #   hN_attn_mul[_M+3]: ... = aten.mul.Tensor(slice2, COS)
    #   hN_attn_add[_M+1]: ... = aten.add.Tensor(mul3, mul4)
    #   hN_attn_cat[_M]: ... = aten.cat([add1, add2], 3)

    lines = source.split('\n')
    new_lines = []
    i = 0
    count = 0

    while i < len(lines):
        line = lines[i]
        # Look for start: slice.Tensor(xxx, 3, 0, 64)
        m_start = re.match(
            r'    (\w+_attn_\w*slice\w*): .* = aten\.slice\.Tensor\((\w+), 3, 0, 64\)', line
        )
        if m_start and i + 9 < len(lines):
            # Verify the 10th line is a cat
            cat_line = lines[i + 9]
            m_cat = re.match(r'    (\w+_attn_\w*cat\w*): .* = aten\.cat\(\[', cat_line)
            if m_cat:
                input_var = m_start.group(2)
                cat_var = m_cat.group(1)
                # Find cos from mul line (line i+2): mul.Tensor(slice_var, COS)
                mul_line = lines[i + 2]
                m_mul = re.match(r'    \w+: .* = aten\.mul\.Tensor\(\w+, (\w+)\)', mul_line)
                # Find sin from mul line (line i+3): mul.Tensor(slice_var, SIN)
                mul_line2 = lines[i + 3]
                m_mul2 = re.match(r'    \w+: .* = aten\.mul\.Tensor\(\w+, (\w+)\)', mul_line2)
                if m_mul and m_mul2:
                    cos_var = m_mul.group(1)
                    sin_var = m_mul2.group(1)
                    new_lines.append(
                        f"    {cat_var} = triton_rope_fwd({input_var}, {cos_var}, {sin_var})"
                        f"  # FUSED: RoPE forward via Triton (10 ops -> 1)"
                    )
                    count += 1
                    i += 10
                    continue
        new_lines.append(line)
        i += 1

    source = '\n'.join(new_lines)

    # Also need to update the forward return tuple: after fusing RoPE, the
    # hN_attn_neg and hN_attn_neg_1 vars are still referenced in the return
    # statement (they are saved for backward). We need to keep them defined.
    # The dedup pass (opt 1) already made them aliases to _neg_sin, so they
    # remain defined. If opt 1 wasn't applied, the neg lines are consumed
    # by the RoPE fusion above (the neg line is line i+5 of the 10-line
    # block). In that case we need to re-introduce them.
    # Check if any neg var from the return is undefined:
    for layer in range(8):
        for suffix in ['neg', 'neg_1']:
            varname = f"h{layer}_attn_{suffix}"
            # Check if defined somewhere in the source
            defn = re.search(rf'    {varname}[\s:=]', source)
            if not defn:
                # It was consumed. Add definition after the cat line
                cat_pat = re.compile(
                    rf"(    h{layer}_attn_\w*cat\w* = triton_rope_fwd\([^)]+\).*\n)"
                )
                m_cat = cat_pat.search(source)
                if m_cat:
                    source = source[:m_cat.end()] + \
                        f"    {varname} = _neg_sin  # needed for backward\n" + \
                        source[m_cat.end():]

    print(f"  [rope_fwd] {count} replacements")
    return source


# ======================================================================
# Optimization 7: Fuse squared ReLU forward
# ======================================================================

def apply_sqrelu_fwd(source: str) -> str:
    """Replace relu + detach + _to_copy(fp32) + pow(2) + _to_copy(bf16) with fused kernel."""
    source = insert_kernel(source, TRITON_SQRELU_FWD)

    # Pattern per layer:
    #   hN_mlp_relu: bf16 = aten.relu(INPUT)
    #   hN_mlp_detach: bf16 = aten.detach(hN_mlp_relu)
    #   hN_mlp__to_copy: fp32 = aten._to_copy(hN_mlp_relu, dtype=torch.float32)
    #   hN_mlp_pow: fp32 = aten.pow.Tensor_Scalar(hN_mlp__to_copy, 2)
    #   <possibly comment lines>
    #   hN_mlp_c_proj__to_copy: bf16[512,2048] = ... (weight cast, skip)
    #   hN_mlp_c_proj__to_copy_1: bf16 = aten._to_copy(hN_mlp_pow, dtype=torch.bfloat16)
    #
    # We need to capture relu->detach->_to_copy->pow and then also find the
    # _to_copy(pow, bf16) that feeds into c_proj.

    pat = re.compile(
        r"    (h(\d+)_mlp_relu): '([^']*)' = aten\.relu\((\w+)\).*\n"
        r"    (h\2_mlp_detach): '[^']*' = aten\.detach\(\1\).*\n"
        r"    (h\2_mlp__to_copy): '[^']*' = aten\._to_copy\(\1, dtype=torch\.float32\).*\n"
        r"    (h\2_mlp_pow): '[^']*' = aten\.pow\.Tensor_Scalar\(\6, 2\).*\n"
    )

    count = [0]

    def _replace(m):
        count[0] += 1
        relu_var = m.group(1)      # hN_mlp_relu
        layer = m.group(2)         # N
        dtype_shape = m.group(3)   # e.g. bfloat16[32, 2048, 2048]
        input_var = m.group(4)     # the relu input
        detach_var = m.group(5)    # hN_mlp_detach
        to_copy_var = m.group(6)   # hN_mlp__to_copy (fp32 relu)
        pow_var = m.group(7)       # hN_mlp_pow

        return (
            f"    {relu_var}, {to_copy_var}, _sqrelu_fwd_sq_{layer} = "
            f"triton_squared_relu_fwd({input_var})"
            f"  # FUSED: squared ReLU forward via Triton\n"
            f"    {detach_var} = {relu_var}  # alias for backward saved tensor\n"
            f"    {pow_var} = {to_copy_var}  # alias (backward uses relu_fp32 directly)\n"
        )

    new_source = pat.sub(_replace, source)

    # Now replace the _to_copy(pow, bf16) that converts pow output for c_proj input
    # Pattern: hN_mlp_c_proj__to_copy_1: bf16 = aten._to_copy(hN_mlp_pow, dtype=torch.bfloat16)
    for layer in range(8):
        pow_var = f"h{layer}_mlp_pow"
        sq_var = f"_sqrelu_fwd_sq_{layer}"
        # Replace the cast line
        cast_pat = re.compile(
            rf"    (h{layer}_mlp_c_proj__to_copy_1): '[^']*' = aten\._to_copy\({pow_var}, dtype=torch\.bfloat16\).*\n"
        )
        m = cast_pat.search(new_source)
        if m:
            cast_var = m.group(1)
            new_source = new_source[:m.start()] + \
                f"    {cast_var} = {sq_var}  # FUSED: already bf16 from triton_squared_relu_fwd\n" + \
                new_source[m.end():]

    print(f"  [sqrelu_fwd] {count[0]} replacements")
    return new_source


# ======================================================================
# Optimization 8: Fuse backward RoPE
# ======================================================================

def apply_rope_bwd(source: str) -> str:
    """Replace backward RoPE pattern with triton_rope_fwd using negated sin.

    The backward RoPE gradient is mathematically the same as forward RoPE
    with negated sin. The pattern per Q/K backward:
      slice(grad, 3, 0, 64) -> slice(grad, 3, 64, 128)
      mul(slice2, cos) -> mul(slice2, -sin) -> mul(slice1, sin) -> add -> add
      -> slice_backward -> slice_backward -> add
    This replaces the 13-op sequence with triton_rope_fwd(grad, cos, _bwd_neg_sin)
    """

    # Add precomputation of _bwd_neg_sin at start of backward
    # The backward uses slice_1 (cos) and slice_2 (sin) as params.
    # We need neg(sin) = -sin for the backward RoPE.
    # But wait - the backward already has neg_N vars passed as saved tensors.
    # Actually the backward RoPE uses slice_1 (=cos from fwd) and neg_N (=neg(sin) from fwd).
    # Let me trace the exact pattern again.

    # In backward, the RoPE grad for K (getitem_3 from rms_norm_backward):
    #   slice(grad, 3, 0, 64)
    #   slice(grad, 3, 64, 128)
    #   mul(slice_hi, cos_slice)       # hi*cos
    #   mul(slice_hi, neg_sin)         # hi*(-sin)
    #   mul(slice_lo, sin_slice)       # lo*sin
    #   add(hi*cos, lo*sin)            # -> y_lo
    #   mul(slice_lo, cos_slice)       # lo*cos
    #   add(hi*(-sin), lo*cos)         # -> y_hi
    #   slice_backward(y_lo, shape, 3, 64, end, 1)
    #   slice_backward(y_hi, shape, 3, 0, 64, 1)
    #   add(sb_lo, sb_hi)              # combined grad

    # The backward RoPE is: given grad [B,T,H,D], split into lo=[0:D/2] and hi=[D/2:D]
    # out_lo = hi*cos + lo*sin
    # out_hi = hi*(-sin) + lo*cos
    # Then reassembled via slice_backward+add.
    # This is the SAME as triton_rope_fwd(grad, cos, neg_sin) if we think of it as:
    # Actually no - the backward transposes the halves. Let me look more carefully.
    # The forward RoPE is: out_lo = x_lo*cos + x_hi*sin, out_hi = x_lo*(-sin) + x_hi*cos
    # The backward is: grad_lo = grad_hi*cos + grad_lo*sin (wait, that doesn't make sense)
    # Let me look at the actual code again.

    # From the actual backward code (h0 layer, K grad):
    #   grad_h0_attn_slice:   grad[..., 0:64]   (lo half of grad)
    #   grad_h0_attn_slice_1: grad[..., 64:128]  (hi half of grad)
    #   mul(slice_1_hi, slice_1_cos) = hi_grad * cos
    #   mul(slice_1_hi, neg_1)       = hi_grad * (-sin)
    #   mul(slice_lo, slice_2_sin)   = lo_grad * sin
    #   add(hi*cos, lo*sin) -> goes to slice_backward(..., 64, end) = upper half output
    #   mul(slice_lo, slice_1_cos)   = lo_grad * cos
    #   add(hi*(-sin), lo*cos) -> goes to slice_backward(..., 0, 64) = lower half output
    #
    # So out[64:128] = hi_grad*cos + lo_grad*sin
    #    out[0:64]   = hi_grad*(-sin) + lo_grad*cos
    #
    # This is: triton_rope_fwd(SWAPPED_grad, cos, sin) where SWAPPED means hi<->lo
    # OR equivalently triton_rope_fwd(grad, sin, cos) with swapped trig? No.
    #
    # Actually if we define backward_grad = cat([hi_grad, lo_grad]):
    # then out = triton_rope_fwd(backward_grad, cos, sin)
    # But we can't easily create that cat without extra work.
    #
    # Instead: the structure is exactly RoPE with cos=slice_1(cos) and sin=slice_2(sin)
    # applied to the SWAPPED halves. This is equivalent to a RoPE with sin negated
    # and halves swapped. For now, let's just do pattern matching replacement without
    # the triton kernel (it would need a different variant).

    # Actually, re-reading the user spec: "These can be replaced with
    # triton_rope_fwd(grad, cos, _bwd_neg_sin) where _bwd_neg_sin is precomputed once.
    # The backward RoPE is mathematically the same as forward RoPE with negated sin."
    #
    # Let me verify: Forward RoPE: out[0:D/2] = x[0:D/2]*cos + x[D/2:D]*sin
    #                               out[D/2:D] = x[0:D/2]*(-sin) + x[D/2:D]*cos
    # Backward (transpose of RoPE rotation matrix):
    #   grad_x[0:D/2] = grad[0:D/2]*cos + grad[D/2:D]*(-sin) = grad[0:D/2]*cos + grad[D/2:D]*(-sin)
    #   grad_x[D/2:D] = grad[0:D/2]*sin + grad[D/2:D]*cos
    # This equals forward RoPE with sin negated:
    #   triton_rope_fwd(grad, cos, -sin) = [grad_lo*cos + grad_hi*(-sin), grad_lo*sin + grad_hi*cos]
    # Wait that gives: out[0:D/2] = grad_lo*cos + grad_hi*(-sin)  [correct]
    #                  out[D/2:D] = grad_lo*(-(-sin)) + grad_hi*cos = grad_lo*sin + grad_hi*cos [correct!]
    # Yes! So backward RoPE = triton_rope_fwd(grad, cos, -sin)

    # First, add _bwd_neg_sin computation at the start of backward
    # In backward, slice_1 = cos, slice_2 = sin (passed as saved tensors)
    bwd_neg_sin_insert = re.compile(
        r"(def backward\([^)]*\):\n)"  # This is too broad, we need after param list
    )
    # Better: insert after the first real code line in backward
    # Find the first aten. call in backward
    bwd_first_op = re.search(
        r"(    # /.autoresearch_repo/train\.py:283\n"
        r"    # loss = F\.cross_entropy.*\n)",
        source
    )
    if bwd_first_op:
        insert_pos = bwd_first_op.start()
        source = (
            source[:insert_pos]
            + "    _bwd_neg_sin: 'bfloat16[1, 2048, 1, 64]' = aten.neg(slice_2)"
            + "  # PRECOMPUTE: neg(sin) for backward RoPE\n\n"
            + source[insert_pos:]
        )

    # Now replace each backward RoPE pattern (13 lines) with triton_rope_fwd call.
    # Pattern:
    #   grad_hN_attn_slice:   [..., 0:64]  from getitem_3 or getitem_4
    #   grad_hN_attn_slice_1: [..., 64:128]
    #   grad_hN_attn_mul:     slice_1 * slice_1(cos)
    #   grad_hN_attn_mul_1:   slice_1 * neg_N
    #   grad_hN_attn_mul_2:   slice * slice_2(sin)
    #   add_N:                mul + mul_2
    #   grad_hN_attn_mul_3:   slice * slice_1(cos)
    #   add_N+1:              mul_1 + mul_3
    #   grad_hN_attn_slice_backward:   (add_N, shape, 3, 64, end, 1)
    #   grad_hN_attn_slice_backward_1: (add_N+1, shape, 3, 0, 64, 1)
    #   add_N+2:              sb + sb_1

    lines = source.split('\n')
    new_lines = []
    i = 0
    count = 0

    while i < len(lines):
        line = lines[i]
        # Look for start: grad_hN_attn_slice: ... = aten.slice.Tensor(INPUT, 3, 0, 64)
        m_start = re.match(
            r'    (grad_h\d+_attn_slice(?:_\d+)?): .* = aten\.slice\.Tensor\((\w+), 3, 0, 64\)',
            line,
        )
        if m_start and i + 10 < len(lines):
            # Verify this is the backward RoPE pattern by checking line i+8 is slice_backward
            sb_line = lines[i + 8]
            m_sb = re.match(
                r'    (grad_h\d+_attn_slice_backward(?:_\d+)?): .* = aten\.slice_backward\(',
                sb_line,
            )
            if m_sb:
                # This is a backward RoPE block (11 lines: 2 slices + 4 muls + 2 adds + 2 slice_backward + 1 final add)
                # The input to slice is the rms_norm_backward output
                input_var = m_start.group(2)
                # The final add is at line i+10
                final_add_line = lines[i + 10]
                m_add = re.match(r'    (\w+): .* = aten\.add\.Tensor\(', final_add_line)
                if m_add:
                    output_var = m_add.group(1)
                    new_lines.append(
                        f"    {output_var} = triton_rope_fwd({input_var}, slice_1, _bwd_neg_sin)"
                        f"  # FUSED: backward RoPE via Triton"
                    )
                    count += 1
                    i += 11
                    continue
        new_lines.append(line)
        i += 1

    source = '\n'.join(new_lines)
    print(f"  [rope_bwd] {count} replacements")
    return source


# ======================================================================
# Optimization 9: Eliminate redundant bf16->fp32 casts in squared relu backward
# ======================================================================

def apply_elim_cast(source: str) -> str:
    """Eliminate redundant _to_copy(bf16->fp32) before triton_squared_relu_bwd.

    After fusing softcap_bwd, the grad flowing into the lm_head backward is already
    bf16. But some intermediate _to_copy casts from bf16 to fp32 may still be present
    that feed into triton_squared_relu_bwd which handles the conversion internally.

    Specifically, look for:
      grad_hN_mlp_c_proj__to_copy: fp32 = aten._to_copy(GRAD_BF16, dtype=torch.float32, ...)
    where the output is used as the grad_fp32 input to triton_squared_relu_bwd.
    Since triton_squared_relu_bwd already casts grad to fp32 internally, we can
    pass the bf16 tensor directly and skip the cast.
    """
    # Pattern: find _to_copy lines whose output feeds into triton_squared_relu_bwd
    # grad_hN_mlp_c_proj__to_copy: fp32 = aten._to_copy(BF16_INPUT, dtype=torch.float32, ...)
    # ... (possibly other lines)
    # OUTPUT = triton_squared_relu_bwd(grad_hN_mlp_c_proj__to_copy, ...)

    # Find all triton_squared_relu_bwd calls
    bwd_calls = list(re.finditer(
        r'triton_squared_relu_bwd\((\w+),',
        source,
    ))

    count = 0
    for m_call in bwd_calls:
        grad_var = m_call.group(1)
        # Check if this variable is defined by a _to_copy from bf16 to fp32
        cast_pat = re.compile(
            rf"    {re.escape(grad_var)}: '[^']*' = aten\._to_copy\((\w+), dtype=torch\.float32.*\n"
        )
        m_cast = cast_pat.search(source)
        if m_cast:
            bf16_input = m_cast.group(1)
            # Check that bf16_input is bf16 (the source line mentions bfloat16 or the view line)
            # Just do the replacement: remove the cast line and rename the variable
            source = source[:m_cast.start()] + source[m_cast.end():]
            # Replace usage of grad_var with bf16_input in the triton call
            source = source.replace(
                f"triton_squared_relu_bwd({grad_var},",
                f"triton_squared_relu_bwd({bf16_input},",
            )
            count += 1

    print(f"  [elim_cast] {count} redundant casts eliminated")
    return source


# ======================================================================
# Optimization registry
# ======================================================================

OPTIMIZATIONS = [
    ("dedup_neg_sin_mask", apply_dedup_neg_sin_and_mask),
    ("softcap_fwd", apply_softcap_fwd),
    ("softcap_bwd", apply_softcap_bwd),
    ("lambda_scale", apply_lambda_scale),
    ("sqrelu_bwd", apply_sqrelu_bwd),
    ("rope_fwd", apply_rope_fwd),
    ("sqrelu_fwd", apply_sqrelu_fwd),
    ("rope_bwd", apply_rope_bwd),
    ("elim_cast", apply_elim_cast),
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply kernel fusion optimizations to aten file")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        help="Optimization name to apply, or 'all' (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="List available optimizations")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument(
        "--file",
        type=Path,
        default=ATEN_FILE,
        help="Path to aten file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: overwrite input)",
    )
    args = parser.parse_args()

    if args.list:
        print("Available optimizations (applied in order):")
        for i, (name, fn) in enumerate(OPTIMIZATIONS, 1):
            doc = fn.__doc__.split("\n")[0] if fn.__doc__ else ""
            print(f"  {i}. {name:<25s} {doc}")
        return

    aten_path = args.file
    if not aten_path.exists():
        print(f"ERROR: aten file not found: {aten_path}", file=sys.stderr)
        sys.exit(1)

    source = aten_path.read_text()
    original_lines = len(source.split("\n"))
    print(f"Input: {aten_path}")
    print(f"  Lines: {original_lines}")
    print()

    # Determine which optimizations to apply
    if args.target == "all":
        targets = [name for name, _ in OPTIMIZATIONS]
    else:
        targets = [args.target]
        if args.target not in dict(OPTIMIZATIONS):
            print(f"ERROR: unknown optimization '{args.target}'", file=sys.stderr)
            print(f"Available: {', '.join(name for name, _ in OPTIMIZATIONS)}", file=sys.stderr)
            sys.exit(1)

    # Apply in order. For single-target mode, we apply all preceding
    # optimizations silently as prerequisites (each builds on the previous).
    applied = []
    target_reached = False
    for name, fn in OPTIMIZATIONS:
        if name in targets:
            # This is a requested optimization
            print(f"Applying: {name}")
            new_source = fn(source)
            if new_source == source:
                print(f"  WARNING: no changes made")
            else:
                delta = len(new_source.split("\n")) - len(source.split("\n"))
                print(f"  Lines: {len(new_source.split(chr(10)))} ({delta:+d})")
            source = new_source
            applied.append((name, fn))
            target_reached = True
        elif args.target != "all" and not target_reached:
            # Prerequisite: apply silently before the requested target
            print(f"  (prerequisite: {name})")
            source = fn(source)
        # else: skip (target already applied, or "all" mode and this isn't in targets)

    if not applied:
        print("No optimizations applied.")
        return

    output_path = args.output or aten_path
    if args.dry_run:
        print(f"\n[DRY RUN] Would write {len(source.split(chr(10)))} lines to {output_path}")
    else:
        output_path.write_text(source)
        final_lines = len(source.split("\n"))
        print(f"\nWrote: {output_path}")
        print(f"  Lines: {original_lines} -> {final_lines} ({final_lines - original_lines:+d})")
        print(f"  Applied: {', '.join(name for name, _ in applied)}")


if __name__ == "__main__":
    main()
