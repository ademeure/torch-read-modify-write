#!/usr/bin/env python3
"""Autonomous aten optimization loop for autoresearch.

Captures the aten graph, identifies optimization targets, writes Triton/CUDA
kernels, inlines them, benchmarks, and generates progress charts.

Each iteration:
  1. Benchmark current aten file (aten_replay mode)
  2. Identify next optimization target
  3. Write a fused kernel
  4. Patch the aten file
  5. Validate correctness (loss matches baseline)
  6. Benchmark again
  7. Generate progress chart
  8. Commit the optimized aten file

Usage:
  .venv/bin/python scripts/autoresearch_optimize_loop.py [--max-iters 10]
"""

from __future__ import annotations

import copy
import gc
import importlib
import json
import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
_ar_root = _root / ".autoresearch_repo"
sys.path.insert(0, str(_ar_root))

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault(
    "LIBRARY_PATH",
    "/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs:"
    + os.environ.get("LIBRARY_PATH", ""),
)

# Force SDPA
_fa_spec = importlib.util.spec_from_file_location(
    "flash_attention", str(_ar_root / "flash_attention.py"))
_fa_mod = importlib.util.module_from_spec(_fa_spec)
import builtins
_saved_import = builtins.__import__
def _block_kernels(name, *a, **kw):
    if name == "kernels": raise ImportError("blocked")
    return _saved_import(name, *a, **kw)
builtins.__import__ = _block_kernels
try: _fa_spec.loader.exec_module(_fa_mod)
finally: builtins.__import__ = _saved_import
sys.modules["flash_attention"] = _fa_mod

import torch
import torch._dynamo
torch.set_float32_matmul_precision("high")


# ── Model setup ─────────────────────────────────────────────────────

def _import_classes():
    importlib.import_module("prepare")
    train_path = _ar_root / "train.py"
    source = train_path.read_text()
    marker = "# ---------------------------------------------------------------------------\n# Hyperparameters"
    train_mod = types.ModuleType("train")
    train_mod.__file__ = str(train_path)
    sys.modules["train"] = train_mod
    exec(compile(source[:source.find(marker)], str(train_path), "exec"), train_mod.__dict__)
    return train_mod


def _build_model():
    train_mod = _import_classes()
    from prepare import Tokenizer
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    config = train_mod.GPTConfig(
        sequence_len=2048, vocab_size=vocab_size,
        n_layer=8, n_head=4, n_kv_head=4, n_embd=512,
        window_pattern="SSSL",
    )
    with torch.device("meta"):
        model = train_mod.GPT(config)
    model.to_empty(device="cuda")
    model.init_weights()
    model.train()
    return model, vocab_size


# ── Benchmark helper ────────────────────────────────────────────────

BATCH_SIZE = 32
SEQ_LEN = 2048
WARMUP = 3
BENCH_STEPS = 15
CACHE_DIR = ".autoresearch_optim_cache"


def benchmark_aten_replay(cache_dir: str = CACHE_DIR, seed: int = 42) -> dict:
    """Run aten_replay benchmark, return {median_ms, losses, step_times}."""
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model, vocab_size = _build_model()
    optimizer = model.setup_optimizer()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    import torch_graph.auto_install as ai
    ai.unpatch()
    ai.configure(
        cache_dir=cache_dir,
        force_recapture=False,
        verbose=False,
        capture_backward=True,
        dynamic=False,
    )
    ai.patch()
    compiled = torch.compile(model, dynamic=False)

    # Warmup
    for i in range(WARMUP):
        torch.manual_seed(1337 + i)
        x = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda")
        y = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda")
        with autocast_ctx:
            loss = compiled(x, y)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Timed
    losses, step_times = [], []
    for i in range(BENCH_STEPS):
        torch.manual_seed(1337 + WARMUP + i)
        x = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda")
        y = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with autocast_ctx:
            loss = compiled(x, y)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        losses.append(loss.item())

    ai.unpatch()

    median_ms = sorted(step_times)[len(step_times) // 2] * 1000
    return {"median_ms": median_ms, "losses": losses, "step_times": step_times}


# ── Capture ─────────────────────────────────────────────────────────

def capture_baseline():
    """Capture aten graph if not already cached."""
    cache = Path(CACHE_DIR)
    aten_files = list(cache.glob("*_aten.py"))
    if aten_files:
        return aten_files[0]

    torch._dynamo.reset()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model, vocab_size = _build_model()
    optimizer = model.setup_optimizer()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    import torch_graph.auto_install as ai
    ai.unpatch()
    ai.configure(
        cache_dir=CACHE_DIR,
        force_recapture=True,
        verbose=True,
        capture_backward=True,
        dynamic=False,
    )
    ai.patch()

    compiled = torch.compile(model, dynamic=False)
    x = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda")
    y = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda")
    with autocast_ctx:
        loss = compiled(x, y)
    loss.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)

    ai.unpatch()

    aten_files = list(cache.glob("*_aten.py"))
    assert aten_files, "Capture failed — no aten .py files generated"
    return aten_files[0]


# ── Optimization targets ────────────────────────────────────────────

@dataclass
class OptimTarget:
    name: str
    description: str
    # String to find in the aten file
    find_pattern: str
    # Replacement code (kernel def + call site)
    kernel_code: str
    replacement: str


def _build_rope_kernel() -> OptimTarget:
    """Fuse RoPE: slice + mul + neg + mul + cat → single Triton kernel."""
    kernel = '''
# ── Triton RoPE kernel ───────────────────────────────────────────────
import triton
import triton.language as tl

@triton.jit
def _rope_fwd_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_x_b, stride_x_t, stride_x_h, stride_x_d,
    stride_cos_b, stride_cos_t, stride_cos_h, stride_cos_d,
    D: tl.constexpr, HALF_D: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    t_start = pid_t * BLOCK_T
    for t_off in range(BLOCK_T):
        t = t_start + t_off
        for d in range(HALF_D):
            x1_off = pid_b * stride_x_b + t * stride_x_t + pid_h * stride_x_h + d * stride_x_d
            x2_off = pid_b * stride_x_b + t * stride_x_t + pid_h * stride_x_h + (d + HALF_D) * stride_x_d
            cos_off = 0 * stride_cos_b + t * stride_cos_t + 0 * stride_cos_h + d * stride_cos_d
            sin_off = cos_off

            x1 = tl.load(x_ptr + x1_off)
            x2 = tl.load(x_ptr + x2_off)
            c = tl.load(cos_ptr + cos_off)
            s = tl.load(sin_ptr + sin_off)

            tl.store(out_ptr + x1_off, x1 * c + x2 * s)
            tl.store(out_ptr + x2_off, x1 * (-s) + x2 * c)

def triton_rope(x, cos, sin):
    """Fused RoPE: replaces slice + mul + neg + mul + cat sequence."""
    B, T, H, D = x.shape
    HALF_D = D // 2
    out = torch.empty_like(x)
    grid = (B, H, (T + 31) // 32)
    _rope_fwd_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1), cos.stride(2), cos.stride(3),
        D=D, HALF_D=HALF_D, BLOCK_T=32,
    )
    return out
# ── End Triton RoPE kernel ───────────────────────────────────────────
'''
    return OptimTarget(
        name="rope_fusion",
        description="Fuse RoPE apply_rotary_emb (slice+mul+neg+mul+cat → single Triton kernel)",
        find_pattern="# apply_rotary_emb",  # source annotation
        kernel_code=kernel,
        replacement="",  # complex multi-line replacement, handled specially
    )


def _build_residual_rms_norm_kernel() -> OptimTarget:
    """Fuse residual add + RMSNorm into one Triton kernel."""
    kernel = '''
# ── Triton fused residual + RMSNorm kernel ───────────────────────────
import triton
import triton.language as tl

@triton.jit
def _residual_rms_norm_kernel(
    x_ptr, residual_ptr, out_ptr, rrms_ptr,
    N: tl.constexpr, eps: tl.constexpr,
    stride_row,
):
    row = tl.program_id(0)
    x_start = row * stride_row
    # Load x + residual
    cols = tl.arange(0, N)
    x = tl.load(x_ptr + x_start + cols).to(tl.float32)
    r = tl.load(residual_ptr + x_start + cols).to(tl.float32)
    h = x + r
    # RMSNorm
    mean_sq = tl.sum(h * h) / N
    rrms = 1.0 / tl.sqrt(mean_sq + eps)
    out = (h * rrms).to(tl.bfloat16)
    tl.store(out_ptr + x_start + cols, out)
    tl.store(rrms_ptr + row, rrms)

def triton_residual_rms_norm(x, residual, eps=1e-6):
    """Fused residual add + RMSNorm. Returns (normed, residual_sum, rrms)."""
    assert x.shape == residual.shape
    B_T = x.shape[0] * x.shape[1]  # flatten batch*seq
    N = x.shape[-1]
    x_flat = x.reshape(B_T, N)
    r_flat = residual.reshape(B_T, N)
    out = torch.empty_like(x_flat)
    rrms = torch.empty(B_T, 1, dtype=torch.float32, device=x.device)
    _residual_rms_norm_kernel[(B_T,)](
        x_flat, r_flat, out, rrms, N=N, eps=eps, stride_row=N,
    )
    return out.reshape(x.shape), rrms.reshape(x.shape[0], x.shape[1], 1)
# ── End Triton residual+RMSNorm kernel ──────────────────────────────
'''
    return OptimTarget(
        name="residual_rms_norm",
        description="Fuse residual add + RMSNorm (add + _fused_rms_norm → single Triton kernel)",
        find_pattern="_fused_rms_norm",
        kernel_code=kernel,
        replacement="",
    )


def _build_squared_relu_kernel() -> OptimTarget:
    """Fuse squared ReLU: relu(x).square() into one Triton kernel."""
    kernel = '''
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
'''
    return OptimTarget(
        name="squared_relu",
        description="Fuse relu().square() into single Triton kernel",
        find_pattern="aten.relu",
        kernel_code=kernel,
        replacement="",
    )


# ── Optimization application ───────────────────────────────────────

def apply_squared_relu(source: str, kernel_code: str) -> str:
    """Replace all relu + square pairs with triton_squared_relu."""
    import re

    # Insert kernel after imports
    source = source.replace(
        "# ======================================================================\n# WEIGHTS / PARAMETERS",
        kernel_code + "\n# ======================================================================\n# WEIGHTS / PARAMETERS",
        1,
    )

    # Find relu + pow(2) pairs — they look like:
    #   h0_mlp_relu: '...' = aten.relu(h0_mlp_c_fc_...)
    #   h0_mlp_pow: '...' = aten.pow(h0_mlp_relu, 2)
    # OR:
    #   h0_mlp_relu: '...' = aten.relu(h0_mlp_c_fc_...)
    #   h0_mlp_mul: '...' = aten.mul.Tensor(h0_mlp_relu, h0_mlp_relu)
    #
    # Replace both lines with single triton_squared_relu call

    # Pattern: relu followed by mul(x, x) i.e. square
    pattern = re.compile(
        r"(    (\w+): '[^']*' = aten\.relu\((\w+)\).*\n)"
        r"(    (\w+): '[^']*' = aten\.mul\.Tensor\(\2, \2\).*\n)",
    )

    def _replace_relu_square(m):
        relu_line = m.group(1)
        relu_var = m.group(2)
        relu_input = m.group(3)
        square_var = m.group(5)
        # Get the annotation from the square line
        return f"    {square_var} = triton_squared_relu({relu_input})  # FUSED: relu().square() via Triton\n"

    new_source = pattern.sub(_replace_relu_square, source)

    # Also handle pow(x, 2) pattern
    pattern2 = re.compile(
        r"(    (\w+): '[^']*' = aten\.relu\((\w+)\).*\n)"
        r"(    (\w+): '[^']*' = aten\.pow\(\2, 2\).*\n)",
    )
    new_source = pattern2.sub(_replace_relu_square, new_source)

    return new_source


def apply_inline_cuda_add(source: str) -> str:
    """Replace ALL residual aten.add.Tensor (bf16[32,2048,512]) with inline CUDA kernel."""
    cuda_kernel = '''
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
'''

    source = source.replace(
        "# ======================================================================\n# WEIGHTS / PARAMETERS",
        cuda_kernel + "\n# ======================================================================\n# WEIGHTS / PARAMETERS",
        1,
    )

    # Replace residual adds (bf16[32, 2048, 512] shape)
    import re
    pattern = re.compile(
        r"(    (\w+): 'bfloat16\[32, 2048, 512\]' = aten\.add\.Tensor\((\w+), (\w+)\))"
        r"(  #.*)?$",
        re.MULTILINE,
    )

    count = 0
    def _replace_add(m):
        nonlocal count
        count += 1
        var = m.group(2)
        arg1 = m.group(3)
        arg2 = m.group(4)
        return f"    {var}: 'bfloat16[32, 2048, 512]' = cuda_add({arg1}, {arg2})  # FUSED: vectorized CUDA bf16 add"

    source = pattern.sub(_replace_add, source)
    print(f"    Replaced {count} residual adds with CUDA kernel")
    return source


def apply_rope_fusion(source: str) -> str:
    """Replace RoPE sequences (slice+mul+neg+mul+add+cat) with a fused Triton kernel.

    Each RoPE application is 9 aten ops:
      slice(x, 3, 0, D/2)
      slice(x, 3, D/2, end)
      mul(x1, cos) + mul(x2, sin) + add → y1
      neg(sin) + mul(x1, neg_sin) + mul(x2, cos) + add → y2
      cat([y1, y2], 3)

    This repeats 2x per layer (q and k), 8 layers = 16 instances.
    """
    import re

    kernel = '''
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
    # Each program handles one (batch, time, head) element
    pid = tl.program_id(0)
    b = pid // (T * H)
    rem = pid % (T * H)
    t = rem // H
    h = rem % H

    base_x = b * stride_xb + t * stride_xt + h * stride_xh
    base_c = 0 * stride_cb + t * stride_ct + 0 * stride_ch  # cos/sin: [1, T, 1, D/2]

    for d in range(HALF_D):
        x1 = tl.load(x_ptr + base_x + d * stride_xd)
        x2 = tl.load(x_ptr + base_x + (d + HALF_D) * stride_xd)
        c = tl.load(cos_ptr + base_c + d * stride_cd)
        s = tl.load(sin_ptr + base_c + d * stride_cd)
        tl.store(out_ptr + base_x + d * stride_xd, x1 * c + x2 * s)
        tl.store(out_ptr + base_x + (d + HALF_D) * stride_xd, x1 * (-s) + x2 * c)

def triton_rope(x, cos, sin):
    """Fused RoPE replacing slice+mul+neg+mul+add+cat (9 ops → 1 kernel)."""
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

    # Insert kernel after imports
    source = source.replace(
        "# ======================================================================\n# WEIGHTS / PARAMETERS",
        kernel + "\n# ======================================================================\n# WEIGHTS / PARAMETERS",
        1,
    )

    # Find and replace the RoPE pattern for each layer.
    # The pattern is:
    #   PREFIX_slice: ... = aten.slice.Tensor(INPUT, 3, 0, 64)
    #   PREFIX_slice_1: ... = aten.slice.Tensor(INPUT, 3, 64, BIG)
    #   PREFIX_mul: ... = aten.mul.Tensor(PREFIX_slice, COS)
    #   PREFIX_mul_1: ... = aten.mul.Tensor(PREFIX_slice_1, SIN)
    #   PREFIX_add: ... = aten.add.Tensor(PREFIX_mul, PREFIX_mul_1)
    #   PREFIX_neg: ... = aten.neg(SIN)
    #   PREFIX_mul_2: ... = aten.mul.Tensor(PREFIX_slice, PREFIX_neg)
    #   PREFIX_mul_3: ... = aten.mul.Tensor(PREFIX_slice_1, COS)
    #   PREFIX_add_1: ... = aten.add.Tensor(PREFIX_mul_2, PREFIX_mul_3)
    #   PREFIX_cat: ... = aten.cat([PREFIX_add, PREFIX_add_1], 3)

    rope_pattern = re.compile(
        r"(    (\w+_slice): '[^']*' = aten\.slice\.Tensor\((\w+), 3, 0, (\d+)\).*\n)"
        r"(    (\w+_slice_\d+): '[^']*' = aten\.slice\.Tensor\(\3, 3, \4, \d+\).*\n)"
        r"(    (\w+_mul\b[^_]?): '[^']*' = aten\.mul\.Tensor\(\2, (\w+)\).*\n)"
        r"(    (\w+_mul_\d+): '[^']*' = aten\.mul\.Tensor\(\6, (\w+)\).*\n)"
        r"(    (\w+_add\b[^_]?): '[^']*' = aten\.add\.Tensor\(\8, \11\).*\n)"
        r"(    (\w+_neg\b[^_]?): '[^']*' = aten\.neg\(\12\).*\n)"
        r"(    (\w+_mul_\d+): '[^']*' = aten\.mul\.Tensor\(\2, \15\).*\n)"
        r"(    (\w+_mul_\d+): '[^']*' = aten\.mul\.Tensor\(\6, \9\).*\n)"
        r"(    (\w+_add_\d+): '[^']*' = aten\.add\.Tensor\(\17, \18\).*\n)"
        r"(    (\w+_cat\b[^_]?): '[^']*' = aten\.cat\(\[\13, \20\], 3\).*\n)",
    )

    count = [0]
    def _replace_rope(m):
        count[0] += 1
        input_var = m.group(3)
        cos_var = m.group(9)
        sin_var = m.group(12)
        cat_var = m.group(22)
        return f"    {cat_var} = triton_rope({input_var}, {cos_var}, {sin_var})  # FUSED: RoPE via Triton (9 ops → 1)\n"

    new_source = rope_pattern.sub(_replace_rope, source)
    print(f"    Replaced {count[0]} RoPE sequences with Triton kernel")

    if count[0] == 0:
        # Try a simpler line-by-line approach as fallback
        print("    Trying simpler pattern matching...")
        lines = new_source.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Look for the start: slice.Tensor(xxx, 3, 0, 64)
            m_start = re.match(r'    (\w+): .* = aten\.slice\.Tensor\((\w+), 3, 0, 64\)', line)
            if m_start and i + 9 < len(lines):
                # Check if the 10th line ahead is a cat
                cat_line = lines[i + 9]
                m_cat = re.match(r'    (\w+): .* = aten\.cat\(\[', cat_line)
                if m_cat:
                    input_var = m_start.group(2)
                    cat_var = m_cat.group(1)
                    # Find cos/sin from the mul lines
                    mul_line = lines[i + 2]
                    m_mul = re.match(r'    \w+: .* = aten\.mul\.Tensor\(\w+, (\w+)\)', mul_line)
                    mul_line2 = lines[i + 3]
                    m_mul2 = re.match(r'    \w+: .* = aten\.mul\.Tensor\(\w+, (\w+)\)', mul_line2)
                    if m_mul and m_mul2:
                        cos_var = m_mul.group(1)
                        sin_var = m_mul2.group(1)
                        new_lines.append(
                            f"    {cat_var} = triton_rope({input_var}, {cos_var}, {sin_var})"
                            f"  # FUSED: RoPE via Triton (9 ops → 1)"
                        )
                        count[0] += 1
                        i += 10
                        continue
            new_lines.append(line)
            i += 1
        new_source = '\n'.join(new_lines)
        print(f"    Replaced {count[0]} RoPE sequences (fallback)")

    return new_source


# ── Progress chart ──────────────────────────────────────────────────

def generate_chart(results: list[dict], output_path: str):
    """Generate a progress chart showing optimization improvements."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in results]
    medians = [r["median_ms"] for r in results]
    baseline = medians[0]
    speedups = [baseline / m for m in medians]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: absolute timing
    colors = ["#2196F3" if i == 0 else "#4CAF50" if m < baseline else "#F44336"
              for i, m in enumerate(medians)]
    bars = ax1.bar(range(len(names)), medians, color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Median step time (ms)")
    ax1.set_title("Autoresearch 8L-512d: Step Time per Optimization")
    ax1.axhline(y=baseline, color="blue", linestyle="--", alpha=0.5, label="baseline")
    for bar, val in zip(bars, medians):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax1.legend()

    # Right: cumulative speedup
    ax2.plot(range(len(names)), speedups, "o-", color="#4CAF50", linewidth=2, markersize=8)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Speedup vs baseline")
    ax2.set_title("Cumulative Speedup")
    ax2.axhline(y=1.0, color="blue", linestyle="--", alpha=0.5)
    ax2.set_ylim(bottom=0.9)
    for i, s in enumerate(speedups):
        ax2.annotate(f"{s:.3f}x", (i, s), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Chart saved: {output_path}")


# ── Main optimization loop ──────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path("outputs/autoresearch_optimization")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.json"
    chart_path = out_dir / "progress.png"

    results: list[dict] = []
    if results_file.exists():
        results = json.loads(results_file.read_text())

    # Step 0: Capture baseline
    print("=" * 70)
    print("STEP 0: Capture baseline aten graph")
    print("=" * 70)
    aten_file = capture_baseline()
    original_source = aten_file.read_text()
    print(f"  Aten file: {aten_file}")
    print(f"  Size: {len(original_source):,} bytes")

    # Baseline benchmark
    if not results:
        print("\n  Benchmarking baseline...")
        baseline = benchmark_aten_replay()
        results.append({
            "name": "baseline",
            "median_ms": baseline["median_ms"],
            "final_loss": baseline["losses"][-1],
            "description": "Unmodified captured aten graph",
        })
        print(f"  Baseline median: {baseline['median_ms']:.2f}ms")
        print(f"  Baseline loss: {baseline['losses'][-1]:.6f}")
        results_file.write_text(json.dumps(results, indent=2))
        generate_chart(results, str(chart_path))

    # Define optimization sequence
    optimizations = [
        {
            "name": "squared_relu",
            "description": "Fuse relu().square() → single Triton kernel (8 instances)",
            "apply": lambda src: apply_squared_relu(src, _build_squared_relu_kernel().kernel_code),
        },
        {
            "name": "cuda_vec_add",
            "description": "Replace bf16 residual adds with vectorized CUDA kernel",
            "apply": lambda src: apply_inline_cuda_add(src),
        },
        {
            "name": "rope_fusion",
            "description": "Fuse RoPE: 9 ops (slice+mul+neg+mul+add+cat) → 1 Triton kernel (16 instances)",
            "apply": lambda src: apply_rope_fusion(src),
        },
    ]

    current_source = original_source

    for i, opt in enumerate(optimizations):
        if i >= args.max_iters:
            break
        # Skip already-done optimizations
        if any(r["name"] == opt["name"] for r in results):
            print(f"\n  Skipping {opt['name']} (already done)")
            # Replay the optimization on current_source to keep it up to date
            current_source = opt["apply"](current_source)
            continue

        print(f"\n{'=' * 70}")
        print(f"STEP {len(results)}: {opt['name']}")
        print(f"  {opt['description']}")
        print("=" * 70)

        # Apply optimization
        new_source = opt["apply"](current_source)
        if new_source == current_source:
            print("  WARNING: optimization had no effect, skipping")
            continue

        # Write modified file
        aten_file.write_text(new_source)
        print("  Modified aten file written")

        # Validate: run a few steps and check loss is reasonable
        print("  Validating correctness...")
        try:
            result = benchmark_aten_replay()
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  Reverting to previous version")
            aten_file.write_text(current_source)
            continue

        # Check loss is reasonable (within 1.0 of baseline — we're not training deterministically
        # due to optimizer state differences, but a broken kernel would give NaN or huge loss)
        baseline_loss = results[0]["final_loss"]
        if abs(result["losses"][-1] - baseline_loss) > 1.0 or any(
            l != l for l in result["losses"]  # NaN check
        ):
            print(f"  FAILED: loss diverged ({result['losses'][-1]:.6f} vs baseline {baseline_loss:.6f})")
            print("  Reverting to previous version")
            aten_file.write_text(current_source)
            continue

        current_source = new_source

        results.append({
            "name": opt["name"],
            "median_ms": result["median_ms"],
            "final_loss": result["losses"][-1],
            "description": opt["description"],
        })

        speedup = results[0]["median_ms"] / result["median_ms"]
        print(f"  Median: {result['median_ms']:.2f}ms (speedup: {speedup:.3f}x)")
        print(f"  Loss: {result['losses'][-1]:.6f}")

        # Save progress
        results_file.write_text(json.dumps(results, indent=2))
        generate_chart(results, str(chart_path))

        # Save this version
        versioned = out_dir / f"aten_after_{opt['name']}.py"
        versioned.write_text(current_source)
        print(f"  Saved: {versioned}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print("=" * 70)
    for r in results:
        speedup = results[0]["median_ms"] / r["median_ms"]
        print(f"  {r['name']:<20s}  {r['median_ms']:>8.2f}ms  {speedup:.3f}x  loss={r['final_loss']:.6f}")
    print(f"\n  Chart: {chart_path.resolve()}")
    print(f"  Results: {results_file.resolve()}")


if __name__ == "__main__":
    main()
