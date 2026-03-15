# Kernel Optimization Workflow: KernelBox -> Inline CUDA -> Production

End-to-end guide for optimizing individual aten ops using KernelBox, integrating
the optimized kernels back into captured aten files, and validating performance
against torch.compile.

## Overview

```
capture aten graph ──> generate kbox scripts ──> benchmark in kernelbox
                                                        │
                                                        ▼
                                                 write CUDA/Triton kernel
                                                        │
                                                        ▼
                                              inline kernel in aten .py
                                                        │
                                                        ▼
                                              run autoresearch with it
                                                        │
                                                        ▼
                                        compare: modified vs original vs torch.compile
```

## Prerequisites

```bash
make setup                    # torch-graph venv
# kernelbox-dev (for iterate/benchmark mode)
# autoresearch repo at .autoresearch_repo/ (sdpa-blackwell-compat branch)
```

## Step 1: Capture aten graphs + H5 tensor dump

Capture the full autoresearch model. The H5 dump records every intermediate
tensor so kernelbox scripts can run individual ops in isolation.

```bash
# Capture autoresearch (8L-512d, batch=32, seq=2048)
.venv/bin/python scripts/autoresearch_capture.py \
    --depth 8 --batch-size 32 --seq-len 2048 --steps 3
```

This generates:
```
.autoresearch_cache/
  GPT_<hash>_2a_train_aten.py    # 400KB — full forward+backward as aten ops
  GPT_<hash>_2a_train_aten.html  # 1.7MB — interactive HTML graph visualization
  GPT_<hash>_2a_train_aten.meta  # metadata (shapes, mutations, primal ordering)
```

For H5 tensor dumps (needed for kernelbox), add `dump_h5=True` to the
auto_install config in the capture script, or capture manually:

```python
from torch_graph.export import capture_aten_graphs
from torch_graph.op_dump import dump_grouped_tensors

output, capture = capture_aten_graphs(model, *args, run_backward=True, ...)

dump_grouped_tensors(
    capture,
    "autoresearch.h5",
    group_by=["line"],
    which="both",           # forward + backward
    include_params=True,
    replay_scripts=True,
)
```

## Step 2: Generate kernelbox scripts

### Per-operation scripts

One `.py` file per aten operation, with its own `.h5` data file:

```bash
.venv/bin/python -m torch_graph kbox autoresearch.h5 \
    --all --section forward --out-dir kbox_scripts/
```

Generates files like:
```
kbox_scripts/
  fw_000_embedding.py        # WTE embedding
  fw_011_masked_fill.py      # attention mask
  fw_023_scaled_dot_product_attention.py  # SDPA
  fw_045_mm.py               # linear layer matmul
  data/
    fw_000_embedding.h5      # input/output tensors for each op
    fw_011_masked_fill.h5
    ...
```

### Chain scripts (full forward/backward as one script)

```bash
.venv/bin/python -m torch_graph kbox autoresearch.h5 \
    --chain --section both --out-dir kbox_scripts/
```

Generates:
```
kbox_scripts/
  forward_line_chain.py    # all forward ops chained
  backward_line_chain.py   # all backward ops chained
```

### Unique-group scripts (deduplicated)

For repeated ops (e.g., the same `mm` at every layer), generates one script
with multiple test instances:

```bash
.venv/bin/python -m torch_graph kbox autoresearch.h5 \
    --all --section forward --out-dir kbox_scripts/
# Automatically generates grp_mm.py, grp_rms_norm.py, etc.
# with data/grp_mm/instance_0.h5, instance_1.h5, ...
```

## Step 3: Benchmark in kernelbox

### What a kbox script looks like

```python
"""011_model.py:28 [blocks.0] | att = att.masked_fill(...)

Inputs (5.0KB total):
  primals_8    float32[1x1x16x16]  strides=(256,256,16,1) C  1.0KB
  mul          float32[2x2x16x16]  strides=(512,256,16,1) C  4.0KB
Outputs (4.0KB total):
  masked_fill  float32[2x2x16x16]  strides=(512,256,16,1) C  4.0KB
Ops: alias, eq, masked_fill  (3 ops)

    kbox iterate fw_011_masked_fill.py
"""
import torch

def init_once():
    return {"h5": "data/fw_011_masked_fill.h5"}

def run(inputs):
    alias = torch.ops.aten.alias.default(inputs.primals_8)
    eq = torch.ops.aten.eq.Scalar(alias, 0)
    masked_fill = torch.ops.aten.masked_fill.Scalar(inputs.mul, eq, -inf)
    return [masked_fill]
```

### Run once with validation + benchmark

```bash
PYTHONPATH=/path/to/kernelbox-dev/python \
python3 /path/to/kernelbox-dev/tools/kbox_iterate.py \
    kbox_scripts/fw_011_masked_fill.py \
    --once --bench --warmup 3 --iters 10
```

This validates the output against the expected tensors in the H5 file and
reports timing.

### Live iterate mode (hot-reload on edit)

```bash
PYTHONPATH=/path/to/kernelbox-dev/python \
python3 /path/to/kernelbox-dev/tools/kbox_iterate.py \
    kbox_scripts/fw_011_masked_fill.py \
    --bench
```

Now edit `run()` in the script. Kernelbox detects the file change, re-runs,
validates against expected outputs, and reports the new timing. This is the
inner loop for kernel optimization.

### Benchmark the full forward chain

```bash
PYTHONPATH=/path/to/kernelbox-dev/python \
python3 /path/to/kernelbox-dev/tools/kbox_iterate.py \
    kbox_scripts/forward_line_chain.py \
    --once --bench --warmup 3 --iters 10
```

## Step 4: Write an optimized CUDA kernel

### Example: replace aten.add with an inline CUDA kernel

This is the pattern for inlining any custom CUDA code. The kernel is compiled
at module load time via `torch.utils.cpp_extension.load_inline`.

```python
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
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hadd(a[idx], b[idx]);
    }
}

torch::Tensor cuda_bf16_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dtype() == torch::kBFloat16, "expected bf16");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
    auto out = torch::empty_like(a);
    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
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
    name="cuda_bf16_add",
    cpp_sources=_cuda_add_cpp,
    cuda_sources=_cuda_add_cu,
    functions=["cuda_bf16_add"],
    verbose=False,
)

def cuda_add(a, b):
    """Drop-in replacement for aten.add.Tensor."""
    return _cuda_add_mod.cuda_bf16_add(a.contiguous(), b.contiguous())
```

### Validate in kernelbox first

Before integrating into the aten file, test in a kbox script:

```python
def run(inputs):
    # Original: masked_fill = torch.ops.aten.masked_fill.Scalar(inputs.mul, eq, -inf)
    # Modified: use custom kernel
    result = my_custom_masked_fill(inputs.mul, inputs.primals_8)
    return [result]
```

Run with `--bench` to compare timing and validate correctness.

### Triton kernels work too

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)

def triton_add(a, b):
    out = torch.empty_like(a)
    n = a.numel()
    add_kernel[(n + 1023) // 1024](a, b, out, n, BLOCK=1024)
    return out
```

## Step 5: Inline the kernel in the aten .py file

Once validated, edit the captured aten file directly.

### Insert kernel definition after imports

In `.autoresearch_cache/GPT_<hash>_2a_train_aten.py`, add the kernel code
after the `import` block but before the weight definitions:

```python
import operator
import os
import torch
aten = torch.ops.aten

# ── Custom CUDA kernel: fused residual add ───────────────────────
import torch.utils.cpp_extension
# ... (kernel code from step 4) ...
# ── End custom kernel ────────────────────────────────────────────

# ======================================================================
# WEIGHTS / PARAMETERS
# ======================================================================
```

### Replace the aten op call

Find the target operation in the `forward()` function and replace it:

```python
# BEFORE (original aten op):
h0_add: 'bfloat16[32, 2048, 512]' = aten.add.Tensor(add_add, h0_attn_c_proj__unsafe_view)

# AFTER (custom kernel):
h0_add: 'bfloat16[32, 2048, 512]' = cuda_add(add_add, h0_attn_c_proj__unsafe_view)
```

The auto_install system reloads the modified .py file on next forward call.
No restart needed.

## Step 6: Run autoresearch with the modified aten file

```bash
# Re-run with the modified aten file (auto_install loads it from cache)
.venv/bin/python scripts/autoresearch_capture.py --steps 10
```

The system detects the cached file exists and loads it instead of re-capturing.
Your kernel modification takes effect immediately.

### Validate correctness

```bash
.venv/bin/python scripts/autoresearch_modify_validate.py
```

This runs three tests:
1. **Baseline**: unmodified aten file, records loss
2. **Breaking change**: zeros out embeddings to prove the file is used (loss diverges by ~0.21)
3. **CUDA kernel**: your kernel replaces an aten.add (loss should be bit-identical to baseline)

## Step 7: Performance comparison

Use the benchmark scripts to compare all modes:

```bash
# Compare: eager vs aten_replay vs torch.compile vs compile_aten
.venv/bin/python bench_autoresearch.py \
    --depth 8 --batch-size 32 --seq-len 2048 \
    --modes eager aten_replay compile compile_aten
```

### What the modes measure

| Mode | What runs | Purpose |
|------|-----------|---------|
| `eager` | Original PyTorch model, no compilation | Baseline |
| `aten_replay` | Captured aten ops via autograd.Function | Our capture system faithfulness |
| `compile` | `torch.compile(model)` with inductor | Best-case compiled perf |
| `compile_aten` | Captured aten ops compiled via inductor | Compile-on-aten perf |

### Expected results (autoresearch 8L-512d, B200)

| Mode | Median | vs compile |
|------|--------|------------|
| eager | 85ms | 2.43x |
| aten_replay | 85ms | 2.42x |
| compile | 35ms | 1.00x |
| compile_aten | 39ms | 1.12x |

The 7-12% gap between compile and compile_aten comes from decomposed RoPE/RMSNorm
ops. Models using `F.scaled_dot_product_attention` (like autoresearch) keep the
attention op fused, minimizing the gap.

### Comparing with vs without your custom kernel

To A/B test your kernel:

1. Run benchmark with unmodified aten file → record `aten_replay` timing
2. Edit the aten file with your kernel
3. Run benchmark again → compare timing
4. Also run `compile_aten` to see if your kernel helps inductor too

```bash
# A: baseline aten replay
.venv/bin/python bench_autoresearch.py --modes aten_replay --steps 30
# Note the median

# B: edit .autoresearch_cache/GPT_*_aten.py with your kernel

# C: re-run (auto_install picks up the edit)
.venv/bin/python bench_autoresearch.py --modes aten_replay --steps 30
# Compare median to A
```

## Optimization targets in autoresearch

The autoresearch forward has 892 aten ops (32 unique op types). Good
optimization targets based on the captured graph:

| Op | Count | Description | Optimization opportunity |
|----|-------|-------------|-------------------------|
| `aten.mm` | 80 | Matrix multiplies (linear layers) | cuBLAS tuning, fp8 |
| `aten._fused_rms_norm` | 17 | RMS normalization | Already fused |
| `aten.scaled_dot_product_attention` | 8 | SDPA attention | Already fused (FA3/SDPA) |
| `aten.mul.Tensor` | 72 | Elementwise multiply | Fuse with adjacent ops |
| `aten.add.Tensor` | 48 | Elementwise add (residuals) | Fuse with multiply |
| `aten.cat` | 16 | Concatenation (RoPE) | Fuse into RoPE kernel |
| `aten.embedding` | 13 | Embedding lookups | Memory-bound, hard to optimize |
| `aten.select.int` | 16 | Scalar selects (lambdas) | Negligible cost |

### High-value fusion targets

1. **RoPE**: `slice + mul + cat` (apply_rotary_emb) — 6 ops per layer,
   fusible into a single Triton kernel
2. **Residual + RMSNorm**: `add + _fused_rms_norm` — fuse the residual add
   with the subsequent normalization
3. **Squared ReLU + MLP**: `mm + relu + square + mm` — fuse the nonlinearity
   with the surrounding matmuls

## File reference

| File | Purpose |
|------|---------|
| `scripts/autoresearch_capture.py` | Capture autoresearch aten graphs + HTML |
| `scripts/autoresearch_modify_validate.py` | Three-test validation (baseline / break / CUDA kernel) |
| `bench_autoresearch.py` | Performance benchmark (4 modes) |
| `bench_perf.py` | Performance benchmark for NanoGPT/nanochat |
| `docs/PERF_BENCHMARK.md` | Full performance results and analysis |
| `docs/KERNELBOX_ITERATE.md` | KernelBox iterate mode basics |
| `torch_graph/kbox_gen.py` | KernelBox script generation from H5 dumps |
| `.autoresearch_cache/GPT_*_aten.py` | The captured aten file (edit this) |
| `.autoresearch_cache/GPT_*_aten.html` | Interactive graph visualization |
