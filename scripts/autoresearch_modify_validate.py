#!/usr/bin/env python3
"""Validate that modifications to captured aten files take effect.

Three tests:
  1. Baseline: run captured aten graph, record loss
  2. Break: zero out the first residual add → loss should change dramatically
  3. CUDA kernel: replace an aten.add with an inline CUDA kernel → loss should match baseline

Usage:
  .venv/bin/python scripts/autoresearch_modify_validate.py
"""

from __future__ import annotations

import copy
import importlib
import os
import re
import shutil
import sys
import time
import types
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


# ── The inline CUDA kernel ──────────────────────────────────────────

CUDA_ADD_KERNEL = r'''
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
    """Drop-in replacement for aten.add.Tensor using inline CUDA kernel."""
    return _cuda_add_mod.cuda_bf16_add(a.contiguous(), b.contiguous())
# ── End inline CUDA kernel ───────────────────────────────────────────
'''


def _import_classes():
    importlib.import_module("prepare")
    train_path = _ar_root / "train.py"
    source = train_path.read_text()
    marker = "# ---------------------------------------------------------------------------\n# Hyperparameters"
    idx = source.find(marker)
    train_mod = types.ModuleType("train")
    train_mod.__file__ = str(train_path)
    sys.modules["train"] = train_mod
    exec(compile(source[:idx], str(train_path), "exec"), train_mod.__dict__)
    return train_mod


def _run_steps(cache_dir: str, n_steps: int = 5, seed: int = 42) -> list[float]:
    """Run n_steps through auto_install with the given cache dir, return losses."""
    torch._dynamo.reset()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_mod = _import_classes()
    GPT, GPTConfig = train_mod.GPT, train_mod.GPTConfig
    from prepare import Tokenizer

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    config = GPTConfig(
        sequence_len=2048, vocab_size=vocab_size,
        n_layer=8, n_head=4, n_kv_head=4, n_embd=512,
        window_pattern="SSSL",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cuda")
    model.init_weights()
    model.train()

    optimizer = model.setup_optimizer()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    import torch_graph.auto_install as ai
    ai.unpatch()
    ai.configure(
        cache_dir=cache_dir,
        force_recapture=False,  # USE the cached .py file
        verbose=True,
        capture_backward=True,
        dynamic=False,
    )
    ai.patch()

    compiled = torch.compile(model, dynamic=False)

    losses = []
    for step in range(n_steps):
        torch.manual_seed(1337 + step)
        x = torch.randint(0, vocab_size, (32, 2048), device="cuda")
        y = torch.randint(0, vocab_size, (32, 2048), device="cuda")
        with autocast_ctx:
            loss = compiled(x, y)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        losses.append(loss.item())

    ai.unpatch()
    return losses


def main():
    cache_dir = ".autoresearch_cache"
    aten_file = Path(cache_dir) / "GPT_07554748d291_2a_train_aten.py"

    if not aten_file.exists():
        print(f"ERROR: {aten_file} not found. Run scripts/autoresearch_capture.py first.")
        sys.exit(1)

    original_source = aten_file.read_text()

    # ── Test 1: Baseline ────────────────────────────────────────────
    print("═══ Test 1: Baseline (unmodified aten file) ═══")
    baseline_losses = _run_steps(cache_dir)
    for i, l in enumerate(baseline_losses):
        print(f"  step {i}: loss={l:.6f}")

    # ── Test 2: Breaking change ─────────────────────────────────────
    print("\n═══ Test 2: Breaking change (zero out block 0 residual add) ═══")

    # Replace the WTE embedding with zeros — this should destroy the model completely
    broken_source = original_source.replace(
        "wte_embedding: 'bfloat16[32, 2048, 512]' = aten.embedding(transformer_wte_weight, input__32__2048)",
        "wte_embedding: 'bfloat16[32, 2048, 512]' = aten.mul.Tensor(aten.embedding(transformer_wte_weight, input__32__2048), 0.0)  # MODIFIED: zeroed embeddings",
    )
    assert broken_source != original_source, "Failed to apply breaking change"
    aten_file.write_text(broken_source)

    broken_losses = _run_steps(cache_dir)
    for i, l in enumerate(broken_losses):
        print(f"  step {i}: loss={l:.6f}")

    # Verify the change actually affected results
    max_diff = max(abs(baseline_losses[i] - broken_losses[i]) for i in range(len(baseline_losses)))
    print(f"\n  Max |loss diff| vs baseline: {max_diff:.6f}")
    if max_diff > 0.01:
        print("  PASS: modification took effect (loss changed significantly)")
    else:
        print("  FAIL: modification had no effect!")

    # ── Test 3: Inline CUDA kernel ──────────────────────────────────
    print("\n═══ Test 3: Inline CUDA kernel (replace aten.add with custom kernel) ═══")

    # Insert the CUDA kernel definition after the imports, before the weights
    cuda_source = original_source.replace(
        "# ======================================================================\n# WEIGHTS / PARAMETERS\n# ======================================================================",
        CUDA_ADD_KERNEL + "\n# ======================================================================\n# WEIGHTS / PARAMETERS\n# ======================================================================",
    )

    # Replace the first residual add (block 0: x = x + attn(norm(x)))
    cuda_source = cuda_source.replace(
        "h0_add: 'bfloat16[32, 2048, 512]' = aten.add.Tensor(add_add, h0_attn_c_proj__unsafe_view)",
        "h0_add: 'bfloat16[32, 2048, 512]' = cuda_add(add_add, h0_attn_c_proj__unsafe_view)  # MODIFIED: inline CUDA kernel",
    )
    assert cuda_source != original_source, "Failed to apply CUDA kernel change"
    aten_file.write_text(cuda_source)

    print("  Compiling inline CUDA kernel...")
    cuda_losses = _run_steps(cache_dir)
    for i, l in enumerate(cuda_losses):
        print(f"  step {i}: loss={l:.6f}")

    max_diff_cuda = max(abs(baseline_losses[i] - cuda_losses[i]) for i in range(len(cuda_losses)))
    print(f"\n  Max |loss diff| vs baseline: {max_diff_cuda:.8f}")
    if max_diff_cuda < 0.01:
        print("  PASS: CUDA kernel produces same results as aten.add")
    else:
        print(f"  FAIL: CUDA kernel diverges from aten.add (diff={max_diff_cuda:.8f})")

    # ── Restore original ────────────────────────────────────────────
    aten_file.write_text(original_source)
    print("\nRestored original aten file.")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n═══ Summary ═══")
    print(f"  Baseline loss (step 4):     {baseline_losses[-1]:.6f}")
    print(f"  Broken loss (step 4):       {broken_losses[-1]:.6f}  (diff={abs(baseline_losses[-1] - broken_losses[-1]):.6f})")
    print(f"  CUDA kernel loss (step 4):  {cuda_losses[-1]:.6f}  (diff={abs(baseline_losses[-1] - cuda_losses[-1]):.8f})")
    print(f"\n  Aten file: {aten_file}")
    print(f"  HTML graph: {aten_file.with_suffix('.html')}")


if __name__ == "__main__":
    main()
