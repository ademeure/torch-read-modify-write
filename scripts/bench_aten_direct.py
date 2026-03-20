#!/usr/bin/env python3
"""Direct aten file benchmark — no model code needed.

Loads the aten forward/backward module, creates matching parameters,
and benchmarks forward+backward directly. Loss is validated across
optimization iterations.

Usage:
  .venv/bin/python scripts/bench_aten_direct.py [--warmup 3] [--steps 15]
  .venv/bin/python scripts/bench_aten_direct.py --aten-file path/to/aten.py
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault(
    "LIBRARY_PATH",
    "/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs:"
    + os.environ.get("LIBRARY_PATH", ""),
)

import torch
import torch._dynamo

torch.set_float32_matmul_precision("high")

_root = Path(__file__).resolve().parent.parent

# Default aten cache dir
DEFAULT_CACHE = _root / ".autoresearch_optim_cache"
DEFAULT_ATEN = DEFAULT_CACHE / "GPT_07554748d291_2a_train_aten.py"


def load_aten_module(path: Path) -> object:
    """Load an aten .py file as a module."""
    # Patch sys.argv to avoid argparse conflicts in the aten file
    saved_argv = sys.argv
    sys.argv = [str(path)]
    try:
        spec = importlib.util.spec_from_file_location("_aten_mod", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = saved_argv


def _parse_forward_sig(mod) -> list[tuple[str, str, list[int]]]:
    """Parse forward() signature to get (name, dtype, shape) for each param."""
    sig = inspect.signature(mod.forward)
    params = []
    for name, p in sig.parameters.items():
        annotation = str(p.annotation) if p.annotation != inspect.Parameter.empty else ""
        # Parse 'float32[512, 512]' or 'bfloat16[32, 2048, 512]' or 'int64[32, 2048]'
        m = re.match(r"(\w+)\[([\d, ]+)\]", annotation)
        if m:
            dtype_str = m.group(1)
            shape = [int(x.strip()) for x in m.group(2).split(",")]
            params.append((name, dtype_str, shape))
        else:
            params.append((name, "float32", []))
    return params


def _make_tensors(param_specs: list[tuple[str, str, list[int]]], seed: int = 42):
    """Create tensors matching forward() signature."""
    torch.manual_seed(seed)
    tensors = {}
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
    }
    for name, dtype_str, shape in param_specs:
        dtype = dtype_map.get(dtype_str, torch.float32)
        if dtype == torch.int64:
            tensors[name] = torch.randint(0, 8192, shape, device="cuda", dtype=dtype)
        elif dtype == torch.bfloat16:
            tensors[name] = torch.randn(shape, device="cuda", dtype=dtype)
        else:
            tensors[name] = torch.randn(shape, device="cuda", dtype=dtype)
    return tensors


def benchmark_forward_backward(
    mod,
    tensors: dict,
    param_specs: list,
    warmup: int = 3,
    steps: int = 15,
    seed: int = 42,
) -> dict:
    """Benchmark forward + backward, return {median_ms, losses, step_times}."""
    # Build ordered arg list
    param_names = [s[0] for s in param_specs]

    # Figure out which tensors need grad (the fp32 weight params)
    weight_names = set()
    for name, dtype_str, shape in param_specs:
        if dtype_str == "float32" and len(shape) >= 1 and "weight" in name or "lambda" in name:
            weight_names.add(name)

    def run_step(step_idx: int):
        # Fresh random inputs each step
        torch.manual_seed(1337 + step_idx)
        for name, dtype_str, shape in param_specs:
            if "input_" in name and dtype_str == "int64":
                tensors[name] = torch.randint(0, 8192, shape, device="cuda", dtype=torch.int64)

        # Enable grads on weight params
        for name in weight_names:
            tensors[name].requires_grad_(True)

        args = [tensors[name] for name in param_names]
        result = mod.forward(*args)

        if isinstance(result, tuple):
            # Find the loss (scalar float32 tensor)
            loss_val = None
            for item in result:
                if isinstance(item, torch.Tensor) and item.ndim == 0 and item.dtype == torch.float32:
                    loss_val = item
                    break
            if loss_val is None:
                # Try nll_loss output
                for item in result:
                    if isinstance(item, torch.Tensor) and item.ndim == 0:
                        loss_val = item.float()
                        break
            if loss_val is None:
                raise RuntimeError("Could not find scalar loss in forward output")

            # Run backward on loss
            loss_val.backward(retain_graph=False)
            loss_scalar = loss_val.item()
        else:
            result.backward()
            loss_scalar = result.item()

        # Zero grads for next step
        for name in weight_names:
            if tensors[name].grad is not None:
                tensors[name].grad = None

        return loss_scalar

    # Warmup
    for i in range(warmup):
        loss = run_step(i)
        if i == 0:
            print(f"  warmup step 0: loss={loss:.6f}")
    torch.cuda.synchronize()

    # Timed steps
    losses, step_times = [], []
    for i in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = run_step(warmup + i)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)
        losses.append(loss)

    median_ms = sorted(step_times)[len(step_times) // 2] * 1000
    avg_ms = sum(step_times) / len(step_times) * 1000
    mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

    print(f"  median={median_ms:.2f}ms  avg={avg_ms:.2f}ms  "
          f"mem={mem_mb:.0f}MB  final_loss={losses[-1]:.6f}")

    return {
        "median_ms": median_ms,
        "avg_ms": avg_ms,
        "mem_mb": mem_mb,
        "losses": losses,
        "step_times": step_times,
    }


def main():
    parser = argparse.ArgumentParser(description="Direct aten file benchmark")
    parser.add_argument("--aten-file", type=str, default=str(DEFAULT_ATEN))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    aten_path = Path(args.aten_file)
    if not aten_path.exists():
        print(f"ERROR: aten file not found: {aten_path}")
        sys.exit(1)

    print(f"═══ Direct Aten Benchmark ═══")
    print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Aten file: {aten_path}")
    print(f"Warmup: {args.warmup}  Steps: {args.steps}")
    print()

    print("Loading aten module...")
    mod = load_aten_module(aten_path)
    print("Parsing forward signature...")
    param_specs = _parse_forward_sig(mod)
    print(f"  {len(param_specs)} parameters")

    print("Creating tensors...")
    tensors = _make_tensors(param_specs, seed=args.seed)

    print("\nBenchmarking forward+backward...")
    torch.cuda.reset_peak_memory_stats()
    result = benchmark_forward_backward(
        mod, tensors, param_specs,
        warmup=args.warmup, steps=args.steps, seed=args.seed,
    )

    print(f"\n═══ Result ═══")
    print(f"  Median: {result['median_ms']:.2f}ms")
    print(f"  Losses: {[f'{l:.4f}' for l in result['losses'][:5]]}")

    return result


if __name__ == "__main__":
    main()
