#!/usr/bin/env python3
"""kbox batch — Run all test files in a directory, single process.

Reuses one KernelSession (one worker daemon, one VMM pool) across all
tests via reconfigure(). Each test takes ~50ms instead of ~1.5s.

Usage:
    kbox batch <directory>
    kbox batch torch_graph/cuda_ref_kernels/
"""
import argparse
import sys
import os
import time
import glob
import importlib.util
import inspect
import traceback
from pathlib import Path


def _ensure_kernelbox_importable():
    try:
        import kernelbox  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent / "python", here.parent.parent.parent / "python"]:
        if (candidate / "kernelbox").is_dir():
            sys.path.insert(0, str(candidate))
            return


_ensure_kernelbox_importable()


def _load_test_module(path):
    """Load a test .py file as a module."""
    name = os.path.basename(path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verify(result, expected, atol, rtol):
    """Compare result tensors to expected. Returns (ok, message)."""
    import torch
    if not isinstance(result, (list, tuple)):
        result = [result]
    for i, (r, e) in enumerate(zip(result, expected)):
        if not isinstance(r, torch.Tensor) or not isinstance(e, torch.Tensor):
            continue
        if r.dtype != e.dtype:
            r = r.to(e.dtype)
        rf, ef = r.float().flatten(), e.float().flatten()
        if rf.shape != ef.shape:
            return False, f"output[{i}]: shape {tuple(rf.shape)} != {tuple(ef.shape)}"
        if e.dtype in (torch.bool, torch.long, torch.int32, torch.int64):
            if not torch.equal(rf, ef):
                return False, f"output[{i}]: int/bool mismatch"
        else:
            diff = (rf - ef).abs().max().item()
            if not torch.allclose(rf, ef, atol=atol, rtol=rtol):
                return False, f"output[{i}]: max_err={diff:.2e} (atol={atol})"
    return True, "ok"


def main():
    p = argparse.ArgumentParser(
        prog="kbox batch",
        description="Run all test files in a directory (single process, fast).",
    )
    p.add_argument("directory", help="Directory containing test .py files")
    p.add_argument("--pattern", default="aten_*.py", help="Glob pattern (default: aten_*.py)")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--filter", default=None, help="Only run tests matching this substring")
    args = p.parse_args()

    directory = os.path.abspath(args.directory)
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Add parent dirs to path so test imports work
    for depth in range(4):
        parent = directory
        for _ in range(depth):
            parent = os.path.dirname(parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

    files = sorted(glob.glob(os.path.join(directory, args.pattern)))
    if not files:
        print(f"No files matching '{args.pattern}' in {directory}")
        sys.exit(1)

    import torch
    from kernelbox.dev import KernelSession

    # Create one session with a dummy kernel — will reconfigure per test
    dummy_src = 'extern "C" __global__ void _dummy(float *x, unsigned int n) {}'
    session = KernelSession(kernel_source=dummy_src)

    passed = failed = skipped = 0
    failures = []
    t0 = time.time()

    for f in files:
        name = os.path.basename(f)[:-3]
        if args.filter and args.filter not in name:
            continue

        try:
            mod = _load_test_module(f)
            init_fn = getattr(mod, "init_once", None) or getattr(mod, "init", None)
            if not init_fn:
                skipped += 1
                continue

            state = init_fn()
            if not isinstance(state, dict):
                failed += 1
                failures.append((name, "init must return dict"))
                print(f"FAIL {name}: init must return dict", flush=True)
                continue

            inputs = state.get("inputs", [])
            expected = state.get("expected", [])
            atol = state.get("atol", args.atol)
            rtol = state.get("rtol", args.rtol)
            run_fn = getattr(mod, "run", None)

            if run_fn is None:
                failed += 1
                failures.append((name, "no run"))
                print(f"FAIL {name}: no run function", flush=True)
                continue

            sig = inspect.signature(run_fn)
            needs_kernel = len(sig.parameters) >= 2

            if needs_kernel and "kernel_source" in state:
                session.reconfigure(
                    kernel_source=state["kernel_source"],
                    outputs=state.get("outputs", 1),
                    grid=state.get("grid"),
                    block=state.get("block"),
                    smem=state.get("smem"),
                )
                result = run_fn(inputs, session)
            elif needs_kernel:
                skipped += 1
                continue
            else:
                result = run_fn(inputs)

            ok, msg = _verify(result, expected, atol, rtol)
            if ok:
                passed += 1
                print(f"PASS {name}", flush=True)
            else:
                failed += 1
                failures.append((name, msg))
                print(f"FAIL {name}: {msg}", flush=True)

        except Exception as ex:
            failed += 1
            failures.append((name, str(ex)[:100]))
            print(f"FAIL {name}: {ex}", flush=True)

    session.close()
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    total = passed + failed + skipped
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped out of {total} ({elapsed:.1f}s)")
    if failures:
        print("Failed:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
