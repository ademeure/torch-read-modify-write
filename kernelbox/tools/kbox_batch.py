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
    """Compare result tensors to expected. Returns (ok, message).

    Handles NaN: NaN in both result and expected at the same position = match.
    Handles Inf: must match sign.
    """
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
            # Mask out positions where both are NaN (those are correct)
            both_nan = torch.isnan(rf) & torch.isnan(ef)
            # Check NaN mismatches: one is NaN but not the other
            nan_mismatch = torch.isnan(rf) != torch.isnan(ef)
            if nan_mismatch.any():
                idx = nan_mismatch.nonzero(as_tuple=True)[0][0].item()
                return False, f"output[{i}]: NaN mismatch at [{idx}] (got={rf[idx]}, expected={ef[idx]})"
            # Compare non-NaN values
            mask = ~both_nan
            if mask.any():
                diff = (rf[mask] - ef[mask]).abs().max().item()
                if not torch.allclose(rf[mask], ef[mask], atol=atol, rtol=rtol):
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
    p.add_argument("--seeds", type=int, default=1,
                   help="Number of random seeds to test per op (seed 0 = special values). Default: 1")
    p.add_argument("--sizes", action="store_true",
                   help="Also test with different tensor sizes (16, 256, 4096) per op")
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

            run_fn = getattr(mod, "run", None)
            if run_fn is None:
                failed += 1
                failures.append((name, "no run"))
                print(f"FAIL {name}: no run function", flush=True)
                continue

            sig = inspect.signature(run_fn)
            needs_kernel = len(sig.parameters) >= 2

            # Reconfigure session once per op (kernel source doesn't change per seed)
            if needs_kernel and "kernel_source" in state:
                session.reconfigure(
                    kernel_source=state["kernel_source"],
                    outputs=state.get("outputs", 1),
                    grid=state.get("grid"),
                    block=state.get("block"),
                    smem=state.get("smem"),
                )

            # Determine seeds and sizes to test
            has_make_inputs = hasattr(mod, "make_inputs")
            has_expected_fn = hasattr(mod, "expected")
            num_seeds = args.seeds if (has_make_inputs and has_expected_fn) else 1

            # Check if make_inputs accepts 'n' parameter for size testing
            has_n_param = False
            if has_make_inputs:
                mi_sig = inspect.signature(mod.make_inputs)
                has_n_param = 'n' in mi_sig.parameters

            # Build test configs: [(n, seed), ...]
            seeds = list(range(num_seeds)) if num_seeds > 1 else [1]
            if args.sizes and has_n_param and has_make_inputs and has_expected_fn:
                # Test multiple sizes, but only with seed=1 for non-default sizes
                configs = [(1024, s) for s in seeds]  # default size, all seeds
                for sz in [16, 256, 4096]:
                    configs.append((sz, 1))  # extra sizes, seed=1 only
            else:
                configs = [(1024, s) for s in seeds]

            atol = state.get("atol", args.atol)
            rtol = state.get("rtol", args.rtol)

            all_ok = True
            fail_msg = ""
            num_tested = 0
            for test_n, seed in configs:
                if has_make_inputs and has_expected_fn:
                    if has_n_param:
                        inputs = mod.make_inputs(n=test_n, seed=seed)
                    else:
                        inputs = mod.make_inputs(seed=seed)
                    exp = mod.expected(inputs)
                else:
                    inputs = state.get("inputs", [])
                    exp = state.get("expected", [])

                if exp == "skip":
                    # RNG ops etc — just verify kernel runs without error
                    if needs_kernel and "kernel_source" in state:
                        run_fn(inputs, session)
                    else:
                        run_fn(inputs)
                    all_ok = True
                    break

                if needs_kernel and "kernel_source" in state:
                    result = run_fn(inputs, session)
                elif needs_kernel:
                    skipped += 1
                    all_ok = None
                    break
                else:
                    result = run_fn(inputs)

                ok, msg = _verify(result, exp, atol, rtol)
                if not ok:
                    all_ok = False
                    fail_msg = f"n={test_n},seed={seed}: {msg}"
                    break
                num_tested += 1

            if all_ok is None:
                continue  # skipped
            elif all_ok:
                label = f"({num_tested} configs)" if num_tested > 1 else ""
                passed += 1
                print(f"PASS {name} {label}", flush=True)
            else:
                failed += 1
                failures.append((name, fail_msg))
                print(f"FAIL {name}: {fail_msg}", flush=True)

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
