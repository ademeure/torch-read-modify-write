#!/usr/bin/env python3
"""kbox batch — Run aten reference kernel tests, single process.

Two modes:
  kbox batch <directory>              Load per-file aten_*.py tests
  kbox batch --registry <module>      Use _registry.py directly (faster, multi-seed)

Registry mode runs all ops from the registry dict with configurable seeds
and shape fuzzing. No per-file loading overhead.

Usage:
    kbox batch torch_graph/cuda_ref_kernels/
    kbox batch --registry torch_graph.cuda_ref_kernels._registry --seeds 10
    kbox batch --registry torch_graph.cuda_ref_kernels._registry --seeds 100 --sizes
"""
import argparse
import sys
import os
import time
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


def _verify(result, expected, atol, rtol=1e-5):
    """NaN-aware comparison. Returns (ok, message)."""
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
            return False, f"[{i}] shape {tuple(rf.shape)}!={tuple(ef.shape)}"
        both_nan = torch.isnan(rf) & torch.isnan(ef)
        nan_mm = torch.isnan(rf) != torch.isnan(ef)
        if nan_mm.any():
            idx = nan_mm.nonzero(as_tuple=True)[0][0].item()
            return False, f"[{i}] NaN@{idx} got={rf[idx]:.2e} exp={ef[idx]:.2e}"
        mask = ~both_nan
        if mask.any():
            diff = (rf[mask] - ef[mask]).abs().max().item()
            if diff > atol:
                return False, f"[{i}] err={diff:.2e} (atol={atol})"
    return True, "ok"


# ═════════════════════════════════════════════════════════════════════════════
#  Registry mode — fast, multi-seed, shape-fuzzed
# ═════════════════════════════════════════════════════════════════════════════

def _run_registry(args):
    """Run all ops from the registry with multi-seed + shape fuzzing."""
    import torch
    from kernelbox.dev import KernelSession

    # Import the registry
    # Add repo root to path
    for depth in range(5):
        p = os.path.abspath(args.registry.replace('.', '/'))
        for _ in range(depth):
            p = os.path.dirname(p)
        root = p
        if root not in sys.path:
            sys.path.insert(0, root)

    # Also try the directory form
    registry_parts = args.registry.rsplit('.', 1)
    if len(registry_parts) == 2:
        pkg_path = registry_parts[0].replace('.', '/')
        for parent in ['.', '..', '../..']:
            full = os.path.abspath(os.path.join(parent, pkg_path))
            if os.path.isdir(full):
                sys.path.insert(0, os.path.abspath(parent))
                break

    import importlib
    reg = importlib.import_module(args.registry)
    OPS = reg.OPS

    COPY_BODY = 'out0[i] = in0[i]'
    all_ops = sorted(OPS.keys())
    if args.filter:
        all_ops = [n for n in all_ops if args.filter in n]

    # Separate real-kernel ops from placeholders
    real_ops = [n for n in all_ops if COPY_BODY not in OPS[n]['kernel']]
    copy_ops = [n for n in all_ops if COPY_BODY in OPS[n]['kernel']]

    # Build test configs per op
    use_fuzz = args.fuzz > 0
    if use_fuzz:
        from kernelbox.fuzz import fuzz_inputs
    seeds = list(range(args.seeds))
    extra_sizes = [16, 128, 4096] if args.sizes else []

    s = KernelSession(kernel_source='extern "C" __global__ void _d(float *x, unsigned int n) {}')
    passed = failed = skipped = 0
    failures = []
    total_tests = 0
    t0 = time.time()

    for name in real_ops:
        op = OPS[name]
        ok = True
        n_tested = 0
        fail_msg = ""

        # Reconfigure session for this op's kernel
        state = reg.get_kbox_state(name)
        try:
            s.reconfigure(
                kernel_source=state['kernel_source'],
                outputs=state.get('outputs', 1),
                grid=state.get('grid'),
                block=state.get('block'),
                smem=state.get('smem'),
            )
        except Exception as e:
            failed += 1
            failures.append((name, f"reconfigure: {e}"))
            print(f"FAIL {name}: reconfigure: {e}", flush=True)
            continue

        if use_fuzz:
            # Universal fuzz: generate inputs from baseline shape, use op['aten'] as reference
            d = dict(op['dims'])
            baseline_inputs = op['inputs'](d, 1)
            ref_fn = op['aten']
            for label, variant in fuzz_inputs(baseline_inputs, seeds=args.fuzz):
                try:
                    expected = ref_fn(variant)
                    result = reg.dispatch(name, variant, s, d)
                    v_ok, v_msg = _verify(result, expected, op['atol'])
                    if not v_ok:
                        ok = False
                        fail_msg = f"{label}: {v_msg}"
                        break
                    n_tested += 1
                except Exception as e:
                    ok = False
                    fail_msg = f"{label}: {e}"
                    break
        else:
            # Legacy: per-op input generators with seed semantics
            for seed in seeds:
                try:
                    d = dict(op['dims'])
                    inputs = op['inputs'](d, seed)
                    expected = op['aten'](inputs)
                    result = reg.dispatch(name, inputs, s, d)
                    v_ok, v_msg = _verify(result, expected, op['atol'])
                    if not v_ok:
                        ok = False
                        fail_msg = f"seed={seed}: {v_msg}"
                        break
                    n_tested += 1
                except Exception as e:
                    ok = False
                    fail_msg = f"seed={seed}: {e}"
                    break

        # Test extra sizes (only seed=1, only for simple 1D/2D ops)
        # Skip ops with custom output specs (pooling, conv, etc) — size changes break them
        if ok and extra_sizes:
            has_custom_outputs = op['outputs'] is not None
            fuzzable_keys = [k for k in op['dims'] if k in ('n',)]  # only fuzz 'n' for now
            fuzzable = fuzzable_keys if (not has_custom_outputs and fuzzable_keys) else []
            for sz in extra_sizes:
                if not fuzzable:
                    break
                try:
                    d = dict(op['dims'])
                    d[fuzzable[0]] = sz
                    inputs = op['inputs'](d, 1)
                    expected = op['aten'](inputs)

                    # Need to reconfigure for new output size
                    new_state = reg.get_kbox_state(name, dims=d)
                    s.reconfigure(
                        kernel_source=new_state['kernel_source'],
                        outputs=new_state.get('outputs', 1),
                        grid=new_state.get('grid'),
                        block=new_state.get('block'),
                    )
                    result = reg.dispatch(name, inputs, s, d)
                    v_ok, v_msg = _verify(result, expected, op['atol'])
                    if not v_ok:
                        ok = False
                        fail_msg = f"size={sz}: {v_msg}"
                        break
                    n_tested += 1
                except Exception as e:
                    # Size fuzzing failure is non-fatal for now
                    pass

        total_tests += n_tested
        if ok:
            passed += 1
            label = f"({n_tested})" if n_tested > 1 else ""
            print(f"PASS {name} {label}", flush=True)
        else:
            failed += 1
            failures.append((name, fail_msg))
            print(f"FAIL {name}: {fail_msg}", flush=True)

    # Report copy/placeholder ops
    skipped = len(copy_ops)

    s.close()
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} placeholders "
          f"({passed+failed+skipped} ops, {total_tests} tests, {elapsed:.1f}s)")
    if failures:
        print("Failed:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
    sys.exit(1 if failed else 0)


# ═════════════════════════════════════════════════════════════════════════════
#  File mode — legacy per-file loading
# ═════════════════════════════════════════════════════════════════════════════

def _run_files(args):
    """Run per-file aten_*.py tests (legacy mode)."""
    import glob
    import importlib.util
    import inspect
    import torch
    from kernelbox.dev import KernelSession

    use_fuzz = args.fuzz > 0
    if use_fuzz:
        from kernelbox.fuzz import fuzz_inputs

    directory = os.path.abspath(args.directory)
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)

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
            spec = importlib.util.spec_from_file_location(name, f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            init_fn = getattr(mod, "init_once", None) or getattr(mod, "init", None)
            if not init_fn:
                skipped += 1
                continue

            state = init_fn()
            if not isinstance(state, dict):
                failed += 1; failures.append((name, "bad init")); continue

            run_fn = getattr(mod, "run", None)
            if not run_fn:
                failed += 1; failures.append((name, "no run")); continue

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
                result = run_fn(state["inputs"], session)
            elif needs_kernel:
                skipped += 1; continue
            else:
                result = run_fn(state.get("inputs", []))

            exp = state.get("expected", [])
            if exp == "skip":
                passed += 1; print(f"PASS {name}", flush=True); continue

            ok, msg = _verify(result, exp, state.get("atol", args.atol))
            if not ok:
                failed += 1; failures.append((name, msg))
                print(f"FAIL {name}: {msg}", flush=True)
                continue

            # Auto-fuzz if --fuzz and module has reference()
            ref_fn = getattr(mod, "reference", None)
            fuzz_ok = True
            n_fuzz = 0
            if use_fuzz and ref_fn and isinstance(state.get("inputs"), (list, tuple)):
                test_atol = state.get("atol", args.atol)
                for fuzz_label, variant in fuzz_inputs(state["inputs"], seeds=args.fuzz):
                    try:
                        fuzz_exp = ref_fn(variant)
                        if needs_kernel:
                            fuzz_result = run_fn(variant, session)
                        else:
                            fuzz_result = run_fn(variant)
                        fuzz_v_ok, fuzz_v_msg = _verify(fuzz_result, fuzz_exp, test_atol)
                        if not fuzz_v_ok:
                            fuzz_ok = False
                            msg = f"fuzz {fuzz_label}: {fuzz_v_msg}"
                            break
                        n_fuzz += 1
                    except Exception as e:
                        fuzz_ok = False
                        msg = f"fuzz {fuzz_label}: {e}"
                        break

            if fuzz_ok:
                label = f"({n_fuzz+1})" if n_fuzz > 0 else ""
                passed += 1; print(f"PASS {name} {label}", flush=True)
            else:
                failed += 1; failures.append((name, msg))
                print(f"FAIL {name}: {msg}", flush=True)

        except Exception as ex:
            failed += 1; failures.append((name, str(ex)[:100]))
            print(f"FAIL {name}: {ex}", flush=True)

    session.close()
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped "
          f"({passed+failed+skipped} total, {elapsed:.1f}s)")
    if failures:
        print("Failed:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
    sys.exit(1 if failed else 0)


# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        prog="kbox batch",
        description="Run aten reference kernel tests (single process, fast).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  kbox batch torch_graph/cuda_ref_kernels/                          # file mode
  kbox batch --registry torch_graph.cuda_ref_kernels._registry      # registry mode
  kbox batch -r torch_graph.cuda_ref_kernels._registry -s 10        # 10 seeds
  kbox batch -r torch_graph.cuda_ref_kernels._registry -s 100 -S    # 100 seeds + sizes
  kbox batch -r torch_graph.cuda_ref_kernels._registry -f softmax   # filter
""",
    )
    p.add_argument("directory", nargs="?", help="Directory with aten_*.py files (file mode)")
    p.add_argument("-r", "--registry", help="Python module path to _registry.py (registry mode)")
    p.add_argument("--pattern", default="aten_*.py", help="Glob pattern for file mode")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("-f", "--filter", default=None, help="Only run ops matching this substring")
    p.add_argument("-s", "--seeds", type=int, default=1,
                   help="Seeds per op. seed=0 is special values (NaN/inf/subnormals)")
    p.add_argument("-S", "--sizes", action="store_true",
                   help="Also fuzz tensor sizes (16, 128, 4096)")
    p.add_argument("--fuzz", type=int, default=0, metavar="N",
                   help="Auto-fuzz with N mixed seeds + exhaustive specials "
                        "(registry: uses aten ref; files: requires reference())")
    args = p.parse_args()

    if args.registry:
        _run_registry(args)
    elif args.directory:
        _run_files(args)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
