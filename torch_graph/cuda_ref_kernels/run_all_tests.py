#!/usr/bin/env python3
"""Run all reference CUDA kernel tests. Reports pass/fail for each op.

Supports two test formats:
  1. kernelbox (init_once + run): PyTorch-only ops run directly, kernel ops skip
  2. Legacy (test()): direct compile + check via load_inline

For full verification of kernel ops, use: kbox iterate aten_*.py --once
"""
import importlib
import sys
import os
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

HERE = Path(__file__).parent
passed = []
skipped = []
failed = []

# Optional filter
filter_name = None
for arg in sys.argv[1:]:
    if not arg.startswith("-"):
        filter_name = arg

for f in sorted(HERE.glob("aten_*.py")):
    if filter_name and filter_name not in f.stem:
        continue
    mod_name = f"torch_graph.cuda_ref_kernels.{f.stem}"
    try:
        mod = importlib.import_module(mod_name)

        if hasattr(mod, "test"):
            # Legacy format
            mod.test()
            passed.append(f.stem)
        elif hasattr(mod, "init_once"):
            import inspect
            state = mod.init_once()
            inputs = state.get("inputs", [])
            expected = state.get("expected", [])

            sig = inspect.signature(mod.run)
            if len(sig.parameters) == 1:
                # PyTorch-only op — run directly
                result = mod.run(inputs)
                import torch
                for i, (r, e) in enumerate(zip(result, expected)):
                    if r.dtype != e.dtype:
                        r = r.to(e.dtype)
                    atol = state.get("atol", 1e-5)
                    if e.dtype in (torch.bool,):
                        assert torch.equal(r, e), f"{f.stem}[{i}]: bool mismatch"
                    elif e.dtype in (torch.long, torch.int32, torch.int64):
                        assert torch.equal(r, e), f"{f.stem}[{i}]: int mismatch"
                    else:
                        diff = (r.float() - e.float()).abs().max().item()
                        assert torch.allclose(r.float(), e.float(), atol=atol, rtol=1e-5), \
                            f"{f.stem}[{i}]: max diff {diff:.2e} (atol={atol})"
                print(f"PASS {f.stem}")
                passed.append(f.stem)
            else:
                # Needs kernel — skip (requires kbox)
                print(f"SKIP {f.stem} (needs kbox)")
                skipped.append(f.stem)
        else:
            print(f"SKIP {f.stem} (no test/init_once)")
            skipped.append(f.stem)

    except Exception as e:
        failed.append((f.stem, str(e)))
        traceback.print_exc()
        print(f"FAIL {f.stem}: {e}\n")

print(f"\n{'='*60}")
print(f"Results: {len(passed)} passed, {len(skipped)} skipped, {len(failed)} failed out of {len(passed)+len(skipped)+len(failed)}")
if failed:
    print("Failed:")
    for name, err in failed:
        print(f"  {name}: {err[:80]}")
