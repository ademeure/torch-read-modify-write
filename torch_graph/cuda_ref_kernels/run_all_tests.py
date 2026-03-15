#!/usr/bin/env python3
"""Run all reference CUDA kernel tests. Reports pass/fail for each op."""
import importlib
import sys
import os
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

HERE = Path(__file__).parent
passed = []
failed = []

for f in sorted(HERE.glob("aten_*.py")):
    mod_name = f"torch_graph.cuda_ref_kernels.{f.stem}"
    try:
        mod = importlib.import_module(mod_name)
        mod.test()
        passed.append(f.stem)
    except Exception as e:
        failed.append((f.stem, str(e)))
        traceback.print_exc()
        print(f"FAIL {f.stem}: {e}\n")

print(f"\n{'='*60}")
print(f"Results: {len(passed)} passed, {len(failed)} failed out of {len(passed)+len(failed)}")
if failed:
    print("Failed:")
    for name, err in failed:
        print(f"  {name}: {err[:80]}")
