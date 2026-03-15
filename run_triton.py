#!/usr/bin/env python3
"""Capture Triton kernels from torch.compile and show aten-to-kernel mapping.

Usage:
    python run_triton.py              # requires CUDA GPU

Outputs:
    outputs/triton_kernels/           - individual Triton kernel source files
    stdout                            - kernel summary with fused op lists
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "test_repo")

import torch
from torch_graph.triton import capture_triton_kernels, save_triton_kernels
from model import NanoGPT

if not torch.cuda.is_available():
    print("ERROR: Triton capture requires a CUDA GPU.")
    print("Run on a machine with a GPU, or skip this script.")
    sys.exit(1)


print("=" * 60)
print(" Triton Kernel Capture: NanoGPT")
print("=" * 60)
print()

model = NanoGPT().cuda()
idx = torch.randint(0, 64, (2, 16)).cuda()

output, tcap, _ = capture_triton_kernels(model, idx)
print()
print(tcap.summary())
print()

saved = save_triton_kernels(tcap, "outputs/triton_kernels/", prefix="nanogpt_")
print(f"\nSaved {len(saved)} kernel files to outputs/triton_kernels/")
for f in saved:
    print(f"  {f}")

print("""
What this shows:
  - Each Triton kernel fuses multiple aten ops into one GPU kernel
  - 'extern' kernels are cuBLAS/cuDNN calls (matmul, conv, etc.)
  - The full inductor output code is saved for inspection

Look at the kernel files to see the actual Triton code that runs on GPU.
""")
