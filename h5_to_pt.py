#!/usr/bin/env python3
"""Convert an H5 tensor dump to a .pt file.

Usage:
    python h5_to_pt.py model.h5                        # -> model.pt
    python h5_to_pt.py model.h5 --out custom_name.pt   # custom output path
    python h5_to_pt.py model.h5 --bf16                 # store as bfloat16
"""

import argparse
import sys
from pathlib import Path

import torch

from torch_graph._utils import load_h5_tensors


def h5_to_pt(h5_path: str, pt_path: str | None = None,
             storage_dtype: torch.dtype | None = None) -> str:
    """Convert all tensors from an H5 file to a .pt file.

    Returns the output path.
    """
    if pt_path is None:
        pt_path = str(Path(h5_path).with_suffix(".pt"))

    pt_data: dict[str, torch.Tensor] = load_h5_tensors(h5_path)

    if not pt_data:
        print(f"No /tensors group in {h5_path}", file=sys.stderr)
        sys.exit(1)

    if storage_dtype is not None:
        for name, val in pt_data.items():
            if val.is_floating_point():
                pt_data[name] = val.to(storage_dtype)

    # Preserve metadata if present
    import h5py
    with h5py.File(h5_path, "r") as f:
        if "_meta" in f:
            meta = {}
            for k, v in f["_meta"].attrs.items():
                meta[k] = v
            pt_data["_meta"] = meta

    torch.save(pt_data, pt_path)
    return pt_path


def main():
    p = argparse.ArgumentParser(
        description="Convert H5 tensor dump to .pt file.",
    )
    p.add_argument("h5", help="Path to the H5 file")
    p.add_argument("--out", help="Output .pt path (default: same name with .pt extension)")
    p.add_argument("--bf16", action="store_true",
                   help="Store floating-point tensors as bfloat16")

    args = p.parse_args()

    storage_dtype = torch.bfloat16 if args.bf16 else None
    pt_path = h5_to_pt(args.h5, args.out, storage_dtype=storage_dtype)

    pt_size = Path(pt_path).stat().st_size / (1024 * 1024)
    h5_size = Path(args.h5).stat().st_size / (1024 * 1024)
    print(f"Converted {args.h5} ({h5_size:.1f} MiB) -> {pt_path} ({pt_size:.1f} MiB)")


if __name__ == "__main__":
    main()
