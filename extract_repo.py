#!/usr/bin/env python3
"""Extract real aten graphs from any PyTorch repo.

Usage
-----
# nanoGPT with real Shakespeare data, capture at step 0 (fresh init):
python extract_repo.py --recipe recipes/nanogpt.py

# Capture after 5 warmup training steps:
python extract_repo.py --recipe recipes/nanogpt.py --warmup 5

# Any other repo — write a recipe (see recipes/nanogpt.py for the pattern):
python extract_repo.py --recipe recipes/my_model.py --warmup 10 --output outputs/my_model

Recipe convention
-----------------
A recipe is a Python file that defines::

    def setup() -> dict:
        return {
            "model":       nn.Module,           # required
            "sample_args": tuple of tensors,     # required – one batch
            "loss_fn":     callable | None,      # output -> scalar
            "get_batch":   callable | None,      # step -> (args, kwargs)
            "optimizer":   Optimizer | None,
        }
"""

import argparse
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(__file__))

from torch_graph.extract import extract_training_step, extract_function, load_recipe


def main():
    parser = argparse.ArgumentParser(
        description="Extract aten graphs from a real PyTorch training step",
    )
    parser.add_argument(
        "--recipe", required=True,
        help="Path to a recipe Python file (see recipes/ for examples)",
    )
    parser.add_argument(
        "--warmup", type=int, default=None,
        help="Training steps before capture (legacy; prefer --steps)",
    )
    parser.add_argument(
        "--steps", type=str, default=None,
        help="Comma-separated step numbers to capture at, e.g. '0,5,10'",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: outputs/<recipe_name>)",
    )
    parser.add_argument(
        "--prefix", default=None,
        help="Filename prefix (default: derived from recipe name)",
    )
    parser.add_argument(
        "--max-intermediates-mb", type=float, default=None,
        help="Cap total intermediate tensor storage at N MiB (inputs/outputs always saved). "
             "Skips optimizer extraction when set.",
    )
    parser.add_argument(
        "--storage-dtype", default=None, choices=["bfloat16", "float16"],
        help="Store tensors in this dtype to reduce .pt file size (e.g. bfloat16)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Device for extraction (default: auto = CUDA if available)",
    )
    parser.add_argument(
        "--triton", action="store_true", default=False,
        help="Capture Triton kernels via inductor (requires CUDA)",
    )
    parser.add_argument(
        "--no-record", action="store_true", default=False,
        help="Disable intermediate tensor recording (faster capture)",
    )
    parser.add_argument(
        "--record-lines", type=str, default=None,
        help='Selectively record intermediates by source lines, e.g. "model.py:42,55,100-120"',
    )
    parser.add_argument(
        "--record-module", type=str, default=None,
        help='Selectively record intermediates by module path, e.g. "blocks.0.attn"',
    )
    parser.add_argument(
        "--record-pattern", type=str, default=None,
        help='Selectively record intermediates by node name glob, e.g. "addmm*"',
    )
    parser.add_argument(
        "--capture-optimizer", action="store_true", default=False,
        help="Trace optimizer.step() via torch.compile to capture optimizer aten ops",
    )
    parser.add_argument(
        "--setup-fn", default="setup",
        help="Name of the setup function in the recipe (default: setup). "
             "E.g. setup_sft, setup_rl for nanochat variants.",
    )
    parser.add_argument(
        "--graph-only", action="store_true", default=False,
        help="Only generate the graph HTML (skip .pt weights file for faster runs)",
    )
    args = parser.parse_args()

    if args.steps is not None:
        steps = [int(s.strip()) for s in args.steps.split(",")]
    elif args.warmup is not None:
        steps = [args.warmup]
    else:
        steps = [0]

    recipe_name = os.path.splitext(os.path.basename(args.recipe))[0]
    output_dir = args.output or os.path.join("outputs", recipe_name)
    prefix = args.prefix or recipe_name

    print("=" * 60)
    print(f" Loading recipe: {args.recipe}")
    print("=" * 60)

    storage_dtype = None
    if args.storage_dtype == "bfloat16":
        storage_dtype = torch.bfloat16
    elif args.storage_dtype == "float16":
        storage_dtype = torch.float16

    recipe = load_recipe(args.recipe, setup_fn=args.setup_fn)

    # Build record_filter from CLI flags
    record_filter = None
    if args.record_lines or args.record_module or args.record_pattern:
        record_filter = {}
        if args.record_lines:
            record_filter["lines"] = args.record_lines
        if args.record_module:
            record_filter["module"] = args.record_module
        if args.record_pattern:
            record_filter["pattern"] = args.record_pattern

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ── Function recipe mode ──────────────────────────────────────
    # A recipe can return "fn" + "args" instead of "model" + "sample_args"
    # to capture an arbitrary callable rather than a full training step.
    if "fn" in recipe:
        fn = recipe["fn"]
        fn_args = recipe.get("args", ())
        fn_kwargs = recipe.get("kwargs", {})
        loss_fn = recipe.get("loss_fn")
        run_backward = recipe.get("run_backward", loss_fn is not None)
        param_names = recipe.get("param_names")

        print(f"  Mode: function capture")
        for i, a in enumerate(fn_args):
            if hasattr(a, "shape"):
                print(f"  Input {i}: shape={list(a.shape)} dtype={a.dtype}")

        result = extract_function(
            fn, *fn_args,
            run_backward=run_backward,
            loss_fn=loss_fn,
            output_dir=output_dir,
            prefix=prefix,
            max_intermediates_mb=args.max_intermediates_mb,
            storage_dtype=storage_dtype,
            device=device,
            record_real_tensors=not args.no_record,
            record_filter=record_filter,
            param_names=param_names,
            **fn_kwargs,
        )
    else:
        # ── Standard model recipe mode ────────────────────────────
        model = recipe["model"]
        sample_args = recipe["sample_args"]
        loss_fn = recipe.get("loss_fn")
        get_batch = recipe.get("get_batch")
        optimizer = recipe.get("optimizer")
        step_fn = recipe.get("step_fn")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")
        for i, a in enumerate(sample_args):
            if hasattr(a, "shape"):
                print(f"  Input {i}: shape={list(a.shape)} dtype={a.dtype}")
        if args.max_intermediates_mb is not None:
            print(f"  Intermediates cap: {args.max_intermediates_mb} MiB")
        if args.no_record:
            print(f"  Intermediate recording: disabled")
        if record_filter:
            print(f"  Record filter: {record_filter}")
        if args.capture_optimizer:
            print(f"  Optimizer capture: enabled")
        if storage_dtype is not None:
            print(f"  Storage dtype: {args.storage_dtype}")

        if args.triton:
            if not torch.cuda.is_available():
                print("ERROR: --triton requires CUDA but no GPU is available.")
                sys.exit(1)
            print(f"  Triton capture: enabled")

        steps_str = ", ".join(str(s) for s in sorted(steps))
        print()
        print("=" * 60)
        print(f" Extracting at step(s): {steps_str}")
        print("=" * 60)

        result = extract_training_step(
            model=model,
            sample_args=sample_args,
            loss_fn=loss_fn,
            optimizer=optimizer,
            steps=steps,
            get_batch=get_batch,
            output_dir=output_dir,
            prefix=prefix,
            max_intermediates_mb=args.max_intermediates_mb,
            storage_dtype=storage_dtype,
            device=device,
            triton=args.triton,
            record_real_tensors=not args.no_record,
            record_filter=record_filter,
            capture_optimizer=args.capture_optimizer,
            graph_only=args.graph_only,
            step_fn=step_fn,
        )

    print()
    print("=" * 60)
    print(" Done!")
    print("=" * 60)
    for f in result["files"]:
        print(f"  {f}")
    print()
    py_file = next((f for f in result["files"] if f.endswith("_aten.py")), None)
    if py_file:
        print("  Verify the exported program:")
        if len(steps) > 1:
            for s in sorted(steps):
                print(f"    python {py_file} --step {s}")
        else:
            print(f"    python {py_file}")
    print()
    html_file = next((f for f in result["files"] if f.endswith(".html")), None)
    if html_file:
        print(f"  View the graph:")
        print(f"    open {html_file}")


if __name__ == "__main__":
    main()
