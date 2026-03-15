#!/usr/bin/env python3
"""Capture modded-nanogpt aten graphs with Triton kernel support.

Generates:
  outputs/test_models/modded_nanogpt/
    modded_nanogpt_aten.py  - Editable aten forward+backward graph with Triton kernels

Usage:
    python scripts/capture_modded_nanogpt.py              # forward + backward
    python scripts/capture_modded_nanogpt.py --forward-only  # forward only
    python scripts/capture_modded_nanogpt.py --optimizer     # also capture optimizer step
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    from recipes.modded_nanogpt_wrapper import setup
    from torch_graph.export import capture_aten_graphs, export_aten_program

    print("Setting up modded-nanogpt...")
    recipe = setup()
    model = recipe["model"]
    sample_args = recipe["sample_args"]
    sample_kwargs = recipe["sample_kwargs"]
    loss_fn = recipe["loss_fn"]

    print(f"Model: {type(model).__name__}, params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Args: {len(sample_args)}")
    for i, a in enumerate(sample_args):
        if isinstance(a, torch.Tensor):
            print(f"  [{i}] tensor shape={a.shape} dtype={a.dtype} device={a.device}")
        else:
            print(f"  [{i}] {type(a).__name__}: {a}")

    # Capture forward + backward
    run_backward = "--forward-only" not in sys.argv
    print(f"\nCapturing graph (run_backward={run_backward})...")
    torch._dynamo.reset()
    t0 = time.time()
    try:
        out, capture = capture_aten_graphs(
            model, *sample_args, **sample_kwargs,
            run_backward=run_backward,
            loss_fn=loss_fn if run_backward else None,
            record_real_tensors=True,
        )
        t1 = time.time()
        print(f"Capture done in {t1-t0:.1f}s")
        print(f"Forward graphs: {len(capture.forward_graphs)}")
        print(f"Backward graphs: {len(capture.backward_graphs)}")
        if capture.forward_graphs:
            gm = capture.forward_graphs[0].graph_module
            nodes = list(gm.graph.nodes)
            triton_nodes = [n for n in nodes if n.op == 'call_function'
                           and 'triton' in str(n.target).lower()]
            print(f"  Total nodes: {len(nodes)}")
            print(f"  Triton nodes: {len(triton_nodes)}")
            for n in triton_nodes:
                print(f"    {n.name}: kernel_idx={n.kwargs.get('kernel_idx')}")
    except Exception as e:
        print(f"Capture failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Optimizer capture (optional) — captures optimizer step as aten ops
    if "--optimizer" in sys.argv:
        opt_capture = _capture_optimizer(model, sample_args, sample_kwargs, loss_fn)
        if opt_capture is not None:
            capture.optimizer_capture = opt_capture

    # Export
    outdir = "outputs/test_models/modded_nanogpt"
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "modded_nanogpt_aten.py")

    print(f"\nExporting to {out_path}...")
    try:
        export_aten_program(
            capture,
            output_path=out_path,
            include_test_harness=True,
        )
        print(f"Export done: {os.path.getsize(out_path):,} bytes")
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Try running the exported file
    print(f"\nRunning exported file...")
    import subprocess
    result = subprocess.run(
        [sys.executable, out_path],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        print("SUCCESS!")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    else:
        print(f"FAILED (exit code {result.returncode})")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)


def _capture_optimizer(model, sample_args, sample_kwargs, loss_fn):
    """Capture optimizer step as aten ops. Returns AtenCapture or None."""
    from recipes.modded_nanogpt_wrapper import setup_optimizer
    from torch_graph.export import capture_optimizer_aten

    print("\n--- Optimizer Capture ---")
    optimizer = setup_optimizer(model)
    print(f"Optimizer: {type(optimizer).__name__}")

    # Prime optimizer state with one training step
    print("  Priming optimizer with one training step...")
    model.zero_grad(set_to_none=True)
    out = model(*sample_args, **sample_kwargs)
    loss = loss_fn(out)
    loss.backward()
    optimizer.step(do_adam=True)
    model.zero_grad(set_to_none=True)

    # Do another forward/backward to get fresh gradients
    out = model(*sample_args, **sample_kwargs)
    loss = loss_fn(out)
    loss.backward()

    print("  Capturing optimizer.step() via capture_optimizer_aten...")
    torch._dynamo.reset()
    try:
        opt_capture = capture_optimizer_aten(
            optimizer,
            record_real_tensors=False,
            param_name_map={id(p): n for n, p in model.named_parameters()},
            step_fn=lambda: optimizer.step(do_adam=True),
        )
        n_graphs = len(opt_capture.forward_graphs)
        if n_graphs > 0:
            n_nodes = len(list(opt_capture.forward_graphs[0].graph_module.graph.nodes))
            print(f"  Optimizer capture: {n_graphs} graph(s), {n_nodes} nodes")
        else:
            print("  Optimizer capture: no graphs produced")
        return opt_capture
    except Exception as e:
        print(f"  Optimizer capture failed: {e}")
        return None


if __name__ == "__main__":
    main()
