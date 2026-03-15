#!/usr/bin/env python3
"""Capture NanoGPT (test_repo/) aten graphs: forward, backward + HTML + aten .py + H5 + kbox.

Generates:
  outputs/nanogpt/
    nanogpt_aten.py              - Editable aten forward+backward graph
    nanogpt_forward.html         - Interactive HTML visualization (forward)
    nanogpt_backward.html        - Interactive HTML visualization (backward)
    nanogpt_forward.json         - Forward graph JSON
    nanogpt_combined.json        - Combined forward+backward JSON
    nanogpt.h5                   - H5 tensor dump (forward + backward)
    nanogpt_scripts/             - Replay scripts
    nanogpt_kbox/                - Kernelbox test scripts

Usage:
    python scripts/capture_nanogpt.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test_repo"))

import torch
from torch_graph.export import capture_aten_graphs, export_aten_program
from torch_graph.op_dump import dump_grouped_tensors
from torch_graph.visualizer import GraphVisualizer
from torch_graph import compute_tensor_stats

OUTPUT_DIR = os.path.join(ROOT, "outputs", "nanogpt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Build model ──────────────────────────────────────────────────────

print("=" * 70)
print(" NANOGPT CAPTURE")
print("=" * 70)

from model import NanoGPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NanoGPT(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32).to(device)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Device: {device}")

inputs = (torch.randint(0, 64, (2, 16), device=device),)

# ══════════════════════════════════════════════════════════════════════
# Phase 1: Capture forward + backward aten graphs
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 1: Capture forward + backward aten graphs")
print("=" * 70)

torch._dynamo.reset()
out, capture = capture_aten_graphs(
    model, *inputs,
    run_backward=True,
    record_real_tensors=True,
)
print(f"  Forward graph: {len(capture.forward_graphs[0].graph_module.graph.nodes)} nodes")
if capture.backward_graphs:
    print(f"  Backward graph: {len(capture.backward_graphs[0].graph_module.graph.nodes)} nodes")

# ══════════════════════════════════════════════════════════════════════
# Phase 2: Export aten Python files
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 2: Export aten Python files")
print("=" * 70)

aten_path = os.path.join(OUTPUT_DIR, "nanogpt_aten.py")
export_aten_program(capture, aten_path)
print(f"  Wrote: {aten_path}")

# ══════════════════════════════════════════════════════════════════════
# Phase 3: HTML visualizations
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 3: HTML visualizations")
print("=" * 70)

fg = capture.forward_graphs[0]
bw = capture.backward_graphs[0] if capture.backward_graphs else None

fw_stats = (compute_tensor_stats(capture.forward_intermediates)
            if capture.forward_intermediates else None)
bw_stats = (compute_tensor_stats(capture.backward_intermediates)
            if capture.backward_intermediates else None)

viz_fw = GraphVisualizer(fg)

# Forward HTML
fw_html_path = os.path.join(OUTPUT_DIR, "nanogpt_forward.html")
viz_fw.save_html(
    fw_html_path,
    title="NanoGPT Forward",
    tensor_stats=fw_stats,
    backward_source=bw,
    bw_tensor_stats=bw_stats,
)
print(f"  Forward HTML: {fw_html_path}")

# JSON
json_path = os.path.join(OUTPUT_DIR, "nanogpt_forward.json")
with open(json_path, "w") as f:
    json.dump(viz_fw.to_json(), f, indent=2, default=str)
print(f"  Forward JSON: {json_path}")

# Backward HTML
if bw:
    try:
        combined_json_path = os.path.join(OUTPUT_DIR, "nanogpt_combined.json")
        viz_fw.save_json(combined_json_path, backward_source=capture)
        print(f"  Combined JSON: {combined_json_path}")
    except Exception as e:
        print(f"  Combined JSON skipped: {e}")

    bw_html_path = os.path.join(OUTPUT_DIR, "nanogpt_backward.html")
    GraphVisualizer(bw).save_html(
        bw_html_path,
        title="NanoGPT Backward",
        tensor_stats=bw_stats,
    )
    print(f"  Backward HTML: {bw_html_path}")

# ══════════════════════════════════════════════════════════════════════
# Phase 4: H5 tensor dump
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 4: H5 tensor dump")
print("=" * 70)

h5_path = os.path.join(OUTPUT_DIR, "nanogpt.h5")
dump_grouped_tensors(
    capture, h5_path,
    group_by=["line", "module"],
    which="both",
    include_params=True,
    stats=True,
    replay_scripts=True,
    scripts_dir=os.path.join(OUTPUT_DIR, "nanogpt_scripts"),
)
print(f"  H5: {h5_path}")
print(f"  Scripts: {os.path.join(OUTPUT_DIR, 'nanogpt_scripts')}/")

# ══════════════════════════════════════════════════════════════════════
# Phase 5: Kbox generation
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 5: Kbox test scripts")
print("=" * 70)

from torch_graph.kbox_gen import generate_all as generate_kbox_scripts

kbox_dir = os.path.join(OUTPUT_DIR, "nanogpt_kbox")

try:
    scripts = generate_kbox_scripts(h5_path, out_dir=kbox_dir)
    print(f"  Kbox dir: {kbox_dir}/")
    print(f"  Generated {len(scripts)} kbox test scripts")
except Exception as e:
    print(f"  Kbox generation failed: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
# Phase 6: H5-embedded HTML
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 6: H5-embedded HTML")
print("=" * 70)

h5_html_path = os.path.join(OUTPUT_DIR, "nanogpt_graph_h5web.html")
GraphVisualizer(fg).save_html(
    h5_html_path,
    title="NanoGPT Forward",
    tensor_stats=fw_stats,
    backward_source=bw,
    bw_tensor_stats=bw_stats,
    embed_h5=h5_path,
)
print(f"  H5-embedded HTML: {h5_html_path}")

# ══════════════════════════════════════════════════════════════════════
# Phase 7: IR JSON (lossless)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 7: IR JSON (lossless)")
print("=" * 70)

from torch_graph.ir_json import save_ir_json, capture_to_ir_json

ir_json_path = os.path.join(OUTPUT_DIR, "nanogpt_ir.json")
save_ir_json(capture, ir_json_path)
print(f"  IR JSON: {ir_json_path}")

# Also save per-graph IR
fw_ir_path = os.path.join(OUTPUT_DIR, "nanogpt_forward_ir.json")
save_ir_json(fg, fw_ir_path)
print(f"  Forward IR JSON: {fw_ir_path}")

if bw:
    bw_ir_path = os.path.join(OUTPUT_DIR, "nanogpt_backward_ir.json")
    save_ir_json(bw, bw_ir_path)
    print(f"  Backward IR JSON: {bw_ir_path}")

# ══════════════════════════════════════════════════════════════════════
# Phase 8: Condensed IR / DAG JSON
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 8: Condensed IR / DAG JSON")
print("=" * 70)

from torch_graph.condense_ir import condense_ir_json

ir_bundle = capture_to_ir_json(capture)

# Standard condensed
condensed = condense_ir_json(ir_bundle)
condensed_path = os.path.join(OUTPUT_DIR, "nanogpt_condensed.json")
with open(condensed_path, "w") as f:
    json.dump(condensed, f, indent=2, default=str)
print(f"  Condensed IR: {condensed_path}")

# Folded condensed (collapsed fused op groups)
condensed_folded = condense_ir_json(ir_bundle, fold=True)
folded_path = os.path.join(OUTPUT_DIR, "nanogpt_condensed_folded.json")
with open(folded_path, "w") as f:
    json.dump(condensed_folded, f, indent=2, default=str)
print(f"  Condensed IR (folded): {folded_path}")

# ══════════════════════════════════════════════════════════════════════
# Phase 9: Backward-only HTML with forward links
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 9: Per-graph HTMLs with tensor stats")
print("=" * 70)

if bw:
    bw_h5_html_path = os.path.join(OUTPUT_DIR, "nanogpt_backward_h5web.html")
    GraphVisualizer(bw).save_html(
        bw_h5_html_path,
        title="NanoGPT Backward",
        tensor_stats=bw_stats,
        embed_h5=h5_path,
    )
    print(f"  Backward H5-embedded HTML: {bw_h5_html_path}")

# Backward JSON
if bw:
    bw_json_path = os.path.join(OUTPUT_DIR, "nanogpt_backward.json")
    with open(bw_json_path, "w") as f:
        json.dump(GraphVisualizer(bw).to_json(), f, indent=2, default=str)
    print(f"  Backward JSON: {bw_json_path}")

# ══════════════════════════════════════════════════════════════════════
# Phase 10: Triton / Inductor kernel capture
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 10: Triton / Inductor kernel capture")
print("=" * 70)

if device.type == "cuda":
    from torch_graph.triton import capture_triton_kernels, save_triton_kernels

    torch._dynamo.reset()
    model_fresh = NanoGPT(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32).to(device)
    idx_fresh = torch.randint(0, 64, (2, 16), device=device)

    try:
        triton_out, fw_tcap, bw_tcap = capture_triton_kernels(
            model_fresh, idx_fresh, run_backward=True
        )
        print(f"\n{fw_tcap.summary()}")

        # Save individual kernel files
        triton_dir = os.path.join(OUTPUT_DIR, "nanogpt_triton_kernels")
        saved_kernels = save_triton_kernels(fw_tcap, triton_dir, prefix="nanogpt_")
        print(f"  Saved {len(saved_kernels)} kernel files to {triton_dir}/")
        for kf in saved_kernels:
            print(f"    {os.path.basename(kf)}")

        if bw_tcap:
            bw_triton_dir = os.path.join(OUTPUT_DIR, "nanogpt_triton_kernels_backward")
            saved_bw = save_triton_kernels(bw_tcap, bw_triton_dir, prefix="nanogpt_bw_")
            print(f"  Saved {len(saved_bw)} backward kernel files to {bw_triton_dir}/")

    except Exception as e:
        print(f"  Triton capture failed: {e}")
        import traceback; traceback.print_exc()

    # Also re-capture aten with triton=True to get enriched export
    print("\n  Re-capturing aten with triton enrichment...")
    torch._dynamo.reset()
    model_enriched = NanoGPT(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32).to(device)
    idx_enriched = torch.randint(0, 64, (2, 16), device=device)
    try:
        _, capture_enriched = capture_aten_graphs(
            model_enriched, idx_enriched,
            run_backward=True,
            record_real_tensors=True,
            triton=True,
        )
        enriched_aten_path = os.path.join(OUTPUT_DIR, "nanogpt_aten_with_triton.py")
        export_aten_program(capture_enriched, enriched_aten_path)
        print(f"  Enriched aten export: {enriched_aten_path}")
    except Exception as e:
        print(f"  Enriched capture failed: {e}")
        import traceback; traceback.print_exc()
else:
    print("  Skipped (no CUDA GPU)")

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" CAPTURE COMPLETE")
print("=" * 70)

output_files = []
for root, dirs, files in os.walk(OUTPUT_DIR):
    for f in files:
        fpath = os.path.join(root, f)
        size = os.path.getsize(fpath)
        rel = os.path.relpath(fpath, OUTPUT_DIR)
        output_files.append((rel, size))

output_files.sort()
for rel, size in output_files:
    if size > 1024 * 1024:
        print(f"  {rel}  ({size / 1024 / 1024:.1f} MB)")
    elif size > 1024:
        print(f"  {rel}  ({size / 1024:.1f} KB)")
    else:
        print(f"  {rel}  ({size} bytes)")

print(f"\nTotal: {len(output_files)} files in {OUTPUT_DIR}/")
