#!/usr/bin/env python3
"""Generate nanogpt_graph_h5web.html with embedded H5 tensors."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "test_repo"))

import torch
from torch_graph import capture_aten_graphs, compute_tensor_stats
from torch_graph.op_dump import dump_grouped_tensors
from torch_graph.visualizer import GraphVisualizer

OUTPUT_DIR = "outputs"
PREFIX = "nanogpt"

print("=" * 60)
print(" NanoGPT — Graph + Embedded H5")
print("=" * 60)

from model import NanoGPT

model = NanoGPT()
inputs = (torch.randint(0, 64, (2, 16)),)

print("\nPhase 1: Capture aten graph with real tensors")
out, capture = capture_aten_graphs(
    model, *inputs, run_backward=True, record_real_tensors=True
)
fg = capture.forward_graphs[0]
bw = capture.backward_graphs[0] if capture.backward_graphs else None

print("\nPhase 2: Dump tensors to H5")
os.makedirs(OUTPUT_DIR, exist_ok=True)
h5_path = os.path.join(OUTPUT_DIR, f"{PREFIX}.h5")

dump_grouped_tensors(
    capture,
    h5_path,
    group_by=["line", "module"],
    which="both",
    include_params=True,
    stats=True,
    replay_scripts=True,
    scripts_dir=os.path.join(OUTPUT_DIR, f"{PREFIX}_scripts"),
)

print(f"\nPhase 3: Generate HTML with embedded H5")
fw_stats = (
    compute_tensor_stats(capture.forward_intermediates)
    if capture.forward_intermediates
    else None
)
bw_stats = (
    compute_tensor_stats(capture.backward_intermediates)
    if capture.backward_intermediates
    else None
)

html_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_graph_h5web.html")

GraphVisualizer(fg).save_html(
    html_path,
    title="NanoGPT Forward",
    tensor_stats=fw_stats,
    backward_source=bw,
    bw_tensor_stats=bw_stats,
    embed_h5=h5_path,
)

print(f"  Wrote: {html_path}")
print(f"  H5:    {h5_path}")
print("\nDone.")
