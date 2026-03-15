#!/usr/bin/env python3
"""Export all example models as standalone aten programs + HTML visualizations.

Usage:
    python run_export.py                # export all models
    python run_export.py --named        # use source-derived variable names

Outputs go to outputs/:
    outputs/mlp_aten.py              - MLP forward+backward
    outputs/convbn_aten.py           - Conv+BN+ReLU forward+backward
    outputs/mha_aten.py              - MultiheadAttention forward+backward
    outputs/nanogpt_aten.py          - NanoGPT forward+backward
    outputs/mlp_graph.html           - MLP interactive graph viewer
    outputs/nanogpt_graph.html       - NanoGPT interactive graph viewer
"""

import sys
import argparse

sys.path.insert(0, ".")
sys.path.insert(0, "test_repo")

import torch
import torch.nn as nn
from torch_graph import capture_aten_graphs, export_aten_program, compute_tensor_stats
from torch_graph.visualizer import GraphVisualizer

parser = argparse.ArgumentParser()
parser.add_argument("--named", action="store_true", help="Use source-derived variable names")
args = parser.parse_args()


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class ConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


from model import NanoGPT

EXPORTS = [
    ("MLP",        MLP(),        (torch.randn(2, 8),),           "mlp"),
    ("Conv+BN",    ConvBN().eval(), (torch.randn(1, 3, 8, 8),),  "convbn"),
    ("MHA",        nn.MultiheadAttention(32, 4, batch_first=True),
                                  (torch.randn(2, 8, 32),) * 3,  "mha"),
    ("NanoGPT",    NanoGPT(),     (torch.randint(0, 64, (2, 16)),), "nanogpt"),
]


print("=" * 60)
print(" Exporting aten programs")
print("=" * 60)

for name, model, inputs, prefix in EXPORTS:
    print(f"\n── {name}")
    out, capture = capture_aten_graphs(model, *inputs, run_backward=True,
                                       record_real_tensors=True)

    fw_nodes = len(list(capture.forward_graphs[0].graph_module.graph.nodes))
    bw_nodes = len(list(capture.backward_graphs[0].graph_module.graph.nodes)) if capture.backward_graphs else 0
    n_fw_inter = len(capture.forward_intermediates or {})
    n_bw_inter = len(capture.backward_intermediates or {})
    print(f"   Forward: {fw_nodes} nodes, {n_fw_inter} intermediates recorded")
    print(f"   Backward: {bw_nodes} nodes, {n_bw_inter} intermediates recorded")

    path = f"outputs/{prefix}_aten.py"
    export_aten_program(capture, path, named_intermediates=args.named)
    print(f"   Wrote: {path}")

    # HTML visualization with real tensor stats
    for fg in capture.forward_graphs:
        html_path = f"outputs/{prefix}_graph.html"
        stats = compute_tensor_stats(capture.forward_intermediates) if capture.forward_intermediates else None
        GraphVisualizer(fg).save_html(html_path, f"{name} Forward (aten)", tensor_stats=stats)
        print(f"   Wrote: {html_path} (with tensor stats)")


print("\n" + "=" * 60)
print(" Done! Check the outputs/ directory:")
print("=" * 60)
print("""
  Aten programs (editable Python):
    outputs/mlp_aten.py
    outputs/convbn_aten.py
    outputs/mha_aten.py
    outputs/nanogpt_aten.py

  Each .py file has a .pt file with ALL real tensors from the original model.
  No synthetic data — every value is from the actual PyTorch execution.

  Interactive graph viewers (open in browser):
    outputs/mlp_graph.html
    outputs/nanogpt_graph.html

  Verify any exported program (compares every intermediate against original):
    python outputs/mlp_aten.py
""")
