#!/usr/bin/env python3
"""One-liner graph extraction for any PyTorch model.

Shows how simple it is to use torch_graph with the convenience API.
"""

import sys
sys.path.insert(0, "/root/torch-graph")

import torch
import torch.nn as nn
from torch_graph import capture_graphs, GraphInspector, GraphVisualizer, GraphEditor


# ── Example 1: Any nn.Module ─────────────────────────────────────────

print("=" * 60)
print("Example 1: Simple function")
print("=" * 60)

def my_fn(x, y):
    z = torch.sin(x) * torch.cos(y)
    return z.sum() + (x ** 2).mean()

result, capture = capture_graphs(my_fn, torch.randn(100), torch.randn(100))
for g in capture:
    print(GraphInspector(g).print_table())
    print()


# ── Example 2: Any nn.Module ─────────────────────────────────────────

print("=" * 60)
print("Example 2: ResNet-style block")
print("=" * 60)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.net(self.norm(x))

model = ResBlock(64)
model.eval()
result, capture = capture_graphs(model, torch.randn(8, 64))

for g in capture:
    inspector = GraphInspector(g)
    print(inspector.print_table())
    print(f"\nOp counts: {inspector.op_counts()}")

    # Quick edit: replace GELU with ReLU
    editor = GraphEditor(g)
    gelu_nodes = inspector.find_nodes("gelu")
    for n in gelu_nodes:
        editor.replace_op(n.name, torch.nn.functional.relu, new_kwargs={})
    print(f"\nReplaced {len(gelu_nodes)} GELU -> ReLU")
    print(f"Diff:\n{editor.diff()}")

    # Save HTML
    GraphVisualizer(g).save_html(
        "/root/torch-graph/outputs/resblock_graph.html",
        title="ResBlock FX Graph"
    )
    print(f"\nSaved: /root/torch-graph/outputs/resblock_graph.html")


# ── Example 3: Exploring the low-level aten ops ─────────────────────

print("\n" + "=" * 60)
print("Example 3: See what torch.compile does to simple operations")
print("=" * 60)

def simple_math(x):
    """What does torch.compile turn this into?"""
    y = x.softmax(dim=-1)
    z = torch.layer_norm(y, [y.shape[-1]])
    return z @ z.T

result, capture = capture_graphs(simple_math, torch.randn(32, 64))
for g in capture:
    print("\nGenerated code:")
    print(g.code)
    print(f"\nOp breakdown: {GraphInspector(g).op_counts()}")


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
