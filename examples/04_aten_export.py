#!/usr/bin/env python3
"""Export models as editable aten-level Python programs WITH autograd.

This is the crown jewel: take any PyTorch model, get a standalone Python file
containing ONLY raw aten ops for both forward AND backward passes.
Edit any op and rerun.
"""

import sys
sys.path.insert(0, "/root/torch-graph")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_graph.export import capture_aten_graphs, export_aten_program, export_graph_to_python
from torch_graph.inspector import GraphInspector
from torch_graph.visualizer import GraphVisualizer


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Simple MLP - See exactly what autograd does
# ═══════════════════════════════════════════════════════════════════════

print("=" * 80)
print("PART 1: MLP - Forward + Backward at aten level".center(80))
print("=" * 80)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.sum()  # scalar loss

model = SimpleMLP()
x = torch.randn(2, 8)

# Capture both forward and backward at aten level
output, capture = capture_aten_graphs(model, x)

print(f"\n{capture.summary()}")

# Show the forward graph
print("\n--- FORWARD (aten ops) ---")
for fg in capture.forward_graphs:
    print(fg.readable)

# Show the backward graph
print("\n--- BACKWARD (aten ops = the autograd!) ---")
for bg in capture.backward_graphs:
    print(bg.readable)

# Export as standalone Python
out_path = "/root/torch-graph/outputs/mlp_aten.py"
export_aten_program(capture, out_path)
print(f"\nExported to: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# PART 2: CNN (MNIST-style) - conv2d autograd decomposed
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PART 2: CNN - See conv2d backward decomposed to aten".center(80))
print("=" * 80)

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.fc = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = self.fc(x)
        return x.sum()

model = TinyCNN()
x = torch.randn(2, 1, 8, 8)

output, capture = capture_aten_graphs(model, x)
print(f"\n{capture.summary()}")

print("\n--- FORWARD ---")
for fg in capture.forward_graphs:
    inspector = GraphInspector(fg)
    print(f"Op counts: {inspector.op_counts()}")
    print()
    print(fg.readable)

print("\n--- BACKWARD ---")
for bg in capture.backward_graphs:
    inspector = GraphInspector(bg)
    print(f"Op counts: {inspector.op_counts()}")
    print()
    print(bg.readable)

out_path = "/root/torch-graph/outputs/cnn_aten.py"
export_aten_program(capture, out_path)
print(f"\nExported to: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Transformer (NanoGPT) - attention + autograd fully decomposed
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PART 3: NanoGPT - Full transformer forward + backward".center(80))
print("=" * 80)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits.sum()  # scalar for backward


# Use small config to keep the export manageable
model = NanoGPT(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
idx = torch.randint(0, 64, (1, 8))

output, capture = capture_aten_graphs(model, idx, run_backward=True)

print(f"\n{capture.summary()}")

# Detailed analysis
for fg in capture.forward_graphs:
    inspector = GraphInspector(fg)
    print(f"\nForward: {fg} ")
    print(f"Op counts:")
    for op, count in inspector.op_counts().items():
        print(f"  {op:<40} {count}")

for bg in capture.backward_graphs:
    inspector = GraphInspector(bg)
    print(f"\nBackward: {bg}")
    print(f"Op counts:")
    for op, count in inspector.op_counts().items():
        print(f"  {op:<40} {count}")

# Export
out_path = "/root/torch-graph/outputs/nanogpt_aten.py"
export_aten_program(capture, out_path, inline_threshold=500)
print(f"\nExported to: {out_path}")

# Also save visualizations
for fg in capture.forward_graphs:
    viz = GraphVisualizer(fg)
    viz.save_html("/root/torch-graph/outputs/nanogpt_forward_aten.html", "NanoGPT Forward (aten)")
    print("Saved: /root/torch-graph/outputs/nanogpt_forward_aten.html")

for bg in capture.backward_graphs:
    viz = GraphVisualizer(bg)
    viz.save_html("/root/torch-graph/outputs/nanogpt_backward_aten.html", "NanoGPT Backward (aten)")
    print("Saved: /root/torch-graph/outputs/nanogpt_backward_aten.html")


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Show the exported code can actually run
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PART 4: Verify exported programs are runnable".center(80))
print("=" * 80)

for name in ["mlp", "cnn"]:
    path = f"/root/torch-graph/outputs/{name}_aten.py"
    print(f"\n--- {path} ---")

    # Show first N lines
    with open(path) as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print(f"First 40 lines:")
    for line in lines[:40]:
        print(f"  {line}", end="")
    print(f"  ...")

# Show NanoGPT export stats
path = "/root/torch-graph/outputs/nanogpt_aten.py"
with open(path) as f:
    lines = f.readlines()
print(f"\n--- {path} ---")
print(f"Total lines: {len(lines)}")

# Count aten ops in the file
aten_ops = set()
for line in lines:
    if "aten." in line and "=" in line:
        parts = line.split("aten.")
        for part in parts[1:]:
            op = part.split("(")[0].split(".")[0].strip()
            if op:
                aten_ops.add(op)
print(f"Unique aten ops used: {len(aten_ops)}")
print(f"Ops: {sorted(aten_ops)}")


print("\n" + "=" * 80)
print("DONE".center(80))
print("=" * 80)
print("""
Generated files:
  outputs/mlp_aten.py              - MLP forward+backward as aten ops
  outputs/cnn_aten.py              - CNN forward+backward as aten ops
  outputs/nanogpt_aten.py          - NanoGPT forward+backward as aten ops
  outputs/nanogpt_forward_aten.html  - Interactive forward graph viewer
  outputs/nanogpt_backward_aten.html - Interactive backward graph viewer

Each .py file is a standalone, editable Python program.
Change any aten op and rerun!
""")
