#!/usr/bin/env python3
"""MNIST Demo: Extract, inspect, visualize, and edit TorchDynamo FX graphs.

This is the simplest possible starting point - a small CNN for MNIST,
compiled with torch.compile, with full graph extraction.
"""

import sys
sys.path.insert(0, "/root/torch-graph")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_graph import GraphCapture, GraphInspector, GraphEditor, GraphVisualizer


# ── 1. Define a simple MNIST model ───────────────────────────────────

class MNISTNet(nn.Module):
    """Small CNN for MNIST digit classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ── 2. Capture the compiled graph ────────────────────────────────────

print("=" * 70)
print("STEP 1: Capture FX Graphs from torch.compile")
print("=" * 70)

model = MNISTNet()
model.eval()

# Create graph capture backend
capture = GraphCapture()

# Compile with our capture backend
compiled_model = torch.compile(model, backend=capture.backend)

# Run with dummy MNIST input (batch=4, channels=1, 28x28)
dummy_input = torch.randn(4, 1, 28, 28)
with torch.no_grad():
    output = compiled_model(dummy_input)

print(f"\nModel output shape: {output.shape}")
print(f"\n{capture.summary()}")


# ── 3. Inspect the graph ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 2: Deep Graph Inspection")
print("=" * 70)

for captured in capture:
    inspector = GraphInspector(captured)

    print(f"\n--- Graph {captured.graph_id} ---")
    print(f"\nGenerated Python code:")
    print(captured.code)

    print(f"\nNode table:")
    print(inspector.print_table())

    print(f"\nOp counts:")
    for op, count in inspector.op_counts().items():
        print(f"  {op}: {count}")

    print(f"\nOp categories:")
    for cat, nodes in inspector.op_categories().items():
        print(f"  {cat}: {nodes}")


# ── 4. Visualize the graph ───────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 3: Graph Visualization")
print("=" * 70)

for captured in capture:
    viz = GraphVisualizer(captured)

    # Save interactive HTML
    html_path = f"/root/torch-graph/outputs/mnist_graph_{captured.graph_id}.html"
    viz.save_html(html_path, title=f"MNIST FX Graph {captured.graph_id}")
    print(f"\nInteractive HTML saved: {html_path}")

    # Save JSON
    json_path = f"/root/torch-graph/outputs/mnist_graph_{captured.graph_id}.json"
    viz.save_json(json_path)
    print(f"JSON export saved: {json_path}")

# ── 5. Edit the graph at the op level ────────────────────────────────

print("\n" + "=" * 70)
print("STEP 4: Edit Graph Ops")
print("=" * 70)

for captured in capture:
    editor = GraphEditor(captured)
    inspector = GraphInspector(captured)

    # Find all relu nodes
    relu_nodes = inspector.find_nodes("relu")
    print(f"\nFound {len(relu_nodes)} relu nodes: {[n.name for n in relu_nodes]}")

    # Replace all relu with gelu
    print("\n>> Replacing all relu activations with gelu...")
    replaced = editor.replace_all_ops(torch.ops.aten.relu.default, torch.ops.aten.gelu.default)
    if not replaced:
        # Try the functional form
        for node_info in relu_nodes:
            try:
                editor.replace_op(node_info.name, torch.ops.aten.gelu.default)
                print(f"   Replaced {node_info.name}")
            except Exception as e:
                print(f"   Skipped {node_info.name}: {e}")

    # Show diff
    print(f"\nGraph diff:")
    print(editor.diff())

    # Compile and validate the edited graph
    print("\n>> Compiling edited graph...")
    edited_gm = editor.compile()

    # Dynamo flattens module params into graph inputs, so we use the
    # captured example_inputs (which include weights, biases, and the tensor)
    print("\n>> Running edited model with captured inputs...")
    with torch.no_grad():
        edited_output = edited_gm(*captured.example_inputs)
        original_output = captured.graph_module(*captured.example_inputs)

    print(f"Original output (first 5): {original_output[0][0, :5].tolist()}")
    print(f"Edited output   (first 5): {edited_output[0][0, :5].tolist()}")
    print(f"Outputs differ: {not torch.allclose(original_output[0], edited_output[0])}")

    # Save the edited graph visualization
    viz_edited = GraphVisualizer(edited_gm)
    html_path = f"/root/torch-graph/outputs/mnist_graph_{captured.graph_id}_edited.html"
    viz_edited.save_html(html_path, title=f"MNIST Edited Graph {captured.graph_id}")
    print(f"\nEdited graph HTML saved: {html_path}")


print("\n" + "=" * 70)
print("DONE! Check /root/torch-graph/outputs/ for all generated files.")
print("=" * 70)
