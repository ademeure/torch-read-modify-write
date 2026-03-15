#!/usr/bin/env python3
"""NanoGPT Demo: Extract and manipulate FX graphs from a GPT-2 style transformer.

This demonstrates the full power of torch_graph on a real-world architecture:
- Multi-head self-attention with causal masking
- Layer normalization, GELU activations
- Residual connections
- Full graph extraction, deep inspection, visualization, and editing
"""

import sys
sys.path.insert(0, "/root/torch-graph")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_graph import GraphCapture, GraphInspector, GraphEditor, GraphVisualizer


# ── NanoGPT Model ─────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    """Minimal GPT-2 style language model."""

    def __init__(
        self,
        vocab_size=256,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ═══════════════════════════════════════════════════════════════════════
# Main: Capture, Inspect, Visualize, and Edit NanoGPT graphs
# ═══════════════════════════════════════════════════════════════════════

print("=" * 80)
print("NanoGPT FX Graph Extraction & Manipulation Demo".center(80))
print("=" * 80)

# ── 1. Create model and capture graphs ────────────────────────────────

config = dict(vocab_size=256, block_size=64, n_layer=4, n_head=4, n_embd=128)
model = NanoGPT(**config)
model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"\nModel: NanoGPT ({param_count:,} parameters)")
print(f"Config: {config}")

# Capture graphs
capture = GraphCapture()
compiled_model = torch.compile(model, backend=capture.backend)

# Run with dummy input: batch=2, sequence_length=32
dummy_input = torch.randint(0, 256, (2, 32))
with torch.no_grad():
    logits = compiled_model(dummy_input)

print(f"Output shape: {logits.shape} (batch, seq_len, vocab_size)")
print(f"\n{capture.summary()}")


# ── 2. Deep inspection ───────────────────────────────────────────────

print("\n" + "=" * 80)
print("DEEP GRAPH INSPECTION".center(80))
print("=" * 80)

for i, captured in enumerate(capture):
    inspector = GraphInspector(captured)

    print(f"\n{'─' * 80}")
    print(f"Graph {captured.graph_id}: {captured.num_nodes} nodes, {captured.num_ops} ops")
    print(f"{'─' * 80}")

    # Op frequency table
    print(f"\nOp frequency:")
    for op, count in inspector.op_counts().items():
        bar = "█" * count
        print(f"  {op:<35} {count:>3}  {bar}")

    # Op categories (semantic grouping)
    print(f"\nOp categories:")
    cats = inspector.op_categories()
    for cat, nodes in cats.items():
        print(f"  {cat:<20} {len(nodes):>3} nodes")

    # Show the generated code
    print(f"\nGenerated forward() code:")
    code_lines = captured.code.strip().split("\n")
    for j, line in enumerate(code_lines):
        if j > 60:
            print(f"  ... ({len(code_lines) - 60} more lines)")
            break
        print(f"  {line}")

    # Find attention-related nodes
    print(f"\nAttention-related nodes:")
    attn_nodes = inspector.find_nodes("matmul")
    attn_nodes += inspector.find_nodes("softmax")
    attn_nodes += inspector.find_nodes("bmm")
    attn_nodes += inspector.find_nodes("scaled_dot")
    for n in attn_nodes:
        print(f"  {n.name:<30} {n.op:<16} {n.target}")

    # Find normalization nodes
    print(f"\nNormalization nodes:")
    norm_nodes = inspector.find_nodes("layer_norm")
    norm_nodes += inspector.find_nodes("native_layer_norm")
    for n in norm_nodes:
        print(f"  {n.name:<30} {n.op:<16} {n.target}")

    # Dependency chain for output
    print(f"\nTotal node table (first 40 nodes):")
    table = inspector.print_table()
    table_lines = table.split("\n")
    for line in table_lines[:42]:
        print(f"  {line}")
    if len(table_lines) > 42:
        print(f"  ... ({len(table_lines) - 42} more rows)")


# ── 3. Visualization ─────────────────────────────────────────────────

print("\n" + "=" * 80)
print("GRAPH VISUALIZATION".center(80))
print("=" * 80)

for captured in capture:
    viz = GraphVisualizer(captured)

    # Interactive HTML (the crown jewel)
    html_path = f"/root/torch-graph/outputs/nanogpt_graph_{captured.graph_id}.html"
    viz.save_html(html_path, title=f"NanoGPT FX Graph {captured.graph_id}")
    print(f"  Interactive HTML: {html_path}")

    # JSON export
    json_path = f"/root/torch-graph/outputs/nanogpt_graph_{captured.graph_id}.json"
    viz.save_json(json_path)
    print(f"  JSON export:     {json_path}")

# ── 4. Graph editing: swap activations ────────────────────────────────

print("\n" + "=" * 80)
print("GRAPH EDITING: Replace GELU with SiLU".center(80))
print("=" * 80)

for captured in capture:
    editor = GraphEditor(captured)
    inspector = GraphInspector(captured)

    # Find GELU ops
    gelu_nodes = inspector.find_nodes("gelu")
    print(f"\nFound {len(gelu_nodes)} GELU nodes: {[n.name for n in gelu_nodes]}")

    # Replace each GELU with SiLU (clear kwargs since silu doesn't take 'approximate')
    for node_info in gelu_nodes:
        try:
            editor.replace_op(
                node_info.name,
                torch.ops.aten.silu.default,
                new_kwargs={},
            )
            print(f"  Replaced {node_info.name}: GELU -> SiLU")
        except Exception as e:
            print(f"  Skipped {node_info.name}: {e}")

    # Show the diff
    print(f"\nDiff (first 30 lines):")
    diff_lines = editor.diff().split("\n")
    for line in diff_lines[:30]:
        print(f"  {line}")
    if len(diff_lines) > 30:
        print(f"  ... ({len(diff_lines) - 30} more lines)")

    # Compile and test
    edited_gm = editor.compile()
    with torch.no_grad():
        edited_output = edited_gm(*captured.example_inputs)
        original_output = captured.graph_module(*captured.example_inputs)

    orig_logits = original_output[0] if isinstance(original_output, tuple) else original_output
    edit_logits = edited_output[0] if isinstance(edited_output, tuple) else edited_output

    print(f"\nOriginal logits sample: {orig_logits[0, 0, :5].tolist()}")
    print(f"Edited logits sample:   {edit_logits[0, 0, :5].tolist()}")
    print(f"Max diff: {(orig_logits - edit_logits).abs().max().item():.6f}")

    # Save edited visualization
    viz = GraphVisualizer(edited_gm)
    html_path = f"/root/torch-graph/outputs/nanogpt_graph_{captured.graph_id}_silu.html"
    viz.save_html(html_path, title="NanoGPT - GELU replaced with SiLU")
    print(f"\nEdited graph HTML: {html_path}")


# ── 5. Graph editing: inject logging at attention ─────────────────────

print("\n" + "=" * 80)
print("GRAPH EDITING: Inject ops into the graph".center(80))
print("=" * 80)

for captured in capture:
    editor = GraphEditor(captured)
    inspector = GraphInspector(captured)

    # Find softmax nodes (core of attention)
    softmax_nodes = inspector.find_nodes("softmax")
    print(f"\nSoftmax nodes (attention scores): {[n.name for n in softmax_nodes]}")

    # Insert a scaling op after each softmax (temperature scaling)
    for node_info in softmax_nodes:
        try:
            new_node = editor.insert_after(
                node_info.name,
                torch.mul,
                args_fn=lambda n: (n, 0.95),  # slight temperature scaling
                name=f"temp_scale_{node_info.name}",
            )
            print(f"  Inserted temperature scaling after {node_info.name} -> {new_node.name}")
        except Exception as e:
            print(f"  Failed on {node_info.name}: {e}")

    # Show diff for the insertion
    print(f"\nInsertion diff:")
    diff_lines = editor.diff().split("\n")
    for line in diff_lines:
        if line.startswith("+") or line.startswith("-"):
            print(f"  {line}")

    # Compile and test
    edited_gm = editor.compile()
    with torch.no_grad():
        edited_output = edited_gm(*captured.example_inputs)
        original_output = captured.graph_module(*captured.example_inputs)

    orig_logits = original_output[0] if isinstance(original_output, tuple) else original_output
    edit_logits = edited_output[0] if isinstance(edited_output, tuple) else edited_output

    print(f"\nWith temperature scaling:")
    print(f"  Max logit diff: {(orig_logits - edit_logits).abs().max().item():.6f}")
    print(f"  Mean logit diff: {(orig_logits - edit_logits).abs().mean().item():.6f}")


# ── 6. Graph editing: remove dropout ──────────────────────────────────

print("\n" + "=" * 80)
print("GRAPH EDITING: Remove all dropout ops".center(80))
print("=" * 80)

for captured in capture:
    editor = GraphEditor(captured)
    inspector = GraphInspector(captured)

    dropout_nodes = inspector.find_nodes("dropout")
    print(f"\nDropout nodes: {[n.name for n in dropout_nodes]}")

    removed = 0
    for node_info in dropout_nodes:
        try:
            if editor.remove_node(node_info.name):
                print(f"  Removed {node_info.name}")
                removed += 1
        except Exception as e:
            print(f"  Failed on {node_info.name}: {e}")

    print(f"\nRemoved {removed}/{len(dropout_nodes)} dropout nodes")

    if removed > 0:
        print(f"\nDiff:")
        diff_lines = editor.diff().split("\n")
        for line in diff_lines:
            if line.startswith("+") or line.startswith("-"):
                print(f"  {line}")

        edited_gm = editor.compile()
        with torch.no_grad():
            edited_output = edited_gm(*captured.example_inputs)
        print("  Edited graph runs successfully!")

        viz = GraphVisualizer(edited_gm)
        html_path = f"/root/torch-graph/outputs/nanogpt_graph_{captured.graph_id}_no_dropout.html"
        viz.save_html(html_path, title="NanoGPT - Dropout Removed")
        print(f"  HTML: {html_path}")


# ── Summary ───────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("COMPLETE".center(80))
print("=" * 80)
print(f"""
All outputs saved to /root/torch-graph/outputs/

Files generated:
  - nanogpt_graph_*.html    Interactive graph viewers (open in browser)
  - nanogpt_graph_*.json    JSON graph exports
  - *_silu.html             Graph with GELU->SiLU replacement
  - *_no_dropout.html       Graph with dropout removed

Demonstrated capabilities:
  1. Graph capture via custom TorchDynamo backend
  2. Deep inspection: op counts, categories, shapes, dependencies
  3. Visualization: HTML, JSON
  4. Op replacement: GELU -> SiLU activation swap
  5. Op insertion: temperature scaling after attention softmax
  6. Op removal: stripping dropout from inference graph
""")
