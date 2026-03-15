#!/usr/bin/env python3
"""Capture nanochat aten graphs: forward, backward, optimizer + HTML + H5 + kbox.

Generates:
  outputs/nanochat/
    nanochat_forward_aten.py       - Editable aten forward graph
    nanochat_forward_aten.html     - Interactive HTML visualization (forward)
    nanochat_backward_aten.html    - Interactive HTML visualization (backward)
    nanochat.h5                    - H5 tensor dump (forward + backward)
    nanochat_kbox/                 - Kernelbox test scripts
    (Optimizer capture is attempted but may not succeed for all optimizers)

Usage:
    python scripts/capture_nanochat.py
    python scripts/capture_nanochat.py --depth=8 --seq-len=64
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_graph.export import capture_aten_graphs, export_aten_program
from torch_graph.op_dump import dump_grouped_tensors
from torch_graph.visualizer import GraphVisualizer
from torch_graph import compute_tensor_stats

# ── Args ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Capture nanochat aten graphs")
parser.add_argument("--depth", type=int, default=4, help="Model depth (layers)")
parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
parser.add_argument("--output-dir", default="outputs/nanochat", help="Output directory")
parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Setup nanochat ────────────────────────────────────────────────────

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "repos", "nanochat")

if not os.path.exists(os.path.join(REPO_DIR, "nanochat", "gpt.py")):
    import subprocess
    print(f"Cloning nanochat into {REPO_DIR}...")
    os.makedirs(os.path.dirname(REPO_DIR), exist_ok=True)
    subprocess.check_call(
        ["git", "clone", "--depth", "1",
         "https://github.com/karpathy/nanochat.git", REPO_DIR],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# ── Build model using recipe helpers ─────────────────────────────────

print("=" * 70)
print(" NANOCHAT CAPTURE: building model")
print("=" * 70)

# Add recipes to sys.path
recipes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recipes")
if recipes_dir not in sys.path:
    sys.path.insert(0, recipes_dir)

from nanochat_wrapper import _build_tokenizer, _build_model, _build_optimizer, _make_token_pool
tokenizer = _build_tokenizer()
vocab_size = tokenizer.get_vocab_size()
model, config = _build_model(vocab_size, depth=args.depth, seq_len=args.seq_len)

device = torch.device(args.device)
if device.type == "cuda" and not torch.cuda.is_available():
    print("CUDA not available, falling back to CPU")
    device = torch.device("cpu")

model = model.to(device)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Config: depth={args.depth}, dim={config.n_embd}, heads={config.n_head}, seq_len={args.seq_len}")
print(f"Device: {device}")

# ── Build sample data ────────────────────────────────────────────────

token_pool = _make_token_pool(tokenizer)
batch_size = args.batch_size
seq_len = args.seq_len

torch.manual_seed(1337)
pool_len = len(token_pool)
offsets = torch.randint(0, pool_len - seq_len - 1, (batch_size,))
rows = [token_pool[o.item():o.item() + seq_len + 1] for o in offsets]
batch = torch.tensor(rows, dtype=torch.long, device=device)
x = batch[:, :-1].contiguous()
targets = batch[:, 1:].contiguous()

# ══════════════════════════════════════════════════════════════════════
# Phase 1: Capture forward + backward aten graphs
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 1: Capture forward + backward aten graphs")
print("=" * 70)

torch._dynamo.reset()
out, capture = capture_aten_graphs(
    model, x, targets=targets,
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

aten_path = os.path.join(OUTPUT_DIR, "nanochat_aten.py")
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
fw_html_path = os.path.join(OUTPUT_DIR, "nanochat_forward.html")
viz_fw.save_html(
    fw_html_path,
    title="NanoChat Forward",
    tensor_stats=fw_stats,
    backward_source=bw,
    bw_tensor_stats=bw_stats,
)
print(f"  Forward HTML: {fw_html_path}")

# JSON
import json
json_path = os.path.join(OUTPUT_DIR, "nanochat_forward.json")
with open(json_path, "w") as f:
    json.dump(viz_fw.to_json(), f, indent=2, default=str)
print(f"  Forward JSON: {json_path}")

# Backward HTML (if available)
if bw:
    combined_json_path = os.path.join(OUTPUT_DIR, "nanochat_combined.json")
    viz_fw.save_json(combined_json_path, backward_source=capture)
    print(f"  Combined JSON: {combined_json_path}")

    bw_html_path = os.path.join(OUTPUT_DIR, "nanochat_backward.html")
    GraphVisualizer(bw).save_html(
        bw_html_path,
        title="NanoChat Backward",
        tensor_stats=bw_stats,
    )
    print(f"  Backward HTML: {bw_html_path}")

# ══════════════════════════════════════════════════════════════════════
# Phase 4: H5 tensor dump (forward + backward)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 4: H5 tensor dump")
print("=" * 70)

h5_path = os.path.join(OUTPUT_DIR, "nanochat.h5")
dump_grouped_tensors(
    capture, h5_path,
    group_by=["line", "module"],
    which="both",
    include_params=True,
    stats=True,
    replay_scripts=True,
    scripts_dir=os.path.join(OUTPUT_DIR, "nanochat_scripts"),
)
print(f"  H5: {h5_path}")
print(f"  Scripts: {os.path.join(OUTPUT_DIR, 'nanochat_scripts')}/")

# ══════════════════════════════════════════════════════════════════════
# Phase 5: Kbox generation
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 5: Kbox test scripts")
print("=" * 70)

from torch_graph.kbox_gen import generate_all as generate_kbox_scripts

kbox_dir = os.path.join(OUTPUT_DIR, "nanochat_kbox")

try:
    scripts = generate_kbox_scripts(h5_path, out_dir=kbox_dir)
    print(f"  Kbox dir: {kbox_dir}/")
    print(f"  Generated {len(scripts)} kbox test scripts")
except Exception as e:
    print(f"  Kbox generation failed: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
# Phase 6: Optimizer capture
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 6: Optimizer capture")
print("=" * 70)

optimizer = _build_optimizer(model)

# Do a training step to get the optimizer primed
model.zero_grad(set_to_none=True)
loss = model(x, targets=targets)
loss.backward()
optimizer.step()
model.zero_grad(set_to_none=True)

# Capture the optimizer step as a compiled function
print("  Capturing optimizer.step() via torch.compile...")
torch._dynamo.reset()

try:
    # Capture another forward/backward to have fresh gradients
    loss = model(x, targets=targets)
    loss.backward()

    # Compile and capture the optimizer step
    compiled_step = torch.compile(optimizer.step, backend="aot_eager")
    compiled_step()

    print("  Optimizer capture completed (used compile backend)")
except Exception as e:
    print(f"  Optimizer capture skipped: {e}")
    print("  (MuonAdamW may not be fully traceable via torch.compile)")

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
