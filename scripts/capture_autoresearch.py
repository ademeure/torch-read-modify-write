#!/usr/bin/env python3
"""Capture autoresearch aten graphs: forward, backward, optimizer + supported viz formats.

Generates:
  outputs/autoresearch/
    autoresearch_aten.py          - Editable aten forward+backward graph
    autoresearch_forward.html     - Interactive HTML visualization (forward)
    autoresearch_backward.html    - Interactive HTML visualization (backward)
    autoresearch_forward.json     - JSON graph
    autoresearch.h5               - H5 tensor dump (forward + backward)
    autoresearch_scripts/         - Replay scripts (by line and module)
    autoresearch_kbox/            - Kernelbox test scripts

Usage:
    python scripts/capture_autoresearch.py
    python scripts/capture_autoresearch.py --depth=8 --seq-len=128
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_graph.export import capture_aten_graphs, capture_optimizer_aten, export_aten_program
from torch_graph.op_dump import dump_grouped_tensors
from torch_graph.visualizer import GraphVisualizer
from torch_graph import compute_tensor_stats, save_ir_json

# ── Args ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Capture autoresearch aten graphs")
parser.add_argument("--depth", type=int, default=4, help="Model depth (layers)")
parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
parser.add_argument("--output-dir", default="outputs/autoresearch", help="Output directory")
parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Setup autoresearch ───────────────────────────────────────────────

recipes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recipes")
if recipes_dir not in sys.path:
    sys.path.insert(0, recipes_dir)

from autoresearch_wrapper import _build_model, _build_optimizer, _make_token_pool

print("=" * 70)
print(" AUTORESEARCH CAPTURE: building model")
print("=" * 70)

device = args.device
if device == "cuda" and not torch.cuda.is_available():
    print("CUDA not available, falling back to CPU")
    device = "cpu"

model, config, tokenizer = _build_model(
    depth=args.depth,
    seq_len=args.seq_len,
    device=device,
)

print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Config: depth={args.depth}, dim={config.n_embd}, heads={config.n_head}, seq_len={args.seq_len}")
print(f"Device: {device}")

# ── Build sample data ────────────────────────────────────────────────

token_pool = _make_token_pool(tokenizer, args.seq_len)
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

aten_path = os.path.join(OUTPUT_DIR, "autoresearch_aten.py")
export_aten_program(capture, aten_path)
print(f"  Wrote: {aten_path}")

# Verify exported code is valid Python
with open(aten_path) as f:
    code = f.read()
compile(code, aten_path, "exec")
print(f"  Syntax check: OK ({len(code):,} bytes)")

# ══════════════════════════════════════════════════════════════════════
# Phase 3: HTML + supported visualization formats
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 3: Visualizations (HTML, JSON)")
print("=" * 70)

fg = capture.forward_graphs[0]
bw = capture.backward_graphs[0] if capture.backward_graphs else None

fw_stats = (compute_tensor_stats(capture.forward_intermediates)
            if capture.forward_intermediates else None)
bw_stats = (compute_tensor_stats(capture.backward_intermediates)
            if capture.backward_intermediates else None)

viz_fw = GraphVisualizer(fg)

# Forward HTML
fw_html_path = os.path.join(OUTPUT_DIR, "autoresearch_forward.html")
viz_fw.save_html(
    fw_html_path,
    title="AutoResearch Forward",
    tensor_stats=fw_stats,
    backward_source=bw,
    bw_tensor_stats=bw_stats,
)
print(f"  Forward HTML: {fw_html_path}")

# JSON
json_path = os.path.join(OUTPUT_DIR, "autoresearch_forward.json")
with open(json_path, "w") as f:
    json.dump(viz_fw.to_json(), f, indent=2, default=str)
print(f"  Forward JSON: {json_path}")

# Backward HTML + supported formats
if bw:
    viz_bw = GraphVisualizer(bw)
    combined_json_path = os.path.join(OUTPUT_DIR, "autoresearch_combined.json")
    viz_fw.save_json(combined_json_path, backward_source=capture)
    print(f"  Combined JSON: {combined_json_path}")

    bw_html_path = os.path.join(OUTPUT_DIR, "autoresearch_backward.html")
    viz_bw.save_html(
        bw_html_path,
        title="AutoResearch Backward",
        tensor_stats=bw_stats,
    )
    print(f"  Backward HTML: {bw_html_path}")

    json_path = os.path.join(OUTPUT_DIR, "autoresearch_backward.json")
    with open(json_path, "w") as f:
        json.dump(viz_bw.to_json(), f, indent=2, default=str)
    print(f"  Backward JSON: {json_path}")

# ══════════════════════════════════════════════════════════════════════
# Phase 4: H5 tensor dump (forward + backward)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 4: H5 tensor dump")
print("=" * 70)

h5_path = os.path.join(OUTPUT_DIR, "autoresearch.h5")
scripts_dir = os.path.join(OUTPUT_DIR, "autoresearch_scripts")
dump_grouped_tensors(
    capture, h5_path,
    group_by=["line", "module"],
    which="both",
    include_params=True,
    stats=True,
    replay_scripts=True,
    scripts_dir=scripts_dir,
)
print(f"  H5: {h5_path}")
print(f"  Scripts: {scripts_dir}/")

# ══════════════════════════════════════════════════════════════════════
# Phase 5: Kbox generation
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 5: Kbox test scripts")
print("=" * 70)

from torch_graph.kbox_gen import generate_all as generate_kbox_scripts

kbox_dir = os.path.join(OUTPUT_DIR, "autoresearch_kbox")

try:
    scripts = generate_kbox_scripts(h5_path, out_dir=kbox_dir)
    print(f"  Kbox dir: {kbox_dir}/")
    print(f"  Generated {len(scripts)} kbox test scripts")
except Exception as e:
    print(f"  Kbox generation failed: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
# Phase 6: Optimizer capture (adamw_step_fused + muon_step_fused)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 6: Optimizer capture")
print("=" * 70)

optimizer = _build_optimizer(model)

# Prime the optimizer with a training step
model.zero_grad(set_to_none=True)
with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
    loss = model(x, targets=targets)
loss.backward()
optimizer.step()
model.zero_grad(set_to_none=True)

print("  Capturing optimizer.step() via torch.compile...")
torch._dynamo.reset()

try:
    # Fresh forward/backward for gradients
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        loss = model(x, targets=targets)
    loss.backward()

    # Capture optimizer step as aten graph with exact param-slot metadata
    opt_capture = capture_optimizer_aten(
        optimizer,
        record_real_tensors=False,
        param_name_map={id(p): n for n, p in model.named_parameters()},
    )
    capture.optimizer_capture = opt_capture
    full_json_path = os.path.join(OUTPUT_DIR, "autoresearch_full.json")
    viz_fw.save_json(full_json_path, backward_source=capture, optimizer_source=capture)
    ir_json_path = os.path.join(OUTPUT_DIR, "autoresearch_ir.json")
    save_ir_json(capture, ir_json_path)
    print("  Optimizer capture completed")
    print(f"  Full JSON: {full_json_path}")
    print(f"  IR JSON: {ir_json_path}")
except Exception as e:
    print(f"  Optimizer capture skipped: {e}")
    print("  (MuonAdamW may not be fully traceable)")

# ══════════════════════════════════════════════════════════════════════
# Phase 7: Verification tests
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" Phase 7: Verification")
print("=" * 70)

errors = 0

# Check aten file has expected content
with open(aten_path) as f:
    aten_code = f.read()

checks = {
    "flash_attn_3 ops in aten": "torch.ops.flash_attn_3" in aten_code,
    "custom op import (_find_and_load_so)": "_find_and_load_so" in aten_code,
    "runtime op verification": "_required_ops" in aten_code,
    "forward function defined": "def forward(" in aten_code,
    "backward function defined": "def backward(" in aten_code,
    "valid Python syntax": True,  # already checked above
}

for name, ok in checks.items():
    if ok:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}")
        errors += 1

# Check all output files exist
expected_files = [
    "autoresearch_aten.py",
    "autoresearch_forward.html",
    "autoresearch_forward.json",
    "autoresearch.h5",
]
if bw:
    expected_files += [
        "autoresearch_backward.html",
        "autoresearch_backward.json",
    ]
if getattr(capture, "optimizer_capture", None):
    expected_files += ["autoresearch_full.json", "autoresearch_ir.json"]

for fname in expected_files:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        print(f"  ✓ {fname} ({os.path.getsize(fpath):,} bytes)")
    else:
        print(f"  ✗ {fname} MISSING or empty")
        errors += 1

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

if errors:
    print(f"\n⚠ {errors} verification errors!")
    sys.exit(1)
else:
    print("\nAll verifications passed!")
