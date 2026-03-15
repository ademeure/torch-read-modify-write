#!/usr/bin/env python3
"""Generate H5 tensor dump for LLM.c GPT-2 124M at step 2.

Flags:
  --all          Record all intermediates (default: inputs-only)
  --html         Also generate the graph visualization (slow)
  --bf16         Store bfloat16 as native HDF5 custom float (half size, needs h5wasm support)
"""

import os
import sys
import tempfile
import time

# Root disk is full — redirect all temp/cache to the outputs disk which has space
_TEMP_BASE = os.path.join(os.path.dirname(__file__), "outputs", "_tmp")
os.makedirs(_TEMP_BASE, exist_ok=True)
os.environ["TMPDIR"] = _TEMP_BASE
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(_TEMP_BASE, "torchinductor")
tempfile.tempdir = _TEMP_BASE

sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch_graph.extract import extract_training_step, load_recipe
from torch_graph.op_dump import dump_grouped_tensors

OUTPUT_DIR = "outputs/llmc_gpt2_h5"
PREFIX = "llmc_gpt2"
STEP = 2
RECORD_PATTERN = "_softmax*"  # attention softmax only (excludes _log_softmax = vocab)
INPUTS_ONLY = "--inputs-only" in sys.argv or "--all" not in sys.argv
GEN_HTML = "--html" in sys.argv
USE_BF16_H5 = "--bf16" in sys.argv

t0 = time.time()

print("=" * 70)
_mode_str = "inputs-only" if INPUTS_ONLY else f"selective ({RECORD_PATTERN})"
print(f" LLM.c GPT-2 124M — {_mode_str} H5")
print("=" * 70)

recipe = load_recipe("recipes/llmc_gpt2.py")
model = recipe["model"]
sample_args = recipe["sample_args"]
loss_fn = recipe.get("loss_fn")
get_batch = recipe.get("get_batch")
optimizer = recipe.get("optimizer")

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {n_params:,}")
if INPUTS_ONLY:
    print(f"  Mode: inputs-only (primals + final output, no intermediates)")
else:
    print(f"  Record filter: pattern={RECORD_PATTERN!r}")
print(f"[{time.time()-t0:.1f}s] recipe loaded")

print()
print("=" * 70)
print(f" Phase 1: Extract aten graph at step {STEP}")
print("=" * 70)

_record_filter = {"inputs_only": True} if INPUTS_ONLY else {"pattern": RECORD_PATTERN}
result = extract_training_step(
    model=model,
    sample_args=sample_args,
    loss_fn=loss_fn,
    optimizer=optimizer,
    steps=[STEP],
    get_batch=get_batch,
    output_dir=OUTPUT_DIR,
    prefix=PREFIX,
    storage_dtype=torch.bfloat16,
    device="cuda",
    record_real_tensors=True,
    record_filter=_record_filter,
    capture_optimizer=False,
)

capture = result["capture"]
loss_value = result["loss_value"]
n_fw = len(capture.forward_intermediates) if capture.forward_intermediates else 0
n_bw = len(capture.backward_intermediates) if capture.backward_intermediates else 0
print(f"\n  Loss at step {STEP}: {loss_value}")
print(f"  Recorded intermediates: FW={n_fw} BW={n_bw}")
print(f"[{time.time()-t0:.1f}s] capture done")

print()
print("=" * 70)
print(" Phase 2: Dump tensors to H5")
print("=" * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)
h5_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_step{STEP}.h5")

dump_grouped_tensors(
    capture,
    h5_path,
    group_by=["line", "module"],
    which="forward" if INPUTS_ONLY else "both",
    include_params=True,
    stats=True,
    replay_scripts=True,
    scripts_dir=os.path.join(OUTPUT_DIR, f"{PREFIX}_scripts"),
    inputs_only=INPUTS_ONLY,
    **({"pattern": RECORD_PATTERN} if not INPUTS_ONLY else {}),
)

h5_size_mb = os.path.getsize(h5_path) / (1024 * 1024)
print(f"  H5: {h5_path} ({h5_size_mb:.1f} MiB)")
print(f"[{time.time()-t0:.1f}s] H5 done")

if GEN_HTML:
    print()
    print("=" * 70)
    print(" Phase 3: Generate HTML with embedded H5")
    print("=" * 70)

    from torch_graph.tensor_dump import compute_tensor_stats
    from torch_graph.visualizer import GraphVisualizer

    fw = capture.forward_graphs[0]
    bw = capture.backward_graphs[0] if capture.backward_graphs else None

    fw_stats = compute_tensor_stats(capture.forward_intermediates) if capture.forward_intermediates else None
    bw_stats = compute_tensor_stats(capture.backward_intermediates) if capture.backward_intermediates else None

    html_path = os.path.join(OUTPUT_DIR, f"{PREFIX}_graph_h5.html")

    GraphVisualizer(fw).save_html(
        html_path,
        title=f"LLM.c GPT-2 124M — Step {STEP}",
        tensor_stats=fw_stats,
        source_map=capture.source_map,
        backward_source=bw,
        bw_tensor_stats=bw_stats,
        embed_h5=h5_path,
    )

    html_size_mb = os.path.getsize(html_path) / (1024 * 1024)
    print(f"  HTML: {html_path} ({html_size_mb:.1f} MiB)")

print(f"[{time.time()-t0:.1f}s] done")

print()
print("=" * 70)
print(" Output files:")
print("=" * 70)
for f in result["files"]:
    print(f"  {f}")
print(f"  {h5_path}")
if GEN_HTML:
    print(f"  {os.path.join(OUTPUT_DIR, f'{PREFIX}_graph_h5.html')}")
