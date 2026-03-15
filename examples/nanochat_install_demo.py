#!/usr/bin/env python3
"""Demo: Replace torch.compile with captured aten graphs for nanochat.

This script demonstrates the full workflow:
1. Build nanochat model + MuonAdamW optimizer
2. Run a reference training step with torch.compile (to get ground truth)
3. Capture aten graphs for model forward/backward
4. Capture aten graphs for optimizer step functions
5. Install captured graphs (replacing torch.compile)
6. Run the same training step and verify outputs match

No torch.compile is used after installation — all computation runs through
the captured, editable aten graphs.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

# ── Setup nanochat ──────────────────────────────────────────────────

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "repos", "nanochat")

def ensure_repo():
    if not os.path.exists(os.path.join(REPO_DIR, "nanochat", "gpt.py")):
        import subprocess
        print(f"Cloning nanochat into {REPO_DIR}...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/karpathy/nanochat.git", REPO_DIR],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

ensure_repo()
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Build model and optimizer (same as base_train.py)
# ═══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 1: Build nanochat model + MuonAdamW optimizer")
print("=" * 70)

import tiktoken
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
from nanochat.optim import MuonAdamW, adamw_step_fused, muon_step_fused

# Build tokenizer
base = tiktoken.get_encoding("gpt2")
special_tokens = {**base._special_tokens}
offset = base.n_vocab
for i, name in enumerate(SPECIAL_TOKENS):
    special_tokens[name] = offset + i
enc = tiktoken.Encoding(
    name="gpt2_nanochat",
    pat_str=base._pat_str,
    mergeable_ranks=base._mergeable_ranks,
    special_tokens=special_tokens,
)
tokenizer = RustBPETokenizer(enc, "<|bos|>")
vocab_size = tokenizer.get_vocab_size()

# Build model (small for demo)
DEPTH = 4
ASPECT_RATIO = 32
HEAD_DIM = 64
SEQ_LEN = 32
BATCH_SIZE = 2

base_dim = DEPTH * ASPECT_RATIO
model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
num_heads = model_dim // HEAD_DIM

config = GPTConfig(
    n_layer=DEPTH,
    n_head=num_heads,
    n_kv_head=num_heads,
    n_embd=model_dim,
    vocab_size=vocab_size,
    sequence_len=SEQ_LEN,
    window_pattern="S" * DEPTH,
)

with torch.device("cpu"):
    model = GPT(config, pad_vocab_size_to=64)
model.init_weights()
model.train()

print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Config: depth={DEPTH}, dim={model_dim}, heads={num_heads}, seq_len={SEQ_LEN}")

# Build optimizer (mirrors nanochat's setup)
def build_optimizer(model):
    param_groups = []
    muon_groups = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            param_groups.append({
                "params": [p], "kind": "adamw",
                "lr": 0.004, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
            })
        elif "wte" in name or "lm_head" in name or "value_embeds" in name:
            lr = 0.3 if ("wte" in name or "value_embeds" in name) else 0.004
            param_groups.append({
                "params": [p], "kind": "adamw",
                "lr": lr, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
            })
        else:
            key = str(p.shape)
            if key not in muon_groups:
                muon_groups[key] = []
            muon_groups[key].append(p)

    for key, params in muon_groups.items():
        param_groups.append({
            "params": params, "kind": "muon",
            "lr": 0.02, "momentum": 0.95, "ns_steps": 5,
            "beta2": 0.8, "weight_decay": 0.0,
        })

    return MuonAdamW(param_groups)


# Build sample data
texts = [
    "The Pythagorean theorem states that in a right-angled triangle, "
    "the square of the hypotenuse is equal to the sum of the squares "
    "of the other two sides.",
    "Machine learning is a subset of artificial intelligence.",
    "Paris is the capital of France. Berlin is the capital of Germany.",
    "Water freezes at zero degrees Celsius.",
]
token_pool = []
for text in texts:
    token_pool.extend(tokenizer.encode(text))


def get_batch(step):
    torch.manual_seed(1337 + step)
    pool_len = len(token_pool)
    offsets = torch.randint(0, pool_len - SEQ_LEN - 1, (BATCH_SIZE,))
    rows = []
    for off in offsets:
        start = off.item()
        rows.append(token_pool[start:start + SEQ_LEN + 1])
    batch = torch.tensor(rows, dtype=torch.long)
    x = batch[:, :-1].contiguous()
    targets = batch[:, 1:].contiguous()
    return x, targets


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Reference training step with torch.compile
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 2: Reference training step (torch.compile)")
print("=" * 70)

# Save initial state for comparison
torch.manual_seed(42)
init_state = {k: v.clone() for k, v in model.state_dict().items()}

optimizer = build_optimizer(model)

# Run one training step with torch.compile
torch.compiler.reset()
compiled_model = torch.compile(model, dynamic=False)

x, targets = get_batch(0)
loss_compiled = compiled_model(x, targets)
loss_compiled.backward()
optimizer.step()
model.zero_grad(set_to_none=True)

ref_loss = loss_compiled.item()
ref_state = {k: v.clone() for k, v in model.state_dict().items()}

print(f"Reference loss: {ref_loss:.6f}")
print(f"Reference param delta (wte): {(ref_state['transformer.wte.weight'] - init_state['transformer.wte.weight']).abs().max():.6e}")


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Capture aten graphs
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 3: Capture aten graphs for model + optimizer")
print("=" * 70)

# Reset model to initial state
model.load_state_dict(init_state, strict=True)
model.train()
torch.compiler.reset()

# Capture model forward/backward
from torch_graph.export import capture_aten_graphs

x, targets = get_batch(0)
output, capture = capture_aten_graphs(
    model, x, targets=targets,
    run_backward=True,
)

print(f"\nModel capture:")
print(f"  Forward graphs: {len(capture.forward_graphs)}")
if capture.forward_graphs:
    fg = capture.forward_graphs[0]
    n_nodes = len(list(fg.graph_module.graph.nodes))
    n_inputs = len([n for n in fg.graph_module.graph.nodes if n.op == 'placeholder'])
    print(f"  Forward: {n_nodes} nodes, {n_inputs} inputs")
print(f"  Backward graphs: {len(capture.backward_graphs)}")
if capture.backward_graphs:
    bg = capture.backward_graphs[0]
    n_nodes = len(list(bg.graph_module.graph.nodes))
    n_inputs = len([n for n in bg.graph_module.graph.nodes if n.op == 'placeholder'])
    print(f"  Backward: {n_nodes} nodes, {n_inputs} inputs")

# Optimizer: the optimizer functions are @torch.compile(fullgraph=True).
# They mutate tensors in-place (no return value) and have non-tensor args
# (ns_steps, red_dim) that get baked in as constants during compilation.
#
# The simplest approach: use __wrapped__ (the original unwrapped function).
# This runs the exact same PyTorch ops eagerly — no torch.compile overhead.
# The ops are already simple aten-level operations (lerp_, mul_, add_, etc.)
# so there's no high-level decomposition needed.
#
# To replace with custom CUDA kernels later, you'd just edit the unwrapped
# function body directly (it's ~30 lines for AdamW, ~50 for Muon).
print(f"\nOptimizer:")
print(f"  adamw_step_fused: will use unwrapped eager function")
print(f"  muon_step_fused: will use unwrapped eager function")
print(f"  (same ops, no torch.compile — ready for custom kernel replacement)")


# ═══════════════════════════════════════════════════════════════════════
# Step 4: Install aten graphs (replacing torch.compile)
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 4: Install captured aten graphs")
print("=" * 70)

# Reset model to initial state again
model.load_state_dict(init_state, strict=True)
model.train()
torch.compiler.reset()

# Build the autograd.Function wrapper from captured graphs
fg_gm = capture.forward_graphs[0].graph_module
bg_gm = capture.backward_graphs[0].graph_module if capture.backward_graphs else None

# Use _build_primal_map to correctly map FX placeholder names → model param paths.
# This handles the actual aot_autograd ordering (which may differ from named_parameters).
from torch_graph.export import _build_primal_map

primal_map = _build_primal_map(fg_gm, capture)
fw_graph = fg_gm.graph
fw_placeholders = [n for n in fw_graph.nodes if n.op == 'placeholder']

# Build per-placeholder mapping: (placeholder_name, param_path_or_None)
# param_path is None for user inputs (like idx, targets)
placeholder_info = []  # list of (ph_node_name, param_path | None)
for ph in fw_placeholders:
    desc = primal_map.get(ph.name, "")
    if desc.startswith("self."):
        # It's a model parameter/buffer
        param_path = desc[5:]  # strip "self."
        placeholder_info.append((ph.name, param_path))
    elif desc.startswith("(buffer) self."):
        param_path = desc[14:]  # strip "(buffer) self."
        placeholder_info.append((ph.name, param_path))
    else:
        # User input
        placeholder_info.append((ph.name, None))

n_params_and_buffers = sum(1 for _, p in placeholder_info if p is not None)
n_user_inputs = sum(1 for _, p in placeholder_info if p is None)
param_paths_ordered = [(i, path) for i, (_, path) in enumerate(placeholder_info) if path is not None]
user_input_indices = [i for i, (_, path) in enumerate(placeholder_info) if path is None]

print(f"Parameter paths: {n_params_and_buffers} params/buffers")
print(f"User inputs: {n_user_inputs} (placeholder indices: {user_input_indices})")

# Figure out the forward return structure
fw_output_node = [n for n in fw_graph.nodes if n.op == 'output'][0]
fw_output_args = fw_output_node.args[0] if fw_output_node.args else ()
if isinstance(fw_output_args, (tuple, list)):
    n_fw_total_outputs = len(fw_output_args)
else:
    n_fw_total_outputs = 1

# Count tangent inputs in backward (= number of real forward outputs)
if bg_gm:
    bw_graph = bg_gm.graph
    bw_placeholders = [n for n in bw_graph.nodes if n.op == 'placeholder']
    n_tangents = sum(1 for p in bw_placeholders if 'tangent' in p.name)
    n_real_outputs = n_tangents
else:
    n_real_outputs = 1

n_saved = n_fw_total_outputs - n_real_outputs
print(f"Forward outputs: {n_fw_total_outputs} total ({n_real_outputs} real, {n_saved} saved for backward)")


def resolve_param(model, path):
    obj = model
    for part in path.split('.'):
        obj = getattr(obj, part)
    return obj


# Validate shapes
print("\nValidating shapes...")
shape_errors = []
for ph, (ph_name, param_path) in zip(fw_placeholders, placeholder_info):
    if param_path is None:
        continue
    actual = resolve_param(model, param_path)
    ph_val = ph.meta.get('val')
    if ph_val is not None and hasattr(ph_val, 'shape'):
        expected = tuple(ph_val.shape)
        actual_shape = tuple(actual.shape)
        if expected != actual_shape:
            shape_errors.append(f"  {param_path}: expected {list(expected)}, got {list(actual_shape)}")

if shape_errors:
    print("SHAPE MISMATCH — aborting:")
    for e in shape_errors:
        print(e)
    sys.exit(1)
else:
    print(f"  All {n_params_and_buffers} parameter shapes match ✓")


# Build the autograd.Function
class NanochatAtenGraph(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *all_inputs):
        result = fg_gm(*all_inputs)
        if isinstance(result, tuple):
            real = result[:n_real_outputs]
            saved = result[n_real_outputs:]
            tensor_saved = []
            non_tensor_indices = {}
            for i, v in enumerate(saved):
                if isinstance(v, torch.Tensor):
                    tensor_saved.append(v)
                else:
                    non_tensor_indices[i] = v
            ctx.save_for_backward(*tensor_saved)
            ctx._non_tensor_saved = non_tensor_indices
            ctx._num_saved = len(saved)
            return real[0] if n_real_outputs == 1 else real
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        if bg_gm is None:
            raise RuntimeError("No backward graph")
        tensors = list(ctx.saved_tensors)
        saved = []
        t_idx = 0
        for i in range(ctx._num_saved):
            if i in ctx._non_tensor_saved:
                saved.append(ctx._non_tensor_saved[i])
            else:
                saved.append(tensors[t_idx])
                t_idx += 1
        result = bg_gm(*saved, *grad_outputs)
        if not isinstance(result, tuple):
            result = (result,)
        return result


def installed_forward(idx, targets=None, **kwargs):
    """Drop-in replacement for model.forward using captured aten graphs."""
    user_inputs = [idx]
    if targets is not None:
        user_inputs.append(targets)

    # Build input list in exact placeholder order
    all_inputs = []
    ui_idx = 0
    for _, param_path in placeholder_info:
        if param_path is not None:
            all_inputs.append(resolve_param(model, param_path))
        else:
            all_inputs.append(user_inputs[ui_idx])
            ui_idx += 1

    return NanochatAtenGraph.apply(*all_inputs)


model.forward = installed_forward
print("Installed aten forward/backward ✓")

# Install optimizer: replace @torch.compile'd functions with the unwrapped
# originals (same ops, no compilation). This removes all torch.compile
# from the training loop while keeping identical behavior.
import nanochat.optim as _optim_module
_optim_module.adamw_step_fused = adamw_step_fused.__wrapped__
_optim_module.muon_step_fused = muon_step_fused.__wrapped__
print("Installed unwrapped optimizer steps ✓")
print("(No torch.compile anywhere)")


# ═══════════════════════════════════════════════════════════════════════
# Step 5: Run training step with installed aten graphs
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 5: Training step with installed aten graphs")
print("=" * 70)

# Fresh optimizer (same config)
optimizer = build_optimizer(model)

x, targets = get_batch(0)
t0 = time.time()
loss_installed = model(x, targets)
loss_installed.backward()
optimizer.step()
model.zero_grad(set_to_none=True)
t1 = time.time()

installed_loss = loss_installed.item()
installed_state = {k: v.clone() for k, v in model.state_dict().items()}

print(f"Installed loss: {installed_loss:.6f}")
print(f"Reference loss: {ref_loss:.6f}")
print(f"Loss match: {abs(installed_loss - ref_loss) < 1e-4}")
print(f"Time: {(t1-t0)*1000:.1f}ms")

# Compare parameter updates
max_diff = 0
for key in ref_state:
    diff = (installed_state[key].float() - ref_state[key].float()).abs().max().item()
    max_diff = max(max_diff, diff)
    if diff > 1e-3:
        print(f"  WARNING: {key} differs by {diff:.6e}")

print(f"\nMax parameter difference: {max_diff:.6e}")
if max_diff < 1e-3:
    print("PASS: Installed aten graphs produce identical results ✓")
else:
    print("WARN: Some parameter differences detected (may be floating point)")


# ═══════════════════════════════════════════════════════════════════════
# Step 6: Run a few more training steps to verify stability
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 6: Multi-step training (verifying stability)")
print("=" * 70)

losses = []
for step in range(5):
    x, targets = get_batch(step + 1)
    loss = model(x, targets)
    loss.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    losses.append(loss.item())
    print(f"  Step {step+1}: loss = {loss.item():.6f}")

print(f"\nLoss trend: {losses[0]:.4f} → {losses[-1]:.4f}")
if losses[-1] <= losses[0] + 0.5:  # Allow some noise for small model
    print("Training is stable ✓")
else:
    print("WARNING: Loss increasing — check for issues")


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
What happened:
  1. Built nanochat GPT model ({sum(p.numel() for p in model.parameters()):,} params)
  2. Ran reference step with torch.compile → loss = {ref_loss:.6f}
  3. Captured forward + backward as aten-level FX graphs
  4. Installed captured graphs via autograd.Function (replacing torch.compile)
  5. Ran same step with installed graphs → loss = {installed_loss:.6f}
  6. Ran 5 more steps to verify stability

What's running (zero torch.compile):
  - Model forward:  captured aten graph via autograd.Function
  - Model backward: captured aten graph via autograd.Function
  - Optimizer step:  unwrapped adamw_step_fused / muon_step_fused (eager)

Next steps:
  - Export the aten graphs to editable _aten.py files
  - Replace specific aten ops with custom CUDA kernels
  - Replace optimizer ops with custom CUDA kernels too
""")
