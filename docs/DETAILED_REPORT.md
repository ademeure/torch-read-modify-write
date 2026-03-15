# torch-graph Detailed Technical Report

Generated 2026-03-15 from branch `autoresearch-stress-test-and-improvements`.

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Complete Pipeline: How It Works](#2-complete-pipeline)
3. [Every Public API](#3-every-public-api)
4. [Generated Aten File Anatomy](#4-generated-aten-file-anatomy)
5. [Configuration Reference](#5-configuration-reference)
6. [CLI Reference](#6-cli-reference)
7. [Cache Structure and File Formats](#7-cache-structure)
8. [Optimizer Capture: Monolithic vs Inner-Fn](#8-optimizer-capture)
9. [Dynamic Shapes Internals](#9-dynamic-shapes)
10. [The LLM Agent Workflow](#10-llm-agent-workflow)
11. [Test Suite Reference](#11-test-suite)
12. [Stress Test Results](#12-stress-test-results)
13. [All Known Weaknesses](#13-weaknesses)
14. [Architecture Invariants](#14-invariants)
15. [Codebase Statistics](#15-codebase-stats)

---

## 1. What This Is

torch-graph intercepts `torch.compile`, captures the full forward + backward computation as raw `torch.ops.aten.*` calls, saves them as editable `.py` files, and runs the model through those files instead of Inductor.

The generated files are plain Python. Every line is one aten op. You edit any line — replace `aten.relu` with `aten.gelu`, insert a Triton kernel, fuse two matmuls — and the model immediately runs your edit. No recompilation. Parameters flow live, so optimizer updates work.

**Core value proposition:** You get the full computation graph as editable source code, with the original PyTorch module structure preserved in comments.

---

## 2. Complete Pipeline

### 2.1 Step-by-Step Flow

```
User code                         torch-graph internals                    Disk
─────────                         ─────────────────────                    ────
ai.patch()                    →   torch.compile = _patched_compile
                                  Optimizer.__init__ = _patched_init

torch.compile(model)          →   Returns _CompiledModelProxy(model)
                                  (NOT a real compiled model)

compiled(x, targets)          →   Step 1: _CompiledModelProxy.__call__()
                                  ├─ _variant_key(args, kwargs, training)
                                  │    = (2, frozenset(), True)  # 2 args, train
                                  ├─ _capture_variant()
                                  │    ├─ Deep-copy model (protect RNG state)
                                  │    ├─ capture_aten_graphs(model_copy, *args)
                                  │    │    ├─ REAL torch.compile(model, backend=aot_backend)
                                  │    │    ├─ aot_autograd traces → FX GraphModules
                                  │    │    └─ Records primal_names via data_ptr() matching
                                  │    ├─ export_aten_program(capture, cache_path)
                                  │    │    ├─ Renames primals → param names
                                  │    │    ├─ Adds source annotations
                                  │    │    ├─ Writes forward() + backward()
                                  │    │    └─ Writes .meta with SHA-256 hash
                                  │    └─ _load_variant(cache_path)                    → Model_hash_1a_train_aten.py
                                  │         ├─ _load_aten_module(path)                 → Model_hash_1a_train_aten.meta
                                  │         ├─ _parse_param_paths(aten_mod)            → Model_hash_1a_train_aten.pt
                                  │         └─ build_aten_forward(model, aten_mod)
                                  └─ forward = autograd.Function wrapping aten

compiled(x, targets)          →   Step 2+: _variants[key] returns cached forward
                                  ├─ forward(*args, **kwargs)
                                  │    runs YOUR .py file's forward()
                                  └─ loss.backward() runs YOUR backward()

optimizer.step()              →   Step 1: _auto_capture_step()
                                  ├─ Detect inner compiled fns (MuonAdamW?)
                                  ├─ If standard: capture_optimizer_aten()        → optimizer_AdamW_hash_aten.py
                                  │    └─ Build _OptimizerReplayInfo
                                  └─ If inner fns: record calls, build
                                       _InnerFnReplayPlan

optimizer.step()              →   Step 2+ (if replay_optimizer=True):
                                  ├─ Standard: _run_optimizer_replay()
                                  │    └─ Assemble live params/grads/state → aten → copy_() back
                                  └─ Inner fns: _run_inner_replay()
                                       └─ Step counters, attr restore, stacked params
```

### 2.2 User-Edit Detection

When you edit a cached `.py` file:

1. `_load_aten_module(path)` evicts `.pyc` cache so edits are always re-read
2. `_has_user_modified(path)` compares SHA-256 of file content against `.meta`
3. User-modified files are **always loaded** (never re-captured, even with `force_recapture`)
4. Unmodified files skip aten path for eval mode (eager is 2-5x faster for unedited graphs)

### 2.3 When Things Happen

| Event | What triggers it | What it produces |
|-------|-----------------|-----------------|
| First `compiled(x)` call | `_capture_variant()` | Forward + backward `.py` file |
| First `optimizer.step()` | `_auto_capture_step()` | Optimizer `.py` file |
| Every `compiled(x)` after step 1 | `_variants[key]` lookup | Runs cached forward |
| Every `loss.backward()` after step 1 | autograd.Function | Runs cached backward |
| Every `optimizer.step()` after step 1 | `_run_optimizer_replay()` | Runs cached optimizer (if `replay_optimizer=True`) |

---

## 3. Every Public API

### 3.1 Auto-Install (Patching)

```python
import torch_graph.auto_install as ai

ai.patch()                                    # Intercept torch.compile + Optimizer.__init__
ai.unpatch()                                  # Restore originals
ai.configure(dynamic=True, ...)               # Set config options
ai.get_config()                               # Read current config
ai.status()                                   # Summary of installed replacements
ai.register_optimizer(optimizer, step_fn)     # Register non-standard optimizer
ai.install_from_file(model, "model_aten.py")  # Load + install specific file
ai.install_fn_from_file("fn_aten.py")         # Load function from file
```

### 3.2 Capture & Export

```python
from torch_graph.export import (
    capture_aten_graphs,       # Trace model → AtenCapture (forward + backward FX graphs)
    capture_optimizer_aten,    # Trace optimizer.step() → AtenCapture
    export_aten_program,       # AtenCapture → standalone .py file
    export_graph_to_python,    # Single FX GraphModule → Python function string
    save_step_data,            # Save all tensors to .pt file
    trace_tensors_from_graph,  # Run graph, capture ALL intermediate values
    extract_subgraph,          # Extract subset of ops into standalone function
    list_ops,                  # List all ops with metadata
)
```

**Key function signatures:**

```python
capture_aten_graphs(
    model_or_fn,                    # nn.Module or Callable
    *args,                          # Sample inputs
    run_backward: bool = True,      # Also capture backward
    loss_fn: Callable | None = None,# Custom loss (default: output.sum())
    dynamic: bool = False,          # Symbolic shapes
    record_real_tensors: bool = False,  # Save real tensor values inline
    triton: bool = False,           # Also capture Triton kernels
    use_inductor: bool = False,     # Use Inductor backend
    offload_saved: bool = False,    # Offload activations to CPU
    **kwargs,                       # Forwarded to model
) -> tuple[Any, AtenCapture]

export_aten_program(
    capture: AtenCapture,
    output_path: str | Path,
    *,
    inline_threshold: int = 1000,   # Weights < this many elements: inline
    include_test_harness: bool = True,
    named_intermediates: bool = True,
    skip_pt: bool = False,          # Skip saving .pt file
) -> Path
```

### 3.3 Install (Direct, No Patching)

```python
from torch_graph.install import (
    install,                   # Replace model.forward with aten
    uninstall,                 # Restore original forward
    build_aten_forward,        # Build aten callable without modifying model
    install_optimizer_step,    # Replace @torch.compile'd optimizer step
    capture_and_install,       # One-shot: capture + install
)
```

### 3.4 Extraction

```python
from torch_graph.extract import (
    extract_training_step,     # Capture from real training step (model + optimizer)
    extract_function,          # Capture from arbitrary callable
    load_recipe,               # Load recipe module and call setup()
)
```

### 3.5 Standalone Generation

```python
from torch_graph.standalone import save_standalone_training
# After step 1:
script_path = save_standalone_training(model, optimizer, cache_dir)
# Generates self-contained .py that needs no original model/optimizer
```

### 3.6 Inspection & Verification

```python
from torch_graph import (
    explain, ExplainResult,         # One-liner: capture + inspect + verify
    GraphInspector,                  # Op counts, shapes, categories
    dump_and_compare,                # Run graph 2x, compare intermediates
    verify_against_model,            # Compare aten vs eager output
    dump_model_tensors,              # Dump all intermediates from model
)
```

### 3.7 IR & Visualization

```python
from torch_graph import (
    capture_to_ir_json,       # AtenCapture → structured JSON IR
    graph_to_ir_json,         # Single graph → JSON
    ir_graph_to_python,       # JSON IR → Python code
    save_ir_json,             # Save JSON IR to file
    GraphVisualizer,          # HTML + JSON visualization
)
```

---

## 4. Generated Aten File Anatomy

This is a real generated file for a 3-layer MLP (fc1 → relu → fc2):

```python
"""Auto-generated aten-level PyTorch program.

Parameter mapping:
  self.fc1.weight   [16, 8]
  self.fc1.bias     [16]
  input_0           2                  # ← SymInt (batch size)
  input [s77, 8]    [2, 8]            # ← user input
  self.fc2.weight   [4, 16]
  self.fc2.bias     [4]
"""

import operator
import torch
aten = torch.ops.aten

# ======================================================================
# FORWARD PASS
# ======================================================================

def forward(
    fc1_weight: 'float32[16, 8]',      # ← type annotations show shape/dtype
    fc1_bias: 'float32[16]',
    input_0,                             # ← SymInt placeholder (batch dim)
    input__s77__8: 'float32[s77, 8]',   # ← symbolic shape in annotation
    fc2_weight: 'float32[4, 16]',
    fc2_bias: 'float32[4]',
):
    # ═══════════════ self.fc1 ═══════════════
    fc1_t: 'float32[8, 16]' = aten.t(fc1_weight)
    fc1_addmm: 'float32[s77, 16]' = aten.addmm(fc1_bias, input__s77__8, fc1_t)

    # ═══════════════ self.relu ═══════════════
    relu_relu: 'float32[s77, 16]' = aten.relu(fc1_addmm)          # ← EDIT THIS
    relu_detach: 'float32[s77, 16]' = aten.detach(relu_relu)

    # ═══════════════ self.fc2 ═══════════════
    fc2_t: 'float32[16, 4]' = aten.t(fc2_weight)
    fc2_addmm: 'float32[s77, 4]' = aten.addmm(fc2_bias, relu_relu, fc2_t)

    return (fc2_addmm, input__s77__8, fc1_t, relu_relu, relu_detach, fc2_t, input_0,)
    #       ^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #       real output  saved for backward (DO NOT change this tuple structure)

# ======================================================================
# BACKWARD PASS
# ======================================================================

def backward(
    input_0,                                    # ← SymInt (non-tensor first)
    input__s77__8: 'float32[s77, 8]',          # ← saved tensors
    t: 'float32[8, 16]',
    relu: 'float32[s77, 16]',
    detach: 'float32[s77, 16]',
    t_1: 'float32[16, 4]',
    tangents_1: 'float32[s77, 4]',             # ← grad_output
):
    # ═══════════════ grad of self.fc2 ═══════════════
    grad_fc2_t = aten.t(t_1)
    grad_fc2_mm = aten.mm(tangents_1, grad_fc2_t)          # d_loss/d_relu_out
    grad_fc2_t_1 = aten.t(tangents_1)
    grad_fc2_mm_1 = aten.mm(grad_fc2_t_1, relu)            # d_loss/d_fc2_weight
    grad_fc2_sum = aten.sum.dim_IntList(tangents_1, [0], True)
    grad_fc2_view = aten.view(grad_fc2_sum, [4])            # d_loss/d_fc2_bias

    # ═══════════════ grad of self.relu ═══════════════
    grad_relu_detach = aten.detach(detach)
    grad_relu_threshold_backward = aten.threshold_backward(grad_fc2_mm, grad_relu_detach, 0)

    # ═══════════════ grad of self.fc1 ═══════════════
    grad_fc1_t = aten.t(t)
    grad_fc1_mm = aten.mm(grad_relu_threshold_backward, grad_fc1_t)  # d_loss/d_input
    grad_fc1_t_1 = aten.t(grad_relu_threshold_backward)
    grad_fc1_mm_1 = aten.mm(grad_fc1_t_1, input__s77__8)   # d_loss/d_fc1_weight
    grad_fc1_sum = aten.sum.dim_IntList(grad_relu_threshold_backward, [0], True)
    grad_fc1_view = aten.view(grad_fc1_sum, [16])           # d_loss/d_fc1_bias

    return (grad_fc1_t_3, grad_fc1_view, None, grad_fc1_mm, grad_fc2_t_3, grad_fc2_view,)
    #       fc1.weight    fc1.bias       N/A  input         fc2.weight    fc2.bias
```

### Forward Return Tuple Layout

```
return (buffer_mutations..., real_outputs..., saved_for_backward...)
        │                     │                │
        │ BatchNorm           │ What model     │ Tensors + SymInts
        │ running_mean/var    │ .forward()     │ needed by backward()
        │ updates             │ returned       │
        │                     │                │
        │ Count: num_mutations│ Count: 1       │ Count: varies
```

**Critical invariant:** If you add/remove values from this tuple, the install machinery splits at the wrong position and produces silently wrong results. The system now validates return length and raises `RuntimeError: Did you edit the forward return statement?` on mismatch.

---

## 5. Configuration Reference

All 22 fields of `AutoInstallConfig`:

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `cache_dir` | str | `.torch_graph_cache` | Where aten files are stored/loaded |
| `force_recapture` | bool | False | Always re-capture (still prefers user-modified files) |
| `validate_shapes` | bool | True | Check param shapes at install time |
| `verbose` | bool | True | Print status messages |
| `capture_backward` | bool | True | Capture backward pass (needed for training) |
| `loss_fn` | Callable\|None | None | Custom loss function (default: output.sum()) |
| `num_real_outputs` | int | 1 | Number of real model outputs (vs saved tensors) |
| `dynamic` | bool | True | Symbolic shapes (one graph for any batch size) |
| `record_real_tensors` | bool | False | Save real weight values inline in .py |
| `generate_graph` | bool | False | Generate HTML visualization |
| `dump_h5` | bool | False | Dump H5 tensor files |
| `dump_h5_functions` | bool | False | Also dump H5 for compiled functions |
| `skip_pt` | bool | False | Skip saving .pt tensor files |
| `exit_after_capture` | int | 0 | Exit after N calls (0=never) |
| `capture_batch_size` | int | 0 | Shrink batch dim during capture (0=no shrinking) |
| `use_inductor` | bool | False | Use Inductor backend during capture |
| `offload_saved` | bool | False | Offload activations to CPU |
| `capture_optimizer` | bool | True | Auto-capture optimizer.step() |
| `replay_optimizer` | bool | False | Replay aten on step 2+ (default: capture only) |
| `verify_steps` | int | 0 | Run N steps, print summary, exit |
| `record_steps` | bool | False | Record losses to training_summary.json |
| `save_json_ir` | bool | False | Save JSON IR alongside aten .py |

---

## 6. CLI Reference

```bash
# Primary workflow: run script with aten capture
python3 -m torch_graph install train.py [args...]
python3 -m torch_graph install -m package.module [args...]

# Options:
  --cache-dir DIR      Cache directory (default: .torch_graph_cache)
  --recapture          Force re-capture even if cache exists
  --quiet              Suppress status messages
  --static             Static shapes (default: dynamic)
  --dynamic            Dynamic shapes (default)
  --graph              Generate HTML visualization
  --h5                 Dump H5 tensor files
  --h5-functions       Also dump H5 for compiled functions
  --no-pt              Skip .pt tensor files
  --max-steps N        Exit after N training steps
  --no-capture-optimizer  Skip optimizer capture
  --replay-optimizer   Replay aten optimizer on step 2+
  --verify N           Run N steps, print summary, exit (implies --replay-optimizer)
  --record-steps       Record losses to training_summary.json
  --json-ir            Save JSON IR alongside aten .py
  --capture-batch-size N  Override batch size during capture
  --offload-saved      Offload activations to CPU
  --use-inductor       Use Inductor backend during capture
  --record-tensors     Record real tensor values inline

# Other subcommands:
python3 -m torch_graph script.py          # Auto-extract (no install)
python3 -m torch_graph dump script.py     # Dump tensor data
python3 -m torch_graph kbox file.h5       # Generate kbox test scripts
```

---

## 7. Cache Structure

```
.torch_graph_cache/
├── ModelName_abc123_1a_train_aten.py       # Forward + backward (training, 1 arg pattern a)
├── ModelName_abc123_1a_train_aten.meta     # JSON: SHA-256 hash, param mapping, mutations
├── ModelName_abc123_1a_train_aten.pt       # Tensor data (weights, example inputs)
├── ModelName_abc123_1a_train_aten_gm_fw.pt # Forward GraphModule state_dict
├── ModelName_abc123_0a_eval_aten.py        # Forward only (eval mode)
├── ModelName_abc123_0a_eval_aten.meta
├── optimizer_AdamW_def456_aten.py          # Monolithic optimizer aten
├── optimizer_AdamW_def456_aten.meta        # slot_info, mutated_slot_indices
├── adamw_step_fused_ghi789_aten.py         # Inner compiled fn (MuonAdamW)
├── adamw_step_fused_ghi789_aten.meta
├── muon_step_fused_jkl012_aten.py          # Another inner fn
└── training_summary.json                   # Per-step losses (if record_steps=True)
```

### Cache Key Format

`{ClassName}_{hash}_{variant_suffix}`

- **Hash**: First 12 chars of SHA-256 of `class_name + sorted(param_name:shape:dtype)`
- **Variant suffix**: `{n_args}{letter}_{mode}` where:
  - `n_args`: number of positional args
  - `letter`: disambiguator (a, b, c...) for same n_args with different kwargs
  - `mode`: `train` or `eval`
- Example: `MLP_3d38993c5a1e_2a_train`

### .meta File Format

```json
{
  "file_hash": "sha256:abc123...",
  "model": "MLP",
  "primal_names": ["fc1.weight", "fc1.bias", null, "fc2.weight", "fc2.bias"],
  "num_mutations": 0,
  "num_real_outputs": 1,
  "mutated_buffers": [],
  "dynamic": true,
  "input_shapes": {"arg_0": [2, 8]}
}
```

For optimizer files, `.meta` additionally contains:
```json
{
  "optimizer_class": "AdamW",
  "slot_info": [
    {"role": "param", "group": 0, "index": 0, "name": "fc1.weight"},
    {"role": "grad",  "group": 0, "index": 0},
    {"role": "state", "group": 0, "index": 0, "state_key": "exp_avg"},
    {"role": "state", "group": 0, "index": 0, "state_key": "exp_avg_sq"},
    {"role": "state", "group": 0, "index": 0, "state_key": "step"}
  ],
  "mutated_slot_indices": [0, 2, 3, 4]
}
```

---

## 8. Optimizer Capture

### 8.1 Standard Optimizers (AdamW, SGD)

Captured monolithically: the entire `optimizer.step()` becomes one aten graph.

**Slot info** maps each FX placeholder to its role:
- `param` — a model parameter (`param_groups[g]["params"][i]`)
- `grad` — the gradient of a parameter (`.grad`)
- `state` — optimizer state (`state[param][key]`: exp_avg, exp_avg_sq, step)

**Lazy state enrichment**: AdamW creates state tensors lazily inside `step()`. During capture, these are "unknown" slots. After step 1, `_enrich_unknown_slots()` re-matches them via `data_ptr()`.

**Replay**: On step 2+, `_run_optimizer_replay()`:
1. Assembles FX inputs from live model params, grads, and optimizer state
2. Calls captured aten forward
3. Writes mutated values back via `copy_()` under `torch.no_grad()`

### 8.2 Inner Compiled Functions (MuonAdamW)

Optimizers that use `@torch.compile` on helper functions can't be captured monolithically (Python control flow in the outer `step()` loop). Instead:

1. **Detection**: `_has_inner_compiled_fns(optimizer)` scans for `_CompiledFnProxy` instances
2. **Recording**: During step 1, each inner fn call's arg roles are recorded via `_match_arg_to_optimizer()`
3. **Arg role matching**: Maps each argument to its source:
   - `param` / `grad` / `state` — matched by `data_ptr()` against optimizer state
   - `optimizer_attr` — matched against optimizer instance attributes
   - `stacked_params` / `stacked_grads` — detected by shape+value matching
   - `constant` — non-tensor values
4. **Replay plan**: `_InnerFnReplayPlan` stores the complete call sequence
5. **Full replay**: On step 2+, `_run_inner_replay(plan)` replaces the entire outer loop:
   - Increments step counters
   - Restores per-call optimizer attributes (e.g., muon LR prescaling)
   - Assembles args from live state
   - Calls each inner fn's captured aten graph
   - Copies back stacked params via `torch._foreach_copy_`

---

## 9. Dynamic Shapes

With `dynamic=True` (default), varying dimensions use symbolic names:

```python
def forward(
    input_0,                            # SymInt: batch size (concrete: 2)
    input__s77__8: 'float32[s77, 8]',  # Tensor with symbolic first dim
):
    fc1_addmm: 'float32[s77, 16]' = aten.addmm(...)  # shapes propagate
```

### SymInt Slot Mapping

At install time, `_detect_symint_slots()` maps each SymInt placeholder to a user-input tensor dimension:

```python
symint_map = {
    'input_0': ('input__s77__8', 0),   # input_0 = input.shape[0]
}
```

At runtime, `_SymIntSpec(tensor_user_idx=0, dim_idx=0)` extracts the batch size from the actual user input's shape. One captured graph works for any batch size.

### Multiple SymInts

When multiple SymInts precede the same tensor (e.g., both batch and seq_len are symbolic), each maps to a distinct dimension. The detection uses a `claimed_dims` set to prevent two SymInts from mapping to the same dimension.

---

## 10. The LLM Agent Workflow

### Approach A: Programmatic (Full Control)

```python
from torch_graph.export import capture_aten_graphs, export_aten_program
from torch_graph.install import install
from pathlib import Path
import importlib.util

# 1. EXTRACT
model = MyModel().cuda()
x = torch.randn(16, 256, device="cuda")
output, capture = capture_aten_graphs(model, x, run_backward=True, dynamic=True)
aten_path = Path("model_aten.py")
export_aten_program(capture, aten_path)

# 2. SEND TO LLM
code = aten_path.read_text()
modified = llm_agent.modify(code, "Replace all aten.relu with aten.gelu")
aten_path.write_text(modified)

# 3. INSTALL
spec = importlib.util.spec_from_file_location("aten_mod", str(aten_path))
aten_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aten_mod)

param_paths = [n for n in capture.primal_names if n is not None]
install(model, aten_mod, param_paths=param_paths, num_real_outputs=1)

# 4. RUN
loss = model(x).sum()
loss.backward()
optimizer.step()
```

### Approach B: Auto-Install (File-Based)

```python
import torch_graph.auto_install as ai
ai.patch()
ai.configure(cache_dir="./cache")

model = MyModel().cuda()
compiled = torch.compile(model)

# Step 1: captures to disk
loss = compiled(x).sum(); loss.backward()

# Edit the file (LLM does this)
aten_file = next(Path("./cache").glob("*train_aten.py"))
code = aten_file.read_text()
modified = llm_agent.modify(code, "Fuse the two matmuls")
aten_file.write_text(modified)

# Step 2: auto_install detects edit, loads modified version
loss = compiled(x).sum(); loss.backward()  # runs YOUR code
```

### What to Tell the LLM Agent

> You are editing an aten-level PyTorch computation graph.
>
> **Rules:**
> 1. Each line is a `torch.ops.aten.*` call — the lowest-level PyTorch ops
> 2. The `forward()` return tuple structure is `(buffer_mutations..., real_outputs..., saved_for_backward...)`
> 3. The `backward()` receives `(saved_tensors..., grad_outputs...)` and returns parameter gradients
> 4. **DO NOT** add or remove values from either return tuple — this corrupts the layout
> 5. **DO NOT** change the order of values in the return tuples
> 6. You **CAN** change any op, add new ops between existing ones, or replace ops
> 7. Tensor shapes must be preserved through the graph (inputs/outputs of each op must match)
> 8. Source annotations in comments (e.g., `# self.fc1 (Linear)`) show the original PyTorch module

### Safe LLM Edits

| Edit | Example | Safe? |
|------|---------|-------|
| Replace activation | `aten.relu(x)` → `aten.gelu(x)` | Yes (same shape) |
| Scale output | Add `x = x * 2.0` after any op | Yes |
| Add normalization | Insert `aten.layer_norm(...)` | Yes (if shapes match) |
| Insert print | `print(x.shape, x.mean())` | Yes |
| Replace with Triton kernel | Replace aten sequence with `@triton.jit` | Yes (if interface matches) |
| Add gradient clipping | Insert `aten.clamp(grad, -1, 1)` in backward | Yes |
| Add return value | `return (..., extra_tensor)` | **NO — raises RuntimeError** |
| Remove return value | Remove a saved tensor | **NO — breaks backward** |
| Rename parameter | Change `fc1_weight` to `w1` | **NO — install can't match** |

---

## 11. Test Suite

### 11.1 Test File Reference

| File | Tests | What It Covers | GPU? |
|------|-------|----------------|------|
| `test_auto_install.py` | 49 | Full torch.compile interception: 13+ model architectures, cache, variants, dynamic shapes, optimizer capture/replay, user edits, inner fn detection, corrupt file recovery, return-count validation | Some |
| `test_kbox_gen_h5.py` | 13 | H5 tensor dump, kbox test script generation, replay scripts, group/section scripts | Mixed |
| `test_multi_step.py` | 11 | Multi-step training (SGD, AdamW, BatchNorm), bit-identical gradients, offload_saved, cache reload, optimizer replay | Yes |
| `test_install.py` | 11 | Low-level install(), parameter tracking, export roundtrip, slot naming, extract_function, extract_training_step | No |
| `test_ir_json.py` | 8 | JSON IR generation, code matching, structured args, multi-output, tensor literals, source mapping, memory format kwargs, cross-graph links | No |
| `test_autoresearch_e2e.py` | 6 | Full autoresearch + MuonAdamW, 20 steps vs eager, verify mode, inner fn groups, full replay, live params | H100 |
| `test_h5_roundtrip.py` | 5 | fp16/bf16 bit-pattern preservation, H5 read/write cycle, blosc compression | No |
| `test_inductor_comparison.py` | 5 | Aten output matches torch.compile(inductor) for MLP, Conv, BatchNorm, multi-step, NanoGPT | Yes |
| `test_standalone.py` | 5 | Standalone training script generation, execution, dynamic shapes, MuonAdamW inner fn replay | Mixed |
| `test_explain.py` | 4 | explain() one-liner: basic, verify, summary format, verbose control | No |
| `test_named_intermediates.py` | 4 | Named intermediate variable generation, module path shortening, deduplication | No |
| `test_visualizer_formats.py` | 4 | HTML/JSON-only format support, backward consumer links, grad targets, optimizer links | No |
| `test_nanochat_e2e.py` | 3 | Full nanochat + MuonAdamW, aten vs eager, inner fn detection, live params | H100 |
| `test_triton_capture.py` | 3 | Triton kernel detection in graph, export roundtrip with @triton.jit, full auto_install roundtrip | Yes |
| `test_workarounds.py` | 3 | CUDA unsigned-int bitwise op workarounds (uint16/32/64) | Yes |
| `test_install_cli.py` | 1 | CLI subprocess: `python -m torch_graph install` on nanochat | H100 |
| **Total** | **135** | | |

### 11.2 Other Test Suites

| File | Tests | What It Covers |
|------|-------|----------------|
| `run_tests.py` | 5 sections | Determinism verification, E2E accuracy, dynamic shapes, op dump, summary table |
| `test_models.py` | 80 models | Architecture coverage: MLP → ResNet → ViT → HF BERT/GPT2 (syntax check) |
| `test_fixes.py` | 52 tests | Legacy integration tests for historical fixes |

### 11.3 Running Tests

```bash
make test-quick                    # pytest (skip nanochat/autoresearch) + run_tests.py --quick  ~50s
make test                          # Full pytest + run_tests.py  ~65s
make test-models                   # 80 model recipes  ~3min

# Single test:
python3 -m pytest tests/test_auto_install.py::test_single_linear -v

# By keyword:
python3 -m pytest tests/ -k "optimizer" -v

# With Triton (needs LIBRARY_PATH):
export LIBRARY_PATH="/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH"
python3 -m pytest tests/ -v
```

---

## 12. Stress Test Results

35 stress tests across 5 categories. Run on NVIDIA B200, PyTorch 2.10+cu130.

### 12.1 Extract-Modify-Run Workflow (3/3 passed)

| Test | Result | Detail |
|------|--------|--------|
| Capture MLP, verify aten reproduces | PASS | Bit-identical output |
| Inject `* 3.0` in forward, verify 3x loss | PASS | Ratio = 3.00x exact |
| Modify optimizer aten, verify param changes | PASS | Loss diverged as expected |

### 12.2 Standalone & Replay (11/11 passed)

| Test | Result | Detail |
|------|--------|--------|
| MLP+AdamW 10-step standalone | PASS | Max diff 4.68e-07 vs live |
| BatchNorm standalone | PASS | Buffer mutations correct |
| Dynamic shapes standalone | PASS | Max diff 4.68e-07 |
| Standalone as subprocess | PASS | Losses decrease properly |
| SGD replay 20 steps | PASS | **Bit-identical** (zero drift) |
| AdamW replay 20 steps | PASS | Max diff 1.22e-05 |
| AdamW with weight_decay, custom LR | PASS | Max diff 1.23e-05 |
| Dynamic shapes + optimizer replay | PASS | Batch 16→32 works |
| Custom @torch.compile optimizer | PASS | Detected and replayed |
| LBFGS (closure-based) | PASS | Graceful fallback to eager |
| Corruption detection | PASS | 10% param scale → 0.045 loss diff |

### 12.3 Dynamic Shapes (4/4 passed)

| Test | Result | Detail |
|------|--------|--------|
| 3 symbolic dims (B, T, F) | PASS | Verified at 5 different sizes |
| Capture at batch=4, run at batch=1024 | PASS | No issues |
| Batch=1 edge case | PASS | Verified at batch 2/4/16/64 |
| Attention Q*K^T with varying B and T | PASS | 5 different B,T combos |

### 12.4 Cache & Corruption (4/5 passed)

| Test | Result | Detail |
|------|--------|--------|
| Delete .meta file | PASS | Re-captures gracefully |
| Corrupt .meta JSON | PASS | Re-captures gracefully |
| Truncate aten .py file | PASS (after fix) | Was crashing, now auto-recovers |
| Double capture (force_recapture) | PASS | Bit-identical outputs |
| Same class name, different architectures | PASS | Distinct cache entries |

### 12.5 Multi-Fragment (1/3 passed)

| Test | Result | Detail |
|------|--------|--------|
| Explicit graph_break() | **FAIL** | Export OK, auto-install can't load multi-fragment param mapping |
| if/else branching | PASS | Python bool traced correctly |
| Data-dependent control flow | **FAIL** | Same multi-fragment issue |

### 12.6 Memory (3/3 passed)

| Test | Result | Detail |
|------|--------|--------|
| 50-layer deep model (202 params) | PASS | 264.8 KB aten file |
| Large tensors (64MB model) | PASS | 257 MB peak forward+backward |
| offload_saved mode | PASS | Bit-identical output (but +22% memory for small models) |

### 12.7 Concurrency (2/2 passed)

| Test | Result | Detail |
|------|--------|--------|
| Two different models, same process | PASS | Including interleaved forward+backward |
| Same model compiled twice, different args | PASS | Dynamic shapes handled correctly |

### 12.8 Edge Cases (3/3 fixed/caught)

| Test | Result | Detail |
|------|--------|--------|
| Extra return values in edited file | **Now caught** | Raises RuntimeError with clear message |
| Syntax error in edited file | Caught | SyntaxError with file/line reference |
| Deleted lines from aten file | Caught | IndexError (could be improved) |

---

## 13. All Known Weaknesses

### Critical

1. **Multi-fragment models can't be auto-installed** — Models with graph breaks export correctly but `_load_variant()` fails with `ValueError: Cannot determine parameter mapping`. Affects HF GPT2, models with `graph_break()`, data-dependent control flow. (Known limitation, not yet fixed.)

### High

2. **Backward saved tensor ordering is implicit** — Users can't safely reorder the return tuple's saved-for-backward section. No validation beyond total count.

3. **`num_mutations` miscounting silently corrupts BatchNorm** — Wrong count shifts the forward output split. Tested for 0 and 2 mutations, no test for 3+.

4. **SymInt concretization is irreversible** — `int()` on any SymInt concretizes all related symbolic dims. Code generation order must be: forward/backward code first, then weight-building loop.

### Medium

5. **`offload_saved` increases memory on small models** — CPU↔GPU overhead exceeds savings for small activations. Only beneficial for large models.

6. **FA3 requires Hopper specifically** — Fails on Blackwell/B200 with unhelpful CUDA errors.

7. **Device fixup regex is fragile** — The regex adding `.to(_device)` in multi-fragment export must not match aten ops in source annotation comments. No isolated regression test.

### Low

8. **Truncated user-modified files give unhelpful errors** — Corrupt unmodified caches auto-recover, but user-modified corrupt files produce `IndexError: tuple index out of range`.

9. **Custom ops tracking has zero test coverage** — `custom_ops.py` works in practice but has no pytest.

10. **LBFGS can't be replayed** — Closure-based optimizers produce "no graphs". System handles gracefully (runs eager).

11. **FX-level APIs untested** — `capture.py`, `inspector.py`, `editor.py` exported in `__init__.py` with zero test coverage.

12. **No distributed training support** — DDP/FSDP models untested.

---

## 14. Architecture Invariants

These are the invariants that, if violated, cause silent corruption:

1. **Forward output tuple: `(mutations, real_outputs, saved_for_backward)`** — `num_mutations` must be exact. Return count is now validated.

2. **Backward input ordering: non-tensor values first, then tensors** — Differs from forward's interleaved order. Reordering logic in `_AtenGraph.backward()`.

3. **Primal ordering: data_ptr() matching, NOT named_parameters() order** — User inputs can appear at ANY position among params. `capture.primal_names` is the source of truth.

4. **Multi-fragment backward ordering is REVERSED** — `backward_0` corresponds to `forward_(n-1)`.

5. **SymInt code generation before weight materialization** — Or all symbolic dims get concretized.

6. **Parameters flow live** — `_make_live_attr_getter(model, "fc1.weight")` reads current values at each call. Optimizer updates automatically flow through.

7. **Cache hash includes param shapes+dtypes** — Same class name with different architecture gets distinct cache entries.

8. **User-modified files are never re-captured** — SHA-256 hash in `.meta` detects edits. Edits are permanent until user deletes the file.

---

## 15. Codebase Statistics

| Metric | Value |
|--------|-------|
| Python files in torch_graph/ | 24 |
| Total lines of code (torch_graph/) | ~17,100 |
| auto_install.py | 2,679 lines |
| export.py | 3,441 lines |
| install.py | 643 lines |
| _utils.py | 301 lines |
| standalone.py | ~800 lines |
| visualizer.py | ~1,000 lines |
| tensor_dump.py | ~800 lines |
| Public API exports | 33 symbols |
| Test files | 17 |
| Total pytest tests | 135 |
| Total test_models.py recipes | 80 |
| Total test_fixes.py tests | 52 |
| Config options | 22 |
| CLI flags | 18 |
