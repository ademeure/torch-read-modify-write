# Extract -> Modify -> Run: The torch-graph Workflow

## Quick Reference (TL;DR)

```bash
# 1. EXTRACT: Capture aten graphs from any training script
python3 -m torch_graph install train.py --dynamic

# 2. MODIFY: Edit the generated .py files in .torch_graph_cache/
#    Change any aten op, add Triton kernels, fuse operations

# 3. RUN: Re-run the same script — modifications take effect immediately
python3 -m torch_graph install train.py --dynamic
```

That's it. `torch.compile(model)` is transparently replaced. The model runs
using your edited aten ops instead of Inductor. No recompilation needed.

---

## Succinct Version

### Extract

```python
import torch_graph.auto_install as ai
ai.patch()                          # Intercept torch.compile
ai.configure(dynamic=True)         # Enable dynamic shapes

model = MyModel().cuda()
compiled = torch.compile(model)     # Returns a proxy (not real compile)
optimizer = torch.optim.AdamW(model.parameters())

# Step 1: triggers capture
loss = compiled(x, targets)         # Forward aten captured
loss.backward()                     # Backward aten captured
optimizer.step()                    # Optimizer aten captured
# Files saved to .torch_graph_cache/
```

### Modify

Edit any `.py` file in `.torch_graph_cache/`:
- `ModelName_hash_1a_train_aten.py` — forward + backward
- `optimizer_AdamW_hash_aten.py` — optimizer step

Every line is a `torch.ops.aten.*` call. Change any op:

```python
# Original:
fc1_addmm: 'float32[B, 256]' = aten.addmm(fc1_bias, input, fc1_t)

# Modified (scale output by 2):
fc1_addmm: 'float32[B, 256]' = aten.addmm(fc1_bias, input, fc1_t)
fc1_addmm = fc1_addmm * 2.0   # <-- your edit
```

### Run

```python
# Step 2+: uses your modified aten files
loss = compiled(x, targets)     # Runs YOUR forward
loss.backward()                 # Runs YOUR backward
optimizer.step()                # Runs YOUR optimizer
```

### Key Rules

1. **Don't change the return tuple** — adding/removing return values corrupts the output layout (forward returns `mutations + real_outputs + saved_for_backward`).
2. **Don't rename parameters** — the install machinery matches by name.
3. **Edit eval independently** — train and eval have separate aten files (`_train_aten.py` vs `_eval_aten.py`).
4. **Clear cache to re-capture** — `rm -rf .torch_graph_cache`.

---

## In-Depth Version

### Architecture Overview

```
User Code                    torch-graph                         Disk
──────────                   ───────────                         ────
torch.compile(model)    →    _CompiledModelProxy created
compiled(x, targets)    →    Step 1: capture_aten_graphs()  →    .torch_graph_cache/
                             │  aot_autograd traces forward         Model_hash_aten.py
                             │  aot_autograd traces backward        Model_hash_aten.py (backward section)
                             │  export_aten_program() codegen
                             │  install() wraps in autograd.Function
                             └→ Forward runs through aten
compiled(x, targets)    →    Step 2+: load cached .py        ←   .torch_graph_cache/
                             │  User edits detected via SHA-256
                             └→ Forward runs through YOUR aten
```

### 1. How Capture Works

#### torch.compile Interception

When you call `ai.patch()`, torch-graph replaces `torch.compile` with a patched version:

```python
# What happens inside patch():
_real_torch_compile = torch.compile    # Save original
torch.compile = _patched_compile       # Install interceptor
```

`_patched_compile` returns:
- `_CompiledModelProxy` for `nn.Module` arguments
- `_CompiledFnProxy` for standalone `@torch.compile` functions

#### Forward/Backward Capture

On first call, `_CompiledModelProxy.__call__` triggers `_capture_variant()`:

1. **Deep-copies** the model (to avoid disturbing original RNG state)
2. Calls `capture_aten_graphs(model_copy, *args, dynamic=True)`
3. Inside, calls the **real** `torch.compile(model, backend=aot_backend)` where `aot_backend` intercepts at the aten level
4. `aot_autograd` decomposes into aten ops, producing separate forward and backward `GraphModule`s
5. `export_aten_program()` generates the `.py` file with source annotations

#### The Forward Output Layout

This is the most important invariant. The forward function returns a flat tuple:

```
(buffer_mutations..., real_outputs..., saved_for_backward...)
│                     │                 │
│ BatchNorm           │ What model      │ Tensors + SymInts
│ running_mean/var    │ .forward()      │ needed by backward
│ updates             │ returns         │
│                     │                 │
│ Count: num_mutations│ Count: num_real │ Count: varies
```

**If you add/remove values from this tuple, the layout silently shifts** — mutations get treated as outputs, outputs as saved tensors, etc. The system now validates the return count and raises an error if it changes.

#### Primal Ordering

aot_autograd does NOT preserve `named_parameters()` order. User inputs (x, targets) can appear at ANY position among model parameters. The ordering is recorded in `capture.primal_names`:

```python
primal_names = [None, 'fc1.weight', None, 'fc1.bias', 'fc2.weight', 'fc2.bias']
#               ↑ user input          ↑ user input
```

`None` entries = user inputs. String entries = parameter paths.

### 2. How Install Works

After capture, `install()` in `install.py` builds a `torch.autograd.Function`:

```python
class _AtenGraph(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *all_inputs):
        result = aten_forward(*all_inputs)      # Call YOUR .py file
        # Split: mutations | real_outputs | saved
        ctx.save_for_backward(*saved_tensors)
        # Write mutations back to model buffers
        for writer, val in zip(buffer_writers, mut_vals):
            writer(val)
        return real_outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        saved = ctx.saved_tensors
        return aten_backward(*saved, *grad_outputs)  # Call YOUR .py backward
```

Parameters flow **live** — the function reads current parameter values at each call via `_make_live_attr_getter(model, "fc1.weight")`. This means optimizer updates (which modify `model.parameters()` in-place) automatically flow into the next forward call.

### 3. How Optimizer Capture Works

#### Standard Optimizers (AdamW, SGD)

`ai.patch()` also wraps `torch.optim.Optimizer.__init__`. After any optimizer is created, its `step()` method is wrapped:

```python
optimizer.step()  # First call:
  → capture_optimizer_aten(optimizer, step_fn)   # Trace step as aten
  → Detect slot roles (param/grad/state)
  → Save to optimizer_AdamW_hash_aten.py
  → Build OptimizerReplayInfo

optimizer.step()  # Step 2+ (if replay_optimizer=True):
  → _run_optimizer_replay(replay)
  → Assemble live params/grads/state as FX inputs
  → Call captured aten forward
  → copy_() mutations back under torch.no_grad()
```

**Slot info** maps each FX placeholder to its role:
```python
slot_info = [
    {"role": "param",  "group": 0, "index": 0, "name": "fc1.weight"},
    {"role": "grad",   "group": 0, "index": 0},
    {"role": "state",  "group": 0, "index": 0, "state_key": "exp_avg"},
    {"role": "state",  "group": 0, "index": 0, "state_key": "exp_avg_sq"},
    {"role": "state",  "group": 0, "index": 0, "state_key": "step"},
    ...
]
```

**Lazy state enrichment**: AdamW's state (exp_avg, exp_avg_sq, step) is created lazily inside `step()`. During capture, these are "unknown" slots. After step 1 completes, `_enrich_unknown_slots()` re-matches them using `data_ptr()`.

#### Inner Compiled Functions (MuonAdamW)

Some optimizers use `@torch.compile` on helper functions:

```python
class MuonAdamW(Optimizer):
    @torch.compile
    def adamw_step_fused(self, params, grads, exp_avg, exp_avg_sq, step, ...):
        ...

    @torch.compile
    def muon_step_fused(self, params, grads, ...):
        ...

    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self.adamw_step_fused(...)
            else:
                self.muon_step_fused(...)
```

These can't be captured monolithically (the outer `step()` has Python control flow that aot_autograd can't trace). Instead:

1. **Detection**: `_has_inner_compiled_fns(optimizer)` checks for `_CompiledFnProxy` instances
2. **Recording**: During step 1, each inner fn call's arg roles are recorded (`_match_arg_to_optimizer`)
3. **Replay plan**: `_build_inner_replay_plan()` creates an `_InnerFnReplayPlan` with all metadata
4. **Full replay**: On step 2+, `_run_inner_replay(plan)` replaces the entire outer step loop — the original optimizer code is never called again

### 4. How Editing Works

When you edit a cached `.py` file:

1. Auto_install checks `.meta` file for the original SHA-256 hash
2. If hash differs, the file is marked "user modified"
3. User-modified files are ALWAYS loaded (never re-captured, even with `force_recapture`)
4. The `.pyc` bytecode cache is evicted so your edits are re-read

**What you can safely edit:**
- Change values in aten op arguments (scales, biases, etc.)
- Add new ops between existing ones (as long as return tuple stays the same)
- Replace an op with a different one that produces the same shape/dtype
- Add print statements for debugging

**What breaks silently:**
- ~~Adding/removing return values~~ (now caught with a clear error)
- Changing output shapes (downstream ops will crash or produce wrong results)
- Removing a tensor that's used in backward (backward will get wrong saved values)

**What crashes explicitly:**
- Syntax errors (SyntaxError with file/line reference)
- Missing variables (NameError)
- Shape mismatches in ops (RuntimeError from aten)

### 5. Cache Structure

```
.torch_graph_cache/
├── ModelName_abc123_1a_train_aten.py      # Forward + backward (training mode, 1 arg)
├── ModelName_abc123_1a_train_aten.meta    # SHA-256 hash, param mapping, mutation count
├── ModelName_abc123_1a_train_aten.pt      # Example inputs for testing
├── ModelName_abc123_1a_train_aten_gm_fw.pt # Forward GraphModule state
├── ModelName_abc123_0a_eval_aten.py       # Forward only (eval mode)
├── optimizer_AdamW_def456_aten.py         # Optimizer aten
├── optimizer_AdamW_def456_aten.meta       # Slot info
└── adamw_step_fused_ghi789_aten.py        # Inner compiled fn (MuonAdamW)
```

Cache key format: `{ClassName}_{hash}_{variant}`
- Hash: SHA-256 of class name + param shapes + dtypes
- Variant: `{n_args}{letter}_{mode}` e.g. `2a_train`, `1b_eval`

### 6. Dynamic Shapes

With `dynamic=True` (the default), symbolic dimensions use names like `s33`, `s50`:

```python
def forward(
    input_0,              # SymInt: batch size (concrete value: 2)
    input_1,              # SymInt: seq length (concrete value: 8)
    input__s33__s50__32,  # Tensor: float32[s33, s50, 32]
    ...
):
```

At runtime, SymInt values are extracted from user input tensor shapes via `_SymIntSpec(tensor_user_idx, dim_idx)`. One captured graph works for any batch size / sequence length.

### 7. Standalone Training Loops

After step 1 captures everything, you can generate a fully self-contained script:

```python
from torch_graph.standalone import save_standalone_training
save_standalone_training(model, optimizer, cache_dir=".torch_graph_cache")
```

This produces a Python script that:
1. Loads saved model state + optimizer state from pickle
2. Runs N training steps using only the aten `.py` files
3. Requires NO original model code, NO original optimizer code
4. Produces identical losses to live replay

### 8. Programmatic API (Without auto_install)

```python
from torch_graph.export import capture_aten_graphs, export_aten_program
from torch_graph.install import install

# Capture
output, capture = capture_aten_graphs(model, x, run_backward=True)

# Export (optional — for inspection/editing)
export_aten_program(capture, "my_model_aten.py")

# Install (replace model.forward with aten)
install(model, aten_module, param_paths=[...], num_real_outputs=1)

# Now model runs through aten
loss = model(x).sum()
loss.backward()  # Uses aten backward
```

### 9. Verify Mode

```bash
python3 -m torch_graph install train.py --verify=5
```

Runs 5 training steps, records per-step losses, prints a summary table, and exits. Produces `training_summary.json` with loss trajectory. Useful for validating that captured aten matches eager execution.
