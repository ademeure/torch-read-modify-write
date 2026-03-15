# torch-graph Architecture Analysis

## Lines of Code Breakdown

### Total: ~7,800 lines (core library + examples + recipes)

```
Module                   Lines   %     Role
─────────────────────────────────────────────────────────────────
export.py                2,019   26%   Aten-level capture, export to Python, test harness
viewer_template.html       993   13%   Interactive HTML graph viewer (JS/CSS/HTML)
tensor_dump.py             829   11%   Tensor recording, comparison, verification
visualizer.py              582    7%   Graph visualization (JSON, HTML)
auto.py                    511    7%   Zero-modification script extraction
triton.py                  451    6%   Triton kernel capture + aten→kernel mapping
extract.py                 360    5%   Recipe-based training step extraction
editor.py                  291    4%   FX graph editing with undo
inspector.py               243    3%   Graph inspection and op analysis
capture.py                 145    2%   FX-level capture via custom Dynamo backend
_utils.py                   85    1%   Shared: RecordingInterpreter, is_fake, materialize_tensor
__init__.py                 68    1%   Exports
__main__.py                  4   <1%   CLI entry
─────────────────────────────────────────────────────────────────
Examples                   986   13%   Demo scripts (4 files)
Recipes                    235    3%   Training recipes (3 files)
```

### Where the Complexity Lives

**export.py (2,070 lines)** — by far the most complex module. Breakdown by function:

| Lines | Function | Purpose |
|-------|----------|---------|
| 279 | `export_aten_program()` | Top-level export orchestrator — assembles header, weights, forward/backward/optimizer functions, test harness |
| 168 | `_emit_real_tensor_harness()` | Generates the `if __name__ == "__main__"` verification harness with FX Interpreter re-execution |
| 110 | `extract_subgraph()` | Extract a subset of ops into a standalone function |
| 97 | `save_step_data()` | Serialize tensors to .pt with optional bf16 compression + intermediates capping |
| 80 | `_match_primals_to_params()` | Match aot_autograd primals to model parameters by shape + consumer metadata |
| 57 | `_node_to_python()` | Convert a single FX node to a Python statement |
| 55 | `capture_aten_graphs()` | The core capture function: torch.compile → aot_autograd → fw/bw GraphModules |
| 54 | `trace_tensors()` | Trace intermediate tensors from exported functions |
| 50 | `_format_group_header()` | Format source annotation section headers |
| 42 | `_callable_to_str()` | Convert aten op targets to clean Python strings |
| ~200 | (misc) | Source trace capture, primal mapping, tensor materialization, helpers |

**visualizer.py (1,564 lines)** — appears huge but is mostly the HTML template:

| Lines | Section | Purpose |
|-------|---------|---------|
| 994 | `_HTML_TEMPLATE` | Self-contained HTML/CSS/JS for interactive graph viewer (zoom, pan, search, grouping, info panel, tensor stats) |
| 126 | `_build_graph_data()` | Build JSON graph data with nodes, edges, groups, source info, kernel mapping |
| 121 | `_extract_source_info()` + `_extract_source_group()` | Extract source metadata from FX node `meta` |
| 104 | Preamble | Color schemes, op categories, class init |
| 28 | `to_html()` + `to_json()` | Thin wrappers |

**tensor_dump.py (892 lines)** — tensor verification machinery:

The complexity is in handling all the edge cases of:
- FakeTensor → real tensor materialization
- aot_autograd's parameter flattening order (params + buffers + user inputs)
- Forward output format (mutated_inputs, user_outputs, saved_tensors)
- Building backward inputs from forward outputs
- Dynamic/symbolic shapes with SymInt placeholders

---

## Architecture: How torch.compile / FX / Dynamo / aot_autograd Are Used

### The PyTorch 2.x Compilation Stack

```
User Code (nn.Module)
        │
        ▼
  TorchDynamo          ← Python bytecode tracer (graph break handling)
        │
        ▼
  FX Graph             ← High-level graph: call_function, call_module, etc.
        │
        ▼
  aot_autograd          ← Traces autograd to get forward + backward as FX graphs
        │
        ▼
  Aten IR               ← Decomposed to ~2000 aten ops (the "aten dialect")
        │
        ▼
  Backend (inductor)    ← Generates Triton/C++ code
```

### How torch-graph hooks into this stack

**Level 1: FX-level capture (`capture.py`)**
```python
# Custom TorchDynamo backend — intercepts after Dynamo tracing but BEFORE lowering
@torch.compile(backend=capture.backend)
def fn(x): ...
```
This gives you the high-level FX graph with `call_module` nodes for nn.Modules,
`call_function` nodes for PyTorch ops, and rich metadata (`stack_trace`,
`nn_module_stack`, `source_fn_stack`).

**Level 2: Aten-level capture (`export.py`)**
```python
# Uses aot_module_simplified → gets BOTH forward and backward as aten-op FX graphs
from torch._functorch.aot_autograd import aot_module_simplified
aot_module_simplified(gm, inputs, fw_compiler=fw_compiler, bw_compiler=bw_compiler)
```
The fw_compiler/bw_compiler callbacks receive the decomposed aten-level GraphModules.
This is the **primary capture mechanism** — gives you the complete, editable
forward and backward computation graphs at the aten op level.

**Level 3: Source traces (two-phase capture in `export.py`)**
```
Phase 1: torch.compile with lightweight identity backend
         → Extracts stack_trace metadata (file/line/code) from FX nodes
         → Maps source_fn keys to SourceTrace objects

Phase 2: torch.compile with aot_autograd backend
         → Captures aten-level forward/backward graphs
         → Has nn_module_stack, source_fn_stack (but NO stack_trace)
         → Cross-references with Phase 1 via source_fn keys
```

**Level 4: Inductor debug enrichment (`triton.py`)**
```python
# Runs a second, optional inductor compilation with trace enabled
torch.compile(model, backend="inductor")
# Parses inductor's debug output (output_code.py) to extract kernel definitions
# Attaches kernel metadata back onto the already-captured aten graph
```

### API Surface Summary

```
User-facing capture paths:
  capture_graphs()           → FX-level (Dynamo backend)
  capture_aten_graphs()      → Aten-level (aot_autograd)
  extract_model()            → Aten capture + export to .py/.pt
  extract_from_script()      → Auto-discover models in a script + extract
  extract_training_step()    → Full fw+bw+optimizer capture from recipe
  capture_triton_kernels()   → Backward-compatible Inductor kernel/debug capture
  capture_inductor_debug()   → Explicit Inductor debug enrichment capture

Post-capture tools:
  GraphInspector             → Analyze FX graphs
  GraphEditor                → Edit FX graphs (replace ops, fuse, etc.)
  GraphVisualizer            → Render (HTML, JSON)
  export_aten_program()      → Export as standalone Python + test harness
  dump_and_compare()         → Tensor-level verification
```

---

## Legacy vs Modern PyTorch APIs

### What we use (all modern 2.x path):

| API | Status | Used in |
|-----|--------|---------|
| `torch.compile(backend=...)` | **Modern** (2.0+) | `capture.py`, `export.py`, `triton.py` |
| `torch._functorch.aot_autograd.aot_module_simplified` | **Modern** (2.0+) | `export.py` — the core decomposition mechanism |
| `functorch.compile.make_boxed_func` | **Compat shim** | `export.py` — wraps compiled functions for aot_autograd |
| `torch.fx.GraphModule` / `Graph` / `Node` | **Modern** (1.8+, stable in 2.x) | Everywhere — the graph IR |
| `torch.fx.interpreter.Interpreter` | **Modern** (1.8+) | `tensor_dump.py`, `export.py` — node-by-node execution |
| `torch._inductor.config` | **Modern** (2.0+) | `triton.py` — inductor trace settings |
| `torch._subclasses.fake_tensor` | **Modern** (2.0+) | `tensor_dump.py`, `export.py` — FakeTensor detection |
| `torch.fx.experimental.symbolic_shapes` | **Modern** (2.1+) | `tensor_dump.py`, `export.py` — dynamic shapes |
| `torch._dynamo.backends.registry` | **Modern** (2.0+) | `capture.py` — backend passthrough |

### What we do NOT use (legacy/pre-2.0):

| API | Status | Notes |
|-----|--------|-------|
| `torch.jit.trace` / `torch.jit.script` | **Legacy** | NOT used — the old TorchScript path |
| `torch.fx.symbolic_trace()` | **Semi-legacy** | NOT used — pre-Dynamo symbolic tracing (no dynamic control flow support) |
| `torch.onnx.export()` | **Legacy export** | NOT used |
| `torch.export.export()` | **Modern** (2.1+) | NOT used — could be an alternative to aot_autograd (see below) |
| `torch.fx.Tracer` | **Semi-legacy** | NOT used — low-level tracer, subsumed by Dynamo |

**Verdict: The codebase uses exclusively the modern PyTorch 2.x compilation path.**
No legacy TorchScript, no pre-Dynamo `symbolic_trace`, no ONNX export.

---

## Potential Weaknesses & Design Contradictions

### 1. ~~Double Compilation for Source Traces~~ (FIXED)

**Fixed.** Source traces are now extracted inline from the pre-decomposition FX
graph inside the `aot_backend` callback. The `gm` parameter there has full
`stack_trace`, `source_fn_stack`, and `nn_module_stack` metadata — same data
the separate compile pass was getting. Result: **1.27x speedup** on warm runs,
larger gains on cold start (no redundant Dynamo initialization).

### 2. ~~`functorch.compile.make_boxed_func` — Deprecated Import Path~~ (FIXED)

**Fixed.** All imports now use `from torch._functorch.aot_autograd import make_boxed_func`.

### 3. ~~Duplicated `_RecordInterp` / `_RecordingInterpreter`~~ (FIXED)

**Fixed.** Consolidated into a single `RecordingInterpreter` class in
`torch_graph/_utils.py`. Both `export.py` and `tensor_dump.py` import from there.

### 4. ~~`_materialize_tensor` Duplication~~ (FIXED)

**Fixed.** Single implementation in `_utils.py` as `materialize_tensor()`, used
by both `export.py` and `tensor_dump.py`.

### 5. ~~`_is_fake_tensor` vs `is_fake`~~ (FIXED)

**Fixed.** Single `is_fake()` in `_utils.py`, combining the most thorough checks
from both implementations (official API, type name, meta device, storage, symbolic sizes).

### 6. `torch.export.export()` — The Unused Alternative

PyTorch 2.1+ introduced `torch.export.export()` which provides:
- A single-pass capture to aten IR (no double compilation needed)
- Built-in parameter/buffer metadata (no primal-matching heuristics)
- Guaranteed graph completeness (no graph breaks)
- Source annotations out of the box
- Official, stable API (unlike `aot_module_simplified` which is private)

The current approach uses `aot_module_simplified` which is a private API
(`torch._functorch`). While it works well and gives us the backward graph
(which `torch.export` does not directly provide), it means:
- No stability guarantees across PyTorch versions
- The primal-matching logic (`_match_primals_to_params`, `_map_primals_by_order`)
  is complex and fragile — it exists because aot_autograd flattens parameters
  into an opaque list

For forward-only capture, `torch.export.export()` would be cleaner and more
stable. For backward capture, aot_autograd remains the right choice since
`torch.export` doesn't decompose autograd.

### 7. ~~HTML Template Inline (994 lines of JS/CSS/HTML)~~ (FIXED)

**Fixed.** Extracted to `torch_graph/viewer_template.html` (993 lines), loaded
at runtime with caching. `visualizer.py` went from 1,564 to 582 lines.

### 8. Test Harness Code Generation

`_emit_real_tensor_harness()` (168 lines) generates Python code via `buf.write()`
string concatenation. This is fragile and hard to maintain — any syntax error in
the generated code only shows up at runtime.

### 9. `torch.compiler.reset()` Called Frequently

The code calls `torch.compiler.reset()` before every compilation to clear Dynamo
caches. This is correct for ensuring clean captures, but it's expensive (~0.7s
on first call in a process). When doing multiple captures in sequence (e.g.,
`extract_training_step` which captures forward+backward then optimizer), this
adds up.

---

## Profiling: Why NanoGPT Capture Is Slow

### Measured Timings (4-layer GPT, n_embd=128, CPU)

```
Component                                    Time      Notes
───────────────────────────────────────────────────────────────
torch.compile() (lazy, no trace)            0.76s     First call: initializes Dynamo
First execution (Dynamo trace + backend)    1.00s     Bytecode analysis + FX graph construction
Second execution (guard check)              0.01s     Just guard evaluation
aot_autograd decomposition                  0.91s     FX → aten IR + autograd tracing
Source trace capture (Phase 1)              0.23s     Warm; 4.3s cold
copy.deepcopy(GraphModule)                  0.007s    Negligible
export_graph_to_python                      0.003s    Negligible
───────────────────────────────────────────────────────────────
Total capture_aten_graphs (warm):           1.23s     Phase 1 + Phase 2
Total capture_aten_graphs (cold):           ~5.5s     First run in process
```

### Where Time Goes

1. **TorchDynamo tracing (40-50%)**: Bytecode analysis, guard generation, FX graph
   construction. This is O(model complexity) and unavoidable when using torch.compile.

2. **aot_autograd decomposition (30-40%)**: Decomposes high-level ops to ~2000 aten
   ops, traces the autograd graph. This involves running the model through
   `make_fx` under fake tensor mode.

3. **Source trace phase (5-20%)**: Second torch.compile pass for metadata. Variable
   cost — 0.23s warm, 4.3s cold.

4. **Deep copy of GraphModules (<1%)**: Negligible.

5. **Code generation (<1%)**: String formatting is fast.

### Applied Optimization + Remaining Opportunities

**Applied: Merged source trace capture** — Source traces now extracted from
the pre-decomposition FX graph inside the aot_autograd callback. Measured
**1.27x speedup** on warm NanoGPT captures (1.08s vs 1.36s average).

**Remaining opportunities (ordered by impact):**

1. **Skip `torch.compiler.reset()` when possible** — Track whether a reset is
   actually needed (e.g., same model class = can reuse Dynamo cache). Each reset
   forces full retracing.

2. **Reduce `copy.deepcopy` calls** — Currently deepcopies both the GraphModule
   and all example inputs. The example_inputs copy could use `clone().detach()`
   instead of deep copy (already done for tensors, but the list itself is copied).

3. **For the extract_training_step path**: The optimizer capture does ANOTHER
   `torch.compiler.reset()` + `torch.compile()`. This is a second compilation.
   Total compilations per training step: 2 (aten capture + optimizer).

---

## Dependency Graph

```
capture.py ──────────────────────────────────────────┐
    │ (GraphCapture, CapturedGraph)                   │
    ▼                                                 │
inspector.py (GraphInspector)                         │
editor.py (GraphEditor)                               │
                                                      │
export.py ◄───────────────────────────────────────────┘
    │ (AtenCapture, capture_aten_graphs,              │
    │  export_aten_program, save_step_data)            │
    ▼                                                 │
tensor_dump.py (verify_against_model, etc.)           │
triton.py (capture_triton_kernels)                    │
visualizer.py (GraphVisualizer)                       │
                                                      │
extract.py ◄──── export.py + tensor_dump + visualizer │
    │ (extract_training_step, load_recipe)            │
    ▼                                                 │
auto.py ◄──── export.py                               │
    (extract_from_script, extract_model)
```

The dependency flow is clean — `export.py` is the hub, with `extract.py` and
`auto.py` as higher-level orchestrators. No circular dependencies.

---

## Summary

**Strengths:**
- Uses exclusively modern PyTorch 2.x APIs (torch.compile, Dynamo, aot_autograd)
- No legacy TorchScript or pre-2.0 symbolic_trace anywhere
- Clean separation of concerns (capture → inspect → edit → visualize → export)
- Comprehensive: forward + backward + optimizer + Triton kernels
- The two-phase source trace system is clever (if expensive)

**Fixed weaknesses:**
- ~~Double compilation~~ → merged source trace into aot_autograd callback (1.27x speedup)
- ~~Code duplication~~ → consolidated into `_utils.py`
- ~~Deprecated import~~ → uses `torch._functorch.aot_autograd` consistently
- ~~HTML template inline~~ → extracted to `viewer_template.html`

**Remaining weaknesses:**
- Test harness generated via string concatenation
- Uses private API (`torch._functorch.aot_autograd`) instead of stable `torch.export`
