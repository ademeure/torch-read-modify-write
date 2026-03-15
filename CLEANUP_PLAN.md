# torch-graph Code Quality Cleanup Plan

Comprehensive analysis of hacky, low-quality, unmaintainable, and incorrectly documented
code across the entire codebase, with proposed fixes organized by priority.

---

## Phase 1: Bugs & Correctness Issues (High Priority)

### 1.1 ~~`trace_tensors()` is fundamentally broken~~ — FIXED
Deleted the unfinished `trace_tensors()` function from `export.py`. It had zero callers
and was not exported. `trace_tensors_from_graph()` provides the correct implementation.

### 1.2 ~~`_infer_example_inputs` logic bug~~ — FIXED
Moved `inputs.append(torch.randn(1, in_features))` inside the `Linear` branch before
the `break`, matching the Conv2d pattern.

### 1.3 ~~`kbox_gen.py` hard-codes `.cuda()`~~ — FIXED
Generated scripts now detect device via `torch.cuda.is_available()` and use `.to(_device)`
instead of unconditional `.cuda()`.

### 1.4 Substring matching in `visualizer.py:_semantic_op_color` — DOWNGRADED to Phase 7
Not a correctness bug. The claim that `"mm"` matches `"sum"` is factually wrong
(`"mm" in "sum"` is False). Cross-category matches like `"cat" in "scatter"` and
`"neg" in "negate"` produce correct colors because both ops are in the same category.
This is a fragility/design concern, not a bug — moved to Phase 7.

### 1.5 ~~Duplicate `moe_recipe` in `test_models.py`~~ — FIXED
Renamed second definition to `moe_topk_recipe` and updated its WAVE5_RECIPES key
to `"moe_topk"`. Both recipes now run.

### 1.6 ~~`sys.path`/`cwd` restoration not in `finally`~~ — FIXED
Moved `sys.path[:] = old_path` and `os.chdir(old_cwd)` into the `finally` block
in `extract_from_script`.

### 1.7 ~~`sys.path` leak in `extract_from_module`~~ — FIXED
Added `try/finally` around `spec.loader.exec_module(mod)` to restore `sys.path`.

### 1.8 ~~`triton.py:kernel_by_name` substring matching~~ — FIXED
Changed to exact match only (`k.name == name`). Removed `name in k.name` fallback.

---

## Phase 2: Dead Code Removal

### 2.1 ~~Unused fields in `capture.py`~~ — FIXED
Removed `source_info`, `capture_time`, and `import time`.

### 2.2 ~~Unused imports across codebase~~ — FIXED
Removed: `textwrap`/`time` from export.py (kept `os` — actually used),
`_is_fake_tensor` alias from export.py, `copy`/`sys`/`Sequence` from extract.py,
`torch.fx as fx`/`Any` from editor.py, `copy`/`threading`/`field` from auto.py.

### 2.3 ~~Unused parameters in `export.py:export_graph_to_python`~~ — FIXED
Removed `example_inputs` and `include_weights` params; updated 3 call sites.

### 2.4 ~~Unused functions~~ — FIXED (2 of 3)
Removed `_capture_source_traces` and `NodeInfo.is_op`. Kept
`triton.py:_find_and_parse_debug_artifacts` — it IS called (false positive).

### 2.5 ~~Dead code blocks~~ — FIXED
Removed all: placeholder guard, `group_set`, `external_inputs` param,
pass-loop, shadowed `expected_pairs`, identical torch.bool branches.

### 2.6 ~~Abandoned implementation artifacts~~ — FIXED
Deleted with `trace_tensors()` in Phase 1.

---

## Phase 3: DRY Violations (Extract Shared Utilities)

### 3.1 Target name resolution — NOT EXTRACTED (too small, context-dependent)
6 occurrences of 3-5 line pattern. Each is simple and readable inline.

### 3.2 ~~`L['self']` cleanup — 14 occurrences across 3 files~~ — FIXED
Extracted `clean_self_path(path, keep_self=True)` to `_utils.py`. All 14 call
sites in export.py, op_dump.py, and visualizer.py now use the shared utility.

### 3.3 `_find_node` — NOT EXTRACTED (intentionally different error messages)
Editor version lists available nodes; inspector version doesn't. Keep separate.

### 3.4 `_VIEW_OPS` — NOT EXTRACTED (different semantics per module)
Each module defines its own set for different purposes (recording filter vs.
view detection vs. triton propagation). Not true duplication.

### 3.5 Compute-op predicate — NOT EXTRACTED (trivial one-liner)
Inline `node.op in (...)` is self-documenting. Marginal benefit.

### 3.6 fw_compiler/bw_compiler closures — NOT EXTRACTED (context-dependent)
Each compiler closure has unique logic interspersed with the common pattern.
Extracting would require heavy parameterization that obscures intent.

### 3.7 ~~Primal name sanitization — duplicated in export.py~~ — FIXED
Extracted `_sanitize_primal_names()` function; both call sites now use it.

### 3.8 ~~`_short_name` duplicated in op_dump.py and kbox_gen.py~~ — FIXED
Moved to `_utils.py` as `short_name()` with pre-compiled regexes. Both modules
now import from _utils.

### 3.9 Loss derivation — NOT EXTRACTED (different semantics)
First block returns loss tensor for .backward(); second returns scalar via
.item(). Same file, different purposes.

### 3.10 `nn_module_stack` — NOT EXTRACTED (subtle variations)
Some want first item, some want last, some extract only path. Marginal benefit.

### 3.11 Parameter count — NOT EXTRACTED (trivial one-liner)
`sum(p.numel() for p in model.parameters())` is self-documenting inline.

---

## Phase 4: Silent Error Swallowing

There are 28+ instances of `except Exception: pass` across the codebase. These make
debugging extremely difficult. Proposed fix for each category:

### 4.1 ~~Critical~~ — FIXED
- ~~`export.py` `_capture_source_traces`~~ — deleted in Phase 1
- ~~`op_dump.py` interpreter failure~~ — now logs `logging.warning`
- ~~`op_dump.py` recipe detection~~ — now logs `logging.debug` on failure
- ~~`kbox_gen.py` `.cuda()` failure~~ — fixed device detection in Phase 1

### 4.2 ~~Medium~~ — PARTIALLY FIXED
- `_utils.py:is_fake()` — left alone (intentional defensive fallback)
- `_utils.py:run_node()` — left alone (intentional clone→materialize fallback)
- `export.py:FakeTensorMode` — left alone (safe version-compat fallback)
- ~~`tensor_dump.py` symint~~ — now logs `logging.debug` with fallback info
- ~~`visualizer.py` HDF5 outer~~ — now logs `logging.warning` on failure
- `visualizer.py` HDF5 inner — left alone (skip corrupted items is correct)

### 4.3 ~~Low~~ — FIXED
- ~~`export.py` weight collection~~ — narrowed to `RuntimeError`, added warning
- ~~`export.py` symint fallback~~ — narrowed to `(ImportError, TypeError, ValueError)`
- ~~`op_dump.py` detach/cpu~~ — narrowed to `RuntimeError`

---

## Phase 5: Misleading Documentation

### 5.1 ~~Wrong docstrings~~ — FIXED
- ~~`_short_dtype`~~ — fixed 'f32' → 'fp32' in docstring
- ~~`export.py` `_node_gm`~~ — fixed comment to "node metadata"
- ~~`auto.py` interceptor~~ — fixed comment to "unseen module class"

### 5.2 ~~Incomplete docstrings~~ — FIXED
- ~~`export_graph_to_python`~~ — documented all 10 parameters
- ~~`to_html`~~ — documented all 10 parameters
- ~~`GraphCapture`~~ — documented constructor parameters

### 5.3 ~~Misleading names~~ — FIXED
- ~~`print_table`~~ — fixed docstring to "Return a formatted table string"
- ~~`visualizer` module~~ — added "JSON" to format list
- ~~`_VIEW_OPS`~~ — fixed comment to "reshapes, views, or low-value plumbing"

---

## Phase 6: Functions Too Long / Need Decomposition — DEFERRED

Maintainability improvement only — no correctness or behavioral change. Decomposing
these functions would be a large refactor with significant breakage risk and no
immediate benefit. Deferred until a natural refactoring opportunity arises.

| File | Function | Lines | Suggested split |
|------|----------|-------|-----------------|
| `export.py` | `export_aten_program` | 326 | weight collection, header gen, code emission, test harness |
| `export.py` | `_emit_real_tensor_harness` | 167 | per-step verification helpers |
| `op_dump.py` | `build_op_groups` | 264 | view hiding, I/O computation, group construction |
| `op_dump.py` | `dump_grouped_tensors` | 220 | per-graph processing, file writing, summary |
| `op_dump.py` | `dump_cli` | 126 | recipe detection, arg parsing, execution |
| `viewer_template.html` | `layout()` | 352 | filtering, toposort, barycenter, positioning, grouping |
| `viewer_template.html` | `draw()` | 235 | edges, nodes, labels, highlights |

---

## Phase 7: Design Issues

### 7.1 ~~`editor.py` — unbounded undo history~~ — FIXED
Changed `_history` from `list[GraphModule]` to `deque(maxlen=50)`.

### 7.2 ~~`editor.py:validate()` has side effects~~ — FIXED
Now works on `copy.deepcopy(self.gm)` — the real `self.gm` is never mutated.

### 7.3 ~~`editor.py:compile()` returns mutable alias~~ — FIXED
Added docstring documenting the aliasing behavior and advising deep-copy if needed.

### 7.4 ~~`op_dump.py` — global mutable `_USE_H5_BFLOAT16`~~ — FIXED
Removed global. `h5_bfloat16` is now passed explicitly through
`dump_grouped_tensors` → `_write_h5_multi` → `_torch_to_numpy`.

### 7.5 ~~`inspector.py` — no caching of `nodes()`~~ — FIXED
Result is cached in `self._cached_nodes` after first call.

### 7.6 ~~`inspector.py:op_categories` — keyword substring matching~~ — FIXED
Replaced `any(kw in name ...)` substring matching with exact token set
intersection. Op names are split on `.`/`:` into segments, then segments
on `_` into word tokens. Keywords match via `keywords & tokens` (set
intersection). Added missing keyword variants (`negate`, `conv1d`-`conv3d`,
`convolution`).

### 7.7 `auto.py` — `sum(p.numel() ...)` in monkey-patched `__call__` — REJECTED
Misstated severity. Python's short-circuit `and` at `_seen_classes` guard
(line 161) prevents 99%+ of calls from reaching the expensive operation.
Only the first call per module class hits it.

### 7.8 Circular import fragility — REJECTED (acceptable pattern)
Lazy imports inside function bodies is a standard Python pattern for breaking
circular dependencies. The cycle is stable and well-understood.

---

## Phase 8: Missing Type Hints

### 8.1 ~~High-value missing annotations (public API)~~ — FIXED
- ~~`capture.py`~~ — added return types to `backend`, `graph`, `__len__`,
  `__getitem__`, `__iter__`; added `Any`, `Iterator`, `Graph` imports
- ~~`inspector.py:__init__`~~ — typed `captured: CapturedGraph | GraphModule`
- ~~`visualizer.py:__init__`~~ — typed `source: CapturedGraph | GraphModule`
- ~~`editor.py:_checkpoint`~~ — added `-> None`
- `export.py:_node_to_python` — already has `-> str` (false positive)
- `extract.py`, `tensor_dump.py`, `kbox_gen.py` — already fully annotated (false positive)

---

## Phase 9: `test_models.py` Overhaul

### 9.1 Convert to pytest — DEFERRED
Large refactor. The current script-based approach works and runs all 60+ recipes.
Defer until there's a concrete need for pytest features.

### 9.2 Add behavioral assertions — DEFERRED
Would require establishing expected values for 60+ models. Low ROI vs current
"does it crash + does the exported script run?" coverage.

### 9.3 ~~Fix `sys.path` pollution~~ — FIXED
Added `saved_path = sys.path[:]` before recipe call and `sys.path[:] = saved_path`
in `finally` block of `run_one()`.

### 9.4 ~~Fix duplicate `moe_recipe`~~ — FIXED (Phase 1)
Renamed to `moe_topk_recipe` with key `"moe_topk"`.

---

## Phase 10: `viewer_template.html` Quality

### 10.1 Extract edge routing into shared function — DEFERRED
DRY improvement. Would require careful coordination to avoid breaking hit-testing.

### 10.2 Consolidate op-name filter lists — DEFERRED
Code quality. The lists work correctly as-is.

### 10.3 Reduce global state — DEFERRED
Architectural refactor with no behavioral change. High risk for the template.

### 10.4 Throttle mousemove — DEFERRED
Performance optimization. Not needed unless users report sluggish interaction.

### 10.5 ~~XSS: sanitize innerHTML~~ — FIXED
Added `_esc()` HTML entity escaping function. Applied to all data interpolations
in `showNodeInfo`, tooltip, and `buildGroupSection` — covering node names, targets,
shapes, dtypes, source code, file paths, module paths, kernel names, and group labels.

### 10.6 Python double-brace escaping — DEFERRED
Structural issue. Switching templating would require rewriting 993+ lines of JS
with no functional change.

---

## Suggested Execution Order

1. **Phase 1** (Bugs) — highest impact, some are correctness issues
2. **Phase 2** (Dead code) — quick wins, reduces noise for subsequent phases
3. **Phase 3** (DRY) — create `_utils.py` helpers, biggest maintainability improvement
4. **Phase 4** (Error swallowing) — add logging to the critical cases
5. **Phase 5** (Docs) — fix wrong docstrings while code is fresh in mind
6. **Phase 6** (Long functions) — decompose the worst offenders
7. **Phase 7** (Design) — address architectural issues
8. **Phase 8** (Type hints) — add to public API surfaces
9. **Phase 9** (Tests) — convert to pytest, add assertions
10. **Phase 10** (HTML viewer) — JS quality improvements
