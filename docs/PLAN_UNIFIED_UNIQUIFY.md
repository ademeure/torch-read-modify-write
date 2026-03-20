# Plan: Unified Uniquification from IR JSON

## Problem

There are currently two independent uniquification systems:

1. **export.py** — operates on live FX `GraphModule` objects (in-memory only, during
   capture). Uses `nn_module_stack` metadata for module-tree grouping. Supports
   hierarchical depth + partial matching. Cannot work from saved files.

2. **kbox_gen.py** — operates on H5 files containing replay script strings (text).
   Groups by source line with module instance disambiguator. Uniquifies by
   normalizing variable names in replay scripts and comparing strings. Cannot do
   hierarchical or cross-module pattern detection.

Both are coupled to their input format.  The user wants a single uniquification
engine that works from **our own IR representation** — something we control, that
can be saved to disk and reprocessed later without needing a live PyTorch graph
or an H5 tensor dump.

## The Right Input: Enriched IR JSON

The existing `ir_json.py` / `internal_ir.py` IR is close but missing one critical
piece: **per-node module hierarchy**.  Currently the IR stores at most a single
`source.module_path` and `source.module_type` (the deepest nn.Module in the
stack).  Uniquification needs the full chain to do hierarchical grouping.

### What to add to each IR node

```json
{
  "name": "mm_3",
  "target": "aten.mm.default",
  "args": [...],
  "kwargs": {...},
  "meta": {"shape": [64, 64], "dtype": "torch.float32"},

  "module_stack": [
    {"path": "transformer.h.0", "type": "Block"},
    {"path": "transformer.h.0.attn", "type": "CausalSelfAttention"},
    {"path": "transformer.h.0.attn.c_q", "type": "Linear"}
  ],

  "source": {
    "file": "model.py",
    "line": 74,
    "code": "q = self.c_q(x).view(B, T, self.n_head, self.head_dim)"
  }
}
```

**`module_stack`** is the full `nn_module_stack` chain, serialized as a list of
`{path, type}` dicts.  Only entries with a numeric-indexed ancestor are included
(same filter as `_compute_all_module_levels` but preserving the data in the IR
instead of computing it on-the-fly from live nodes).

**`source`** is simplified: just file, line, code.  No `fn_name` or `module_path`
(those are redundant with `module_stack`).

This gives the uniquification engine everything it needs from a plain JSON file.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Live FX Graph      │  (capture time)
                    │   (GraphModule)      │
                    └──────────┬──────────┘
                               │
                    graph_to_ir() + annotate
                               │
                    ┌──────────▼──────────┐
                    │   Enriched IR JSON   │  ← single source of truth
                    │   (dict / .json file)│
                    └──────────┬──────────┘
                               │
                    uniquify_ir()          ← NEW: the unified engine
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
     ┌────────────────┐ ┌───────────┐  ┌──────────────┐
     │ Python codegen │ │ kbox test │  │ HTML / viz    │
     │ (export.py)    │ │ scripts   │  │ (visualizer)  │
     └────────────────┘ └───────────┘  └──────────────┘
```

### Key principle

**Capture produces IR.  Uniquification consumes IR.  Code generation consumes
uniquified IR.**  No step reaches back into the live FX graph for uniquification
purposes.

---

## The Unified Uniquification Engine: `uniquify_ir()`

### Input

```python
def uniquify_ir(
    ir_graph: dict,
    *,
    strategy: str = "module",       # "module" | "source_line" | "both"
    depth: int = -1,                # -1 = all, 1 = top-level, 2 = one sub-level, ...
    min_ops: int = 0,               # skip groups with fewer ops
) -> UniqueIR:
```

- `ir_graph` — a single graph section from the IR JSON (the `"forward"` or
  `"backward"` dict with `placeholders`, `nodes`, `returns`).
- Returns a `UniqueIR` object containing the detected groups and the residual
  (non-grouped) nodes.

### Output

```python
@dataclass
class UniqueGroup:
    """A set of structurally identical node sequences."""
    fn_name: str                            # "residual_block", "mlp", "linear"
    template_key: str                       # "transformer.h.*" or "model.py:74"
    group_type: str                         # "module" or "source_line"
    module_type: str                        # "ResidualBlock" (for module groups)
    instances: list[GroupInstance]           # one per repetition
    params: list[ParamDef]                  # function signature
    returns: list[ReturnDef]                # return values
    body_nodes: list[dict]                  # IR nodes with genericized names

@dataclass
class GroupInstance:
    """One concrete occurrence of a UniqueGroup."""
    instance_id: str                        # "0", "1", etc.
    input_map: dict[str, str]               # generic_param → actual_node_name
    output_map: dict[str, str]              # generic_return → actual_node_name
    first_node: str                         # name of first node (for ordering)

@dataclass
class UniqueIR:
    """Result of uniquification: groups + residual nodes."""
    groups: list[UniqueGroup]
    # For codegen: which nodes are grouped and where call sites go
    grouped_nodes: set[str]                 # node names consumed by groups
    call_sites: dict[str, tuple[UniqueGroup, GroupInstance]]  # first_node → (group, instance)
```

### Grouping strategies

#### `strategy="module"` (current export.py approach, using `module_stack`)

1. For each node, read `module_stack` and compute template keys at every depth:
   `transformer.h.*`, `transformer.h.*.attn`, etc.
2. Group nodes by template key, bucket by structural signature.
3. Process shallowest-first, skip already-grouped nodes.

This is exactly what `_detect_unique_groups` does today, but operating on IR
dicts instead of live FX nodes.

#### `strategy="source_line"` (current kbox approach, using `source`)

1. For each node, build a grouping key from `source.file` + `source.line` +
   module instance disambiguator (extracted from `module_stack[0].path`).
2. Group nodes by this key.
3. Build structural signatures from the IR nodes in each group.
4. Bucket by signature (same partial-matching logic as module strategy).

This reproduces kbox_gen's `_normalize_replay` approach but using structural
signatures instead of text comparison, which is more robust.

#### `strategy="both"` (new)

Run module strategy first at depth=1 (top-level blocks).  For nodes NOT captured
by a top-level module group, fall back to source_line grouping.  This gets the
best of both: block-level extraction where modules match, plus fine-grained
source-line extraction for the glue code between blocks.

### Structural signature (shared by all strategies)

The existing `_build_structural_signature` algorithm works on IR node dicts.
It normalizes:
- Internal node references → `("@", relative_index)`
- External node references → `("ext", position)`
- Literals → `("lit", value)`

This is already format-agnostic — it works on `{"target": ..., "args": [...], "kwargs": {...}}`
dicts, not on FX Node objects.  The only FX-specific part today is how we get
the set of node names in a group (from `Node.name`).  In the IR JSON, this is
just `node["name"]`.

---

## Implementation Plan

### Step 1: Enrich IR JSON with `module_stack` (small change)

**File**: `torch_graph/internal_ir.py`, function `graph_to_ir()`

Add `module_stack` to each node dict during serialization:

```python
# In graph_to_ir(), inside the node loop:
nn_mod = node.meta.get("nn_module_stack") or node.meta.get("fwd_nn_module_stack", {})
if nn_mod:
    stack = []
    for _, v in nn_mod.items():
        if isinstance(v, tuple) and len(v) >= 2:
            path = clean_self_path(v[0])
            mod_type = v[1].__name__ if hasattr(v[1], "__name__") else str(v[1])
            # Only include entries with a numeric-indexed ancestor
            parts = path.split(".")
            if any(p.isdigit() for p in parts[1:]):  # skip "self"
                stack.append({"path": path, "type": mod_type})
    if stack:
        node_dict["module_stack"] = stack
```

Also add raw source info (file + line + code) directly to nodes, independent
of the source_map lookup.  This can come from `source_fn_stack`:

```python
src_fn = node.meta.get("source_fn_stack") or node.meta.get("fwd_source_fn_stack", [])
if src_fn:
    # The source_fn_stack contains (name, target) tuples
    # We want the source location, which requires the source_map
    # For now, store the source_fn name for source_line grouping
    node_dict["source_fn"] = src_fn[-1][0] if src_fn else ""
```

**Impact**: Every IR JSON file will now contain the module hierarchy per node.
Existing code that reads IR JSON will ignore the new fields (forward compatible).

### Step 2: New module `torch_graph/uniquify.py`

Create a new self-contained module that operates purely on IR JSON dicts.
No imports from `torch`, `torch.fx`, or any live-graph code.

```python
"""Unified uniquification engine operating on IR JSON.

Works on serialized IR dicts — no live PyTorch graph or H5 files needed.
Supports module-tree and source-line grouping strategies.
"""

from __future__ import annotations
import re
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class UniqueGroup:
    fn_name: str
    template_key: str
    group_type: str                 # "module" or "source_line"
    module_type: str
    instances: list[GroupInstance]
    params: list[dict]              # [{name, annotation}, ...]
    returns: list[dict]             # [{name, annotation}, ...]
    body_nodes: list[dict]          # IR nodes with genericized names
    body_code: str                  # generated Python (from IR python field)

@dataclass
class GroupInstance:
    instance_id: str
    input_map: dict[str, str]       # generic → actual
    output_map: dict[str, str]      # generic → actual
    first_node: str

@dataclass
class UniqueIR:
    groups: list[UniqueGroup]
    grouped_nodes: set[str]
    call_sites: dict[str, tuple[UniqueGroup, GroupInstance]]


def uniquify_ir(
    ir_graph: dict,
    *,
    strategy: str = "module",
    depth: int = -1,
    min_ops: int = 0,
) -> UniqueIR:
    """Detect and extract repeated patterns from an IR graph dict."""
    ...
```

#### Core algorithm (shared by both strategies)

The structural signature building and group construction logic is already
strategy-agnostic.  The only difference is how nodes are assigned to
template groups in Phase 1:

```python
def _group_by_module(nodes, depth):
    """Phase 1 for module strategy: group by module_stack template keys."""
    templates = defaultdict(lambda: defaultdict(list))
    for node in nodes:
        for entry in node.get("module_stack", []):
            path = entry["path"]
            template = _normalize_template_key(path)
            instance_idx = _extract_instance_idx(path)
            templates[template][instance_idx].append(node)
    return templates

def _group_by_source_line(nodes):
    """Phase 1 for source_line strategy: group by file:line@instance."""
    templates = defaultdict(lambda: defaultdict(list))
    for node in nodes:
        src = node.get("source", {})
        mod_stack = node.get("module_stack", [])
        if not src.get("file") or not src.get("line"):
            continue
        short_file = src["file"].rsplit("/", 1)[-1]
        # Disambiguator from module_stack (first numeric-indexed path)
        disambig = _instance_disambiguator(mod_stack)
        if disambig:
            key = f"{short_file}:{src['line']}@{disambig}"
            instance_idx = disambig.rsplit(".", 1)[-1]  # the numeric part
        else:
            key = f"{short_file}:{src['line']}"
            instance_idx = "0"
        templates[key][instance_idx].append(node)
    return templates
```

After Phase 1, Phase 2 (signature building + partial matching) and Phase 3+4
(I/O computation + code generation) are identical regardless of strategy.
These are already implemented in `_build_structural_signature` and
`_build_group_from_instances`.

#### Porting `_build_structural_signature` to work on IR dicts

The existing function takes `ir_nodes: list[dict]` and `group_names: set[str]`
— it already works on dicts, not FX nodes.  The only change: it needs to be
importable from `uniquify.py` without importing `export.py`.

Options:
1. Move `_build_structural_signature` to `uniquify.py` (breaking import if
   export.py still needs it)
2. Move it to `internal_ir.py` (shared utility)
3. Keep it in `export.py` and have `uniquify.py` import from there

Option 2 is cleanest — `internal_ir.py` already owns the IR schema.

#### Porting `_build_group_from_instances` to work on IR dicts

The current function takes `instances: dict[str, list[Node]]` where `Node` is
an FX node.  It uses `node.name`, `node.users`, and the graph's output node.

For the IR-based version:
- `node.name` → `node["name"]` (trivial)
- `node.users` → build a usage map from IR: scan all nodes' `args`/`kwargs`
  for `{"node": "name"}` references (one-time precomputation)
- Output node → `ir_graph["returns"]` (already in the IR)

This is a straightforward port.  The core logic (finding external inputs,
computing outputs, genericizing names, generating body code) stays the same.

### Step 3: Migrate export.py to use `uniquify.py`

Replace `_detect_unique_groups` in `export.py` with a thin wrapper:

```python
def _detect_unique_groups(compute_nodes, ir_nodes_by_name, name_remap,
                          source_map, is_backward, *, max_depth, min_ops):
    # Build IR graph dict from the compute nodes
    ir_graph = _build_graph_ir(graph_module, ...)  # already done by caller

    # Enrich with module_stack (from live FX nodes, since we have them)
    _enrich_module_stacks(ir_graph, compute_nodes)

    # Call unified engine
    from torch_graph.uniquify import uniquify_ir
    result = uniquify_ir(ir_graph, strategy="module", depth=max_depth, min_ops=min_ops)

    # Convert UniqueGroup → _UniqueGroup (existing dataclass)
    return [_convert_group(g) for g in result.groups]
```

Or better: change `export_graph_to_python` to call `uniquify_ir` directly
on the `ir_graph` dict it already builds at line 2529, and skip the
`_detect_unique_groups` indirection entirely.

### Step 4: Migrate kbox_gen.py to use `uniquify.py`

Replace `detect_unique_groups` in `kbox_gen.py`:

```python
def detect_unique_groups(groups: list[GroupInfo]) -> tuple[list[UniqueGroupInfo], set[int]]:
    # Convert GroupInfo objects to IR-like node dicts
    ir_nodes = _groups_to_ir_nodes(groups)

    # Call unified engine with source_line strategy
    from torch_graph.uniquify import uniquify_ir
    result = uniquify_ir({"nodes": ir_nodes}, strategy="source_line")

    # Convert back to UniqueGroupInfo
    return _convert_to_kbox_groups(result, groups)
```

The `_groups_to_ir_nodes` conversion parses the replay scripts into IR-like
dicts.  Or better: have `op_dump.py` save the IR nodes directly into the H5
groups (instead of just the replay script string), so `kbox_gen.py` can load
structured IR instead of parsing text.

### Step 5: Add `strategy` parameter to public API

```python
def export_aten_program(
    capture,
    output_path,
    *,
    uniquify: bool = True,
    uniquify_depth: int = -1,
    uniquify_min_ops: int = 0,
    uniquify_strategy: str = "module",   # NEW: "module" | "source_line" | "both"
    ...
)
```

### Step 6: Save/load uniquification from IR JSON files

Since `uniquify_ir` works on plain dicts, it can work on IR JSON loaded from
disk:

```python
import json
from torch_graph.uniquify import uniquify_ir

ir = json.load(open("model_ir.json"))
result = uniquify_ir(ir["forward"], strategy="module", depth=2, min_ops=5)

# result.groups contains all detected patterns
# result.call_sites maps node names to group instances
# Can regenerate Python code, kbox scripts, or visualizations from this
```

This enables offline analysis: capture once, experiment with different
uniquification configs without re-running the model.

---

## What each strategy is good for

### `strategy="module"` (default)

**Groups by**: nn.Module hierarchy (transformer.h.*, transformer.h.*.attn, etc.)
**Granularity**: One group per repeated nn.Module instance
**Best for**: Standard architectures where modules are the natural unit of reuse.
Transformer layers, ResNet blocks, etc.
**Weakness**: Can't detect patterns that cross module boundaries or patterns
within a single module that don't correspond to sub-modules.

### `strategy="source_line"`

**Groups by**: Source file + line number + module instance
**Granularity**: One group per Python statement per module instance
**Best for**: Fine-grained analysis. Identifies exactly which source lines
produce identical op sequences. Useful for kbox-style per-op testing.
**Weakness**: Over-segments. A single attention block becomes 5-10 separate
groups (one per source line). Doesn't naturally produce "attention()" as a
unit.

### `strategy="both"`

**Groups by**: Module at depth=1 first, then source_line for ungrouped nodes
**Granularity**: Coarse for repeated modules, fine for everything else
**Best for**: Maximum extraction. Gets block-level functions where modules
repeat, plus source-line functions for the glue code and one-off patterns.

---

## Migration path

1. **Step 1** (small, safe): Add `module_stack` to IR JSON nodes. No behavior change.
2. **Step 2** (new code): Create `uniquify.py` with the unified engine.
   Write tests that operate on hand-crafted IR dicts (no model capture needed).
3. **Step 3** (refactor): Wire `export.py` through `uniquify.py`.
   Existing tests validate identical behavior.
4. **Step 4** (refactor): Wire `kbox_gen.py` through `uniquify.py`.
   Existing kbox tests validate behavior.
5. **Step 5** (feature): Add `strategy` parameter. New tests for source_line
   and both strategies.
6. **Step 6** (feature): CLI support for offline uniquification from saved
   IR JSON files.

Steps 1-3 can be done without changing any user-visible behavior.  Steps 4-6
add new capabilities.

---

## What moves where

| Current location | Current purpose | New location |
|---|---|---|
| `export.py: _build_structural_signature` | Signature from IR nodes | `internal_ir.py` (shared) |
| `export.py: _detect_unique_groups` | Module-tree grouping | `uniquify.py: uniquify_ir` |
| `export.py: _build_group_from_instances` | I/O + codegen from instances | `uniquify.py: _build_group` |
| `export.py: _compute_all_module_levels` | Multi-depth module paths | `uniquify.py: _module_templates` (from IR) |
| `export.py: _disambiguate_fn_names` | Name dedup | `uniquify.py: _disambiguate` |
| `export.py: _generate_unique_fn_def` | Function def codegen | stays in `export.py` (codegen-specific) |
| `export.py: _generate_call_site` | Call site codegen | stays in `export.py` (codegen-specific) |
| `kbox_gen.py: _normalize_replay` | Text-based dedup | replaced by `uniquify.py` |
| `kbox_gen.py: detect_unique_groups` | Text-based grouping | wrapper around `uniquify.py` |

Code generation (`_generate_unique_fn_def`, `_generate_call_site`) stays in
export.py because it's specific to Python code output.  The detection/grouping
logic moves to `uniquify.py` because it's format-agnostic.

---

## Open questions

1. **Should `module_stack` include non-numeric-indexed modules?**  E.g., `self.embed`
   has no numeric index — should it appear?  Current answer: no, same filter
   as `_compute_all_module_levels`.  But source_line strategy doesn't need
   numeric indices, so maybe it should be the full stack.

2. **Should we store the replay script in IR JSON too?**  It's derivable from
   the `python` field of each node, but having it pre-assembled per source-line
   group would simplify the kbox path.  Probably not worth it — the `python`
   field is sufficient.

3. **H5 format change**: Should `op_dump.py` save IR node dicts in H5 groups
   alongside (or instead of) replay script strings?  This would make kbox_gen's
   path cleaner but changes the H5 schema.  Could be done as a v2 format with
   backward compat.
