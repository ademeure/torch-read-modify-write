# Plan: Hierarchical Uniquification of Captured Aten Graphs

## Problem Statement

Captured aten forward files are huge monolithic functions (3700+ lines for 8-layer GPT).
The current `uniquify=True` flag in `export_aten_program()` requires **exact structural
match** at the top nn.Module level (e.g. `transformer.h.*`). This fails when layers are
heterogeneous (autoresearch GPT has even/odd layer differences: `ve_gate` and `value_embeds`
only on odd layers). Result: uniquified file is actually *larger* than original (3771 vs 3727).

### What we want

```python
# Instead of 3700 lines of flat aten ops, generate:

def rope(q_half1, q_half2, cos, sin):
    """RoPE rotation — shared by all 8 layers, Q and K paths."""
    ...  # 9 ops
    return (rotated,)

def attention(x, c_q_w, c_k_w, c_v_w, c_proj_w, cos, sin, *, ve_gate_w=None):
    """Self-attention — shared by all 8 layers (ve_gate optional)."""
    ...  # ~40 ops, calls rope() twice
    return (attn_out,)

def mlp(x, c_fc_w, c_proj_w):
    """MLP block — identical across all 8 layers."""
    ...  # ~12 ops
    return (mlp_out,)

def transformer_block(x, x0, resid_lambda, x0_lambda, attn_params, mlp_params, *, ve_gate_w=None, ve_w=None):
    """Full transformer block — all 8 layers."""
    ...  # calls attention(), mlp(), ~5 glue ops
    return (block_out,)

def forward(input, cos, sin, wte_weight, resid_lambdas, x0_lambdas, *layer_params, lm_head_weight):
    x = aten.embedding(wte_weight, input)
    for i in range(8):
        x = transformer_block(x, x0, resid_lambdas[i], x0_lambdas[i], ...)
    return aten.linear(x, lm_head_weight)
```

This is ~200 lines instead of 3700. An LLM editing the RoPE kernel edits it once.

---

## Existing Code to Build On

### 1. `export.py` uniquification (lines 1938-2487)
- `_UniqueGroup` dataclass, `_detect_unique_groups()`, `_build_structural_signature()`
- Works well for homogeneous layers. Algorithm: group by template key, compare structural
  signatures, extract function with generic param names.
- **Limitation**: requires ALL instances to have identical signature. No partial matching.
- **Limitation**: only operates at one depth level (the first numeric index in module path).
- **Limitation**: backward graphs explicitly skipped (`if uniquify and not is_backward`).

### 2. `kbox_gen.py` uniquification (lines 453-590)
- `_normalize_replay()` + `detect_unique_groups()` — text-based deduplication of replay scripts.
- Simpler approach: normalize variable names to positional markers (I0, I1, O0, O1),
  then compare normalized strings.
- `UniqueGroupInfo` dataclass, `h5_suite` contract with kernelbox.

### 3. IR JSON pipeline (`ir_json.py`, `condense_ir.py`, `internal_ir.py`)
- `graph_to_ir()` / `graph_to_ir_json()` — lossless IR dictionaries from FX graphs.
- `condense_ir.py` — collapses foldable ops (views, detach, getitem) in JSON.
- These are read-only analysis tools. Not currently connected to code generation.

---

## Implementation Plan

### Phase 1: Sub-Module Level Uniquification (HIGH IMPACT)

**Goal**: Extract functions at every nn.Module depth, not just the top numeric-indexed one.

#### Step 1.1: Multi-depth module grouping

**File**: `torch_graph/export.py`
**Function**: `_compute_node_top_module()` (lines 1955-1986)

Current behavior: returns the FIRST numeric-indexed module path (depth >= 2 preferred, >= 1 fallback).
This means `transformer.h.0.attn` and `transformer.h.0.mlp` both return `transformer.h.0`.

**Change**: New function `_compute_all_module_levels(node)` that returns ALL numeric-indexed
module paths at every depth:

```python
def _compute_all_module_levels(node: Node) -> list[tuple[str, str, str]]:
    """Return [(module_path, instance_idx, module_type), ...] at every nesting depth.

    For a node in transformer.h.0.attn (CausalSelfAttention):
      Returns: [
        ("transformer.h.0", "0", "Block"),           # depth=outermost
        ("transformer.h.0.attn", "0", "CausalSelfAttention"),  # sub-module
      ]
    But the second entry only applies if 'attn' has a numeric index somewhere,
    or if we extend to name-based grouping (see Step 1.2).
    """
```

Wait — the current algorithm requires a **numeric index** in the path to identify instances.
Sub-modules like `attn` and `mlp` don't have numeric indices. They're singletons within
each block. The numeric index comes from the parent (`transformer.h.0`).

**Revised approach**: The real hierarchy is:
```
transformer.h.0          → instance 0 of transformer.h.*
  transformer.h.0.attn   → the attention sub-module of instance 0
  transformer.h.0.mlp    → the MLP sub-module of instance 0
```

So the grouping should be:
1. First group by top-level numeric index: `transformer.h.*` → instances {0,1,...,7}
2. Within each instance's nodes, sub-group by sub-module path:
   `transformer.h.*.attn` → attention nodes, `transformer.h.*.mlp` → MLP nodes
3. Compare sub-groups across instances: are all `attn` sub-groups identical? All `mlp`?
4. If the full block ISN'T identical but sub-modules ARE, extract sub-module functions.

**Algorithm**:

```python
def _detect_unique_groups_hierarchical(
    compute_nodes: list[Node],
    ir_nodes_by_name: dict[str, dict],
    name_remap: dict[str, str],
    source_map: dict | None,
) -> list[_UniqueGroup]:
    """Hierarchical uniquification: extract at every sub-module level."""

    # Step A: Group nodes by FULL module path (not just top-level)
    # For each node, extract its full nn_module_stack and record ALL levels.
    # E.g. node in transformer.h.0.attn.c_q gets tagged with:
    #   - ("transformer.h.*", "0")           → top block
    #   - ("transformer.h.*.attn", "0")      → attention sub-module
    #   - ("transformer.h.*.attn.c_q", "0")  → individual linear (probably too granular)

    # Step B: For each template level, check if ALL instances match structurally.
    # Start from the DEEPEST level and work up (bottom-up).
    # If transformer.h.* matches → extract one function, done.
    # If not → try transformer.h.*.attn, transformer.h.*.mlp separately.
    # If attn matches across all 8 → extract attention().
    # If attn only matches across {0,2,4,6} and {1,3,5,7} → two attention variants.

    # Step C: Avoid double-extraction.
    # If we extract transformer_block() (all instances match), don't also extract
    # attention() and mlp() — they're already inside transformer_block().
    # Only extract sub-modules when the parent level DOESN'T match.

    # Step D: Handle the "glue" ops.
    # After extracting attention() and mlp() from a block, there are leftover ops
    # (RMSNorm, residual add, lambda scaling). These become the block-level function
    # body that CALLS attention() and mlp().
    # If these glue ops are also identical across instances → extract block() too,
    # but now block() calls attention() and mlp() instead of inlining them.
```

**Key implementation detail — nn_module_stack traversal**:

Each FX node's `meta["nn_module_stack"]` is an OrderedDict where keys are module paths
and values are (path, type) tuples. The entries go from outermost to innermost:

```python
OrderedDict([
    ('', ('', GPT)),
    ('transformer', ('transformer', Transformer)),
    ('transformer.h', ('transformer.h', ModuleList)),
    ('transformer.h.0', ('transformer.h.0', Block)),
    ('transformer.h.0.attn', ('transformer.h.0.attn', CausalSelfAttention)),
    ('transformer.h.0.attn.c_q', ('transformer.h.0.attn.c_q', Linear)),
])
```

To build sub-module groups, we need the **deepest nn_module_stack entry that still contains
the parent's numeric index**. For a node in `transformer.h.0.attn.c_q`:
- Top-level group: `transformer.h.*` (instance "0")
- Sub-module: `transformer.h.*.attn` (derived by replacing "0" with "*")
- Sub-sub-module: `transformer.h.*.attn.c_q` (too granular — every Linear is identical anyway)

**Granularity control**: Only create sub-groups for nn.Module types that are "interesting"
(not Linear, not Conv2d — these are single-op modules). Use a heuristic: sub-group if the
module contains >= N ops (e.g., N=5). Or: sub-group if the module type appears in
`nn_module_stack` and has >= 2 distinct named children.

#### Step 1.2: Template key with sub-module suffix

**Current**: `_normalize_template_key("transformer.h.0")` → `"transformer.h.*"`

**New**: Also produce sub-module keys:
```python
_normalize_template_key("transformer.h.0.attn") → "transformer.h.*.attn"
_normalize_template_key("transformer.h.0.mlp")  → "transformer.h.*.mlp"
```

This is straightforward: replace only the numeric segment, keep the rest.

#### Step 1.3: Bottom-up extraction with dedup avoidance

```python
# Pseudo-code for hierarchical extraction:

all_groups = []
extracted_nodes = set()  # nodes already covered by an extracted function

# Sort template keys by depth (deepest first)
for template in sorted(all_templates, key=lambda t: t.count('.'), reverse=True):
    instances = template_instances[template]

    # Skip nodes already extracted at a deeper level
    for idx, nodes in instances.items():
        nodes = [n for n in nodes if n.name not in extracted_nodes]
        instances[idx] = nodes

    # Remove empty instances
    instances = {k: v for k, v in instances.items() if v}
    if len(instances) < 2:
        continue

    # Check structural match (existing algorithm)
    sigs = {idx: _build_structural_signature(ir_nodes, names) for idx, (nodes, ir_nodes, names) in ...}

    # Group instances by signature (allows partial matching — see Phase 2)
    sig_buckets = defaultdict(list)
    for idx, sig in sigs.items():
        sig_buckets[sig].append(idx)

    for sig, matching_idxs in sig_buckets.items():
        if len(matching_idxs) < 2:
            continue
        # Extract function for this bucket
        group = _build_unique_group(template, matching_idxs, ...)
        all_groups.append(group)
        extracted_nodes.update(group.all_node_names)

return all_groups
```

#### Step 1.4: Nested function calls

When a parent-level function is extracted AND it contains calls to child-level functions:

```python
def attention(x, c_q_w, c_k_w, c_v_w, c_proj_w, cos, sin):
    q = aten.mm(x, aten.t(aten._to_copy(c_q_w, dtype=torch.bfloat16)))
    k = aten.mm(x, aten.t(aten._to_copy(c_k_w, dtype=torch.bfloat16)))
    # ... RoPE ...
    q_rot = rope(q_half1, q_half2, cos, sin)  # ← calls extracted sub-function
    k_rot = rope(k_half1, k_half2, cos, sin)
    # ... SDPA + projection ...
    return (attn_out,)
```

**Implementation**: When generating the body of a parent function, check if any node
belongs to a child `_UniqueGroup`. If so, emit a call to the child function instead of
inlining the ops.

This requires ordering the extraction: child functions must be generated BEFORE parent
functions that reference them. The bottom-up traversal in Step 1.3 naturally achieves this.

**Data structure change**: `_UniqueGroup` needs a new field:

```python
@dataclass
class _UniqueGroup:
    # ... existing fields ...
    child_groups: list[_UniqueGroup]  # sub-functions called from this function's body
    child_call_sites: dict[str, tuple[_UniqueGroup, str]]  # node.name -> (child_group, instance_idx)
```

#### Step 1.5: Expected result for autoresearch

With sub-module extraction, even though `transformer.h.*` doesn't match (even vs odd layers),
we get:
- `rope()` — 9 ops, called 16x (Q and K for each of 8 layers)
- `linear_proj()` — 5 ops (to_copy + t + view + mm + unsafe_view), called 32x
  (c_q, c_k, c_v, c_proj for each layer, plus mlp)
  Actually — linear_proj might be too granular. Need MIN_OPS threshold.
- `attention_even()` — ~40 ops (layers 0,2,4,6), calls rope()
- `attention_odd()` — ~45 ops (layers 1,3,5,7, includes ve_gate), calls rope()
- `mlp()` — ~12 ops, called 8x (identical across all layers)
- `transformer_block_even()` — calls attention_even() + mlp() + glue ops
- `transformer_block_odd()` — calls attention_odd() + mlp() + glue ops

Or with optional-param approach (Phase 2): just `attention()` and `transformer_block()`
with `ve_gate_w=None` default.

Estimated result: **~300-400 lines** instead of 3700.

---

### Phase 2: Fuzzy/Optional-Parameter Matching (MEDIUM IMPACT)

**Goal**: Group layers that differ only by a few optional ops (e.g., ve_gate on odd layers).

#### Step 2.1: Signature diff analysis

When two instance groups have different structural signatures, compute their diff:

```python
def _signature_diff(sig_a: tuple, sig_b: tuple) -> SignatureDiff:
    """Compare two structural signatures and return the diff.

    Returns:
        SignatureDiff with:
        - common_ops: ops present in both (with positions)
        - a_only_ops: ops only in sig_a (with positions)
        - b_only_ops: ops only in sig_b (with positions)
        - similarity: float 0.0-1.0 (Jaccard on op sequences)
    """
```

Use sequence alignment (LCS or edit distance) to find the common subsequence.
Two groups are "fuzzy-matchable" if similarity > 0.85 (configurable threshold).

#### Step 2.2: Optional parameter detection

When the diff shows that sig_b has extra ops that reference extra external inputs:

```python
# sig_a (even layers): [..., mm(x, c_proj_w), ...]
# sig_b (odd layers):  [..., mm(x, c_proj_w), sigmoid(mm(v, ve_gate_w)), mul(...), ...]
#                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                              These extra ops use ve_gate_w (extra external input)
```

The extra external inputs become optional parameters with `None` default:

```python
def attention(x, c_q_w, c_k_w, c_v_w, c_proj_w, cos, sin, *, ve_gate_w=None):
    # ... common ops ...
    if ve_gate_w is not None:
        gate = aten.sigmoid(aten.mm(v_reshaped, ve_gate_w))
        v = aten.mul(v, gate)
    # ... rest of common ops ...
```

#### Step 2.3: Conditional code generation

**This is the hardest part**. The generated function body must interleave common ops with
conditional blocks.

**Algorithm**:
1. Align the two op sequences (LCS-based alignment).
2. Emit common ops as-is.
3. At each insertion point (ops in B but not A), emit `if optional_param is not None:` block.
4. At each insertion point (ops in A but not B), also emit conditional (rare — usually
   it's A ⊂ B, not symmetric).

**Constraint**: The conditional must not break data flow. If an optional op produces a value
used by a later common op, the common op must reference the conditional result:

```python
if ve_gate_w is not None:
    gate = aten.sigmoid(...)
    v = aten.mul(v, gate)  # v is reassigned
# later common op uses v — correct whether gate was applied or not
```

This works naturally when the optional ops modify a value in-place (overwrite `v`).
It's harder when the optional ops introduce entirely new data paths.

**Simplification**: For V1, only support the "optional suffix/insertion" pattern:
- The shorter signature is a prefix/subsequence of the longer one
- Extra ops at the end or at clear boundaries (between sub-modules)
- Extra ops only reference extra external inputs (no new internal cross-references
  to common ops that don't exist in the shorter version)

**Fallback**: If fuzzy matching is too complex for a particular pair, fall back to
two separate functions (attention_even, attention_odd). This is still better than
no extraction at all.

#### Step 2.4: Autoresearch-specific: ve_gate pattern

The autoresearch model's even/odd difference is specifically:
- Odd layers have `ve_gate_w` parameter and ~5 extra ops (sigmoid gate on values)
- Odd layers have `value_embeds` (separate embedding lookup, adds to input)

For `value_embeds`: this happens OUTSIDE the transformer block (between blocks), so it's
a separate call in the forward() body, not a conditional inside the block.

For `ve_gate`: this is inside attention, between the V projection and SDPA. It's a clean
insertion point: 3 extra ops (mm, sigmoid, mul) that gate V before attention.

This should be straightforward for the optional-parameter approach.

---

### Phase 3: Backward Graph Uniquification (MEDIUM IMPACT)

**Goal**: Apply the same uniquification to backward graphs.

#### Step 3.1: Why backward is currently skipped

Line 2616: `if uniquify and not is_backward:` — explicitly skipped.

Reasons this was probably done:
1. **Backward node metadata**: Backward FX nodes may lack `nn_module_stack` metadata
   (aot_autograd strips or transforms it). Need to verify what metadata is available.
2. **Saved tensor ordering**: Backward receives saved tensors in a specific order
   (non-tensors first, then tensors). Extracting sub-functions must preserve this.
3. **Gradient accumulation**: Multiple backward paths may contribute to the same gradient.
   Extracting a function must correctly handle gradient accumulation at boundaries.

#### Step 3.2: Investigate backward metadata

**Action**: Run a test capture and inspect backward graph nodes' metadata:

```python
capture = capture_aten_graphs(model, x, run_backward=True)
bw_graph = capture.backward_graphs[0]
for node in bw_graph.graph.nodes:
    if node.op == "call_function":
        print(node.name, node.meta.get("nn_module_stack"), node.meta.get("source_fn_stack"))
```

If `nn_module_stack` is present → we can use the same template-key grouping.
If not → we need an alternative grouping strategy (see Step 3.3).

#### Step 3.3: Alternative backward grouping (if no module metadata)

If backward nodes lack module paths, use **positional correspondence**:
- Forward uniquification identified that nodes [100-150] belong to layer 0's attention.
- The backward graph processes gradients in reverse order.
- Match backward nodes to forward nodes via `source_fn_stack` or via the
  saved-tensor correspondence (backward placeholder N corresponds to forward output N).

**Algorithm**:
1. Build forward→backward node mapping from saved tensor indices.
2. For each forward `_UniqueGroup`, find the corresponding backward nodes.
3. Check if backward groups are also structurally identical.
4. If yes → extract backward helper function with same naming.

#### Step 3.4: Backward function signature

Backward functions receive gradients and saved tensors:

```python
def attention_backward(
    d_attn_out: 'bfloat16[32, 2048, 512]',  # gradient input
    # saved tensors from forward:
    saved_q: 'bfloat16[32, 4, 2048, 128]',
    saved_k: 'bfloat16[32, 4, 2048, 128]',
    saved_v: 'bfloat16[32, 4, 2048, 128]',
    saved_attn_weights: 'bfloat16[32, 4, 2048, 2048]',
    # weight params:
    c_q_w: 'float32[512, 512]',
    c_k_w: 'float32[512, 512]',
) -> tuple['bfloat16[...]', ...]:
    # gradient computation aten ops
    ...
```

The tricky part: saved tensors are indexed positionally in the backward function's
placeholders. Extracting a sub-function must correctly partition which saved tensors
belong to which sub-function.

#### Step 3.5: Joint forward-backward extraction

Ideal output:

```python
# Forward
def attention(x, c_q_w, c_k_w, c_v_w, c_proj_w, cos, sin):
    ...
    return (attn_out, q, k, v, attn_weights)  # last 4 are saved for backward

# Backward
def attention_backward(d_attn_out, saved_q, saved_k, saved_v, saved_attn_weights, c_q_w, c_k_w):
    ...
    return (d_x, d_c_q_w, d_c_k_w, d_c_v_w, d_c_proj_w)
```

This requires coordinating forward and backward extraction. The `_UniqueGroup` for forward
should reference its backward counterpart.

---

### Phase 4: Autoresearch Integration (HIGH IMPACT)

**Goal**: Make uniquified files the default for autoresearch, and make LLM editing seamless.

#### Step 4.1: Enable uniquify in auto_install for autoresearch

**File**: `torch_graph/auto_install.py`

Currently `export_aten_program()` is called with `uniquify=False` by default.
Add a parameter to `auto_install` configuration:

```python
auto_install(
    model,
    uniquify=True,          # NEW: default True for autoresearch
    uniquify_backward=True, # NEW: also uniquify backward (Phase 3)
    min_group_ops=5,        # NEW: minimum ops to extract a sub-function
    fuzzy_threshold=0.85,   # NEW: similarity threshold for fuzzy matching
)
```

#### Step 4.2: Capture script integration

**File**: `scripts/autoresearch_capture.py` (or equivalent)

Pass `uniquify=True` to the export call. Verify the generated file is:
1. Syntactically valid Python (exec() it)
2. Produces bit-identical output vs non-uniquified version
3. Significantly shorter

#### Step 4.3: Autoresearch optimization loop integration

**File**: `scripts/autoresearch_optimize_loop.py`

Currently uses regex to find and replace op sequences across all layers (fragile).
With uniquification, the LLM can:
1. Read the helper function (e.g., `rope()`)
2. Replace its body with a fused kernel
3. All 16 call sites automatically use the new implementation

**Change the optimization prompt** to tell the LLM:
- "Edit the helper functions in the SHARED LAYER FUNCTIONS section"
- "Changes to a helper function apply to all layers automatically"
- "To add a new fused kernel, replace the ops inside the helper function"

#### Step 4.4: Layer-level verification

**New feature**: After editing a helper function, verify it in isolation:

```python
def verify_helper(fn, fn_name, capture, instance_idx=0):
    """Run a single helper function with captured tensors and compare to reference."""
    # Load saved intermediate tensors for the specified instance
    # Call fn() with those tensors
    # Compare output to reference
    # Return (pass/fail, max_error)
```

This enables fast iteration: edit rope(), verify rope() in 0.1s, instead of running
full training step (10s+).

**Implementation**: Save per-helper intermediate tensors during capture:
```python
# During capture, when we detect a unique group:
for group in unique_groups:
    for idx in group.instance_order:
        # Save inputs and outputs for this instance
        instance_inputs = {name: recorded_tensors[actual_name]
                          for name, actual_name in group.input_name_map[idx].items()}
        instance_outputs = {name: recorded_tensors[actual_name]
                           for name, actual_name in group.output_name_map[idx].items()}
        torch.save({"inputs": instance_inputs, "outputs": instance_outputs},
                   f"data/{group.fn_name}_instance_{idx}.pt")
```

---

### Phase 5: KBox Integration (MEDIUM IMPACT)

**Goal**: Generate kernelbox test files from hierarchically-extracted helper functions.

#### Step 5.1: Bridge export.py uniquification → kbox_gen.py

Currently `kbox_gen.py` has its own independent uniquification based on replay script
text comparison. This should be unified with export.py's structural approach.

**Change**: After `export_aten_program()` produces `_UniqueGroup` objects, pass them to
`kbox_gen.py` instead of re-detecting groups from H5 files.

```python
# In the capture pipeline:
unique_groups = []
export_aten_program(capture, ..., uniquify=True, _unique_groups=unique_groups)

# Pass to kbox generation:
generate_kbox_from_unique_groups(unique_groups, h5_path, output_dir)
```

#### Step 5.2: Hierarchical kbox test structure

```
nanogpt_kbox/
├── grp_rope/
│   ├── grp_rope.py              # Test: run rope() against all instances
│   └── data/
│       ├── layer0_q.h5
│       ├── layer0_k.h5
│       ├── layer1_q.h5
│       └── ...                   # 16 instances
├── grp_attention/
│   ├── grp_attention.py          # Test: run attention() against all instances
│   └── data/
│       ├── layer0.h5
│       └── ...
├── grp_mlp/
│   ├── grp_mlp.py
│   └── data/
│       └── ...
└── grp_transformer_block/
    ├── grp_transformer_block.py  # Calls grp_attention + grp_mlp
    └── data/
        └── ...
```

#### Step 5.3: Cascading kernel optimization

When a CUDA kernel is developed for `rope()` in kernelbox:
1. `kbox iterate` verifies it against all 16 instances
2. On success, the kernel is automatically available in the aten file via `load_cuda()`
3. The `attention()` helper calls `rope()` which now uses the CUDA kernel
4. No manual per-layer replacement needed

---

### Phase 6: IR JSON / Condensed IR Integration (LOW IMPACT)

**Goal**: Make the JSON IR representation reflect unique groups.

#### Step 6.1: Add unique_groups to IR JSON

```python
# In ir_json.py capture_to_ir_json():
ir["unique_groups"] = [
    {
        "fn_name": group.fn_name,
        "template_key": group.template_key,
        "module_type": group.module_type,
        "num_instances": len(group.instances),
        "num_ops": len(group.body_code.split('\n')),
        "params": group.params,
        "returns": group.returns,
        "child_groups": [child.fn_name for child in group.child_groups],
    }
    for group in unique_groups
]
```

#### Step 6.2: Condensed IR with function boundaries

Add group membership to each node in condensed IR:

```json
{
    "name": "h0_attn_mm",
    "op": "aten.mm",
    "group": "attention",
    "group_instance": 0,
    "group_role": "q_projection"
}
```

This enables visualization tools to show the hierarchical structure.

---

### Phase 7: Nice-to-Have Features

#### 7.1: Layer diff reporting

When uniquification finds near-matches, report the diff:

```
=== Uniquification Report ===
transformer.h.* (Block): 8 instances
  Matched: {0,2,4,6} — 4 identical instances ("transformer_block_even")
  Matched: {1,3,5,7} — 4 identical instances ("transformer_block_odd")
  Diff: odd layers have +5 ops (ve_gate: sigmoid, mm, mul, expand, mul)

  Sub-modules:
    transformer.h.*.attn: 8 instances
      Even: 42 ops, Odd: 47 ops (+5 ve_gate ops)
      Common: 42/47 ops (89.4% similarity)
      → Merged with optional ve_gate_w parameter
    transformer.h.*.mlp: 8 instances → 100% identical → mlp()

  Result: 3 helper functions extracted
    rope()              — 9 ops, 16 call sites
    attention()         — 42-47 ops (conditional), 8 call sites
    mlp()               — 12 ops, 8 call sites

  Line reduction: 3727 → 342 (90.8%)
```

#### 7.2: `--uniquify` CLI flag

Add to `python -m torch_graph install`:

```bash
python -m torch_graph install --uniquify          # hierarchical (default)
python -m torch_graph install --uniquify=flat     # current behavior
python -m torch_graph install --no-uniquify       # disabled
```

#### 7.3: Inline loop generation

When ALL instances of a template are identical AND called sequentially:

```python
# Instead of 8 explicit calls:
for i in range(8):
    x = transformer_block(
        x, x0,
        resid_lambdas[i], x0_lambdas[i],
        layer_weights[i],  # packed as a list/dict
    )
```

This requires:
- Detecting that external inputs follow a regular pattern (weight_0, weight_1, ...)
- Packing per-layer weights into a list or dict
- Indexing into the list in the loop body

**Constraint**: Only valid when there are no inter-layer skip connections or when the
output of layer N feeds directly into layer N+1 (which is the common case for transformers).

**Weight packing**:
```python
# Generated weight section:
layer_weights = [
    {"c_q_w": transformer_h_0_attn_c_q_weight, "c_k_w": ..., ...},
    {"c_q_w": transformer_h_1_attn_c_q_weight, "c_k_w": ..., ...},
    # ...
]

# Or as stacked tensors for uniform shapes:
all_c_q_w = torch.stack([transformer_h_0_attn_c_q_weight, ...])  # [8, 512, 512]
```

#### 7.4: Cross-graph deduplication

Some op patterns repeat across forward AND backward (e.g., the transpose pattern
for attention is similar in both). Could extract shared utility functions.

Low priority — forward and backward typically have very different structure.

#### 7.5: Weight section compression

With uniquification, the weight section can also be grouped:

```python
# Current: 60+ individual weight declarations
# With grouping:
# === Per-layer weights ===
for i in range(8):
    layer_weights[i] = {
        "c_q_w": torch.randn([512, 512], dtype=torch.float32),
        "c_k_w": torch.randn([512, 512], dtype=torch.float32),
        # ... 6 weights per layer
    }
# === Global weights ===
wte_weight = torch.randn([8192, 512], dtype=torch.bfloat16)
lm_head_weight = torch.randn([8192, 512], dtype=torch.bfloat16)
```

#### 7.6: Standalone training loop integration

`torch_graph/standalone.py` generates self-contained training scripts.
These should also use uniquified helper functions for readability.

#### 7.7: HTML visualization with function boundaries

The HTML visualizer (generated by capture scripts) should show collapsible
function boundaries matching the uniquified structure.

#### 7.8: Profile-guided uniquification

Use CUDA profiling data to annotate each helper function with timing:

```python
def attention(x, ...):  # ~2.3ms per call, 8 calls = 18.4ms (62% of forward)
    ...

def mlp(x, ...):  # ~0.8ms per call, 8 calls = 6.4ms (22% of forward)
    ...

def rope(q, ...):  # ~0.1ms per call, 16 calls = 1.6ms (5% of forward)
    ...
```

This helps LLMs prioritize which helper to optimize.

---

## Implementation Order (Recommended)

### Sprint 1: Sub-module extraction (Phase 1) — 2-3 days
1. `_compute_all_module_levels()` — multi-depth module path extraction
2. `_detect_unique_groups_hierarchical()` — bottom-up extraction
3. Nested function call emission
4. Tests: autoresearch model, ResNet, standard transformer
5. Verify bit-identical output

### Sprint 2: Fuzzy matching (Phase 2) — 1-2 days
1. Signature diff / LCS alignment
2. Optional parameter generation
3. Conditional code blocks
4. Test: autoresearch even/odd layers merged with ve_gate_w=None

### Sprint 3: Backward + autoresearch integration (Phase 3 + 4) — 2 days
1. Investigate backward metadata availability
2. Implement backward uniquification (or positional correspondence fallback)
3. Enable in auto_install, verify full pipeline
4. Layer-level verification infrastructure

### Sprint 4: KBox + polish (Phase 5 + 7) — 1-2 days
1. Bridge export.py groups to kbox_gen.py
2. Hierarchical kbox test structure
3. CLI flag, diff report, weight compression
4. Update CLAUDE.md documentation

---

## Testing Strategy

### Unit tests (add to tests/test_uniquify.py)

1. **test_hierarchical_submodule_extraction** — model with Block containing Attn+MLP,
   verify attn() and mlp() extracted separately when Block is heterogeneous.

2. **test_hierarchical_nested_calls** — verify parent function body contains calls to
   child functions, not inlined ops.

3. **test_fuzzy_matching_optional_param** — model with some layers having extra linear,
   verify merged function with optional parameter.

4. **test_fuzzy_matching_threshold** — near-identical groups below threshold NOT merged.

5. **test_backward_uniquification** — verify backward helpers extracted (Phase 3).

6. **test_autoresearch_uniquification** — autoresearch 8L model, verify:
   - rope(), attention(), mlp() extracted
   - Line count < 500 (vs 3700)
   - Bit-identical output

7. **test_loop_generation** — identical sequential blocks → for loop (Phase 7.3).

8. **test_hierarchical_bit_identical** — run forward+backward with uniquified code,
   compare all intermediate tensors to non-uniquified version.

### Integration tests

1. **Full autoresearch capture** with `uniquify=True` → verify training runs.
2. **KBox generation** from uniquified groups → verify all tests pass.
3. **Standalone training loop** with uniquified functions → verify convergence.

---

## Risks and Mitigations

### Risk 1: nn_module_stack metadata missing for some ops
Some aten ops generated by aot_autograd (e.g., detach, alias) may lack module metadata.
**Mitigation**: Inherit module path from the nearest preceding node that has metadata.
Already partially handled in export.py's annotation logic.

### Risk 2: Backward graph lacks module correspondence
**Mitigation**: Use saved-tensor index mapping as fallback. If that fails, skip backward
uniquification for that model (degrade gracefully).

### Risk 3: Fuzzy matching produces incorrect code
Conditional blocks may break data flow if the optional ops have complex interactions.
**Mitigation**: Always verify bit-identical output. Fall back to separate functions
if verification fails. Start with conservative similarity threshold (0.95).

### Risk 4: Performance regression from function call overhead
Python function calls add ~1us overhead per call. For 8 layers × 3 functions = 24 calls,
that's ~24us — negligible vs the ~200ms step time.
**Mitigation**: Benchmark before/after. If overhead matters, emit inline code with
comments marking function boundaries instead of actual function calls.

### Risk 5: SymInt concretization ordering
Uniquified functions are generated in `export_graph_to_python()` which runs BEFORE
weight materialization. This ordering must be preserved.
**Mitigation**: No change needed — the current code already handles this correctly.
Just ensure the new hierarchical detection also runs within `export_graph_to_python()`.

---

## Key Code Locations Reference

| What | File | Lines | Purpose |
|------|------|-------|---------|
| `_UniqueGroup` | export.py | 1938-1952 | Core data structure |
| `_compute_node_top_module` | export.py | 1955-1986 | Module path extraction (MODIFY) |
| `_normalize_template_key` | export.py | 1989-2000 | Template normalization (MODIFY) |
| `_build_structural_signature` | export.py | 2003-2049 | Signature matching (REUSE) |
| `_detect_unique_groups` | export.py | 2052-2420 | Main detection (REPLACE with hierarchical) |
| `_generate_unique_fn_def` | export.py | 2423-2452 | Function codegen (EXTEND for nesting) |
| `_generate_call_site` | export.py | 2455-2487 | Call site codegen (REUSE) |
| `export_graph_to_python` | export.py | 2490-2744 | Code emission (MODIFY for hierarchy) |
| `export_aten_program` | export.py | 2924-3246 | Main export entry (ADD uniquify options) |
| `_normalize_replay` | kbox_gen.py | 453-507 | Text-based normalization (DEPRECATE?) |
| `detect_unique_groups` | kbox_gen.py | 510-590 | KBox detection (UNIFY with export.py) |
| `test_uniquify.py` | tests/ | 1-482 | Test suite (EXTEND significantly) |

---

## Appendix A: Autoresearch Layer Structure

```
Layer 0 (even — no ve_gate, no value_embeds):
  RMSNorm → residual_lambda_scaling
  → Attention: c_q(Linear) + c_k(Linear) + c_v(Linear) + RoPE(Q) + RoPE(K)
              + QK_RMSNorm + SDPA + c_proj(Linear)
  → MLP: c_fc(Linear) + GELU + c_proj(Linear)
  → residual_add

Layer 1 (odd — has ve_gate AND value_embeds):
  value_embeds_1(Embedding) → add to input    ← EXTRA: separate embedding
  RMSNorm → residual_lambda_scaling
  → Attention: c_q(Linear) + c_k(Linear) + c_v(Linear) + RoPE(Q) + RoPE(K)
              + QK_RMSNorm + ve_gate(sigmoid gate on V) ← EXTRA: 5 ops
              + SDPA + c_proj(Linear)
  → MLP: c_fc(Linear) + GELU + c_proj(Linear)
  → residual_add
```

Key differences between even and odd layers:
1. Odd layers: `value_embeds` embedding lookup + add (happens BEFORE the block, ~2 ops)
2. Odd layers: `ve_gate` on attention values (sigmoid + mm + mul, ~5 ops inside attention)
3. All other ops (RMSNorm, RoPE, SDPA, MLP, residual) are IDENTICAL

So with fuzzy matching:
- `attention()` can have optional `ve_gate_w` — 42 common ops + 5 conditional ops
- `value_embeds` application is outside the block — separate function or inline
- `mlp()` is 100% identical across all 8 layers
- `rope()` is 100% identical across all 16 applications (8 layers × Q,K)

## Appendix B: Name Genericization Strategy

Current algorithm (`_genericize_name`, export.py ~line 2249):
```python
instance_prefix = f"{template_base}_{template_idx_str}_"  # "transformer_h_0_"

def _genericize_name(name: str) -> str:
    if name.startswith(instance_prefix):
        return name[len(instance_prefix):]  # "transformer_h_0_attn_c_q_weight" → "attn_c_q_weight"
    return name
```

For hierarchical extraction, this needs to be level-aware:
- Top-level function: strip "transformer_h_0_" → "attn_c_q_weight"
- Sub-function (attention): strip "transformer_h_0_attn_" → "c_q_weight"
- Sub-sub-function (rope): strip nothing (rope params come from attention, already generic)

**Rule**: The prefix to strip = the template key converted to underscore form + instance index.

## Appendix C: Backward Saved Tensor Mapping

Forward returns: `(buffer_mutations..., real_outputs..., saved_for_backward...)`
Backward receives: `(grad_of_real_outputs..., saved_non_tensors..., saved_tensors...)`

The `saved_for_backward` ↔ `saved_tensors` correspondence is 1:1 (after reordering
non-tensors to front). This mapping is computed in `install.py _AtenGraph.backward()`.

For backward uniquification, we need to:
1. Identify which saved tensors belong to which forward unique group
2. Partition backward ops by which saved tensors they consume
3. Verify the backward partition matches the forward partition structurally

This is feasible because saved tensors from layer 0's attention will be consumed by
backward ops that compute gradients for layer 0's attention weights.
