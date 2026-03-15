# Inductor Enrichment Refactor

## Goal

Make the existing runtime contract explicit:

1. Capture the authoritative aten graph first.
2. Optionally run a second Inductor compilation.
3. Attach parsed kernel/debug metadata back onto the captured graph.

This keeps the current capabilities while making it clear that Inductor debug
artifacts are an enrichment layer, not a second source of truth for graph
identity.

## Why

Today the runtime already behaves this way when `capture_aten_graphs(...,
triton=True)` is used:

- `export.py` captures the forward/backward aten graphs first.
- `triton.py` then runs a separate Inductor compilation.
- Later code maps the parsed kernels back onto the captured graph.

What is missing is an explicit API and documentation boundary between:

- graph capture
- kernel/debug enrichment

That implicit boundary makes the repo feel like it has two competing graph
loaders, even though the code already treats one as authoritative and the
other as derived.

## Target Architecture

### Authoritative Layer

`AtenCapture`

- owns forward/backward `GraphModule`s
- owns source/module metadata
- owns primal ordering / parameter identity
- owns real tensor recordings
- owns graph-break / fallback behavior

### Enrichment Layer

Inductor debug capture

- owns parsed Triton kernels
- owns extern/native kernel calls
- owns Inductor debug artifacts (`output_code.py`, `fx_graph_readable.py`,
  `fx_graph_transformed.py`, IR dumps)
- owns kernel call sequence

### Attachment Layer

Kernel mapping / grouping

- maps captured graph nodes to kernel calls
- groups nodes by kernel invocation
- exposes kernel source code in graph dumps / exports

## Scope

### In Scope

- Add a public Inductor-debug-oriented API that makes the enrichment role
  explicit.
- Keep the old Triton API as a backward-compatible wrapper.
- Rename internal helpers in `export.py` so the control flow reads as
  capture-then-enrich.
- Add clearer attachment helpers/properties on `AtenCapture`.
- Update architecture docs to match the real layering.

### Out of Scope

- Replacing the live capture path with debug-file parsing.
- Introducing a brand new repo-wide `GraphIR`.
- Rewriting all downstream code to stop using existing `triton_capture` field
  names.
- Changing kernel mapping heuristics.

## Implementation Plan

### Phase 1: Document the Contract

Files:

- `docs/INDUCTOR_ENRICHMENT_REFACTOR.md`
- `docs/ARCHITECTURE.md`

Changes:

- state that live aten capture is authoritative
- state that Inductor debug parsing is optional enrichment
- explain that Triton/extern/native kernel metadata is attached back to the
  captured graph

Success criteria:

- docs no longer read as if `triton.py` is a peer graph loader to `export.py`

### Phase 2: Add Explicit Enrichment APIs

Files:

- `torch_graph/triton.py`
- `torch_graph/__init__.py`

Changes:

- add a public `capture_inductor_debug(...)` function
- add a public `enrich_capture_with_inductor_debug(...)` function
- keep `capture_triton_kernels(...)` as a wrapper for backward compatibility
- update docstrings to say these APIs enrich an existing graph capture

Success criteria:

- callers can explicitly request “Inductor debug enrichment” without routing
  through Triton-specific naming

### Phase 3: Make `AtenCapture` Speak Enrichment Natively

Files:

- `torch_graph/export.py`

Changes:

- add attachment helpers/properties on `AtenCapture` for forward/backward
  kernel enrichment
- rename private helper `_triton_capture_from_model(...)` to
  `_enrich_capture_with_inductor_debug(...)`
- route the existing `triton=True` path through the new enrichment helper

Success criteria:

- internal code reads as: capture graph -> enrich capture
- no behavior regression for existing `triton=True` callers

### Phase 4: Update Consumers and Docs

Files:

- `docs/ARCHITECTURE.md`
- `torch_graph/__init__.py`

Changes:

- update wording from “separate Triton path” to “kernel enrichment layer”
- export the new APIs publicly
- preserve old exports for compatibility

Success criteria:

- public API supports both old and new naming
- docs describe the actual layering used by the code

### Phase 5: Validation

Checks:

- import `torch_graph`
- compile the edited modules
- grep references to ensure old names still resolve

Success criteria:

- refactor is additive and backward compatible

## File-Level Execution Checklist

- [x] Add detailed refactor plan
- [x] Add explicit Inductor debug capture/enrichment APIs
- [x] Update `AtenCapture` attachment semantics
- [x] Update architecture docs
- [x] Run lightweight validation

## Notes

This refactor is intentionally conservative. It clarifies the layering that the
repo already relies on without forcing a large migration of field names or a
new intermediate representation through every module.
