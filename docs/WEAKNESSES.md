# Known Weaknesses & Limitations

Comprehensive list of all known weaknesses, sorted by severity.

## Critical

### 1. Multi-fragment models cannot be auto-installed
**Status**: Known limitation, not fixed
**Impact**: Models with graph breaks (explicit `torch._dynamo.graph_break()`, data-dependent control flow, HF models with `DynamicCache`) export successfully but fail to load back via `_load_variant()` with `ValueError: Cannot determine parameter mapping`.

**Root cause**: `_export_multi_fragment()` generates a different file format than `_load_variant` / `_parse_param_paths` expects. The multi-fragment file has per-fragment parameter mappings but no single "Parameter mapping:" section.

**Workaround**: Use `test_models.py` validation (syntax check) rather than full auto-install for multi-fragment models. HF GPT2 passes `test_models.py` but cannot be installed back.

**Fix complexity**: High — requires multi-fragment install support: chaining fragment forwards, reversing backward ordering, per-fragment input/output plumbing.

## High

### 2. Backward saved tensor ordering is implicit
**Impact**: The backward function expects saved values in a specific order: non-tensor values (SymInts) first, then tensor values. This differs from the interleaved order in forward's return tuple. If a user adds a new saved tensor in the wrong position, backward silently gets wrong inputs producing wrong gradients.

**Mitigation**: The reordering logic in `install.py` `_AtenGraph.backward()` handles this correctly, and the return-count validation now catches changed tuple lengths. But the ordering within the tuple is unchecked.

### 3. `num_mutations` miscounting silently corrupts BatchNorm models
**Impact**: If `_compute_num_mutations()` gets the count wrong, the forward output layout splits at the wrong position. Buffer mutations are treated as real outputs, or vice versa. Models with BatchNorm silently produce wrong results.

**Mitigation**: Tested for 0 and 2 mutations. No test for 3+ mutations or for intentionally wrong counts.

### 4. SymInt concretization is irreversible and order-dependent
**Impact**: Calling `int()` on a SymInt concretizes ALL related symbolic dimensions globally. In `export_aten_program`, forward/backward code must be generated BEFORE the weight-building loop. Getting this wrong produces broken dynamic-shape graphs with no obvious error.

**Mitigation**: Documented in CLAUDE.md. Code is correctly ordered. But any refactoring that moves the weight-building loop earlier will silently break dynamic shapes.

## Medium

### 5. ~~Corrupt aten files crash instead of re-capturing~~
**Status**: FIXED
Corrupt/truncated cache `.py` files now log a warning and trigger re-capture automatically. User-modified files still raise errors (intentional — don't silently discard user edits).

### 6. ~~Extra return values in edited aten files silently corrupt output~~
**Status**: FIXED
Added return-tuple-length validation in `_AtenGraph.forward()`. Raises `RuntimeError` with clear message showing expected vs actual count breakdown.

### 7. `offload_saved` increases memory on small models
**Impact**: The CPU<->GPU transfer overhead and bookkeeping exceeds savings for small models. In testing, a small model used 22% MORE memory with `offload_saved=True`.

**Mitigation**: Only use for large models where activation memory dominates (documented but easy to misuse).

### 8. Flash Attention 3 requires Hopper GPUs specifically
**Impact**: FA3 tests and nanochat/autoresearch workflows fail on non-Hopper GPUs (including newer Blackwell/B200). CUDA error: `no kernel image is available for execution on the device`.

**Mitigation**: Tests skip correctly. But users running on non-Hopper GPUs get unhelpful CUDA errors from FA3, not from torch-graph.

### 9. Device fixup regex is fragile
**Impact**: The regex that adds `.to(_device)` to inline tensor constants in `_export_multi_fragment()` must NOT match aten ops. Source annotations in comments span newlines; a greedy `[^)]*` pattern matches into the next line's aten call, producing `device=_device.to(_device)`.

**Mitigation**: Documented in CLAUDE.md. Tested indirectly via HF GPT2 in `test_models.py`. No isolated regression test.

## Low

### 10. Error messages for truncated aten files are unhelpful
**Status**: Partially fixed (corrupt files now auto-recover for unmodified caches). User-modified files that are truncated still produce `IndexError: tuple index out of range` from `_assemble_inputs()` without indicating which file is corrupt.

### 11. FakeTensor detection requires `is_fake()` everywhere
**Impact**: In PyTorch 2.10+, `untyped_storage()` succeeds on FakeTensors. Calling `.data_ptr()`, `.tolist()`, `.numpy()` on a FakeTensor during tracing produces wrong results or crashes. Must always use `_utils.is_fake(t)` first.

**Mitigation**: Documented in CLAUDE.md. Used consistently in current code.

### 12. Custom ops tracking has zero test coverage
**Impact**: `custom_ops.py` intercepts custom op registrations for export. Used by `export.py` but has no tests. A model with C++ CUDA extensions would produce an exported `.py` missing required imports.

**Mitigation**: The modded-nanogpt capture script exercises this path. Failure mode is a clear ImportError at load time, not silent corruption.

### 13. LBFGS and closure-based optimizers can't be replayed
**Impact**: `capture_optimizer_aten()` produces "no graphs" for LBFGS because its `step()` takes a closure. The system handles this gracefully (runs eager), but the user gets no captured optimizer aten file.

**Mitigation**: Expected limitation. Standard optimizers (SGD, AdamW, MuonAdamW) all work.

### 14. FX-level APIs (capture.py, inspector.py, editor.py) are untested
**Impact**: Older FX-level APIs exported in `__init__.py` have zero test coverage. They're not used by the aten pipeline.

**Mitigation**: These should either be tested or removed from `__all__`.

### 15. No distributed training support
**Impact**: DDP/FSDP models are not tested and likely fail (parameter sharding, communication ops).

**Mitigation**: Single-GPU only is the current scope.

## Strengths (verified by stress testing)

- **Bit-identical capture**: SGD replay is bit-identical to eager over 20 steps (zero drift)
- **Dynamic shapes are robust**: Captured at batch=4, verified at batch=1024 with no issues; 3 symbolic dims work correctly
- **Concurrent model compilation**: Two different models in same process work without conflicts
- **Cache collision resistance**: Same class name with different architectures gets distinct cache entries
- **Memory scaling**: 50-layer deep model (202 params) captures successfully, generates 264KB aten file
- **Large tensor handling**: 64MB model with 257MB peak forward+backward works correctly
- **Custom @torch.compile optimizers**: Detected and replayed correctly
- **BatchNorm buffer mutations**: Work correctly in standalone mode
- **Corrupt cache recovery**: Missing .meta or corrupt .meta triggers re-capture
