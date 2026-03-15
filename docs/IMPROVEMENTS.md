# Potential Improvements

## Robustness

### 1. Fix Conv+BN Backward Tensor Verification

**Problem**: `native_batch_norm_backward` produces NaN when forward saved tensors are random (not from a real forward pass). This is because the backward needs the running mean/variance from a real forward, not random noise.

**Fix**: For backward verification, feed the forward graph's actual outputs as backward inputs instead of materializing random tensors. This requires threading the forward interpreter's output tuple through to `_build_real_backward_inputs`. The plumbing exists but the matching between forward outputs and backward inputs is imprecise - the backward expects the forward's *saved tensors* (a subset of outputs), not all outputs.

**Deeper fix**: Run the full forward first, capture the actual saved_tensors tuple, and use those as backward inputs. This requires understanding the `aot_autograd` convention for which forward outputs are "saved for backward" vs "actual outputs".

### 2. Primal Matching for Ambiguous Shapes

**Problem**: When two parameters have the same shape (e.g. two `nn.Linear(128, 128)` layers), shape matching can mismatch them.

**Current mitigation**: Consumer-based disambiguation checks which module each primal feeds into. But this fails when the consumer metadata is missing or ambiguous.

**Better approach**: Record the actual primal-to-parameter mapping during `aot_autograd` by instrumenting `aot_module_simplified`. The function already knows the mapping internally - we just need to extract it. Alternatively, use the parameter access order during tracing (which `aot_autograd` follows) to build the mapping deterministically.

### 3. Dynamic Shapes

**Problem**: All captured graphs assume static shapes. If the model is used with different input sizes, a new graph is compiled.

**Fix**: Support `torch.compile(dynamic=True)` in the capture pipeline. The FX graph will contain symbolic shapes (`s0`, `s1`) instead of concrete sizes. The export would need to preserve these as function parameters rather than hard-coded values.

### 4. Multi-GPU / Distributed

**Problem**: Tensor verification assumes single-device execution. Distributed models split tensors across devices, and collective ops (all_reduce, etc.) aren't handled.

**Fix**: Detect distributed ops in the graph and either skip verification for those nodes or run verification per-rank.

---

## User Friendliness

### 5. Interactive Notebook Widget

**Problem**: The HTML visualizer requires saving to a file and opening in a browser. Awkward in Jupyter notebooks.

**Improvement**: Create an IPython display widget that renders the graph inline in a notebook cell. Use `IPython.display.HTML()` for the interactive canvas, or create a simpler SVG-based renderer for static display.

### 6. Diff View Between Two Models

**Problem**: No way to compare the aten graphs of two models (e.g. before/after an optimization, or two different architectures).

**Improvement**: `diff_graphs(graph1, graph2)` that highlights added/removed/changed ops. Could output a side-by-side HTML view or a colored text diff.

### 7. Op-Level Profiling Annotations

**Problem**: The captured graph shows structure but not performance. Users can't tell which ops are bottlenecks.

**Improvement**: Integrate with `torch.profiler` to annotate each aten op with its wall-clock time and memory usage. The Triton module already maps ops to kernels - adding kernel execution time from the profiler would create a complete performance picture.

### 8. Better Backward Source Annotations

**Problem**: Backward ops have `fwd_nn_module_stack` and `fwd_source_fn_stack` metadata, but the annotation quality is lower than forward. Many backward ops show up as "grad: self (NanoGPT)" without the specific layer.

**Fix**: In Phase 1, also trace the backward (by running `.backward()`) and capture the backward's source metadata. Cross-reference with the forward's source map to produce annotations like "grad of self.blocks.0.attn.c_attn (Linear)".

### 9. One-Line Summary Mode

**Problem**: The full tensor comparison report is verbose. For quick validation, users just want "all good" or "3 mismatches at nodes X, Y, Z".

**Improvement**: Add a `dump_and_compare(..., verbose=False)` mode that prints a single line like:
```
✓ forward: 241/241 matched | backward: 279/279 matched
```

### 10. Exported Program Readability

**Problem**: Exported `.py` files for large models (NanoGPT: 822 lines) are hard to navigate.

**Improvements**:
- **Collapsible sections**: Export as a Jupyter notebook with one cell per source group
- **Table of contents**: Add a comment block at the top listing all source groups with line numbers
- **Named intermediates**: Instead of `t_7`, `mm_3`, use names derived from the source: `fc1_weight_transposed`, `fc1_grad_weight`
- **Hyperlinked HTML version**: Export as HTML with syntax highlighting and clickable source annotations

### 11. CLI Progress Bars

**Problem**: Long-running captures on large models give no progress feedback.

**Improvement**: Use `tqdm` for progress bars during graph capture, tensor tracing, and comparison.

---

## Performance

### 12. Lazy Tensor Materialization

**Problem**: `trace_all_intermediates` clones every tensor at every node. For NanoGPT backward (279 tensors), this uses significant memory.

**Improvement**: Option to only record tensor metadata (shape, dtype, hash) for non-selected tensors, and only fully materialize tensors the user asks for. This would reduce memory from O(sum of all tensor sizes) to O(number of tensors).

### 13. Parallel Triton Capture

**Problem**: Triton capture clears the inductor cache and recompiles from scratch every time.

**Improvement**: Check for existing debug artifacts in the cache before clearing. Only clear and recompile if artifacts are missing. Track the cache key to detect when the model has changed.

### 14. Streaming Export for Large Models

**Problem**: `export_aten_program` builds the entire file in a StringIO buffer, then writes it all at once. For very large models, this could use significant memory.

**Improvement**: Stream directly to the output file. Also, compress the weights file with `torch.save(..., _use_new_zipfile_serialization=True)`.

---

## New Features

### 15. Tensor Value Inspection in Visualizer

**Problem**: The HTML visualizer shows shapes and dtypes but not actual values.

**Improvement**: After running `trace_all_intermediates`, inject the tensor statistics (mean, std, min, max, % zeros, % NaN) into the visualization data. Show these in the node inspector sidebar.

### 16. Automatic Numerical Stability Analysis

**Problem**: Users can't easily find ops that might cause numerical issues.

**Improvement**: After tracing, flag tensors with very large/small values, NaN/Inf, or high variance. Highlight these nodes in the visualizer. Cross-reference with the source code to identify the problematic PyTorch layer.

### 17. Graph Simplification

**Problem**: Aten graphs include many "plumbing" ops (view, t, permute, expand) that clutter the view.

**Improvement**: `simplify_graph()` that collapses chains of view/reshape ops into a single annotated edge, and fuses transpose+matmul into "matmul with transposed input". Show the simplified view by default, with an option to expand.

### 18. Export to Other Formats

**Problem**: Only exports to Python. Some users want ONNX, TorchScript, or C++.

**Improvement**: Add `export_to_onnx()`, `export_to_torchscript()`, and `export_to_cpp()` backends. The FX graph has all the information needed for these conversions.

### 19. Triton Kernel Editing

**Problem**: Users can see the Triton kernels but can't easily modify and test them.

**Improvement**: `edit_triton_kernel(capture, kernel_name)` that:
1. Extracts the kernel source
2. Opens it in the user's editor
3. Recompiles the modified kernel
4. Runs the model with the modified kernel
5. Compares outputs against the original

This would enable Triton kernel development directly from the captured graph.

### 20. Automatic Test Generation

**Problem**: The test harness in exported files is basic.

**Improvement**: Generate comprehensive tests that:
- Verify each intermediate tensor against the reference dump
- Test numerical stability with different dtypes (fp16, bf16, fp32)
- Test with different input sizes (if dynamic shapes are supported)
- Benchmark each kernel's execution time
