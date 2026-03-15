# Next Phase: Detailed Plan

## Phase 2 Goal

Make the tool production-ready for real workloads: fix the remaining edge cases, add end-to-end tensor verification that goes beyond determinism (compare against the real model), and build the Triton integration into something that enables kernel-level debugging and optimization.

---

## P0: Critical Fixes (do first)

### 2.1 Thread Real Forward Outputs to Backward Inputs

**Why**: The Conv+BN backward NaN issue affects all models with BatchNorm (which is most vision models). The root cause is that backward inputs are random instead of the actual saved tensors from forward.

**Plan**:
1. In `dump_and_compare`, run the forward interpreter first and capture its full output tuple
2. Parse the forward graph's output node to understand which outputs are "saved for backward" vs "user outputs" (aot_autograd convention: user outputs first, then saved tensors)
3. Feed the real saved tensors into the backward interpreter
4. Materialize only the gradient tensors (tangents) as random

**Validation**: Conv+BN backward should go from 15/29 to 29/29.

### 2.2 End-to-End Model Comparison (Not Just Determinism)

**Why**: Currently we verify that running the graph twice produces the same result (determinism). But we don't verify that the graph produces the SAME result as the original PyTorch model. This is the actual guarantee users need.

**Plan**:
1. Run the real model with real inputs → get reference output tensor
2. Run the aten graph with the same real inputs (using `_build_real_forward_inputs`) → get graph output
3. Compare the two: `torch.allclose(real_output, graph_output)`
4. For backward: run `loss.backward()` on the real model, capture `.grad` for each parameter, then compare against the backward graph's outputs

**This is the "no cheating" test**: we start with ONLY the model and its inputs, extract the graph, and verify the graph reproduces the model's exact behavior.

**Implementation**:
```python
def verify_against_model(model, *args, atol=1e-5, rtol=1e-4):
    # 1. Real model forward
    real_output = model(*args)
    
    # 2. Capture aten graph
    _, capture = capture_aten_graphs(model, *args, run_backward=False)
    
    # 3. Run graph with real inputs
    gm = capture.forward_graphs[0].graph_module
    inputs = _build_real_forward_inputs(model, args, gm)
    graph_output = gm(*inputs)  # or via interpreter
    
    # 4. Compare (extract user outputs from the full output tuple)
    n_user_outputs = ...  # from graph structure
    assert torch.allclose(real_output, graph_output[:n_user_outputs])
```

### 2.3 Fix FakeTensor Detection in Export

**Why**: The `_safe_copy_inputs` fix uses `type(inp).__name__ == "FakeTensor"` which is fragile. A proper check should handle all FakeTensor subclasses and meta tensors.

**Plan**: Use `torch._subclasses.fake_tensor.is_fake(t)` if available, or check `t.device.type == 'meta'` as a secondary signal.

---

## P1: Triton Integration Depth

### 2.4 Bidirectional Aten↔Triton Mapping

**Why**: Currently we have aten→kernel direction. Users also need kernel→aten: "this kernel is slow, which aten ops are in it, which source code line generated them?"

**Plan**:
1. Build a `KernelProfile` dataclass combining: kernel name, execution time (from profiler), aten ops, source lines
2. Sort by execution time, show the critical path
3. In the HTML visualizer, color aten nodes by which kernel they belong to
4. Click a kernel → highlight all its aten ops in the graph

### 2.5 Triton Kernel Performance Profiling

**Plan**:
1. Run the model under `torch.profiler.profile()` to get per-kernel timing
2. Match profiler kernel names to our captured kernel names
3. Annotate each `TritonKernel` with its wall-clock time
4. Generate a report: "Top 5 slowest kernels, their aten ops, and source lines"

### 2.6 Triton Kernel for Backward

**Why**: Currently we only capture Triton kernels for the forward pass. Backward is typically 2-3x slower and more interesting for optimization.

**Plan**: Run `torch.compile` with backward too:
```python
compiled = torch.compile(model, backend="inductor")
out = compiled(x)
loss = out.sum()
loss.backward()  # This triggers backward kernel compilation
```

Parse the backward kernels the same way. The inductor output code will include both forward and backward kernels.

---

## P2: Tensor Verification Depth

### 2.7 Cross-Dtype Verification

**Why**: Many real workloads use mixed precision (fp16/bf16 forward, fp32 gradients). The aten graph should reproduce this behavior.

**Plan**:
1. Add `dtype` parameter to `dump_and_compare`: `dtype=torch.float16`
2. Cast model and inputs to the target dtype
3. Run comparison with dtype-appropriate tolerances (fp16 needs `atol=1e-3`)
4. Report which ops introduce the most numerical error

### 2.8 Tensor Provenance Tracking

**Why**: When a mismatch is found, users need to trace back to the root cause: which op first diverged?

**Plan**: Instead of comparing at the end, compare after each node in a single combined interpreter run. The first node where `actual != reference` is the root cause. Report the op, its inputs, and the magnitude of divergence.

### 2.9 Gradient Verification Against Autograd

**Why**: The ultimate test - verify that the exported backward graph produces the same gradients as PyTorch's autograd.

**Plan**:
1. Run real model forward + backward, capture `param.grad` for all parameters
2. Run forward aten graph → get saved tensors
3. Run backward aten graph with saved tensors + tangent → get gradient outputs
4. Match backward outputs to parameter gradients (by shape)
5. Compare: `allclose(param.grad, backward_output)` for each parameter

---

## P3: User Experience

### 2.10 `torch_graph.explain(model, x)`

A single entry point that runs everything and produces a comprehensive report:

```python
from torch_graph import explain

report = explain(model, x)
# Prints:
#   Model: NanoGPT (24,576 params)
#   Forward: 168 aten ops, 241 tensors
#   Backward: 242 aten ops, 279 tensors
#   Triton: 12 fused kernels + 13 cuBLAS calls
#   Verification: ✓ forward matches model, ✓ backward matches autograd
#   Top 3 expensive kernels:
#     1. triton_per_fused_..._softmax_... (45μs) — self.blocks.0.attn softmax
#     2. extern_kernels.bmm (38μs) — self.blocks.0.attn Q@K
#     3. triton_per_fused_..._layer_norm_... (22μs) — self.blocks.0.ln_1
#   Exported to: outputs/nanogpt_aten.py

report.save_html("report.html")  # Full interactive report
```

### 2.11 Watch Mode

For iterative development: watch a model file for changes, automatically re-extract and diff the graph.

```bash
python3 -m torch_graph --watch model.py --class GPT --output-dir outputs/
# Re-runs on file change, shows diff of aten ops
```

### 2.12 VS Code Extension

Syntax highlighting for exported aten programs: color aten ops by category, make source annotations clickable (jump to original source), show tensor shapes inline.

---

## P4: Scale

### 2.13 Large Model Support

**Problem**: Models with >1B parameters can't be fully materialized for tensor dumping.

**Plan**:
- Support `device_map="auto"` for multi-GPU models
- Implement per-layer verification (verify one layer at a time, freeing memory between layers)
- Support `torch.distributed` for tensor-parallel models

### 2.14 Streaming/Checkpoint-Based Verification

For very long sequences or large batch sizes:
- Verify in chunks (first N tokens, then next N, etc.)
- Checkpoint intermediate states to disk
- Resume verification from checkpoint

### 2.15 Integration with Training Loops

Hook into training to continuously verify graph correctness:
```python
model = NanoGPT()
verifier = torch_graph.Verifier(model)

for step in range(1000):
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    if step % 100 == 0:
        verifier.check()  # Verify graph hasn't drifted
```

---

## Implementation Order

```
Week 1: P0 (fixes)
  2.1 Real forward→backward threading
  2.2 Model comparison (not just determinism)
  2.3 FakeTensor detection cleanup

Week 2: P1 (Triton depth)
  2.4 Bidirectional aten↔Triton mapping
  2.5 Kernel performance profiling
  2.6 Backward Triton capture

Week 3: P2 (verification depth)
  2.7 Cross-dtype verification
  2.8 Tensor provenance tracking
  2.9 Gradient verification against autograd

Week 4: P3 (UX)
  2.10 explain() one-liner
  2.11 Watch mode
  2.12 VS Code extension (stretch)
```

The P4 items (scale) are future work that depends on having real large-model users.
