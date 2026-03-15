"""Test capturing and exporting Triton kernel calls in aten graphs."""

import os
import subprocess
import sys
import tempfile

import pytest
import torch

# Skip if no CUDA or no triton
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    @triton.jit
    def relu_square_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        x = tl.maximum(x, 0.0)
        output = x * x
        tl.store(output_ptr + offsets, output, mask=mask)

    class TritonAddModel(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = torch.nn.Linear(dim, dim)

        def forward(self, x):
            y = self.linear(x)
            output = torch.empty_like(x)
            n = x.numel()
            grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
            add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
            return output


@pytest.mark.skipif(not HAS_TRITON, reason="triton required")
def test_triton_kernel_capture_forward_only():
    """Test that Triton kernel calls appear in the captured forward graph."""
    from torch_graph.export import capture_aten_graphs

    model = TritonAddModel(128).cuda()
    x = torch.randn(4, 128, device='cuda')

    torch._dynamo.reset()
    out, capture = capture_aten_graphs(model, x, run_backward=False)

    gm = capture.forward_graphs[0].graph_module
    triton_nodes = [n for n in gm.graph.nodes
                    if n.op == 'call_function' and 'triton' in str(n.target).lower()]
    assert len(triton_nodes) >= 1, "Expected at least one triton_kernel_wrapper_functional node"


@pytest.mark.skipif(not HAS_TRITON, reason="triton required")
def test_triton_kernel_export_roundtrip():
    """Test that a model with Triton kernels can be exported and re-executed."""
    from torch_graph.export import capture_aten_graphs, export_aten_program

    model = TritonAddModel(128).cuda()
    x = torch.randn(4, 128, device='cuda')

    torch._dynamo.reset()
    out_eager, capture = capture_aten_graphs(model, x, run_backward=False,
                                             record_real_tensors=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "triton_test_aten.py")
        export_aten_program(
            capture,
            output_path=out_path,
            include_test_harness=False,
        )

        # Read and verify the exported file contains expected elements
        with open(out_path) as f:
            content = f.read()

        assert "import triton" in content, "Missing triton import"
        assert "import triton.language as tl" in content, "Missing triton.language import"
        assert "add_kernel" in content, "Missing kernel definition"
        assert "@triton.jit" in content, "Missing @triton.jit decorator"
        assert "tl.load" in content, "Missing kernel body (tl.load)"

        # Verify the exported file is syntactically valid Python
        compile(content, out_path, "exec")

        # Run the exported file directly (not via exec() - Triton's @jit
        # needs inspect.getsource which requires a real file)
        result = subprocess.run(
            [sys.executable, out_path],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"Exported file failed:\n{result.stderr}"


@pytest.mark.skipif(not HAS_TRITON, reason="triton required")
def test_triton_auto_install_roundtrip():
    """Test that auto_install works end-to-end with a Triton kernel model.

    Verifies the full pipeline: capture → export with kernels → load → forward.
    """
    import shutil
    import tempfile

    import torch_graph.auto_install as ai

    torch._dynamo.reset()
    cache_dir = tempfile.mkdtemp(prefix="triton_auto_")
    try:
        ai.configure(
            cache_dir=cache_dir,
            force_recapture=True,
            capture_backward=False,
            verbose=False,
        )
        ai.patch()

        model = TritonAddModel(128).cuda()
        x = torch.randn(4, 128, device='cuda')

        # Get eager reference
        with torch.no_grad():
            ref = model(x)

        # Apply torch.compile (auto_install intercepts)
        compiled = torch.compile(model)

        torch._dynamo.reset()
        with torch.no_grad():
            result = compiled(x)

        # Verify the aten cache file exists and contains Triton kernel definitions
        from pathlib import Path
        cache_files = list(Path(cache_dir).glob("*_aten.py"))
        assert len(cache_files) >= 1, f"No aten files in {cache_dir}"
        content = cache_files[0].read_text()
        assert "import triton" in content, "Aten file missing triton import"
        assert "add_kernel" in content, "Aten file missing kernel definition"

        # Verify output matches eager
        assert torch.allclose(ref, result, atol=1e-5), (
            f"Auto-installed output differs from eager: "
            f"max_diff={( ref - result).abs().max():.2e}"
        )
    finally:
        ai.unpatch()
        shutil.rmtree(cache_dir, ignore_errors=True)
        torch._dynamo.reset()


if __name__ == "__main__":
    test_triton_kernel_capture_forward_only()
    print("\n" + "="*60 + "\n")
    torch._dynamo.reset()
    test_triton_kernel_export_roundtrip()
