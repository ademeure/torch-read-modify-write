"""Tests for inline CUDA kernel compilation, caching, and integration with uniquified aten exports."""

import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_graph.cuda_inline import (
    load_cuda,
    load_cuda_precompiled,
    clear_cache,
    cuda_kernel_template,
    cuda_wrapper_template,
)

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Test models (same as test_uniquify.py)
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x + self.fc2(F.relu(self.fc1(x))))


class ResNetLike(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(3)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Core load_cuda tests
# ---------------------------------------------------------------------------


class TestLoadCuda:
    """Test the load_cuda() compilation and caching utility."""

    def setup_method(self):
        clear_cache()

    def test_simple_elementwise_kernel(self):
        """Compile a simple CUDA kernel that doubles a tensor."""
        cuda_src = r"""
        #include <torch/extension.h>

        torch::Tensor double_tensor(torch::Tensor x) {
            return x * 2;
        }
        """
        mod = load_cuda("double", cuda_src, ["double_tensor"])
        x = torch.randn(4, 8, device="cuda")
        result = mod.double_tensor(x)
        assert torch.allclose(result, x * 2)

    def test_multi_arg_kernel(self):
        """Compile a kernel that takes multiple tensor arguments."""
        cuda_src = r"""
        #include <torch/extension.h>

        torch::Tensor fused_add_mul(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
            return (a + b) * c;
        }
        """
        mod = load_cuda("add_mul", cuda_src, ["fused_add_mul"])
        a = torch.randn(4, 8, device="cuda")
        b = torch.randn(4, 8, device="cuda")
        c = torch.randn(4, 8, device="cuda")
        result = mod.fused_add_mul(a, b, c)
        assert torch.allclose(result, (a + b) * c)

    def test_multi_return_kernel(self):
        """Compile a kernel that returns multiple tensors."""
        cuda_src = r"""
        #include <torch/extension.h>

        std::tuple<torch::Tensor, torch::Tensor> split_add(
            torch::Tensor x, torch::Tensor y
        ) {
            return std::make_tuple(x + y, x - y);
        }
        """
        mod = load_cuda("split", cuda_src, ["split_add"])
        x = torch.randn(4, 8, device="cuda")
        y = torch.randn(4, 8, device="cuda")
        a, b = mod.split_add(x, y)
        assert torch.allclose(a, x + y)
        assert torch.allclose(b, x - y)

    def test_caching_same_source(self):
        """Same source code should return the cached module (no recompile)."""
        cuda_src = r"""
        #include <torch/extension.h>

        torch::Tensor identity(torch::Tensor x) {
            return x.clone();
        }
        """
        mod1 = load_cuda("cache_test", cuda_src, ["identity"])
        mod2 = load_cuda("cache_test", cuda_src, ["identity"])
        assert mod1 is mod2

    def test_caching_different_source(self):
        """Different source code should compile a new module."""
        src1 = r"""
        #include <torch/extension.h>
        torch::Tensor op_v1(torch::Tensor x) { return x * 2; }
        """
        src2 = r"""
        #include <torch/extension.h>
        torch::Tensor op_v2(torch::Tensor x) { return x * 3; }
        """
        mod1 = load_cuda("version_test", src1, ["op_v1"])
        mod2 = load_cuda("version_test", src2, ["op_v2"])
        assert mod1 is not mod2
        x = torch.randn(4, device="cuda")
        assert torch.allclose(mod1.op_v1(x), x * 2)
        assert torch.allclose(mod2.op_v2(x), x * 3)

    def test_half_precision(self):
        """Kernel should work with float16 tensors."""
        cuda_src = r"""
        #include <torch/extension.h>

        torch::Tensor scale(torch::Tensor x) {
            return x * 0.5;
        }
        """
        mod = load_cuda("half_test", cuda_src, ["scale"])
        x = torch.randn(4, 8, device="cuda", dtype=torch.float16)
        result = mod.scale(x)
        assert result.dtype == torch.float16
        assert torch.allclose(result.float(), (x * 0.5).float(), atol=1e-3)

    def test_bfloat16(self):
        """Kernel should work with bfloat16 tensors."""
        cuda_src = r"""
        #include <torch/extension.h>

        torch::Tensor scale_bf(torch::Tensor x) {
            return x * 2.0;
        }
        """
        mod = load_cuda("bf16_test", cuda_src, ["scale_bf"])
        x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        result = mod.scale_bf(x)
        assert result.dtype == torch.bfloat16
        assert torch.allclose(result.float(), (x * 2.0).float(), atol=1e-2)

    def test_raw_cuda_kernel(self):
        """Compile an actual CUDA kernel (not just ATen C++ ops)."""
        cuda_src = r"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void relu_kernel(const float* x, float* out, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                out[idx] = x[idx] > 0 ? x[idx] : 0;
            }
        }

        torch::Tensor cuda_relu(torch::Tensor x) {
            auto out = torch::empty_like(x);
            int n = x.numel();
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            relu_kernel<<<blocks, threads>>>(
                x.data_ptr<float>(), out.data_ptr<float>(), n
            );
            return out;
        }
        """
        mod = load_cuda("raw_relu", cuda_src, ["cuda_relu"])
        x = torch.randn(1024, device="cuda")
        result = mod.cuda_relu(x)
        expected = torch.relu(x)
        assert torch.allclose(result, expected)

    def test_clear_cache(self):
        """clear_cache() should evict the in-memory cache entry."""
        cuda_src = r"""
        #include <torch/extension.h>
        torch::Tensor pass_through(torch::Tensor x) { return x.clone(); }
        """
        mod1 = load_cuda("clear_test", cuda_src, ["pass_through"])
        clear_cache()
        # After clearing, the next call goes through load_inline again
        # (which may return the same .so from disk — that's fine)
        mod2 = load_cuda("clear_test", cuda_src, ["pass_through"])
        # Both should work correctly regardless
        x = torch.randn(4, device="cuda")
        assert torch.allclose(mod1.pass_through(x), x)
        assert torch.allclose(mod2.pass_through(x), x)


# ---------------------------------------------------------------------------
# Template generation tests
# ---------------------------------------------------------------------------


class TestTemplateGeneration:
    """Test CUDA kernel template and wrapper generation."""

    def test_single_return_template(self):
        """Template for a function with one return value."""
        params = [
            {"name": "x", "annotation": "float32[B, 16]"},
            {"name": "weight", "annotation": "float32[16, 16]"},
        ]
        returns = [{"name": "output", "annotation": "float32[B, 16]"}]
        template = cuda_kernel_template("my_layer", params, returns)

        assert "#include <torch/extension.h>" in template
        assert "torch::Tensor fused_my_layer(" in template
        assert "torch::Tensor x" in template
        assert "torch::Tensor weight" in template
        assert "// float32[B, 16]" in template
        assert "return output;" in template

    def test_multi_return_template(self):
        """Template for a function with multiple return values."""
        params = [{"name": "x", "annotation": "float32[4, 16]"}]
        returns = [
            {"name": "out1", "annotation": "float32[4, 16]"},
            {"name": "out2", "annotation": "float32[4, 16]"},
        ]
        template = cuda_kernel_template("split_op", params, returns)

        assert "std::tuple<torch::Tensor, torch::Tensor>" in template
        assert "std::make_tuple(out_0, out_1)" in template

    def test_wrapper_template(self):
        """Test the Python wrapper generation."""
        params = [
            {"name": "x", "annotation": "float32[B, 16]"},
            {"name": "weight", "annotation": "float32[16, 16]"},
        ]
        returns = [{"name": "output", "annotation": "float32[B, 16]"}]
        wrapper = cuda_wrapper_template("my_layer", params, returns)

        assert "_my_layer_mod = None" in wrapper
        assert "def my_layer(" in wrapper
        assert "load_cuda(" in wrapper
        assert 'fused_my_layer' in wrapper
        assert "return (_my_layer_mod.fused_my_layer(x, weight),)" in wrapper


# ---------------------------------------------------------------------------
# Integration: replace uniquified function with CUDA kernel
# ---------------------------------------------------------------------------


class TestCudaReplacesUniqueFunction:
    """Test replacing a uniquified aten function with a CUDA kernel."""

    def setup_method(self):
        clear_cache()
        torch._dynamo.reset()

    def test_replace_residual_block_with_cuda(self):
        """Replace the aten ops in a ResidualBlock with an ATen C++ kernel.

        This is the primary end-to-end test: capture model → uniquify →
        replace uniquified function with CUDA → verify correct output.
        """
        from torch_graph.export import (
            capture_aten_graphs,
            export_graph_to_python,
            _UniqueGroup,
        )

        torch.manual_seed(42)
        model = ResNetLike(dim=16).cuda()
        x = torch.randn(4, 16, device="cuda")

        _, capture = capture_aten_graphs(model, x, run_backward=False, record_real_tensors=True)
        fg = capture.forward_graphs[0]
        real_inputs = capture.forward_real_inputs

        # Get the uniquified code AND the group metadata
        unique_fn_defs: list[str] = []
        unique_groups: list[_UniqueGroup] = []
        code_uniq = export_graph_to_python(
            fg.graph_module,
            fn_name="forward_uniq",
            source_map=capture.source_map,
            named_intermediates=True,
            uniquify=True,
            _unique_fn_defs=unique_fn_defs,
            _unique_groups=unique_groups,
        )

        if not unique_fn_defs or not unique_groups:
            pytest.skip("No uniquified functions detected")

        # Also get plain (non-uniquified) code for reference
        code_plain = export_graph_to_python(
            fg.graph_module,
            fn_name="forward_ref",
            source_map=capture.source_map,
            named_intermediates=True,
            uniquify=False,
        )

        # Execute the plain code to get reference output
        import operator
        ns_ref = {"torch": torch, "aten": torch.ops.aten, "operator": operator}
        exec(code_plain, ns_ref)
        ref_output = ns_ref["forward_ref"](*real_inputs)

        # Use the group metadata to get the actual parameter names
        group = unique_groups[0]
        fn_name = group.fn_name
        param_names = [p["name"] for p in group.params]

        # The ResidualBlock uniquified function takes:
        #   fc1_weight, fc1_bias, <input>, fc2_weight, fc2_bias, ln_weight, ln_bias
        # We need to identify which param is the "input" (the one that varies per call)
        # and which are weights. The CUDA kernel must match this signature.

        # Build a CUDA kernel that matches the uniquified function's parameter order.
        # The kernel uses ATen C++ ops — same math as the Python aten ops.
        param_decls = ",\n            ".join(f"torch::Tensor {p}" for p in param_names)
        param_args = ", ".join(param_names)

        # Find which param is the "input" (non-weight) — it's the one NOT named
        # like a weight/bias. For ResidualBlock: fc1_weight, fc1_bias, <input>,
        # fc2_weight, fc2_bias, ln_weight, ln_bias
        # We'll build the kernel to match the exact parameter order.
        cuda_src = f"""
        #include <torch/extension.h>

        torch::Tensor fused_block(
            {param_decls}
        ) {{
            // Identify input vs weight params by name convention
            // For ResidualBlock: params are fc1_weight, fc1_bias, input, fc2_weight, fc2_bias, ln_weight, ln_bias
            // The third param is the input (activation from previous layer)
            auto fc1 = torch::addmm({param_names[1]}, {param_names[2]}, {param_names[0]}.t());
            auto h = torch::relu(fc1);
            auto fc2 = torch::addmm({param_names[4]}, h, {param_names[3]}.t());
            auto residual = {param_names[2]} + fc2;
            auto ln = std::get<0>(
                torch::native_layer_norm(residual, {{residual.size(-1)}}, {param_names[5]}, {param_names[6]}, 1e-5)
            );
            return ln;
        }}
        """

        cuda_mod = load_cuda("residual_e2e", cuda_src, ["fused_block"])

        # Create a replacement function that calls the CUDA kernel
        replacement_code = f"""
def {fn_name}({', '.join(param_names)}):
    return (cuda_mod.fused_block({', '.join(param_names)}),)
"""
        ns_cuda = {
            "torch": torch,
            "aten": torch.ops.aten,
            "operator": operator,
            "cuda_mod": cuda_mod,
        }
        exec(replacement_code + "\n" + code_uniq, ns_cuda)
        cuda_output = ns_cuda["forward_uniq"](*real_inputs)

        # Verify outputs match (may not be bit-identical due to ATen C++ vs Python dispatch)
        assert torch.allclose(ref_output, cuda_output, atol=1e-5), \
            f"CUDA kernel output differs. Max diff: {(ref_output - cuda_output).abs().max().item()}"


# ---------------------------------------------------------------------------
# Export integration: CUDA stubs in generated file
# ---------------------------------------------------------------------------


class TestExportCudaStubs:
    """Test that export_aten_program generates CUDA kernel stubs."""

    def setup_method(self):
        torch._dynamo.reset()

    def test_cuda_stubs_emitted(self, tmp_path):
        """export_aten_program with emit_cuda_stubs=True should include CUDA section."""
        from torch_graph.export import capture_aten_graphs, export_aten_program

        torch.manual_seed(42)
        model = ResNetLike(dim=16).cuda()
        x = torch.randn(4, 16, device="cuda")

        _, capture = capture_aten_graphs(model, x, run_backward=False)

        out_path = str(tmp_path / "test_aten.py")
        export_aten_program(
            capture, out_path,
            include_test_harness=False,
            named_intermediates=True,
            skip_pt=True,
            uniquify=True,
            emit_cuda_stubs=True,
        )

        content = (tmp_path / "test_aten.py").read_text()

        # Should have the CUDA KERNEL SOURCES section
        assert "CUDA KERNEL SOURCES" in content
        assert "from torch_graph.cuda_inline import load_cuda" in content
        assert "#include <torch/extension.h>" in content
        assert "fused_" in content
        assert "load_cuda(" in content

    def test_no_cuda_stubs_by_default(self, tmp_path):
        """emit_cuda_stubs=False (default) should not include CUDA section."""
        from torch_graph.export import capture_aten_graphs, export_aten_program

        torch.manual_seed(42)
        model = ResNetLike(dim=16).cuda()
        x = torch.randn(4, 16, device="cuda")

        _, capture = capture_aten_graphs(model, x, run_backward=False)

        out_path = str(tmp_path / "test_aten.py")
        export_aten_program(
            capture, out_path,
            include_test_harness=False,
            named_intermediates=True,
            skip_pt=True,
            uniquify=True,
            emit_cuda_stubs=False,
        )

        content = (tmp_path / "test_aten.py").read_text()
        assert "CUDA KERNEL SOURCES" not in content

    def test_cuda_stubs_without_uniquify(self, tmp_path):
        """emit_cuda_stubs with uniquify=False should not crash or emit stubs."""
        from torch_graph.export import capture_aten_graphs, export_aten_program

        torch.manual_seed(42)
        model = ResNetLike(dim=16).cuda()
        x = torch.randn(4, 16, device="cuda")

        _, capture = capture_aten_graphs(model, x, run_backward=False)

        out_path = str(tmp_path / "test_aten.py")
        export_aten_program(
            capture, out_path,
            include_test_harness=False,
            named_intermediates=True,
            skip_pt=True,
            uniquify=False,
            emit_cuda_stubs=True,
        )

        content = (tmp_path / "test_aten.py").read_text()
        # No uniquified functions = no CUDA stubs
        assert "CUDA KERNEL SOURCES" not in content


# ---------------------------------------------------------------------------
# End-to-end: load_cuda in the install path
# ---------------------------------------------------------------------------


class TestInstallPathIntegration:
    """Test that CUDA kernels work through the auto_install load path."""

    def setup_method(self):
        clear_cache()
        torch._dynamo.reset()

    def test_load_cuda_in_exec_context(self):
        """Verify load_cuda works when called from exec() (as _load_aten_module does)."""
        cuda_src = r"""
        #include <torch/extension.h>

        torch::Tensor add_one(torch::Tensor x) {
            return x + 1;
        }
        """
        # Simulate how _load_aten_module loads .py files: via exec()
        code = textwrap.dedent("""
            from torch_graph.cuda_inline import load_cuda

            _ADD_CUDA = r'''
            #include <torch/extension.h>
            torch::Tensor add_one(torch::Tensor x) { return x + 1; }
            '''

            _mod = None

            def my_forward(x):
                global _mod
                if _mod is None:
                    _mod = load_cuda("exec_test", _ADD_CUDA, ["add_one"])
                return _mod.add_one(x)
        """)

        ns = {}
        exec(code, ns)

        x = torch.randn(4, device="cuda")
        result = ns["my_forward"](x)
        assert torch.allclose(result, x + 1)

        # Second call should use cached module
        result2 = ns["my_forward"](x)
        assert torch.allclose(result2, x + 1)
