"""Test aten_to_cuda: parsing, fusion analysis, and CUDA kernel generation.

Tests are in two parts:
  1. Structural tests (CPU, no CUDA): parsing, classification, fusion grouping,
     kernel string generation.
  2. GPU verification tests (require CUDA): compile generated kernels, run with
     real tensors, compare against PyTorch reference. These catch wrong math.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch_graph.aten_to_cuda import (
    parse_run_function,
    analyze,
    transform,
    _ELEMENTWISE_EXPR,
    _IDENTITY_OPS,
    _LAYOUT_OPS,
    _BARRIER_OPS,
)


# ── Parsing tests ────────────────────────────────────────────────────────────


class TestParsing:
    def test_simple_add(self):
        ops = parse_run_function("""
def run(inputs):
    add = torch.ops.aten.add.Tensor(inputs.x, inputs.y)
    return [add]
""")
        assert len(ops) == 1
        assert ops[0].output_var == "add"
        assert ops[0].op_name == "add"
        assert ops[0].op_variant == "Tensor"
        assert len(ops[0].args) == 2
        assert ops[0].args[0].kind == "input"
        assert ops[0].args[0].name == "x"

    def test_scalar_variant(self):
        ops = parse_run_function("""
def run(inputs):
    eq = torch.ops.aten.eq.Scalar(inputs.x, 0)
    return [eq]
""")
        assert ops[0].op_name == "eq"
        assert ops[0].op_variant == "Scalar"
        assert ops[0].has_scalar_variant
        assert ops[0].args[1].kind == "scalar"
        assert ops[0].args[1].value == 0

    def test_float_scalar_in_tensor_variant(self):
        ops = parse_run_function("""
def run(inputs):
    mul = torch.ops.aten.mul.Tensor(inputs.x, 0.25)
    return [mul]
""")
        assert ops[0].args[1].kind == "scalar"
        assert ops[0].args[1].value == 0.25

    def test_local_var_reference(self):
        ops = parse_run_function("""
def run(inputs):
    relu = torch.ops.aten.relu.default(inputs.x)
    add = torch.ops.aten.add.Tensor(relu, inputs.y)
    return [add]
""")
        assert len(ops) == 2
        assert ops[1].args[0].kind == "var"
        assert ops[1].args[0].name == "relu"

    def test_getitem(self):
        ops = parse_run_function("""
def run(inputs):
    getitem = operator.getitem(inputs.result, 0)
    return [getitem]
""")
        assert len(ops) == 1
        assert ops[0].op_name == "__getitem__"
        assert ops[0].is_special

    def test_list_arg(self):
        ops = parse_run_function("""
def run(inputs):
    view = torch.ops.aten.view.default(inputs.x, [2, 16, 32])
    return [view]
""")
        assert ops[0].op_name == "view"
        assert ops[0].args[1].kind == "scalar"
        assert ops[0].args[1].value == [2, 16, 32]

    def test_negative_inf(self):
        ops = parse_run_function("""
def run(inputs):
    masked_fill = torch.ops.aten.masked_fill.Scalar(inputs.x, inputs.mask, -inf)
    return [masked_fill]
""")
        assert ops[0].args[2].kind == "scalar"
        assert ops[0].args[2].value == float("-inf")

    def test_keyword_arg(self):
        ops = parse_run_function("""
def run(inputs):
    clone = torch.ops.aten.clone.default(inputs.x, memory_format=torch.contiguous_format)
    return [clone]
""")
        assert len(ops[0].args) == 2

    def test_multi_op_chain(self):
        ops = parse_run_function("""
def run(inputs):
    gelu = torch.ops.aten.gelu.default(inputs.x)
    add = torch.ops.aten.add.Tensor(gelu, inputs.y)
    mul = torch.ops.aten.mul.Tensor(add, 2.0)
    return [mul]
""")
        assert len(ops) == 3
        assert all(op.is_elementwise for op in ops)


# ── Classification tests ─────────────────────────────────────────────────────


class TestClassification:
    def test_elementwise_ops(self):
        for op in ["relu", "gelu", "sigmoid", "tanh", "add", "mul", "sub",
                    "div", "neg", "abs", "exp", "log", "eq", "gt", "where"]:
            ops = parse_run_function(f"""
def run(inputs):
    out = torch.ops.aten.{op}.default(inputs.x)
    return [out]
""")
            assert ops[0].is_elementwise, f"{op} should be elementwise"

    def test_identity_ops(self):
        for op in ["alias", "detach"]:
            ops = parse_run_function(f"""
def run(inputs):
    out = torch.ops.aten.{op}.default(inputs.x)
    return [out]
""")
            assert ops[0].is_identity, f"{op} should be identity"
            assert ops[0].is_fusable, f"{op} should be fusable"

    def test_layout_ops(self):
        for op in ["view", "reshape", "transpose", "permute", "expand",
                    "unsqueeze", "squeeze", "clone"]:
            ops = parse_run_function(f"""
def run(inputs):
    out = torch.ops.aten.{op}.default(inputs.x)
    return [out]
""")
            assert ops[0].is_layout, f"{op} should be a layout op"
            assert not ops[0].is_fusable, f"{op} should NOT be fusable"

    def test_barrier_ops(self):
        for op in ["mm", "bmm", "addmm", "convolution", "embedding",
                    "_softmax", "native_layer_norm", "sum", "mean"]:
            ops = parse_run_function(f"""
def run(inputs):
    out = torch.ops.aten.{op}.default(inputs.x)
    return [out]
""")
            assert ops[0].is_barrier, f"{op} should be a barrier"


# ── Fusion analysis tests ────────────────────────────────────────────────────


class TestFusion:
    def test_single_elementwise(self):
        r = analyze("""
def run(inputs):
    add = torch.ops.aten.add.Tensor(inputs.x, inputs.y)
    return [add]
""")
        assert r["stats"]["fusion_groups"] == 1
        assert r["stats"]["ops_fused"] == 1

    def test_chain_fuses(self):
        r = analyze("""
def run(inputs):
    gelu = torch.ops.aten.gelu.default(inputs.x)
    add = torch.ops.aten.add.Tensor(gelu, inputs.y)
    mul = torch.ops.aten.mul.Tensor(add, inputs.z)
    return [mul]
""")
        assert r["stats"]["fusion_groups"] == 1
        assert r["stats"]["ops_fused"] == 3

    def test_barrier_splits_groups(self):
        r = analyze("""
def run(inputs):
    relu = torch.ops.aten.relu.default(inputs.x)
    mm = torch.ops.aten.mm.default(relu, inputs.w)
    gelu = torch.ops.aten.gelu.default(mm)
    return [gelu]
""")
        # relu before mm, gelu after mm → 2 groups
        assert r["stats"]["fusion_groups"] == 2
        assert r["stats"]["barrier"] == 1

    def test_identity_included_in_group(self):
        """alias and detach are identity ops — safe to include in fusion."""
        r = analyze("""
def run(inputs):
    alias = torch.ops.aten.alias.default(inputs.x)
    relu = torch.ops.aten.relu.default(alias)
    detach = torch.ops.aten.detach.default(relu)
    return [detach]
""")
        assert r["stats"]["fusion_groups"] == 1
        assert r["stats"]["ops_fused"] == 3  # alias + relu + detach

    def test_layout_op_breaks_fusion(self):
        """view/transpose/permute are layout ops — they break fusion."""
        r = analyze("""
def run(inputs):
    relu = torch.ops.aten.relu.default(inputs.x)
    view = torch.ops.aten.view.default(relu, [4, 8])
    gelu = torch.ops.aten.gelu.default(view)
    return [gelu]
""")
        # view breaks fusion → relu in one group, gelu in another
        assert r["stats"]["fusion_groups"] == 2

    def test_pure_layout_not_fused(self):
        r = analyze("""
def run(inputs):
    view = torch.ops.aten.view.default(inputs.x, [4, 8])
    transpose = torch.ops.aten.transpose.int(view, 0, 1)
    return [transpose]
""")
        # No elementwise ops → no fusion group
        assert r["stats"]["fusion_groups"] == 0

    def test_special_ops_not_fused(self):
        r = analyze("""
def run(inputs):
    getitem = operator.getitem(inputs.result, 0)
    return [getitem]
""")
        assert r["stats"]["fusion_groups"] == 0
        assert r["stats"]["special"] == 1

    def test_barrier_then_elementwise(self):
        r = analyze("""
def run(inputs):
    native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.x, [32], inputs.w, inputs.b, 1e-05)
    getitem = operator.getitem(native_layer_norm, 0)
    add = torch.ops.aten.add.Tensor(inputs.x, inputs.y)
    return [add]
""")
        assert r["stats"]["fusion_groups"] == 1
        assert r["stats"]["barrier"] == 1
        assert r["stats"]["special"] == 1


# ── Kernel generation tests ──────────────────────────────────────────────────


class TestKernelGen:
    def test_add_kernel(self):
        r = analyze("""
def run(inputs):
    add = torch.ops.aten.add.Tensor(inputs.x, inputs.y)
    return [add]
""")
        assert len(r["kernels"]) == 1
        _, name, src = r["kernels"][0]
        assert "in0[i] + in1[i]" in src
        assert "float *out0" in src
        assert "const float *in0" in src
        assert "const float *in1" in src

    def test_mul_scalar_kernel(self):
        r = analyze("""
def run(inputs):
    mul = torch.ops.aten.mul.Tensor(inputs.x, 0.25)
    return [mul]
""")
        _, _, src = r["kernels"][0]
        assert "0.25f" in src
        assert "in0[i]" in src

    def test_gelu_add_fused_kernel(self):
        r = analyze("""
def run(inputs):
    gelu = torch.ops.aten.gelu.default(inputs.x)
    add = torch.ops.aten.add.Tensor(gelu, inputs.residual)
    return [add]
""")
        _, _, src = r["kernels"][0]
        # Should contain gelu math fused with add
        assert "erff" in src
        assert "in1[i]" in src  # residual
        assert "out0[i]" in src

    def test_eq_scalar_kernel(self):
        r = analyze("""
def run(inputs):
    eq = torch.ops.aten.eq.Scalar(inputs.x, 0)
    return [eq]
""")
        _, _, src = r["kernels"][0]
        assert "0.0f" in src  # scalar 0 converted to float

    def test_alias_passthrough_in_fusion(self):
        """alias is identity — forwarded through the fused kernel."""
        r = analyze("""
def run(inputs):
    alias = torch.ops.aten.alias.default(inputs.x)
    relu = torch.ops.aten.relu.default(alias)
    return [relu]
""")
        _, _, src = r["kernels"][0]
        # alias is passthrough, relu reads same data
        assert "in0[i]" in src
        assert "> 0.0f" in src

    def test_masked_fill_fusion(self):
        r = analyze("""
def run(inputs):
    alias = torch.ops.aten.alias.default(inputs.mask)
    eq = torch.ops.aten.eq.Scalar(alias, 0)
    masked_fill = torch.ops.aten.masked_fill.Scalar(inputs.x, eq, -inf)
    return [masked_fill]
""")
        _, _, src = r["kernels"][0]
        # eq result feeds into masked_fill — no intermediate write
        assert "-INFINITY" in src
        assert "in0[i]" in src  # mask
        assert "in1[i]" in src  # x

    def test_multiple_outputs(self):
        r = analyze("""
def run(inputs):
    relu = torch.ops.aten.relu.default(inputs.x)
    neg = torch.ops.aten.neg.default(relu)
    return [relu, neg]
""")
        group, _, src = r["kernels"][0]
        assert len(group.output_vars) == 2
        assert "out0[i]" in src
        assert "out1[i]" in src


# ── Transform output tests ───────────────────────────────────────────────────


class TestTransform:
    def test_preserves_init_once(self):
        out = transform("""
import torch

def init_once():
    return {"h5": "data/test.h5"}

def run(inputs):
    add = torch.ops.aten.add.Tensor(inputs.x, inputs.y)
    return [add]
""")
        assert 'def init_once():' in out
        assert '"h5": "data/test.h5"' in out

    def test_contains_kernel_source(self):
        out = transform("""
def run(inputs):
    gelu = torch.ops.aten.gelu.default(inputs.x)
    return [gelu]
""")
        assert "KERNEL_FUSED_0" in out
        assert "extern" in out
        assert "__global__" in out

    def test_contains_pytorch_fallback(self):
        out = transform("""
def run(inputs):
    add = torch.ops.aten.add.Tensor(inputs.x, inputs.y)
    return [add]
""")
        # Should keep the original PyTorch call as reference
        assert "torch.ops.aten.add.Tensor" in out

    def test_barrier_ops_unchanged(self):
        out = transform("""
def run(inputs):
    mm = torch.ops.aten.mm.default(inputs.x, inputs.w)
    return [mm]
""")
        assert "torch.ops.aten.mm.default" in out
        assert "KERNEL" not in out  # no kernel for mm

    def test_mixed_barrier_and_elementwise(self):
        out = transform("""
def run(inputs):
    mm = torch.ops.aten.mm.default(inputs.x, inputs.w)
    relu = torch.ops.aten.relu.default(mm)
    return [relu]
""")
        assert "torch.ops.aten.mm.default" in out  # barrier kept
        assert "KERNEL_FUSED_0" in out  # relu fused
        assert "relu_kernel" not in out or "fused_0" in out


# ═══════════════════════════════════════════════════════════════════════════════
# GPU verification: compile generated CUDA kernels and run with real tensors
# ═══════════════════════════════════════════════════════════════════════════════

_GPU = torch.cuda.is_available()
_skip_no_gpu = pytest.mark.skipif(not _GPU, reason="CUDA not available")


def _compile_and_run_kernel(kernel_src: str, input_tensors: list[torch.Tensor],
                            n_outputs: int = 1) -> list[torch.Tensor]:
    """Compile a CUDA kernel string and run it on the given input tensors.

    Uses torch.utils.cpp_extension.load_inline to compile on-the-fly.
    """
    import re
    from torch.utils.cpp_extension import load_inline

    # Extract kernel function name
    m = re.search(r'void\s+(\w+)\s*\(', kernel_src)
    func_name = m.group(1)
    n = input_tensors[0].numel()

    # Build C++ wrapper
    in_args = ", ".join(f"torch::Tensor in{i}" for i in range(len(input_tensors)))
    in_ptrs = ", ".join(f"in{i}.data_ptr<float>()" for i in range(len(input_tensors)))
    out_allocs = "\n".join(
        f"    auto out{i} = torch::empty_like(in0);" for i in range(n_outputs))
    out_ptrs = ", ".join(f"out{i}.data_ptr<float>()" for i in range(n_outputs))
    out_vec = ", ".join(f"out{i}" for i in range(n_outputs))

    wrapper_name = f"{func_name}_wrapper"
    cuda_src = f"""
#include <torch/extension.h>
#include <math.h>

{kernel_src}

std::vector<torch::Tensor> {wrapper_name}({in_args}) {{
{out_allocs}
    int n = in0.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    {func_name}<<<blocks, threads>>>({in_ptrs}, {out_ptrs}, n);
    return {{{out_vec}}};
}}
"""
    ext = load_inline(
        name=f"test_{func_name}",
        cpp_sources="",
        cuda_sources=[cuda_src],
        functions=[wrapper_name],
        verbose=False,
    )
    fn = getattr(ext, wrapper_name)
    return fn(*input_tensors)


@_skip_no_gpu
class TestGPUVerification:
    """Compile generated CUDA kernels and verify numerical correctness."""

    def test_add_kernel_correct(self):
        r = analyze("""
def run(inputs):
    add = torch.ops.aten.add.Tensor(inputs.x, inputs.y)
    return [add]
""")
        _, _, ksrc = r["kernels"][0]
        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        [result] = _compile_and_run_kernel(ksrc, [x, y])
        expected = x + y
        assert torch.allclose(result, expected, atol=1e-6), (
            f"add kernel wrong: max diff {(result - expected).abs().max():.2e}")

    def test_mul_scalar_kernel_correct(self):
        r = analyze("""
def run(inputs):
    mul = torch.ops.aten.mul.Tensor(inputs.x, 0.25)
    return [mul]
""")
        _, _, ksrc = r["kernels"][0]
        x = torch.randn(1024, device="cuda")
        [result] = _compile_and_run_kernel(ksrc, [x])
        expected = x * 0.25
        assert torch.allclose(result, expected, atol=1e-6), (
            f"mul scalar kernel wrong: max diff {(result - expected).abs().max():.2e}")

    def test_relu_kernel_correct(self):
        r = analyze("""
def run(inputs):
    relu = torch.ops.aten.relu.default(inputs.x)
    return [relu]
""")
        _, _, ksrc = r["kernels"][0]
        x = torch.randn(2048, device="cuda")
        [result] = _compile_and_run_kernel(ksrc, [x])
        expected = torch.relu(x)
        assert torch.equal(result, expected), "relu kernel wrong"

    def test_gelu_kernel_correct(self):
        r = analyze("""
def run(inputs):
    gelu = torch.ops.aten.gelu.default(inputs.x)
    return [gelu]
""")
        _, _, ksrc = r["kernels"][0]
        x = torch.randn(2048, device="cuda")
        [result] = _compile_and_run_kernel(ksrc, [x])
        expected = torch.nn.functional.gelu(x)
        assert torch.allclose(result, expected, atol=1e-5), (
            f"gelu kernel wrong: max diff {(result - expected).abs().max():.2e}")

    def test_sigmoid_kernel_correct(self):
        r = analyze("""
def run(inputs):
    sigmoid = torch.ops.aten.sigmoid.default(inputs.x)
    return [sigmoid]
""")
        _, _, ksrc = r["kernels"][0]
        x = torch.randn(2048, device="cuda")
        [result] = _compile_and_run_kernel(ksrc, [x])
        expected = torch.sigmoid(x)
        assert torch.allclose(result, expected, atol=1e-5), (
            f"sigmoid kernel wrong: max diff {(result - expected).abs().max():.2e}")

    def test_neg_kernel_correct(self):
        r = analyze("""
def run(inputs):
    neg = torch.ops.aten.neg.default(inputs.x)
    return [neg]
""")
        _, _, ksrc = r["kernels"][0]
        x = torch.randn(1024, device="cuda")
        [result] = _compile_and_run_kernel(ksrc, [x])
        expected = -x
        assert torch.equal(result, expected), "neg kernel wrong"

    def test_eq_scalar_kernel_correct(self):
        r = analyze("""
def run(inputs):
    eq = torch.ops.aten.eq.Scalar(inputs.x, 0)
    return [eq]
""")
        _, _, ksrc = r["kernels"][0]
        x = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0], device="cuda")
        [result] = _compile_and_run_kernel(ksrc, [x])
        expected = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0], device="cuda")
        assert torch.equal(result, expected), (
            f"eq.Scalar kernel wrong: got {result}, expected {expected}")

    def test_gelu_add_fused_correct(self):
        """Two ops fused: gelu(x) + y — verify the FUSED kernel is correct."""
        r = analyze("""
def run(inputs):
    gelu = torch.ops.aten.gelu.default(inputs.x)
    add = torch.ops.aten.add.Tensor(gelu, inputs.y)
    return [add]
""")
        assert r["stats"]["fusion_groups"] == 1
        assert r["stats"]["ops_fused"] == 2
        _, _, ksrc = r["kernels"][0]
        x = torch.randn(2048, device="cuda")
        y = torch.randn(2048, device="cuda")
        [result] = _compile_and_run_kernel(ksrc, [x, y])
        expected = torch.nn.functional.gelu(x) + y
        assert torch.allclose(result, expected, atol=1e-5), (
            f"gelu+add fused kernel wrong: max diff "
            f"{(result - expected).abs().max():.2e}")

    def test_gelu_add_mul_triple_fused_correct(self):
        """Three ops fused: gelu(x) + y then * z."""
        r = analyze("""
def run(inputs):
    gelu = torch.ops.aten.gelu.default(inputs.x)
    add = torch.ops.aten.add.Tensor(gelu, inputs.y)
    mul = torch.ops.aten.mul.Tensor(add, inputs.z)
    return [mul]
""")
        assert r["stats"]["ops_fused"] == 3
        _, _, ksrc = r["kernels"][0]
        x = torch.randn(2048, device="cuda")
        y = torch.randn(2048, device="cuda")
        z = torch.randn(2048, device="cuda")
        [result] = _compile_and_run_kernel(ksrc, [x, y, z])
        expected = (torch.nn.functional.gelu(x) + y) * z
        assert torch.allclose(result, expected, atol=1e-5), (
            f"gelu+add+mul fused kernel wrong: max diff "
            f"{(result - expected).abs().max():.2e}")

    def test_masked_fill_fused_correct(self):
        """alias → eq.Scalar → masked_fill.Scalar fused into one kernel."""
        r = analyze("""
def run(inputs):
    alias = torch.ops.aten.alias.default(inputs.mask)
    eq = torch.ops.aten.eq.Scalar(alias, 0)
    masked_fill = torch.ops.aten.masked_fill.Scalar(inputs.x, eq, -inf)
    return [masked_fill]
""")
        assert r["stats"]["ops_fused"] == 3
        _, _, ksrc = r["kernels"][0]
        # mask: 1s on diagonal (causal), 0s above
        mask = torch.tril(torch.ones(4, 4, device="cuda"))
        x = torch.randn(4, 4, device="cuda")
        mask_flat = mask.reshape(-1)
        x_flat = x.reshape(-1)
        [result] = _compile_and_run_kernel(ksrc, [mask_flat, x_flat])
        # Reference: where mask==0, fill -inf; else keep x
        expected = x_flat.clone()
        expected[mask_flat == 0] = float("-inf")
        assert torch.equal(result, expected), (
            f"masked_fill fused kernel wrong:\n"
            f"  result:   {result}\n"
            f"  expected: {expected}")

    def test_multiple_outputs_correct(self):
        """Two outputs from one fused group: relu(x) and neg(relu(x))."""
        r = analyze("""
def run(inputs):
    relu = torch.ops.aten.relu.default(inputs.x)
    neg = torch.ops.aten.neg.default(relu)
    return [relu, neg]
""")
        group, _, ksrc = r["kernels"][0]
        assert len(group.output_vars) == 2
        x = torch.randn(1024, device="cuda")
        results = _compile_and_run_kernel(ksrc, [x], n_outputs=2)
        expected_relu = torch.relu(x)
        expected_neg = -expected_relu
        assert torch.equal(results[0], expected_relu), "relu output wrong"
        assert torch.equal(results[1], expected_neg), "neg output wrong"

    def test_silu_kernel_correct(self):
        r = analyze("""
def run(inputs):
    silu = torch.ops.aten.silu.default(inputs.x)
    return [silu]
""")
        _, _, ksrc = r["kernels"][0]
        x = torch.randn(2048, device="cuda")
        [result] = _compile_and_run_kernel(ksrc, [x])
        expected = torch.nn.functional.silu(x)
        assert torch.allclose(result, expected, atol=1e-5), (
            f"silu kernel wrong: max diff {(result - expected).abs().max():.2e}")

    def test_tanh_kernel_correct(self):
        r = analyze("""
def run(inputs):
    tanh = torch.ops.aten.tanh.default(inputs.x)
    return [tanh]
""")
        _, _, ksrc = r["kernels"][0]
        x = torch.randn(2048, device="cuda")
        [result] = _compile_and_run_kernel(ksrc, [x])
        expected = torch.tanh(x)
        assert torch.allclose(result, expected, atol=1e-6), (
            f"tanh kernel wrong: max diff {(result - expected).abs().max():.2e}")

    def test_sub_div_chain_correct(self):
        """sub then div fused."""
        r = analyze("""
def run(inputs):
    sub = torch.ops.aten.sub.Tensor(inputs.x, inputs.y)
    div = torch.ops.aten.div.Tensor(sub, inputs.z)
    return [div]
""")
        assert r["stats"]["ops_fused"] == 2
        _, _, ksrc = r["kernels"][0]
        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        z = torch.randn(1024, device="cuda").abs() + 0.1  # avoid div by zero
        [result] = _compile_and_run_kernel(ksrc, [x, y, z])
        expected = (x - y) / z
        assert torch.allclose(result, expected, atol=1e-5), (
            f"sub+div fused wrong: max diff {(result - expected).abs().max():.2e}")
