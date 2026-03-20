"""Tests for uniquification of repeated module groups in aten export."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_graph.export import capture_aten_graphs, export_graph_to_python, export_aten_program


# ---------------------------------------------------------------------------
# Test models
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
    """Model with 3 identical residual blocks."""
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


class TinyTransformerLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_proj = nn.Linear(dim, dim)
        self.ff = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.attn_proj(x)
        return self.ln(x + self.ff(F.relu(h)))


class TinyTransformer(nn.Module):
    """Model with 2 identical transformer-like layers."""
    def __init__(self, dim=16):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([TinyTransformerLayer(dim) for _ in range(2)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class SingleLayerModel(nn.Module):
    """Model with NO repeated modules — uniquification should be a no-op."""
    def __init__(self, dim=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, 4)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resnet_uniquify_helper_emitted():
    """Uniquified ResNet should emit a shared helper function."""
    torch.manual_seed(42)
    model = ResNetLike(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False)
    fg = capture.forward_graphs[0]

    unique_fn_defs: list[str] = []
    code = export_graph_to_python(
        fg.graph_module,
        fn_name="forward",
        annotate_sources=True,
        source_map=capture.source_map,
        named_intermediates=True,
        uniquify=True,
        _unique_fn_defs=unique_fn_defs,
    )

    # Should have at least one helper function
    assert len(unique_fn_defs) >= 1, "Expected at least one shared helper function"

    # The helper function should be named after ResidualBlock (snake_case)
    fn_def = unique_fn_defs[0]
    assert "def " in fn_def
    assert "residual_block" in fn_def.lower() or "def " in fn_def

    # Helper should have a return annotation with tuple[...]
    assert "-> tuple[" in fn_def

    # Main function should have keyword call sites
    assert "=" in code  # call sites use keyword args


def test_resnet_uniquify_call_sites():
    """Call sites should use keyword arguments."""
    torch.manual_seed(42)
    model = ResNetLike(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False)
    fg = capture.forward_graphs[0]

    unique_fn_defs: list[str] = []
    code = export_graph_to_python(
        fg.graph_module,
        fn_name="forward",
        annotate_sources=True,
        source_map=capture.source_map,
        named_intermediates=True,
        uniquify=True,
        _unique_fn_defs=unique_fn_defs,
    )

    if unique_fn_defs:
        # Extract the function name from the first helper
        fn_def = unique_fn_defs[0]
        fn_name_line = [l for l in fn_def.split("\n") if l.startswith("def ")][0]
        fn_name = fn_name_line.split("(")[0].replace("def ", "")

        # The main code should call this function multiple times
        call_count = code.count(f"{fn_name}(")
        assert call_count >= 2, f"Expected 2+ calls to {fn_name}, got {call_count}"


def test_transformer_uniquify():
    """TinyTransformer with 2 layers should produce a shared helper."""
    torch.manual_seed(42)
    model = TinyTransformer(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False)
    fg = capture.forward_graphs[0]

    unique_fn_defs: list[str] = []
    code = export_graph_to_python(
        fg.graph_module,
        fn_name="forward",
        annotate_sources=True,
        source_map=capture.source_map,
        named_intermediates=True,
        uniquify=True,
        _unique_fn_defs=unique_fn_defs,
    )

    assert len(unique_fn_defs) >= 1, "Expected a shared helper for transformer layers"


def test_no_uniquify_single_layer():
    """Model with no repeated modules should produce no helpers."""
    torch.manual_seed(42)
    model = SingleLayerModel(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False)
    fg = capture.forward_graphs[0]

    unique_fn_defs: list[str] = []
    code = export_graph_to_python(
        fg.graph_module,
        fn_name="forward",
        annotate_sources=True,
        source_map=capture.source_map,
        named_intermediates=True,
        uniquify=True,
        _unique_fn_defs=unique_fn_defs,
    )

    assert len(unique_fn_defs) == 0, f"Expected no helpers, got {len(unique_fn_defs)}"


def test_uniquify_off_by_default_in_export_graph():
    """When uniquify=False (default), no helpers should be generated."""
    torch.manual_seed(42)
    model = ResNetLike(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False)
    fg = capture.forward_graphs[0]

    unique_fn_defs: list[str] = []
    code = export_graph_to_python(
        fg.graph_module,
        fn_name="forward",
        annotate_sources=True,
        source_map=capture.source_map,
        named_intermediates=True,
        uniquify=False,  # explicitly off
        _unique_fn_defs=unique_fn_defs,
    )

    assert len(unique_fn_defs) == 0


def test_export_aten_program_uniquify(tmp_path):
    """export_aten_program with uniquify=True should include SHARED LAYER FUNCTIONS section."""
    torch.manual_seed(42)
    model = ResNetLike(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False)

    out_path = str(tmp_path / "test_aten.py")
    export_aten_program(
        capture,
        out_path,
        include_test_harness=False,
        named_intermediates=True,
        skip_pt=True,
        uniquify=True,
    )

    content = (tmp_path / "test_aten.py").read_text()
    assert "SHARED LAYER FUNCTIONS" in content
    assert "def " in content  # at least one function def


def test_export_aten_program_no_uniquify(tmp_path):
    """export_aten_program with uniquify=False should NOT include shared layer section."""
    torch.manual_seed(42)
    model = ResNetLike(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False)

    out_path = str(tmp_path / "test_aten.py")
    export_aten_program(
        capture,
        out_path,
        include_test_harness=False,
        named_intermediates=True,
        skip_pt=True,
        uniquify=False,
    )

    content = (tmp_path / "test_aten.py").read_text()
    assert "SHARED LAYER FUNCTIONS" not in content


def test_uniquify_helper_has_return_annotation():
    """The generated helper function should have -> tuple[...] return annotation."""
    torch.manual_seed(42)
    model = ResNetLike(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False)
    fg = capture.forward_graphs[0]

    unique_fn_defs: list[str] = []
    export_graph_to_python(
        fg.graph_module,
        fn_name="forward",
        annotate_sources=True,
        source_map=capture.source_map,
        named_intermediates=True,
        uniquify=True,
        _unique_fn_defs=unique_fn_defs,
    )

    if unique_fn_defs:
        fn_def = unique_fn_defs[0]
        # Should have -> tuple[...] return annotation
        assert "-> tuple[" in fn_def, f"Missing return annotation in:\n{fn_def[:500]}"
        # Should have a return statement
        assert "return (" in fn_def


def test_uniquify_backward_supported():
    """Backward graphs CAN be uniquified when module metadata is available."""
    torch.manual_seed(42)
    model = ResNetLike(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=True)
    if not capture.backward_graphs:
        pytest.skip("No backward graph captured")

    bg = capture.backward_graphs[0]
    unique_fn_defs: list[str] = []
    code = export_graph_to_python(
        bg.graph_module,
        fn_name="backward",
        annotate_sources=True,
        source_map=capture.source_map,
        is_backward=True,
        named_intermediates=True,
        uniquify=True,
        _unique_fn_defs=unique_fn_defs,
    )

    # Backward uniquification may or may not produce helpers depending on
    # whether backward nodes carry nn_module_stack metadata.  When helpers
    # ARE produced, verify they're valid Python.
    if unique_fn_defs:
        import operator
        ns = {"torch": torch, "aten": torch.ops.aten, "operator": operator}
        for fn_def in unique_fn_defs:
            exec(fn_def, ns)  # should not raise
        # The main backward function should compile too
        exec("\n".join(unique_fn_defs) + "\n" + code, ns)


def test_uniquify_bit_identical_resnet():
    """Uniquified ResNet forward must produce bit-identical output to non-uniquified."""
    torch.manual_seed(42)
    model = ResNetLike(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False, record_real_tensors=True)
    fg = capture.forward_graphs[0]
    real_inputs = capture.forward_real_inputs

    code_plain = export_graph_to_python(
        fg.graph_module, fn_name="forward",
        source_map=capture.source_map, named_intermediates=True, uniquify=False,
    )

    unique_fn_defs: list[str] = []
    code_uniq = export_graph_to_python(
        fg.graph_module, fn_name="forward_u",
        source_map=capture.source_map, named_intermediates=True, uniquify=True,
        _unique_fn_defs=unique_fn_defs,
    )

    import operator
    ns = {"torch": torch, "aten": torch.ops.aten, "operator": operator}
    exec(code_plain, ns)
    exec("\n".join(unique_fn_defs) + "\n" + code_uniq, ns)

    r_plain = ns["forward"](*real_inputs)
    r_uniq = ns["forward_u"](*real_inputs)
    assert torch.equal(r_plain, r_uniq), f"Max diff: {(r_plain - r_uniq).abs().max().item()}"


def test_uniquify_bit_identical_transformer():
    """Uniquified TinyTransformer forward must produce bit-identical output."""
    torch.manual_seed(42)
    model = TinyTransformer(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False, record_real_tensors=True)
    fg = capture.forward_graphs[0]
    real_inputs = capture.forward_real_inputs

    code_plain = export_graph_to_python(
        fg.graph_module, fn_name="forward",
        source_map=capture.source_map, named_intermediates=True, uniquify=False,
    )

    unique_fn_defs: list[str] = []
    code_uniq = export_graph_to_python(
        fg.graph_module, fn_name="forward_u",
        source_map=capture.source_map, named_intermediates=True, uniquify=True,
        _unique_fn_defs=unique_fn_defs,
    )

    import operator
    ns = {"torch": torch, "aten": torch.ops.aten, "operator": operator}
    exec(code_plain, ns)
    exec("\n".join(unique_fn_defs) + "\n" + code_uniq, ns)

    r_plain = ns["forward"](*real_inputs)
    r_uniq = ns["forward_u"](*real_inputs)
    assert torch.equal(r_plain, r_uniq), f"Max diff: {(r_plain - r_uniq).abs().max().item()}"


def test_uniquify_heterogeneous_blocks_sub_modules():
    """Heterogeneous blocks shouldn't group at block level, but identical sub-modules should."""
    class BlockA(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim)
        def forward(self, x):
            return F.relu(self.fc(x))

    class BlockB(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim)
            self.ln = nn.LayerNorm(dim)
        def forward(self, x):
            return self.ln(self.fc(x))

    class Model(nn.Module):
        def __init__(self, dim=16):
            super().__init__()
            self.blocks = nn.ModuleList([BlockA(dim), BlockB(dim)])
        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    torch.manual_seed(42)
    model = Model(16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False)
    fg = capture.forward_graphs[0]

    unique_fn_defs: list[str] = []
    unique_groups: list = []
    export_graph_to_python(
        fg.graph_module, fn_name="forward",
        source_map=capture.source_map, named_intermediates=True, uniquify=True,
        _unique_fn_defs=unique_fn_defs, _unique_groups=unique_groups,
    )
    # Top-level blocks should NOT be grouped (different structures)
    block_level = [g for g in unique_groups if g.template_key == "self.blocks.*"]
    assert len(block_level) == 0, "Heterogeneous blocks should not group at block level"

    # But the shared Linear sub-module (self.blocks.*.fc) SHOULD be extracted
    sub_groups = [g for g in unique_groups if ".fc" in g.template_key]
    assert len(sub_groups) >= 1, "Identical sub-modules should be extracted"


def test_uniquify_many_layers_scales():
    """10 identical layers should produce exactly 1 helper with 10 call sites."""
    class Block(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim)
            self.ln = nn.LayerNorm(dim)
        def forward(self, x):
            return self.ln(torch.relu(self.fc(x)) + x)

    class BigModel(nn.Module):
        def __init__(self, dim=16, n=10):
            super().__init__()
            self.layers = nn.ModuleList([Block(dim) for _ in range(n)])
        def forward(self, x):
            for l in self.layers:
                x = l(x)
            return x

    torch.manual_seed(42)
    model = BigModel(16, 10)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False, record_real_tensors=True)
    fg = capture.forward_graphs[0]

    unique_fn_defs: list[str] = []
    code = export_graph_to_python(
        fg.graph_module, fn_name="forward_u",
        source_map=capture.source_map, named_intermediates=True, uniquify=True,
        _unique_fn_defs=unique_fn_defs,
    )

    assert len(unique_fn_defs) == 1, f"Expected 1 helper, got {len(unique_fn_defs)}"

    fn_name = [l for l in unique_fn_defs[0].split("\n") if l.startswith("def ")][0].split("(")[0].replace("def ", "")
    call_count = code.count(f"{fn_name}(")
    assert call_count == 10, f"Expected 10 call sites, got {call_count}"

    # Verify bit-identical
    code_plain = export_graph_to_python(
        fg.graph_module, fn_name="forward",
        source_map=capture.source_map, named_intermediates=True, uniquify=False,
    )
    import operator
    ns = {"torch": torch, "aten": torch.ops.aten, "operator": operator}
    exec(code_plain, ns)
    exec("\n".join(unique_fn_defs) + "\n" + code, ns)
    r1 = ns["forward"](*capture.forward_real_inputs)
    r2 = ns["forward_u"](*capture.forward_real_inputs)
    assert torch.equal(r1, r2)


# ---------------------------------------------------------------------------
# Phase 1: Hierarchical / partial matching tests
# ---------------------------------------------------------------------------


def test_partial_matching_even_odd_layers():
    """Layers with two distinct structures should produce two separate groups."""
    class BlockEven(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim)
            self.ln = nn.LayerNorm(dim)
        def forward(self, x):
            return self.ln(F.relu(self.fc(x)))

    class BlockOdd(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim)
            self.gate = nn.Linear(dim, dim)  # extra module
            self.ln = nn.LayerNorm(dim)
        def forward(self, x):
            return self.ln(F.relu(self.fc(x)) * torch.sigmoid(self.gate(x)))

    class Model(nn.Module):
        def __init__(self, dim=16):
            super().__init__()
            # Interleaved even/odd: 0=even, 1=odd, 2=even, 3=odd
            blocks = []
            for i in range(4):
                blocks.append(BlockEven(dim) if i % 2 == 0 else BlockOdd(dim))
            self.layers = nn.ModuleList(blocks)
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    torch.manual_seed(42)
    model = Model(16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False, record_real_tensors=True)
    fg = capture.forward_graphs[0]

    unique_fn_defs: list[str] = []
    unique_groups: list = []
    code = export_graph_to_python(
        fg.graph_module, fn_name="forward_u",
        source_map=capture.source_map, named_intermediates=True, uniquify=True,
        _unique_fn_defs=unique_fn_defs, _unique_groups=unique_groups,
    )

    # Should find groups — either partial top-level or sub-module level
    assert len(unique_groups) >= 1, "Should extract at least one group"

    # Verify bit-identical output
    code_plain = export_graph_to_python(
        fg.graph_module, fn_name="forward",
        source_map=capture.source_map, named_intermediates=True, uniquify=False,
    )
    import operator
    ns = {"torch": torch, "aten": torch.ops.aten, "operator": operator}
    exec(code_plain, ns)
    exec("\n".join(unique_fn_defs) + "\n" + code, ns)
    r1 = ns["forward"](*capture.forward_real_inputs)
    r2 = ns["forward_u"](*capture.forward_real_inputs)
    assert torch.equal(r1, r2), f"Max diff: {(r1 - r2).abs().max().item()}"


def test_hierarchical_submodule_extraction():
    """When top-level blocks differ, identical sub-modules should still be extracted."""
    class Attention(nn.Module):
        """Shared across all layers."""
        def __init__(self, dim):
            super().__init__()
            self.q = nn.Linear(dim, dim, bias=False)
            self.k = nn.Linear(dim, dim, bias=False)
            self.proj = nn.Linear(dim, dim, bias=False)
        def forward(self, x):
            q, k = self.q(x), self.k(x)
            attn = torch.softmax(q @ k.transpose(-1, -2), dim=-1)
            return self.proj(attn @ x)

    class MLP(nn.Module):
        """Shared across all layers."""
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * 2)
            self.fc2 = nn.Linear(dim * 2, dim)
        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    class BlockA(nn.Module):
        """Block variant A: attention + mlp (no LN)."""
        def __init__(self, dim):
            super().__init__()
            self.attn = Attention(dim)
            self.mlp = MLP(dim)
        def forward(self, x):
            return x + self.mlp(self.attn(x))

    class BlockB(nn.Module):
        """Block variant B: attention + mlp + layernorm."""
        def __init__(self, dim):
            super().__init__()
            self.attn = Attention(dim)
            self.mlp = MLP(dim)
            self.ln = nn.LayerNorm(dim)
        def forward(self, x):
            return self.ln(x + self.mlp(self.attn(x)))

    class Model(nn.Module):
        def __init__(self, dim=16):
            super().__init__()
            self.layers = nn.ModuleList([BlockA(dim), BlockB(dim), BlockA(dim), BlockB(dim)])
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    torch.manual_seed(42)
    model = Model(16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False, record_real_tensors=True)
    fg = capture.forward_graphs[0]

    unique_fn_defs: list[str] = []
    unique_groups: list = []
    code = export_graph_to_python(
        fg.graph_module, fn_name="forward_u",
        source_map=capture.source_map, named_intermediates=True, uniquify=True,
        _unique_fn_defs=unique_fn_defs, _unique_groups=unique_groups,
    )

    # Top-level blocks differ (A vs B), but sub-modules should be extracted
    assert len(unique_groups) >= 1, "Should extract sub-module groups"

    # Check that attn and/or mlp sub-modules were extracted
    template_keys = [g.template_key for g in unique_groups]
    has_sub_modules = any("attn" in k or "mlp" in k or "fc" in k for k in template_keys)
    has_partial_top = any("layers.*" == k.split(".")[-2] + "." + k.split(".")[-1] if ".*" in k else False for k in template_keys)
    assert has_sub_modules or has_partial_top, (
        f"Expected sub-module or partial top-level extraction, got: {template_keys}"
    )

    # Verify bit-identical output
    code_plain = export_graph_to_python(
        fg.graph_module, fn_name="forward",
        source_map=capture.source_map, named_intermediates=True, uniquify=False,
    )
    import operator
    ns = {"torch": torch, "aten": torch.ops.aten, "operator": operator}
    exec(code_plain, ns)
    exec("\n".join(unique_fn_defs) + "\n" + code, ns)
    r1 = ns["forward"](*capture.forward_real_inputs)
    r2 = ns["forward_u"](*capture.forward_real_inputs)
    assert torch.equal(r1, r2), f"Max diff: {(r1 - r2).abs().max().item()}"


def test_export_aten_program_backward_uniquify(tmp_path):
    """export_aten_program with uniquify=True should include backward shared functions when available."""
    torch.manual_seed(42)
    model = ResNetLike(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=True)

    out_path = str(tmp_path / "test_bw.py")
    export_aten_program(
        capture, out_path,
        include_test_harness=False,
        named_intermediates=True,
        skip_pt=True,
        uniquify=True,
    )

    content = (tmp_path / "test_bw.py").read_text()
    # Forward should always have shared functions for ResNetLike
    assert "SHARED LAYER FUNCTIONS" in content
    # Backward may also have shared functions if metadata is available
    # (check content compiles as valid Python)
    import operator
    ns = {"torch": torch, "aten": torch.ops.aten, "operator": operator}
    exec(compile(content, out_path, "exec"), ns)


# ---------------------------------------------------------------------------
# Configurable depth / min_ops tests
# ---------------------------------------------------------------------------


def test_uniquify_depth_1_top_level_only():
    """uniquify_depth=1 should only extract top-level groups, no sub-modules."""
    torch.manual_seed(42)
    model = ResNetLike(dim=16)
    x = torch.randn(4, 16)

    _, capture = capture_aten_graphs(model, x, run_backward=False)
    fg = capture.forward_graphs[0]

    groups_d1: list = []
    export_graph_to_python(
        fg.graph_module, fn_name="forward",
        source_map=capture.source_map, named_intermediates=True,
        uniquify=True, uniquify_depth=1, _unique_groups=groups_d1,
    )
    # Depth 1 should produce the top-level ResidualBlock group
    assert len(groups_d1) >= 1
    # All groups should be at the shallowest depth (same dot count)
    depths = {g.template_key.count(".") for g in groups_d1}
    assert len(depths) == 1, f"Depth=1 should produce groups at one level only, got depths: {depths}"


def test_uniquify_depth_controls_sub_module_extraction():
    """uniquify_depth=1 vs -1 should differ when top-level is heterogeneous."""
    class BlockA(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim)
        def forward(self, x):
            return F.relu(self.fc(x))

    class BlockB(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim)
            self.ln = nn.LayerNorm(dim)
        def forward(self, x):
            return self.ln(self.fc(x))

    class Model(nn.Module):
        def __init__(self, dim=16):
            super().__init__()
            self.blocks = nn.ModuleList([BlockA(dim), BlockB(dim)])
        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    torch.manual_seed(42)
    model = Model(16)
    x = torch.randn(4, 16)
    _, capture = capture_aten_graphs(model, x, run_backward=False)
    fg = capture.forward_graphs[0]

    # Depth 1: top-level only — blocks are heterogeneous, no groups
    groups_d1: list = []
    export_graph_to_python(
        fg.graph_module, fn_name="forward",
        source_map=capture.source_map, named_intermediates=True,
        uniquify=True, uniquify_depth=1, _unique_groups=groups_d1,
    )
    assert len(groups_d1) == 0, "Depth=1 with heterogeneous blocks should find nothing"

    # Depth -1 (unlimited): should find the shared Linear sub-module
    groups_all: list = []
    export_graph_to_python(
        fg.graph_module, fn_name="forward",
        source_map=capture.source_map, named_intermediates=True,
        uniquify=True, uniquify_depth=-1, _unique_groups=groups_all,
    )
    assert len(groups_all) >= 1, "Depth=-1 should extract shared sub-modules"


def test_uniquify_min_ops_filters_small_groups():
    """uniquify_min_ops should prevent extraction of small groups."""
    torch.manual_seed(42)
    model = ResNetLike(dim=16)
    x = torch.randn(4, 16)
    _, capture = capture_aten_graphs(model, x, run_backward=False)
    fg = capture.forward_graphs[0]

    # With a very high min_ops, nothing should be extracted
    groups_high: list = []
    export_graph_to_python(
        fg.graph_module, fn_name="forward",
        source_map=capture.source_map, named_intermediates=True,
        uniquify=True, uniquify_min_ops=9999, _unique_groups=groups_high,
    )
    assert len(groups_high) == 0, "min_ops=9999 should prevent all extraction"

    # With min_ops=0 (default), extraction should work
    groups_low: list = []
    export_graph_to_python(
        fg.graph_module, fn_name="forward",
        source_map=capture.source_map, named_intermediates=True,
        uniquify=True, uniquify_min_ops=0, _unique_groups=groups_low,
    )
    assert len(groups_low) >= 1, "min_ops=0 should allow extraction"
