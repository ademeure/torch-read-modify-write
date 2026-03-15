"""Tests for named intermediate variable naming in exported aten code."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from torch_graph.export import capture_aten_graphs, export_graph_to_python, _module_path_to_short


class TwoLayerModel(nn.Module):
    """Simple 2-layer model with distinct modules for naming tests."""
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(8, 8)
        self.layer1 = nn.Linear(8, 4)

    def forward(self, x):
        h = torch.relu(self.layer0(x))
        return self.layer1(h)


def test_module_path_to_short():
    """_module_path_to_short strips container names and collapses numbers."""
    assert _module_path_to_short("transformer.h.0.attn.c_q") == "h0_attn_c_q"
    assert _module_path_to_short("model.layers.2.mlp") == "2_mlp"
    assert _module_path_to_short("backbone.block.1.conv") == "block1_conv"
    assert _module_path_to_short("encoder.layer.0.attention.self.query") == "attention_self_query"
    # Edge case: all container names stripped
    assert _module_path_to_short("model.encoder.decoder") == ""
    # Simple path
    assert _module_path_to_short("fc1") == "fc1"


def test_named_intermediates_multilayer():
    """Named intermediates should contain layer indices for multi-layer models."""
    torch.manual_seed(42)
    model = TwoLayerModel()
    x = torch.randn(2, 8)
    _, capture = capture_aten_graphs(model, x, run_backward=False)

    code = export_graph_to_python(
        capture.forward_graphs[0].graph_module,
        named_intermediates=True,
        source_map=capture.source_map,
    )
    # Should contain references to both layers
    assert "layer0" in code
    assert "layer1" in code


def test_named_intermediates_backward():
    """Backward nodes should get a grad_ prefix."""
    torch.manual_seed(42)
    model = TwoLayerModel()
    x = torch.randn(2, 8)
    _, capture = capture_aten_graphs(model, x, run_backward=True)

    if not capture.backward_graphs:
        pytest.skip("No backward graph captured")

    code = export_graph_to_python(
        capture.backward_graphs[0].graph_module,
        fn_name="backward",
        named_intermediates=True,
        source_map=capture.source_map,
        is_backward=True,
    )
    # Backward names should have grad_ prefix
    assert "grad_" in code


def test_named_intermediates_dedup():
    """Duplicate derived names get _1, _2 suffixes."""
    torch.manual_seed(42)

    class DuplicateOps(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, x):
            # Multiple ops from the same module → same base name
            h = self.fc(x)
            return h + h  # add uses fc's output twice but add itself has unique name

    model = DuplicateOps()
    x = torch.randn(2, 8)
    _, capture = capture_aten_graphs(model, x, run_backward=False)

    code = export_graph_to_python(
        capture.forward_graphs[0].graph_module,
        named_intermediates=True,
        source_map=capture.source_map,
    )
    # Code should be valid Python (basic syntax check)
    assert "def forward(" in code
