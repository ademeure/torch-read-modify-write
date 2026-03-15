"""Tests for torch_graph.explain one-liner API."""

import os
import sys
from io import StringIO

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from torch_graph.explain import explain, ExplainResult


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_explain_basic():
    """explain() returns a fully populated ExplainResult."""
    torch.manual_seed(42)
    torch._dynamo.reset()
    model = SimpleMLP()
    x = torch.randn(2, 8)

    result = explain(model, x, verbose=False)

    assert isinstance(result, ExplainResult)
    assert result.model_name == "SimpleMLP"
    assert result.num_parameters > 0
    assert result.param_memory_mb > 0
    assert result.num_forward_ops > 0
    assert result.num_backward_ops > 0
    assert len(result.op_counts) > 0
    assert len(result.op_categories) > 0
    assert len(result.shapes) > 0
    assert result.capture is not None
    assert result.capture_time_s > 0
    assert result.verification is None
    assert result.profile_data is None


def test_explain_with_verify():
    """explain(verify=True) includes verification results."""
    torch.manual_seed(42)
    torch._dynamo.reset()
    model = SimpleMLP()
    x = torch.randn(2, 8)

    result = explain(model, x, verify=True, verbose=False)

    assert result.verification is not None
    # Forward verification should pass
    fwd = result.verification.get("forward")
    assert fwd is not None


def test_explain_summary_format():
    """Summary string contains expected sections."""
    torch.manual_seed(42)
    torch._dynamo.reset()
    model = SimpleMLP()
    x = torch.randn(2, 8)

    result = explain(model, x, verbose=False)
    summary = result.summary()

    assert "SimpleMLP" in summary
    assert "Parameters:" in summary
    assert "Forward ops:" in summary
    assert "Backward ops:" in summary
    assert "Top ops:" in summary
    assert "Op categories:" in summary


def test_explain_no_verbose(capsys):
    """verbose=False suppresses stdout output."""
    torch.manual_seed(42)
    torch._dynamo.reset()
    model = SimpleMLP()
    x = torch.randn(2, 8)

    result = explain(model, x, verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ""

    # verbose=True should produce output
    torch._dynamo.reset()
    result2 = explain(model, x, verbose=True)
    captured2 = capsys.readouterr()
    assert "SimpleMLP" in captured2.out
