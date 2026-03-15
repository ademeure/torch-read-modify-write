#!/usr/bin/env python3

import copy
import inspect
import os
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch_graph.auto_install as ai
from torch_graph.export import (
    _shorten_source_path,
    capture_aten_graphs,
    capture_optimizer_aten,
    export_graph_to_python,
)
from torch_graph.extract import extract_function, extract_training_step
from torch_graph.install import install


class SingleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture(autouse=True)
def _ensure_real_torch_compile():
    was_patched = ai._real_torch_compile is not None
    ai.unpatch()
    # Reset global state to prevent inter-test leaks
    ai._optimizer_captured.clear()
    ai._registered_optimizers.clear()
    ai._config.capture_optimizer = True
    try:
        yield
    finally:
        if was_patched:
            ai.patch()


def capture_and_install(model: nn.Module, sample_input: torch.Tensor) -> None:
    capture_model = copy.deepcopy(model)
    _, capture = capture_aten_graphs(capture_model, sample_input.clone(), run_backward=True)

    user_idx = 0
    fw_names = []
    for name in capture.primal_names:
        if name is None:
            fw_names.append(f"user_input_{user_idx}")
            user_idx += 1
        else:
            fw_names.append(name.replace(".", "_"))

    def aten_forward(*args):
        return capture.forward_graphs[0].graph_module(*args)

    aten_forward.__signature__ = inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in fw_names
    ])

    aten_mod = types.SimpleNamespace(
        forward=aten_forward,
        backward=capture.backward_graphs[0].graph_module.forward,
    )
    install(
        model,
        aten_mod,
        validate=False,
        param_paths=[name for name in capture.primal_names if name is not None],
    )


def test_capture_and_install_tracks_replaced_parameters():
    torch.manual_seed(0)
    model = SingleLinear()
    ref = copy.deepcopy(model)
    x = torch.randn(3, 4)

    capture_and_install(model, x)

    with torch.no_grad():
        ref.fc.weight = nn.Parameter(ref.fc.weight + 1.25)
        ref.fc.bias = nn.Parameter(ref.fc.bias - 0.5)
        model.fc.weight = nn.Parameter(model.fc.weight + 1.25)
        model.fc.bias = nn.Parameter(model.fc.bias - 0.5)

    out = model(x)
    expected = ref(x)
    assert torch.allclose(out, expected, atol=1e-5)

    out.sum().backward()
    assert model.fc.weight.grad is not None
    assert model.fc.bias.grad is not None


def test_capture_and_install_tracks_replaced_submodule():
    torch.manual_seed(0)
    model = SingleLinear()
    ref = copy.deepcopy(model)
    x = torch.randn(3, 4)

    capture_and_install(model, x)

    new_fc = nn.Linear(4, 2)
    with torch.no_grad():
        new_fc.weight.copy_(torch.randn_like(new_fc.weight))
        new_fc.bias.copy_(torch.randn_like(new_fc.bias))

    ref.fc = copy.deepcopy(new_fc)
    model.fc = copy.deepcopy(new_fc)

    out = model(x)
    expected = ref(x)
    assert torch.allclose(out, expected, atol=1e-5)

    out.sum().backward()
    assert model.fc.weight.grad is not None
    assert model.fc.bias.grad is not None


def test_export_graph_to_python_uses_compact_source_comments():
    def optimizer_style_fn(exp_avg_sq, grad, beta2_t):
        exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
        return exp_avg_sq

    _, capture = capture_aten_graphs(
        optimizer_style_fn,
        torch.zeros(4),
        torch.ones(4),
        torch.tensor(0.5),
        run_backward=False,
        dynamic=False,
    )

    code = export_graph_to_python(
        capture.forward_graphs[0].graph_module,
        annotate_sources=True,
        source_map=capture.source_map,
    )

    assert "# Source:" not in code
    assert "# ───" not in code
    assert __file__ not in code
    assert "# /test_install.py:" in code
    assert code.count("# exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)") == 1


def test_export_graph_to_python_uses_compact_module_headers():
    model = SingleLinear()

    _, capture = capture_aten_graphs(
        model,
        torch.randn(3, 4),
        run_backward=False,
        dynamic=False,
    )

    code = export_graph_to_python(
        capture.forward_graphs[0].graph_module,
        annotate_sources=True,
        source_map=capture.source_map,
    )

    assert "# Source:" not in code
    assert "# ───" not in code
    assert "# self.fc (Linear)" in code
    assert "# /test_install.py:" in code


def test_shorten_source_path_dist_packages_does_not_use_local_common_root():
    common_source_dir = "/root/torch-graph/outputs/repos/nanochat/nanochat"

    assert (
        _shorten_source_path(
            "/root/torch-graph/outputs/repos/nanochat/nanochat/optim.py",
            common_source_dir,
        )
        == "/optim.py"
    )
    assert (
        _shorten_source_path(
            "/usr/local/lib/python3.12/dist-packages/torch/nn/functional.py",
            common_source_dir,
        )
        == "/python3.12/dist-packages/torch/nn/functional.py"
    )


def test_load_aten_module_reloads_same_size_edit(tmp_path):
    path = tmp_path / "edited_aten.py"
    initial = "VALUE = 1\n\ndef forward():\n    return VALUE\n"
    edited = "VALUE = 2\n\ndef forward():\n    return VALUE\n"
    assert len(initial) == len(edited)

    path.write_text(initial)
    mod1 = ai._load_aten_module(path)
    assert mod1.forward() == 1

    path.write_text(edited)
    assert path.stat().st_size == len(initial)

    mod2 = ai._load_aten_module(path)
    assert mod2.forward() == 2


def test_extract_function_captures_standalone_fn(tmp_path):
    """Test extract_function with a plain function (not an nn.Module)."""
    def my_fn(x, w):
        return torch.nn.functional.linear(torch.relu(x), w)

    x = torch.randn(2, 4)
    w = torch.randn(3, 4)

    result = extract_function(
        my_fn, x, w,
        output_dir=str(tmp_path),
        prefix="standalone_fn",
    )

    assert "capture" in result
    assert "output" in result
    assert "files" in result
    assert len(result["files"]) >= 1

    # The aten .py file should exist and be valid Python
    py_files = [f for f in result["files"] if f.endswith(".py")]
    assert len(py_files) == 1
    with open(py_files[0]) as f:
        content = f.read()
    compile(content, py_files[0], "exec")

    # Output should match eager
    eager_out = my_fn(x, w)
    assert torch.allclose(result["output"], eager_out, atol=1e-5)


def test_extract_function_captures_nn_module(tmp_path):
    """Test extract_function with an nn.Module."""
    model = SingleLinear()
    x = torch.randn(2, 4)

    result = extract_function(
        model, x,
        output_dir=str(tmp_path),
        prefix="module_fn",
    )

    assert result["capture"].forward_graphs
    eager_out = model(x)
    assert torch.allclose(result["output"], eager_out, atol=1e-5)


def test_extract_training_step_with_optimizer_capture(tmp_path):
    """Test extract_training_step with capture_optimizer=True."""
    torch.manual_seed(42)
    model = SingleLinear()
    x = torch.randn(3, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Prime optimizer state with one step
    out = model(x)
    out.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    torch._dynamo.reset()
    result = extract_training_step(
        model=model,
        sample_args=(x,),
        loss_fn=lambda o: o.sum(),
        optimizer=optimizer,
        output_dir=str(tmp_path),
        prefix="opt_test",
        record_real_tensors=True,
        capture_optimizer=True,
    )

    capture = result["capture"]
    assert capture.forward_graphs, "No forward graphs captured"
    assert capture.backward_graphs, "No backward graphs captured"
    assert capture.optimizer_capture is not None, "Optimizer capture missing"
    assert capture.optimizer_capture.forward_graphs, "No optimizer graphs"

    # Verify optimizer aten graph has actual ops (not empty)
    opt_gm = capture.optimizer_capture.forward_graphs[0].graph_module
    opt_nodes = [n for n in opt_gm.graph.nodes if n.op == "call_function"]
    assert len(opt_nodes) > 5, f"Optimizer graph too small: {len(opt_nodes)} call_function nodes"

    # Verify .py file was generated and is valid Python
    py_files = [f for f in result["files"] if f.endswith(".py")]
    assert len(py_files) == 1
    with open(py_files[0]) as f:
        content = f.read()
    compile(content, py_files[0], "exec")

    # The exported file should contain an optimizer_step function
    assert "optimizer_step" in content, "Exported file missing optimizer_step"


def test_capture_optimizer_aten_uses_stable_slot_names():
    """Optimizer capture should use reliable group/param slot names by default."""
    torch.manual_seed(42)
    model = SingleLinear()
    x = torch.randn(3, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    out = model(x)
    out.sum().backward()
    capture = capture_optimizer_aten(optimizer, record_real_tensors=True)

    slot_info = capture.optimizer_slot_info[0]
    param_slots = [info for info in slot_info if info.get("role") == "param"]
    assert param_slots, "No optimizer param slots recorded"
    assert all(info["param_name"].startswith("group0.param") for info in param_slots)


def test_extract_training_step_with_custom_step_fn(tmp_path):
    """Test extract_training_step with a custom step_fn for non-standard optimizers."""
    torch.manual_seed(42)
    model = SingleLinear()
    x = torch.randn(3, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    step_calls = []

    def custom_step():
        step_calls.append(1)
        optimizer.step()

    torch._dynamo.reset()
    result = extract_training_step(
        model=model,
        sample_args=(x,),
        loss_fn=lambda o: o.sum(),
        optimizer=optimizer,
        output_dir=str(tmp_path),
        prefix="step_fn_test",
        record_real_tensors=True,
        capture_optimizer=True,
        step_fn=custom_step,
    )

    # custom_step should have been called (once for capture)
    assert len(step_calls) >= 1, f"step_fn was called {len(step_calls)} times, expected >= 1"
    assert result["capture"].optimizer_capture is not None
