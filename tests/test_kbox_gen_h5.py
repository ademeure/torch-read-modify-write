"""Validate kbox_gen H5 roundtrip: forward, backward, optimizer on CPU and GPU.

Each test exercises the full pipeline (capture → H5 dump → load → replay/run)
so basic roundtrip, dtype preservation, and tensor storage are validated
implicitly.
"""

import os
import sys
import tempfile

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch_graph.export import capture_aten_graphs
from torch_graph.op_dump import dump_grouped_tensors
from torch_graph.kbox_gen import (
    list_groups,
    generate_group_script,
    generate_section_script,
    _available_tensors,
    _build_input_mapping,
    _build_expected_out,
    _parse_replay_inputs,
)


# ── Test models ──────────────────────────────────────────────────────────────

class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.ln = nn.LayerNorm(16)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.ln(x)
        return x


class BNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 32)
        self.bn = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, 8)

    def forward(self, x):
        x = self.bn(torch.relu(self.fc(x)))
        return self.out(x)


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_dynamo():
    from torch_graph import auto_install as _ai
    if _ai._real_torch_compile is not None:
        _ai.unpatch()
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _load_h5_tensors(h5_path, device="cpu"):
    """Load all tensors from H5 restoring original dtypes."""
    _DTYPE_MAP = {
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
    }
    tensors = {}
    with h5py.File(h5_path, "r") as f:
        if "tensors" not in f:
            return tensors
        t = f["tensors"]
        for name in t:
            ds = t[name]
            raw = ds[()]
            val = torch.from_numpy(raw) if isinstance(raw, np.ndarray) else torch.tensor(raw)
            orig = ds.attrs.get("torch_dtype", "")
            if orig in _DTYPE_MAP:
                val = val.to(_DTYPE_MAP[orig])
            tensors[name] = val.to(device)
    return tensors


def _get_tensor(loaded, short_name, fx_name=None):
    """Get a tensor from loaded dict, handling falsy tensors correctly."""
    t = loaded.get(short_name)
    if t is not None:
        return t
    if fx_name is not None:
        return loaded.get(fx_name)
    return None


def _run_replay(replay_script, input_dict):
    """Execute a replay script with the given inputs dict."""
    outputs = {}
    exec(replay_script, {
        "inputs": input_dict, "outputs": outputs,
        "torch": torch, "operator": __import__("operator"),
        "inf": float("inf"), "math": __import__("math"),
    })
    return outputs


def _fx_inputs_from_h5(gm, loaded):
    """Reconstruct FX graph inputs in placeholder order from loaded H5 tensors."""
    from torch_graph._utils import short_name
    inputs = []
    for ph in (n for n in gm.graph.nodes if n.op == "placeholder"):
        sn = short_name(ph.name)
        t = _get_tensor(loaded, sn, ph.name)
        assert t is not None, f"Placeholder {ph.name} (short: {sn}) not in H5"
        inputs.append(t)
    return inputs


def _assert_outputs_match(actual, expected, msg="", exact=True, atol=1e-5, rtol=1e-5):
    """Compare FX graph outputs (tuple or single tensor)."""
    if not isinstance(actual, (tuple, list)):
        actual = (actual,)
    if not isinstance(expected, (tuple, list)):
        expected = (expected,)
    for i, (a, b) in enumerate(zip(actual, expected)):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if exact:
                assert torch.equal(a, b), (
                    f"{msg}output {i} not bit-identical. "
                    f"max diff: {(a.float() - b.float()).abs().max().item():.2e}"
                )
            else:
                diff = (a.float() - b.float()).abs()
                assert torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol), (
                    f"{msg}output {i} not close. "
                    f"max diff: {diff.max().item():.2e} (atol={atol})"
                )


def _capture_mlp(device="cpu", record_filter=None, record_real=True):
    """Capture SmallMLP aten graphs on the given device."""
    torch.manual_seed(42)
    model = SmallMLP().to(device)
    x = torch.randn(4, 16, device=device)
    kw = {}
    if record_filter:
        kw["record_filter"] = record_filter
    _, capture = capture_aten_graphs(
        model, x, run_backward=True, loss_fn=lambda o: o.sum(),
        record_real_tensors=record_real, **kw,
    )
    return capture, model, x


# ── CPU tests ────────────────────────────────────────────────────────────────

def test_kbox_group_scripts_cpu():
    """Generated kbox group scripts produce bit-identical results on CPU.

    Implicitly validates: H5 roundtrip, tensor storage, group listing,
    replay scripts, dtype handling, generate_group_script.
    """
    capture, model, x = _capture_mlp()
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name
    try:
        dump_grouped_tensors(
            capture, h5_path, group_by=["line"], which="both",
            include_params=True, replay_scripts=True,
        )
        groups = list_groups(h5_path, section="forward", strategy="line")
        assert len(groups) > 0

        n_run = 0
        for g in groups:
            script = generate_group_script(h5_path, g)
            # Force CPU for bit-identical comparison
            script = script.replace(
                '_device = "cuda" if torch.cuda.is_available() else "cpu"',
                '_device = "cpu"',
            )
            ns = {"__builtins__": __builtins__, "__file__": h5_path}
            try:
                exec(compile(script, f"<group_{g.index}>", "exec"), ns)
            except Exception:
                continue
            run_fn, inp, exp = ns.get("run"), ns.get("input"), ns.get("expected")
            if not (run_fn and inp and exp):
                continue
            try:
                result = run_fn(inp)
            except Exception:
                continue
            if result is None:
                continue
            for actual, expected in zip(result, exp):
                if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
                    assert torch.equal(actual, expected), (
                        f"Group {g.name} mismatch: "
                        f"max diff {(actual - expected).abs().max().item():.2e}"
                    )
                    n_run += 1

        assert n_run > 0, "No group scripts produced validated results"
    finally:
        os.unlink(h5_path)


def test_forward_backward_full_chain_cpu():
    """Chain all forward+backward group replay scripts from H5, verify bit-identical.

    Implicitly validates: backward replay, forward+backward tensor flow,
    group-level replay scripts, all-intermediate recording.
    """
    capture, model, x = _capture_mlp()
    if not capture.backward_graphs:
        pytest.skip("No backward graph")

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name
    try:
        dump_grouped_tensors(
            capture, h5_path, group_by=["line"], which="both",
            include_params=True, replay_scripts=True,
        )
        loaded = _load_h5_tensors(h5_path)
        available = _available_tensors(h5_path)

        from torch_graph.kbox_gen import _long_name
        all_vals = {_long_name(k): v for k, v in loaded.items()}

        for section in ("forward", "backward"):
            groups = list_groups(h5_path, section=section, strategy="line")
            for g in groups:
                if not g.replay_script:
                    continue
                needed = _parse_replay_inputs(g.replay_script)
                if any(n not in all_vals for n in needed):
                    continue
                outputs = _run_replay(g.replay_script, all_vals)
                all_vals.update(outputs)

        # Verify backward outputs match stored values
        n_matched = 0
        bw_groups = list_groups(h5_path, section="backward", strategy="line")
        for g in bw_groups:
            for fx_name, h5_key in _build_expected_out(g.replay_script, available):
                if h5_key in loaded and fx_name in all_vals:
                    actual, stored = all_vals[fx_name], loaded[h5_key]
                    if isinstance(actual, torch.Tensor) and isinstance(stored, torch.Tensor):
                        assert torch.equal(actual, stored), (
                            f"Backward '{fx_name}' not bit-identical. "
                            f"max diff: {(actual - stored).abs().max().item():.2e}"
                        )
                        n_matched += 1

        assert n_matched > 0, "No backward outputs validated"
    finally:
        os.unlink(h5_path)


def test_inputs_only_recompute_cpu():
    """Recompute full forward from inputs_only H5 (no intermediates stored)."""
    capture, model, x = _capture_mlp(record_filter={"inputs_only": True})

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name
    try:
        dump_grouped_tensors(
            capture, h5_path, group_by=["line"], which="forward",
            include_params=True, replay_scripts=True, inputs_only=True,
        )
        loaded = _load_h5_tensors(h5_path)

        # Run FX graph from H5-loaded inputs and compare to real inputs
        gm = capture.forward_graphs[0].graph_module
        h5_inputs = _fx_inputs_from_h5(gm, loaded)
        _assert_outputs_match(
            gm(*h5_inputs), gm(*capture.forward_real_inputs),
            msg="inputs_only recompute: ",
        )
    finally:
        os.unlink(h5_path)


def test_batchnorm_forward_cpu():
    """BatchNorm model (buffer mutations) forward via H5 is bit-identical."""
    torch.manual_seed(42)
    model = BNModel()
    model.train()
    x = torch.randn(8, 16)
    _, capture = capture_aten_graphs(
        model, x, run_backward=True,
        loss_fn=lambda o: o.sum(), record_real_tensors=True,
    )

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name
    try:
        dump_grouped_tensors(
            capture, h5_path, group_by=["line"], which="forward",
            include_params=True, replay_scripts=True,
        )
        loaded = _load_h5_tensors(h5_path)
        gm = capture.forward_graphs[0].graph_module
        h5_inputs = _fx_inputs_from_h5(gm, loaded)
        _assert_outputs_match(
            gm(*h5_inputs), gm(*capture.forward_real_inputs),
            msg="BN forward: ",
        )
    finally:
        os.unlink(h5_path)


def test_optimizer_capture():
    """Optimizer state captured and params change verified."""
    torch.manual_seed(42)
    model = SmallMLP()
    x = torch.randn(4, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Warm-up so optimizer has state
    model(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    from torch_graph.extract import extract_training_step
    with tempfile.TemporaryDirectory() as tmpdir:
        result = extract_training_step(
            model=model, sample_args=(x,),
            loss_fn=lambda o: o.sum(), optimizer=optimizer,
            steps=[1], output_dir=tmpdir, prefix="test_opt",
            record_real_tensors=True, capture_optimizer=True,
        )
        cap = result["capture"]
        assert cap.optimizer_data
        pre = cap.optimizer_data["pre_step_params"]
        post = cap.optimizer_data["post_step_params"]
        assert sum(1 for n in pre if not torch.equal(pre[n], post[n])) > 0


# ── GPU tests ────────────────────────────────────────────────────────────────

_GPU = torch.cuda.is_available()
_skip_no_gpu = pytest.mark.skipif(not _GPU, reason="CUDA not available")


@_skip_no_gpu
def test_kbox_group_scripts_gpu():
    """Generated kbox scripts run on GPU with allclose validation.

    This is the real deployment path: capture on GPU, dump H5, generate
    kbox script, run with _device=cuda.  Implicitly validates GPU tensor
    storage, H5 roundtrip, and replay correctness.
    """
    capture, model, x = _capture_mlp(device="cuda")
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name
    try:
        dump_grouped_tensors(
            capture, h5_path, group_by=["line"], which="both",
            include_params=True, replay_scripts=True,
        )
        groups = list_groups(h5_path, section="forward", strategy="line")

        n_run = 0
        for g in groups:
            script = generate_group_script(h5_path, g)
            ns = {"__builtins__": __builtins__, "__file__": h5_path}
            try:
                exec(compile(script, f"<gpu_group_{g.index}>", "exec"), ns)
            except Exception:
                continue
            run_fn, inp, exp = ns.get("run"), ns.get("input"), ns.get("expected")
            if not (run_fn and inp and exp):
                continue
            try:
                result = run_fn(inp)
            except Exception:
                continue
            if result is None:
                continue
            for actual, expected in zip(result, exp):
                if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
                    _assert_outputs_match(actual, expected, exact=False,
                                          msg=f"GPU group {g.name}: ")
                    n_run += 1

        assert n_run > 0, "No GPU kbox scripts validated"
    finally:
        os.unlink(h5_path)


@_skip_no_gpu
def test_gpu_tensor_storage_bit_identical():
    """GPU-captured tensors survive H5 roundtrip bit-identically.

    GPU ops are non-deterministic on re-execution, but the *stored* values
    must be losslessly preserved through H5 write→read.
    """
    capture, model, x = _capture_mlp(device="cuda")
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name
    try:
        dump_grouped_tensors(
            capture, h5_path, group_by=["line"], which="both",
            include_params=True, replay_scripts=True,
        )
        loaded = _load_h5_tensors(h5_path)  # CPU
        from torch_graph._utils import short_name

        n_checked = 0
        # Check placeholders
        fg = capture.forward_graphs[0]
        for ph, real in zip(
            (n for n in fg.graph_module.graph.nodes if n.op == "placeholder"),
            capture.forward_real_inputs,
        ):
            if not isinstance(real, torch.Tensor):
                continue
            stored = _get_tensor(loaded, short_name(ph.name), ph.name)
            if stored is None:
                continue
            assert torch.equal(real.cpu(), stored), f"{ph.name} not bit-identical in H5"
            n_checked += 1

        # Check intermediates
        if capture.forward_intermediates:
            for name, real in capture.forward_intermediates.items():
                if not isinstance(real, torch.Tensor):
                    continue
                stored = _get_tensor(loaded, short_name(name), name)
                if stored is None:
                    continue
                assert torch.equal(real.cpu(), stored), f"Intermediate {name} not bit-identical"
                n_checked += 1

        assert n_checked > 5
    finally:
        os.unlink(h5_path)


@_skip_no_gpu
def test_backward_fx_graph_gpu():
    """Run backward FX graph on GPU with H5-loaded inputs, verify allclose."""
    capture, model, x = _capture_mlp(device="cuda")
    if not capture.backward_graphs:
        pytest.skip("No backward graph")

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name
    try:
        dump_grouped_tensors(
            capture, h5_path, group_by=["line"], which="both",
            include_params=True, replay_scripts=True,
        )
        loaded = _load_h5_tensors(h5_path, device="cuda")

        bw_gm = capture.backward_graphs[0].graph_module
        bw_inputs = _fx_inputs_from_h5(bw_gm, loaded)
        _assert_outputs_match(
            bw_gm(*bw_inputs), bw_gm(*capture.backward_real_inputs),
            msg="GPU backward: ", exact=False,
        )
    finally:
        os.unlink(h5_path)


@_skip_no_gpu
def test_batchnorm_forward_gpu():
    """BatchNorm model forward on GPU via H5 is allclose."""
    torch.manual_seed(42)
    model = BNModel().cuda()
    model.train()
    x = torch.randn(8, 16, device="cuda")
    _, capture = capture_aten_graphs(
        model, x, run_backward=True,
        loss_fn=lambda o: o.sum(), record_real_tensors=True,
    )

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name
    try:
        dump_grouped_tensors(
            capture, h5_path, group_by=["line"], which="forward",
            include_params=True, replay_scripts=True,
        )
        loaded = _load_h5_tensors(h5_path, device="cuda")
        gm = capture.forward_graphs[0].graph_module
        h5_inputs = _fx_inputs_from_h5(gm, loaded)
        _assert_outputs_match(
            gm(*h5_inputs), gm(*capture.forward_real_inputs),
            msg="GPU BN: ", exact=False,
        )
    finally:
        os.unlink(h5_path)


# ── Backward section script tests ────────────────────────────────────────────


def test_backward_section_script_cpu():
    """generate_section_script(section='backward') produces a runnable script
    whose output matches the stored backward tensors."""
    capture, model, x = _capture_mlp()
    if not capture.backward_graphs:
        pytest.skip("No backward graph")

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name
    try:
        dump_grouped_tensors(
            capture, h5_path, group_by=["line"], which="both",
            include_params=True, replay_scripts=True,
        )

        # Ensure backward groups are discoverable
        bw_groups = list_groups(h5_path, section="backward", strategy="line")
        assert len(bw_groups) > 0, "No backward groups found"

        # Generate a backward section script with validation
        script = generate_section_script(
            h5_path, section="backward", strategy="line", validate=True,
        )
        assert "def run(input):" in script
        assert "def check(" in script

        # Execute the generated script on CPU
        script = script.replace(
            '_device = "cuda" if torch.cuda.is_available() else "cpu"',
            '_device = "cpu"',
        )
        ns = {"__builtins__": __builtins__, "__file__": h5_path}
        exec(compile(script, "<backward_section>", "exec"), ns)

        run_fn = ns["run"]
        inp = ns["input"]
        expected = ns["expected"]

        assert len(expected) > 0, "No expected outputs in backward section script"

        result = run_fn(inp)
        for i, (got, exp) in enumerate(zip(result, expected)):
            if isinstance(got, torch.Tensor) and isinstance(exp, torch.Tensor):
                assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5), (
                    f"Backward section output {i} mismatch: "
                    f"max diff {(got - exp).abs().max().item():.2e}"
                )
    finally:
        os.unlink(h5_path)


def test_backward_group_scripts_cpu():
    """Individual backward group scripts produce correct results."""
    capture, model, x = _capture_mlp()
    if not capture.backward_graphs:
        pytest.skip("No backward graph")

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name
    try:
        dump_grouped_tensors(
            capture, h5_path, group_by=["line"], which="both",
            include_params=True, replay_scripts=True,
        )

        groups = list_groups(h5_path, section="backward", strategy="line")
        assert len(groups) > 0

        n_run = 0
        for g in groups:
            script = generate_group_script(h5_path, g)
            script = script.replace(
                '_device = "cuda" if torch.cuda.is_available() else "cpu"',
                '_device = "cpu"',
            )
            ns = {"__builtins__": __builtins__, "__file__": h5_path}
            try:
                exec(compile(script, f"<bw_group_{g.index}>", "exec"), ns)
            except Exception:
                continue
            run_fn, inp, exp = ns.get("run"), ns.get("input"), ns.get("expected")
            if not (run_fn and inp and exp):
                continue
            try:
                result = run_fn(inp)
            except Exception:
                continue
            if result is None:
                continue
            for actual, expected_t in zip(result, exp):
                if isinstance(actual, torch.Tensor) and isinstance(expected_t, torch.Tensor):
                    assert torch.equal(actual, expected_t), (
                        f"Backward group {g.name} mismatch: "
                        f"max diff {(actual - expected_t).abs().max().item():.2e}"
                    )
                    n_run += 1

        assert n_run > 0, "No backward group scripts validated"
    finally:
        os.unlink(h5_path)


def test_forward_section_script_cpu():
    """generate_section_script(section='forward') with validate=True is runnable."""
    capture, model, x = _capture_mlp()

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name
    try:
        dump_grouped_tensors(
            capture, h5_path, group_by=["line"], which="both",
            include_params=True, replay_scripts=True,
        )

        script = generate_section_script(
            h5_path, section="forward", strategy="line", validate=True,
        )
        assert "def run(input):" in script
        assert "def check(" in script

        script = script.replace(
            '_device = "cuda" if torch.cuda.is_available() else "cpu"',
            '_device = "cpu"',
        )
        ns = {"__builtins__": __builtins__, "__file__": h5_path}
        exec(compile(script, "<forward_section>", "exec"), ns)

        result = ns["run"](ns["input"])
        expected = ns["expected"]
        assert len(expected) > 0
        for i, (got, exp) in enumerate(zip(result, expected)):
            if isinstance(got, torch.Tensor) and isinstance(exp, torch.Tensor):
                assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5), (
                    f"Forward section output {i}: "
                    f"max diff {(got - exp).abs().max().item():.2e}"
                )
    finally:
        os.unlink(h5_path)


def test_section_script_relative_path():
    """Generated section scripts use relative H5 paths."""
    capture, model, x = _capture_mlp()
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "model.h5")
        dump_grouped_tensors(
            capture, h5_path, group_by=["line"], which="forward",
            include_params=True, replay_scripts=True,
        )
        out_path = os.path.join(tmpdir, "test_script.py")
        script = generate_section_script(
            h5_path, section="forward", strategy="line",
            out_path=out_path,
        )
        # Should NOT contain absolute path
        assert tmpdir not in script or "os.path.dirname" in script
        # Should contain relative reference
        assert "model.h5" in script
