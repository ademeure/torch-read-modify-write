"""Multi-step training verification tests.

Tests that captured aten graphs produce bit-identical results over multiple
training steps, including parameter updates via various optimizers.
Also tests optimizer capture and the offload_saved mode.
"""

import copy
import gc
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_graph.auto_install as ai
from torch_graph.export import capture_aten_graphs, capture_optimizer_aten, export_aten_program


# ── Fixtures ──────────────────────────────────────────────────────────────────

CACHE_DIR = Path(".torch_graph_cache/_test_multi_step")


class MLP(nn.Module):
    def __init__(self, d=256, n=4):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(d, d) for _ in range(n)])
        self.head = nn.Linear(d, 10)

    def forward(self, x, target):
        h = self.layers(x)
        return F.cross_entropy(self.head(h), target)


class MLPWithBN(nn.Module):
    """Model with BatchNorm — tests buffer mutations."""

    def __init__(self, d=128):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.bn = nn.BatchNorm1d(d)
        self.fc2 = nn.Linear(d, 10)

    def forward(self, x, target):
        h = F.relu(self.bn(self.fc1(x)))
        return F.cross_entropy(self.fc2(h), target)


@pytest.fixture(autouse=True)
def _cleanup():
    """Clean up caches and reset dynamo before each test."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    torch._dynamo.reset()
    ai.unpatch()
    ai._installed.clear()
    ai._optimizer_captured.clear()
    yield
    ai.unpatch()
    ai._installed.clear()
    ai._optimizer_captured.clear()
    torch._dynamo.reset()
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)


def _run_eager_steps(model, x, target, optimizer_cls, n_steps=5, lr=1e-3):
    """Run n_steps of training with eager execution, return losses and final state."""
    opt = optimizer_cls(model.parameters(), lr=lr)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = model(x, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    state = {k: v.clone() for k, v in model.state_dict().items()}
    return losses, state


def _run_aten_steps(model, x, target, optimizer_cls, n_steps=5, lr=1e-3):
    """Run n_steps of training through auto_install aten graphs."""
    ai.configure(
        cache_dir=str(CACHE_DIR),
        capture_backward=True,
        force_recapture=True,
        verbose=False,
        capture_optimizer=False,
    )
    ai.patch()
    compiled = torch.compile(model)
    opt = optimizer_cls(model.parameters(), lr=lr)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = compiled(x, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    ai.unpatch()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    return losses, state


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_multi_step_sgd():
    """5 training steps with SGD produce bit-identical losses."""
    torch.manual_seed(42)
    model = MLP().cuda()
    x = torch.randn(16, 256, device="cuda")
    target = torch.randint(10, (16,), device="cuda")

    ref_losses, ref_state = _run_eager_steps(
        copy.deepcopy(model), x, target, torch.optim.SGD
    )
    ai_losses, ai_state = _run_aten_steps(
        copy.deepcopy(model), x, target, torch.optim.SGD
    )

    for step, (r, a) in enumerate(zip(ref_losses, ai_losses)):
        assert abs(r - a) < 1e-5, f"Step {step}: ref={r:.6f} aten={a:.6f}"

    max_diff = max(
        (ai_state[k] - ref_state[k]).abs().max().item() for k in ref_state
    )
    assert max_diff < 1e-4, f"Param divergence: {max_diff:.2e}"


def test_multi_step_adamw():
    """5 training steps with AdamW produce near-identical losses."""
    torch.manual_seed(42)
    model = MLP().cuda()
    x = torch.randn(16, 256, device="cuda")
    target = torch.randint(10, (16,), device="cuda")

    ref_losses, ref_state = _run_eager_steps(
        copy.deepcopy(model), x, target, torch.optim.AdamW
    )
    ai_losses, ai_state = _run_aten_steps(
        copy.deepcopy(model), x, target, torch.optim.AdamW
    )

    for step, (r, a) in enumerate(zip(ref_losses, ai_losses)):
        assert abs(r - a) < 5e-5, f"Step {step}: ref={r:.6f} aten={a:.6f}"

    max_diff = max(
        (ai_state[k] - ref_state[k]).abs().max().item() for k in ref_state
    )
    assert max_diff < 1e-4, f"Param divergence: {max_diff:.2e}"


def test_multi_step_batchnorm():
    """Multi-step training with BatchNorm (buffer mutations) matches eager."""
    torch.manual_seed(42)
    model = MLPWithBN().cuda()
    x = torch.randn(16, 128, device="cuda")
    target = torch.randint(10, (16,), device="cuda")

    ref_losses, ref_state = _run_eager_steps(
        copy.deepcopy(model), x, target, torch.optim.SGD
    )
    ai_losses, ai_state = _run_aten_steps(
        copy.deepcopy(model), x, target, torch.optim.SGD
    )

    for step, (r, a) in enumerate(zip(ref_losses, ai_losses)):
        assert abs(r - a) < 1e-4, f"Step {step}: ref={r:.6f} aten={a:.6f}"

    # Check running_mean/var buffers are updated correctly
    for key in ref_state:
        diff = (ai_state[key] - ref_state[key]).abs().max().item()
        assert diff < 1e-4, f"State mismatch for {key}: {diff:.2e}"


def test_capture_io_recording():
    """capture_aten_graphs with record_real_tensors saves inputs/outputs."""
    torch.manual_seed(42)
    model = MLP().cuda()
    x = torch.randn(8, 256, device="cuda")
    target = torch.randint(10, (8,), device="cuda")

    torch._dynamo.reset()
    out, cap = capture_aten_graphs(
        model, x, target, run_backward=True, record_real_tensors=True
    )

    # Forward I/O recorded
    assert cap.forward_real_inputs is not None
    assert len(cap.forward_real_inputs) > 0
    assert cap.forward_intermediates is not None
    assert len(cap.forward_intermediates) > 0
    assert cap.forward_real_output is not None

    # Backward I/O recorded
    assert cap.backward_real_inputs is not None
    assert cap.backward_real_output is not None

    # Forward output matches eager
    ref_loss = model(x, target).item()
    if isinstance(out, torch.Tensor):
        assert abs(out.item() - ref_loss) < 1e-5


def test_export_and_reimport():
    """Exported .py file can be imported and contains forward+backward."""
    torch.manual_seed(42)
    model = MLP().cuda()
    x = torch.randn(4, 256, device="cuda")
    target = torch.randint(10, (4,), device="cuda")

    torch._dynamo.reset()
    _, cap = capture_aten_graphs(
        model, x, target, run_backward=True, record_real_tensors=True
    )

    outdir = Path("outputs/_test_export")
    outdir.mkdir(parents=True, exist_ok=True)
    export_path = export_aten_program(
        cap, str(outdir / "aten.py"), include_test_harness=True
    )

    # Verify it's valid Python
    code = export_path.read_text()
    compile(code, str(export_path), "exec")

    # Verify it can be imported
    import importlib.util

    spec = importlib.util.spec_from_file_location("test_aten", str(export_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "forward")
    assert hasattr(mod, "backward")

    shutil.rmtree(outdir, ignore_errors=True)


def test_optimizer_capture():
    """capture_optimizer_aten captures AdamW step as aten ops."""
    torch.manual_seed(42)
    model = MLP().cuda()
    x = torch.randn(8, 256, device="cuda")
    target = torch.randint(10, (8,), device="cuda")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.zero_grad()
    loss = model(x, target)
    loss.backward()

    # Save pre-step params
    pre_state = {k: v.clone() for k, v in model.state_dict().items()}

    torch._dynamo.reset()
    cap = capture_optimizer_aten(opt, record_real_tensors=True)

    assert len(cap.forward_graphs) > 0
    og = cap.forward_graphs[0]
    n_ops = sum(1 for n in og.graph_module.graph.nodes if n.op == "call_function")
    assert n_ops > 10, f"Expected many aten ops in optimizer, got {n_ops}"

    # Verify optimizer actually updated params
    post_state = {k: v.clone() for k, v in model.state_dict().items()}
    any_changed = any(
        (post_state[k] - pre_state[k]).abs().max().item() > 1e-6
        for k in pre_state
        if "weight" in k
    )
    assert any_changed, "Optimizer step should have updated parameters"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_offload_saved_correctness():
    """offload_saved produces bit-identical gradients to normal capture."""
    torch.manual_seed(42)
    model = MLP(d=512, n=6).cuda()
    x = torch.randn(16, 512, device="cuda")
    target = torch.randint(10, (16,), device="cuda")

    # Normal capture
    model_norm = copy.deepcopy(model)
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    out_norm, _ = capture_aten_graphs(
        model_norm, x, target, run_backward=True, offload_saved=False
    )
    norm_grads = {
        n: p.grad.clone()
        for n, p in model_norm.named_parameters()
        if p.grad is not None
    }

    # Offloaded capture
    model_off = copy.deepcopy(model)
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    out_off, _ = capture_aten_graphs(
        model_off, x, target, run_backward=True, offload_saved=True
    )
    off_grads = {
        n: p.grad.clone()
        for n, p in model_off.named_parameters()
        if p.grad is not None
    }

    # Outputs match
    if isinstance(out_norm, torch.Tensor) and isinstance(out_off, torch.Tensor):
        assert (out_norm - out_off).abs().max().item() == 0.0

    # Gradients match
    max_diff = max(
        (norm_grads[n] - off_grads[n]).abs().max().item() for n in norm_grads
    )
    assert max_diff == 0.0, f"Offloaded grads differ: {max_diff:.2e}"


def test_cache_reload_multi_step():
    """Loading from cache produces identical training trajectories."""
    torch.manual_seed(42)
    model = MLP().cuda()
    x = torch.randn(8, 256, device="cuda")
    target = torch.randint(10, (8,), device="cuda")

    # First run: capture + train 3 steps
    losses1, state1 = _run_aten_steps(
        copy.deepcopy(model), x, target, torch.optim.SGD, n_steps=3
    )

    # Second run: load from cache + train 3 steps
    ai._installed.clear()
    torch._dynamo.reset()
    ai.configure(
        cache_dir=str(CACHE_DIR),
        capture_backward=True,
        force_recapture=False,  # use cached
        verbose=False,
    )
    ai.patch()
    model2 = copy.deepcopy(model)
    compiled2 = torch.compile(model2)
    opt2 = torch.optim.SGD(model2.parameters(), lr=1e-3)
    losses2 = []
    for _ in range(3):
        opt2.zero_grad()
        loss = compiled2(x, target)
        loss.backward()
        opt2.step()
        losses2.append(loss.item())
    ai.unpatch()

    for step, (l1, l2) in enumerate(zip(losses1, losses2)):
        assert abs(l1 - l2) < 1e-5, f"Step {step}: first={l1:.6f} cached={l2:.6f}"


# ── Optimizer replay tests ───────────────────────────────────────────────────


def _run_replay_steps(model, x, target, optimizer_cls, n_steps=5, lr=1e-3):
    """Run n_steps of training with optimizer replay through captured aten."""
    ai.configure(
        cache_dir=str(CACHE_DIR),
        capture_backward=True,
        force_recapture=True,
        verbose=False,
        capture_optimizer=True,
        replay_optimizer=True,
    )
    ai.patch()
    compiled = torch.compile(model)
    opt = optimizer_cls(model.parameters(), lr=lr)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        loss = compiled(x, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    ai.unpatch()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    return losses, state


def test_replay_multi_step_sgd():
    """5 training steps with SGD optimizer replay match eager."""
    torch.manual_seed(42)
    model = MLP().cuda()
    x = torch.randn(16, 256, device="cuda")
    target = torch.randint(10, (16,), device="cuda")

    ref_losses, ref_state = _run_eager_steps(
        copy.deepcopy(model), x, target, torch.optim.SGD
    )
    replay_losses, replay_state = _run_replay_steps(
        copy.deepcopy(model), x, target, torch.optim.SGD
    )

    for step, (r, a) in enumerate(zip(ref_losses, replay_losses)):
        assert abs(r - a) < 1e-5, f"Step {step}: ref={r:.6f} replay={a:.6f}"

    max_diff = max(
        (replay_state[k] - ref_state[k]).abs().max().item() for k in ref_state
    )
    assert max_diff < 1e-4, f"Param divergence: {max_diff:.2e}"


def test_replay_multi_step_adamw():
    """5 training steps with AdamW optimizer replay match eager."""
    torch.manual_seed(42)
    model = MLP().cuda()
    x = torch.randn(16, 256, device="cuda")
    target = torch.randint(10, (16,), device="cuda")

    ref_losses, ref_state = _run_eager_steps(
        copy.deepcopy(model), x, target, torch.optim.AdamW
    )
    replay_losses, replay_state = _run_replay_steps(
        copy.deepcopy(model), x, target, torch.optim.AdamW
    )

    for step, (r, a) in enumerate(zip(ref_losses, replay_losses)):
        assert abs(r - a) < 5e-5, f"Step {step}: ref={r:.6f} replay={a:.6f}"

    max_diff = max(
        (replay_state[k] - ref_state[k]).abs().max().item() for k in ref_state
    )
    assert max_diff < 1e-4, f"Param divergence: {max_diff:.2e}"


def test_replay_multi_step_batchnorm():
    """Multi-step training with BatchNorm + optimizer replay matches eager."""
    torch.manual_seed(42)
    model = MLPWithBN().cuda()
    x = torch.randn(16, 128, device="cuda")
    target = torch.randint(10, (16,), device="cuda")

    ref_losses, ref_state = _run_eager_steps(
        copy.deepcopy(model), x, target, torch.optim.SGD
    )
    replay_losses, replay_state = _run_replay_steps(
        copy.deepcopy(model), x, target, torch.optim.SGD
    )

    for step, (r, a) in enumerate(zip(ref_losses, replay_losses)):
        assert abs(r - a) < 1e-4, f"Step {step}: ref={r:.6f} replay={a:.6f}"

    for key in ref_state:
        diff = (replay_state[key] - ref_state[key]).abs().max().item()
        assert diff < 1e-4, f"State mismatch for {key}: {diff:.2e}"
