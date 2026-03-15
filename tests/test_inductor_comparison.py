"""Inductor comparison tests.

Verify that captured aten graphs produce outputs matching what
torch.compile (inductor backend) computes. This validates that
the captured aten is a faithful representation of what the compiler
actually executes.
"""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_graph.auto_install as ai
from torch_graph.export import capture_aten_graphs


# ── Models ───────────────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    def __init__(self, d=128, n=3):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(d, d) for _ in range(n)])
        self.head = nn.Linear(d, 10)

    def forward(self, x, target):
        h = self.layers(x)
        return F.cross_entropy(self.head(h), target)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x, target):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h).flatten(1)
        return F.cross_entropy(self.fc(h), target)


class MLPWithBN(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.bn = nn.BatchNorm1d(d)
        self.fc2 = nn.Linear(d, 10)

    def forward(self, x, target):
        h = F.relu(self.bn(self.fc1(x)))
        return F.cross_entropy(self.fc2(h), target)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _cleanup():
    """Reset dynamo and auto_install between tests."""
    torch._dynamo.reset()
    ai.unpatch()
    ai._installed.clear()
    ai._optimizer_captured.clear()
    yield
    ai.unpatch()
    ai._installed.clear()
    ai._optimizer_captured.clear()
    torch._dynamo.reset()


# ── Helper ───────────────────────────────────────────────────────────────────

def _run_inductor(model, *inputs):
    """Run model through torch.compile with inductor backend, return output + grads."""
    torch._dynamo.reset()
    compiled = torch.compile(model, backend="inductor")
    out = compiled(*inputs)
    out.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return out.item(), grads


def _run_captured_aten(model, *inputs):
    """Run model through captured aten graphs, return output + grads."""
    torch._dynamo.reset()
    out, _cap = capture_aten_graphs(model, *inputs, run_backward=True)
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    if isinstance(out, torch.Tensor):
        return out.item(), grads
    return out, grads


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for inductor")
def test_inductor_vs_aten_mlp():
    """Captured aten forward matches inductor for MLP."""
    torch.manual_seed(42)
    model = SimpleMLP().cuda()
    x = torch.randn(8, 128, device="cuda")
    target = torch.randint(10, (8,), device="cuda")

    ind_loss, ind_grads = _run_inductor(copy.deepcopy(model), x, target)
    aten_loss, aten_grads = _run_captured_aten(copy.deepcopy(model), x, target)

    assert abs(ind_loss - aten_loss) < 1e-4, \
        f"Loss mismatch: inductor={ind_loss:.6f} aten={aten_loss:.6f}"

    for name in ind_grads:
        diff = (ind_grads[name] - aten_grads[name]).abs().max().item()
        assert diff < 1e-3, \
            f"Grad mismatch for {name}: {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for inductor")
def test_inductor_vs_aten_conv():
    """Captured aten forward matches inductor for ConvNet."""
    torch.manual_seed(42)
    model = ConvNet().cuda()
    x = torch.randn(4, 3, 16, 16, device="cuda")
    target = torch.randint(10, (4,), device="cuda")

    ind_loss, ind_grads = _run_inductor(copy.deepcopy(model), x, target)
    aten_loss, aten_grads = _run_captured_aten(copy.deepcopy(model), x, target)

    assert abs(ind_loss - aten_loss) < 1e-4, \
        f"Loss mismatch: inductor={ind_loss:.6f} aten={aten_loss:.6f}"

    for name in ind_grads:
        diff = (ind_grads[name] - aten_grads[name]).abs().max().item()
        assert diff < 1e-3, \
            f"Grad mismatch for {name}: {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for inductor")
def test_inductor_vs_aten_batchnorm():
    """Captured aten forward matches inductor for model with BatchNorm."""
    torch.manual_seed(42)
    model = MLPWithBN().cuda()
    x = torch.randn(8, 64, device="cuda")
    target = torch.randint(10, (8,), device="cuda")

    ind_loss, ind_grads = _run_inductor(copy.deepcopy(model), x, target)
    aten_loss, aten_grads = _run_captured_aten(copy.deepcopy(model), x, target)

    assert abs(ind_loss - aten_loss) < 1e-4, \
        f"Loss mismatch: inductor={ind_loss:.6f} aten={aten_loss:.6f}"

    for name in ind_grads:
        diff = (ind_grads[name] - aten_grads[name]).abs().max().item()
        assert diff < 1e-3, \
            f"Grad mismatch for {name}: {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for inductor")
def test_inductor_vs_aten_multi_step():
    """Multi-step training: captured aten + optimizer replay matches inductor."""
    torch.manual_seed(42)
    model = SimpleMLP(d=64, n=2).cuda()
    x = torch.randn(8, 64, device="cuda")
    target = torch.randint(10, (8,), device="cuda")

    # Run inductor for 3 steps
    ind_model = copy.deepcopy(model)
    torch._dynamo.reset()
    ind_compiled = torch.compile(ind_model, backend="inductor")
    ind_opt = torch.optim.SGD(ind_model.parameters(), lr=0.01)
    ind_losses = []
    for _ in range(3):
        ind_opt.zero_grad()
        loss = ind_compiled(x, target)
        loss.backward()
        ind_opt.step()
        ind_losses.append(loss.item())
    ind_state = {k: v.clone() for k, v in ind_model.state_dict().items()}

    # Run captured aten with optimizer replay for 3 steps
    import shutil
    from pathlib import Path
    cache_dir = Path(".torch_graph_cache/_test_inductor_multi")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    torch._dynamo.reset()
    ai.configure(
        cache_dir=str(cache_dir),
        capture_backward=True,
        force_recapture=True,
        verbose=False,
        capture_optimizer=True,
        replay_optimizer=True,
        dynamic=False,
    )
    ai.patch()

    aten_model = copy.deepcopy(model)
    aten_compiled = torch.compile(aten_model)
    aten_opt = torch.optim.SGD(aten_model.parameters(), lr=0.01)
    aten_losses = []
    for _ in range(3):
        aten_opt.zero_grad()
        loss = aten_compiled(x, target)
        loss.backward()
        aten_opt.step()
        aten_losses.append(loss.item())
    ai.unpatch()
    aten_state = {k: v.clone() for k, v in aten_model.state_dict().items()}

    # Compare losses (allow slightly more tolerance for inductor vs aten)
    for step, (il, al) in enumerate(zip(ind_losses, aten_losses)):
        assert abs(il - al) < 1e-3, \
            f"Step {step}: inductor={il:.6f} aten={al:.6f}"

    # Compare final parameters
    max_diff = max(
        (aten_state[k] - ind_state[k]).abs().max().item() for k in ind_state
    )
    assert max_diff < 1e-2, f"Param divergence after 3 steps: {max_diff:.2e}"

    if cache_dir.exists():
        shutil.rmtree(cache_dir)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for inductor")
def test_inductor_vs_aten_nanogpt():
    """NanoGPT: captured aten forward+backward matches inductor."""
    import sys, os
    test_repo = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_repo")
    if test_repo not in sys.path:
        sys.path.insert(0, test_repo)
    from model import NanoGPT

    class NanoGPTWithLoss(nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.gpt = NanoGPT(**kw)
        def forward(self, idx, target):
            logits = self.gpt(idx)
            return F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

    torch.manual_seed(42)
    model = NanoGPTWithLoss(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32).cuda()
    idx = torch.randint(64, (2, 16), device="cuda")
    target = torch.randint(64, (2, 16), device="cuda")

    ind_loss, ind_grads = _run_inductor(copy.deepcopy(model), idx, target)
    aten_loss, aten_grads = _run_captured_aten(copy.deepcopy(model), idx, target)

    assert abs(ind_loss - aten_loss) < 1e-4, \
        f"Loss mismatch: inductor={ind_loss:.6f} aten={aten_loss:.6f}"

    for name in ind_grads:
        diff = (ind_grads[name] - aten_grads[name]).abs().max().item()
        assert diff < 1e-3, \
            f"Grad mismatch for {name}: {diff:.2e}"
