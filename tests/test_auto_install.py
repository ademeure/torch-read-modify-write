#!/usr/bin/env python3
"""Thorough tests for torch_graph.auto_install.

Tests auto_install with progressively complex models:
  1. Single linear layer
  2. MLP (multiple layers, relu)
  3. Model with buffers (BatchNorm)
  4. Model with kwargs (targets)
  5. Conv + Pool model
  6. Multi-output model
  7. Model with dropout (training mode)
  8. LSTM-based model (via fallback)
  9. Training loop (multiple steps with optimizer)
 10. Nanochat GPT (real-world model)
 11. Cache reload correctness
 12. force_recapture flag
 13. Nested compiled submodules
"""

import os
import shutil
import sys
import tempfile
import traceback

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

# We need to import auto_install BEFORE any torch.compile calls.
# But we want per-test isolation, so we'll unpatch/repatch around each test.
import torch_graph.auto_install as ai

CACHE_DIR: str | None = None
RESULTS = []


def setup_test(test_name, cache_dir=None, **config_overrides):
    """Reset state for a new test."""
    global CACHE_DIR

    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix=f"torch_graph_{test_name}_")
    cache_dir = os.fspath(cache_dir)

    # Clean cache
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    CACHE_DIR = cache_dir

    # Reset internal state
    try:
        torch.compiler.reset()
    except Exception:
        pass
    ai._installed.clear()
    ai._optimizer_captured.clear()
    ai._registered_optimizers.clear()

    # Reset ALL config fields to defaults (prevents leaks between tests)
    defaults = ai.AutoInstallConfig()
    ai.configure(
        cache_dir=cache_dir,
        verbose=False,
        force_recapture=defaults.force_recapture,
        validate_shapes=defaults.validate_shapes,
        capture_backward=defaults.capture_backward,
        loss_fn=defaults.loss_fn,
        num_real_outputs=defaults.num_real_outputs,
        dynamic=False,
        record_real_tensors=defaults.record_real_tensors,
        generate_graph=defaults.generate_graph,
        dump_h5=defaults.dump_h5,
        dump_h5_functions=defaults.dump_h5_functions,
        skip_pt=defaults.skip_pt,
        exit_after_capture=defaults.exit_after_capture,
        capture_batch_size=defaults.capture_batch_size,
        use_inductor=defaults.use_inductor,
        offload_saved=defaults.offload_saved,
        capture_optimizer=defaults.capture_optimizer,
        replay_optimizer=defaults.replay_optimizer,
        verify_steps=defaults.verify_steps,
        record_steps=defaults.record_steps,
    )
    # Apply overrides
    for k, v in config_overrides.items():
        ai.configure(**{k: v})

    # Ensure patched
    ai.unpatch()
    ai.patch()

    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")


def record_result(name, passed, error=None):
    RESULTS.append((name, passed, error))
    if passed:
        print(f"  PASS: {name}")
    else:
        print(f"  FAIL: {name}")
        if error:
            print(f"    Error: {error}")


def run_test(name, fn, **config_overrides):
    """Run a test function with setup/teardown."""
    setup_test(name, **config_overrides)
    try:
        fn()
        record_result(name, True)
    except Exception as e:
        traceback.print_exc()
        record_result(name, False, str(e))


@pytest.fixture(autouse=True)
def _pytest_setup(request, tmp_path):
    """Mirror the script runner's per-test setup when run under pytest."""
    setup_test(request.node.name, cache_dir=tmp_path / ".torch_graph_cache")


def _only_cache_py() -> str:
    py_files = sorted(
        name for name in os.listdir(CACHE_DIR)
        if name.endswith(".py")
    )
    assert len(py_files) == 1, py_files
    return os.path.join(CACHE_DIR, py_files[0])


# ═══════════════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════════════

class SingleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class MLPWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        return self.fc2(F.relu(self.bn(self.fc1(x))))

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

class ModelWithKwargs(nn.Module):
    """Model that takes idx and targets kwargs, like a language model."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.fc = nn.Linear(16, 32)
    def forward(self, idx, targets=None):
        x = self.embed(idx)
        logits = self.fc(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, 32), targets.view(-1))
            return loss
        return logits

class MultiHeadModel(nn.Module):
    """Model with multiple output heads."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(8, 16)
        self.head1 = nn.Linear(16, 4)
        self.head2 = nn.Linear(16, 4)
    def forward(self, x):
        h = F.relu(self.shared(x))
        return self.head1(h), self.head2(h)

class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.fc1(x))))

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        return self.ln(x + self.fc2(F.relu(self.fc1(x))))

class ResNet(nn.Module):
    """Mini residual network."""
    def __init__(self, dim=16, n_blocks=3):
        super().__init__()
        self.embed = nn.Linear(8, dim)
        self.blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, 4)
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

class TinyTransformer(nn.Module):
    """Minimal transformer encoder."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(8, 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64,
            dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(32, 4)
    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


# ═══════════════════════════════════════════════════════════════════════
# Test functions
# ═══════════════════════════════════════════════════════════════════════

def test_single_linear():
    model = SingleLinear()
    x = torch.randn(2, 4)
    ref = model(x).clone()

    model2 = SingleLinear()
    model2.load_state_dict(model.state_dict())
    compiled = torch.compile(model2)
    out = compiled(x)

    diff = (ref - out).abs().max().item()
    assert diff < 1e-5, f"Forward mismatch: {diff}"

    # Test backward
    out.sum().backward()
    assert model2.fc.weight.grad is not None, "No gradient on weight"
    assert model2.fc.bias.grad is not None, "No gradient on bias"


def test_tiny_mlp():
    model = TinyMLP()
    x = torch.randn(2, 8)
    ref = model(x).clone()

    model2 = TinyMLP()
    model2.load_state_dict(model.state_dict())
    compiled = torch.compile(model2)
    out = compiled(x)

    diff = (ref - out).abs().max().item()
    assert diff < 1e-5, f"Forward mismatch: {diff}"

    out.sum().backward()
    assert model2.fc1.weight.grad is not None


def test_batchnorm():
    torch.manual_seed(42)
    model = MLPWithBatchNorm()
    model.train()
    x = torch.randn(4, 8)  # batch_size=4 for batchnorm

    # Save state BEFORE running reference to avoid running_stats drift
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    ref = model(x).clone()

    model2 = MLPWithBatchNorm()
    model2.load_state_dict(init_state)
    model2.train()
    compiled = torch.compile(model2)
    out = compiled(x)

    diff = (ref - out).abs().max().item()
    assert diff < 1e-4, f"Forward mismatch: {diff}"

    out.sum().backward()
    assert model2.fc1.weight.grad is not None


def test_conv_model():
    model = ConvModel()
    x = torch.randn(2, 1, 8, 8)
    ref = model(x).clone()

    model2 = ConvModel()
    model2.load_state_dict(model.state_dict())
    compiled = torch.compile(model2)
    out = compiled(x)

    diff = (ref - out).abs().max().item()
    assert diff < 1e-5, f"Forward mismatch: {diff}"

    out.sum().backward()
    assert model2.conv1.weight.grad is not None


def test_model_with_kwargs():
    """Test model that uses kwargs (idx, targets) like a language model."""
    torch.manual_seed(42)
    model = ModelWithKwargs()
    idx = torch.randint(0, 32, (2, 4))
    targets = torch.randint(0, 32, (2, 4))

    ref_loss = model(idx, targets=targets).clone()

    model2 = ModelWithKwargs()
    model2.load_state_dict(model.state_dict())
    compiled = torch.compile(model2)
    loss = compiled(idx, targets=targets)

    diff = (ref_loss - loss).abs().item()
    assert diff < 1e-5, f"Loss mismatch: {diff}"

    loss.backward()
    assert model2.embed.weight.grad is not None


def test_multi_output():
    """Test model with multiple output heads."""
    model = MultiHeadModel()
    x = torch.randn(2, 8)
    ref1, ref2 = model(x)
    ref1 = ref1.clone()
    ref2 = ref2.clone()

    model2 = MultiHeadModel()
    model2.load_state_dict(model.state_dict())
    # multi-output needs num_real_outputs=2
    ai.configure(num_real_outputs=2)
    compiled = torch.compile(model2)
    out1, out2 = compiled(x)

    diff1 = (ref1 - out1).abs().max().item()
    diff2 = (ref2 - out2).abs().max().item()
    assert diff1 < 1e-5, f"Head 1 mismatch: {diff1}"
    assert diff2 < 1e-5, f"Head 2 mismatch: {diff2}"


def test_dropout():
    """Test model with dropout (training mode)."""
    torch.manual_seed(42)
    model = ModelWithDropout()
    model.train()
    x = torch.randn(2, 8)

    model2 = ModelWithDropout()
    model2.load_state_dict(model.state_dict())
    model2.train()
    compiled = torch.compile(model2)
    # With dropout, outputs won't match exactly, but should be reasonable
    out = compiled(x)
    assert out.shape == (2, 4), f"Wrong shape: {out.shape}"

    out.sum().backward()
    assert model2.fc1.weight.grad is not None


def test_residual_network():
    """Test model with residual connections and LayerNorm."""
    model = ResNet(dim=16, n_blocks=3)
    x = torch.randn(2, 8)
    ref = model(x).clone()

    model2 = ResNet(dim=16, n_blocks=3)
    model2.load_state_dict(model.state_dict())
    compiled = torch.compile(model2)
    out = compiled(x)

    diff = (ref - out).abs().max().item()
    assert diff < 1e-4, f"Forward mismatch: {diff}"

    out.sum().backward()
    assert model2.embed.weight.grad is not None
    assert model2.blocks[0].fc1.weight.grad is not None


def test_transformer():
    """Test mini transformer encoder."""
    torch.manual_seed(42)
    model = TinyTransformer()
    model.eval()  # eval mode to avoid dropout issues
    x = torch.randn(2, 5, 8)  # batch=2, seq=5, feat=8
    ref = model(x).clone()

    model2 = TinyTransformer()
    model2.load_state_dict(model.state_dict())
    model2.eval()
    compiled = torch.compile(model2)
    out = compiled(x)

    diff = (ref - out).abs().max().item()
    assert diff < 1e-4, f"Forward mismatch: {diff}"


def test_training_loop():
    """Test multiple training steps with optimizer."""
    torch.manual_seed(42)
    model = TinyMLP()
    x = torch.randn(4, 8)
    target = torch.randn(4, 4)

    # Reference: train with eager
    model_ref = TinyMLP()
    model_ref.load_state_dict(model.state_dict())
    opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.01)
    for _ in range(5):
        opt_ref.zero_grad()
        loss = F.mse_loss(model_ref(x), target)
        loss.backward()
        opt_ref.step()
    ref_loss = loss.item()
    ref_state = {k: v.clone() for k, v in model_ref.state_dict().items()}

    # Now with auto_install
    model2 = TinyMLP()
    model2.load_state_dict(model.state_dict())
    compiled = torch.compile(model2)
    opt = torch.optim.SGD(model2.parameters(), lr=0.01)
    for _ in range(5):
        opt.zero_grad()
        loss = F.mse_loss(compiled(x), target)
        loss.backward()
        opt.step()
    ai_loss = loss.item()

    loss_diff = abs(ref_loss - ai_loss)
    assert loss_diff < 1e-4, f"Loss diverged: ref={ref_loss:.6f} ai={ai_loss:.6f} diff={loss_diff:.2e}"

    # Check parameter updates match
    max_param_diff = 0
    for key in ref_state:
        diff = (model2.state_dict()[key] - ref_state[key]).abs().max().item()
        max_param_diff = max(max_param_diff, diff)
    assert max_param_diff < 1e-4, f"Param divergence: {max_param_diff:.2e}"


def test_cache_reload():
    """Test that loading from cache produces identical results."""
    model = TinyMLP()
    x = torch.randn(2, 8)

    # First compile: captures and saves
    model1 = TinyMLP()
    model1.load_state_dict(model.state_dict())
    compiled1 = torch.compile(model1)
    out1 = compiled1(x).clone()
    out1.sum().backward()
    grad1 = model1.fc1.weight.grad.clone()

    # Second compile: should load from cache
    ai._installed.clear()
    model2 = TinyMLP()
    model2.load_state_dict(model.state_dict())
    compiled2 = torch.compile(model2)
    out2 = compiled2(x).clone()
    out2.sum().backward()
    grad2 = model2.fc1.weight.grad.clone()

    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-5, f"Output mismatch on cache reload: {diff}"

    grad_diff = (grad1 - grad2).abs().max().item()
    assert grad_diff < 1e-5, f"Gradient mismatch on cache reload: {grad_diff}"


def test_force_recapture():
    """Test that force_recapture=True re-captures even with cache."""
    model = TinyMLP()
    x = torch.randn(2, 8)

    # First compile
    model1 = TinyMLP()
    model1.load_state_dict(model.state_dict())
    compiled1 = torch.compile(model1)
    compiled1(x)

    # Check cache exists
    cache_files = os.listdir(CACHE_DIR)
    assert any(f.endswith('.py') for f in cache_files), "No cache file created"

    # Get mtime of cache file
    py_file = [f for f in cache_files if f.endswith('.py')][0]
    mtime1 = os.path.getmtime(os.path.join(CACHE_DIR, py_file))

    # Force recapture
    import time
    time.sleep(0.1)
    ai.configure(force_recapture=True)
    ai._installed.clear()
    model2 = TinyMLP()
    model2.load_state_dict(model.state_dict())
    compiled2 = torch.compile(model2)
    compiled2(x)

    mtime2 = os.path.getmtime(os.path.join(CACHE_DIR, py_file))
    assert mtime2 > mtime1, "Cache file not updated despite force_recapture"


def test_force_recapture_eval_variant():
    """force_recapture=True should also refresh eval-only cache files."""
    model = TinyMLP().eval()
    x = torch.randn(2, 8)

    compiled1 = torch.compile(model)
    compiled1(x)

    py_file = [
        f for f in os.listdir(CACHE_DIR)
        if f.endswith("_eval_aten.py")
    ][0]
    cache_path = os.path.join(CACHE_DIR, py_file)
    mtime1 = os.path.getmtime(cache_path)

    import time
    time.sleep(0.1)
    ai.configure(force_recapture=True)
    ai._installed.clear()

    model2 = TinyMLP().eval()
    model2.load_state_dict(model.state_dict())
    compiled2 = torch.compile(model2)
    compiled2(x)

    mtime2 = os.path.getmtime(cache_path)
    assert mtime2 > mtime1, "Eval cache file not updated despite force_recapture"


def test_h5_dump_creates_file():
    """Test that dump_h5=True creates an H5 file alongside the .py cache file."""
    ai.configure(dump_h5=True)

    model = SingleLinear()
    x = torch.randn(2, 4)

    compiled = torch.compile(model)
    compiled(x)

    # Check that an H5 file was created
    h5_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.h5')]
    assert len(h5_files) == 1, f"Expected 1 H5 file, found: {h5_files}"

    # Verify it has a /tensors group with entries
    import h5py
    h5_path = os.path.join(CACHE_DIR, h5_files[0])
    with h5py.File(h5_path, "r") as f:
        assert "tensors" in f, f"Missing /tensors group. Keys: {list(f.keys())}"
        tensor_keys = list(f["tensors"].keys())
        assert len(tensor_keys) > 0, "No tensors stored in /tensors/"


def test_user_modified_model_backward_changes_gradients():
    """Editing backward() in the cached model file should affect parameter grads."""
    torch.manual_seed(42)
    x = torch.randn(2, 4)

    ref_model = SingleLinear()
    ref_state = {k: v.clone() for k, v in ref_model.state_dict().items()}
    ref_out = ref_model(x).clone()
    ref_out.sum().backward()
    ref_weight_grad = ref_model.fc.weight.grad.clone()
    ref_bias_grad = ref_model.fc.bias.grad.clone()

    captured_model = SingleLinear()
    captured_model.load_state_dict(ref_state)
    compiled = torch.compile(captured_model)
    compiled(x)

    cache_file = _only_cache_py()
    with open(cache_file) as f:
        code = f.read()
    # Find the backward return line (LAST return statement with 3 values)
    import re as _re
    all_returns = list(_re.finditer(r"^(    return \((\w+), (\w+), (\w+),\)\n)", code, _re.MULTILINE))
    assert len(all_returns) >= 2, f"Expected ≥2 return lines in {cache_file}, found {len(all_returns)}"
    bw_return = all_returns[-1]  # backward is the last one
    old = bw_return.group(0)
    v1, v2, v3 = bw_return.group(2), bw_return.group(3), bw_return.group(4)
    new = f"    return ({v1} * 2.0, {v2} * 2.0, {v3},)\n"
    with open(cache_file, "w") as f:
        f.write(code.replace(old, new, 1))

    ai._installed.clear()

    edited_model = SingleLinear()
    edited_model.load_state_dict(ref_state)
    compiled_edited = torch.compile(edited_model)
    out = compiled_edited(x)
    out.sum().backward()

    assert torch.allclose(out, ref_out, atol=1e-6)
    assert torch.allclose(
        edited_model.fc.weight.grad,
        ref_weight_grad * 2.0,
        atol=1e-6,
    )
    assert torch.allclose(
        edited_model.fc.bias.grad,
        ref_bias_grad * 2.0,
        atol=1e-6,
    )


def test_user_modified_optimizer_function_changes_updates():
    """Editing a cached optimizer-style function should change the in-place update."""

    def make_step_fn():
        @torch.compile(dynamic=False, fullgraph=True)
        def step_fn(param, grad, lr):
            param.add_(grad, alpha=-lr)
            return None

        return step_fn

    grad = torch.tensor([0.1, 0.2])
    lr = 0.5
    initial = torch.tensor([1.0, 2.0])

    baseline_param = initial.clone()
    make_step_fn()(baseline_param, grad, lr)
    assert torch.allclose(baseline_param, torch.tensor([0.95, 1.9]))

    cache_file = _only_cache_py()
    with open(cache_file) as f:
        code = f.read()
    old = 'aten.mul.Tensor(input__2_1, -0.5)'
    new = 'aten.mul.Tensor(input__2_1, -1.0)'
    assert old in code, cache_file
    with open(cache_file, "w") as f:
        f.write(code.replace(old, new, 1))

    ai._installed.clear()

    edited_param = initial.clone()
    make_step_fn()(edited_param, grad, lr)

    assert torch.allclose(edited_param, torch.tensor([0.9, 1.8]), atol=1e-6)

    entry = next(iter(ai._installed.values()))
    assert entry.source == "loaded_from_disk"


def test_nested_compiled():
    """Test nested models where submodules are individually compiled."""
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
        def forward(self, x):
            return F.relu(self.fc(x))

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)
        def forward(self, x):
            return self.fc(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.compile(Encoder())
            self.decoder = torch.compile(Decoder())
        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Model()
    x = torch.randn(2, 8)
    out = model(x)
    assert out.shape == (2, 4)

    out.sum().backward()
    assert model.encoder.fc.weight.grad is not None
    assert model.decoder.fc.weight.grad is not None


def test_compiled_function():
    """Test @torch.compile on a standalone function."""
    @torch.compile
    def my_fn(a, b, lr):
        return a - lr * b

    p = torch.randn(4)
    g = torch.randn(4)
    result = my_fn(p, g, 0.01)
    expected = p - 0.01 * g
    diff = (result - expected).abs().max().item()
    assert diff < 1e-6, f"Function mismatch: {diff}"


def test_compiled_function_decorator_with_args():
    """Test @torch.compile(...) with kwargs."""
    @torch.compile(fullgraph=True)
    def my_fn(a, b):
        return a + b

    result = my_fn(torch.ones(4), torch.ones(4))
    expected = torch.ones(4) * 2
    diff = (result - expected).abs().max().item()
    assert diff < 1e-6, f"Decorator with args mismatch: {diff}"


def test_compiled_function_static_shape_variants():
    """dynamic=False functions should capture separate static shape variants."""
    @torch.compile(dynamic=False, fullgraph=True)
    def branch_fn(x):
        if x.size(-2) > x.size(-1):
            return x.sum(dim=-1)
        return x.sum(dim=-2)

    tall = torch.randn(3, 2)
    tall_out = branch_fn(tall)
    assert torch.equal(tall_out, tall.sum(dim=-1))

    wide = torch.randn(2, 3)
    wide_out = branch_fn(wide)
    assert torch.equal(wide_out, wide.sum(dim=-2))

    cache_files = [
        name for name in os.listdir(CACHE_DIR)
        if name.endswith("_aten.py")
    ]
    assert len(cache_files) == 2, cache_files

    entry = next(iter(ai._installed.values()))
    assert entry.source != "unwrapped"


def test_proxy_attribute_forwarding():
    """Test that _CompiledModelProxy forwards attribute access correctly."""
    model = TinyMLP()
    compiled = torch.compile(model)

    # Should forward to underlying model
    assert list(compiled.parameters()) is not None
    assert compiled.fc1.weight.shape == (16, 8)

    # .train() and .eval() should work
    compiled.train()
    assert model.training
    compiled.eval()
    assert not model.training


def test_status():
    """Test the status() function."""
    model = SingleLinear()
    compiled = torch.compile(model)
    x = torch.randn(2, 4)
    compiled(x)

    s = ai.status()
    assert "aten replacement" in s
    assert "SingleLinear" in s


def test_install_from_file():
    """Test explicit install_from_file API."""
    model = TinyMLP()
    x = torch.randn(2, 8)

    # First, capture to create a cache file
    model1 = TinyMLP()
    model1.load_state_dict(model.state_dict())
    compiled = torch.compile(model1)
    ref_out = compiled(x).clone()

    # Find the cache file
    py_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.py')]
    assert py_files, "No cache file created"
    cache_file = os.path.join(CACHE_DIR, py_files[0])

    # Install from file onto a fresh model
    model2 = TinyMLP()
    model2.load_state_dict(model.state_dict())
    ai.install_from_file(model2, cache_file)
    out = model2(x)

    diff = (ref_out - out).abs().max().item()
    assert diff < 1e-5, f"install_from_file mismatch: {diff}"


def test_inference_only():
    """Test with capture_backward=False (inference only)."""
    model = TinyMLP()
    x = torch.randn(2, 8)
    ref = model(x).clone()

    model2 = TinyMLP()
    model2.load_state_dict(model.state_dict())
    ai.configure(capture_backward=False)
    compiled = torch.compile(model2)
    out = compiled(x)

    diff = (ref - out).abs().max().item()
    assert diff < 1e-5, f"Inference mismatch: {diff}"


def test_larger_batch():
    """Test with larger batch size to catch shape issues."""
    model = TinyMLP()
    x = torch.randn(32, 8)
    ref = model(x).clone()

    model2 = TinyMLP()
    model2.load_state_dict(model.state_dict())
    compiled = torch.compile(model2)
    out = compiled(x)

    diff = (ref - out).abs().max().item()
    assert diff < 1e-5, f"Larger batch mismatch: {diff}"


def _nanochat_setup():
    """Shared nanochat setup: returns (model, vocab_size, config, device) or None if unavailable."""
    REPO_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "repos", "nanochat"
    )
    if not os.path.exists(os.path.join(REPO_DIR, "nanochat", "gpt.py")):
        return None

    # nanochat uses Flash Attention 3 which requires CUDA
    if not torch.cuda.is_available():
        return None

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    from nanochat.gpt import GPT, GPTConfig

    # Use a fixed vocab size (GPT-2 base + special tokens, padded)
    vocab_size = 50304
    device = torch.device("cuda")

    DEPTH = 2
    HEAD_DIM = 64
    base_dim = DEPTH * 32
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM

    config = GPTConfig(
        n_layer=DEPTH, n_head=num_heads, n_kv_head=num_heads,
        n_embd=model_dim, vocab_size=vocab_size, sequence_len=16,
        window_pattern="S" * DEPTH,
    )

    torch.manual_seed(42)
    with torch.device("cpu"):
        model = GPT(config, pad_vocab_size_to=64)
    model.init_weights()
    model.train()
    model = model.to(device)

    return model, vocab_size, config, device


def test_nanochat():
    """Test with nanochat GPT model — the primary real-world use case.

    This mimics what base_train.py does: build model, torch.compile(model),
    then forward+backward. auto_install intercepts torch.compile transparently.
    """
    setup = _nanochat_setup()
    if setup is None:
        pytest.skip("nanochat repo not found or CUDA not available")
    model, vocab_size, config, device = setup

    from nanochat.gpt import GPT

    idx = torch.randint(0, vocab_size, (2, 16), device=device)
    targets = torch.randint(0, vocab_size, (2, 16), device=device)

    # Reference loss (eager)
    ref_loss = model(idx, targets=targets).item()

    # Same flow as base_train.py: just torch.compile(model)
    model2 = GPT(config, pad_vocab_size_to=64)
    model2.load_state_dict(model.state_dict())
    model2.train()
    model2 = model2.to(device)

    # This is ALL the user does — torch.compile is intercepted by auto_install
    compiled = torch.compile(model2, dynamic=False)
    loss = compiled(idx, targets=targets)
    ai_loss = loss.item()

    diff = abs(ref_loss - ai_loss)
    assert diff == 0, f"Nanochat loss not bit-identical: ref={ref_loss:.6f} ai={ai_loss:.6f} diff={diff:.2e}"

    loss.backward()
    has_grad = any(p.grad is not None for p in model2.parameters())
    assert has_grad, "No gradients after backward"


def test_nanochat_training_loop():
    """Test nanochat training loop: compile + MuonAdamW + multiple steps.

    Uses nanochat's own optimizer, exactly as base_train.py would.
    """
    setup = _nanochat_setup()
    if setup is None:
        pytest.skip("nanochat repo not found or CUDA not available")
    model, vocab_size, config, device = setup

    from nanochat.optim import MuonAdamW

    # Build optimizer exactly like nanochat does
    param_groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        param_groups.append({
            "params": [p], "kind": "adamw",
            "lr": 1e-3, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
        })
    optimizer = MuonAdamW(param_groups)

    # torch.compile — intercepted by auto_install
    compiled = torch.compile(model, dynamic=False)

    losses = []
    for step in range(5):
        idx = torch.randint(0, vocab_size, (2, 16), device=device)
        targets = torch.randint(0, vocab_size, (2, 16), device=device)
        optimizer.zero_grad()
        loss = compiled(idx, targets=targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] + 1.0, \
        f"Training unstable: losses went from {losses[0]:.4f} to {losses[-1]:.4f}"


def test_nanochat_optimizer_detected_as_inner_compiled():
    """MuonAdamW is detected as using inner @torch.compile functions.

    Since MuonAdamW uses @torch.compile on adamw_step_fused etc., the
    optimizer should be detected as using inner compiled fns and NOT
    captured monolithically.
    """
    setup = _nanochat_setup()
    if setup is None:
        pytest.skip("nanochat repo not found or CUDA not available")
    model, vocab_size, config, device = setup

    from nanochat.optim import MuonAdamW

    setup_test("nanochat_optimizer_detected_as_inner_compiled")

    param_groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        param_groups.append({
            "params": [p], "kind": "adamw",
            "lr": 1e-3, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
        })
    optimizer = MuonAdamW(param_groups)

    compiled = torch.compile(model, dynamic=False)
    idx = torch.randint(0, vocab_size, (2, 16), device=device)
    targets = torch.randint(0, vocab_size, (2, 16), device=device)

    loss = compiled(idx, targets=targets)
    loss.backward()
    optimizer.step()

    # Verify optimizer was detected as using inner compiled functions
    opt_entry = ai._optimizer_captured.get(id(optimizer))
    assert opt_entry is not None, "Optimizer should have been captured"
    assert opt_entry.get("uses_inner_compiled"), \
        f"MuonAdamW should use inner compiled fns, got: {opt_entry}"

    # Inner compiled functions should be installed
    inner_fn_names = [
        e.name for e in ai._installed.values()
        if e.kind == "function"
    ]
    assert len(inner_fn_names) > 0, \
        f"Expected inner compiled functions to be installed, got none"


def test_nanochat_optimizer_replay_inner_fns():
    """MuonAdamW inner compiled functions are captured individually and replay works.

    Verifies that adamw_step_fused (and muon_step_fused if present) are
    individually captured via _CompiledFnProxy, and that training loss
    over 5 steps matches eager within tolerance.
    """
    setup = _nanochat_setup()
    if setup is None:
        pytest.skip("nanochat repo not found or CUDA not available")
    model, vocab_size, config, device = setup

    from nanochat.gpt import GPT
    from nanochat.optim import MuonAdamW

    # ── Reference: fully eager ──
    torch.manual_seed(42)
    ref_model = GPT(config, pad_vocab_size_to=64)
    ref_model.load_state_dict(model.state_dict())
    ref_model.train()
    ref_model = ref_model.to(device)

    ref_param_groups = []
    for name, p in ref_model.named_parameters():
        if not p.requires_grad:
            continue
        ref_param_groups.append({
            "params": [p], "kind": "adamw",
            "lr": 1e-3, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
        })
    ref_optimizer = MuonAdamW(ref_param_groups)

    ref_losses = []
    for step in range(5):
        torch.manual_seed(100 + step)
        idx = torch.randint(0, vocab_size, (2, 16), device=device)
        targets = torch.randint(0, vocab_size, (2, 16), device=device)
        ref_optimizer.zero_grad()
        loss = ref_model(idx, targets=targets)
        loss.backward()
        ref_optimizer.step()
        ref_losses.append(loss.item())

    # ── Auto-install with inner fn capture ──
    setup_test("nanochat_optimizer_replay_inner_fns")

    torch.manual_seed(42)
    test_model = GPT(config, pad_vocab_size_to=64)
    test_model.load_state_dict(model.state_dict())
    test_model.train()
    test_model = test_model.to(device)

    test_param_groups = []
    for name, p in test_model.named_parameters():
        if not p.requires_grad:
            continue
        test_param_groups.append({
            "params": [p], "kind": "adamw",
            "lr": 1e-3, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
        })
    test_optimizer = MuonAdamW(test_param_groups)

    compiled = torch.compile(test_model, dynamic=False)

    test_losses = []
    for step in range(5):
        torch.manual_seed(100 + step)
        idx = torch.randint(0, vocab_size, (2, 16), device=device)
        targets = torch.randint(0, vocab_size, (2, 16), device=device)
        test_optimizer.zero_grad()
        loss = compiled(idx, targets=targets)
        loss.backward()
        test_optimizer.step()
        test_losses.append(loss.item())

    # Verify optimizer was detected as using inner compiled functions
    opt_entry = ai._optimizer_captured.get(id(test_optimizer))
    assert opt_entry is not None, "Optimizer should have been captured"
    assert opt_entry.get("uses_inner_compiled"), \
        f"MuonAdamW should be detected as using inner compiled fns, got: {opt_entry}"

    # Verify inner compiled functions were installed
    inner_fn_names = [
        e.name for e in ai._installed.values()
        if e.kind == "function"
    ]
    assert any("adam" in n.lower() or "fused" in n.lower() for n in inner_fn_names), \
        f"Expected inner compiled function names to contain 'adam' or 'fused', got: {inner_fn_names}"

    # Losses should match eager within tolerance
    for step, (r, a) in enumerate(zip(ref_losses, test_losses)):
        assert abs(r - a) < 1e-4, \
            f"Step {step}: ref={r:.6f} auto={a:.6f} diff={abs(r-a):.2e}"


def test_inner_fn_full_replay():
    """Full inner fn replay: MuonAdamW.step() is NEVER called after step 1.

    With replay_optimizer=True, the inner fn replay plan completely replaces
    the optimizer's outer loop.  Step counters, scalar tensor fills, and
    aten graph calls are all managed by _run_inner_replay.
    """
    setup = _nanochat_setup()
    if setup is None:
        pytest.skip("nanochat repo not found or CUDA not available")
    model, vocab_size, config, device = setup

    from nanochat.gpt import GPT
    from nanochat.optim import MuonAdamW

    # ── Reference: fully eager ──
    torch.manual_seed(42)
    ref_model = GPT(config, pad_vocab_size_to=64)
    ref_model.load_state_dict(model.state_dict())
    ref_model.train()
    ref_model = ref_model.to(device)

    ref_param_groups = []
    for name, p in ref_model.named_parameters():
        if not p.requires_grad:
            continue
        ref_param_groups.append({
            "params": [p], "kind": "adamw",
            "lr": 1e-3, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
        })
    ref_optimizer = MuonAdamW(ref_param_groups)

    ref_losses = []
    for step in range(5):
        torch.manual_seed(100 + step)
        idx = torch.randint(0, vocab_size, (2, 16), device=device)
        targets = torch.randint(0, vocab_size, (2, 16), device=device)
        ref_optimizer.zero_grad()
        loss = ref_model(idx, targets=targets)
        loss.backward()
        ref_optimizer.step()
        ref_losses.append(loss.item())

    # ── Auto-install with inner fn FULL replay ──
    setup_test("inner_fn_full_replay")
    ai.configure(replay_optimizer=True)  # Enable full replay

    torch.manual_seed(42)
    test_model = GPT(config, pad_vocab_size_to=64)
    test_model.load_state_dict(model.state_dict())
    test_model.train()
    test_model = test_model.to(device)

    test_param_groups = []
    for name, p in test_model.named_parameters():
        if not p.requires_grad:
            continue
        test_param_groups.append({
            "params": [p], "kind": "adamw",
            "lr": 1e-3, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": 0.0,
        })
    test_optimizer = MuonAdamW(test_param_groups)

    compiled = torch.compile(test_model, dynamic=False)

    test_losses = []
    for step in range(5):
        torch.manual_seed(100 + step)
        idx = torch.randint(0, vocab_size, (2, 16), device=device)
        targets = torch.randint(0, vocab_size, (2, 16), device=device)
        test_optimizer.zero_grad()
        loss = compiled(idx, targets=targets)
        loss.backward()
        test_optimizer.step()
        test_losses.append(loss.item())

    # Verify inner replay plan was built
    opt_entry = ai._optimizer_captured.get(id(test_optimizer))
    assert opt_entry is not None, "Optimizer should have been captured"
    assert opt_entry.get("uses_inner_compiled"), \
        "Should be detected as inner compiled"
    inner_replay = opt_entry.get("inner_replay")
    assert inner_replay is not None, \
        "Inner replay plan should have been built (replay_optimizer=True)"
    assert len(inner_replay.calls) > 0, \
        f"Replay plan should have calls, got {len(inner_replay.calls)}"
    assert len(inner_replay.step_attr_names) > 0, \
        f"Replay plan should detect step attrs, got {inner_replay.step_attr_names}"

    # Verify step counters were managed correctly (step 1 by outer loop,
    # steps 2-5 by _run_inner_replay)
    for group in test_optimizer.param_groups:
        for param in group["params"]:
            state = test_optimizer.state.get(param, {})
            if "step" in state:
                assert state["step"] == 5, \
                    f"Expected step=5 after 5 training steps, got {state['step']}"

    # Losses should match eager within tolerance
    for step, (r, a) in enumerate(zip(ref_losses, test_losses)):
        assert abs(r - a) < 1e-4, \
            f"Step {step}: ref={r:.6f} auto={a:.6f} diff={abs(r-a):.2e}"


# ═══════════════════════════════════════════════════════════════════════
# Auto optimizer capture
# ═══════════════════════════════════════════════════════════════════════


def test_auto_optimizer_capture():
    """optimizer.step() is auto-captured without torch.compile wrapping."""
    torch.manual_seed(42)
    model = TinyMLP()
    x = torch.randn(4, 8)
    target = torch.randn(4, 4)

    compiled = torch.compile(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # First forward triggers model capture
    loss = F.mse_loss(compiled(x), target)
    loss.backward()

    # optimizer.step() should trigger auto-capture
    optimizer.step()

    # Verify optimizer was captured
    assert id(optimizer) in ai._optimizer_captured, "Optimizer was not auto-captured"
    entry = ai._optimizer_captured[id(optimizer)]
    assert entry["source"] == "captured"
    assert entry["cache_path"] is not None

    # Verify cache file exists on disk
    import os
    assert os.path.exists(entry["cache_path"]), \
        f"Optimizer cache file not found: {entry['cache_path']}"

    # Subsequent step should run eagerly (not re-capture)
    optimizer.zero_grad()
    loss = F.mse_loss(compiled(x), target)
    loss.backward()
    optimizer.step()  # should not re-capture


def test_auto_optimizer_capture_disabled():
    """capture_optimizer=False skips optimizer capture."""
    ai.configure(capture_optimizer=False)

    torch.manual_seed(42)
    model = TinyMLP()
    x = torch.randn(4, 8)
    target = torch.randn(4, 4)

    compiled = torch.compile(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss = F.mse_loss(compiled(x), target)
    loss.backward()
    optimizer.step()

    assert id(optimizer) not in ai._optimizer_captured, \
        "Optimizer was captured despite capture_optimizer=False"


def test_auto_optimizer_training_matches_eager():
    """Training with auto-captured optimizer produces same results as eager."""
    torch.manual_seed(42)
    model = TinyMLP()
    x = torch.randn(4, 8)
    target = torch.randn(4, 4)

    # Reference: fully eager
    ref_model = TinyMLP()
    ref_model.load_state_dict(model.state_dict())
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01)
    for _ in range(5):
        ref_opt.zero_grad()
        loss = F.mse_loss(ref_model(x), target)
        loss.backward()
        ref_opt.step()
    ref_state = {k: v.clone() for k, v in ref_model.state_dict().items()}

    # Auto-install: model compiled, optimizer auto-captured
    model2 = TinyMLP()
    model2.load_state_dict(model.state_dict())
    compiled = torch.compile(model2)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    for _ in range(5):
        opt2.zero_grad()
        loss = F.mse_loss(compiled(x), target)
        loss.backward()
        opt2.step()

    # Parameters should match
    max_diff = 0
    for key in ref_state:
        diff = (model2.state_dict()[key] - ref_state[key]).abs().max().item()
        max_diff = max(max_diff, diff)
    assert max_diff < 1e-4, f"Param divergence: {max_diff:.2e}"


def test_auto_optimizer_capture_uses_distinct_cache_keys_for_distinct_layouts():
    """Same-class optimizers with different parameter layouts should not share cache files."""
    class WiderMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    x = torch.randn(4, 8)
    target = torch.randn(4, 4)

    model1 = TinyMLP()
    compiled1 = torch.compile(model1)
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    loss1 = F.mse_loss(compiled1(x), target)
    loss1.backward()
    opt1.step()

    model2 = WiderMLP()
    compiled2 = torch.compile(model2)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    loss2 = F.mse_loss(compiled2(x), target)
    loss2.backward()
    opt2.step()

    entry1 = ai._optimizer_captured[id(opt1)]
    entry2 = ai._optimizer_captured[id(opt2)]
    assert entry1["source"] == "captured"
    assert entry2["source"] == "captured"
    assert entry1["cache_path"] != entry2["cache_path"]


def test_registered_optimizer_preserves_first_step_return_value():
    """register_optimizer should return the original step value on first capture."""
    class DummyOptimizer:
        def __init__(self):
            self.calls = []

        def step(self, do_adam=False):
            self.calls.append(do_adam)
            return "VALUE"

    opt = DummyOptimizer()
    ai.register_optimizer(opt, step_fn=lambda _orig=opt.step: _orig(do_adam=True))

    result = opt.step()

    assert result == "VALUE"
    assert opt.calls == [True]
    assert ai._optimizer_captured[id(opt)]["source"] == "failed"


def test_registered_optimizer_preserves_custom_step_after_capture():
    """register_optimizer should keep using the custom step_fn after first capture."""
    class DummyOptimizer:
        def __init__(self):
            self.calls = []

        def step(self, do_adam=False):
            self.calls.append(do_adam)
            return "VALUE"

    opt = DummyOptimizer()
    ai.register_optimizer(opt, step_fn=lambda _orig=opt.step: _orig(do_adam=True))

    first = opt.step()
    second = opt.step()

    assert first == "VALUE"
    assert second == "VALUE"
    assert opt.calls == [True, True]
    assert ai._optimizer_captured[id(opt)]["source"] == "failed"


def test_registered_optimizer_forwards_step_args_on_first_capture():
    """register_optimizer should preserve call-time step args on first capture."""
    class DummyOptimizer:
        def __init__(self):
            self.calls = []

        def step(self, flag):
            self.calls.append(flag)
            return flag

    opt = DummyOptimizer()
    ai.register_optimizer(opt)

    result = opt.step("VALUE")

    assert result == "VALUE"
    assert opt.calls == ["VALUE"]
    assert ai._optimizer_captured[id(opt)]["source"] == "failed"


# ═══════════════════════════════════════════════════════════════════════
# Dynamic batch sizes
# ═══════════════════════════════════════════════════════════════════════


class ViewBatchModel(nn.Module):
    """Model that uses explicit view(batch_size, ...) — needs dynamic=True."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(4, 2)
    def forward(self, x):
        h = self.fc1(x)
        B = x.shape[0]
        h = h.view(B, 4, 4)
        h = h.sum(dim=-1)
        return self.fc2(h)


@pytest.fixture()
def _dynamic_setup(request, tmp_path):
    """Override the default setup with dynamic=True."""
    setup_test(
        request.node.name,
        cache_dir=tmp_path / ".torch_graph_cache",
        dynamic=True,
    )


def test_dynamic_batch_forward(_dynamic_setup):
    """Forward works with varying batch sizes after dynamic capture."""
    torch.manual_seed(42)
    model = ViewBatchModel()
    ref_state = {k: v.clone() for k, v in model.state_dict().items()}

    compiled = torch.compile(model)
    # Capture with batch=2
    out = compiled(torch.randn(2, 8))
    assert out.shape == (2, 2)

    # Test with different batch sizes
    for bs in [1, 3, 4, 8]:
        torch.manual_seed(100 + bs)
        x = torch.randn(bs, 8)
        ref_model = ViewBatchModel()
        ref_model.load_state_dict(ref_state)
        ref_out = ref_model(x)
        inst_out = compiled(x)
        diff = (ref_out - inst_out).abs().max().item()
        assert diff == 0.0, f"batch={bs}: fwd diff={diff}"


def test_dynamic_batch_backward(_dynamic_setup):
    """Forward+backward work with varying batch sizes after dynamic capture."""
    torch.manual_seed(42)
    model = ViewBatchModel()
    ref_state = {k: v.clone() for k, v in model.state_dict().items()}

    compiled = torch.compile(model)
    # Capture with batch=2
    out = compiled(torch.randn(2, 8))
    out.sum().backward()

    # Test backward with different batch sizes
    for bs in [1, 3, 4, 8]:
        torch.manual_seed(200 + bs)
        x = torch.randn(bs, 8)

        ref_model = ViewBatchModel()
        ref_model.load_state_dict(ref_state)
        ref_out = ref_model(x)
        ref_out.sum().backward()

        model.zero_grad()
        inst_out = compiled(x)
        inst_out.sum().backward()

        fwd_diff = (ref_out - inst_out).abs().max().item()
        assert fwd_diff == 0.0, f"batch={bs}: fwd diff={fwd_diff}"

        for name, rp in ref_model.named_parameters():
            ip = dict(model.named_parameters())[name]
            grad_diff = (rp.grad - ip.grad).abs().max().item()
            assert grad_diff == 0.0, f"batch={bs} {name}: grad diff={grad_diff}"


def test_dynamic_cache_reload(_dynamic_setup):
    """Cache-loaded dynamic graph works with varying batch sizes."""
    torch.manual_seed(42)
    model = ViewBatchModel()
    ref_state = {k: v.clone() for k, v in model.state_dict().items()}

    compiled = torch.compile(model)
    out = compiled(torch.randn(2, 8))
    out.sum().backward()

    # Create a fresh model, should load from cache
    model2 = ViewBatchModel()
    model2.load_state_dict(ref_state)
    ai._installed.clear()

    compiled2 = torch.compile(model2)
    for bs in [1, 4, 8]:
        torch.manual_seed(300 + bs)
        x = torch.randn(bs, 8)

        ref_model = ViewBatchModel()
        ref_model.load_state_dict(ref_state)
        ref_out = ref_model(x)
        ref_out.sum().backward()

        model2.zero_grad()
        inst_out = compiled2(x)
        inst_out.sum().backward()

        fwd_diff = (ref_out - inst_out).abs().max().item()
        assert fwd_diff == 0.0, f"batch={bs}: fwd diff={fwd_diff}"

        for name, rp in ref_model.named_parameters():
            ip = dict(model2.named_parameters())[name]
            grad_diff = (rp.grad - ip.grad).abs().max().item()
            assert grad_diff == 0.0, f"batch={bs} {name}: grad diff={grad_diff}"


def test_dynamic_training_loop(_dynamic_setup):
    """Training loop works with varying batch sizes."""
    torch.manual_seed(42)
    model = ViewBatchModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    compiled = torch.compile(model)

    losses = []
    for step in range(5):
        bs = [2, 4, 1, 8, 3][step]
        x = torch.randn(bs, 8)
        target = torch.randn(bs, 2)
        optimizer.zero_grad()
        out = compiled(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(l < 100 for l in losses), f"Training produced unreasonable losses: {losses}"


# ═══════════════════════════════════════════════════════════════════════
# Dynamic shape edge cases
# ═══════════════════════════════════════════════════════════════════════


class MultiDimModel(nn.Module):
    """Model with varying batch AND feature dimensions — uses element-wise ops only."""
    def __init__(self, d=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d))
        self.bias = nn.Parameter(torch.randn(d))
    def forward(self, x):
        # x: (B, T, D) — element-wise ops avoid view/reshape of symbolic dims
        h = x * self.weight + self.bias
        return h.sum(dim=1)  # (B, D)


class SeqModel(nn.Module):
    """Model with varying sequence length — softmax over T dimension."""
    def __init__(self, d=8):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(d) * 0.1)
        self.bias = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        # x: (B, T, D) — softmax over seq dim, then weighted sum
        h = x * self.scale + self.bias
        w = torch.softmax(h.sum(dim=-1, keepdim=True), dim=1)  # (B, T, 1)
        return (h * w).sum(dim=1)  # (B, D)


def test_dynamic_multiple_symbolic_dims(_dynamic_setup):
    """Model with varying batch AND seq_len (B, T, D) — element-wise ops."""
    torch.manual_seed(42)
    model = MultiDimModel(d=8)
    ref_state = {k: v.clone() for k, v in model.state_dict().items()}

    compiled = torch.compile(model)
    # Capture with B=2, T=4
    out = compiled(torch.randn(2, 4, 8))
    assert out.shape == (2, 8)

    # Test with different (B, T) combinations
    for b, t in [(1, 3), (3, 6), (4, 2), (2, 8)]:
        x = torch.randn(b, t, 8)
        ref_model = MultiDimModel(d=8)
        ref_model.load_state_dict(ref_state)
        ref_out = ref_model(x)
        inst_out = compiled(x)
        diff = (ref_out - inst_out).abs().max().item()
        assert diff == 0.0, f"(B={b}, T={t}): fwd diff={diff}"


def test_dynamic_seq_len_forward(_dynamic_setup):
    """Softmax over varying sequence length — forward correctness."""
    torch.manual_seed(42)
    model = SeqModel(d=8)
    ref_state = {k: v.clone() for k, v in model.state_dict().items()}

    compiled = torch.compile(model)
    # Capture with T=4
    out = compiled(torch.randn(2, 4, 8))
    assert out.shape == (2, 8)

    # Test with different sequence lengths
    for t in [3, 6, 8]:
        torch.manual_seed(100 + t)
        x = torch.randn(2, t, 8)

        ref_model = SeqModel(d=8)
        ref_model.load_state_dict(ref_state)
        ref_out = ref_model(x)
        inst_out = compiled(x)

        diff = (ref_out - inst_out).abs().max().item()
        assert diff == 0.0, f"T={t}: fwd diff={diff}"


def test_dynamic_optimizer_replay(_dynamic_setup):
    """Training loop with varying batch sizes + replay_optimizer=True."""
    setup_test("dynamic_optimizer_replay",
               dynamic=True, replay_optimizer=True)

    torch.manual_seed(42)
    model = ViewBatchModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    compiled = torch.compile(model)

    losses = []
    for step in range(5):
        bs = [2, 4, 1, 8, 3][step]
        x = torch.randn(bs, 8)
        target = torch.randn(bs, 2)
        optimizer.zero_grad()
        out = compiled(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(l < 100 for l in losses), f"Training produced unreasonable losses: {losses}"
    assert len(losses) == 5


# ═══════════════════════════════════════════════════════════════════════
# Optimizer replay
# ═══════════════════════════════════════════════════════════════════════


def test_optimizer_replay_sgd():
    """Optimizer replay with SGD produces bit-identical results to eager."""
    setup_test("optimizer_replay_sgd", replay_optimizer=True)

    torch.manual_seed(42)
    model = TinyMLP()
    x = torch.randn(4, 8)
    target = torch.randn(4, 4)

    # Reference: fully eager
    ref_model = TinyMLP()
    ref_model.load_state_dict(model.state_dict())
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01)
    ref_losses = []
    for _ in range(5):
        ref_opt.zero_grad()
        loss = F.mse_loss(ref_model(x), target)
        loss.backward()
        ref_opt.step()
        ref_losses.append(loss.item())
    ref_state = {k: v.clone() for k, v in ref_model.state_dict().items()}

    # Auto-install with optimizer replay
    model2 = TinyMLP()
    model2.load_state_dict(model.state_dict())
    compiled = torch.compile(model2)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    replay_losses = []
    for _ in range(5):
        opt2.zero_grad()
        loss = F.mse_loss(compiled(x), target)
        loss.backward()
        opt2.step()
        replay_losses.append(loss.item())

    # Verify optimizer was captured with replay
    entry = ai._optimizer_captured[id(opt2)]
    assert entry["replay"] is not None, "Replay info should be set"

    # Losses should match
    for step, (r, a) in enumerate(zip(ref_losses, replay_losses)):
        assert abs(r - a) < 1e-5, f"Step {step}: ref={r:.6f} replay={a:.6f}"

    # Parameters should match
    max_diff = 0
    for key in ref_state:
        diff = (model2.state_dict()[key] - ref_state[key]).abs().max().item()
        max_diff = max(max_diff, diff)
    assert max_diff < 1e-4, f"Param divergence: {max_diff:.2e}"


def test_optimizer_replay_adamw():
    """Optimizer replay with AdamW produces near-identical results to eager."""
    setup_test("optimizer_replay_adamw", replay_optimizer=True)

    torch.manual_seed(42)
    model = TinyMLP()
    x = torch.randn(4, 8)
    target = torch.randn(4, 4)

    # Reference: fully eager
    ref_model = TinyMLP()
    ref_model.load_state_dict(model.state_dict())
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-3)
    ref_losses = []
    for _ in range(5):
        ref_opt.zero_grad()
        loss = F.mse_loss(ref_model(x), target)
        loss.backward()
        ref_opt.step()
        ref_losses.append(loss.item())
    ref_state = {k: v.clone() for k, v in ref_model.state_dict().items()}

    # Auto-install with optimizer replay
    model2 = TinyMLP()
    model2.load_state_dict(model.state_dict())
    compiled = torch.compile(model2)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    replay_losses = []
    for _ in range(5):
        opt2.zero_grad()
        loss = F.mse_loss(compiled(x), target)
        loss.backward()
        opt2.step()
        replay_losses.append(loss.item())

    # Verify optimizer was captured with replay
    entry = ai._optimizer_captured[id(opt2)]
    assert entry["replay"] is not None, "Replay info should be set"

    # Losses should match
    for step, (r, a) in enumerate(zip(ref_losses, replay_losses)):
        assert abs(r - a) < 5e-5, f"Step {step}: ref={r:.6f} replay={a:.6f}"

    # Parameters should match
    max_diff = 0
    for key in ref_state:
        diff = (model2.state_dict()[key] - ref_state[key]).abs().max().item()
        max_diff = max(max_diff, diff)
    assert max_diff < 1e-4, f"Param divergence: {max_diff:.2e}"


def test_optimizer_replay_slot_info_stored_in_meta():
    """Replay metadata (slot_info, mutated_slots) is stored in .meta file."""
    setup_test("optimizer_replay_meta", replay_optimizer=True)

    torch.manual_seed(42)
    model = TinyMLP()
    x = torch.randn(4, 8)
    target = torch.randn(4, 4)

    compiled = torch.compile(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss = F.mse_loss(compiled(x), target)
    loss.backward()
    optimizer.step()

    # Read the .meta file
    entry = ai._optimizer_captured[id(optimizer)]
    from pathlib import Path
    import json
    meta_path = Path(entry["cache_path"]).with_suffix(".meta")
    assert meta_path.exists(), f"Meta file not found: {meta_path}"
    with open(meta_path) as f:
        meta = json.load(f)

    assert "slot_info" in meta, "slot_info missing from meta"
    assert "mutated_slot_indices" in meta, "mutated_slot_indices missing from meta"
    assert len(meta["slot_info"]) > 0, "slot_info is empty"
    assert len(meta["mutated_slot_indices"]) > 0, "mutated_slot_indices is empty"

    # Verify slot roles are present
    roles = {info["role"] for info in meta["slot_info"]}
    assert "param" in roles, "No param slots found"
    assert "grad" in roles, "No grad slots found"


def test_corrupt_cache_file_triggers_recapture():
    """Corrupt aten .py file auto-recovers by re-capturing."""
    setup_test("corrupt_cache_recovery")

    torch.manual_seed(42)
    model = TinyMLP()
    compiled = torch.compile(model)
    x = torch.randn(4, 8)

    # First call: captures to disk
    out1 = compiled(x)
    assert out1.shape == (4, 4)

    # Corrupt the cached file
    import glob
    files = glob.glob(os.path.join(CACHE_DIR, "*train_aten.py"))
    assert files, "No cached aten file found"
    with open(files[0], "w") as f:
        f.write("def forward(: BROKEN SYNTAX\n")

    # Second call: should recover via re-capture
    torch._dynamo.reset()
    ai.unpatch()
    ai.patch()
    setup_test("corrupt_cache_recovery")

    model2 = TinyMLP()
    compiled2 = torch.compile(model2)
    out2 = compiled2(x)
    assert out2.shape == (4, 4), "Recovery from corrupt cache failed"


def test_edited_forward_return_count_raises_error():
    """Editing forward to return extra values raises RuntimeError."""
    setup_test("return_count_validation")

    torch.manual_seed(42)
    model = TinyMLP()
    compiled = torch.compile(model)
    x = torch.randn(4, 8)

    loss = compiled(x).sum()
    loss.backward()

    # Modify the cached file: add extra return value
    import glob
    files = glob.glob(os.path.join(CACHE_DIR, "*train_aten.py"))
    assert files, "No cached aten file found"
    with open(files[0]) as f:
        code = f.read()
    code = code.replace("return (", "return (torch.zeros(1), ")
    with open(files[0], "w") as f:
        f.write(code)

    # New proxy loads the user-modified file on first call
    ai._installed.clear()
    model2 = TinyMLP()
    compiled2 = torch.compile(model2)
    with pytest.raises(RuntimeError, match="Did you edit the forward return statement"):
        compiled2(x).sum()


# ═══════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Simple models first
    run_test("single_linear", test_single_linear)
    run_test("tiny_mlp", test_tiny_mlp)
    run_test("batchnorm", test_batchnorm)
    run_test("conv_model", test_conv_model)
    run_test("model_with_kwargs", test_model_with_kwargs)
    run_test("multi_output", test_multi_output, num_real_outputs=2)
    run_test("dropout", test_dropout)
    run_test("residual_network", test_residual_network)
    run_test("transformer", test_transformer)

    # Functional tests
    run_test("training_loop", test_training_loop)
    run_test("cache_reload", test_cache_reload)
    run_test("force_recapture", test_force_recapture)
    run_test("nested_compiled", test_nested_compiled)
    run_test("compiled_function", test_compiled_function)
    run_test("compiled_function_decorator_args", test_compiled_function_decorator_with_args)
    run_test("proxy_attribute_forwarding", test_proxy_attribute_forwarding)
    run_test("status", test_status)
    run_test("install_from_file", test_install_from_file)
    run_test("inference_only", test_inference_only, capture_backward=False)
    run_test("larger_batch", test_larger_batch)

    # Dynamic batch sizes
    run_test("dynamic_batch_forward", test_dynamic_batch_forward, dynamic=True)
    run_test("dynamic_batch_backward", test_dynamic_batch_backward, dynamic=True)
    run_test("dynamic_cache_reload", test_dynamic_cache_reload, dynamic=True)
    run_test("dynamic_training_loop", test_dynamic_training_loop, dynamic=True)

    # Real-world models
    run_test("nanochat", test_nanochat)
    run_test("nanochat_training_loop", test_nanochat_training_loop)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, p, _ in RESULTS if p)
    failed = sum(1 for _, p, _ in RESULTS if not p)
    print(f"\n{passed}/{passed+failed} tests passed")
    if failed:
        print(f"\nFailed tests:")
        for name, p, err in RESULTS:
            if not p:
                print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")

    # Cleanup
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
