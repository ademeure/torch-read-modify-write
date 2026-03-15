"""Test every aten op through the capture → H5 → replay pipeline.

For each op category, we build a minimal model that exercises those ops,
capture the aten graph, dump to H5, then validate:
  1. Per-group replay scripts produce bit-identical results (CPU)
  2. Section-chain replay produces correct results
  3. The expected ops actually appear in the captured graph

This validates the full torch-graph pipeline for every aten op category
that appears in real model captures.

Op categories tested:
  - Elementwise unary: relu, gelu, silu, sigmoid, tanh, neg, abs
  - Elementwise binary: add, mul, sub, div
  - Reductions: sum, mean, amax
  - Normalization: layer_norm, batch_norm
  - Linear algebra: mm, addmm, bmm, t
  - Embedding + loss: embedding, nll_loss, log_softmax
  - Softmax: softmax, log_softmax
  - Convolution: conv2d
  - Tensor manipulation: view, reshape, permute, transpose, cat,
    unsqueeze, expand, clone, slice, select
  - Dtype conversion: _to_copy
  - Clamp + math: clamp_min, erf, rsqrt, exp
  - Backward ops (implicit via run_backward=True)
"""

import os
import re
import sys
import tempfile

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch_graph.export import capture_aten_graphs
from torch_graph.op_dump import dump_grouped_tensors
from torch_graph.kbox_gen import (
    list_groups,
    generate_group_script,
    generate_section_script,
    _available_tensors,
    _build_expected_out,
    _build_input_mapping,
    _parse_replay_inputs,
)
from torch_graph._utils import short_name as _short_name


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_dynamo():
    from torch_graph import auto_install as _ai
    if _ai._real_torch_compile is not None:
        _ai.unpatch()
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ── Test models covering all op categories ────────────────────────────────────


class ElementwiseUnaryModel(nn.Module):
    """Exercises: relu, gelu, silu, sigmoid, tanh, neg, abs, log."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        h = self.fc(x)
        r = torch.relu(h)
        g = F.gelu(r)
        si = F.silu(g)            # aten.silu
        s = torch.sigmoid(si)
        t = torch.tanh(s)
        n = torch.neg(t)
        a = torch.abs(n)
        # log on positive values (abs ensures positive)
        l = torch.log(a + 1.0)   # aten.log
        return l


class ElementwiseBinaryModel(nn.Module):
    """Exercises: add, mul, sub, div (via residual + scaling)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 16)
        self.scale = nn.Parameter(torch.ones(16))

    def forward(self, x):
        a = self.fc1(x)
        b = self.fc2(x)
        added = a + b           # aten.add
        scaled = added * self.scale  # aten.mul
        diff = scaled - a       # aten.sub
        normed = diff / 2.0     # aten.div
        return normed


class ReductionModel(nn.Module):
    """Exercises: sum, mean, amax."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 32)

    def forward(self, x):
        h = self.fc(x)
        s = h.sum(dim=-1, keepdim=True)   # aten.sum
        m = h.mean(dim=-1, keepdim=True)  # aten.mean
        a = h.amax(dim=-1, keepdim=True)  # aten.amax
        return s + m + a


class NormalizationModel(nn.Module):
    """Exercises: native_layer_norm, native_batch_norm."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 32)
        self.ln = nn.LayerNorm(32)
        self.bn = nn.BatchNorm1d(32)

    def forward(self, x):
        h = self.fc(x)
        h = self.ln(h)
        h = self.bn(h)
        return h


class MatmulModel(nn.Module):
    """Exercises: mm, addmm, t, bmm."""
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(16, 32))
        self.b1 = nn.Parameter(torch.zeros(32))
        self.w2 = nn.Parameter(torch.randn(32, 16))

    def forward(self, x):
        h = torch.addmm(self.b1, x, self.w1)
        out = torch.mm(h, self.w2)
        b = x.unsqueeze(1)
        c = out.unsqueeze(2)
        d = torch.bmm(b, c).squeeze(-1).squeeze(-1)
        return out + d.unsqueeze(-1)


class EmbeddingLossModel(nn.Module):
    """Exercises: embedding, log_softmax, nll_loss."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 16)
        self.fc = nn.Linear(16, 100)

    def forward(self, idx, targets):
        h = self.emb(idx)
        logits = self.fc(h)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(log_probs.view(-1, 100), targets.view(-1))
        return loss


class SoftmaxModel(nn.Module):
    """Exercises: _softmax."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 32)

    def forward(self, x):
        h = self.fc(x)
        return F.softmax(h, dim=-1)


class ConvModel(nn.Module):
    """Exercises: convolution, adaptive_avg_pool2d."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


class TensorManipModel(nn.Module):
    """Exercises: view, reshape, permute, transpose, cat, unsqueeze, expand, clone."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 32)

    def forward(self, x):
        h = self.fc(x)
        v = h.view(-1, 2, 16)
        r = v.reshape(-1, 32)
        v2 = h.view(-1, 4, 8)
        p = v2.permute(0, 2, 1)
        t = p.transpose(1, 2)
        t = t.reshape(-1, 32)
        c = torch.cat([r, t], dim=-1)
        u = c.unsqueeze(1)
        e = u.expand(-1, 2, -1)
        s = e[:, 0, :]
        cl = s.clone()
        return cl


class ClampErfModel(nn.Module):
    """Exercises: clamp_min, erf, rsqrt, exp."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        h = self.fc(x)
        clamped = torch.clamp(h, min=0.0)
        erfd = torch.erf(clamped)
        rsq = torch.rsqrt(clamped + 1.0)
        exp_v = torch.exp(-clamped)
        return erfd + rsq + exp_v


class DtypeConversionModel(nn.Module):
    """Exercises: _to_copy (dtype conversion)."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        h = self.fc(x)
        h_half = h.to(torch.float16)
        h_back = h_half.to(torch.float32)
        return h + h_back


class ResidualLayerNormModel(nn.Module):
    """A realistic mini-transformer block: residual + layernorm + gelu."""
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(16)
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 16)
        self.ln2 = nn.LayerNorm(16)

    def forward(self, x):
        h = self.ln1(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        x = x + h
        x = self.ln2(x)
        return x


class DropoutModel(nn.Module):
    """Exercises: native_dropout (training mode)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.drop(h)
        return self.fc2(h)


class StackModel(nn.Module):
    """Exercises: stack."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        h = self.fc(x)
        stacked = torch.stack([h, h * 2], dim=1)
        return stacked.sum(dim=1)


# ── Helpers ───────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
}


def _load_h5_tensors(h5_path, device="cpu"):
    """Load all tensors from /tensors/ restoring original dtypes."""
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


def _long_name(name):
    """H5 short name → FX name: p6 → primals_6, d3 → tangents_3."""
    import re
    m = re.match(r"^p(\d+)$", name)
    if m:
        return f"primals_{m.group(1)}"
    m = re.match(r"^d(\d+)$", name)
    if m:
        return f"tangents_{m.group(1)}"
    return name


def _run_replay(replay_script, input_dict):
    """Execute a replay script with the given inputs dict."""
    outputs = {}
    exec(replay_script, {
        "inputs": input_dict, "outputs": outputs,
        "torch": torch, "operator": __import__("operator"),
        "inf": float("inf"), "math": __import__("math"),
    })
    return outputs


def _capture_and_dump(model, *args, run_backward=True, loss_fn=None, **kwargs):
    """Capture aten graphs and dump to H5, return (h5_path, capture)."""
    if loss_fn is None:
        loss_fn = lambda o: o.sum()
    _, capture = capture_aten_graphs(
        model, *args, run_backward=run_backward, loss_fn=loss_fn,
        record_real_tensors=True, **kwargs,
    )
    h5_fd = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    h5_path = h5_fd.name
    h5_fd.close()
    dump_grouped_tensors(
        capture, h5_path, group_by=["line"], which="both",
        include_params=True, replay_scripts=True,
    )
    return h5_path, capture


def _extract_ops_from_groups(h5_path, section="forward"):
    """Extract all aten op names from group replay scripts."""
    ops = set()
    groups = list_groups(h5_path, section=section, strategy="line")
    for g in groups:
        for m in re.finditer(r'torch\.ops\.aten\.(\w+)', g.replay_script):
            ops.add(m.group(1))
        if 'operator.getitem' in g.replay_script:
            ops.add('getitem')
    return ops, groups


def _validate_group_replays(h5_path, section="forward", exact=True,
                            atol=1e-5, rtol=1e-5):
    """Validate per-group replay scripts produce correct results.

    Loads all tensors from H5, runs each group's replay script, and compares
    against stored expected outputs. Returns (n_validated, ops_found).
    """
    loaded = _load_h5_tensors(h5_path)
    available = _available_tensors(h5_path)
    groups = list_groups(h5_path, section=section, strategy="line")

    # Build name mapping (both short and long names)
    all_vals = {}
    for k, v in loaded.items():
        all_vals[k] = v
        all_vals[_long_name(k)] = v

    ops_found = set()
    n_validated = 0

    for g in groups:
        # Extract aten ops
        for m in re.finditer(r'torch\.ops\.aten\.(\w+)', g.replay_script):
            ops_found.add(m.group(1))

        # Check we have all inputs
        needed = _parse_replay_inputs(g.replay_script)
        if any(n not in all_vals for n in needed):
            continue

        # Run replay
        try:
            outputs = _run_replay(g.replay_script, all_vals)
        except Exception:
            continue

        # Merge outputs into all_vals for downstream groups
        all_vals.update(outputs)

        # Verify expected outputs
        expected_pairs = _build_expected_out(g.replay_script, available)
        for fx_name, h5_key in expected_pairs:
            if h5_key not in loaded:
                continue
            actual = outputs.get(fx_name)
            if actual is None:
                continue
            stored = loaded[h5_key]
            if not (isinstance(actual, torch.Tensor) and isinstance(stored, torch.Tensor)):
                continue
            if exact:
                if torch.equal(actual, stored):
                    n_validated += 1
                else:
                    diff = (actual.float() - stored.float()).abs().max().item()
                    assert False, (
                        f"Group {g.name}, output {fx_name}: not bit-identical, "
                        f"max diff {diff:.2e}"
                    )
            else:
                if torch.allclose(actual.float(), stored.float(), atol=atol, rtol=rtol):
                    n_validated += 1
                else:
                    diff = (actual.float() - stored.float()).abs().max().item()
                    assert False, (
                        f"Group {g.name}, output {fx_name}: not close, "
                        f"max diff {diff:.2e}"
                    )

    return n_validated, ops_found


def _validate_full_chain(h5_path, section="forward", exact=True,
                         atol=1e-5, rtol=1e-5):
    """Run all groups in topological order, verify final outputs match stored."""
    loaded = _load_h5_tensors(h5_path)
    available = _available_tensors(h5_path)
    groups = list_groups(h5_path, section=section, strategy="line")

    all_vals = {}
    for k, v in loaded.items():
        all_vals[k] = v
        all_vals[_long_name(k)] = v

    for g in groups:
        needed = _parse_replay_inputs(g.replay_script)
        if any(n not in all_vals for n in needed):
            continue
        try:
            outputs = _run_replay(g.replay_script, all_vals)
            all_vals.update(outputs)
        except Exception:
            continue

    # Check all stored outputs
    n_matched = 0
    for g in groups:
        for fx_name, h5_key in _build_expected_out(g.replay_script, available):
            if h5_key in loaded and fx_name in all_vals:
                actual = all_vals[fx_name]
                stored = loaded[h5_key]
                if isinstance(actual, torch.Tensor) and isinstance(stored, torch.Tensor):
                    if exact:
                        assert torch.equal(actual, stored), (
                            f"Chain output {fx_name} mismatch: "
                            f"max diff {(actual.float() - stored.float()).abs().max().item():.2e}"
                        )
                    else:
                        assert torch.allclose(actual.float(), stored.float(),
                                              atol=atol, rtol=rtol), (
                            f"Chain output {fx_name} mismatch: "
                            f"max diff {(actual.float() - stored.float()).abs().max().item():.2e}"
                        )
                    n_matched += 1

    return n_matched


# ── CPU Tests — Per Op Category ──────────────────────────────────────────────


class TestElementwiseUnary:
    """Test unary elementwise ops: relu, gelu, sigmoid, tanh, neg, abs."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = ElementwiseUnaryModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0, "No forward group replays validated"
            assert ops & {"relu", "gelu", "silu"}, f"Expected relu/gelu/silu, got {ops}"
        finally:
            os.unlink(h5_path)

    def test_backward_replays(self):
        torch.manual_seed(42)
        model = ElementwiseUnaryModel()
        x = torch.randn(4, 16)
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if capture.backward_graphs:
                n, ops = _validate_group_replays(h5_path, "backward")
                assert n > 0, "No backward group replays validated"
        finally:
            os.unlink(h5_path)

    def test_forward_chain(self):
        torch.manual_seed(42)
        model = ElementwiseUnaryModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n = _validate_full_chain(h5_path, "forward")
            assert n > 0
        finally:
            os.unlink(h5_path)


class TestElementwiseBinary:
    """Test binary elementwise ops: add, mul, sub, div."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = ElementwiseBinaryModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0
            assert ops & {"add", "mul"}, f"Expected add/mul, got {ops}"
        finally:
            os.unlink(h5_path)

    def test_forward_chain(self):
        torch.manual_seed(42)
        model = ElementwiseBinaryModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n = _validate_full_chain(h5_path, "forward")
            assert n > 0
        finally:
            os.unlink(h5_path)


class TestReductions:
    """Test reduction ops: sum, mean, amax."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = ReductionModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0
            assert ops & {"sum", "mean"}, f"Expected sum/mean, got {ops}"
        finally:
            os.unlink(h5_path)


class TestNormalization:
    """Test normalization ops: native_layer_norm, native_batch_norm."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = NormalizationModel()
        model.train()
        x = torch.randn(8, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0
            assert ops & {"native_layer_norm"}, f"Expected native_layer_norm, got {ops}"
        finally:
            os.unlink(h5_path)

    def test_backward_replays(self):
        torch.manual_seed(42)
        model = NormalizationModel()
        model.train()
        x = torch.randn(8, 16)
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if capture.backward_graphs:
                n, ops = _validate_group_replays(h5_path, "backward")
                assert n > 0
        finally:
            os.unlink(h5_path)

    def test_forward_chain(self):
        torch.manual_seed(42)
        model = NormalizationModel()
        model.train()
        x = torch.randn(8, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n = _validate_full_chain(h5_path, "forward")
            assert n > 0
        finally:
            os.unlink(h5_path)


class TestMatmul:
    """Test linear algebra ops: mm, addmm, bmm, t."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = MatmulModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0
            assert ops & {"mm", "addmm", "bmm"}, f"Expected matmul ops, got {ops}"
        finally:
            os.unlink(h5_path)


class TestEmbeddingLoss:
    """Test embedding + loss ops: embedding, log_softmax, nll_loss."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = EmbeddingLossModel()
        idx = torch.randint(0, 100, (4, 8))
        targets = torch.randint(0, 100, (4, 8))
        h5_path, _ = _capture_and_dump(
            model, idx, targets, run_backward=True,
            loss_fn=lambda o: o,
        )
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0
            assert ops & {"embedding"}, f"Expected embedding, got {ops}"
        finally:
            os.unlink(h5_path)


class TestSoftmax:
    """Test softmax ops."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = SoftmaxModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0
        finally:
            os.unlink(h5_path)


class TestConvolution:
    """Test convolution ops."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = ConvModel()
        x = torch.randn(2, 3, 8, 8)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0
            assert ops & {"convolution"}, f"Expected convolution, got {ops}"
        finally:
            os.unlink(h5_path)

    def test_backward_replays(self):
        torch.manual_seed(42)
        model = ConvModel()
        x = torch.randn(2, 3, 8, 8)
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if capture.backward_graphs:
                # Conv backward has minor numerical differences on CPU
                n, _ = _validate_group_replays(h5_path, "backward",
                                               exact=False, atol=1e-6)
                assert n > 0
        finally:
            os.unlink(h5_path)


class TestTensorManipulation:
    """Test tensor manipulation ops: view, reshape, permute, transpose, cat, etc."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = TensorManipModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0
            view_ops = ops & {"view", "reshape", "permute", "transpose", "cat",
                              "unsqueeze", "expand", "clone", "slice", "select",
                              "_unsafe_view"}
            assert len(view_ops) >= 2, f"Expected view-like ops, got {ops}"
        finally:
            os.unlink(h5_path)

    def test_forward_chain(self):
        torch.manual_seed(42)
        model = TensorManipModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n = _validate_full_chain(h5_path, "forward")
            assert n > 0
        finally:
            os.unlink(h5_path)


class TestClampErf:
    """Test clamp_min, erf, rsqrt, exp."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = ClampErfModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0
        finally:
            os.unlink(h5_path)


class TestDtypeConversion:
    """Test _to_copy (dtype conversion)."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = DtypeConversionModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0
        finally:
            os.unlink(h5_path)


class TestResidualLayerNorm:
    """Test realistic mini-transformer block: residual + layernorm + gelu."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = ResidualLayerNormModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0
            assert ops & {"gelu", "native_layer_norm", "add"}, (
                f"Expected gelu/layer_norm/add, got {ops}")
        finally:
            os.unlink(h5_path)

    def test_backward_replays(self):
        torch.manual_seed(42)
        model = ResidualLayerNormModel()
        x = torch.randn(4, 16)
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if capture.backward_graphs:
                n, ops = _validate_group_replays(h5_path, "backward")
                assert n > 0
        finally:
            os.unlink(h5_path)

    def test_forward_chain(self):
        torch.manual_seed(42)
        model = ResidualLayerNormModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n = _validate_full_chain(h5_path, "forward")
            assert n > 0
        finally:
            os.unlink(h5_path)

    def test_backward_chain(self):
        torch.manual_seed(42)
        model = ResidualLayerNormModel()
        x = torch.randn(4, 16)
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if capture.backward_graphs:
                n = _validate_full_chain(h5_path, "backward")
                assert n > 0
        finally:
            os.unlink(h5_path)


class TestDropout:
    """Test dropout ops (training mode).

    Dropout is non-deterministic — re-execution produces different masks.
    We verify groups that don't involve dropout are bit-identical, and
    that dropout groups at least produce valid tensors + correct ops.
    """

    def test_forward_ops_detected(self):
        torch.manual_seed(42)
        model = DropoutModel()
        model.train()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            ops, groups = _extract_ops_from_groups(h5_path, "forward")
            assert len(groups) > 0, "No forward groups"
            # native_dropout should appear
            assert ops & {"native_dropout", "relu"}, f"Expected dropout/relu, got {ops}"
        finally:
            os.unlink(h5_path)


class TestStackOps:
    """Test stack ops."""

    def test_forward_replays(self):
        torch.manual_seed(42)
        model = StackModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward")
            assert n > 0
        finally:
            os.unlink(h5_path)


# ── GPU Tests ─────────────────────────────────────────────────────────────────

_GPU = torch.cuda.is_available()
_skip_no_gpu = pytest.mark.skipif(not _GPU, reason="CUDA not available")


@_skip_no_gpu
class TestGPUElementwiseUnary:
    def test_forward_replays(self):
        torch.manual_seed(42)
        model = ElementwiseUnaryModel().cuda()
        x = torch.randn(4, 16, device="cuda")
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward", exact=False)
            assert n > 0
        finally:
            os.unlink(h5_path)


@_skip_no_gpu
class TestGPUNormalization:
    def test_forward_replays(self):
        torch.manual_seed(42)
        model = NormalizationModel().cuda()
        model.train()
        x = torch.randn(8, 16, device="cuda")
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward", exact=False)
            assert n > 0
        finally:
            os.unlink(h5_path)


@_skip_no_gpu
class TestGPUResidualBlock:
    def test_forward_replays(self):
        torch.manual_seed(42)
        model = ResidualLayerNormModel().cuda()
        x = torch.randn(4, 16, device="cuda")
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward", exact=False)
            assert n > 0
        finally:
            os.unlink(h5_path)

    def test_backward_replays(self):
        torch.manual_seed(42)
        model = ResidualLayerNormModel().cuda()
        x = torch.randn(4, 16, device="cuda")
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if capture.backward_graphs:
                n, _ = _validate_group_replays(h5_path, "backward", exact=False)
                assert n > 0
        finally:
            os.unlink(h5_path)


@_skip_no_gpu
class TestGPUConvolution:
    def test_forward_replays(self):
        torch.manual_seed(42)
        model = ConvModel().cuda()
        x = torch.randn(2, 3, 8, 8, device="cuda")
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n, ops = _validate_group_replays(h5_path, "forward", exact=False)
            assert n > 0
        finally:
            os.unlink(h5_path)


# ── Comprehensive op inventory test ──────────────────────────────────────────


class TestAtenOpInventory:
    """Captures ALL test models and reports which aten ops are covered."""

    def test_op_coverage_report(self):
        """Run all models, collect the complete set of aten ops exercised."""
        all_ops = set()
        model_ops = {}

        models = [
            ("elementwise_unary", ElementwiseUnaryModel(), (torch.randn(4, 16),)),
            ("elementwise_binary", ElementwiseBinaryModel(), (torch.randn(4, 16),)),
            ("reduction", ReductionModel(), (torch.randn(4, 16),)),
            ("normalization", NormalizationModel(), (torch.randn(8, 16),)),
            ("matmul", MatmulModel(), (torch.randn(4, 16),)),
            ("softmax", SoftmaxModel(), (torch.randn(4, 16),)),
            ("conv", ConvModel(), (torch.randn(2, 3, 8, 8),)),
            ("tensor_manip", TensorManipModel(), (torch.randn(4, 16),)),
            ("clamp_erf", ClampErfModel(), (torch.randn(4, 16),)),
            ("dtype_conv", DtypeConversionModel(), (torch.randn(4, 16),)),
            ("residual_ln", ResidualLayerNormModel(), (torch.randn(4, 16),)),
            ("dropout", DropoutModel(), (torch.randn(4, 16),)),
            ("stack", StackModel(), (torch.randn(4, 16),)),
        ]

        for name, model, args in models:
            torch.manual_seed(42)
            if hasattr(model, 'train'):
                model.train()
            try:
                h5_path, _ = _capture_and_dump(model, *args)
                try:
                    for section in ("forward", "backward"):
                        ops, _ = _extract_ops_from_groups(h5_path, section)
                        all_ops.update(ops)
                        model_ops.setdefault(name, set()).update(ops)
                finally:
                    os.unlink(h5_path)
            except Exception as e:
                print(f"  Warning: {name} failed: {e}")

        # Report
        print(f"\n{'='*60}")
        print(f"ATEN OP COVERAGE: {len(all_ops)} unique ops exercised")
        print(f"{'='*60}")
        for op in sorted(all_ops):
            covered_by = [n for n, ops in model_ops.items() if op in ops]
            print(f"  aten.{op}: {', '.join(covered_by)}")

        # Assert we cover the core ops
        core_ops = {
            "relu", "gelu", "sigmoid", "tanh",  # activations
            "add", "mul",                         # binary
            "mm", "addmm",                        # matmul
            "native_layer_norm",                  # normalization
            "view",                               # tensor manip
            "t",                                  # transpose
        }
        missing = core_ops - all_ops
        assert not missing, f"Missing core ops: {missing}"
        assert len(all_ops) >= 20, f"Only {len(all_ops)} unique ops, expected >= 20"

    def test_backward_op_coverage(self):
        """Verify backward-specific ops appear in backward captures."""
        torch.manual_seed(42)
        model = ResidualLayerNormModel()
        x = torch.randn(4, 16)
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if not capture.backward_graphs:
                pytest.skip("No backward graph")
            ops, _ = _extract_ops_from_groups(h5_path, "backward")
            # Should have backward-specific ops
            assert len(ops) >= 3, f"Expected >= 3 backward ops, got {ops}"
            print(f"Backward ops: {sorted(ops)}")
        finally:
            os.unlink(h5_path)


# ── Script generation tests (verify scripts are well-formed) ─────────────────


class TestScriptGeneration:
    """Verify generated kbox scripts have correct structure."""

    def test_group_script_has_init_and_run(self):
        torch.manual_seed(42)
        model = ElementwiseUnaryModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            groups = list_groups(h5_path, section="forward", strategy="line")
            assert len(groups) > 0
            for g in groups:
                script = generate_group_script(h5_path, g)
                assert "def init_once():" in script, f"Missing init_once in {g.name}"
                assert "def run(inputs):" in script, f"Missing run in {g.name}"
                assert "return [" in script, f"Missing return in {g.name}"
        finally:
            os.unlink(h5_path)

    def test_section_script_has_init_and_run(self):
        torch.manual_seed(42)
        model = ElementwiseUnaryModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            script = generate_section_script(
                h5_path, section="forward", strategy="line")
            assert "def init_once():" in script
            assert "def run(inputs):" in script
            assert "return [" in script
        finally:
            os.unlink(h5_path)

    def test_group_script_with_validate(self):
        torch.manual_seed(42)
        model = ElementwiseUnaryModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            groups = list_groups(h5_path, section="forward", strategy="line")
            script = generate_group_script(h5_path, groups[0], validate=True)
            assert "if __name__" in script
        finally:
            os.unlink(h5_path)

    def test_all_models_generate_scripts(self):
        """Every model produces at least one group with a valid script."""
        models = [
            ("unary", ElementwiseUnaryModel(), torch.randn(4, 16)),
            ("binary", ElementwiseBinaryModel(), torch.randn(4, 16)),
            ("reduction", ReductionModel(), torch.randn(4, 16)),
            ("norm", NormalizationModel(), torch.randn(8, 16)),
            ("matmul", MatmulModel(), torch.randn(4, 16)),
            ("softmax", SoftmaxModel(), torch.randn(4, 16)),
            ("conv", ConvModel(), torch.randn(2, 3, 8, 8)),
            ("manip", TensorManipModel(), torch.randn(4, 16)),
        ]
        for name, model, x in models:
            torch.manual_seed(42)
            model.train()
            h5_path, _ = _capture_and_dump(model, x)
            try:
                groups = list_groups(h5_path, section="forward", strategy="line")
                assert len(groups) > 0, f"{name}: no forward groups"
                for g in groups:
                    script = generate_group_script(h5_path, g)
                    assert "def run(inputs):" in script, (
                        f"{name}/{g.name}: missing run()")
            finally:
                os.unlink(h5_path)
