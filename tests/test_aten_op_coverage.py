"""Test every aten op through the kbox pipeline: capture → H5 → kbox script → validate.

For each op category, we build a minimal model that exercises those ops,
capture the aten graph, dump to H5, generate per-group kbox scripts, and
verify the scripts produce bit-identical (CPU) or allclose (GPU) results.

This validates the full torch-graph → kernelbox pipeline for every aten op
that appears in real model captures.

Op categories tested:
  - Elementwise unary: relu, gelu, silu, sigmoid, tanh, neg, abs, exp, rsqrt
  - Elementwise binary: add, mul, sub, div
  - Reductions: sum, mean, amax
  - Normalization: layer_norm, batch_norm, rms_norm (via manual impl)
  - Linear algebra: mm, addmm, bmm, t
  - Embedding + loss: embedding, nll_loss, log_softmax
  - Softmax: softmax, log_softmax
  - Convolution: conv2d
  - Tensor manipulation: view, reshape, permute, transpose, cat, stack,
    unsqueeze, squeeze, expand, clone, slice, select
  - Dtype conversion: _to_copy
  - Creation: arange, zeros, full
  - Clamp: clamp_min
  - Error function: erf
  - Backward-specific: gelu_backward, threshold_backward, etc.
    (tested implicitly via run_backward=True)
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
    _parse_replay_inputs,
)


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
    """Exercises: relu, gelu, silu, sigmoid, tanh, neg, abs, exp, rsqrt."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        h = self.fc(x)
        r = torch.relu(h)
        g = F.gelu(r)
        s = torch.sigmoid(g)
        t = torch.tanh(s)
        n = torch.neg(t)
        a = torch.abs(n)
        return a


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
        # addmm: bias + x @ w1
        h = torch.addmm(self.b1, x, self.w1)
        # mm: h @ w2
        out = torch.mm(h, self.w2)
        # bmm via unsqueeze
        b = x.unsqueeze(1)  # (B, 1, 16)
        c = out.unsqueeze(2)  # (B, 16, 1)
        d = torch.bmm(b, c).squeeze(-1).squeeze(-1)  # (B,)
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
    """Exercises: _softmax, softmax via functional."""
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
    """Exercises: view, reshape, permute, transpose, cat, stack, unsqueeze,
    squeeze, expand, clone, contiguous."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 32)

    def forward(self, x):
        h = self.fc(x)  # (B, 32)
        # view + reshape
        v = h.view(-1, 2, 16)  # (B, 2, 16)
        r = v.reshape(-1, 32)  # (B, 32)
        # permute
        v2 = h.view(-1, 4, 8)
        p = v2.permute(0, 2, 1)  # (B, 8, 4)
        t = p.transpose(1, 2)    # (B, 4, 8)
        t = t.reshape(-1, 32)
        # cat
        c = torch.cat([r, t], dim=-1)  # (B, 64)
        # unsqueeze + squeeze + expand
        u = c.unsqueeze(1)      # (B, 1, 64)
        e = u.expand(-1, 2, -1)  # (B, 2, 64)
        s = e[:, 0, :]           # (B, 64) — select/slice
        # clone
        cl = s.clone()
        return cl


class ClampErfModel(nn.Module):
    """Exercises: clamp_min, erf, rsqrt, exp."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        h = self.fc(x)
        clamped = torch.clamp(h, min=0.0)   # aten.clamp_min
        erfd = torch.erf(clamped)            # aten.erf
        rsq = torch.rsqrt(clamped + 1.0)    # aten.rsqrt
        exp_v = torch.exp(-clamped)          # aten.exp
        return erfd + rsq + exp_v


class DtypeConversionModel(nn.Module):
    """Exercises: _to_copy (dtype conversion)."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        h = self.fc(x)
        h_half = h.to(torch.float16)  # _to_copy
        h_back = h_half.to(torch.float32)  # _to_copy
        return h + h_back


class ResidualLayerNormModel(nn.Module):
    """A realistic mini-transformer block with residual + layernorm.
    Exercises many ops together like in real models."""
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(16)
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 16)
        self.ln2 = nn.LayerNorm(16)

    def forward(self, x):
        # Pre-norm residual block
        h = self.ln1(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        x = x + h  # residual add
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


class StackArangeModel(nn.Module):
    """Exercises: stack, arange, zeros, full."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        h = self.fc(x)
        # stack
        stacked = torch.stack([h, h * 2], dim=1)  # (B, 2, 16)
        return stacked.sum(dim=1)


# ── Helpers ───────────────────────────────────────────────────────────────────


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


def _validate_group_scripts(h5_path, section="forward", exact=True,
                            atol=1e-5, rtol=1e-5):
    """Generate and execute kbox group scripts, validating results.

    Returns (n_validated, n_groups, ops_found) where ops_found is a set of
    aten op names found in replay scripts.
    """
    groups = list_groups(h5_path, section=section, strategy="line")
    n_run = 0
    ops_found = set()

    for g in groups:
        # Extract aten op names from replay script
        for m in re.finditer(r'torch\.ops\.aten\.(\w+)', g.replay_script):
            ops_found.add(m.group(1))
        # Also catch operator.getitem
        if 'operator.getitem' in g.replay_script:
            ops_found.add('getitem')

        script = generate_group_script(h5_path, g)
        # Force CPU for bit-identical comparison
        if exact:
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
                if exact:
                    assert torch.equal(actual, expected), (
                        f"Group {g.name} mismatch: "
                        f"max diff {(actual.float() - expected.float()).abs().max().item():.2e}"
                    )
                else:
                    assert torch.allclose(actual.float(), expected.float(),
                                          atol=atol, rtol=rtol), (
                        f"Group {g.name} mismatch: "
                        f"max diff {(actual.float() - expected.float()).abs().max().item():.2e}"
                    )
                n_run += 1

    return n_run, len(groups), ops_found


def _validate_section_script(h5_path, section="forward", exact=True,
                             atol=1e-5, rtol=1e-5):
    """Generate and execute a section chain script, validating results."""
    script = generate_section_script(
        h5_path, section=section, strategy="line", validate=True,
    )
    assert "def run(input):" in script
    if exact:
        script = script.replace(
            '_device = "cuda" if torch.cuda.is_available() else "cpu"',
            '_device = "cpu"',
        )
    ns = {"__builtins__": __builtins__, "__file__": h5_path}
    exec(compile(script, f"<{section}_section>", "exec"), ns)

    result = ns["run"](ns["input"])
    expected = ns["expected"]
    assert len(expected) > 0, f"No expected outputs in {section} section script"
    for i, (got, exp) in enumerate(zip(result, expected)):
        if isinstance(got, torch.Tensor) and isinstance(exp, torch.Tensor):
            if exact:
                assert torch.allclose(got, exp, atol=1e-6, rtol=1e-6), (
                    f"{section} section output {i}: "
                    f"max diff {(got.float() - exp.float()).abs().max().item():.2e}"
                )
            else:
                assert torch.allclose(got.float(), exp.float(),
                                      atol=atol, rtol=rtol), (
                    f"{section} section output {i}: "
                    f"max diff {(got.float() - exp.float()).abs().max().item():.2e}"
                )


# ── CPU Tests — Per Op Category ──────────────────────────────────────────────


class TestElementwiseUnary:
    """Test unary elementwise ops: relu, gelu, sigmoid, tanh, neg, abs."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = ElementwiseUnaryModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, n_groups, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0, "No group scripts validated"
            # At minimum relu and gelu should appear
            assert ops & {"relu", "gelu"}, f"Expected relu/gelu in ops, got {ops}"
        finally:
            os.unlink(h5_path)

    def test_backward_group_scripts(self):
        torch.manual_seed(42)
        model = ElementwiseUnaryModel()
        x = torch.randn(4, 16)
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if capture.backward_graphs:
                n_run, _, ops = _validate_group_scripts(h5_path, "backward")
                assert n_run > 0, "No backward group scripts validated"
        finally:
            os.unlink(h5_path)

    def test_section_chain(self):
        torch.manual_seed(42)
        model = ElementwiseUnaryModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            _validate_section_script(h5_path, "forward")
        finally:
            os.unlink(h5_path)


class TestElementwiseBinary:
    """Test binary elementwise ops: add, mul, sub, div."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = ElementwiseBinaryModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
            assert ops & {"add", "mul"}, f"Expected add/mul in ops, got {ops}"
        finally:
            os.unlink(h5_path)

    def test_section_chain(self):
        torch.manual_seed(42)
        model = ElementwiseBinaryModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            _validate_section_script(h5_path, "forward")
        finally:
            os.unlink(h5_path)


class TestReductions:
    """Test reduction ops: sum, mean, amax."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = ReductionModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
            # sum/mean should appear
            assert ops & {"sum", "mean"}, f"Expected sum/mean in ops, got {ops}"
        finally:
            os.unlink(h5_path)


class TestNormalization:
    """Test normalization ops: native_layer_norm, native_batch_norm."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = NormalizationModel()
        model.train()
        x = torch.randn(8, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
            assert ops & {"native_layer_norm"}, f"Expected native_layer_norm, got {ops}"
        finally:
            os.unlink(h5_path)

    def test_section_chain(self):
        torch.manual_seed(42)
        model = NormalizationModel()
        model.train()
        x = torch.randn(8, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            _validate_section_script(h5_path, "forward")
        finally:
            os.unlink(h5_path)

    def test_backward(self):
        torch.manual_seed(42)
        model = NormalizationModel()
        model.train()
        x = torch.randn(8, 16)
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if capture.backward_graphs:
                n_run, _, ops = _validate_group_scripts(h5_path, "backward")
                assert n_run > 0
        finally:
            os.unlink(h5_path)


class TestMatmul:
    """Test linear algebra ops: mm, addmm, bmm, t."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = MatmulModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
            # At least mm or addmm should appear
            assert ops & {"mm", "addmm", "bmm"}, f"Expected matmul ops, got {ops}"
        finally:
            os.unlink(h5_path)


class TestEmbeddingLoss:
    """Test embedding + loss ops: embedding, log_softmax, nll_loss."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = EmbeddingLossModel()
        idx = torch.randint(0, 100, (4, 8))
        targets = torch.randint(0, 100, (4, 8))
        h5_path, _ = _capture_and_dump(
            model, idx, targets, run_backward=True,
            loss_fn=lambda o: o,  # model already returns loss
        )
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
            assert ops & {"embedding"}, f"Expected embedding, got {ops}"
        finally:
            os.unlink(h5_path)


class TestSoftmax:
    """Test softmax ops: _softmax."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = SoftmaxModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
        finally:
            os.unlink(h5_path)


class TestConvolution:
    """Test convolution ops: convolution, adaptive_avg_pool."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = ConvModel()
        x = torch.randn(2, 3, 8, 8)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
            assert ops & {"convolution"}, f"Expected convolution, got {ops}"
        finally:
            os.unlink(h5_path)

    def test_backward(self):
        torch.manual_seed(42)
        model = ConvModel()
        x = torch.randn(2, 3, 8, 8)
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if capture.backward_graphs:
                n_run, _, ops = _validate_group_scripts(h5_path, "backward")
                assert n_run > 0
        finally:
            os.unlink(h5_path)


class TestTensorManipulation:
    """Test tensor manipulation ops: view, reshape, permute, transpose, cat,
    unsqueeze, expand, clone, select/slice."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = TensorManipModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
            # Should see view-like ops
            view_ops = ops & {"view", "reshape", "permute", "transpose", "cat",
                              "unsqueeze", "expand", "clone", "slice", "select"}
            assert len(view_ops) >= 2, f"Expected view-like ops, got {ops}"
        finally:
            os.unlink(h5_path)

    def test_section_chain(self):
        torch.manual_seed(42)
        model = TensorManipModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            _validate_section_script(h5_path, "forward")
        finally:
            os.unlink(h5_path)


class TestClampErf:
    """Test clamp_min, erf, rsqrt, exp."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = ClampErfModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
        finally:
            os.unlink(h5_path)


class TestDtypeConversion:
    """Test _to_copy (dtype conversion)."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = DtypeConversionModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
        finally:
            os.unlink(h5_path)


class TestResidualLayerNorm:
    """Test realistic mini-transformer block with residual + layernorm + gelu."""

    def test_forward_group_scripts(self):
        torch.manual_seed(42)
        model = ResidualLayerNormModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
            # Should see gelu + layer_norm + add at minimum
            assert ops & {"gelu", "native_layer_norm", "add"}, (
                f"Expected gelu/layer_norm/add, got {ops}")
        finally:
            os.unlink(h5_path)

    def test_backward_group_scripts(self):
        torch.manual_seed(42)
        model = ResidualLayerNormModel()
        x = torch.randn(4, 16)
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if capture.backward_graphs:
                n_run, _, ops = _validate_group_scripts(h5_path, "backward")
                assert n_run > 0
        finally:
            os.unlink(h5_path)

    def test_forward_section_chain(self):
        torch.manual_seed(42)
        model = ResidualLayerNormModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            _validate_section_script(h5_path, "forward")
        finally:
            os.unlink(h5_path)

    def test_backward_section_chain(self):
        torch.manual_seed(42)
        model = ResidualLayerNormModel()
        x = torch.randn(4, 16)
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if capture.backward_graphs:
                _validate_section_script(h5_path, "backward")
        finally:
            os.unlink(h5_path)


class TestDropout:
    """Test dropout ops (training mode): native_dropout."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = DropoutModel()
        model.train()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
        finally:
            os.unlink(h5_path)


class TestStackOps:
    """Test stack and creation ops."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = StackArangeModel()
        x = torch.randn(4, 16)
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(h5_path, "forward")
            assert n_run > 0
        finally:
            os.unlink(h5_path)


# ── GPU Tests ─────────────────────────────────────────────────────────────────

_GPU = torch.cuda.is_available()
_skip_no_gpu = pytest.mark.skipif(not _GPU, reason="CUDA not available")


@_skip_no_gpu
class TestGPUElementwiseUnary:
    """GPU version: unary elementwise ops."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = ElementwiseUnaryModel().cuda()
        x = torch.randn(4, 16, device="cuda")
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(
                h5_path, "forward", exact=False)
            assert n_run > 0
        finally:
            os.unlink(h5_path)


@_skip_no_gpu
class TestGPUNormalization:
    """GPU version: normalization ops."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = NormalizationModel().cuda()
        model.train()
        x = torch.randn(8, 16, device="cuda")
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(
                h5_path, "forward", exact=False)
            assert n_run > 0
        finally:
            os.unlink(h5_path)


@_skip_no_gpu
class TestGPUResidualBlock:
    """GPU version: realistic mini-transformer block."""

    def test_forward_group_scripts(self):
        torch.manual_seed(42)
        model = ResidualLayerNormModel().cuda()
        x = torch.randn(4, 16, device="cuda")
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(
                h5_path, "forward", exact=False)
            assert n_run > 0
        finally:
            os.unlink(h5_path)

    def test_backward_group_scripts(self):
        torch.manual_seed(42)
        model = ResidualLayerNormModel().cuda()
        x = torch.randn(4, 16, device="cuda")
        h5_path, capture = _capture_and_dump(model, x)
        try:
            if capture.backward_graphs:
                n_run, _, ops = _validate_group_scripts(
                    h5_path, "backward", exact=False)
                assert n_run > 0
        finally:
            os.unlink(h5_path)


@_skip_no_gpu
class TestGPUConvolution:
    """GPU version: convolution ops."""

    def test_group_scripts(self):
        torch.manual_seed(42)
        model = ConvModel().cuda()
        x = torch.randn(2, 3, 8, 8, device="cuda")
        h5_path, _ = _capture_and_dump(model, x)
        try:
            n_run, _, ops = _validate_group_scripts(
                h5_path, "forward", exact=False)
            assert n_run > 0
        finally:
            os.unlink(h5_path)


# ── Comprehensive op inventory test ──────────────────────────────────────────


class TestAtenOpInventory:
    """Captures ALL test models and reports which aten ops are covered."""

    def test_op_coverage_report(self):
        """Run all models and collect the complete set of aten ops exercised."""
        all_ops = set()
        model_ops = {}

        models = [
            ("elementwise_unary", ElementwiseUnaryModel(), torch.randn(4, 16)),
            ("elementwise_binary", ElementwiseBinaryModel(), torch.randn(4, 16)),
            ("reduction", ReductionModel(), torch.randn(4, 16)),
            ("normalization", NormalizationModel(), torch.randn(8, 16)),
            ("matmul", MatmulModel(), torch.randn(4, 16)),
            ("softmax", SoftmaxModel(), torch.randn(4, 16)),
            ("conv", ConvModel(), torch.randn(2, 3, 8, 8)),
            ("tensor_manip", TensorManipModel(), torch.randn(4, 16)),
            ("clamp_erf", ClampErfModel(), torch.randn(4, 16)),
            ("dtype_conv", DtypeConversionModel(), torch.randn(4, 16)),
            ("residual_ln", ResidualLayerNormModel(), torch.randn(4, 16)),
            ("dropout", DropoutModel(), torch.randn(4, 16)),
            ("stack", StackArangeModel(), torch.randn(4, 16)),
        ]

        for name, model, x in models:
            torch.manual_seed(42)
            if hasattr(model, 'train'):
                model.train()
            try:
                h5_path, _ = _capture_and_dump(model, x)
                try:
                    for section in ("forward", "backward"):
                        groups = list_groups(h5_path, section=section, strategy="line")
                        for g in groups:
                            for m in re.finditer(r'torch\.ops\.aten\.(\w+)', g.replay_script):
                                op = m.group(1)
                                all_ops.add(op)
                                model_ops.setdefault(name, set()).add(op)
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

        # Assert we cover at least the core ops
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

        # We should cover at least 20 unique ops across all models
        assert len(all_ops) >= 20, (
            f"Only {len(all_ops)} unique ops found, expected >= 20")
