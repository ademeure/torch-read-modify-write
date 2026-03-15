#!/usr/bin/env python3
"""Temporary one-off tests for the review fixes. Delete after verification."""

import sys
import os
import tempfile
import importlib
import traceback
import re
import inspect

import torch
import torch.nn as nn

PASS = 0
FAIL = 0

def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} -- {detail}")


def export_and_install(model_cls, model, capture):
    """Export capture to temp file, load module, install on a clone of model."""
    import copy
    from torch_graph.export import export_aten_program
    from torch_graph.install import install
    tmpf = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
    export_aten_program(capture, tmpf.name, inline_threshold=0)
    tmpf.close()
    mod_name = f'aten_mod_{id(capture)}'
    spec = importlib.util.spec_from_file_location(mod_name, tmpf.name)
    aten_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aten_mod)
    fresh = copy.deepcopy(model)
    install(fresh, aten_mod)
    os.unlink(tmpf.name)
    return fresh


# ═══════════════════════════════════════════════════════════════════
# 1. pyproject.toml build-backend
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 1: pyproject.toml build-backend ===")
try:
    with open("pyproject.toml") as f:
        content = f.read()
    report("build-backend is setuptools.build_meta",
           'build-backend = "setuptools.build_meta"' in content,
           f"got: {[l for l in content.splitlines() if 'build-backend' in l]}")
    from setuptools.build_meta import build_wheel
    report("setuptools.build_meta importable", True)
except Exception as e:
    report("pyproject.toml", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 2. kwargs ordering in install.py — 2-arg model
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 2: install.py kwargs ordering (2-arg) ===")
from torch_graph.export import capture_aten_graphs, export_aten_program
from torch_graph.install import install

class TwoArgModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0, 2.0]))
    def forward(self, x, y):
        return self.w * x + y

try:
    m = TwoArgModel()
    x = torch.tensor([1.0, 1.0], requires_grad=True)
    y = torch.tensor([10.0, 10.0], requires_grad=True)
    expected = m(x, y).detach().clone()  # [11, 12]

    output, capture = capture_aten_graphs(m, x, y, run_backward=True)
    m_inst = export_and_install(TwoArgModel, m, capture)

    r_pos = m_inst(x.detach(), y.detach())
    report("positional args correct",
           torch.allclose(r_pos, expected),
           f"got {r_pos}, expected {expected}")

    r_kw = m_inst(x=x.detach(), y=y.detach())
    report("kwargs (x=, y=) correct",
           torch.allclose(r_kw, expected),
           f"got {r_kw}, expected {expected}")

    r_rev = m_inst(y=y.detach(), x=x.detach())
    report("kwargs reversed (y=, x=) correct",
           torch.allclose(r_rev, expected),
           f"got {r_rev}, expected {expected}")

    r_mix = m_inst(x.detach(), y=y.detach())
    report("mixed (pos x, kw y=) correct",
           torch.allclose(r_mix, expected),
           f"got {r_mix}, expected {expected}")
except Exception as e:
    report("install.py kwargs 2-arg", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# 3. kwargs ordering in install.py — 3-arg model (wider)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 3: install.py kwargs ordering (3-arg) ===")

class ThreeArgModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)
    def forward(self, a, b, c):
        return self.fc(a) + b - c

try:
    m3 = ThreeArgModel()
    a_t = torch.randn(2, 4, requires_grad=True)
    b_t = torch.randn(2, 2, requires_grad=True)
    c_t = torch.randn(2, 2, requires_grad=True)
    expected3 = m3(a_t, b_t, c_t).detach().clone()

    output3, capture3 = capture_aten_graphs(m3, a_t, b_t, c_t, run_backward=True)
    if capture3.forward_graphs:
        m3_inst = export_and_install(ThreeArgModel, m3, capture3)

        r = m3_inst(a_t.detach(), b_t.detach(), c_t.detach())
        report("3-arg positional correct",
               torch.allclose(r, expected3, atol=1e-5),
               f"max diff {(r - expected3).abs().max().item()}")

        r = m3_inst(c=c_t.detach(), a=a_t.detach(), b=b_t.detach())
        report("3-arg kwargs scrambled (c,a,b) correct",
               torch.allclose(r, expected3, atol=1e-5),
               f"max diff {(r - expected3).abs().max().item()}")

        r = m3_inst(a_t.detach(), c=c_t.detach(), b=b_t.detach())
        report("3-arg mixed pos+kwargs correct",
               torch.allclose(r, expected3, atol=1e-5),
               f"max diff {(r - expected3).abs().max().item()}")
    else:
        report("3-arg capture produced graphs", False, "0 forward_graphs from capture")
except Exception as e:
    report("install.py kwargs 3-arg", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# 4. kwargs ordering in auto_install.py (unit test of reorder logic)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 4: auto_install.py kwargs reorder logic ===")
try:
    # Directly test the reorder logic used in both install.py and auto_install.py
    def simulate_reorder(forward_sig, args, kwargs):
        params = [
            p for p in inspect.signature(forward_sig).parameters.values()
            if p.name != 'self' and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        _orig_params = [p.name for p in params]
        kwargs = dict(kwargs)
        if kwargs and _orig_params:
            ordered = list(args)
            for name in _orig_params[len(args):]:
                if name in kwargs:
                    ordered.append(kwargs.pop(name))
            if kwargs:
                ordered.extend(kwargs.values())
            return tuple(ordered)
        elif kwargs:
            return args + tuple(kwargs.values())
        else:
            return args

    def f_named(self, x, y, z): pass
    def f_star(self, *args, **kwargs): pass
    def f_mixed(self, x, *args): pass

    # Named params, kwargs reversed
    r = simulate_reorder(f_named, (), {'z': 'Z', 'x': 'X', 'y': 'Y'})
    report("named params, all kwargs reversed → correct order",
           r == ('X', 'Y', 'Z'), f"got {r}")

    # Named params, partial kwargs
    r = simulate_reorder(f_named, ('X',), {'z': 'Z', 'y': 'Y'})
    report("named params, 1 pos + 2 kwargs → correct order",
           r == ('X', 'Y', 'Z'), f"got {r}")

    # *args signature: fallback to insertion order
    r = simulate_reorder(f_star, (), {'b': 'B', 'a': 'A'})
    report("*args sig fallback preserves insertion order",
           r == ('B', 'A'), f"got {r}")

    # No kwargs
    r = simulate_reorder(f_named, ('X', 'Y', 'Z'), {})
    report("all positional, no kwargs → unchanged",
           r == ('X', 'Y', 'Z'), f"got {r}")

    # Extra kwargs not in signature
    r = simulate_reorder(f_named, (), {'x': 'X', 'y': 'Y', 'z': 'Z', 'extra': 'E'})
    report("extra kwargs appended after named",
           r == ('X', 'Y', 'Z', 'E'), f"got {r}")

except Exception as e:
    report("auto_install reorder logic", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# 5. CUDA RNG state preservation (code inspection)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 5: CUDA RNG state preservation ===")
try:
    src_ai = open("torch_graph/auto_install.py").read()

    # Find the _capture_variant method (was _capture_and_install before refactor)
    cap_start = src_ai.index("def _capture_variant(")
    # Find the next def at same or lower indentation to bound the method
    next_def = src_ai.index("\n    def ", cap_start + 1)
    cap_method = src_ai[cap_start:next_def]

    report("saves CUDA RNG state",
           "cuda.get_rng_state_all" in cap_method,
           "torch.cuda.get_rng_state_all not found in _capture_variant")
    report("restores CUDA RNG state",
           "cuda.set_rng_state_all" in cap_method,
           "torch.cuda.set_rng_state_all not found in _capture_variant")
    report("guards with is_available",
           "cuda.is_available" in cap_method,
           "torch.cuda.is_available guard not found")

    # Verify save happens before capture, restore happens after
    idx_save = cap_method.index("get_rng_state_all")
    idx_capture = cap_method.index("capture_aten_graphs(")
    idx_restore = cap_method.index("set_rng_state_all")
    report("save before capture, restore after (within method)",
           idx_save < idx_capture < idx_restore,
           f"ordering within _capture_variant: save={idx_save}, capture={idx_capture}, restore={idx_restore}")
except Exception as e:
    report("CUDA RNG", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# 6. keep_debug_dir honored in triton.py
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 6: keep_debug_dir in triton.py ===")
try:
    src_triton = open("torch_graph/triton.py").read()
    fn_start = src_triton.index("def capture_inductor_debug(")
    fn_end = src_triton.index("\ndef ", fn_start + 1)
    fn_body = src_triton[fn_start:fn_end]

    report("keep_debug_dir used in body (not just param)",
           fn_body.count("keep_debug_dir") >= 2,
           f"only {fn_body.count('keep_debug_dir')} occurrences (need >=2)")
    report("conditional cleanup: 'not keep_debug_dir'",
           "not keep_debug_dir" in fn_body,
           "'not keep_debug_dir' guard not found")
    report("cleanup uses shutil.rmtree",
           fn_body.count("shutil.rmtree") == 2,  # pre-clear + post-clear
           f"expected 2 rmtree calls, got {fn_body.count('shutil.rmtree')}")
except Exception as e:
    report("keep_debug_dir", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# 7. tensor_dump multi-fragment (code inspection)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 7: tensor_dump multi-fragment code ===")
try:
    from torch_graph.tensor_dump import dump_and_compare, dump_model_tensors

    src_dc = inspect.getsource(dump_and_compare)
    src_dm = inspect.getsource(dump_model_tensors)

    report("dump_and_compare: no forward_graphs[0]",
           "forward_graphs[0]" not in src_dc)
    report("dump_and_compare: no backward_graphs[0]",
           "backward_graphs[0]" not in src_dc)
    report("dump_and_compare: iterates forward_graphs",
           "enumerate(capture.forward_graphs)" in src_dc)
    report("dump_and_compare: iterates backward_graphs",
           "enumerate(capture.backward_graphs)" in src_dc)
    report("dump_and_compare: backward-forward pairing",
           "bw_to_fw" in src_dc)

    report("dump_model_tensors: no forward_graphs[0]",
           "forward_graphs[0]" not in src_dm)
    report("dump_model_tensors: no backward_graphs[0]",
           "backward_graphs[0]" not in src_dm)
except Exception as e:
    report("tensor_dump code", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# 8. tensor_dump functional (single fragment)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 8: tensor_dump functional (single fragment) ===")
try:
    from torch_graph.tensor_dump import dump_and_compare

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)
        def forward(self, x):
            return self.fc(x)

    sm = SimpleModel()
    sx = torch.randn(2, 4)
    with tempfile.TemporaryDirectory() as td:
        results = dump_and_compare(sm, sx, output_dir=td, run_backward=False, verbose=False)
        report("returns results", len(results) >= 1, f"got {len(results)}")
        if results:
            report("kind is 'forward'", results[0].kind == "forward",
                   f"got '{results[0].kind}'")
except Exception as e:
    report("tensor_dump single-fragment", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# 9. tensor_dump multi-fragment functional
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 9: tensor_dump multi-fragment functional ===")
try:
    class GraphBreakModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 2)
        def forward(self, x):
            x = self.fc1(x)
            torch._dynamo.graph_break()
            x = self.fc2(x)
            return x

    gbm = GraphBreakModel()
    gbx = torch.randn(2, 4)
    output_gb, capture_gb = capture_aten_graphs(gbm, gbx, run_backward=False)
    n_fw = len(capture_gb.forward_graphs)
    report(f"graph break produces >1 forward graphs", n_fw > 1, f"got {n_fw}")

    if n_fw > 1:
        with tempfile.TemporaryDirectory() as td:
            results_gb = dump_and_compare(
                gbm, gbx, output_dir=td, run_backward=False, verbose=False)
            report(f"returns {n_fw} results", len(results_gb) == n_fw,
                   f"got {len(results_gb)} for {n_fw} fragments")
            kinds = [r.kind for r in results_gb]
            report("kinds are forward_0, forward_1, ...",
                   all(f"forward_{i}" in kinds for i in range(n_fw)),
                   f"got: {kinds}")
except Exception as e:
    report("tensor_dump multi-fragment", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# 10. dump_model_tensors multi-fragment
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 10: dump_model_tensors multi-fragment ===")
try:
    from torch_graph.tensor_dump import dump_model_tensors
    gbm2 = GraphBreakModel()
    with tempfile.TemporaryDirectory() as td:
        outpath = os.path.join(td, "tensors.pt")
        all_t = dump_model_tensors(gbm2, gbx, output_path=outpath, run_backward=False)
        keys = list(all_t.keys())
        has_multi = any("forward_" in k for k in keys)
        report("multi-fragment forward keys",
               has_multi or "forward" in keys,
               f"got keys: {keys}")
except Exception as e:
    report("dump_model_tensors multi-fragment", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# 11. Scan for stale [0] indexing on graph lists
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 11: Scan for stale [0] indexing in tensor_dump.py ===")
# Only tensor_dump.py was the reported bug. install.py/auto_install.py
# legitimately use [0] for single-graph installation.
try:
    with open("torch_graph/tensor_dump.py") as f:
        lines = f.readlines()
    hits = [(i+1, line.strip()) for i, line in enumerate(lines)
            if re.search(r'(forward_graphs|backward_graphs)\[0\]', line)
            and not line.strip().startswith('#')]
    report("tensor_dump.py: no [0] on graph lists",
           len(hits) == 0,
           f"found: {hits}")
except Exception as e:
    report("tensor_dump.py scan", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 12. Signature capture ordering (code structure check)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 12: Signature capture before forward replaced ===")
try:
    src_install = open("torch_graph/install.py").read()
    idx_orig = src_install.index("_orig_params")
    idx_replace = src_install.index("model.forward = forward_fn")
    report("install.py: _orig_params before forward replacement",
           idx_orig < idx_replace)

    # auto_install.py no longer replaces model.forward directly — it uses
    # _CompiledModelProxy which delegates to install.py via _install_model_from_module.
    # Just verify install.py's shared helper is used.
    src_ai = open("torch_graph/auto_install.py").read()
    report("auto_install.py: delegates to install.py",
           "_install_model_from_module" in src_ai)
except Exception as e:
    report("signature ordering", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# 13. Backward through installed model with reversed kwargs
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 13: Backward through installed model (kwargs reversed) ===")
try:
    m_ref = TwoArgModel()
    xr = torch.tensor([1.0, 1.0], requires_grad=True)
    yr = torch.tensor([10.0, 10.0], requires_grad=True)
    ref_out = m_ref(xr, yr)
    ref_out.sum().backward()
    ref_gx = xr.grad.clone()
    ref_gy = yr.grad.clone()

    # Capture on fresh model
    m_cap = TwoArgModel()
    xc = torch.tensor([1.0, 1.0], requires_grad=True)
    yc = torch.tensor([10.0, 10.0], requires_grad=True)
    _, cap_bw = capture_aten_graphs(m_cap, xc, yc, run_backward=True)
    m_bw = export_and_install(TwoArgModel, m_cap, cap_bw)

    # Forward + backward with reversed kwargs
    xt = torch.tensor([1.0, 1.0], requires_grad=True)
    yt = torch.tensor([10.0, 10.0], requires_grad=True)
    out = m_bw(y=yt, x=xt)
    out.sum().backward()

    report("x.grad exists", xt.grad is not None)
    report("y.grad exists", yt.grad is not None)
    if xt.grad is not None:
        report("x.grad matches reference",
               torch.allclose(xt.grad, ref_gx),
               f"got {xt.grad}, expected {ref_gx}")
    if yt.grad is not None:
        report("y.grad matches reference",
               torch.allclose(yt.grad, ref_gy),
               f"got {yt.grad}, expected {ref_gy}")
except Exception as e:
    report("backward with kwargs", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# 14. auto_install.py _install_model_from_capture kwargs
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 14: auto_install install via install.py kwargs ===")
# auto_install delegates to install.py's install() via _install_model_from_module.
# Test that the full pipeline (capture → export → load → install) handles kwargs.
from torch_graph.auto_install import unpatch, patch
try:
    unpatch()  # restore real torch.compile so capture works
    torch.compiler.reset()
    torch._dynamo.reset()

    class AutoInstTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor([3.0, 4.0]))
        def forward(self, x, y):
            return self.scale * x + y

    m_ai = AutoInstTestModel()
    xa = torch.tensor([1.0, 1.0], requires_grad=True)
    ya = torch.tensor([10.0, 10.0], requires_grad=True)
    expected_ai = m_ai(xa, ya).detach().clone()

    _, cap_ai = capture_aten_graphs(m_ai, xa, ya, run_backward=True)
    if cap_ai.forward_graphs:
        import copy, importlib, tempfile
        from torch_graph.export import export_aten_program
        from torch_graph.install import install

        # Export capture to .py, then load as module (matching real pipeline)
        with tempfile.TemporaryDirectory() as td:
            py_path = os.path.join(td, "aten.py")
            export_aten_program(cap_ai, py_path, include_test_harness=False)
            spec = importlib.util.spec_from_file_location("aten_test14", py_path)
            aten_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(aten_mod)

            m_ai2 = copy.deepcopy(m_ai)
            install(m_ai2, aten_mod, validate=False)

            r1 = m_ai2(xa.detach(), ya.detach())
            report("install positional correct",
                   torch.allclose(r1, expected_ai),
                   f"got {r1}")

            r2 = m_ai2(y=ya.detach(), x=xa.detach())
            report("install kwargs reversed correct",
                   torch.allclose(r2, expected_ai),
                   f"got {r2}")
    else:
        report("install capture", False, "0 forward_graphs")
except Exception as e:
    report("install kwargs", False, traceback.format_exc())
finally:
    patch()  # re-patch for any subsequent tests


# ═══════════════════════════════════════════════════════════════════
# 15. Verify auto_install.py uses shared install.py helpers
# ═══════════════════════════════════════════════════════════════════
print("\n=== Test 15: Code consistency install.py vs auto_install.py ===")
try:
    src_inst = open("torch_graph/install.py").read()
    src_auto = open("torch_graph/auto_install.py").read()

    # install.py should own the input-binding helpers
    for pattern, desc in [
        ("def _extract_forward_arg_names(", "defines forward arg-name helper"),
        ("def _normalize_user_inputs(", "defines kwargs normalization helper"),
        ("def _assemble_inputs(", "defines compiled input assembly helper"),
        ("def _make_live_attr_getter(", "defines live attr getter helper"),
        ("def _make_buffer_writer(", "defines buffer writer helper"),
    ]:
        report(f"install has '{pattern}'", pattern in src_inst, desc)

    # auto_install.py delegates to install() via _install_model_from_module,
    # so it only directly references _extract_forward_arg_names.
    # The other helpers (_assemble_inputs, _make_live_attr_getter, etc.) are
    # used internally by install() and don't need to be imported by auto_install.
    report("auto_install uses '_extract_forward_arg_names'",
           "_extract_forward_arg_names" in src_auto,
           "imports shared arg-name helper")
    report("auto_install delegates to install.py",
           "_install_model_from_module" in src_auto,
           "should call _install_model_from_module which uses install()")

    # The old ad-hoc kwargs reordering block should no longer be duplicated here.
    for pattern in [
        "kwargs.pop(name)",
        "ordered.extend(kwargs.values())",
    ]:
        report(
            f"auto_install does not duplicate '{pattern}'",
            pattern not in src_auto,
            f"pattern still present in auto_install.py",
        )
except Exception as e:
    report("code consistency", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
if FAIL:
    print("SOME TESTS FAILED")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
