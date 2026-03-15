#!/usr/bin/env python3
"""Run all tensor verification tests across multiple model architectures.

Usage:
    python run_tests.py              # run everything
    python run_tests.py --quick      # skip NanoGPT (faster)
    python run_tests.py --verbose    # show step-by-step detail

Outputs:
    Prints a summary table of results to stdout.
    With --verbose, prints all intermediate step details too.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "test_repo")

import torch
import torch.nn as nn
from torch_graph.tensor_dump import dump_and_compare, verify_against_model

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="Skip larger models")
parser.add_argument("--verbose", "-v", action="store_true", help="Show step-by-step")
args = parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Define test models
# ═══════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class ConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class LNGelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(16)
    def forward(self, x):
        return torch.nn.functional.gelu(self.ln(x))


MODELS = [
    ("Linear",           nn.Linear(4, 3),             (torch.randn(2, 4),)),
    ("MLP",              MLP(),                        (torch.randn(2, 8),)),
    ("Conv+BN+ReLU",     ConvBN().eval(),              (torch.randn(1, 3, 8, 8),)),
    ("LayerNorm+GELU",   LNGelu(),                     (torch.randn(2, 8, 16),)),
    ("MultiheadAttn",    nn.MultiheadAttention(32, 4, batch_first=True),
                                                       (torch.randn(2, 8, 32),) * 3),
]

if not args.quick:
    from model import NanoGPT
    MODELS.append(
        ("NanoGPT",      NanoGPT(),                    (torch.randint(0, 64, (2, 16)),)),
    )


# ═══════════════════════════════════════════════════════════════════════
# Part 1: Determinism verification (run graph twice, compare)
# ═══════════════════════════════════════════════════════════════════════

print("=" * 74)
print(" DETERMINISM VERIFICATION: run graph 2x with same inputs, compare all")
print("=" * 74)
print()

det_results = []
for name, model, inputs in MODELS:
    print(f"── {name} ", "─" * (60 - len(name)))
    results = dump_and_compare(model, *inputs, run_backward=True, verbose=args.verbose)
    if not args.verbose:
        pass  # dump_and_compare already prints the one-liner
    fw = results[0] if results else None
    bw = results[1] if len(results) > 1 else None
    det_results.append((name, fw, bw))
    print()


# ═══════════════════════════════════════════════════════════════════════
# Part 2: End-to-end verification (graph output vs real model output)
# ═══════════════════════════════════════════════════════════════════════

print()
print("=" * 74)
print(" END-TO-END VERIFICATION: graph output vs real model output")
print("=" * 74)
print()

e2e_results = []
for name, model, inputs in MODELS:
    print(f"── {name} ", "─" * (60 - len(name)))
    r = verify_against_model(model, *inputs, run_backward=False, verbose=True)
    fw_comps = r.get("forward", [])
    fw_ok = all(c.matches for c in fw_comps) if fw_comps else False
    e2e_results.append((name, fw_ok, len(fw_comps)))
    print()


# ═══════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════

print()
print("=" * 74)
print(" SUMMARY")
print("=" * 74)
print()
print(f"{'Model':<20} {'FW Det':>10} {'BW Det':>10} {'E2E FW':>10}")
print("-" * 54)

all_pass = True
for (name, fw, bw), (_, e2e_ok, _) in zip(det_results, e2e_results):
    fw_str = fw.oneline().split(":")[1].strip() if fw else "n/a"
    bw_str = bw.oneline().split(":")[1].strip() if bw else "n/a"
    e2e_str = "pass" if e2e_ok else "FAIL"
    print(f"{name:<20} {fw_str:>10} {bw_str:>10} {e2e_str:>10}")
    if fw and not fw.all_match:
        all_pass = False
    if bw and not bw.all_match:
        all_pass = False
    if not e2e_ok:
        all_pass = False

# ═══════════════════════════════════════════════════════════════════════
# DYNAMIC SHAPES VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

print()
print("=" * 74)
print(" DYNAMIC SHAPES VERIFICATION: capture with dynamic=True, verify determinism")
print("=" * 74)
print()

print("── MLP (dynamic=True) ", "─" * 50)
dynamic_results = dump_and_compare(
    MLP(), torch.randn(2, 8),
    run_backward=True,
    dynamic=True,
    verbose=args.verbose,
)
dynamic_fw = dynamic_results[0] if dynamic_results else None
dynamic_bw = dynamic_results[1] if len(dynamic_results) > 1 else None

if dynamic_fw and dynamic_bw:
    if dynamic_fw.all_match and dynamic_bw.all_match:
        print("  Dynamic shapes: FW and BW determinism PASS")
    else:
        print("  Dynamic shapes: FAIL")
        all_pass = False
else:
    print("  Dynamic shapes: no results")
    all_pass = False

# ═══════════════════════════════════════════════════════════════════════
# OP DUMP VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

print()
print("=" * 74)
print(" OP DUMP VERIFICATION: grouping strategies, H5 structure, replay scripts")
print("=" * 74)
print()

import tempfile, os
from torch_graph.export import capture_aten_graphs
from torch_graph.op_dump import dump_grouped_tensors, dump_model_ops

def _test_dump(label, test_fn):
    """Run a test and return (label, pass/fail, detail)."""
    try:
        test_fn()
        print(f"  {label}: PASS")
        return (label, True, "")
    except Exception as e:
        print(f"  {label}: FAIL — {e}")
        return (label, False, str(e))

dump_results = []

# Capture once for reuse
_mlp = MLP()
_mlp_x = torch.randn(2, 8)
_, _mlp_cap = capture_aten_graphs(_mlp, _mlp_x, record_real_tensors=True)

def test_groupby_line():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        groups = dump_grouped_tensors(_mlp_cap, path, group_by="line")
        assert len(groups) > 0, "No groups produced"
        assert all(g.name for g in groups), "Group has empty name"
        assert all(len(g.ops) > 0 for g in groups), "Group has no ops"
        import h5py
        with h5py.File(path, "r") as hf:
            assert "groups" in hf, "Missing /groups"
            assert "tensors" in hf, "Missing /tensors"
            assert len(hf["tensors"]) > 0, "No tensors stored"
            for gname in hf["groups"]:
                g = hf["groups"][gname]
                assert "num_ops" in g.attrs, f"{gname} missing num_ops attr"
    finally:
        os.unlink(path)

dump_results.append(_test_dump("group_by=line (H5 structure)", test_groupby_line))

def test_groupby_module():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        groups = dump_grouped_tensors(_mlp_cap, path, group_by="module")
        assert len(groups) > 0, "No groups produced"
        has_fc1 = any("fc1" in g.name for g in groups)
        has_fc2 = any("fc2" in g.name for g in groups)
        assert has_fc1 and has_fc2, f"Expected fc1/fc2 groups, got: {[g.name for g in groups]}"
    finally:
        os.unlink(path)

dump_results.append(_test_dump("group_by=module (correct groups)", test_groupby_module))

def test_groupby_multi():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        groups = dump_grouped_tensors(_mlp_cap, path, group_by=["line", "module"])
        import h5py
        with h5py.File(path, "r") as hf:
            sections = list(hf["_meta"].attrs["sections"])
            assert "line" in sections and "module" in sections, \
                f"Expected line+module sections, got: {sections}"
            assert "line" in hf and "module" in hf, \
                f"Missing section groups, got: {list(hf.keys())}"
    finally:
        os.unlink(path)

dump_results.append(_test_dump("group_by=[line,module] (multi-section)", test_groupby_multi))

def test_hide_views():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path_full = f.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path_hidden = f.name
    try:
        g_full = dump_grouped_tensors(_mlp_cap, path_full, group_by="line")
        g_hidden = dump_grouped_tensors(_mlp_cap, path_hidden, group_by="line", hide_views=True)
        total_full = sum(len(g.ops) for g in g_full)
        total_hidden = sum(len(g.ops) for g in g_hidden)
        assert total_hidden <= total_full, \
            f"hide_views should reduce ops ({total_hidden} > {total_full})"
    finally:
        os.unlink(path_full)
        os.unlink(path_hidden)

dump_results.append(_test_dump("hide_views reduces visible ops", test_hide_views))

def test_replay_scripts():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        groups = dump_grouped_tensors(_mlp_cap, path, group_by="line")
        has_group_script = any(g.replay_script for g in groups)
        has_op_script = any(op.replay_script for g in groups for op in g.ops)
        assert has_group_script, "No group-level replay scripts generated"
        assert has_op_script, "No op-level replay scripts generated"
        import h5py
        with h5py.File(path, "r") as hf:
            # Scripts are in consolidated /scripts/replay dataset
            assert "scripts" in hf, "Missing /scripts group"
            assert "replay" in hf["scripts"], "Missing /scripts/replay dataset"
            scripts_ds = hf["scripts"]["replay"]
            assert len(scripts_ds) > 0, "/scripts/replay is empty"
            text = scripts_ds[0].decode("utf-8")
            assert "outputs[" in text, "/scripts/replay[0] has no outputs assignments"
            assert "import" not in text, "/scripts/replay[0] should not contain imports"
            # Groups should have _script_idx attrs pointing into the store
            found_idx = False
            for gname in hf["groups"]:
                g = hf["groups"][gname]
                if "_script_idx" in g.attrs:
                    found_idx = True
                    break
            assert found_idx, "No _script_idx attr found in any group"
    finally:
        os.unlink(path)

dump_results.append(_test_dump("replay scripts (generated + readable)", test_replay_scripts))

def test_replay_executes():
    """Actually execute a replay script and verify outputs are produced."""
    groups = dump_grouped_tensors(
        _mlp_cap, "/dev/null", group_by="op", replay_scripts=True)
    all_tensors = _mlp_cap.forward_intermediates or {}
    executed = 0
    for g in groups:
        if not g.replay_script or not g.external_inputs:
            continue
        inputs = {}
        for iname in g.external_inputs:
            t = all_tensors.get(iname)
            if t is not None:
                inputs[iname] = t
        if len(inputs) != len(g.external_inputs):
            continue
        import operator as _op
        outputs = {}
        ns = {"inputs": inputs, "outputs": outputs, "torch": torch, "operator": _op}
        exec(g.replay_script, ns)
        if outputs:
            executed += 1
    assert executed > 0, "No replay scripts executed successfully"

dump_results.append(_test_dump("replay script execution", test_replay_executes))

def test_pt_format():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        groups = dump_grouped_tensors(_mlp_cap, path, group_by="line")
        data = torch.load(path, weights_only=False)
        assert "tensors" in data, "Missing tensors in .pt"
        assert "sections" in data, "Missing sections in .pt"
        assert len(data["tensors"]) > 0, "No tensors in .pt"
    finally:
        os.unlink(path)

dump_results.append(_test_dump("PT format output", test_pt_format))

def test_dump_model_ops_convenience():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        groups = dump_model_ops(MLP(), torch.randn(2, 8), path=path, group_by="module")
        assert len(groups) > 0, "dump_model_ops returned no groups"
    finally:
        os.unlink(path)

dump_results.append(_test_dump("dump_model_ops convenience API", test_dump_model_ops_convenience))

# NanoGPT-specific tests (richer model)
if not args.quick:
    from model import NanoGPT as _NanoGPT
    _gpt = _NanoGPT()
    _gpt_x = torch.randint(0, 50, (1, 16))
    _, _gpt_cap = capture_aten_graphs(_gpt, _gpt_x, record_real_tensors=True)

    def test_human_aliases():
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            dump_grouped_tensors(_gpt_cap, path, group_by="line")
            import h5py
            with h5py.File(path, "r") as hf:
                all_names = []
                def collect(name, obj):
                    if isinstance(obj, h5py.Dataset) and "tensors" not in name:
                        all_names.append(name.split("/")[-1])
                hf.visititems(collect)
                has_alias = any("___q" in n or "___k" in n or "___v" in n
                                or "___x" in n or "___logits" in n
                                for n in all_names)
                assert has_alias, \
                    f"No human-readable aliases found in tensor links"
        finally:
            os.unlink(path)

    dump_results.append(_test_dump("NanoGPT human-readable aliases", test_human_aliases))

    def test_backward_dump():
        _, cap_bwd = capture_aten_graphs(
            _gpt, _gpt_x, record_real_tensors=True,
            run_backward=True, loss_fn=lambda o: o.sum())
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            dump_grouped_tensors(cap_bwd, path, group_by="module", which="both")
            import h5py
            with h5py.File(path, "r") as hf:
                assert "forward" in hf, "Missing /forward section"
                assert "backward" in hf, "Missing /backward section"
                assert len(hf["forward"]) > 0, "Empty forward section"
                assert len(hf["backward"]) > 0, "Empty backward section"
        finally:
            os.unlink(path)

    dump_results.append(_test_dump("NanoGPT forward+backward dump", test_backward_dump))

def test_scripts_dir():
    import shutil
    scripts_dir = tempfile.mkdtemp(prefix="scripts_")
    try:
        groups = dump_grouped_tensors(
            _mlp_cap, "/dev/null", group_by="line",
            replay_scripts=True, scripts_dir=scripts_dir)
        py_files = list(Path(scripts_dir).glob("*.py"))
        assert len(py_files) > 0, "No .py scripts written"
        for py_file in py_files:
            text = py_file.read_text()
            assert "import torch" in text, f"{py_file.name} missing import torch"
            assert "outputs = {}" in text, f"{py_file.name} missing outputs init"
            assert 'outputs["' in text, f"{py_file.name} has no assignments"
    finally:
        shutil.rmtree(scripts_dir)

dump_results.append(_test_dump("scripts_dir standalone .py files", test_scripts_dir))

def test_scripts_dir_multisection():
    import shutil
    scripts_dir = tempfile.mkdtemp(prefix="scripts_multi_")
    try:
        groups = dump_grouped_tensors(
            _mlp_cap, "/dev/null", group_by=["line", "module"],
            replay_scripts=True, scripts_dir=scripts_dir)
        subdirs = [p for p in Path(scripts_dir).iterdir() if p.is_dir()]
        assert len(subdirs) >= 2, f"Expected >=2 subdirs for multi-section, got {len(subdirs)}"
        for sd in subdirs:
            py_files = list(sd.glob("*.py"))
            assert len(py_files) > 0, f"No scripts in {sd.name}/"
    finally:
        shutil.rmtree(scripts_dir)

dump_results.append(_test_dump("scripts_dir multi-section subdirs", test_scripts_dir_multisection))

dump_pass = all(ok for _, ok, _ in dump_results)
if not dump_pass:
    all_pass = False

print()
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED - see details above")
    sys.exit(1)
