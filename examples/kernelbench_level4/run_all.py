"""Test all KernelBench level4 scripts with torch-graph."""
import gc
import importlib.util
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def configure_runtime():
    warnings.filterwarnings(
        "ignore",
        message=r"Dynamo does not know how to trace the builtin `numpy\.random\.mtrand\.seed\.`.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"To copy construct from a tensor, it is recommended.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"torch\.compile produced \d+ graph fragments \(graph breaks\)\. Using multi-fragment export\.",
    )
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    try:
        torch._logging.set_logs(graph_breaks=False, recompiles=False)
    except Exception:
        pass
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity_error()
    except Exception:
        pass


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for KernelBench tests. Run this script outside the sandbox on a GPU-enabled host."
        )
    return torch.device("cuda")


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    return value


configure_runtime()

def load_kb_module(path):
    """Load a KernelBench script as a module."""
    spec = importlib.util.spec_from_file_location("kb", path)
    mod = importlib.util.module_from_spec(spec)
    old_path = sys.path.copy()
    sys.path.insert(0, str(Path(path).parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = old_path
    return mod


def apply_batch_override(mod, name):
    raw_override = os.environ.get("TORCH_GRAPH_BATCH_OVERRIDE")
    if raw_override is None or raw_override == "":
        return None
    try:
        override = int(raw_override)
    except ValueError as e:
        raise RuntimeError("TORCH_GRAPH_BATCH_OVERRIDE must be a positive integer") from e
    if override <= 0:
        raise RuntimeError("TORCH_GRAPH_BATCH_OVERRIDE must be a positive integer")
    if hasattr(mod, "batch_size"):
        mod.batch_size = override
        return override
    return None

def test_script(path):
    """Test a single KernelBench script. Returns (success, info_dict)."""
    name = Path(path).stem
    info = {
        "name": name,
        "error": None,
        "fw_nodes": 0,
        "bw_nodes": 0,
        "params": 0,
        "time": 0,
        "batch_override": None,
    }
    mod = model = inputs = output = capture = None

    try:
        device = require_cuda()
        mod = load_kb_module(path)
        info["batch_override"] = apply_batch_override(mod, name)
        model = mod.Model(*mod.get_init_inputs()).eval().to(device)
        inputs = move_to_device(mod.get_inputs(), device)
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        info["params"] = sum(p.numel() for p in model.parameters())

        from torch_graph.export import capture_aten_graphs

        torch.cuda.synchronize(device)
        t0 = time.time()
        output, capture = capture_aten_graphs(model, *inputs, run_backward=False)
        torch.cuda.synchronize(device)
        info["time"] = time.time() - t0

        info["fw_nodes"] = sum(
            len(list(g.graph_module.graph.nodes)) for g in capture.forward_graphs
        )
        info["bw_nodes"] = sum(
            len(list(g.graph_module.graph.nodes)) for g in capture.backward_graphs
        )
        info["fw_graphs"] = len(capture.forward_graphs)
        info["source_traces"] = len(capture.source_map) if capture.source_map else 0

        return True, info

    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["time"] = 0
        return False, info

    finally:
        del mod, model, inputs, output, capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    scripts = sorted(p for p in script_dir.glob("*.py") if p.stem and p.stem[0].isdigit())

    # Group by model to download weights once
    from collections import defaultdict
    by_model = defaultdict(list)
    for s in scripts:
        # Quick parse to get model_name
        text = s.read_text()
        for line in text.splitlines():
            if line.startswith("model_name = "):
                model_id = line.split("=", 1)[1].strip().strip('"').strip("'")
                by_model[model_id].append(s)
                break

    results = []
    total = len(scripts)
    done = 0

    for model_id, model_scripts in by_model.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_id} ({len(model_scripts)} scripts)")
        print(f"{'='*70}")

        for s in model_scripts:
            done += 1
            name = s.stem
            print(f"\n[{done}/{total}] {name} ... ", end="", flush=True)

            ok, info = test_script(str(s))
            results.append((ok, info))

            if ok:
                print(f"OK  ({info['params']:,} params, {info['fw_nodes']} nodes, "
                      f"{info['source_traces']} traces, {info['time']:.1f}s)")
            else:
                print(f"FAIL: {info['error']}")

        # Free model weights between model groups
        gc.collect()

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    passed = sum(1 for ok, _ in results if ok)
    failed = sum(1 for ok, _ in results if not ok)
    print(f"\nPassed: {passed}/{total}")
    if failed:
        print(f"\nFailed ({failed}):")
        for ok, info in results:
            if not ok:
                print(f"  {info['name']}: {info['error']}")

    print(f"\nPassed ({passed}):")
    for ok, info in results:
        if ok:
            print(f"  {info['name']}: {info['params']:,} params, "
                  f"{info['fw_nodes']} fw nodes, {info['source_traces']} traces")
