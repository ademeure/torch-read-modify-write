"""Test a single KernelBench script. Usage: python3 examples/kernelbench_level4/run_one.py <script.py>"""
import gc
import importlib.util
import logging
import os
import sys
import time
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


configure_runtime()

def main():
    path = sys.argv[1]
    name = Path(path).stem
    device = require_cuda()

    # Load module
    spec = importlib.util.spec_from_file_location("kb", path)
    mod = importlib.util.module_from_spec(spec)
    old_path = sys.path.copy()
    sys.path.insert(0, str(Path(path).parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = old_path

    batch_override = apply_batch_override(mod, name)

    # Construct model
    t0 = time.time()
    model = mod.Model(*mod.get_init_inputs()).eval().to(device)
    t_init = time.time() - t0
    params = sum(p.numel() for p in model.parameters())

    # Get inputs
    inputs = move_to_device(mod.get_inputs(), device)
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    input_shapes = [tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__ for x in inputs]

    # Capture
    from torch_graph.export import capture_aten_graphs
    torch.cuda.synchronize(device)
    t1 = time.time()
    output, capture = capture_aten_graphs(model, *inputs, run_backward=False)
    torch.cuda.synchronize(device)
    t_capture = time.time() - t1

    fw_nodes = sum(len(list(g.graph_module.graph.nodes)) for g in capture.forward_graphs)
    fw_graphs = len(capture.forward_graphs)
    traces = len(capture.source_map) if capture.source_map else 0

    print(f"RESULT: OK | {name}")
    print(f"  params: {params:,}")
    print(f"  input_shapes: {input_shapes}")
    if batch_override is not None:
        print(f"  batch_override: {batch_override}")
    print(f"  fw_graphs: {fw_graphs}")
    print(f"  fw_nodes: {fw_nodes}")
    print(f"  source_traces: {traces}")
    print(f"  init_time: {t_init:.1f}s")
    print(f"  capture_time: {t_capture:.1f}s")
    del mod, model, inputs, output, capture
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        name = Path(sys.argv[1]).stem
        print(f"RESULT: FAIL | {name}")
        print(f"  error: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
