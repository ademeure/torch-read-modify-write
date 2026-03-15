"""
Zero-modification graph extraction from arbitrary PyTorch repos.

Three levels of usage, from simplest to most flexible:

    # 1. Give us a model and inputs, get aten graphs:
    result = extract_model(model, input_tensor, output_dir="outputs/")

    # 2. Point at a script, we'll intercept every nn.Module forward call:
    results = extract_from_script("nanogpt/train.py", output_dir="outputs/")

    # 3. Point at a module + class name, we'll instantiate and export:
    result = extract_from_module("model.py", model_class_name="GPT")

    # CLI:
    python -m torch_graph.auto train.py --output-dir outputs/
    python -m torch_graph.auto model.py --class GPT --kwargs '{"n_layer": 12}'
"""

from __future__ import annotations

import importlib.util
import logging
import os
import runpy
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from torch_graph.export import AtenCapture, capture_aten_graphs, export_aten_program

logger = logging.getLogger("torch_graph")

# -----------------------------------------------------------------------------
# Result container

@dataclass
class ExtractedModel:
    """What you get back from extract_model / extract_from_script."""
    model_class: type
    model_name: str
    model: nn.Module
    example_inputs: tuple
    output: Any
    capture: AtenCapture
    export_path: str | None = None

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters())
        n_fw, n_bw = len(self.capture.forward_graphs), len(self.capture.backward_graphs)
        return f"ExtractedModel({self.model_name}, {n_params:,} params, {n_fw} fw + {n_bw} bw graphs)"

# -----------------------------------------------------------------------------
# Level 1: extract from a model + inputs (simplest API)

def extract_model(
    model: nn.Module, *args,
    output_dir: str | None = ".",
    name: str | None = None,
    run_backward: bool = True,
    loss_fn: Callable | None = None,
    inline_threshold: int = 500,
    record_real_tensors: bool = False,
    triton: bool = False,
    **kwargs,
) -> ExtractedModel:
    """Extract aten-level graphs from a model with given inputs.

    The inline_threshold controls whether small tensors are embedded directly
    in the exported .py file (convenient for inspection) vs saved to a .pt
    file (safe for reuse across different inputs/steps). Default 500.
    """
    model_name = name or model.__class__.__name__.lower()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Extracting: {model.__class__.__name__} ({model_name}, {n_params:,} params)")

    output, capture = capture_aten_graphs(
        model, *args, run_backward=run_backward, loss_fn=loss_fn,
        record_real_tensors=record_real_tensors, triton=triton, **kwargs,
    )
    logger.info(f"  {capture.summary()}")
    if capture.source_map:
        logger.debug(f"  Source traces: {len(capture.source_map)} ops mapped to source")

    export_path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        export_path = str(Path(output_dir) / f"{model_name}_aten.py")
        export_aten_program(capture, export_path, inline_threshold=inline_threshold)
        logger.info(f"  Exported: {export_path}")

    return ExtractedModel(
        model_class=model.__class__, model_name=model_name, model=model,
        example_inputs=args, output=output, capture=capture, export_path=export_path,
    )

# -----------------------------------------------------------------------------
# Level 2: extract from a script (zero modification)
#
# We monkey-patch nn.Module.__call__ to intercept every forward call during
# script execution. This is invasive but lets us capture models from arbitrary
# training scripts without any code changes.
#
# NOTE: we only capture one instance per model class (by type identity).
# If the same class is instantiated multiple times with different configs,
# we only get the first one.

class _ModelInterceptor:
    """Monkey-patches nn.Module.__call__ to record forward calls and inputs."""

    def __init__(self, max_captures: int = 10, min_params: int = 100):
        self.max_captures = max_captures
        self.min_params = min_params
        self.captured: list[tuple[nn.Module, tuple, dict]] = []
        self._original_call = None
        self._seen_classes: set[type] = set()
        self._active = False

    def install(self):
        self._original_call = nn.Module.__call__
        interceptor = self

        def patched_call(module_self, *args, **kwargs):
            # Only intercept each unseen class once (skips submodules already seen)
            if (interceptor._active
                    and type(module_self) not in interceptor._seen_classes
                    and sum(p.numel() for p in module_self.parameters()) >= interceptor.min_params
                    and len(interceptor.captured) < interceptor.max_captures):
                interceptor._seen_classes.add(type(module_self))
                # Snapshot inputs so later mutations in the script don't affect us
                saved_args = tuple(a.clone().detach() if isinstance(a, torch.Tensor) else a for a in args)
                saved_kwargs = {k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
                interceptor.captured.append((module_self, saved_args, saved_kwargs))
            return interceptor._original_call(module_self, *args, **kwargs)

        nn.Module.__call__ = patched_call
        self._active = True

    def uninstall(self):
        if self._original_call is not None:
            nn.Module.__call__ = self._original_call
            self._original_call = None
        self._active = False


def _try_kernelbench_convention(
    module_globals: dict | None, script_path: str, *,
    output_dir: str | None, run_backward: bool, inline_threshold: int,
    record_real_tensors: bool, triton: bool,
) -> list[ExtractedModel] | None:
    """Detect and use the KernelBench convention if present.

    KernelBench scripts define Model + get_init_inputs() + get_inputs().
    If all three are present, we use them directly instead of the interceptor.
    """
    if module_globals is None:
        return None

    model_cls = module_globals.get("Model")
    get_init_inputs = module_globals.get("get_init_inputs")
    get_inputs = module_globals.get("get_inputs")

    if not (model_cls is not None and isinstance(model_cls, type)
            and issubclass(model_cls, nn.Module)
            and callable(get_init_inputs) and callable(get_inputs)):
        return None

    script_name = Path(script_path).stem
    logger.info(f"Detected KernelBench convention (Model + get_inputs + get_init_inputs)")

    try:
        model = model_cls(*get_init_inputs())
    except Exception as e:
        logger.error(f"  Model construction failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    try:
        inputs = get_inputs()
    except Exception as e:
        logger.error(f"  get_inputs() failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    # Auto-move to CUDA when --triton is requested
    if triton and torch.cuda.is_available():
        model = model.cuda()
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  {model_cls.__name__}: {n_params:,} params")
    logger.info(f"  Inputs: {[tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__ for x in inputs]}")

    try:
        result = extract_model(
            model, *inputs, output_dir=output_dir, name=script_name,
            run_backward=run_backward, inline_threshold=inline_threshold,
            record_real_tensors=record_real_tensors, triton=triton,
        )
    except Exception as e:
        logger.error(f"  Extraction failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    logger.info(f"Done: 1 model exported")
    if result.export_path:
        logger.info(f"  {result.export_path}")
    return [result]


def extract_from_script(
    script_path: str,
    output_dir: str | None = ".",
    max_models: int = 5,
    min_params: int = 100,
    run_backward: bool = True,
    inline_threshold: int = 500,
    record_real_tensors: bool = False,
    triton: bool = False,
) -> list[ExtractedModel]:
    """Run a Python script, intercept all nn.Module forward calls, export each as aten graphs.

    Requires ZERO modifications to the target script — we monkey-patch nn.Module.__call__
    and capture models + inputs as they flow through. After the script finishes (or crashes),
    we process each captured model through the full aten export pipeline.

    The script's exceptions are caught and logged, not re-raised — we may have captured
    useful models before the error occurred.
    """
    script_path = os.path.abspath(script_path)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    script_dir = os.path.dirname(script_path)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Auto-extracting from: {script_path}")
    if output_dir is not None:
        logger.info(f"Output directory: {output_dir}")

    interceptor = _ModelInterceptor(max_captures=max_models, min_params=min_params)
    interceptor.install()

    # Run the target script with its directory on sys.path so its imports work.
    # We save and restore sys.path and cwd to avoid polluting our own process.
    module_globals = None
    try:
        old_path, old_cwd = sys.path.copy(), os.getcwd()
        sys.path.insert(0, script_dir)
        os.chdir(script_dir)
        logger.info(f"Running script: {os.path.basename(script_path)}")
        try:
            module_globals = runpy.run_path(script_path, run_name="__main__")
        except SystemExit:
            pass  # script called sys.exit(), that's fine
        except Exception as e:
            # Script crashed — that's OK, we may have captured models before the error
            logger.warning(f"Script raised {type(e).__name__}: {e}")
    finally:
        sys.path[:] = old_path
        os.chdir(old_cwd)
        interceptor.uninstall()

    # Check for KernelBench convention: Model class + get_inputs() + get_init_inputs()
    # This handles scripts that define the model but don't call it at module level.
    kb_result = _try_kernelbench_convention(
        module_globals, script_path, output_dir=output_dir,
        run_backward=run_backward, inline_threshold=inline_threshold,
        record_real_tensors=record_real_tensors, triton=triton,
    )
    if kb_result is not None:
        return kb_result

    if not interceptor.captured:
        logger.warning(f"No models captured. No nn.Module with >= {min_params} params was called.")
        return []

    logger.info(f"Captured {len(interceptor.captured)} model(s), processing...")

    # Export each captured model
    results = []
    for i, (model, args, kwargs) in enumerate(interceptor.captured):
        cls_name = model.__class__.__name__
        model_name = cls_name.lower() if len(interceptor.captured) == 1 else f"{cls_name.lower()}_{i}"
        try:
            result = extract_model(
                model, *args, output_dir=output_dir, name=model_name,
                run_backward=run_backward, inline_threshold=inline_threshold,
                record_real_tensors=record_real_tensors, triton=triton, **kwargs,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"  Failed to export {cls_name}: {type(e).__name__}: {e}")
            logger.debug(traceback.format_exc())

    logger.info(f"Done: {len(results)}/{len(interceptor.captured)} models exported")
    return results

# -----------------------------------------------------------------------------
# Level 3: extract from a module + class name (targeted)

def extract_from_module(
    module_path: str,
    model_class_name: str | None = None,
    constructor_args: tuple = (),
    constructor_kwargs: dict | None = None,
    example_inputs: tuple | None = None,
    output_dir: str = ".",
    run_backward: bool = True,
    inline_threshold: int = 500,
    triton: bool = False,
) -> ExtractedModel:
    """Import a module, instantiate a model class, and export its aten graphs.

    If model_class_name is None, we try every nn.Module subclass in the module
    and pick the one with the most parameters. This is a best-effort heuristic —
    provide the class name explicitly for reliable results.
    """
    constructor_kwargs = constructor_kwargs or {}

    # Import: support both file paths ("model.py") and dotted paths ("nanogpt.model")
    if module_path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("_target_module", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {module_path}")
        mod = importlib.util.module_from_spec(spec)
        _old_path = sys.path.copy()
        sys.path.insert(0, str(Path(module_path).parent))
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path[:] = _old_path
    else:
        mod = importlib.import_module(module_path)

    # Find or instantiate the model
    if model_class_name:
        model = getattr(mod, model_class_name)(*constructor_args, **constructor_kwargs)
    else:
        model, model_class_name = _find_largest_module(mod, constructor_args, constructor_kwargs)

    # Generate example inputs if not provided
    # NOTE: _infer_example_inputs is a pile of heuristics. It works for common patterns
    # (transformers, CNNs) but will produce garbage for unusual architectures.
    # Always prefer providing explicit inputs.
    if example_inputs is None:
        example_inputs = _infer_example_inputs(model)

    return extract_model(
        model, *example_inputs, output_dir=output_dir,
        name=model_class_name.lower(), run_backward=run_backward,
        inline_threshold=inline_threshold, triton=triton,
    )


def _find_largest_module(mod, constructor_args, constructor_kwargs) -> tuple[nn.Module, str]:
    """Try to instantiate every nn.Module subclass in a module, return the biggest one."""
    candidates = [
        (name, obj) for name in dir(mod)
        if isinstance(obj := getattr(mod, name), type)
        and issubclass(obj, nn.Module) and obj is not nn.Module
    ]
    if not candidates:
        raise ValueError("No nn.Module subclasses found in module")

    # Many candidates will fail to instantiate without the right args — that's expected
    best_name, best_n, best_inst = None, -1, None
    for name, cls in candidates:
        try:
            inst = cls(*constructor_args, **constructor_kwargs)
            n = sum(p.numel() for p in inst.parameters())
            if n > best_n:
                best_name, best_n, best_inst = name, n, inst
        except Exception:
            continue

    if best_inst is None:
        raise ValueError("Could not instantiate any nn.Module subclass (try passing --class and --kwargs)")

    logger.info(f"Auto-selected: {best_name} ({best_n:,} params)")
    return best_inst, best_name


def _infer_example_inputs(model: nn.Module) -> tuple:
    """Best-effort heuristic to guess valid inputs for a model's forward().

    Strategy: inspect the forward() signature, guess tensor shapes from parameter
    names ("idx" -> LongTensor, "mask" -> BoolTensor, etc.) and from the model's
    first layer (Linear -> in_features, Conv2d -> in_channels, Embedding -> vocab_size).

    This is intentionally hacky — it covers ~80% of common model patterns but will
    fail on anything unusual. Users should always prefer providing explicit inputs.
    """
    import inspect
    sig = inspect.signature(model.forward)
    input_params = [p for p in sig.parameters.values() if p.name != "self"]
    if not input_params:
        return (torch.randn(1, 16),)

    inputs = []
    for p in input_params:
        name = p.name.lower()
        if "idx" in name or "token" in name or "input_id" in name:
            # Token indices — find vocab_size from the first Embedding layer
            vocab_size = 100
            for m in model.modules():
                if isinstance(m, nn.Embedding):
                    vocab_size = m.num_embeddings
                    break
            inputs.append(torch.randint(0, vocab_size, (1, 16)))
        elif "mask" in name:
            inputs.append(torch.ones(1, 16, dtype=torch.bool))
        elif "label" in name:
            inputs.append(None)  # usually optional
        else:
            # Default: float tensor shaped to match the first Linear or Conv2d
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    inputs.append(torch.randn(1, m.in_features))
                    break
                elif isinstance(m, nn.Conv2d):
                    inputs.append(torch.randn(1, m.in_channels, 32, 32))
                    break
            else:
                inputs.append(torch.randn(1, 16))

    inputs = [i for i in inputs if i is not None]
    return tuple(inputs) if inputs else (torch.randn(1, 16),)

# -----------------------------------------------------------------------------
# CLI entry point

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract aten-level computation graphs from PyTorch scripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # KernelBench script (auto-detects Model + get_inputs + get_init_inputs):
  python -m torch_graph 7_gpt2_bs32_seq256.py -o outputs/

  # With Triton kernel capture (requires CUDA):
  python -m torch_graph 7_gpt2_bs32_seq256.py -o outputs/ --triton

  # Extract from a training script (auto-discovers models):
  python -m torch_graph train.py -o outputs/

  # Extract a specific class:
  python -m torch_graph model.py --class GPT -o outputs/

  # With constructor kwargs:
  python -m torch_graph model.py --class GPT \\
      --kwargs '{"vocab_size": 50257, "n_layer": 12}'
""",
    )
    parser.add_argument("script", help="Path to Python script or module")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory")
    parser.add_argument("--class", dest="model_class", help="Model class name (skip auto-detection)")
    parser.add_argument("--max-models", type=int, default=5, help="Max models to capture in script mode")
    parser.add_argument("--min-params", type=int, default=100, help="Min params to consider a model")
    parser.add_argument("--no-backward", action="store_true", help="Skip backward pass capture")
    parser.add_argument("--inline-threshold", type=int, default=500, help="Tensor inline threshold")
    parser.add_argument("--kwargs", help="JSON constructor kwargs for --class mode")
    parser.add_argument("--triton", action="store_true", help="Capture Triton kernels (requires CUDA)")

    # Set up logging so CLI users see output by default
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parser.parse_args()

    if args.model_class:
        import json
        kwargs = json.loads(args.kwargs) if args.kwargs else {}
        extract_from_module(
            args.script, model_class_name=args.model_class, constructor_kwargs=kwargs,
            output_dir=args.output_dir, run_backward=not args.no_backward,
            inline_threshold=args.inline_threshold, triton=args.triton,
        )
    else:
        extract_from_script(
            args.script, output_dir=args.output_dir, max_models=args.max_models,
            min_params=args.min_params, run_backward=not args.no_backward,
            inline_threshold=args.inline_threshold, triton=args.triton,
        )


if __name__ == "__main__":
    main()
