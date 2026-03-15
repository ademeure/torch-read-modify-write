"""torch_graph - Extract, visualize, and edit PyTorch TorchDynamo/FX compilation graphs."""

import torch_graph._workarounds  # noqa: F401 — apply PyTorch workarounds on import

from torch_graph.capture import GraphCapture, capture_graphs
from torch_graph.inspector import GraphInspector
from torch_graph.editor import GraphEditor
from torch_graph.visualizer import GraphVisualizer
from torch_graph.export import (
    capture_aten_graphs,
    capture_optimizer_aten,
    export_aten_program,
    export_graph_to_python,
    save_step_data,
    trace_tensors_from_graph,
    extract_subgraph,
    list_ops,
    SourceTrace,
)
from torch_graph.triton import (
    capture_inductor_debug,
    capture_triton_kernels,
    enrich_capture_with_inductor_debug,
    attach_inductor_debug,
    save_triton_kernels,
    build_kernel_node_map,
    TritonKernel,
    TritonCapture,
    KernelMapping,
)
from torch_graph.tensor_dump import (
    dump_and_compare,
    dump_model_tensors,
    trace_all_intermediates,
    compare_tensors,
    compute_tensor_stats,
    verify_against_model,
    DumpResult,
    TensorComparison,
)
from torch_graph.op_dump import dump_grouped_tensors, dump_model_ops
from torch_graph.extract import extract_training_step, extract_function, load_recipe
from torch_graph.ir_json import capture_to_ir_json, graph_to_ir_json, ir_graph_to_python, save_ir_json
from torch_graph.explain import explain, ExplainResult

__all__ = [
    "GraphCapture",
    "capture_graphs",
    "GraphInspector",
    "GraphEditor",
    "GraphVisualizer",
    "capture_aten_graphs",
    "export_aten_program",
    "export_graph_to_python",
    "save_step_data",
    "trace_tensors_from_graph",
    "extract_subgraph",
    "list_ops",
    "SourceTrace",
    "capture_inductor_debug",
    "capture_triton_kernels",
    "enrich_capture_with_inductor_debug",
    "attach_inductor_debug",
    "save_triton_kernels",
    "build_kernel_node_map",
    "TritonKernel",
    "TritonCapture",
    "KernelMapping",
    "dump_and_compare",
    "dump_model_tensors",
    "trace_all_intermediates",
    "compare_tensors",
    "compute_tensor_stats",
    "verify_against_model",
    "DumpResult",
    "TensorComparison",
    "dump_grouped_tensors",
    "dump_model_ops",
    "capture_optimizer_aten",
    "extract_training_step",
    "extract_function",
    "load_recipe",
    "capture_to_ir_json",
    "graph_to_ir_json",
    "ir_graph_to_python",
    "save_ir_json",
    "explain",
    "ExplainResult",
]
