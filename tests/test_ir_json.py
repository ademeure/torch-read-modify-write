from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from torch_graph.export import export_graph_to_python
from torch_graph.ir_json import capture_to_ir_json, graph_to_ir_json, ir_graph_to_python


class TinyModule(nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1


def test_ir_json_python_matches_export_graph_to_python():
    gm = symbolic_trace(TinyModule())

    ir_graph = graph_to_ir_json(gm)
    ir_code = ir_graph_to_python(ir_graph)
    export_code = export_graph_to_python(gm, annotate_sources=False)

    assert ir_graph["schema"] == "torch_graph.ir_json/v1"
    assert ir_code == export_code
    assert all("line" not in node for node in ir_graph["nodes"])


def test_ir_json_preserves_structured_args_and_kwargs():
    class SliceModule(nn.Module):
        def forward(self, x):
            y = torch.ops.aten.slice.Tensor(x, 1, 0, 2)
            return torch.ops.aten.clamp.default(y, min=-1.0, max=1.0)

    gm = symbolic_trace(SliceModule())
    ir_graph = graph_to_ir_json(gm)
    nodes = {node["name"]: node for node in ir_graph["nodes"]}

    slice_node = next(node for node in ir_graph["nodes"] if node["target"] == "slice.Tensor")
    clamp_node = next(node for node in ir_graph["nodes"] if node["target"] == "clamp.default")

    assert slice_node["args"][0] == {"node": "x"}
    assert [arg.get("kind") for arg in slice_node["args"][1:]] == ["int", "int", "int"]
    assert clamp_node["kwargs"]["min"] == {"kind": "float", "value": -1.0}
    assert clamp_node["kwargs"]["max"] == {"kind": "float", "value": 1.0}
    assert "line" not in slice_node
    assert "slice_tensor" in nodes
    assert "clamp_default" in nodes


def test_ir_json_preserves_multi_output_returns():
    class MultiOutputModule(nn.Module):
        def forward(self, x):
            relu = torch.relu(x)
            add = relu + 1
            return relu, add

    gm = symbolic_trace(MultiOutputModule())
    ir_graph = graph_to_ir_json(gm)
    code = ir_graph_to_python(ir_graph)

    assert ir_graph["returns"] == [
        {"node": "relu"},
        {"node": "add"},
    ]
    assert "return (relu, add,)" in code


def test_ir_json_preserves_get_attr_tensor_literals():
    class ParamModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        def forward(self, x):
            return x + self.weight

    module = ParamModule()
    gm = symbolic_trace(module)
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            node.meta["val"] = module.weight.detach()

    ir_graph = graph_to_ir_json(gm)
    get_attr_node = next(node for node in ir_graph["nodes"] if node["fx_op"] == "get_attr")

    assert get_attr_node["meta"]["shape"] == [2, 2]
    assert "torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)" in get_attr_node["python"]


def test_capture_to_ir_json_keeps_forward_backward_and_optimizer_sections():
    class ForwardModule(nn.Module):
        def forward(self, x, weight):
            return x * weight

    class BackwardModule(nn.Module):
        def forward(self, grad_out):
            return grad_out

    class OptimizerModule(nn.Module):
        def forward(self, opt_param, opt_grad):
            return opt_param - opt_grad

    fw = symbolic_trace(ForwardModule())
    bw = symbolic_trace(BackwardModule())
    opt = symbolic_trace(OptimizerModule())
    capture = SimpleNamespace(
        forward_graphs=[SimpleNamespace(graph_module=fw)],
        backward_graphs=[SimpleNamespace(graph_module=bw)],
        optimizer_capture=SimpleNamespace(
            forward_graphs=[SimpleNamespace(graph_module=opt)],
            optimizer_slot_info=[[{"slot_index": 0, "role": "param", "param_name": "weight"}]],
        ),
        primal_names=[None, "weight"],
        param_names=["weight"],
        buffer_names=[],
        param_shapes=[],
        buffer_shapes=[],
    )

    ir_bundle = capture_to_ir_json(capture)

    assert ir_bundle["schema"] == "torch_graph.ir_json_bundle/v1"
    assert ir_bundle["forward"]["placeholders"][1]["display_name"] == "self.weight"
    assert ir_bundle["backward"]["fn_name"] == "backward"
    assert ir_bundle["optimizer"]["fn_name"] == "optimizer_step"
    assert ir_bundle["optimizer"]["slot_info"] == [
        {"slot_index": 0, "role": "param", "param_name": "weight"}
    ]


def test_ir_json_uses_original_source_line_when_source_map_is_available():
    class SourceModule(nn.Module):
        def forward(self, x):
            y = torch.relu(x)
            return y + 1

    gm = symbolic_trace(SourceModule())
    capture = SimpleNamespace(
        primal_names=[],
        param_names=[],
        buffer_names=[],
        param_shapes=[],
        buffer_shapes=[],
        source_map={
            "forward": SimpleNamespace(
                file="/tmp/example.py",
                line=12,
                code="y = torch.relu(x)",
                fn_name="forward",
                module_path="self",
                module_type="SourceModule",
            )
        },
    )

    for node in gm.graph.nodes:
        if node.name == "relu":
            node.meta["source_fn_stack"] = [("forward", 12)]
            node.meta["nn_module_stack"] = {"self": ("self", SourceModule)}

    ir_graph = graph_to_ir_json(gm, capture=capture, source_map=capture.source_map)
    relu_node = next(node for node in ir_graph["nodes"] if node["name"] == "relu")

    assert "line" not in relu_node
    assert relu_node["source"]["code"] == "y = torch.relu(x)"
    assert relu_node["source"]["file"] == "/tmp/example.py"
    assert relu_node["source"]["line_number"] == 12


def test_capture_to_ir_json_annotates_cross_graph_links():
    """annotate=True adds backward_users, backward_grads, grad_of to IR nodes."""
    from torch_graph.export import capture_aten_graphs

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 4)

        def forward(self, x):
            return self.linear(x).sum()

    model = M()
    x = torch.randn(2, 8)
    _, cap = capture_aten_graphs(model, x, run_backward=True, loss_fn=lambda out: out)

    ir = capture_to_ir_json(cap, annotate=True)
    fw_all = ir["forward"]["placeholders"] + ir["forward"]["nodes"]
    bw_all = ir["backward"]["placeholders"] + ir["backward"]["nodes"]

    # Forward placeholders should have param_name for weight and bias
    param_names = {n["param_name"] for n in fw_all if "param_name" in n}
    assert "linear.weight" in param_names
    assert "linear.bias" in param_names

    # Forward nodes should have backward_users (saved tensors consumed by bw)
    assert any("backward_users" in n for n in fw_all)

    # Forward placeholders should have backward_grads (which bw node is their grad)
    assert any("backward_grads" in n for n in fw_all)

    # Backward nodes should have grad_of linking back to forward placeholder
    grad_of_nodes = [n for n in bw_all if "grad_of" in n]
    assert len(grad_of_nodes) > 0
    assert any(n.get("grad_of_param") == "linear.weight" for n in grad_of_nodes)

    # annotate=False should have none of these
    ir_no = capture_to_ir_json(cap, annotate=False)
    fw_no = ir_no["forward"]["placeholders"] + ir_no["forward"]["nodes"]
    bw_no = ir_no["backward"]["placeholders"] + ir_no["backward"]["nodes"]
    assert not any("backward_users" in n for n in fw_no)
    assert not any("backward_grads" in n for n in fw_no)
    assert not any("grad_of" in n for n in bw_no)
    assert not any("param_name" in n for n in fw_no)


def test_ir_json_renders_memory_format_kwargs():
    class CloneModule(nn.Module):
        def forward(self, x):
            return torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)

    gm = symbolic_trace(CloneModule())
    ir_graph = graph_to_ir_json(gm)
    code = ir_graph_to_python(ir_graph)

    clone_node = next(node for node in ir_graph["nodes"] if node["target"] == "clone.default")
    assert clone_node["kwargs"]["memory_format"] == {
        "kind": "torch_memory_format",
        "value": "torch.contiguous_format",
    }
    assert "memory_format=torch.contiguous_format" in code
