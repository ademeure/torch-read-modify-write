from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from torch_graph.visualizer import GraphVisualizer


class TinyModule(nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1


def test_visualizer_supports_only_html_json():
    gm = symbolic_trace(TinyModule())
    viz = GraphVisualizer(gm)

    assert not hasattr(GraphVisualizer, "to_dot")
    assert not hasattr(GraphVisualizer, "save_dot")
    assert not hasattr(GraphVisualizer, "to_ascii")
    assert not hasattr(GraphVisualizer, "to_mermaid")

    html = viz.to_html()
    assert "FX Graph Viewer" in html

    graph_json = viz.to_json()
    assert "nodes" in graph_json
    assert "code" in graph_json
    add_node = next(node for node in graph_json["nodes"] if node["name"] == "add")
    assert add_node["op"] == "add"
    assert add_node["fx_op"] == "call_function"


def test_visualizer_combined_json_adds_backward_users():
    class ForwardModule(nn.Module):
        def forward(self, x):
            relu = torch.relu(x)
            sig = torch.sigmoid(x)
            out = relu + sig
            return out, relu, sig

    class BackwardModule(nn.Module):
        def forward(self, saved_relu, saved_sig, grad_out):
            return saved_relu * grad_out + saved_sig

    fw = symbolic_trace(ForwardModule())
    bw = symbolic_trace(BackwardModule())
    shape = (2, 3)

    for node in fw.graph.nodes:
        if node.name in {"relu", "sigmoid", "add"}:
            node.meta["val"] = torch.zeros(shape)
    for node in bw.graph.nodes:
        if node.name in {"saved_relu", "saved_sig", "grad_out"}:
            node.meta["val"] = torch.zeros(shape)

    capture = SimpleNamespace(
        backward_graphs=[SimpleNamespace(graph_module=bw)],
        forward_real_output=torch.zeros(shape),
    )
    combined = GraphVisualizer(fw).to_json(backward_source=capture)

    assert "forward" in combined
    assert "backward" in combined

    fw_nodes = {node["name"]: node for node in combined["forward"]["nodes"]}
    bw_nodes = {node["name"]: node for node in combined["backward"]["nodes"]}

    assert "relu" in fw_nodes
    assert fw_nodes["relu"]["backward_users"] == ["saved_relu"]
    assert fw_nodes["sigmoid"]["backward_users"] == ["saved_sig"]
    assert fw_nodes["add"]["backward_users"] == ["grad_out"]
    assert bw_nodes["saved_relu"]["op"] == "placeholder"
    assert bw_nodes["saved_relu"]["fx_op"] == "placeholder"


def test_visualizer_combined_json_adds_backward_grad_targets():
    class ForwardModule(nn.Module):
        def forward(self, x, weight):
            prod = x * weight
            saved = torch.relu(prod)
            return prod, saved

    class BackwardModule(nn.Module):
        def forward(self, saved_prod, grad_out):
            grad_x = grad_out + saved_prod
            grad_weight = grad_out * saved_prod
            return grad_x, grad_weight

    fw = symbolic_trace(ForwardModule())
    bw = symbolic_trace(BackwardModule())
    shape = (2, 3)

    for node in fw.graph.nodes:
        if node.name in {"x", "weight", "mul", "relu"}:
            node.meta["val"] = torch.zeros(shape)
    for node in bw.graph.nodes:
        if node.name in {"saved_prod", "grad_out", "add", "mul"}:
            node.meta["val"] = torch.zeros(shape)

    capture = SimpleNamespace(
        backward_graphs=[SimpleNamespace(graph_module=bw)],
        forward_real_output=torch.zeros(shape),
        primal_names=[None, "weight"],
    )
    combined = GraphVisualizer(fw).to_json(backward_source=capture)

    bw_nodes = {node["name"]: node for node in combined["backward"]["nodes"]}

    assert bw_nodes["add"]["grad_of"] == "x"
    assert bw_nodes["mul"]["grad_of"] == "weight"
    assert bw_nodes["mul"]["grad_of_display"] == "self.weight"


def test_visualizer_combined_json_adds_optimizer_links():
    class ForwardModule(nn.Module):
        def forward(self, x, weight):
            prod = x * weight
            saved = torch.relu(prod)
            return prod, saved

    class BackwardModule(nn.Module):
        def forward(self, saved_prod, grad_out):
            grad_x = grad_out + saved_prod
            grad_weight = grad_out * saved_prod
            return grad_x, grad_weight

    class OptimizerModule(nn.Module):
        def forward(self, opt_param, opt_grad, opt_state):
            new_state = opt_state + opt_grad
            new_param = opt_param - opt_grad
            return new_param, new_state

    fw = symbolic_trace(ForwardModule())
    bw = symbolic_trace(BackwardModule())
    opt = symbolic_trace(OptimizerModule())
    shape = (2, 3)

    for node in fw.graph.nodes:
        if node.name in {"x", "weight", "mul", "relu"}:
            node.meta["val"] = torch.zeros(shape)
    for node in bw.graph.nodes:
        if node.name in {"saved_prod", "grad_out", "add", "mul"}:
            node.meta["val"] = torch.zeros(shape)
    for node in opt.graph.nodes:
        if node.name in {"opt_param", "opt_grad", "opt_state", "sub", "add"}:
            node.meta["val"] = torch.zeros(shape)

    optimizer_capture = SimpleNamespace(
        forward_graphs=[SimpleNamespace(graph_module=opt)],
        optimizer_slot_info=[[
            {"slot_index": 0, "role": "param", "param_name": "weight"},
            {"slot_index": 1, "role": "grad", "param_name": "weight"},
            {"slot_index": 2, "role": "state", "state_key": "exp_avg", "param_name": "weight"},
        ]],
    )
    capture = SimpleNamespace(
        backward_graphs=[SimpleNamespace(graph_module=bw)],
        optimizer_capture=optimizer_capture,
        forward_real_output=torch.zeros(shape),
        primal_names=[None, "weight"],
        param_names=["weight"],
        buffer_names=[],
    )
    combined = GraphVisualizer(fw).to_json(
        backward_source=capture,
        optimizer_source=capture,
    )

    fw_nodes = {node["name"]: node for node in combined["forward"]["nodes"]}
    bw_nodes = {node["name"]: node for node in combined["backward"]["nodes"]}
    opt_nodes = {node["name"]: node for node in combined["optimizer"]["nodes"]}

    assert fw_nodes["weight"]["param_name"] == "weight"
    assert fw_nodes["weight"]["backward_grads"] == ["mul"]
    assert fw_nodes["weight"]["optimizer_users"] == ["opt_param"]
    assert fw_nodes["weight"]["optimizer_grad_users"] == ["opt_grad"]
    assert fw_nodes["weight"]["optimizer_state_users"] == ["opt_state"]

    assert bw_nodes["mul"]["grad_of_param"] == "weight"
    assert bw_nodes["mul"]["optimizer_users"] == ["opt_grad"]
    assert bw_nodes["mul"]["optimizer_param_users"] == ["opt_param"]
    assert bw_nodes["mul"]["optimizer_state_users"] == ["opt_state"]

    assert opt_nodes["opt_param"]["forward_param"] == "weight"
    assert opt_nodes["opt_param"]["backward_grad"] == "mul"
    assert opt_nodes["opt_param"]["optimizer_role"] == "param"
    assert opt_nodes["opt_grad"]["optimizer_role"] == "grad"
    assert opt_nodes["opt_state"]["optimizer_state_key"] == "exp_avg"
