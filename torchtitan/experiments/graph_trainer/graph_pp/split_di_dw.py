# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import operator
from dataclasses import dataclass

import torch
import torch.fx as fx
from torch._functorch.partitioners import (
    _extract_fwd_bwd_modules,
    _extract_fwd_bwd_outputs,
    _extract_graph_with_inputs_outputs,
    is_sym_node,
)
from torch.utils._ordered_set import OrderedSet

from torchtitan.experiments.graph_trainer.graph_pp.partition import (
    GraphPPSlotDescriptor,
    _output_descs,
    _placeholder_descs,
)


@dataclass(frozen=True, slots=True)
class GraphPPDiDwSplit:
    """Backward graph split into dI and dW GraphPP callables."""

    bw_di_module: fx.GraphModule
    bw_dw_module: fx.GraphModule
    num_input_grads: int
    bw_di_input_descs: tuple[GraphPPSlotDescriptor, ...]
    bw_di_output_descs: tuple[GraphPPSlotDescriptor, ...]
    bw_dw_input_descs: tuple[GraphPPSlotDescriptor, ...]
    bw_dw_output_descs: tuple[GraphPPSlotDescriptor, ...]


def _rename_placeholder_node(
    gm: fx.GraphModule, node: fx.Node, new_name: str
) -> None:
    if node.op != "placeholder":
        raise ValueError(f"Can only rename placeholder nodes, got {node.op}")
    with gm.graph.inserting_before(node):
        new_node = gm.graph.placeholder(new_name)
    new_node.meta.update(node.meta)
    node.replace_all_uses_with(new_node)
    gm.graph.erase_node(node)


def _remove_recompute_tags(gm: fx.GraphModule) -> None:
    for node in gm.graph.nodes:
        node.meta.pop("recompute", None)


def _reorder_backward_outputs_for_di(
    gm: fx.GraphModule, *, num_param_buffer_grads: int
) -> int:
    outputs = gm.graph.find_nodes(op="output")
    if len(outputs) != 1:
        raise ValueError(f"Expected exactly one output node, found {len(outputs)}")
    output = outputs[0]
    if not isinstance(output.args[0], tuple):
        raise ValueError("Backward graph output must be a tuple")

    output_values = output.args[0]
    if len(output_values) < num_param_buffer_grads:
        raise ValueError(
            f"Backward graph has {len(output_values)} outputs but "
            f"{num_param_buffer_grads} param/buffer grads were requested"
        )

    param_buffer_grads = output_values[:num_param_buffer_grads]
    input_grads = output_values[num_param_buffer_grads:]
    with gm.graph.inserting_after(output):
        new_output = gm.graph.output(tuple(input_grads + param_buffer_grads))
    new_output.meta.update(output.meta)
    gm.graph.erase_node(output)
    gm.graph.lint()
    gm.recompile()
    return len(input_grads)


def _is_fake_tensor_value(node: fx.Node) -> bool:
    return isinstance(node.meta.get("val"), torch._subclasses.FakeTensor)


def _collect_saved_values_for_dw(
    bw_gm: fx.GraphModule,
    di_graph: fx.Graph,
) -> tuple[list[fx.Node], list[fx.Node]]:
    di_node_names = OrderedSet(
        node.name for node in di_graph.nodes if node.op != "output"
    )
    saved_values: list[fx.Node] = []
    saved_sym_nodes: list[fx.Node] = []

    for node in bw_gm.graph.nodes:
        if node.name not in di_node_names:
            continue
        if is_sym_node(node):
            saved_sym_nodes.append(node)
        elif (
            "tensor_meta" not in node.meta
            and node.op == "call_function"
            and not _is_fake_tensor_value(node)
        ):
            users = list(node.users)
            if not all(user.target == operator.getitem for user in users):
                raise ValueError(
                    f"Non-tensor multi-output node {node.name} has unexpected users"
                )
            saved_values.extend(users)
        else:
            dw_users = [user for user in node.users if user.name not in di_node_names]
            if "tensor_meta" in node.meta and all(is_sym_node(user) for user in dw_users):
                saved_sym_nodes.extend(dw_users)
            else:
                saved_values.append(node)

    return (
        list(dict.fromkeys(saved_values).keys()),
        list(dict.fromkeys(saved_sym_nodes).keys()),
    )


def split_di_dw_graph(
    bw_module: fx.GraphModule,
    *,
    num_param_buffer_grads: int,
) -> GraphPPDiDwSplit | None:
    """Split a backward graph into input-gradient and weight-gradient graphs.

    The backward graph is expected to return param/buffer gradients first,
    followed by input gradients.  If there are no input gradients, the caller
    should skip `BACKWARD_INPUT` and run the original full backward graph at the
    `BACKWARD_WEIGHT` action.
    """
    if num_param_buffer_grads < 0:
        raise ValueError(
            f"num_param_buffer_grads must be non-negative, got {num_param_buffer_grads}"
        )

    bw_gm = copy.deepcopy(bw_module)
    for placeholder in list(bw_gm.graph.find_nodes(op="placeholder")):
        if placeholder.name.startswith("tangent"):
            _rename_placeholder_node(
                bw_gm,
                placeholder,
                f"not_tngnt{placeholder.name[len('tangent') :]}",
            )

    _remove_recompute_tags(bw_gm)
    num_input_grads = _reorder_backward_outputs_for_di(
        bw_gm, num_param_buffer_grads=num_param_buffer_grads
    )
    if num_input_grads == 0:
        return None

    placeholders = list(bw_gm.graph.find_nodes(op="placeholder"))
    di_outputs, _, di_output_descs, _ = _extract_fwd_bwd_outputs(
        bw_gm, num_fwd_outputs=num_input_grads
    )
    di_graph = _extract_graph_with_inputs_outputs(
        bw_gm.graph,
        placeholders,
        di_outputs,
        di_output_descs,
        "forward",
        ignore_must_be_in_fw_bw=True,
    )
    saved_values, saved_sym_nodes = _collect_saved_values_for_dw(bw_gm, di_graph)
    bw_di_module, bw_dw_module = _extract_fwd_bwd_modules(
        bw_gm,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_input_grads,
        ignore_must_be_in_fw_bw=True,
        omit_aot_autograd_runtime=True,
    )
    bw_di_module.graph.lint()
    bw_dw_module.graph.lint()
    bw_di_module.recompile()
    bw_dw_module.recompile()

    return GraphPPDiDwSplit(
        bw_di_module=bw_di_module,
        bw_dw_module=bw_dw_module,
        num_input_grads=num_input_grads,
        bw_di_input_descs=_placeholder_descs(bw_di_module),
        bw_di_output_descs=_output_descs(bw_di_module),
        bw_dw_input_descs=_placeholder_descs(bw_dw_module),
        bw_dw_output_descs=_output_descs(bw_dw_module),
    )
