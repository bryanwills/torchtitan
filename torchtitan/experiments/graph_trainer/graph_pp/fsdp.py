# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import operator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import torch
import torch.fx as fx
import torch.fx.node
import torch.utils._pytree as pytree
from torch._functorch._aot_autograd.descriptors import AOTOutput
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs

from torchtitan.experiments.graph_trainer.graph_pp.partition import (
    GraphPPSlotDescriptor,
    _output_descs,
    _placeholder_descs,
)


@dataclasses.dataclass(frozen=True)
class FSDPUnshardOutput(AOTOutput):
    pass


@dataclasses.dataclass(frozen=True)
class FSDPReduceGradInput(AOTOutput):
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class GraphPPFSDPForwardSplit:
    fw_fsdp_module: fx.GraphModule | None
    unshard_module: fx.GraphModule | None
    fw_no_fsdp_module: fx.GraphModule
    unshard_output_descs: tuple[GraphPPSlotDescriptor, ...]
    fw_no_fsdp_input_descs: tuple[GraphPPSlotDescriptor, ...]
    fw_no_fsdp_output_descs: tuple[GraphPPSlotDescriptor, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class GraphPPFSDPBackwardSplit:
    bw_no_fsdp_module: fx.GraphModule
    reduce_grad_module: fx.GraphModule | None
    bw_no_fsdp_output_descs: tuple[GraphPPSlotDescriptor, ...]
    reduce_grad_input_descs: tuple[GraphPPSlotDescriptor, ...]


@contextmanager
def _exclude_from_fx_side_effectful(exclude_vals: set[Any]):
    original_val = torch.fx.node._side_effectful_functions.copy()
    try:
        torch.fx.node._side_effectful_functions -= exclude_vals
        yield
    finally:
        torch.fx.node._side_effectful_functions.clear()
        torch.fx.node._side_effectful_functions.update(original_val)


def _is_wait_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def _is_all_gather_into_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
    )


def _is_reduce_scatter_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
    )


def _find_last_all_gather_in_chain(start_node: fx.Node) -> fx.Node | None:
    node = start_node
    last_ag_node = None
    while True:
        if len(node.users) != 1:
            break
        user = next(iter(node.users))
        if len(user.all_input_nodes) > 1:
            break
        node = user
        if _is_all_gather_into_tensor(node):
            last_ag_node = node
    return last_ag_node


def _find_last_user_in_wait_chain(wait_node: fx.Node) -> fx.Node:
    node = wait_node
    while True:
        if len(node.users) != 1:
            if node.op == "call_function" and node.target == torch.ops.aten.split.Tensor:
                if all(
                    user.op == "call_function"
                    and user.target == operator.getitem
                    and len(user.users) == 1
                    for user in node.users
                ):
                    getitem_users = [next(iter(user.users)) for user in node.users]
                    potential_cat = getitem_users[0]
                    if all(user == potential_cat for user in getitem_users) and (
                        potential_cat.op == "call_function"
                        and potential_cat.target == torch.ops.aten.cat.default
                    ):
                        node = potential_cat
                        continue
            break
        user = next(iter(node.users))
        if len(user.all_input_nodes) > 1:
            break
        node = user
    return node


def _find_last_non_view_node_in_chain(node: fx.Node) -> fx.Node:
    result = node
    while hasattr(result.target, "is_view") and result.target.is_view:
        if len(result.all_input_nodes) != 1:
            raise ValueError(f"View node {result.name} should have exactly one input")
        result = result.all_input_nodes[0]
    return result


def _make_graph_module_like(gm: fx.GraphModule, graph: fx.Graph) -> fx.GraphModule:
    return torch.fx._lazy_graph_module._make_graph_module(gm, graph)


def _graph_outputs(graph: fx.Graph) -> tuple[Any, ...]:
    outputs = graph.find_nodes(op="output")
    if len(outputs) != 1:
        raise ValueError(f"Expected one output node, found {len(outputs)}")
    return tuple(pytree.arg_tree_leaves(*(node.args for node in outputs)))


def split_forward_fsdp_collectives(
    fw_module: fx.GraphModule,
    *,
    num_params: int,
) -> GraphPPFSDPForwardSplit:
    """Split forward FSDP all-gather chains from a forward graph."""
    if num_params < 0:
        raise ValueError(f"num_params must be non-negative, got {num_params}")

    graph = deepcopy(fw_module.graph)
    placeholders = graph.find_nodes(op="placeholder")
    param_inputs = placeholders[:num_params]
    remaining_inputs = placeholders[num_params:]
    unshard_outputs: list[Any] = []
    found_collective = False

    for param_input in param_inputs:
        last_ag = _find_last_all_gather_in_chain(param_input)
        if last_ag is None:
            unshard_outputs.append(param_input)
            continue
        found_collective = True
        wait_node = next(iter(last_ag.users))
        if not _is_wait_tensor(wait_node):
            raise ValueError(
                f"Expected wait_tensor after all_gather node {last_ag.name}, "
                f"got {wait_node.name}"
            )
        wait_chain_user = _find_last_user_in_wait_chain(wait_node)
        unshard_outputs.append(_find_last_non_view_node_in_chain(wait_chain_user))

    if not found_collective:
        return GraphPPFSDPForwardSplit(
            fw_fsdp_module=None,
            unshard_module=None,
            fw_no_fsdp_module=fw_module,
            unshard_output_descs=(),
            fw_no_fsdp_input_descs=_placeholder_descs(fw_module),
            fw_no_fsdp_output_descs=_output_descs(fw_module),
        )

    graph_outputs = _graph_outputs(graph)
    output_node = graph.find_nodes(op="output")[0]
    graph_output_descs = pytree.arg_tree_leaves(
        output_node.meta.get("desc", [None] * len(graph_outputs))
    )
    unshard_output_descs = [FSDPUnshardOutput() for _ in unshard_outputs]

    with _exclude_from_fx_side_effectful(
        {
            torch.ops._c10d_functional.wait_tensor,
            torch.ops._c10d_functional.wait_tensor.default,
        }
    ):
        unshard_graph = _extract_graph_with_inputs_outputs(
            graph,
            param_inputs,
            unshard_outputs,
            unshard_output_descs,
            ignore_must_be_in_fw_bw=True,
        )
        fw_fsdp_graph = _extract_graph_with_inputs_outputs(
            graph,
            placeholders,
            list(graph_outputs) + unshard_outputs,
            graph_output_descs + unshard_output_descs,
            ignore_must_be_in_fw_bw=True,
        )
        fw_no_fsdp_graph = _extract_graph_with_inputs_outputs(
            graph,
            unshard_outputs + remaining_inputs,
            list(graph_outputs),
            graph_output_descs,
            ignore_must_be_in_fw_bw=True,
        )

    unshard_module = _make_graph_module_like(fw_module, unshard_graph)
    fw_fsdp_module = _make_graph_module_like(fw_module, fw_fsdp_graph)
    fw_no_fsdp_module = _make_graph_module_like(fw_module, fw_no_fsdp_graph)
    return GraphPPFSDPForwardSplit(
        fw_fsdp_module=fw_fsdp_module,
        unshard_module=unshard_module,
        fw_no_fsdp_module=fw_no_fsdp_module,
        unshard_output_descs=_output_descs(unshard_module),
        fw_no_fsdp_input_descs=_placeholder_descs(fw_no_fsdp_module),
        fw_no_fsdp_output_descs=_output_descs(fw_no_fsdp_module),
    )


def split_backward_fsdp_collectives(
    bw_module: fx.GraphModule,
    *,
    num_param_buffer_grads: int,
) -> GraphPPFSDPBackwardSplit:
    """Split backward FSDP reduce-scatter epilogues from a backward graph."""
    if num_param_buffer_grads < 0:
        raise ValueError(
            f"num_param_buffer_grads must be non-negative, got {num_param_buffer_grads}"
        )

    graph = deepcopy(bw_module.graph)
    placeholders = graph.find_nodes(op="placeholder")
    graph_outputs = _graph_outputs(graph)
    grad_outputs = graph_outputs[:num_param_buffer_grads]
    remaining_outputs = graph_outputs[num_param_buffer_grads:]
    output_node = graph.find_nodes(op="output")[0]
    output_descs = pytree.arg_tree_leaves(
        output_node.meta.get("desc", [None] * len(graph_outputs))
    )
    grad_output_descs = output_descs[:num_param_buffer_grads]
    remaining_output_descs = output_descs[num_param_buffer_grads:]

    reduce_grad_inputs = []
    found_collective = False
    for grad_output in grad_outputs:
        node = grad_output
        earliest_rs_input = None
        while isinstance(node, fx.Node) and len(node.all_input_nodes) == 1:
            input_node = node.all_input_nodes[0]
            if len(input_node.users) > 1:
                break
            previous_node = node
            node = input_node
            if _is_reduce_scatter_tensor(previous_node):
                earliest_rs_input = node
        if earliest_rs_input is not None:
            found_collective = True
            reduce_grad_inputs.append(earliest_rs_input)
        else:
            reduce_grad_inputs.append(grad_output)

    if not found_collective:
        return GraphPPFSDPBackwardSplit(
            bw_no_fsdp_module=bw_module,
            reduce_grad_module=None,
            bw_no_fsdp_output_descs=_output_descs(bw_module),
            reduce_grad_input_descs=(),
        )

    reduce_grad_input_descs = [FSDPReduceGradInput() for _ in reduce_grad_inputs]
    with _exclude_from_fx_side_effectful(
        {
            torch.ops._c10d_functional.wait_tensor,
            torch.ops._c10d_functional.wait_tensor.default,
        }
    ):
        bw_no_fsdp_graph = _extract_graph_with_inputs_outputs(
            graph,
            placeholders,
            reduce_grad_inputs + list(remaining_outputs),
            reduce_grad_input_descs + remaining_output_descs,
            ignore_must_be_in_fw_bw=True,
        )
        reduce_grad_graph = _extract_graph_with_inputs_outputs(
            graph,
            reduce_grad_inputs,
            list(grad_outputs),
            grad_output_descs,
            ignore_must_be_in_fw_bw=True,
        )

    bw_no_fsdp_module = _make_graph_module_like(bw_module, bw_no_fsdp_graph)
    reduce_grad_module = _make_graph_module_like(bw_module, reduce_grad_graph)
    return GraphPPFSDPBackwardSplit(
        bw_no_fsdp_module=bw_no_fsdp_module,
        reduce_grad_module=reduce_grad_module,
        bw_no_fsdp_output_descs=_output_descs(bw_no_fsdp_module),
        reduce_grad_input_descs=_placeholder_descs(reduce_grad_module),
    )
