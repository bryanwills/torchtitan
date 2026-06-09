# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.fx as fx
from torch._functorch.partitioners import (
    _extract_graph_with_inputs_outputs,
    default_partition,
)


GraphPPSlotKind = Literal[
    "tensor",
    "symint",
    "symfloat",
    "symbool",
    "none",
    "primitive",
    "opaque",
]


@dataclass(frozen=True, slots=True)
class GraphPPSlotDescriptor:
    """Stable description for one GraphPP graph input or output slot."""

    index: int
    name: str
    kind: GraphPPSlotKind
    source_node: str | None = None
    target: str | None = None


@dataclass(frozen=True, slots=True)
class GraphPPGraphMeta:
    """Calling-convention metadata for a partitioned GraphPP stage graph pair."""

    fwd_input_descs: tuple[GraphPPSlotDescriptor, ...]
    fwd_user_output_descs: tuple[GraphPPSlotDescriptor, ...]
    saved_for_backward_descs: tuple[GraphPPSlotDescriptor, ...]
    bwd_runtime_input_descs: tuple[GraphPPSlotDescriptor, ...]
    bwd_input_descs: tuple[GraphPPSlotDescriptor, ...]
    bwd_output_descs: tuple[GraphPPSlotDescriptor, ...]

    @property
    def num_fwd_inputs(self) -> int:
        return len(self.fwd_input_descs)

    @property
    def num_fwd_user_outputs(self) -> int:
        return len(self.fwd_user_output_descs)

    @property
    def num_saved_for_backward(self) -> int:
        return len(self.saved_for_backward_descs)

    @property
    def num_bwd_inputs(self) -> int:
        return len(self.bwd_input_descs)

    @property
    def num_bwd_runtime_inputs(self) -> int:
        return len(self.bwd_runtime_input_descs)

    @property
    def num_bwd_outputs(self) -> int:
        return len(self.bwd_output_descs)


@dataclass(frozen=True, slots=True)
class GraphPPPartitionedGraphs:
    """Forward/backward GraphModules extracted from one joint FX graph."""

    fw_module: fx.GraphModule
    bw_module: fx.GraphModule
    meta: GraphPPGraphMeta


def _slot_kind_from_value(value: Any) -> GraphPPSlotKind:
    if isinstance(value, torch.Tensor):
        return "tensor"
    if isinstance(value, torch.SymInt):
        return "symint"
    if isinstance(value, torch.SymFloat):
        return "symfloat"
    if isinstance(value, torch.SymBool):
        return "symbool"
    if value is None:
        return "none"
    if isinstance(value, (int, float, bool, str)):
        return "primitive"
    return "opaque"


def _node_descriptor(index: int, node: fx.Node) -> GraphPPSlotDescriptor:
    return GraphPPSlotDescriptor(
        index=index,
        name=node.name,
        kind=_slot_kind_from_value(node.meta.get("val")),
        source_node=node.name,
        target=str(node.target),
    )


def _value_descriptor(index: int, value: Any, *, name: str) -> GraphPPSlotDescriptor:
    if isinstance(value, fx.Node):
        return _node_descriptor(index, value)
    return GraphPPSlotDescriptor(index=index, name=name, kind=_slot_kind_from_value(value))


def _graph_output_values(gm: fx.GraphModule) -> tuple[Any, ...]:
    outputs = gm.graph.find_nodes(op="output")
    if len(outputs) != 1:
        raise ValueError(f"Expected exactly one output node in {gm}, found {len(outputs)}")
    output_arg = outputs[0].args[0]
    if not isinstance(output_arg, tuple):
        output_arg = (output_arg,)
    return tuple(output_arg)


def _placeholder_descs(gm: fx.GraphModule) -> tuple[GraphPPSlotDescriptor, ...]:
    return tuple(
        _node_descriptor(index, node)
        for index, node in enumerate(gm.graph.find_nodes(op="placeholder"))
    )


def _output_descs(gm: fx.GraphModule) -> tuple[GraphPPSlotDescriptor, ...]:
    return tuple(
        _value_descriptor(index, value, name=f"output_{index}")
        for index, value in enumerate(_graph_output_values(gm))
    )


def _placeholder_dependencies(values: list[Any]) -> set[fx.Node]:
    dependencies: set[fx.Node] = set()
    queue = [value for value in values if isinstance(value, fx.Node)]
    while queue:
        node = queue.pop()
        if node.op == "placeholder":
            dependencies.add(node)
            continue
        queue.extend(node.all_input_nodes)
    return dependencies


def partition_joint_graph(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...],
    *,
    num_fwd_outputs: int,
    backward_only_input_indices: tuple[int, ...] = (),
) -> GraphPPPartitionedGraphs:
    """Partition a `minimal_fx_tracer` joint graph into forward/backward graphs.

    The initial implementation deliberately uses the public functorch
    partitioner entrypoint instead of AOTAutograd stage2a.  It preserves the
    direct GraphPP contract: forward returns user outputs followed by saved
    slots, and backward consumes exactly those saved slots as its leading
    placeholders.
    """
    if num_fwd_outputs < 1:
        raise ValueError(f"num_fwd_outputs must be positive, got {num_fwd_outputs}")

    joint = copy.deepcopy(gm)
    fw_module, bw_module = default_partition(
        joint,
        example_inputs,
        num_fwd_outputs=num_fwd_outputs,
    )

    backward_only_input_indices_set = set(backward_only_input_indices)
    joint_placeholders = joint.graph.find_nodes(op="placeholder")
    backward_only_names = {
        node.name
        for index, node in enumerate(joint_placeholders)
        if index in backward_only_input_indices_set
    }
    fw_placeholders = fw_module.graph.find_nodes(op="placeholder")
    fw_output_node = fw_module.graph.find_nodes(op="output")[0]
    fw_outputs = _graph_output_values(fw_module)
    fw_output_descs = fw_output_node.meta.get("desc", [None] * len(fw_outputs))
    user_outputs = list(fw_outputs[:num_fwd_outputs])
    user_output_descs = list(fw_output_descs[:num_fwd_outputs])
    saved_outputs_by_name: dict[str, Any] = {}
    saved_descs_by_name: dict[str, Any] = {}
    for output, desc in zip(
        fw_outputs[num_fwd_outputs:],
        fw_output_descs[num_fwd_outputs:],
        strict=True,
    ):
        if isinstance(output, fx.Node):
            saved_outputs_by_name[output.name] = output
            saved_descs_by_name[output.name] = desc

    ordered_saved_outputs = []
    ordered_saved_descs = []
    for bwd_placeholder in bw_module.graph.find_nodes(op="placeholder"):
        if bwd_placeholder.name in backward_only_names:
            continue
        if bwd_placeholder.name not in saved_outputs_by_name:
            raise ValueError(
                "Backward placeholder is neither a forward saved output nor a "
                "runtime backward input: "
                f"{bwd_placeholder.name}"
            )
        ordered_saved_outputs.append(saved_outputs_by_name[bwd_placeholder.name])
        ordered_saved_descs.append(saved_descs_by_name[bwd_placeholder.name])

    kept_outputs = user_outputs + ordered_saved_outputs
    kept_output_descs = user_output_descs + ordered_saved_descs
    needed_fw_placeholders = _placeholder_dependencies(kept_outputs)
    fw_inputs = [
        node
        for node in fw_placeholders
        if node in needed_fw_placeholders and node.name not in backward_only_names
    ]

    if list(fw_inputs) != list(fw_placeholders) or list(fw_outputs) != kept_outputs:
        fw_graph = _extract_graph_with_inputs_outputs(
            fw_module.graph,
            fw_inputs,
            kept_outputs,
            kept_output_descs,
            ignore_must_be_in_fw_bw=True,
        )
        fw_module = torch.fx._lazy_graph_module._make_graph_module(
            fw_module, fw_graph
        )

    fw_module.graph.lint()
    bw_module.graph.lint()
    fw_module.recompile()
    bw_module.recompile()

    fwd_output_descs = _output_descs(fw_module)
    if len(fwd_output_descs) < num_fwd_outputs:
        raise ValueError(
            f"Forward graph returned {len(fwd_output_descs)} outputs, "
            f"expected at least {num_fwd_outputs}"
        )
    saved_for_backward_descs = fwd_output_descs[num_fwd_outputs:]
    bwd_input_descs = _placeholder_descs(bw_module)
    bwd_runtime_input_descs = tuple(
        _node_descriptor(index, node)
        for index, node in enumerate(bw_module.graph.find_nodes(op="placeholder"))
        if node.name in backward_only_names
    )

    expected_bwd_input_count = (
        len(saved_for_backward_descs) + len(bwd_runtime_input_descs)
    )
    if expected_bwd_input_count != len(bwd_input_descs):
        raise ValueError(
            "Forward saved-slot plus runtime backward input count must match "
            "backward placeholder count: "
            f"{expected_bwd_input_count} != {len(bwd_input_descs)}"
        )

    saved_names = {desc.name for desc in saved_for_backward_descs}
    runtime_names = {desc.name for desc in bwd_runtime_input_descs}
    for bwd_input in bwd_input_descs:
        if bwd_input.name not in saved_names and bwd_input.name not in runtime_names:
            raise ValueError(
                "Backward placeholder is not supplied by forward saved slots "
                "or runtime backward inputs: "
                f"{bwd_input.name!r}"
            )

    return GraphPPPartitionedGraphs(
        fw_module=fw_module,
        bw_module=bw_module,
        meta=GraphPPGraphMeta(
            fwd_input_descs=_placeholder_descs(fw_module),
            fwd_user_output_descs=fwd_output_descs[:num_fwd_outputs],
            saved_for_backward_descs=saved_for_backward_descs,
            bwd_runtime_input_descs=bwd_runtime_input_descs,
            bwd_input_descs=bwd_input_descs,
            bwd_output_descs=_output_descs(bw_module),
        ),
    )
