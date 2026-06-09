# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from itertools import dropwhile

import torch.fx as fx
from torch._logging import trace_structured


def _copy_prefixed_tensor_constants(
    dst: fx.GraphModule,
    src: fx.GraphModule,
    *,
    prefix: str,
) -> dict[str, str]:
    remap: dict[str, str] = {}
    for attr_name in dir(src):
        if not attr_name.startswith("_tensor_constant"):
            continue
        new_attr_name = f"{prefix}{attr_name}"
        setattr(dst, new_attr_name, getattr(src, attr_name))
        remap[attr_name] = new_attr_name
    return remap


def multiplex_fw_bw_graph(
    fw_gm: fx.GraphModule,
    bw_gm: fx.GraphModule,
) -> fx.GraphModule:
    """Concatenate backward and forward graphs into one boxed GraphPP callable.

    The returned graph has placeholders ordered as ``bw_inputs + fw_inputs``
    and outputs ordered as ``bw_outputs + fw_outputs``.  DualPipeV uses this
    for the ``OVERLAP_F_B`` action.  EP-overlap annotations are intentionally
    not applied inside this graph; pass ordering keeps EP-overlap on the
    standalone no-FSDP forward/backward graphs.
    """
    old_to_new: dict[fx.Node, fx.Node] = {}
    multiplexed_gm = copy.deepcopy(bw_gm)
    fw_constant_remap = _copy_prefixed_tensor_constants(
        multiplexed_gm,
        fw_gm,
        prefix="fw",
    )

    fw_placeholders = fw_gm.graph.find_nodes(op="placeholder")
    insert_point = multiplexed_gm.graph.find_nodes(op="placeholder")[-1]
    for node in fw_placeholders:
        with multiplexed_gm.graph.inserting_after(insert_point):
            new_placeholder = multiplexed_gm.graph.placeholder(f"fw_{node.name}")
        new_placeholder.meta = copy.copy(node.meta)
        old_to_new[node] = new_placeholder
        insert_point = new_placeholder

    fw_nodes = iter(fw_gm.graph.nodes)
    fw_nodes = dropwhile(lambda node: node.op == "placeholder", fw_nodes)
    insert_point = multiplexed_gm.graph.find_nodes(op="output")[-1]
    for node in fw_nodes:
        if node.op == "output":
            break
        with multiplexed_gm.graph.inserting_before(insert_point):
            new_node = multiplexed_gm.graph.node_copy(node, lambda arg: old_to_new[arg])
        new_node.meta = copy.copy(node.meta)
        if new_node.op == "get_attr" and new_node.target in fw_constant_remap:
            new_node.target = fw_constant_remap[str(new_node.target)]
        old_to_new[node] = new_node

    fw_output_node = fw_gm.graph.find_nodes(op="output")[0]
    multiplexed_output = multiplexed_gm.graph.find_nodes(op="output")[0]
    fw_outputs = [
        old_to_new[value] if isinstance(value, fx.Node) else value
        for value in fw_output_node.args[0]
    ]
    bw_outputs = list(multiplexed_output.args[0])
    multiplexed_output.args = (tuple(bw_outputs + fw_outputs),)

    multiplexed_gm.graph.eliminate_dead_code()
    multiplexed_gm.graph.lint()
    multiplexed_gm.recompile()
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "graph_pp_multiplexed_graph",
            "encoding": "string",
        },
        payload_fn=lambda: multiplexed_gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
        ),
    )
    return multiplexed_gm
