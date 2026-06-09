# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.graph_pp import (
    split_backward_fsdp_collectives,
    split_forward_fsdp_collectives,
)


def _targets(gm: fx.GraphModule) -> set[object]:
    return {
        node.target
        for node in gm.graph.nodes
        if node.op == "call_function"
    }


def _placeholder_names(gm: fx.GraphModule) -> list[str]:
    return [node.name for node in gm.graph.find_nodes(op="placeholder")]


def _make_forward_fsdp_graph() -> fx.GraphModule:
    graph = fx.Graph()
    param = graph.placeholder("param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(param, 1, "0"),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(wait, x))
    graph.output((out,))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def _make_forward_no_fsdp_graph() -> fx.GraphModule:
    graph = fx.Graph()
    param = graph.placeholder("param")
    x = graph.placeholder("x")
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(param, x))
    graph.output((out,))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def _make_backward_fsdp_graph() -> fx.GraphModule:
    graph = fx.Graph()
    grad = graph.placeholder("grad")
    reduce_scatter = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        args=(grad, "sum", 1, "0"),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(reduce_scatter,),
    )
    graph.output((wait,))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


def _make_backward_no_fsdp_graph() -> fx.GraphModule:
    graph = fx.Graph()
    grad = graph.placeholder("grad")
    out = graph.call_function(torch.ops.aten.neg.default, args=(grad,))
    graph.output((out,))
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


class GraphPPFSDPCollectiveSplitTest(unittest.TestCase):
    def test_forward_split_extracts_unshard_graph(self) -> None:
        split = split_forward_fsdp_collectives(
            _make_forward_fsdp_graph(),
            num_params=1,
        )

        self.assertIsNotNone(split.fw_fsdp_module)
        self.assertIsNotNone(split.unshard_module)
        assert split.unshard_module is not None
        assert split.fw_fsdp_module is not None
        self.assertEqual(_placeholder_names(split.unshard_module), ["param"])
        self.assertEqual(_placeholder_names(split.fw_fsdp_module), ["param", "x"])
        self.assertEqual(len(_placeholder_names(split.fw_no_fsdp_module)), 2)
        self.assertEqual(_placeholder_names(split.fw_no_fsdp_module)[1], "x")
        self.assertIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _targets(split.unshard_module),
        )
        self.assertIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _targets(split.fw_fsdp_module),
        )
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _targets(split.fw_no_fsdp_module),
        )
        self.assertEqual(len(split.unshard_output_descs), 1)
        self.assertEqual(len(_placeholder_names(split.fw_fsdp_module)), 2)
        self.assertEqual(len(split.fw_fsdp_module.graph.find_nodes(op="output")), 1)
        self.assertEqual(len(split.fw_no_fsdp_input_descs), 2)
        self.assertEqual(len(split.fw_no_fsdp_output_descs), 1)

    def test_forward_split_no_fsdp_is_noop(self) -> None:
        gm = _make_forward_no_fsdp_graph()
        split = split_forward_fsdp_collectives(gm, num_params=1)

        self.assertIsNone(split.fw_fsdp_module)
        self.assertIsNone(split.unshard_module)
        self.assertIs(split.fw_no_fsdp_module, gm)
        self.assertEqual(_placeholder_names(split.fw_no_fsdp_module), ["param", "x"])

    def test_backward_split_extracts_reduce_grad_graph(self) -> None:
        split = split_backward_fsdp_collectives(
            _make_backward_fsdp_graph(),
            num_param_buffer_grads=1,
        )

        self.assertIsNotNone(split.reduce_grad_module)
        assert split.reduce_grad_module is not None
        self.assertEqual(_placeholder_names(split.bw_no_fsdp_module), ["grad"])
        self.assertEqual(_placeholder_names(split.reduce_grad_module), ["grad"])
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            _targets(split.bw_no_fsdp_module),
        )
        self.assertIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            _targets(split.reduce_grad_module),
        )
        self.assertEqual(len(split.bw_no_fsdp_output_descs), 1)
        self.assertEqual(len(split.reduce_grad_input_descs), 1)

    def test_backward_split_no_fsdp_is_noop(self) -> None:
        gm = _make_backward_no_fsdp_graph()
        split = split_backward_fsdp_collectives(gm, num_param_buffer_grads=1)

        self.assertIsNone(split.reduce_grad_module)
        self.assertIs(split.bw_no_fsdp_module, gm)
        self.assertEqual(_placeholder_names(split.bw_no_fsdp_module), ["grad"])


if __name__ == "__main__":
    unittest.main()
