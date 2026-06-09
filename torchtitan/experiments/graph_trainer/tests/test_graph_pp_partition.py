# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from torch.nn.utils import stateless

from torchtitan.experiments.graph_trainer.graph_pp import (
    execute_graph_boxed,
    partition_joint_graph,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
)


class _TinyTrainStep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _make_train_step(model: nn.Module):
    def train_step(x: torch.Tensor, target: torch.Tensor):
        out = model(x)
        loss = (out - target).pow(2).sum()
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params)
        return [loss, *grads]

    return train_step


def _make_stage_step(model: nn.Module):
    def stage_step(x: torch.Tensor, output_grad: torch.Tensor):
        out = model(x)
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(out, [*params, x], grad_outputs=output_grad)
        return [out, *grads]

    return stage_step


def _flat_runtime_inputs(model: nn.Module, *user_inputs: torch.Tensor) -> list[object]:
    model_state = extract_module_state(model)
    return [*model_state.values(), *user_inputs]


def _trace_with_module(fn, model: nn.Module, *args):
    def _stateless_fn(state: dict[str, torch.Tensor], *user_args):
        with stateless._reparametrize_module(model, state):
            return fn(*user_args)

    return minimal_fx_tracer(_stateless_fn)(extract_module_state(model), *args)


class GraphPPPartitionTest(unittest.TestCase):
    def test_partition_matches_joint_graph(self) -> None:
        torch.manual_seed(0)
        model = _TinyTrainStep()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        traced = _trace_with_module(_make_train_step(model), model, x, target)

        partitioned = partition_joint_graph(
            traced.gm,
            traced.example_inputs,
            num_fwd_outputs=1,
        )

        flat_inputs = _flat_runtime_inputs(model, x, target)
        joint_outputs = traced.gm(*flat_inputs)

        fw_args = list(flat_inputs)
        fw_outputs = execute_graph_boxed(partitioned.fw_module, fw_args)
        self.assertEqual(fw_args, [])
        self.assertTrue(torch.equal(fw_outputs[0], joint_outputs[0]))

        bw_args = list(fw_outputs[1:])
        bw_outputs = execute_graph_boxed(partitioned.bw_module, bw_args)
        self.assertEqual(bw_args, [])
        self.assertEqual(len(bw_outputs), len(joint_outputs) - 1)
        for actual, expected in zip(bw_outputs, joint_outputs[1:], strict=True):
            self.assertTrue(torch.equal(actual, expected))

    def test_metadata_describes_calling_convention(self) -> None:
        model = _TinyTrainStep()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        traced = _trace_with_module(_make_train_step(model), model, x, target)

        partitioned = partition_joint_graph(
            traced.gm,
            traced.example_inputs,
            num_fwd_outputs=1,
        )
        meta = partitioned.meta

        self.assertEqual(meta.num_fwd_user_outputs, 1)
        self.assertEqual(meta.num_fwd_inputs, len(traced.example_inputs))
        self.assertEqual(meta.num_saved_for_backward, meta.num_bwd_inputs)
        self.assertEqual(
            [slot.name for slot in meta.saved_for_backward_descs],
            [slot.name for slot in meta.bwd_input_descs],
        )
        self.assertEqual({slot.kind for slot in meta.fwd_input_descs}, {"tensor"})

    def test_partition_does_not_require_optional_source_metadata(self) -> None:
        model = _TinyTrainStep()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        traced = _trace_with_module(_make_train_step(model), model, x, target)
        for node in traced.gm.graph.nodes:
            node.meta.pop("custom", None)
            node.meta.pop("seq_nr", None)

        partitioned = partition_joint_graph(
            traced.gm,
            traced.example_inputs,
            num_fwd_outputs=1,
        )

        self.assertEqual(partitioned.meta.num_fwd_user_outputs, 1)
        self.assertEqual(
            partitioned.meta.num_saved_for_backward,
            partitioned.meta.num_bwd_inputs,
        )

    def test_partition_keeps_backward_only_tangent_out_of_forward(self) -> None:
        model = _TinyTrainStep()
        x = torch.randn(2, 4, requires_grad=True)
        output_grad = torch.randn(2, 3)
        traced = _trace_with_module(_make_stage_step(model), model, x, output_grad)

        partitioned = partition_joint_graph(
            traced.gm,
            traced.example_inputs,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced.example_inputs) - 1,),
        )

        self.assertNotIn(
            partitioned.meta.bwd_runtime_input_descs[0].name,
            [slot.name for slot in partitioned.meta.fwd_input_descs],
        )
        self.assertEqual(partitioned.meta.num_bwd_runtime_inputs, 1)
        self.assertEqual(
            partitioned.meta.num_bwd_inputs,
            partitioned.meta.num_saved_for_backward
            + partitioned.meta.num_bwd_runtime_inputs,
        )

        flat_inputs = _flat_runtime_inputs(model, x)
        fw_outputs = execute_graph_boxed(partitioned.fw_module, flat_inputs)
        self.assertEqual(flat_inputs, [])
        self.assertEqual(len(fw_outputs), partitioned.meta.num_fwd_user_outputs + 2)

        bw_args = [*fw_outputs[1:], output_grad]
        bw_outputs = execute_graph_boxed(partitioned.bw_module, bw_args)
        self.assertEqual(bw_args, [])
        joint_outputs = traced.gm(*_flat_runtime_inputs(model, x, output_grad))
        for actual, expected in zip(bw_outputs, joint_outputs[1:], strict=True):
            self.assertTrue(torch.equal(actual, expected))

    def test_invalid_forward_output_count_raises(self) -> None:
        model = _TinyTrainStep()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        traced = _trace_with_module(_make_train_step(model), model, x, target)

        with self.assertRaisesRegex(ValueError, "num_fwd_outputs"):
            partition_joint_graph(
                traced.gm,
                traced.example_inputs,
                num_fwd_outputs=0,
            )


if __name__ == "__main__":
    unittest.main()
