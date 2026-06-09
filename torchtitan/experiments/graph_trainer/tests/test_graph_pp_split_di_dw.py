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
    split_di_dw_graph,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
)


class _TinyStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _make_train_step(model: nn.Module, *, include_input_grad: bool):
    def train_step(x: torch.Tensor, target: torch.Tensor):
        out = model(x)
        loss = (out - target).pow(2).sum()
        params = [p for p in model.parameters() if p.requires_grad]
        grad_inputs = [*params, x] if include_input_grad else params
        grads = torch.autograd.grad(loss, grad_inputs)
        return [loss, *grads]

    return train_step


def _flat_runtime_inputs(model: nn.Module, x: torch.Tensor, target: torch.Tensor):
    return [*extract_module_state(model).values(), x, target]


def _trace_with_module(fn, model: nn.Module, *args):
    def _stateless_fn(state: dict[str, torch.Tensor], *user_args):
        with stateless._reparametrize_module(model, state):
            return fn(*user_args)

    return minimal_fx_tracer(_stateless_fn)(extract_module_state(model), *args)


class GraphPPSplitDiDwTest(unittest.TestCase):
    def test_split_di_dw_reconstructs_full_backward(self) -> None:
        torch.manual_seed(0)
        model = _TinyStage()
        x = torch.randn(2, 4, requires_grad=True)
        target = torch.randn(2, 3)
        traced = _trace_with_module(
            _make_train_step(model, include_input_grad=True),
            model,
            x,
            target,
        )
        partitioned = partition_joint_graph(
            traced.gm,
            traced.example_inputs,
            num_fwd_outputs=1,
        )

        split = split_di_dw_graph(
            partitioned.bw_module,
            num_param_buffer_grads=2,
        )
        self.assertIsNotNone(split)
        assert split is not None

        fw_outputs = execute_graph_boxed(
            partitioned.fw_module,
            _flat_runtime_inputs(model, x, target),
        )
        full_bw_outputs = execute_graph_boxed(
            partitioned.bw_module,
            list(fw_outputs[1:]),
        )

        di_outputs = execute_graph_boxed(split.bw_di_module, list(fw_outputs[1:]))
        input_grads = di_outputs[: split.num_input_grads]
        dw_live_ins = di_outputs[split.num_input_grads :]
        dw_outputs = execute_graph_boxed(split.bw_dw_module, list(dw_live_ins))

        self.assertEqual(split.num_input_grads, 1)
        self.assertEqual(len(dw_outputs), 2)
        for actual, expected in zip(dw_outputs, full_bw_outputs[:2], strict=True):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(input_grads[0], full_bw_outputs[2]))

    def test_first_stage_no_input_grads_returns_none(self) -> None:
        model = _TinyStage()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        traced = _trace_with_module(
            _make_train_step(model, include_input_grad=False),
            model,
            x,
            target,
        )
        partitioned = partition_joint_graph(
            traced.gm,
            traced.example_inputs,
            num_fwd_outputs=1,
        )

        split = split_di_dw_graph(
            partitioned.bw_module,
            num_param_buffer_grads=2,
        )

        self.assertIsNone(split)


if __name__ == "__main__":
    unittest.main()
