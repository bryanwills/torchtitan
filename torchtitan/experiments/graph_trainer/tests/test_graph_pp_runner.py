# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    GraphPPRunner,
    _run_dI_bw_module,
    _run_dW_bw_module,
    _run_full_bw_module,
    _run_fw_module,
    _trace_stage_graphs,
)
from torchtitan.config import ParallelismConfig
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
    _validate_graph_pp_config,
)


class GraphPPRunnerTraceTest(unittest.TestCase):
    def test_graph_pp_accumulates_grads_only_for_trainable_params(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        for param in model[0].parameters():
            param.requires_grad_(False)

        stage = types.SimpleNamespace(
            submod=model,
            state={
                "sharded_params": [],
                "unsharded_params": [],
                "buffers": [],
                "sharded_grads": [],
                "unsharded_grads": [],
                "trainable_params": [],
            },
        )
        runner = GraphPPRunner.__new__(GraphPPRunner)
        runner._populate_stage_states(stage)

        self.assertEqual(len(stage.state["sharded_params"]), 4)
        self.assertEqual(len(stage.state["trainable_params"]), 2)
        self.assertEqual(len(stage.state["unsharded_grads"]), 2)

    def test_single_stage_schedule_hard_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime PP schedule"):
            _validate_graph_pp_config(
                compile_config=GraphTrainerCompileConfig(),
                parallelism=ParallelismConfig(pipeline_parallel_schedule="1F1B"),
            )

    def test_runtime_schedule_validation_accepts_interleaved(self) -> None:
        _validate_graph_pp_config(
            compile_config=GraphTrainerCompileConfig(),
            parallelism=ParallelismConfig(
                pipeline_parallel_schedule="Interleaved1F1B"
            ),
        )

    def test_intermediate_stage_graphs_match_eager_grads(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 3)
        stage = types.SimpleNamespace(
            submod=model,
            is_last=False,
            loss_fn=None,
            stage_index=0,
        )
        x = torch.randn(2, 4, requires_grad=True)
        output_grad = torch.randn(2, 3)

        _trace_stage_graphs(stage, (x,), {}, None, {})

        state = [*model.parameters()]
        output, saved = _run_fw_module(stage.graph_callables.fw, stage.graph_meta, [*state, x])
        self.assertTrue(torch.allclose(output, model(x)))

        input_grads, param_grads = _run_full_bw_module(
            stage.graph_callables.full_bw,
            stage.graph_meta,
            [*saved, output_grad],
        )
        expected_grads = torch.autograd.grad(
            model(x),
            [*model.parameters(), x],
            grad_outputs=output_grad,
        )
        for actual, expected in zip(param_grads + input_grads, expected_grads, strict=True):
            self.assertTrue(torch.allclose(actual, expected))

        dI_grads, dW_inputs = _run_dI_bw_module(
            stage.graph_callables.bw_dI,
            stage.graph_meta,
            [*saved, output_grad],
        )
        dW_grads = _run_dW_bw_module(stage.graph_callables.bw_dW, dW_inputs)
        for actual, expected in zip(dW_grads + dI_grads, expected_grads, strict=True):
            self.assertTrue(torch.allclose(actual, expected))

    def test_last_stage_graphs_return_loss_and_input_grad(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 3)

        def loss_fn(pred, target, global_valid_tokens):
            return ((pred - target) ** 2).sum() / global_valid_tokens

        stage = types.SimpleNamespace(
            submod=model,
            is_last=True,
            loss_fn=loss_fn,
            stage_index=1,
        )
        x = torch.randn(2, 4, requires_grad=True)
        target = torch.randn(2, 3)
        global_valid_tokens = torch.tensor(2.0)

        _trace_stage_graphs(
            stage,
            (x,),
            {},
            target,
            {"global_valid_tokens": global_valid_tokens},
        )

        state = [*model.parameters()]
        loss, saved = _run_fw_module(
            stage.graph_callables.fw,
            stage.graph_meta,
            [*state, x, target, global_valid_tokens],
        )
        expected_loss = loss_fn(model(x), target, global_valid_tokens)
        self.assertTrue(torch.allclose(loss, expected_loss))

        input_grads, param_grads = _run_full_bw_module(
            stage.graph_callables.full_bw,
            stage.graph_meta,
            list(saved),
        )
        expected_grads = torch.autograd.grad(
            expected_loss,
            [*model.parameters(), x],
        )
        for actual, expected in zip(param_grads + input_grads, expected_grads, strict=True):
            self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
