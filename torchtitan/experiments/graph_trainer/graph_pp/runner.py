# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import re
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.nn.utils import stateless
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineContext,
    _PipelineScheduleRuntime,
    _wait_batch_p2p,
    BACKWARD_INPUT,
    BACKWARD_WEIGHT,
    FORWARD,
    FULL_BACKWARD,
    REDUCE_GRAD,
    RESHARD,
    UNSHARD,
)
from torch.distributed.pipelining.stage import (
    _normalize_model_output_as_tuple,
    PipelineStage,
)
from torch.distributed.tensor import DTensor

from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.passes import (
    annotate_flex_attention_for_regional_inductor_pass,
    full_inductor_compilation_pass,
    regional_inductor_pass,
)
from torchtitan.experiments.graph_trainer.graph_pp.boxed import execute_graph_boxed
from torchtitan.experiments.graph_trainer.graph_pp.fsdp import (
    split_backward_fsdp_collectives,
    split_forward_fsdp_collectives,
)
from torchtitan.experiments.graph_trainer.graph_pp.partition import (
    GraphPPGraphMeta,
    partition_joint_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.split_di_dw import (
    GraphPPDiDwSplit,
    split_di_dw_graph,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _unwrap_subclasses,
    extract_module_state,
    minimal_fx_tracer,
)
from torchtitan.tools.logging import logger


@dataclasses.dataclass(slots=True)
class GraphCallables:
    fw: fx.GraphModule
    full_bw: fx.GraphModule
    fw_fsdp: fx.GraphModule | None = None
    bw_dI: fx.GraphModule | None = None
    bw_dW: fx.GraphModule | None = None
    unshard: fx.GraphModule | None = None
    reduce_grad: fx.GraphModule | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class GraphMeta:
    num_user_outputs: int
    num_saved_for_backward: int
    num_bwd_runtime_inputs: int
    num_params: int
    num_buffers: int
    num_input_grads: int
    partition: GraphPPGraphMeta
    fwd_input_descs: tuple[Any, ...]
    unshard_output_descs: tuple[Any, ...] = ()
    num_fused_unshard_outputs: int = 0


@dataclasses.dataclass(slots=True)
class StageTraceSpec:
    output_spec: pytree.TreeSpec | None = None
    output_grad_spec: pytree.TreeSpec | None = None

def _execute_graph(graph: fx.GraphModule | Callable[[list[Any]], Any], args: list[Any]):
    return execute_graph_boxed(graph, args)


def _local_tensor(value: Any) -> Any:
    return value.to_local() if isinstance(value, DTensor) else value


def _requires_grad_like(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        value = value.detach().requires_grad_(True)
    return value


def _zero_like_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    return value


def _unwrapped_flat_count(values: list[Any]) -> int:
    unwrapped, _ = _unwrap_subclasses(values)
    return len(unwrapped)


def _graph_outputs(gm: fx.GraphModule) -> tuple[Any, ...]:
    outputs = gm.graph.find_nodes(op="output")
    if len(outputs) != 1:
        raise ValueError(f"Expected exactly one output node, found {len(outputs)}")
    output_arg = outputs[0].args[0]
    if not isinstance(output_arg, tuple):
        output_arg = (output_arg,)
    return tuple(output_arg)


def _example_inputs_from_placeholders(gm: fx.GraphModule) -> tuple[Any, ...]:
    example_inputs = []
    for node in gm.graph.find_nodes(op="placeholder"):
        if "val" not in node.meta:
            raise ValueError(
                "GraphPP cannot compile graph without placeholder metadata: "
                f"{node.name}"
            )
        example_inputs.append(node.meta["val"])
    return tuple(example_inputs)


def _compile_graph_pp_module(
    gm: fx.GraphModule | None,
    *,
    compile_config: GraphTrainerCompileConfig,
    graph_name: str,
) -> fx.GraphModule | None:
    if gm is None or not compile_config.enable_passes:
        return gm
    if compile_config.backend == "aot_eager":
        return gm
    if compile_config.backend != "inductor":
        raise ValueError(
            "GraphPP aot_fx_trace supports --compile.backend aot_eager or "
            f"inductor, got {compile_config.backend!r}"
        )

    example_inputs = _example_inputs_from_placeholders(gm)
    if compile_config.inductor_compilation == "regional":
        from torchtitan.models.common.attention import FlexAttention

        gm = annotate_flex_attention_for_regional_inductor_pass(
            gm,
            example_inputs,
            flex_compile_config=FlexAttention.inductor_configs,
        )
        gm = regional_inductor_pass(gm, example_inputs)
    elif compile_config.inductor_compilation == "full":
        gm = full_inductor_compilation_pass(gm, example_inputs)
    else:
        raise ValueError(
            "GraphPP supports --compile.inductor_compilation regional or full, "
            f"got {compile_config.inductor_compilation!r}"
        )
    logger.info(
        "GraphPP compiled %s with %s inductor",
        graph_name,
        compile_config.inductor_compilation,
    )
    return gm


def _compile_graph_pp_callables(
    callables: GraphCallables,
    *,
    compile_config: GraphTrainerCompileConfig,
    stage_index: int,
) -> GraphCallables:
    return GraphCallables(
        fw=_compile_graph_pp_module(
            callables.fw,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_fw",
        ),
        full_bw=_compile_graph_pp_module(
            callables.full_bw,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_full_bw",
        ),
        fw_fsdp=_compile_graph_pp_module(
            callables.fw_fsdp,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_fw_fsdp",
        ),
        bw_dI=_compile_graph_pp_module(
            callables.bw_dI,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_bw_dI",
        ),
        bw_dW=_compile_graph_pp_module(
            callables.bw_dW,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_bw_dW",
        ),
        unshard=_compile_graph_pp_module(
            callables.unshard,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_unshard",
        ),
        reduce_grad=_compile_graph_pp_module(
            callables.reduce_grad,
            compile_config=compile_config,
            graph_name=f"stage_{stage_index}_reduce_grad",
        ),
    )


def _minimal_fx_trace_stage_step(
    stage: "GraphPipelineStage",
    fn: Callable,
    *args: Any,
):
    def _stateless_fn(state: dict[str, torch.Tensor], *trace_args: Any) -> Any:
        with stateless._reparametrize_module(stage.submod, state):
            return fn(*trace_args)

    return minimal_fx_tracer(_stateless_fn)(extract_module_state(stage.submod), *args)


def _trace_stage_graphs(
    stage: "GraphPipelineStage",
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
) -> None:
    stage_args = pytree.tree_map_only(torch.Tensor, _requires_grad_like, args)
    stage_kwargs = pytree.tree_map_only(torch.Tensor, _requires_grad_like, kwargs)

    state_params = [p for _, p in stage.submod.named_parameters(remove_duplicate=False)]
    grad_params = [p for p in state_params if p.requires_grad]
    buffers = [b for _, b in stage.submod.named_buffers(remove_duplicate=False)]
    num_state_params = len(state_params)
    num_grad_params = len(grad_params)
    num_buffers = len(buffers)
    trace_spec = StageTraceSpec()

    if stage.is_last:

        def stage_step(
            stage_args,
            stage_kwargs,
            target,
            loss_kwargs,
        ):
            pred = stage.submod(*stage_args, **stage_kwargs)
            loss = stage.loss_fn(pred, target, **loss_kwargs)
            grad_params = [
                p
                for _, p in stage.submod.named_parameters(remove_duplicate=False)
                if p.requires_grad
            ]
            flat_inputs, _ = pytree.tree_flatten((stage_args, stage_kwargs))
            grad_inputs = [
                *grad_params,
                *[
                    value
                    for value in flat_inputs
                    if isinstance(value, torch.Tensor) and value.requires_grad
                ],
            ]
            grads = torch.autograd.grad(loss, grad_inputs, allow_unused=True)
            return [loss, *pytree.tree_map_only(DTensor, _local_tensor, grads)]

        traced = _minimal_fx_trace_stage_step(
            stage,
            stage_step,
            stage_args,
            stage_kwargs,
            target,
            loss_kwargs,
        )
        num_fwd_outputs = 1
        backward_only_indices: tuple[int, ...] = ()
    else:
        with torch.no_grad():
            output_example = stage.submod(*args, **kwargs)
        output_grads = pytree.tree_map_only(
            torch.Tensor, _zero_like_tree, output_example
        )
        _, trace_spec.output_spec = pytree.tree_flatten(output_example)
        _, trace_spec.output_grad_spec = pytree.tree_flatten(output_grads)

        def stage_step(stage_args, stage_kwargs, output_grads):
            output = stage.submod(*stage_args, **stage_kwargs)
            flat_outputs, trace_spec.output_spec = pytree.tree_flatten(output)
            flat_output_grads, trace_spec.output_grad_spec = pytree.tree_flatten(
                output_grads
            )
            flat_inputs, _ = pytree.tree_flatten((stage_args, stage_kwargs))
            grad_params = [
                p
                for _, p in stage.submod.named_parameters(remove_duplicate=False)
                if p.requires_grad
            ]
            grad_inputs = [
                *grad_params,
                *[
                    value
                    for value in flat_inputs
                    if isinstance(value, torch.Tensor) and value.requires_grad
                ],
            ]
            grads = torch.autograd.grad(
                flat_outputs,
                grad_inputs,
                grad_outputs=flat_output_grads,
                allow_unused=True,
            )
            return [
                *flat_outputs,
                *pytree.tree_map_only(DTensor, _local_tensor, grads),
            ]

        traced = _minimal_fx_trace_stage_step(
            stage,
            stage_step,
            stage_args,
            stage_kwargs,
            output_grads,
        )
        num_fwd_outputs = len(pytree.tree_leaves(output_example))
        state_flat, _ = pytree.tree_flatten(extract_module_state(stage.submod))
        prefix_user_flat, _ = pytree.tree_flatten(((stage_args, stage_kwargs), {}))
        backward_only_start = _unwrapped_flat_count([*state_flat, *prefix_user_flat])
        backward_only_count = _unwrapped_flat_count(pytree.tree_leaves(output_grads))
        backward_only_indices = tuple(
            range(backward_only_start, backward_only_start + backward_only_count)
        )

    partitioned = partition_joint_graph(
        traced.gm,
        traced.example_inputs,
        num_fwd_outputs=num_fwd_outputs,
        backward_only_input_indices=backward_only_indices,
    )
    fsdp_fw = split_forward_fsdp_collectives(
        partitioned.fw_module,
        num_params=num_state_params,
    )
    fsdp_bw = split_backward_fsdp_collectives(
        partitioned.bw_module,
        num_param_buffer_grads=num_grad_params,
    )
    didw_split: GraphPPDiDwSplit | None = split_di_dw_graph(
        fsdp_bw.bw_no_fsdp_module,
        num_param_buffer_grads=num_grad_params,
    )
    num_input_grads = 0 if didw_split is None else didw_split.num_input_grads
    graph_callables = GraphCallables(
        fw=fsdp_fw.fw_no_fsdp_module,
        full_bw=fsdp_bw.bw_no_fsdp_module,
        fw_fsdp=fsdp_fw.fw_fsdp_module,
        bw_dI=None if didw_split is None else didw_split.bw_di_module,
        bw_dW=None if didw_split is None else didw_split.bw_dw_module,
        unshard=fsdp_fw.unshard_module,
        reduce_grad=fsdp_bw.reduce_grad_module,
    )
    stage.graph_callables = _compile_graph_pp_callables(
        graph_callables,
        compile_config=getattr(
            stage,
            "compile_config",
            GraphTrainerCompileConfig(enable_passes=False),
        ),
        stage_index=stage.stage_index,
    )
    stage.graph_meta = GraphMeta(
        num_user_outputs=partitioned.meta.num_fwd_user_outputs,
        num_saved_for_backward=partitioned.meta.num_saved_for_backward,
        num_bwd_runtime_inputs=partitioned.meta.num_bwd_runtime_inputs,
        num_params=num_grad_params,
        num_buffers=num_buffers,
        num_input_grads=num_input_grads,
        partition=partitioned.meta,
        fwd_input_descs=fsdp_fw.fw_no_fsdp_input_descs,
        unshard_output_descs=fsdp_fw.unshard_output_descs,
        num_fused_unshard_outputs=len(fsdp_fw.unshard_output_descs),
    )
    stage.trace_spec = trace_spec
    logger.info(
        "GraphPP traced stage %s: fwd_outputs=%s saved=%s bwd_runtime_inputs=%s",
        stage.stage_index,
        stage.graph_meta.num_user_outputs,
        stage.graph_meta.num_saved_for_backward,
        stage.graph_meta.num_bwd_runtime_inputs,
    )


def _run_fw_module(
    fw_module: fx.GraphModule,
    graph_meta: GraphMeta,
    fw_args: list[Any],
) -> tuple[Any, tuple[Any, ...]]:
    fw_placeholders = fw_module.graph.find_nodes(op="placeholder")
    if len(fw_args) != len(fw_placeholders):
        raise ValueError(
            "GraphPP forward graph input mismatch: "
            f"expected {len(fw_placeholders)} args, got {len(fw_args)}. "
            "Placeholders: "
            f"{[node.name for node in fw_placeholders]}. "
            f"Graph meta inputs: {[desc.name for desc in graph_meta.fwd_input_descs]}."
        )
    fw_outputs = _execute_graph(fw_module, fw_args)
    user_outputs = fw_outputs[: graph_meta.num_user_outputs]
    saved_intermediates = tuple(fw_outputs[graph_meta.num_user_outputs :])
    output = user_outputs[0] if len(user_outputs) == 1 else tuple(user_outputs)
    return output, saved_intermediates


def _run_fw_fsdp_module(
    fw_fsdp_module: fx.GraphModule,
    graph_meta: GraphMeta,
    fw_args: list[Any],
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    fw_outputs = _execute_graph(fw_fsdp_module, fw_args)
    expected_min_outputs = (
        graph_meta.num_user_outputs
        + graph_meta.num_saved_for_backward
        + graph_meta.num_fused_unshard_outputs
    )
    if len(fw_outputs) < expected_min_outputs:
        raise ValueError(
            "GraphPP fused FSDP forward returned too few outputs: "
            f"expected at least {expected_min_outputs}, got {len(fw_outputs)}"
        )
    user_outputs = fw_outputs[: graph_meta.num_user_outputs]
    saved_start = graph_meta.num_user_outputs
    saved_end = saved_start + graph_meta.num_saved_for_backward
    saved_intermediates = tuple(fw_outputs[saved_start:saved_end])
    unsharded_params = tuple(fw_outputs[saved_end:expected_min_outputs])
    output = user_outputs[0] if len(user_outputs) == 1 else tuple(user_outputs)
    return output, saved_intermediates, unsharded_params


def _run_full_bw_module(
    bw_module: fx.GraphModule,
    graph_meta: GraphMeta,
    bw_args: list[Any],
) -> tuple[list[Any], list[Any]]:
    bw_outputs = _execute_graph(bw_module, bw_args)
    num_param_grads = graph_meta.num_params
    param_buffer_grads = list(bw_outputs[:num_param_grads])
    input_grads = list(bw_outputs[num_param_grads:])
    return input_grads, param_buffer_grads


def _run_dI_bw_module(
    bw_dI_module: fx.GraphModule,
    graph_meta: GraphMeta,
    bw_args: list[Any],
) -> tuple[list[Any], list[Any]]:
    outputs = _execute_graph(bw_dI_module, bw_args)
    return (
        list(outputs[: graph_meta.num_input_grads]),
        list(outputs[graph_meta.num_input_grads :]),
    )


def _run_dW_bw_module(
    bw_dW_module: fx.GraphModule,
    bw_args: list[Any],
) -> list[Any]:
    return list(_execute_graph(bw_dW_module, bw_args))


class GraphPipelineStage(PipelineStage):
    def __init__(
        self,
        submodule: nn.Module,
        *,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        loss_fn: Callable,
        compile_config: GraphTrainerCompileConfig | None = None,
        input_args: Any = None,
        output_args: Any = None,
        group: torch.distributed.ProcessGroup | None = None,
        get_mesh: Callable | None = None,
    ) -> None:
        super().__init__(
            submodule,
            stage_index,
            num_stages,
            device,
            input_args=input_args,
            output_args=output_args,
            group=group,
            get_mesh=get_mesh,
        )
        self.loss_fn = loss_fn
        self.compile_config = compile_config or GraphTrainerCompileConfig()
        self.graph_callables: GraphCallables | None = None
        self.graph_meta: GraphMeta | None = None
        self.trace_spec = StageTraceSpec()
        self.state: dict[str, list[Any]] = {
            "sharded_params": [],
            "unsharded_params": [],
            "buffers": [],
            "sharded_grads": [],
            "unsharded_grads": [],
            "trainable_params": [],
        }
        self.bwd_activation_cache: dict[int, tuple[Any, ...]] = {}
        self._graph_pp_fused_first_forward_done = False
        self._graph_pp_fused_reduce_done = False

    def ensure_graphs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        loss_kwargs: dict[str, Any],
    ) -> None:
        if self.graph_callables is not None:
            return
        _trace_stage_graphs(self, args, kwargs, target, loss_kwargs)

    def _ensure_unsharded_params(self) -> None:
        assert self.graph_callables is not None
        if self.state["unsharded_params"]:
            return
        if self.graph_callables.unshard is None:
            self.state["unsharded_params"] = list(self.state["sharded_params"])
        else:
            self.state["unsharded_params"] = list(
                _execute_graph(
                    self.graph_callables.unshard,
                    list(self.state["sharded_params"]),
                )
            )

    def _accumulate_stage_unsharded_grads(self, grads: list[Any]) -> None:
        unsharded_grads = self.state["unsharded_grads"]
        grads_to_accumulate = grads[: len(unsharded_grads)]
        for index, grad in enumerate(grads_to_accumulate):
            if grad is None:
                continue
            if unsharded_grads[index] is None:
                unsharded_grads[index] = grad
            else:
                unsharded_grads[index] += grad

    def scale_grads(self, grad_scale_factor: int) -> None:
        if grad_scale_factor == 1:
            return
        for grad in self.state["unsharded_grads"]:
            if grad is not None:
                grad.div_(grad_scale_factor)


def _get_stage_from_action(
    action: _Action,
    ctx: _PipelineContext,
) -> tuple[_PipelineScheduleRuntime, dict[int, GraphPipelineStage], GraphPipelineStage]:
    schedule = ctx.schedule_ref
    assert isinstance(schedule, _PipelineScheduleRuntime)
    stage_index_to_stage = {
        stage.stage_index: cast(GraphPipelineStage, stage) for stage in schedule._stages
    }
    return schedule, stage_index_to_stage, stage_index_to_stage[action.stage_index]


def _prepare_fwd_common(action: _Action, ctx: _PipelineContext):
    schedule, stage_index_to_stage, stage = _get_stage_from_action(action, ctx)
    mb_index = action.microbatch_index
    assert mb_index is not None
    is_next_stage_on_this_rank = stage.stage_index + 1 in stage_index_to_stage
    is_prev_stage_on_this_rank = stage.stage_index - 1 in stage_index_to_stage
    if not stage.is_first and not is_prev_stage_on_this_rank:
        fwd_recv_ops = schedule.fwd_recv_ops
        assert (stage.stage_index, mb_index) in fwd_recv_ops
        _wait_batch_p2p(fwd_recv_ops.pop((stage.stage_index, mb_index)))
    return (
        schedule,
        stage_index_to_stage,
        stage,
        mb_index,
        is_next_stage_on_this_rank,
    )


def _prepare_fwd_user_args(
    stage: GraphPipelineStage,
    mb_index: int,
    ctx: _PipelineContext,
) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
    arg_mbs = ctx.arg_mbs
    kwarg_mbs = ctx.kwarg_mbs
    assert arg_mbs is not None and kwarg_mbs is not None
    kwargs = kwarg_mbs[mb_index]
    if stage.is_first:
        args = arg_mbs[mb_index]
    else:
        args = _normalize_model_output_as_tuple(
            stage._retrieve_recv_activations(mb_index)
        )
    target = ctx.target_mbs[mb_index] if stage.is_last and ctx.target_mbs else None
    return tuple(args), kwargs, target


def _flatten_stage_inputs(
    stage: GraphPipelineStage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
) -> list[Any]:
    state_flat, _ = pytree.tree_flatten(extract_module_state(stage.submod))
    if stage.is_last:
        flat_user_inputs, _ = pytree.tree_flatten(
            ((args, kwargs, target, loss_kwargs), {})
        )
    else:
        flat_user_inputs, _ = pytree.tree_flatten(((args, kwargs), {}))
    flat_inputs, _ = _unwrap_subclasses([*state_flat, *flat_user_inputs])
    return flat_inputs


def _prepare_fwd_args_from_descs(
    stage: GraphPipelineStage,
    fwd_input_descs: tuple[Any, ...],
    flat_inputs: list[Any],
) -> list[Any]:
    unsharded_by_name = {
        desc.name: value
        for desc, value in zip(
            stage.graph_meta.unshard_output_descs,
            stage.state["unsharded_params"],
            strict=False,
        )
    }
    fw_args = []
    for desc in stage.graph_meta.fwd_input_descs:
        match = re.fullmatch(r"arg(\d+)_\d+", desc.name)
        if match is not None:
            input_index = int(match.group(1))
            if input_index >= len(flat_inputs):
                raise ValueError(
                    "GraphPP forward placeholder index is out of range: "
                    f"{desc.name} indexes {input_index}, but runtime has "
                    f"{len(flat_inputs)} flattened inputs"
                )
            fw_args.append(flat_inputs[input_index])
        elif desc.name in unsharded_by_name:
            fw_args.append(unsharded_by_name[desc.name])
        else:
            raise ValueError(f"Missing GraphPP forward input {desc.name}")
    return fw_args


def _prepare_fwd_graph_args(
    stage: GraphPipelineStage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
) -> list[Any]:
    stage._ensure_unsharded_params()
    return _prepare_fwd_args_from_descs(
        stage,
        stage.graph_meta.fwd_input_descs,
        _flatten_stage_inputs(stage, args, kwargs, target, loss_kwargs),
    )


def _prepare_fwd_fsdp_graph_args(
    stage: GraphPipelineStage,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: Any,
    loss_kwargs: dict[str, Any],
) -> list[Any]:
    assert stage.graph_meta is not None
    return _prepare_fwd_args_from_descs(
        stage,
        stage.graph_meta.partition.fwd_input_descs,
        _flatten_stage_inputs(stage, args, kwargs, target, loss_kwargs),
    )


def _post_fwd_common(
    stage: GraphPipelineStage,
    mb_index: int,
    output: Any,
    saved_intermediates: tuple[Any, ...],
    schedule: _PipelineScheduleRuntime,
    stage_index_to_stage: dict[int, GraphPipelineStage],
    ctx: _PipelineContext,
    is_next_stage_on_this_rank: bool,
) -> None:
    output_tuple = _normalize_model_output_as_tuple(output)
    if stage.is_last:
        stage.output_chunks.append(output)
        if ctx.losses is not None:
            ctx.losses.append(output)
        schedule._internal_losses.append(output)
    stage.fwd_cache[mb_index] = (output_tuple, saved_intermediates)
    if is_next_stage_on_this_rank:
        stage_index_to_stage[stage.stage_index + 1].set_local_fwd_input(
            output, mb_index
        )


def stage_forward(action: _Action, ctx: _PipelineContext) -> None:
    (
        schedule,
        stage_index_to_stage,
        stage,
        mb_index,
        is_next_stage_on_this_rank,
    ) = _prepare_fwd_common(action, ctx)
    args, kwargs, target = _prepare_fwd_user_args(stage, mb_index, ctx)
    loss_kwargs = getattr(schedule, "_graph_pp_loss_kwargs", {})
    stage.ensure_graphs(args, kwargs, target, loss_kwargs)
    assert stage.graph_callables is not None and stage.graph_meta is not None
    output, saved_intermediates = _run_fw_module(
        stage.graph_callables.fw,
        stage.graph_meta,
        _prepare_fwd_graph_args(stage, args, kwargs, target, loss_kwargs),
    )
    _post_fwd_common(
        stage,
        mb_index,
        output,
        saved_intermediates,
        schedule,
        stage_index_to_stage,
        ctx,
        is_next_stage_on_this_rank,
    )


def _prepare_backward_common(action: _Action, ctx: _PipelineContext):
    schedule, stage_index_to_stage, stage = _get_stage_from_action(action, ctx)
    mb_index = action.microbatch_index
    assert mb_index is not None
    is_next_stage_on_this_rank = stage.stage_index + 1 in stage_index_to_stage
    is_prev_stage_on_this_rank = stage.stage_index - 1 in stage_index_to_stage
    if not stage.is_last and not is_next_stage_on_this_rank:
        bwd_recv_ops = schedule.bwd_recv_ops
        assert (stage.stage_index, mb_index) in bwd_recv_ops
        _wait_batch_p2p(bwd_recv_ops.pop((stage.stage_index, mb_index)))
    schedule.backward_counter[stage.stage_index] += 1
    return schedule, stage_index_to_stage, stage, mb_index, is_prev_stage_on_this_rank


def _prepare_backward_args(stage: GraphPipelineStage, mb_index: int) -> list[Any]:
    _, saved_intermediates = stage.fwd_cache.pop(mb_index)
    assert stage.graph_meta is not None
    if stage.is_last:
        runtime_bwd_inputs: tuple[Any, ...] = ()
    else:
        runtime_bwd_inputs = _normalize_model_output_as_tuple(
            stage._retrieve_recv_grads(mb_index)
        )
    saved_by_name = {
        desc.name: value
        for desc, value in zip(
            stage.graph_meta.partition.saved_for_backward_descs,
            saved_intermediates,
            strict=True,
        )
    }
    runtime_by_name = {
        desc.name: value
        for desc, value in zip(
            stage.graph_meta.partition.bwd_runtime_input_descs,
            runtime_bwd_inputs,
            strict=True,
        )
    }
    bwd_args = []
    for desc in stage.graph_meta.partition.bwd_input_descs:
        if desc.name in saved_by_name:
            bwd_args.append(saved_by_name[desc.name])
        elif desc.name in runtime_by_name:
            bwd_args.append(runtime_by_name[desc.name])
        else:
            raise ValueError(f"Missing backward input {desc.name}")
    return bwd_args


def _post_backward_common(
    stage: GraphPipelineStage,
    mb_index: int,
    input_grads: list[Any],
    stage_index_to_stage: dict[int, GraphPipelineStage],
    is_prev_stage_on_this_rank: bool,
) -> None:
    stage.bwd_cache[mb_index] = tuple(input_grads)
    if is_prev_stage_on_this_rank:
        stage_index_to_stage[stage.stage_index - 1].set_local_bwd_input(
            stage.get_local_bwd_output(mb_index),
            mb_index,
        )


def _maybe_finish_last_backward(
    stage: GraphPipelineStage,
    schedule: _PipelineScheduleRuntime,
    *,
    last_backward: bool,
) -> None:
    if not last_backward:
        return
    stage.scale_grads(schedule._n_microbatches if schedule.scale_grads else 1)


def stage_full_backward(action: _Action, ctx: _PipelineContext) -> None:
    (
        schedule,
        stage_index_to_stage,
        stage,
        mb_index,
        is_prev_stage_on_this_rank,
    ) = _prepare_backward_common(action, ctx)
    if not stage.has_backward:
        return
    assert stage.graph_callables is not None and stage.graph_meta is not None
    last_backward = schedule.backward_counter[stage.stage_index] == schedule._n_microbatches
    input_grads, param_buffer_grads = _run_full_bw_module(
        stage.graph_callables.full_bw,
        stage.graph_meta,
        _prepare_backward_args(stage, mb_index),
    )
    stage._accumulate_stage_unsharded_grads(param_buffer_grads)
    _post_backward_common(
        stage,
        mb_index,
        input_grads,
        stage_index_to_stage,
        is_prev_stage_on_this_rank,
    )
    _maybe_finish_last_backward(stage, schedule, last_backward=last_backward)


def stage_backward_input(action: _Action, ctx: _PipelineContext) -> None:
    _, _, stage = _get_stage_from_action(action, ctx)
    assert stage.graph_callables is not None
    if stage.graph_callables.bw_dI is None:
        logger.debug("GraphPP skipping BACKWARD_INPUT for stage %s", stage.stage_index)
        return
    (
        schedule,
        stage_index_to_stage,
        stage,
        mb_index,
        is_prev_stage_on_this_rank,
    ) = _prepare_backward_common(action, ctx)
    if not stage.has_backward:
        return
    assert stage.graph_meta is not None and stage.graph_callables.bw_dI is not None
    input_grads, activations = _run_dI_bw_module(
        stage.graph_callables.bw_dI,
        stage.graph_meta,
        _prepare_backward_args(stage, mb_index),
    )
    stage.bwd_activation_cache[mb_index] = tuple(activations)
    _post_backward_common(
        stage,
        mb_index,
        input_grads,
        stage_index_to_stage,
        is_prev_stage_on_this_rank,
    )


def stage_backward_weight(action: _Action, ctx: _PipelineContext) -> None:
    schedule, _, stage = _get_stage_from_action(action, ctx)
    mb_index = action.microbatch_index
    assert mb_index is not None
    assert stage.graph_callables is not None and stage.graph_meta is not None
    if stage.graph_callables.bw_dW is None:
        new_action = _Action(
            action.stage_index,
            FULL_BACKWARD,
            action.microbatch_index,
            action.sub_actions,
        )
        stage_full_backward(new_action, ctx)
        return
    last_backward = schedule.backward_counter[stage.stage_index] == schedule._n_microbatches
    if not stage.has_backward:
        return
    activations = stage.bwd_activation_cache.pop(mb_index)
    param_buffer_grads = _run_dW_bw_module(stage.graph_callables.bw_dW, list(activations))
    stage._accumulate_stage_unsharded_grads(param_buffer_grads)
    _maybe_finish_last_backward(stage, schedule, last_backward=last_backward)


def stage_unshard(action: _Action, ctx: _PipelineContext) -> None:
    _, _, stage = _get_stage_from_action(action, ctx)
    if stage.graph_callables is not None:
        stage._ensure_unsharded_params()


def stage_reshard(action: _Action, ctx: _PipelineContext) -> None:
    _, _, stage = _get_stage_from_action(action, ctx)
    stage.state["unsharded_params"] = []


def stage_reduce_grad(action: _Action, ctx: _PipelineContext) -> None:
    _, _, stage = _get_stage_from_action(action, ctx)
    assert stage.graph_callables is not None
    if stage.graph_callables.reduce_grad is None:
        stage.state["sharded_grads"] = list(stage.state["unsharded_grads"])
    else:
        stage.state["sharded_grads"] = list(
            _execute_graph(
                stage.graph_callables.reduce_grad,
                list(stage.state["unsharded_grads"]),
            )
        )


class GraphPPRunner:
    def __init__(self, schedule: _PipelineScheduleRuntime) -> None:
        self.schedule = schedule
        self.schedule._has_backward = True
        for stage in schedule._stages:
            if not isinstance(stage, GraphPipelineStage):
                raise TypeError(
                    "GraphPPRunner requires GraphPipelineStage instances, got "
                    f"{type(stage).__name__}"
                )

    def _populate_stage_states(self, stage: GraphPipelineStage) -> None:
        sharded_params = []
        trainable_params = []
        for _, value in stage.submod.named_parameters(remove_duplicate=False):
            local_value = _local_tensor(value)
            sharded_params.append(local_value)
            if value.requires_grad:
                trainable_params.append(value)
        buffers = [
            _local_tensor(value)
            for _, value in stage.submod.named_buffers(remove_duplicate=False)
        ]
        stage.state["sharded_params"] = sharded_params
        stage.state["buffers"] = buffers
        stage.state["trainable_params"] = trainable_params
        stage.state["unsharded_grads"] = [None] * len(trainable_params)
        stage._graph_pp_fused_first_forward_done = False
        stage._graph_pp_fused_reduce_done = False

    def _ensure_reduced_grads(self, stage: GraphPipelineStage) -> None:
        if stage.state["sharded_grads"]:
            return
        if not any(grad is not None for grad in stage.state["unsharded_grads"]):
            return
        if stage.graph_callables is None:
            return
        if stage.graph_callables.reduce_grad is None:
            stage.state["sharded_grads"] = list(stage.state["unsharded_grads"])
        else:
            stage.state["sharded_grads"] = list(
                _execute_graph(
                    stage.graph_callables.reduce_grad,
                    list(stage.state["unsharded_grads"]),
                )
            )

    def _accumulate_stage_sharded_grads(self, stage: GraphPipelineStage) -> None:
        self._ensure_reduced_grads(stage)
        for param, grad in zip(
            stage.state["trainable_params"],
            stage.state["sharded_grads"],
            strict=False,
        ):
            if grad is None:
                continue
            if isinstance(param, DTensor):
                spec = param._spec
                grad = DTensor.from_local(
                    grad,
                    device_mesh=spec.device_mesh,
                    placements=spec.placements,
                    shape=spec.shape,
                    stride=spec.stride,
                )
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad

    def step(self, *args, **kwargs) -> None:
        loss_kwargs = kwargs.get("loss_kwargs") or {}
        setattr(self.schedule, "_graph_pp_loss_kwargs", loss_kwargs)
        for stage in self.schedule._stages:
            self._populate_stage_states(cast(GraphPipelineStage, stage))
        self.schedule.step(*args, **kwargs)
        for stage in self.schedule._stages:
            graph_stage = cast(GraphPipelineStage, stage)
            self._accumulate_stage_sharded_grads(graph_stage)
            for key in graph_stage.state:
                graph_stage.state[key] = []

    def eval(self, *args, **kwargs):
        return self.schedule.eval(*args, **kwargs)


def register_graph_pp_schedule(schedule: _PipelineScheduleRuntime) -> GraphPPRunner:
    schedule.register_custom_function(FORWARD, stage_forward)
    schedule.register_custom_function(FULL_BACKWARD, stage_full_backward)
    schedule.register_custom_function(UNSHARD, stage_unshard)
    schedule.register_custom_function(RESHARD, stage_reshard)
    schedule.register_custom_function(REDUCE_GRAD, stage_reduce_grad)
    schedule.register_custom_function(BACKWARD_INPUT, stage_backward_input)
    schedule.register_custom_function(BACKWARD_WEIGHT, stage_backward_weight)
    return GraphPPRunner(schedule)
