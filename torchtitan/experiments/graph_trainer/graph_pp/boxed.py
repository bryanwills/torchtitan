# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Any

import torch.fx as fx


def execute_graph_boxed(graph: fx.GraphModule | Callable[[list[Any]], Any], boxed_args: list[Any]) -> Any:
    """Execute a GraphPP callable with boxed argument lifetime semantics.

    GraphPP schedules keep many per-microbatch activations alive.  Always
    executing through a mutable boxed input list lets the runner drop its input
    references immediately after the graph consumes them, instead of retaining a
    long-lived Python argument tuple.
    """
    try:
        if isinstance(graph, fx.GraphModule):
            return fx.Interpreter(graph).boxed_run(boxed_args)
        return graph(boxed_args)
    finally:
        boxed_args.clear()
