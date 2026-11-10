"""RapidOCR's torch networks as flat ``torch.fx`` graphs: the same kernels, less Python per forward.

A RapidOCR forward makes some 300 module calls (convolution, batch norm, activation, the
blocks around them). Each call goes through ``Module.__call__`` (hooks, tracing state), attribute
lookups and the module's own ``forward`` before it reaches the one torch function that queues a
kernel; on a GPU worker with few CPU cores that Python costs more than the GPU work.

:func:`inline` traces a network once into a ``torch.fx.GraphModule`` whose generated ``forward``
calls those torch functions directly, in the same order and with the same arguments, so every
kernel and every result is the one of the eager module. Batch norm (in eval mode) becomes the
``F.batch_norm`` call its ``forward`` makes. Shapes stay dynamic: shape arithmetic in the
networks is recorded as graph nodes, and anything a trace cannot represent fails the trace, in
which case the eager module is used.
"""

from __future__ import annotations

import linecache
import logging
import sys
from typing import Any

import torch
from torch import fx, nn
from torch.nn import functional as F  # noqa: N812

logger = logging.getLogger(__name__)


class _Inliner(fx.Tracer):
    """Traces through every module except batch norm, which is rewritten afterwards."""

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:  # noqa: ARG002
        return isinstance(m, nn.modules.batchnorm._BatchNorm)  # noqa: SLF001

    def to_bool(self, obj: fx.Proxy) -> bool:
        """Let ``assert`` statements on shapes pass (they check, they do not compute); refuse real branches."""
        frame = sys._getframe(1)  # noqa: SLF001
        while frame is not None and "/torch/fx/" in frame.f_code.co_filename.replace("\\", "/"):
            frame = frame.f_back
        if frame is not None:
            line = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
            if line.startswith("assert "):
                return True
        return super().to_bool(obj)


def _inline_batch_norm(gm: fx.GraphModule) -> None:
    """Replace eval-mode batch norm modules with the ``F.batch_norm`` call their ``forward`` makes."""
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "call_module":
            continue
        bn = gm.get_submodule(node.target)
        if not isinstance(bn, nn.modules.batchnorm._BatchNorm):  # noqa: SLF001
            continue
        if bn.training or bn.running_mean is None or bn.running_var is None:
            continue
        momentum = 0.0 if bn.momentum is None else bn.momentum
        with graph.inserting_before(node):
            args = [graph.get_attr(f"{node.target}.{name}") for name in ("running_mean", "running_var")]
            weight = graph.get_attr(f"{node.target}.weight") if bn.weight is not None else None
            bias = graph.get_attr(f"{node.target}.bias") if bn.bias is not None else None
            call = graph.call_function(F.batch_norm, (node.args[0], *args, weight, bias, False, momentum, bn.eps))
        node.replace_all_uses_with(call)
        graph.erase_node(node)
    graph.lint()
    gm.recompile()


def inline(net: Any) -> Any:  # noqa: ANN401
    """``net`` as a flat graph module with the same kernels, or ``net`` itself when it cannot be traced."""
    if not isinstance(net, nn.Module) or net.training:
        return net
    try:
        gm = fx.GraphModule(net, _Inliner().trace(net))
        _inline_batch_norm(gm)
    except Exception:  # noqa: BLE001
        logger.info("rapidocr-fast: %s runs eagerly (no fx graph)", type(net).__name__, exc_info=True)
        return net
    return gm.eval()


def same_outputs(a: Any, b: Any) -> bool:  # noqa: ANN401
    """Whether two network outputs (tensors, or dicts/lists/tuples of them) are equal, bit for bit."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_outputs(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(map(same_outputs, a, b))
    return a == b
