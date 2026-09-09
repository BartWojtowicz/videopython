from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
_MISSING = object()


class _TokenGraph:
    """One graph for a decoder module's two-row, single-token forward pass.

    Prefill, CPU, training and other layouts retain their original forward path.
    The output buffer is consumed inside the decoder layer before its next call;
    this wrapper is deliberately limited to Chatterbox's MLP and RMSNorm modules.
    """

    def __init__(self, module: Any, owner: SpeechGraphs):
        self.module = module
        self.owner = owner
        self.original = module.forward
        self.previous = vars(module).get("forward", _MISSING)
        self.children = tuple(module.modules())[1:]
        self.state: tuple[Any, Any, Any] | None = None
        self.signature: Any = None

    def _capture(self, x: Any) -> tuple[Any, Any, Any]:
        import torch

        static = x.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.original(static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        if self.owner.pool is None:
            self.owner.pool = torch.cuda.graph_pool_handle()
        # Decoder layers execute serially in capture order. Each output remains
        # alive in its wrapper until unload, so later captures cannot reuse it.
        with torch.cuda.graph(graph, pool=self.owner.pool):
            output = self.original(static)
        return graph, static, output

    def __call__(self, x: Any) -> Any:
        import torch

        if (
            self.owner.disabled
            or self.module.training
            or torch.is_grad_enabled()
            or x.device.type != "cuda"
            or x.dtype != torch.float32
            or x.ndim != 3
            or x.shape[:2] != (2, 1)
            or not x.is_contiguous()
            or any(child._forward_hooks or child._forward_pre_hooks for child in self.children)
        ):
            return self.original(x)
        signature = (tuple(x.shape), x.dtype, x.device, tuple(x.stride()))
        if self.state is None:
            try:
                state = self._capture(x)
            except RuntimeError:
                # Release other graphs too, so a failed optimization cannot
                # keep their private pools resident while falling back.
                self.owner.close()
                logger.warning("CUDA speech graph capture unavailable; using the original forward path.")
            else:
                self.state = state
                self.signature = signature
            # Leave the exception scope before retrying: its traceback can
            # otherwise keep failed-capture tensors and their pool alive.
            if self.state is None:
                torch.cuda.empty_cache()
                return self.original(x)
        if signature != self.signature:
            return self.original(x)
        graph, static, output = self.state
        # Captured buffers may be inference tensors even when a later caller
        # uses no_grad. Mutations must retain the inference context in that case.
        with torch.inference_mode():
            static.copy_(x)
            graph.replay()
        return output


class SpeechGraphs:
    """Own decoder graphs for one loaded model and restore forwards on unload."""

    def __init__(self, model: Any):
        self.disabled = False
        self.pool: Any = None
        self._wrappers: list[_TokenGraph] = []
        transformer = getattr(getattr(model, "t3", None), "tfmr", None)
        for layer in getattr(transformer, "layers", ()):
            for name in ("mlp", "input_layernorm", "post_attention_layernorm"):
                module = getattr(layer, name, None)
                if module is None or type(module).__name__ not in {"LlamaMLP", "LlamaRMSNorm"}:
                    continue
                wrapper = _TokenGraph(module, self)
                self._wrappers.append(wrapper)
                module.forward = wrapper

    def close(self) -> None:
        self.disabled = True
        for wrapper in self._wrappers:
            if wrapper.previous is _MISSING:
                del wrapper.module.forward
            else:
                wrapper.module.forward = wrapper.previous
            wrapper.state = None
        self._wrappers.clear()
        self.pool = None
