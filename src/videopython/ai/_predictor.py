"""Context-manager mixin for VRAM-releasing predictors.

Every predictor under ``videopython.ai`` already exposes an ``unload()`` method
that drops its model reference (``self._model = None``, plus any processor /
pipeline fields) and calls :func:`videopython.ai._device.release_device_memory`
to free the allocator cache. Today that has to be called by hand -- the dubbing
pipeline, for example, unloads each stage manually once it is done.

:class:`ManagedPredictor` makes that bookkeeping automatic by turning any such
predictor into a context manager::

    with SceneVLM(...) as vlm:
        ...  # use vlm
    # vlm.unload() has fired here, releasing VRAM

The mixin is deliberately tiny and dependency-free: it imports no torch /
transformers / ultralytics and holds no state, so it is safe to mix into any
predictor class regardless of how it is constructed.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from typing_extensions import Self


class ManagedPredictor:
    """Adds ``with``-statement support that calls ``unload()`` on exit.

    Subclasses MUST define their own ``unload()`` method; this mixin does not
    provide one. The expected contract is the one shared by every predictor in
    ``videopython.ai``: ``unload()`` clears the model reference(s) and releases
    device memory, and is safe to call more than once (including before the
    model was ever loaded).

    ``__exit__`` always returns ``False`` so exceptions raised inside the ``with``
    block are never suppressed -- ``unload()`` runs on both the success and
    failure paths.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.unload()  # type: ignore[attr-defined]
        return False
