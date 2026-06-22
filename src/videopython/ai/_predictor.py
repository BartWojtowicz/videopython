"""Context-manager base with a default VRAM-releasing ``unload()``.

Every predictor under ``videopython.ai`` holds a lazily-loaded model (and
sometimes a processor / pipeline / VAD field) on the selected device, and needs
to drop those references and free the allocator cache when done. That teardown
is mechanical and identical across predictors: set the model field(s) to
``None`` and call :func:`videopython.ai._device.release_device_memory`.

:class:`ManagedPredictor` provides that as a default ``unload()`` driven by two
class attributes -- ``_model_attrs`` (the fields holding model state) and
``_device_attr`` (the field holding the device) -- and turns any predictor into
a context manager::

    with SceneVLM(...) as vlm:
        ...  # use vlm
    # vlm.unload() has fired here, releasing VRAM

Subclasses with non-default model fields just declare ``_model_attrs``; those
whose teardown isn't "null the fields + release" (e.g. delegating to a client's
own ``unload()``) override ``unload()`` instead.

The base imports no torch / transformers / ultralytics (``release_device_memory``
defers its torch import), so it stays safe to mix into any predictor regardless
of how it is constructed.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Literal

from videopython.ai._device import release_device_memory

if TYPE_CHECKING:
    from typing_extensions import Self


class ManagedPredictor:
    """Adds a default ``unload()`` plus ``with``-statement support.

    ``unload()`` clears each attribute named in ``_model_attrs`` to ``None`` and
    releases the cache for the device named by ``_device_attr``. It is idempotent
    (safe before the model is loaded and on repeated calls). ``__exit__`` always
    returns ``False`` so exceptions inside the ``with`` block propagate --
    ``unload()`` runs on both the success and failure paths.
    """

    # Attributes holding model state, cleared to None on unload. Override per
    # predictor (e.g. ("_model", "_processor")).
    _model_attrs: tuple[str, ...] = ("_model",)
    # Attribute holding the resolved device passed to release_device_memory.
    _device_attr: str = "device"

    def unload(self) -> None:
        """Drop the model reference(s) and release device memory. Idempotent."""
        for attr in self._model_attrs:
            setattr(self, attr, None)
        release_device_memory(getattr(self, self._device_attr, None))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.unload()
        return False
