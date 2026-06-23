"""Shared lazy detector base for the understanding layer.

``ObjectDetector`` (COCO objects, D-FINE via transformers) and ``_FaceDetector``
(faces, OpenCV YuNet) share the same machinery: ``cpu``/``gpu``/``auto`` device
resolution, lazy load, the ``detect`` / ``detect_batch`` shape, and ``unload``.
:class:`DetectorBase` holds that machinery; subclasses supply just ``_load_model``
(populate the ``_model_attrs`` fields) and ``_infer`` (map a batch of frames to a
list of per-frame detections).

The two backends are completely different (transformers vs OpenCV DNN), so the
base imports no model library itself; each subclass defers its heavy import into
``_load_model`` (guarded by ``require`` where the dep lives in the ``[ai]`` extra).
"""

from __future__ import annotations

import logging
from typing import Generic, Literal, TypeVar

import numpy as np

from videopython.ai._device import select_device
from videopython.ai._predictor import ManagedPredictor

logger = logging.getLogger(__name__)

T = TypeVar("T")
Backend = Literal["cpu", "gpu", "auto"]


class DetectorBase(ManagedPredictor, Generic[T]):
    """Lazy detector base: device resolution, lazy load, detect/detect_batch, unload.

    Context-managed via :class:`ManagedPredictor`; ``unload()`` drops the model
    reference(s) named in ``_model_attrs`` and frees the resolved device. Subclass
    contract:
      * ``_FEATURE`` -- name surfaced in the ``[ai]``-extra ``ImportError``.
      * ``_model_attrs`` -- attribute(s) holding loaded model state; the first one
        doubles as the "is loaded" sentinel.
      * ``_load_model()`` -- populate the ``_model_attrs`` fields plus any label state.
      * ``_infer(images)`` -- map a list of frames to a list of per-frame detections.
    """

    _FEATURE = "detector"
    _device_attr = "_resolved_device"

    def __init__(self, *, backend: Backend = "auto") -> None:
        self.backend: Backend = backend
        self._resolved_device: Literal["cpu", "cuda"] | None = None

    def _resolve_device(self) -> Literal["cpu", "cuda"]:
        if self._resolved_device is not None:
            return self._resolved_device

        if self.backend == "cpu":
            self._resolved_device = "cpu"
        elif self.backend == "gpu":
            if select_device(None, mps_allowed=False) != "cuda":
                raise ValueError("GPU backend requested but CUDA is not available.")
            self._resolved_device = "cuda"
        else:
            self._resolved_device = "cuda" if select_device(None, mps_allowed=False) == "cuda" else "cpu"
        return self._resolved_device

    def execution_device(self) -> Literal["cpu", "cuda"]:
        """Resolved execution device for this detector."""
        return self._resolve_device()

    # --- subclass hooks -----------------------------------------------------
    def _load_model(self) -> None:
        raise NotImplementedError

    def _infer(self, images: list[np.ndarray]) -> list[list[T]]:
        raise NotImplementedError

    # --- public API ---------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if getattr(self, self._model_attrs[0], None) is None:
            self._load_model()

    def detect(self, image: np.ndarray) -> list[T]:
        """Detect in a single ``(H, W, 3)`` frame."""
        self._ensure_loaded()
        results = self._infer([image])
        return results[0] if results else []

    def detect_batch(self, images: list[np.ndarray] | np.ndarray) -> list[list[T]]:
        """Detect in a batch of frames (list or stacked ``(N, H, W, 3)``)."""
        if isinstance(images, np.ndarray):
            images = [images[i] for i in range(images.shape[0])] if images.ndim == 4 else [images]
        if not images:
            return []

        self._ensure_loaded()
        return self._infer(images)
