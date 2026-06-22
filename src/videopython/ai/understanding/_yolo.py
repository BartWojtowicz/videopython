"""Shared lazy YOLOv8 detector wrapper for the understanding layer.

``ObjectDetector`` (COCO objects) and ``_FaceDetector`` (faces) were nearly
identical YOLO wrappers: same ``cpu``/``gpu``/``auto`` device resolution, same
lazy load + ``.to("cuda")``, same ``detect`` / ``detect_batch`` shape, differing
only in how the model is loaded and how a result is parsed into detections.
:class:`YoloDetector` holds all the shared machinery; subclasses supply just
``_load_model`` (build ``self._yolo_model``), ``_parse`` (one result -> list of
detections), and ``_conf`` (the confidence passed to YOLO).

The heavy ``ultralytics`` import is deferred to :meth:`_build_yolo` (guarded by
``require``), so constructing a detector pulls in no torch until it actually runs.
"""

from __future__ import annotations

import logging
from typing import Any, Generic, Literal, TypeVar

import numpy as np

from videopython.ai._device import select_device
from videopython.ai._predictor import ManagedPredictor

logger = logging.getLogger(__name__)

T = TypeVar("T")
Backend = Literal["cpu", "gpu", "auto"]


class YoloDetector(ManagedPredictor, Generic[T]):
    """Lazy YOLOv8 wrapper: device resolution, lazy load, detect/detect_batch, unload.

    Context-managed via :class:`ManagedPredictor`; ``unload()`` drops the model
    and frees the resolved device. Subclass contract:
      * ``_FEATURE`` -- name surfaced in the ``[ai]``-extra ``ImportError``.
      * ``_load_model()`` -- set ``self._yolo_model`` (call :meth:`_build_yolo`)
        plus any label state.
      * ``_parse(result)`` -- map one ultralytics result to a list of detections.
      * ``_conf()`` -- the ``conf`` threshold passed to the model.
    """

    _FEATURE = "YoloDetector"
    _model_attrs = ("_yolo_model",)
    _device_attr = "_resolved_device"

    def __init__(self, *, backend: Backend = "auto") -> None:
        self.backend: Backend = backend
        self._resolved_device: Literal["cpu", "cuda"] | None = None
        self._yolo_model: Any = None

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

    def _build_yolo(self, model_path: str) -> None:
        """Construct ``self._yolo_model`` from a model id/path and move it to the device."""
        from videopython.ai._optional import require

        YOLO = require("ultralytics", feature=self._FEATURE).YOLO
        self._yolo_model = YOLO(model_path)
        if self._resolve_device() == "cuda":
            self._yolo_model.to("cuda")

    # --- subclass hooks -----------------------------------------------------
    def _load_model(self) -> None:
        raise NotImplementedError

    def _conf(self) -> float:
        raise NotImplementedError

    def _parse(self, result: Any) -> list[T]:
        raise NotImplementedError

    # --- public API ---------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._yolo_model is None:
            self._load_model()

    def detect(self, image: np.ndarray) -> list[T]:
        """Detect in a single ``(H, W, 3)`` frame."""
        self._ensure_loaded()
        results = self._yolo_model(image, conf=self._conf(), verbose=False)
        if not results:
            return []
        return self._parse(results[0])

    def detect_batch(self, images: list[np.ndarray] | np.ndarray) -> list[list[T]]:
        """Detect in a batch of frames (list or stacked ``(N, H, W, 3)``)."""
        if isinstance(images, np.ndarray):
            images = [images[i] for i in range(images.shape[0])] if images.ndim == 4 else [images]
        if not images:
            return []

        self._ensure_loaded()
        results = self._yolo_model(images, conf=self._conf(), verbose=False)
        return [self._parse(result) for result in results]
