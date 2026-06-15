"""General object detection for the understanding layer.

``ObjectDetector`` is the object-detection counterpart to the face detector in
``faces.py``: a lazy YOLOv8-COCO wrapper returning
:class:`~videopython.base.description.DetectedObject` with normalized bounding
boxes. It mirrors ``_FaceDetector`` (lazy init, device selection, ``detect`` /
``detect_batch``) so the two share one mental model. Consumed by
``videopython.ai.effects.ObjectDetectionOverlay``; usable directly for any
per-frame object analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from videopython.ai._device import select_device
from videopython.base.description import BoundingBox, DetectedObject

logger = logging.getLogger(__name__)

__all__ = ["ObjectDetector"]


class ObjectDetector:
    """Lazy YOLOv8-COCO object detector returning normalized detections.

    The Ultralytics weights (default ``yolov8n.pt``) auto-download on first
    real use; class names come from the loaded model. Detection is gated by
    ``confidence_threshold`` and optionally restricted to ``class_filter``.
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        class_filter: tuple[str, ...] = (),
        backend: Literal["cpu", "gpu", "auto"] = "auto",
    ):
        """Initialize the detector.

        Args:
            model_name: Ultralytics COCO model id or path (e.g. ``yolov8n.pt``,
                ``yolov8s.pt``, ``yolov8m.pt``). Downloaded on first use.
            confidence_threshold: Minimum detection confidence in ``[0, 1]``.
            class_filter: If non-empty, only these COCO class names are kept.
            backend: Detection device - ``"cpu"``, ``"gpu"``, or ``"auto"``.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.class_filter = class_filter
        self.backend: Literal["cpu", "gpu", "auto"] = backend
        self._resolved_device: Literal["cpu", "cuda"] | None = None
        self._yolo_model: Any = None
        self._class_names: dict[int, str] = {}
        logger.info("ObjectDetector initialized with model=%s backend=%s", model_name, backend)

    def _resolve_device(self) -> Literal["cpu", "cuda"]:
        if self._resolved_device is not None:
            return self._resolved_device

        if self.backend == "cpu":
            self._resolved_device = "cpu"
            return self._resolved_device

        if self.backend == "gpu":
            resolved = select_device(None, mps_allowed=False)
            if resolved != "cuda":
                raise ValueError("GPU backend requested but CUDA is not available.")
            self._resolved_device = "cuda"
            return self._resolved_device

        resolved_auto = select_device(None, mps_allowed=False)
        self._resolved_device = "cuda" if resolved_auto == "cuda" else "cpu"
        return self._resolved_device

    def execution_device(self) -> Literal["cpu", "cuda"]:
        """Resolved execution device for this detector."""
        return self._resolve_device()

    def _init_yolo(self) -> None:
        from videopython.ai._optional import require

        YOLO = require("ultralytics", "vision", feature="ObjectDetector").YOLO

        # No revision pin: ultralytics resolves/downloads this asset from its own
        # GitHub release assets, not a HF repo, and YOLO() takes no revision arg
        # (see videopython.ai._revisions module docstring).
        self._yolo_model = YOLO(self.model_name)
        self._class_names = dict(self._yolo_model.names)

        if self._resolve_device() == "cuda":
            self._yolo_model.to("cuda")

    def _objects_from_yolo_result(self, result: Any) -> list[DetectedObject]:
        detected: list[DetectedObject] = []
        boxes = result.boxes
        if boxes is None:
            return detected

        img_h, img_w = result.orig_shape
        for i in range(len(boxes)):
            label = self._class_names.get(int(boxes.cls[i]), str(int(boxes.cls[i])))
            if self.class_filter and label not in self.class_filter:
                continue

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            detected.append(
                DetectedObject(
                    label=label,
                    confidence=float(boxes.conf[i]),
                    bounding_box=BoundingBox(
                        x=x1 / img_w,
                        y=y1 / img_h,
                        width=(x2 - x1) / img_w,
                        height=(y2 - y1) / img_h,
                    ),
                )
            )
        detected.sort(key=lambda d: d.confidence, reverse=True)
        return detected

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        """Detect objects in a single ``(H, W, 3)`` frame."""
        if self._yolo_model is None:
            self._init_yolo()
        assert self._yolo_model is not None

        results = self._yolo_model(image, conf=self.confidence_threshold, verbose=False)
        if not results:
            return []
        return self._objects_from_yolo_result(results[0])

    def detect_batch(self, images: list[np.ndarray] | np.ndarray) -> list[list[DetectedObject]]:
        """Detect objects in a batch of frames (list or stacked ``(N, H, W, 3)``)."""
        if isinstance(images, np.ndarray):
            images = [images[i] for i in range(images.shape[0])] if images.ndim == 4 else [images]
        if not images:
            return []

        if self._yolo_model is None:
            self._init_yolo()
        assert self._yolo_model is not None

        results = self._yolo_model(images, conf=self.confidence_threshold, verbose=False)
        return [self._objects_from_yolo_result(result) for result in results]
