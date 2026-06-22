"""General object detection for the understanding layer.

``ObjectDetector`` is the object-detection counterpart to the face detector in
``faces.py``: a lazy YOLOv8-COCO wrapper returning
:class:`~videopython.base.description.DetectedObject` with normalized bounding
boxes. Both share :class:`~videopython.ai.understanding._yolo.YoloDetector`
(lazy init, device selection, ``detect`` / ``detect_batch`` / ``unload``), so the
two stay one mental model. Consumed by
``videopython.ai.effects.ObjectDetectionOverlay``; usable directly for any
per-frame object analysis.
"""

from __future__ import annotations

import logging
from typing import Any

from videopython.ai.understanding._yolo import Backend, YoloDetector
from videopython.base.description import BoundingBox, DetectedObject

logger = logging.getLogger(__name__)

__all__ = ["ObjectDetector"]


class ObjectDetector(YoloDetector[DetectedObject]):
    """Lazy YOLOv8-COCO object detector returning normalized detections.

    The Ultralytics weights (default ``yolov8n.pt``) auto-download on first
    real use; class names come from the loaded model. Detection is gated by
    ``confidence_threshold`` and optionally restricted to ``class_filter``.
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    _FEATURE = "ObjectDetector"

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        class_filter: tuple[str, ...] = (),
        backend: Backend = "auto",
    ):
        """Initialize the detector.

        Args:
            model_name: Ultralytics COCO model id or path (e.g. ``yolov8n.pt``,
                ``yolov8s.pt``, ``yolov8m.pt``). Downloaded on first use.
            confidence_threshold: Minimum detection confidence in ``[0, 1]``.
            class_filter: If non-empty, only these COCO class names are kept.
            backend: Detection device - ``"cpu"``, ``"gpu"``, or ``"auto"``.
        """
        super().__init__(backend=backend)
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.class_filter = class_filter
        self._class_names: dict[int, str] = {}
        logger.info("ObjectDetector initialized with model=%s backend=%s", model_name, backend)

    def _conf(self) -> float:
        return self.confidence_threshold

    def _load_model(self) -> None:
        # No revision pin: ultralytics resolves/downloads this asset from its own
        # GitHub release assets, not a HF repo, and YOLO() takes no revision arg
        # (see videopython.ai._revisions module docstring).
        self._build_yolo(self.model_name)
        self._class_names = dict(self._yolo_model.names)

    def _parse(self, result: Any) -> list[DetectedObject]:
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
