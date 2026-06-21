"""AI-powered video effects that require object detection.

Effects here are real :class:`~videopython.editing.operation.Effect` subclasses
(shape-preserving, streamable) that physically live in ``videopython.ai`` so the
``videopython.editing`` layer keeps no AI dependency -- the same direction
``FaceTrackingCrop`` imports ``Operation``. The pixel work is delegated to the
AI-free renderer in :mod:`videopython.base.draw_detections`.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import Field, PrivateAttr

from videopython.ai.understanding.objects import ObjectDetector
from videopython.base.description import DetectedObject
from videopython.base.draw_detections import DetectionStyle, draw_detections
from videopython.editing.operation import Effect

__all__ = ["ObjectDetectionOverlay"]


class ObjectDetectionOverlay(Effect):
    """Detect objects per frame and overlay labelled bounding boxes.

    Runs a YOLOv8-COCO detector and composites tidy, colour-coded boxes with
    class labels (and optional confidence) onto every frame in the window.

    Detection runs on a ``detection_interval`` cadence in the streaming path and
    boxes are held between detections, so the cost is *compute*-bound, not
    *memory*-bound: ``"streamable"`` here means bounded memory, not bounded
    compute. On long clips, cap cost with ``window`` (limit the time range),
    a larger ``detection_interval``, a ``class_filter``, and/or the smaller
    ``model_size``. Only ``streaming_init`` and ``process_frame`` are
    overridden; the streaming engine drives that contract for bounded-memory
    execution.
    """

    op: Literal["object_detection_overlay"] = "object_detection_overlay"

    confidence_threshold: float = Field(0.5, ge=0, le=1, description="Minimum detection confidence to draw a box, 0-1.")
    class_filter: list[str] | None = Field(
        None,
        description='Only draw these COCO class names, e.g. ["person", "car", "dog"]. Null draws all classes.',
    )
    show_confidence: bool = Field(True, description="Append the detection confidence as a percentage to each label.")
    box_color: tuple[int, int, int] | None = Field(
        None,
        description="Fixed box color as [R, G, B] (0-255) for every box, or null for distinct per-class colors.",
    )
    line_thickness: float = Field(
        0.003,
        gt=0,
        le=0.05,
        description="Box stroke width as a fraction of the frame's longer side (0.003 = ~3px at 1080p).",
    )
    label_font_size: float = Field(
        0.022,
        gt=0,
        le=0.2,
        description="Label text height as a fraction of the frame's longer side (0.022 = ~24px at 1080p).",
    )
    detection_interval: int = Field(
        2,
        ge=1,
        description="Run detection every Nth frame and reuse the last result in between. Higher is faster.",
    )
    model_size: Literal["n", "s", "m"] = Field(
        "n",
        description=(
            "YOLOv8 model size: 'n' (nano, fastest), 's' (small), 'm' (medium, most accurate). "
            "Larger detects better but is slower."
        ),
    )
    backend: Literal["cpu", "gpu", "auto"] = Field(
        "auto",
        description="Detection device: 'cpu', 'gpu', or 'auto'.",
        json_schema_extra={"llm_hidden": True},
    )

    _detector: ObjectDetector | None = PrivateAttr(default=None)
    _last: list[DetectedObject] = PrivateAttr(default_factory=list)

    def _style(self) -> DetectionStyle:
        return DetectionStyle(
            box_color=self.box_color,
            line_thickness=self.line_thickness,
            show_confidence=self.show_confidence,
            label_font_size=self.label_font_size,
            min_confidence=self.confidence_threshold,
        )

    def _init_detector(self) -> None:
        """Build the detector lazily. Single patch point for tests."""
        if self._detector is None:
            self._detector = ObjectDetector(
                model_name=f"yolov8{self.model_size}.pt",
                confidence_threshold=self.confidence_threshold,
                class_filter=tuple(self.class_filter or ()),
                backend=self.backend,
            )

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int, **_context: Any) -> None:
        self._last = []
        self._init_detector()

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        if self._detector is None:
            self._init_detector()
        assert self._detector is not None
        # frame_index is 0-based within the effect's window, so frame 0 always
        # detects; intermediate frames reuse the last result.
        if frame_index % self.detection_interval == 0:
            self._last = self._detector.detect(frame)
        return draw_detections(frame, self._last, self._style())
