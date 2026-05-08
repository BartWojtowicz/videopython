"""Face detection and per-shot tracking for the understanding layer.

Lifted from ``ai/transforms.py`` so analysis code (``VideoAnalyzer``) and
transforms (``FaceTrackingCrop`` / ``SplitScreenComposite``) can share a
single source. M6 lip-sync also consumes this directly.

Tracking is IoU-only — no embedding re-id. Tracks do not survive across
shot/scene boundaries; a shot here means a ``SceneBoundary`` produced by
``SemanticSceneDetector``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from videopython.ai._device import select_device
from videopython.base.description import BoundingBox, DetectedFace, FaceTrack

logger = logging.getLogger(__name__)


# Hamming/IoU tunables. Module-level constants — same convention as the
# M1 voice-sample thresholds. Not user-facing; revisit if eval data
# shows tracks fragmenting on real footage.
DEFAULT_IOU_MATCH_THRESHOLD = 0.3
DEFAULT_MAX_MISSED_FRAMES = 3


class _FaceDetector:
    """Internal YOLOv8-face detector. Renamed from ``_FaceDetectionBackend``.

    Identical behaviour to the previous transforms-layer implementation —
    only the import path changed. Producers in ``transforms.py`` reach for
    this class via the lifted module path.
    """

    CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        min_face_size: int = 30,
        backend: Literal["cpu", "gpu", "auto"] = "auto",
    ):
        self.min_face_size = min_face_size
        self.backend: Literal["cpu", "gpu", "auto"] = backend
        self._resolved_device: Literal["cpu", "cuda"] | None = None
        self._yolo_model: Any = None

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

    def _init_yolo_face(self) -> None:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt",
        )
        self._yolo_model = YOLO(model_path)

        device = self._resolve_device()
        if device == "cuda":
            self._yolo_model.to("cuda")

    def _faces_from_yolo_result(self, result: Any) -> list[DetectedFace]:
        detected_faces: list[DetectedFace] = []
        boxes = result.boxes
        if boxes is None:
            return detected_faces

        img_h, img_w = result.orig_shape
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])

            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < self.min_face_size or face_h < self.min_face_size:
                continue

            detected_faces.append(
                DetectedFace(
                    bounding_box=BoundingBox(
                        x=x1 / img_w,
                        y=y1 / img_h,
                        width=face_w / img_w,
                        height=face_h / img_h,
                    ),
                    confidence=conf,
                )
            )
        detected_faces.sort(key=lambda f: f.area or 0, reverse=True)
        return detected_faces

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        if self._yolo_model is None:
            self._init_yolo_face()
        assert self._yolo_model is not None

        results = self._yolo_model(image, conf=self.CONFIDENCE_THRESHOLD, verbose=False)
        if not results:
            return []
        return self._faces_from_yolo_result(results[0])

    def detect_batch(self, images: list[np.ndarray] | np.ndarray) -> list[list[DetectedFace]]:
        if isinstance(images, np.ndarray):
            images = [images[i] for i in range(images.shape[0])] if images.ndim == 4 else [images]
        if not images:
            return []

        if self._yolo_model is None:
            self._init_yolo_face()
        assert self._yolo_model is not None

        results = self._yolo_model(images, conf=self.CONFIDENCE_THRESHOLD, verbose=False)
        return [self._faces_from_yolo_result(result) for result in results]


def _bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Standard IoU on normalized bounding boxes."""
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.width, b.y + b.height

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = (a.width * a.height) + (b.width * b.height) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


class FaceTracker:
    """Face tracking utility with per-frame smoothing and per-shot tracks.

    Two surfaces:

    - ``detect_and_track(frame, frame_index)`` / ``track_video(frames)`` —
      legacy single-subject API used by ``FaceTrackingCrop`` /
      ``SplitScreenComposite``. Returns a smoothed
      ``(cx, cy, w, h)`` tuple.
    - ``track_shot(frames, frame_indices)`` — new per-shot multi-track API
      returning ``list[FaceTrack]``. Used by the analysis pipeline (M5)
      and lip-sync (M6) to bind detections to subjects across the
      frames of one shot. IoU-only association — tracks do not survive
      across shot boundaries.
    """

    def __init__(
        self,
        selection_strategy: Literal["largest", "centered", "index"] = "largest",
        face_index: int = 0,
        smoothing: float = 0.8,
        detection_interval: int = 3,
        min_face_size: int = 30,
        backend: Literal["cpu", "gpu", "auto"] = "auto",
        sample_rate: int = 1,
        batch_size: int = 16,
        iou_match_threshold: float = DEFAULT_IOU_MATCH_THRESHOLD,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
    ):
        """Initialize face tracker.

        Args:
            selection_strategy: How to select which face to track (legacy
                single-subject API).
                - "largest": Track the face with the largest bounding box.
                - "centered": Track the face closest to frame center.
                - "index": Track the face at a specific index (sorted by area).
            face_index: Index of face to track when using "index" strategy.
            smoothing: Exponential moving average factor (0-1). Higher = smoother.
            detection_interval: Run detection every N frames, interpolate between.
            min_face_size: Minimum face size in pixels for detection.
            backend: Detection backend - "cpu", "gpu", or "auto".
            sample_rate: For GPU backend, detect every Nth frame and interpolate.
                Only used by track_video(). Default 1 (every frame).
            batch_size: Batch size for GPU detection. Default 16.
            iou_match_threshold: Minimum IoU between consecutive detections to
                continue an existing per-shot track. Used by ``track_shot``.
            max_missed_frames: How many consecutive frames a per-shot track
                can go without a detection before it's closed.
        """
        self.selection_strategy = selection_strategy
        self.face_index = face_index
        self.smoothing = smoothing
        self.detection_interval = detection_interval
        self.min_face_size = min_face_size
        self.backend: Literal["cpu", "gpu", "auto"] = backend
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.iou_match_threshold = iou_match_threshold
        self.max_missed_frames = max_missed_frames

        self._detector: _FaceDetector | None = None
        self._last_position: tuple[float, float] | None = None
        self._last_size: tuple[float, float] | None = None
        self._smoothed_position: tuple[float, float] | None = None
        self._smoothed_size: tuple[float, float] | None = None
        logger.info("FaceTracker initialized with backend=%s", self.backend)

    def _init_detector(self) -> None:
        """Initialize face detector lazily."""
        self._detector = _FaceDetector(
            min_face_size=self.min_face_size,
            backend=self.backend,
        )

    def _select_face(
        self,
        faces: list,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float, float, float] | None:
        """Select a face based on the configured strategy.

        Args:
            faces: List of DetectedFace objects.
            frame_width: Width of the frame.
            frame_height: Height of the frame.

        Returns:
            Tuple of (center_x, center_y, width, height) in normalized coords, or None.
        """
        if not faces:
            return None

        if self.selection_strategy == "largest":
            face = faces[0]
        elif self.selection_strategy == "centered":
            frame_center = (0.5, 0.5)
            face = min(
                faces,
                key=lambda f: (
                    (f.bounding_box.center[0] - frame_center[0]) ** 2
                    + (f.bounding_box.center[1] - frame_center[1]) ** 2
                ),
            )
        elif self.selection_strategy == "index":
            if self.face_index < len(faces):
                face = faces[self.face_index]
            else:
                face = faces[0]
        else:
            face = faces[0]

        bbox = face.bounding_box
        return (bbox.center[0], bbox.center[1], bbox.width, bbox.height)

    def detect_and_track(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> tuple[float, float, float, float] | None:
        """Detect face in frame and return smoothed position.

        Args:
            frame: Video frame as numpy array (H, W, 3).
            frame_index: Index of current frame.

        Returns:
            Tuple of (center_x, center_y, width, height) in normalized coords,
            or None if no face detected and no fallback available.
        """
        if self._detector is None:
            self._init_detector()
            assert self._detector is not None

        h, w = frame.shape[:2]

        if frame_index % self.detection_interval == 0:
            faces = self._detector.detect(frame)
            face_info = self._select_face(faces, w, h)
            if face_info is not None:
                self._last_position = (face_info[0], face_info[1])
                self._last_size = (face_info[2], face_info[3])
        elif self._last_position is not None and self._last_size is not None:
            face_info = (*self._last_position, *self._last_size)
        else:
            face_info = None

        return self._smooth(face_info)

    def _smooth(
        self,
        face_info: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        """Apply EMA smoothing, or replay the last smoothed value when no detection.

        Returns ``None`` when no detection has been seen yet.
        """
        if face_info is not None:
            cx, cy, fw, fh = face_info
            if self._smoothed_position is None:
                self._smoothed_position = (cx, cy)
                self._smoothed_size = (fw, fh)
            else:
                assert self._smoothed_size is not None
                alpha = 1 - self.smoothing
                self._smoothed_position = (
                    self._smoothed_position[0] * self.smoothing + cx * alpha,
                    self._smoothed_position[1] * self.smoothing + cy * alpha,
                )
                self._smoothed_size = (
                    self._smoothed_size[0] * self.smoothing + fw * alpha,
                    self._smoothed_size[1] * self.smoothing + fh * alpha,
                )
            return (*self._smoothed_position, *self._smoothed_size)

        if self._smoothed_position is not None and self._smoothed_size is not None:
            return (*self._smoothed_position, *self._smoothed_size)
        return None

    def reset(self) -> None:
        """Reset tracker state for a new video."""
        self._last_position = None
        self._last_size = None
        self._smoothed_position = None
        self._smoothed_size = None

    @staticmethod
    def _interpolate_bbox(
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float],
        t: float,
    ) -> tuple[float, float, float, float]:
        """Linearly interpolate between two bounding boxes."""
        return (
            bbox1[0] + (bbox2[0] - bbox1[0]) * t,
            bbox1[1] + (bbox2[1] - bbox1[1]) * t,
            bbox1[2] + (bbox2[2] - bbox1[2]) * t,
            bbox1[3] + (bbox2[3] - bbox1[3]) * t,
        )

    def track_video(
        self,
        frames: np.ndarray,
    ) -> list[tuple[float, float, float, float] | None]:
        """Track face through entire video using optimized batch detection.

        Optimized for GPU backends with frame sampling and interpolation
        for smooth tracking with reduced computation.

        Args:
            frames: Video frames array of shape (N, H, W, 3).

        Returns:
            List of face positions (cx, cy, w, h) for each frame, or None if
            no face detected and no fallback available.
        """
        if self._detector is None:
            self._init_detector()
            assert self._detector is not None

        n_frames = len(frames)
        if n_frames == 0:
            return []

        h, w = frames[0].shape[:2]

        execution_device_getter = getattr(self._detector, "execution_device", None)
        if callable(execution_device_getter):
            resolved = execution_device_getter()
            backend_execution_device = resolved if resolved in {"cpu", "cuda"} else None
        else:
            backend_execution_device = None
        if backend_execution_device is None:
            backend_execution_device = "cuda" if self.backend == "gpu" else "cpu"

        use_sampled_interpolation = self.sample_rate > 1 and backend_execution_device == "cuda"

        if use_sampled_interpolation:
            sample_indices = list(range(0, n_frames, self.sample_rate))
            if sample_indices[-1] != n_frames - 1:
                sample_indices.append(n_frames - 1)
        else:
            sample_indices = list(range(n_frames))

        sampled_frames = [frames[i] for i in sample_indices]

        sampled_detections: list[list] = []
        for batch_start in range(0, len(sampled_frames), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(sampled_frames))
            batch = sampled_frames[batch_start:batch_end]
            batch_results = self._detector.detect_batch(batch)
            sampled_detections.extend(batch_results)

        sampled_faces: list[tuple[float, float, float, float] | None] = []
        for faces in sampled_detections:
            face_info = self._select_face(faces, w, h)
            sampled_faces.append(face_info)

        if not use_sampled_interpolation:
            self.reset()
            return [self._smooth(face_info) for face_info in sampled_faces]

        all_positions: list[tuple[float, float, float, float] | None] = [None] * n_frames

        for idx, sample_idx in enumerate(sample_indices):
            all_positions[sample_idx] = sampled_faces[idx]

        for i in range(len(sample_indices) - 1):
            start_idx = sample_indices[i]
            end_idx = sample_indices[i + 1]
            start_face = sampled_faces[i]
            end_face = sampled_faces[i + 1]

            if start_face is None and end_face is None:
                continue
            elif start_face is None:
                for j in range(start_idx, end_idx):
                    all_positions[j] = end_face
            elif end_face is None:
                for j in range(start_idx + 1, end_idx + 1):
                    all_positions[j] = start_face
            else:
                gap = end_idx - start_idx
                for j in range(start_idx + 1, end_idx):
                    t = (j - start_idx) / gap
                    all_positions[j] = self._interpolate_bbox(start_face, end_face, t)

        self.reset()
        return [self._smooth(face_info) for face_info in all_positions]

    def track_shot(
        self,
        frames: list[np.ndarray] | np.ndarray,
        frame_indices: list[int] | None = None,
    ) -> list[FaceTrack]:
        """Per-shot multi-track association via IoU.

        Detection is run on every input frame (caller is expected to have
        already chosen the sampling cadence -- the analysis pipeline
        passes one frame per scene-VLM sample, lip-sync passes every
        frame in the shot). Tracks are stitched together greedily by
        best IoU above ``iou_match_threshold``; tracks with no match for
        ``max_missed_frames`` consecutive frames are closed and won't
        accept future associations.

        Track ids are integers starting at 1 within this shot. They are
        **not** stable across shots — embedding re-id is deferred.

        Args:
            frames: Frames in the shot (list or stacked ndarray).
            frame_indices: Source-video frame indices. Defaults to
                ``range(len(frames))`` when omitted.

        Returns:
            List of ``FaceTrack`` objects, one per distinct subject
            tracked in the shot.
        """
        if isinstance(frames, np.ndarray):
            frame_list = [frames[i] for i in range(frames.shape[0])] if frames.ndim == 4 else [frames]
        else:
            frame_list = list(frames)

        if not frame_list:
            return []

        if frame_indices is None:
            frame_indices = list(range(len(frame_list)))
        if len(frame_indices) != len(frame_list):
            raise ValueError("frame_indices length must match frames length")

        if self._detector is None:
            self._init_detector()
            assert self._detector is not None

        per_frame_detections: list[list[DetectedFace]] = []
        for batch_start in range(0, len(frame_list), self.batch_size):
            batch = frame_list[batch_start : batch_start + self.batch_size]
            per_frame_detections.extend(self._detector.detect_batch(batch))

        active: list[_OpenTrack] = []
        finished: list[_OpenTrack] = []
        next_id = 1

        for relative_idx, faces in enumerate(per_frame_detections):
            absolute_idx = frame_indices[relative_idx]
            available = [face for face in faces if face.bounding_box is not None]
            assignments: dict[int, DetectedFace] = {}

            for track in active:
                best_face: DetectedFace | None = None
                best_iou = self.iou_match_threshold
                last_box = track.last_box
                if last_box is None:
                    continue
                for face in available:
                    if face in assignments.values() or face.bounding_box is None:
                        continue
                    iou = _bbox_iou(last_box, face.bounding_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_face = face
                if best_face is not None:
                    assignments[track.track_id] = best_face

            for track in active:
                if track.track_id in assignments:
                    face = assignments[track.track_id]
                    assert face.bounding_box is not None
                    track.frame_indices.append(absolute_idx)
                    track.boxes.append(face.bounding_box)
                    track.confidences.append(face.confidence)
                    track.last_box = face.bounding_box
                    track.missed = 0
                else:
                    track.missed += 1

            for face in available:
                if face in assignments.values() or face.bounding_box is None:
                    continue
                track = _OpenTrack(track_id=next_id, last_box=face.bounding_box)
                next_id += 1
                track.frame_indices.append(absolute_idx)
                track.boxes.append(face.bounding_box)
                track.confidences.append(face.confidence)
                active.append(track)

            still_active: list[_OpenTrack] = []
            for track in active:
                if track.missed > self.max_missed_frames:
                    finished.append(track)
                else:
                    still_active.append(track)
            active = still_active

        finished.extend(active)

        return [
            FaceTrack(
                track_id=track.track_id,
                frame_indices=track.frame_indices,
                boxes=track.boxes,
                confidences=track.confidences,
            )
            for track in finished
            if track.frame_indices
        ]


class _OpenTrack:
    """Mutable scratch state used by ``FaceTracker.track_shot``."""

    __slots__ = ("track_id", "last_box", "frame_indices", "boxes", "confidences", "missed")

    def __init__(self, track_id: int, last_box: BoundingBox):
        self.track_id = track_id
        self.last_box: BoundingBox | None = last_box
        self.frame_indices: list[int] = []
        self.boxes: list[BoundingBox] = []
        self.confidences: list[float] = []
        self.missed = 0


__all__ = [
    "FaceTracker",
    "_FaceDetector",
    "DEFAULT_IOU_MATCH_THRESHOLD",
    "DEFAULT_MAX_MISSED_FRAMES",
]
