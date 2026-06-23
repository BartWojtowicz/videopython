"""Face detection and per-shot tracking for the understanding layer.

Lifted from ``ai/transforms.py`` so analysis code (``VideoAnalyzer``) and
transforms (``FaceTrackingCrop``) can share a single source. M6 lip-sync
also consumes this directly.

Tracking is IoU-only — no embedding re-id. Tracks do not survive across
shot/scene boundaries; a shot here means a ``SceneBoundary`` produced by
``SemanticSceneDetector``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import cv2
import numpy as np

from videopython.ai._predictor import ManagedPredictor
from videopython.ai._revisions import pinned
from videopython.ai.understanding._detector import DetectorBase
from videopython.base.description import BoundingBox, DetectedFace, FaceTrack

logger = logging.getLogger(__name__)

# OpenCV YuNet face detector (MIT). Pinned in ``_revisions.py``.
_YUNET_REPO = "opencv/face_detection_yunet"
_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"


# Hamming/IoU tunables. Module-level constants — same convention as the
# M1 voice-sample thresholds. Not user-facing; revisit if eval data
# shows tracks fragmenting on real footage.
DEFAULT_IOU_MATCH_THRESHOLD = 0.3
DEFAULT_MAX_MISSED_FRAMES = 3


class _FaceDetector(DetectorBase[DetectedFace]):
    """Internal OpenCV YuNet face detector over the shared :class:`DetectorBase`.

    Pulls the pinned ``opencv/face_detection_yunet`` ONNX checkpoint (MIT) and
    filters detections below ``min_face_size`` pixels. Runs on CPU via OpenCV DNN.
    Producers in ``transforms.py`` reach for this class via the lifted module path.
    """

    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.3
    TOP_K = 5000
    _FEATURE = "face tracking"
    _model_attrs = ("_yunet",)

    def __init__(self, min_face_size: int = 30):
        super().__init__(backend="cpu")  # YuNet runs on CPU via OpenCV DNN
        self.min_face_size = min_face_size
        self._yunet: Any = None
        self._input_size: tuple[int, int] | None = None  # (w, h) currently set on the model

    def execution_device(self) -> Literal["cpu", "cuda"]:
        """YuNet runs on CPU via OpenCV DNN."""
        return "cpu"

    def _load_model(self) -> None:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=_YUNET_REPO,
            filename=_YUNET_FILENAME,
            revision=pinned(_YUNET_REPO),
        )
        # input_size is a placeholder; setInputSize() per frame overrides it.
        self._yunet = cv2.FaceDetectorYN.create(  # type: ignore[attr-defined]
            model_path,
            "",
            (320, 320),
            self.CONFIDENCE_THRESHOLD,
            self.NMS_THRESHOLD,
            self.TOP_K,
        )
        self._input_size = (320, 320)

    def _infer(self, images: list[np.ndarray]) -> list[list[DetectedFace]]:
        # YuNet has no batch API; detect one frame at a time.
        return [self._detect_one(image) for image in images]

    def _detect_one(self, frame: np.ndarray) -> list[DetectedFace]:
        img_h, img_w = frame.shape[:2]
        size = (img_w, img_h)
        if self._input_size != size:
            self._yunet.setInputSize(size)
            self._input_size = size
        # videopython frames are RGB; OpenCV DNN expects BGR.
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, faces = self._yunet.detect(bgr)
        return self._parse(faces, img_w, img_h)

    def _parse(self, faces: np.ndarray | None, img_w: int, img_h: int) -> list[DetectedFace]:
        detected_faces: list[DetectedFace] = []
        if faces is None or len(faces) == 0:
            return detected_faces

        # YuNet rows: [x, y, w, h, 5x(lx, ly) landmarks, score]; coords are pixels.
        for row in faces:
            x, y, face_w, face_h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            conf = float(row[14])
            if face_w < self.min_face_size or face_h < self.min_face_size:
                continue

            detected_faces.append(
                DetectedFace(
                    bounding_box=BoundingBox(
                        x=x / img_w,
                        y=y / img_h,
                        width=face_w / img_w,
                        height=face_h / img_h,
                    ),
                    confidence=conf,
                )
            )
        detected_faces.sort(key=lambda f: f.area or 0, reverse=True)
        return detected_faces


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


class _FaceTrackerBase(ManagedPredictor):
    """Shared detector lifecycle for the face trackers.

    Both trackers lazily build one :class:`_FaceDetector` and release it through
    the :class:`ManagedPredictor` context-manager contract.
    """

    def __init__(self, *, min_face_size: int) -> None:
        self.min_face_size = min_face_size
        self._detector: _FaceDetector | None = None

    def _init_detector(self) -> None:
        """Initialize the face detector lazily."""
        self._detector = _FaceDetector(min_face_size=self.min_face_size)

    def _ensure_detector(self) -> _FaceDetector:
        if self._detector is None:
            self._init_detector()
        assert self._detector is not None
        return self._detector

    def unload(self) -> None:
        """Release the underlying face-detection model (idempotent)."""
        if self._detector is not None:
            self._detector.unload()


class FaceSmoothingTracker(_FaceTrackerBase):
    """Single-subject face tracker with EMA position smoothing.

    Selects one face per frame (``selection_strategy``) and returns a smoothed
    ``(cx, cy, w, h)`` tuple in normalized coords via ``detect_and_track`` /
    ``track_video``. Used by ``FaceTrackingCrop`` to drive a follow-the-speaker
    crop.
    """

    def __init__(
        self,
        selection_strategy: Literal["largest", "centered", "index"] = "largest",
        face_index: int = 0,
        smoothing: float = 0.8,
        detection_interval: int = 3,
        min_face_size: int = 30,
        batch_size: int = 16,
    ):
        """Initialize the smoothing tracker.

        Args:
            selection_strategy: Which face to track — "largest" (biggest box),
                "centered" (closest to frame center), or "index" (``face_index``).
            face_index: Index of face to track when using the "index" strategy.
            smoothing: Exponential moving average factor (0-1). Higher = smoother.
            detection_interval: Run detection every N frames, hold position between.
            min_face_size: Minimum face size in pixels for detection.
            batch_size: Frames per detection batch in ``track_video``. Default 16.
        """
        super().__init__(min_face_size=min_face_size)
        self.selection_strategy = selection_strategy
        self.face_index = face_index
        self.smoothing = smoothing
        self.detection_interval = detection_interval
        self.batch_size = batch_size
        self._last_position: tuple[float, float] | None = None
        self._last_size: tuple[float, float] | None = None
        self._smoothed_position: tuple[float, float] | None = None
        self._smoothed_size: tuple[float, float] | None = None
        logger.info("FaceSmoothingTracker initialized (detection_interval=%s)", self.detection_interval)

    def _select_face(
        self,
        faces: list[DetectedFace],
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
        faces_with_box = [(f, f.bounding_box) for f in faces if f.bounding_box is not None]
        if not faces_with_box:
            return None

        if self.selection_strategy == "largest":
            _, bbox = faces_with_box[0]
        elif self.selection_strategy == "centered":
            frame_center = (0.5, 0.5)
            _, bbox = min(
                faces_with_box,
                key=lambda fb: ((fb[1].center[0] - frame_center[0]) ** 2 + (fb[1].center[1] - frame_center[1]) ** 2),
            )
        elif self.selection_strategy == "index":
            idx = self.face_index if self.face_index < len(faces_with_box) else 0
            _, bbox = faces_with_box[idx]
        else:
            _, bbox = faces_with_box[0]

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

    def track_video(
        self,
        frames: np.ndarray,
    ) -> list[tuple[float, float, float, float] | None]:
        """Track the face through a whole clip via batched per-frame detection.

        Detection runs on every frame (the YuNet detector is CPU-only), then each
        frame's selected face is EMA-smoothed.

        Args:
            frames: Video frames array of shape (N, H, W, 3).

        Returns:
            List of face positions (cx, cy, w, h) for each frame, or None where
            no face was detected and no fallback was available.
        """
        if self._detector is None:
            self._init_detector()
            assert self._detector is not None

        n_frames = len(frames)
        if n_frames == 0:
            return []

        h, w = frames[0].shape[:2]

        detections: list[list[DetectedFace]] = []
        for batch_start in range(0, n_frames, self.batch_size):
            batch = [frames[i] for i in range(batch_start, min(batch_start + self.batch_size, n_frames))]
            detections.extend(self._detector.detect_batch(batch))

        faces = [self._select_face(frame_faces, w, h) for frame_faces in detections]
        self.reset()
        return [self._smooth(face_info) for face_info in faces]


class FaceShotTracker(_FaceTrackerBase):
    """Per-shot multi-track face association via IoU.

    Detects faces on every input frame and stitches them into ``FaceTrack``s
    greedily by best IoU. Tracks do not survive across shot boundaries
    (IoU-only association; no embedding re-id). Used by the video-analysis
    pipeline to bind detections to subjects within one shot.
    """

    def __init__(
        self,
        min_face_size: int = 30,
        batch_size: int = 16,
        iou_match_threshold: float = DEFAULT_IOU_MATCH_THRESHOLD,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
    ):
        """Initialize the per-shot tracker.

        Args:
            min_face_size: Minimum face size in pixels for detection.
            batch_size: Batch size for detection. Default 16.
            iou_match_threshold: Minimum IoU between consecutive detections to
                continue an existing track.
            max_missed_frames: Consecutive frames a track may go without a
                detection before it is closed.
        """
        super().__init__(min_face_size=min_face_size)
        self.batch_size = batch_size
        self.iou_match_threshold = iou_match_threshold
        self.max_missed_frames = max_missed_frames
        logger.info("FaceShotTracker initialized (min_face_size=%s)", self.min_face_size)

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
    """Mutable scratch state used by ``FaceShotTracker.track_shot``."""

    __slots__ = ("track_id", "last_box", "frame_indices", "boxes", "confidences", "missed")

    def __init__(self, track_id: int, last_box: BoundingBox):
        self.track_id = track_id
        self.last_box: BoundingBox | None = last_box
        self.frame_indices: list[int] = []
        self.boxes: list[BoundingBox] = []
        self.confidences: list[float] = []
        self.missed = 0


__all__ = [
    "FaceSmoothingTracker",
    "FaceShotTracker",
    "_FaceDetector",
    "DEFAULT_IOU_MATCH_THRESHOLD",
    "DEFAULT_MAX_MISSED_FRAMES",
]
