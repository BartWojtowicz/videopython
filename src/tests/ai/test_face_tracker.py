"""Unit tests for ``FaceShotTracker.track_shot`` IoU association."""

from __future__ import annotations

import numpy as np

from videopython.ai.understanding import faces as faces_mod
from videopython.ai.understanding.faces import FaceShotTracker
from videopython.base.description import BoundingBox, DetectedFace


class _StaticDetector:
    """Returns a fixed sequence of per-frame detections without running YOLO."""

    def __init__(self, per_frame: list[list[DetectedFace]]):
        self._per_frame = per_frame

    def detect_batch(self, frames):
        n = frames.shape[0] if hasattr(frames, "shape") and not isinstance(frames, list) else len(frames)
        return [self._per_frame[i] for i in range(n)]


def _box(x: float, y: float, w: float = 0.1, h: float = 0.1) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=w, height=h)


def _face(box: BoundingBox, conf: float = 0.9) -> DetectedFace:
    return DetectedFace(bounding_box=box, confidence=conf)


def test_iou_helper_disjoint_zero() -> None:
    a = _box(0.0, 0.0, 0.1, 0.1)
    b = _box(0.5, 0.5, 0.1, 0.1)
    assert faces_mod._bbox_iou(a, b) == 0.0


def test_iou_helper_full_overlap_one() -> None:
    a = _box(0.0, 0.0, 0.2, 0.2)
    assert faces_mod._bbox_iou(a, a) == 1.0


def test_track_shot_associates_two_consecutive_frames(monkeypatch) -> None:
    tracker = FaceShotTracker()
    detector = _StaticDetector(
        [
            [_face(_box(0.10, 0.20))],
            [_face(_box(0.11, 0.21))],
        ]
    )
    monkeypatch.setattr(tracker, "_init_detector", lambda: setattr(tracker, "_detector", detector))

    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    tracks = tracker.track_shot(frames)

    assert len(tracks) == 1
    track = tracks[0]
    assert track.track_id == 1
    assert track.frame_indices == [0, 1]
    assert track.length == 2
    assert track.confidences == [0.9, 0.9]


def test_track_shot_starts_new_track_when_iou_below_threshold(monkeypatch) -> None:
    tracker = FaceShotTracker(iou_match_threshold=0.5)
    detector = _StaticDetector(
        [
            [_face(_box(0.10, 0.10))],
            [_face(_box(0.80, 0.80))],
        ]
    )
    monkeypatch.setattr(tracker, "_init_detector", lambda: setattr(tracker, "_detector", detector))

    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    tracks = tracker.track_shot(frames)

    assert len(tracks) == 2
    track_ids = sorted(t.track_id for t in tracks)
    assert track_ids == [1, 2]


def test_track_shot_preserves_two_distinct_subjects(monkeypatch) -> None:
    tracker = FaceShotTracker()
    detector = _StaticDetector(
        [
            [_face(_box(0.10, 0.10)), _face(_box(0.70, 0.70))],
            [_face(_box(0.11, 0.11)), _face(_box(0.71, 0.71))],
        ]
    )
    monkeypatch.setattr(tracker, "_init_detector", lambda: setattr(tracker, "_detector", detector))

    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    tracks = tracker.track_shot(frames)

    assert len(tracks) == 2
    for track in tracks:
        assert track.length == 2


def test_track_shot_closes_track_after_max_missed(monkeypatch) -> None:
    tracker = FaceShotTracker(max_missed_frames=1)
    detector = _StaticDetector(
        [
            [_face(_box(0.10, 0.10))],
            [],
            [],
            [_face(_box(0.10, 0.10))],
        ]
    )
    monkeypatch.setattr(tracker, "_init_detector", lambda: setattr(tracker, "_detector", detector))

    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    tracks = tracker.track_shot(frames)

    # First track survives one miss but gets closed by the second; the
    # later detection starts a fresh track id.
    assert len(tracks) == 2
    track_ids = sorted(t.track_id for t in tracks)
    assert track_ids == [1, 2]
