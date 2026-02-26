"""Tests for VideoAnalysis serialization and bounded-memory orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import videopython.ai.video_analysis as va
from videopython.ai.understanding.detection import CombinedFrameAnalysis
from videopython.base.description import BoundingBox, DetectedObject
from videopython.base.video import VideoMetadata


def test_combined_frame_analysis_roundtrip_dict() -> None:
    """CombinedFrameAnalysis should roundtrip via to_dict/from_dict."""
    original = CombinedFrameAnalysis(
        detected_objects=[
            DetectedObject(
                label="person",
                confidence=0.91,
                bounding_box=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
            )
        ],
        detected_text=["EXIT"],
        face_count=2,
        shot_type="medium",
    )

    restored = CombinedFrameAnalysis.from_dict(original.to_dict())
    assert restored == original


def test_analyze_path_extracts_source_metadata(monkeypatch, tmp_path: Path) -> None:
    """Source metadata should include title, creation time normalization, and geo when available."""
    metadata = VideoMetadata(height=1080, width=1920, fps=30.0, frame_count=300, total_seconds=10.0)
    monkeypatch.setattr(va.VideoMetadata, "from_path", lambda _path: metadata)
    monkeypatch.setattr(
        va.VideoAnalyzer,
        "_extract_source_tags",
        lambda self, _path: {
            "title": "Vacation Clip",
            "creation_time": "2025-01-02T12:00:00+02:00",
            "com.apple.quicktime.location.iso6709": "+37.3317-122.0307+005.0/",
        },
    )

    analyzer = va.VideoAnalyzer(config=va.VideoAnalysisConfig(enabled_analyzers=set()))
    analysis = analyzer.analyze_path(tmp_path / "vacation.mp4")

    assert analysis.source.title == "Vacation Clip"
    assert analysis.source.width == 1920
    assert analysis.source.height == 1080
    assert analysis.source.duration == 10.0
    assert analysis.source.creation_time == "2025-01-02T10:00:00Z"
    assert analysis.source.geo is not None
    assert analysis.source.geo.latitude == 37.3317
    assert analysis.source.geo.longitude == -122.0307


def test_analyze_path_uses_chunked_sampling_without_video_from_path(monkeypatch, tmp_path: Path) -> None:
    """analyze_path should not load full videos into memory for frame analysis."""
    metadata = VideoMetadata(height=4, width=4, fps=10.0, frame_count=20, total_seconds=2.0)
    monkeypatch.setattr(va.VideoMetadata, "from_path", lambda _path: metadata)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Video.from_path should not be called by analyze_path")

    monkeypatch.setattr(va.Video, "from_path", fail_if_called)
    monkeypatch.setattr(va.VideoAnalyzer, "_extract_source_tags", lambda self, _path: {})

    def fake_extract_frames_at_indices(_path, frame_indices):
        frames = np.zeros((len(frame_indices), 4, 4, 3), dtype=np.uint8)
        for i in range(len(frame_indices)):
            frames[i, :, :, :] = i
        return frames

    monkeypatch.setattr(va, "extract_frames_at_indices", fake_extract_frames_at_indices)

    class FakeObjectDetector:
        def __init__(self, **_kwargs):
            pass

        def detect(self, _frame):
            return [
                DetectedObject(
                    label="person",
                    confidence=0.9,
                    bounding_box=BoundingBox(x=0.1, y=0.1, width=0.5, height=0.5),
                )
            ]

    class FakeCameraMotionDetector:
        def __init__(self, **_kwargs):
            pass

        def detect(self, _frame1, _frame2):
            return "pan"

    monkeypatch.setattr(va, "ObjectDetector", FakeObjectDetector)
    monkeypatch.setattr(va, "CameraMotionDetector", FakeCameraMotionDetector)

    config = va.VideoAnalysisConfig(
        enabled_analyzers={va.OBJECT_DETECTOR, va.CAMERA_MOTION_DETECTOR},
        optional_analyzers=set(),
        frame_sampling_mode="uniform",
        frames_per_second=1.0,
        max_frames=2,
        frame_chunk_size=1,
    )
    analyzer = va.VideoAnalyzer(config=config)

    analysis = analyzer.analyze_path(tmp_path / "sample.mp4")

    assert analysis.frames is not None
    assert analysis.frames.sampling.access_mode == "chunked"
    assert len(analysis.frames.samples) == 2
    assert analysis.frames.samples[0].objects
    assert analysis.steps[va.OBJECT_DETECTOR].status == "succeeded"
    assert analysis.steps[va.CAMERA_MOTION_DETECTOR].status == "succeeded"

    assert analysis.motion is not None
    assert len(analysis.motion.camera_motion_samples) == 1
    assert analysis.motion.camera_motion_samples[0].label == "pan"

    restored = va.VideoAnalysis.from_json(analysis.to_json())
    assert restored.frames is not None
    assert restored.steps[va.OBJECT_DETECTOR].status == "succeeded"
