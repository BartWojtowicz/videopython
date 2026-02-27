"""Tests for VideoAnalysis serialization and bounded-memory orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import videopython.ai.video_analysis as va
from videopython.base.description import BoundingBox, DetectedAction, DetectedObject, DetectedText, SceneBoundary
from videopython.base.video import VideoMetadata


def test_frame_analysis_sample_roundtrip_dict() -> None:
    """FrameAnalysisSample should roundtrip via to_dict/from_dict."""
    original = va.FrameAnalysisSample(
        timestamp=1.25,
        frame_index=30,
        image_caption="A person standing in a room",
        objects=[
            DetectedObject(
                label="person",
                confidence=0.91,
                bounding_box=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
            ),
        ],
        faces=[],
        text=["EXIT"],
        step_results={va.OBJECT_DETECTOR: "ok", va.IMAGE_TO_TEXT: "ok"},
    )

    restored = va.FrameAnalysisSample.from_dict(original.to_dict())
    assert restored == original


def test_video_analysis_roundtrip_json() -> None:
    """VideoAnalysis should roundtrip through JSON with nested sections."""
    original = va.VideoAnalysis(
        source=va.VideoAnalysisSource(
            title="clip",
            path="/tmp/clip.mp4",
            filename="clip.mp4",
            duration=2.0,
            fps=30.0,
            width=640,
            height=360,
            frame_count=60,
            creation_time="2026-01-01T00:00:00Z",
            raw_tags={"title": "clip"},
        ),
        config=va.VideoAnalysisConfig(
            enabled_analyzers={va.OBJECT_DETECTOR, va.TEXT_DETECTOR},
            action_scope="adaptive",
            max_action_scenes=4,
        ),
        run_info=va.AnalysisRunInfo(created_at="2026-01-01T00:00:00Z", mode="path", elapsed_seconds=0.1),
        steps={
            va.OBJECT_DETECTOR: va.AnalysisStepStatus(status="succeeded", duration_seconds=0.01),
            va.TEXT_DETECTOR: va.AnalysisStepStatus(status="succeeded", duration_seconds=0.01),
        },
        frames=va.FrameAnalysisSection(
            sampling=va.FrameSamplingReport(
                mode="uniform",
                frames_per_second=1.0,
                max_frames=10,
                sampled_indices=[0],
                sampled_timestamps=[0.0],
                access_mode="chunked",
                effective_max_frames=10,
            ),
            samples=[
                va.FrameAnalysisSample(
                    timestamp=0.0,
                    frame_index=0,
                    objects=[
                        DetectedObject(
                            label="person",
                            confidence=0.9,
                            bounding_box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.3),
                        )
                    ],
                    text=["exit"],
                    text_regions=[
                        DetectedText(
                            text="exit",
                            confidence=0.8,
                            bounding_box=BoundingBox(x=0.4, y=0.4, width=0.1, height=0.05),
                        )
                    ],
                )
            ],
        ),
        summary={"frame_sample_count": 1},
    )

    restored = va.VideoAnalysis.from_json(original.to_json())
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


def test_analyze_path_redacts_geo_from_raw_tags(monkeypatch, tmp_path: Path) -> None:
    """Geo redaction should remove location payload from raw tags."""
    metadata = VideoMetadata(height=1080, width=1920, fps=30.0, frame_count=300, total_seconds=10.0)
    monkeypatch.setattr(va.VideoMetadata, "from_path", lambda _path: metadata)
    monkeypatch.setattr(
        va.VideoAnalyzer,
        "_extract_source_tags",
        lambda self, _path: {
            "title": "Vacation Clip",
            "com.apple.quicktime.location.iso6709": "+37.3317-122.0307+005.0/",
            "creation_time": "2025-01-02T12:00:00+02:00",
        },
    )

    analyzer = va.VideoAnalyzer(
        config=va.VideoAnalysisConfig(enabled_analyzers=set(), redact_geo=True),
    )
    analysis = analyzer.analyze_path(tmp_path / "vacation.mp4")

    assert analysis.source.geo is None
    assert analysis.source.raw_tags is not None
    assert "com.apple.quicktime.location.iso6709" not in analysis.source.raw_tags
    assert analysis.source.raw_tags["title"] == "Vacation Clip"


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


def test_config_rejects_unknown_analyzer_ids() -> None:
    """Config should fail fast on unsupported analyzer ids."""
    with pytest.raises(ValueError, match="Unknown analyzer ids in enabled_analyzers"):
        va.VideoAnalysisConfig(enabled_analyzers={"not_a_real_analyzer"})


def test_action_scope_scene_runs_per_scene(monkeypatch, tmp_path: Path) -> None:
    """Scene action scope should run action recognition for each selected scene."""
    metadata = VideoMetadata(height=1080, width=1920, fps=30.0, frame_count=240, total_seconds=8.0)
    monkeypatch.setattr(va.VideoMetadata, "from_path", lambda _path: metadata)
    monkeypatch.setattr(va.VideoAnalyzer, "_extract_source_tags", lambda self, _path: {})

    class FakeSceneDetector:
        def __init__(self, **_kwargs):
            pass

        def detect_streaming(self, _path):
            return [
                SceneBoundary(start=0.0, end=4.0, start_frame=0, end_frame=120),
                SceneBoundary(start=4.0, end=8.0, start_frame=120, end_frame=240),
            ]

    calls: list[tuple[float | None, float | None]] = []

    class FakeActionRecognizer:
        def __init__(self, **_kwargs):
            pass

        def recognize_path(self, _path, top_k=5, start_second=None, end_second=None):
            calls.append((start_second, end_second))
            return [
                DetectedAction(
                    label=f"segment_{len(calls)}",
                    confidence=0.9,
                    start_time=start_second,
                    end_time=end_second,
                )
            ]

    monkeypatch.setattr(va, "SemanticSceneDetector", FakeSceneDetector)
    monkeypatch.setattr(va, "ActionRecognizer", FakeActionRecognizer)

    config = va.VideoAnalysisConfig(
        enabled_analyzers={va.SEMANTIC_SCENE_DETECTOR, va.ACTION_RECOGNIZER},
        action_scope="scene",
    )
    analyzer = va.VideoAnalyzer(config=config)

    analysis = analyzer.analyze_path(tmp_path / "sample.mp4")

    assert analysis.temporal is not None
    assert len(analysis.temporal.actions) == 2
    assert calls == [(0.0, 4.0), (4.0, 8.0)]


def test_memory_budget_limits_frame_sampling(monkeypatch, tmp_path: Path) -> None:
    """max_memory_mb should constrain effective sampled frame count."""
    metadata = VideoMetadata(height=100, width=100, fps=25.0, frame_count=100, total_seconds=4.0)
    monkeypatch.setattr(va.VideoMetadata, "from_path", lambda _path: metadata)
    monkeypatch.setattr(va.VideoAnalyzer, "_extract_source_tags", lambda self, _path: {})

    analyzer = va.VideoAnalyzer(
        config=va.VideoAnalysisConfig(
            enabled_analyzers=set(),
            frame_sampling_mode="uniform",
            frames_per_second=25.0,
            max_frames=None,
            max_memory_mb=1,
        )
    )
    analysis = analyzer.analyze_path(tmp_path / "sample.mp4")

    assert analysis.frames is not None
    assert analysis.frames.sampling.effective_max_frames is not None
    assert len(analysis.frames.sampling.sampled_indices) <= analysis.frames.sampling.effective_max_frames


def test_rich_understanding_preset_includes_full_coverage() -> None:
    """Rich preset should enable all analyzers and include semantic captioning by default."""
    config = va.VideoAnalysisConfig.rich_understanding_preset()
    assert set(va.ALL_ANALYZER_IDS).issubset(config.enabled_analyzers)
    assert va.IMAGE_TO_TEXT in config.enabled_analyzers
    assert config.frames_per_second >= 1.0
