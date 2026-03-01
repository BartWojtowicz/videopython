"""Tests for VideoAnalysis serialization and bounded-memory orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import videopython.ai.video_analysis as va
from videopython.base.description import (
    AudioClassification,
    AudioEvent,
    BoundingBox,
    DetectedAction,
    DetectedObject,
    DetectedText,
    SceneBoundary,
)
from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord
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
    )

    restored = va.VideoAnalysis.from_json(original.to_json())
    assert restored == original
    assert restored.summary["frame_sample_count"] == 1


def test_build_summary_prioritizes_high_level_information() -> None:
    """Summary should provide a readable high-level overview of video content."""
    analyzer = va.VideoAnalyzer(config=va.VideoAnalysisConfig(enabled_analyzers=set()))

    temporal_section = va.TemporalAnalysisSection(
        scenes=[
            SceneBoundary(start=0.0, end=3.0, start_frame=0, end_frame=90),
            SceneBoundary(start=3.0, end=7.0, start_frame=90, end_frame=210),
        ],
        actions=[
            DetectedAction(label="running", confidence=0.93, start_time=0.8, end_time=2.9),
            DetectedAction(label="running", confidence=0.88, start_time=3.5, end_time=5.4),
            DetectedAction(label="jumping", confidence=0.81, start_time=5.8, end_time=6.4),
        ],
    )
    audio_section = va.AudioAnalysisSection(
        transcription=Transcription(
            segments=[
                TranscriptionSegment(start=0.4, end=2.2, text="welcome to the park race", words=[]),
                TranscriptionSegment(start=2.2, end=4.6, text="the runners are speeding up", words=[]),
            ]
        ),
        classification=AudioClassification(
            events=[
                AudioEvent(start=0.0, end=1.0, label="Speech", confidence=0.97),
                AudioEvent(start=1.2, end=6.5, label="Music", confidence=0.84),
            ],
            clip_predictions={"Speech": 0.76, "Music": 0.58},
        ),
    )
    motion_section = va.MotionAnalysisSection(
        camera_motion_samples=[
            va.CameraMotionSample(start=0.0, end=1.0, label="pan"),
            va.CameraMotionSample(start=1.0, end=2.0, label="pan"),
            va.CameraMotionSample(start=2.0, end=3.0, label="static"),
        ]
    )
    frame_samples = [
        va.FrameAnalysisSample(
            timestamp=0.8,
            frame_index=24,
            image_caption="A runner moving quickly through a city park",
            objects=[DetectedObject(label="person", confidence=0.95), DetectedObject(label="track", confidence=0.78)],
            text=["RACE", "START"],
        ),
        va.FrameAnalysisSample(
            timestamp=4.2,
            frame_index=126,
            image_caption="Several people sprinting near the finish line",
            objects=[DetectedObject(label="person", confidence=0.92), DetectedObject(label="crowd", confidence=0.8)],
            text=["FINISH"],
        ),
    ]

    summary = analyzer._build_summary(
        temporal_section=temporal_section,
        audio_section=audio_section,
        motion_section=motion_section,
        frame_samples=frame_samples,
    )

    assert isinstance(summary.get("overview"), str)
    assert "running" in summary["primary_actions"]
    assert "person" in summary["primary_subjects"]
    assert "speech/dialogue" in summary["audio_cues"]
    assert "running" in summary["topic_keywords"]
    assert summary["highlights"]
    assert any(item["summary"].startswith("Action:") for item in summary["highlights"])
    assert summary["pace"] in {"fast pacing", "moderate pacing", "slower pacing"}
    assert "transcript_full" in summary
    assert "runners are speeding up" in summary["transcript_full"]
    assert summary["transcript_reliability"]["is_reliable"] is True


def test_build_summary_with_no_signals_returns_fallback_overview() -> None:
    """Summary should still be readable when analyzers produce little/no signal."""
    analyzer = va.VideoAnalyzer(config=va.VideoAnalysisConfig(enabled_analyzers=set()))
    summary = analyzer._build_summary(
        temporal_section=va.TemporalAnalysisSection(),
        audio_section=va.AudioAnalysisSection(),
        motion_section=va.MotionAnalysisSection(),
        frame_samples=[],
    )

    assert summary["scene_count"] == 0
    assert summary["action_count"] == 0
    assert summary["frame_sample_count"] == 0
    assert summary["audio_events_count"] == 0
    assert summary["overview"].startswith("Limited analysis signals")
    assert "highlights" not in summary


def test_build_summary_omits_full_transcript_when_unreliable() -> None:
    """Low-quality transcript should not be expanded into full summary payload."""
    analyzer = va.VideoAnalyzer(config=va.VideoAnalysisConfig(enabled_analyzers=set()))
    audio_section = va.AudioAnalysisSection(
        transcription=Transcription(
            segments=[
                TranscriptionSegment(start=1.0, end=1.0, text="um um um", words=[]),
                TranscriptionSegment(start=0.5, end=0.6, text="uh", words=[]),
                TranscriptionSegment(start=2.0, end=2.0, text="noise", words=[]),
            ]
        )
    )
    summary = analyzer._build_summary(
        temporal_section=va.TemporalAnalysisSection(),
        audio_section=audio_section,
        motion_section=va.MotionAnalysisSection(),
        frame_samples=[],
    )

    assert "transcript_excerpt" in summary
    assert "transcript_full" not in summary
    assert summary["transcript_reliability"]["is_reliable"] is False


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


def test_video_analysis_editing_export_prunes_noise_and_keeps_signal() -> None:
    """Editing export should keep high-signal fields and remove raw-heavy sections."""
    analysis = va.VideoAnalysis(
        source=va.VideoAnalysisSource(
            title="fight_clip",
            filename="fight_clip.mp4",
            duration=10.0,
            fps=30.0,
            width=1280,
            height=720,
        ),
        config=va.VideoAnalysisConfig(enabled_analyzers=set()),
        run_info=va.AnalysisRunInfo(created_at="2026-01-01T00:00:00Z", mode="path"),
        audio=va.AudioAnalysisSection(
            transcription=Transcription(
                segments=[
                    TranscriptionSegment(
                        start=0.0,
                        end=2.0,
                        text="Round one starts now",
                        words=[
                            TranscriptionWord(start=0.0, end=0.4, word="Round"),
                            TranscriptionWord(start=0.4, end=0.8, word="one"),
                            TranscriptionWord(start=0.8, end=1.2, word="starts"),
                            TranscriptionWord(start=1.2, end=1.6, word="now"),
                        ],
                    )
                ]
            ),
            classification=AudioClassification(
                events=[
                    AudioEvent(start=0.0, end=2.0, label="Speech", confidence=0.95),
                    AudioEvent(start=2.0, end=6.0, label="Music", confidence=0.81),
                ],
                clip_predictions={"Speech": 0.9, "Music": 0.72, "Noise": 0.05},
            ),
        ),
        temporal=va.TemporalAnalysisSection(
            scenes=[
                SceneBoundary(start=0.0, end=5.0, start_frame=0, end_frame=150),
                SceneBoundary(start=5.0, end=10.0, start_frame=150, end_frame=300),
            ],
            actions=[
                DetectedAction(label="boxing", confidence=0.81, start_time=1.0, end_time=3.0),
                DetectedAction(label="dodgeball", confidence=0.12, start_time=6.0, end_time=7.0),
            ],
        ),
        motion=va.MotionAnalysisSection(
            camera_motion_samples=[
                va.CameraMotionSample(start=0.0, end=5.0, label="zoom"),
                va.CameraMotionSample(start=5.0, end=10.0, label="pan"),
            ],
        ),
        frames=va.FrameAnalysisSection(
            sampling=va.FrameSamplingReport(
                mode="uniform",
                frames_per_second=1.0,
                max_frames=10,
                sampled_indices=[0, 30, 60, 90],
                sampled_timestamps=[0.0, 1.0, 2.0, 3.0],
            ),
            samples=[
                va.FrameAnalysisSample(
                    timestamp=0.0,
                    frame_index=0,
                    image_caption="Two fighters in a cage",
                    objects=[
                        DetectedObject(label="person", confidence=0.88),
                        DetectedObject(label="tv", confidence=0.2),
                    ],
                    text_regions=[DetectedText(text="ROUND 1", confidence=0.72)],
                ),
                va.FrameAnalysisSample(
                    timestamp=1.0,
                    frame_index=30,
                    image_caption="Fighters trading punches",
                    objects=[DetectedObject(label="person", confidence=0.82)],
                    text_regions=[DetectedText(text="ROUND 1", confidence=0.76)],
                ),
                va.FrameAnalysisSample(
                    timestamp=2.0,
                    frame_index=60,
                    image_caption="Referee steps in",
                    objects=[DetectedObject(label="person", confidence=0.78)],
                    text_regions=[],
                ),
            ],
        ),
    )

    editing = analysis.filter(target="editing")

    assert editing.source.path is None
    assert editing.source.raw_tags is None
    assert editing.audio is not None and editing.audio.transcription is not None
    assert editing.audio.transcription.segments[0].text == "Round one starts now"
    assert editing.audio.transcription.segments[0].words == []
    assert editing.temporal is not None
    assert len(editing.temporal.actions) == 1
    assert editing.temporal.actions[0].label == "boxing"
    assert editing.frames is not None
    assert editing.frames.samples
    assert all(sample.step_results is None for sample in editing.frames.samples)
    assert all(obj.label != "tv" for sample in editing.frames.samples for obj in (sample.objects or []))
    assert editing.motion is not None
    assert editing.motion.video_motion == []
    assert editing.motion.motion_timeline == []
    assert "overview" in editing.summary


def test_editing_export_does_not_cap_high_quality_moments() -> None:
    """Editing export should include all high-quality moments without hard caps."""
    scenes = []
    actions = []
    for i in range(20):
        start = float(i * 5)
        end = start + 5.0
        scenes.append(SceneBoundary(start=start, end=end, start_frame=i * 150, end_frame=(i + 1) * 150))
        actions.append(
            DetectedAction(
                label=f"action_{i}",
                confidence=0.9,
                start_time=start + 1.0,
                end_time=start + 2.0,
            )
        )

    analysis = va.VideoAnalysis(
        source=va.VideoAnalysisSource(duration=100.0, fps=30.0),
        config=va.VideoAnalysisConfig(enabled_analyzers=set()),
        run_info=va.AnalysisRunInfo(created_at="2026-01-01T00:00:00Z", mode="path"),
        temporal=va.TemporalAnalysisSection(scenes=scenes, actions=actions),
    )

    editing = analysis.filter(target="editing")
    assert editing.temporal is not None
    assert len(editing.temporal.actions) == 20


def test_filter_rejects_non_editing_targets() -> None:
    """Only editing target should be supported for filtering."""
    analysis = va.VideoAnalysis(
        source=va.VideoAnalysisSource(duration=2.0, fps=30.0),
        config=va.VideoAnalysisConfig(enabled_analyzers=set()),
        run_info=va.AnalysisRunInfo(created_at="2026-01-01T00:00:00Z", mode="path"),
    )
    with pytest.raises(ValueError, match="Supported targets: editing"):
        analysis.filter(target="rich")


def test_editing_filter_preserves_motion_timeline_when_camera_samples_absent() -> None:
    """Editing filter should keep timeline motion signal if camera samples are unavailable."""
    analysis = va.VideoAnalysis(
        source=va.VideoAnalysisSource(duration=4.0, fps=30.0),
        config=va.VideoAnalysisConfig(enabled_analyzers=set()),
        run_info=va.AnalysisRunInfo(created_at="2026-01-01T00:00:00Z", mode="path"),
        motion=va.MotionAnalysisSection(
            motion_timeline=[
                va.MotionTimelineSample(
                    timestamp=1.0,
                    motion=va.MotionInfo(motion_type="pan", magnitude=0.4, raw_magnitude=8.0),
                ),
                va.MotionTimelineSample(
                    timestamp=2.0, motion=va.MotionInfo(motion_type="zoom", magnitude=0.5, raw_magnitude=10.0)
                ),
            ],
            camera_motion_samples=[],
        ),
    )

    editing = analysis.filter()
    assert editing.motion is not None
    assert editing.motion.camera_motion_samples == []
    assert len(editing.motion.motion_timeline) == 2
    assert editing.motion.motion_timeline[0].motion.motion_type == "pan"
    assert editing.motion.motion_timeline[1].motion.motion_type == "zoom"


def test_summary_is_computed_runtime_not_from_serialized_snapshot() -> None:
    """Serialized summary snapshots should be ignored in favor of runtime summary."""
    analysis = va.VideoAnalysis(
        source=va.VideoAnalysisSource(duration=6.0, fps=30.0),
        config=va.VideoAnalysisConfig(enabled_analyzers=set()),
        run_info=va.AnalysisRunInfo(created_at="2026-01-01T00:00:00Z", mode="path"),
        temporal=va.TemporalAnalysisSection(
            scenes=[SceneBoundary(start=0.0, end=6.0, start_frame=0, end_frame=180)],
            actions=[],
        ),
    )

    payload = analysis.to_dict()
    payload["summary"] = {"scene_count": 999, "overview": "stale summary"}
    restored = va.VideoAnalysis.from_dict(payload)

    assert restored.summary["scene_count"] == 1
    assert restored.summary["overview"] != "stale summary"
