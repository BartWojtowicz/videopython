"""Scene-first VideoAnalysis tests."""

from __future__ import annotations

import numpy as np
import pytest

import videopython.ai.video_analysis as va
from videopython.ai.video_analysis import analyzer as _analyzer
from videopython.ai.video_analysis import stages as _stages
from videopython.base.description import (
    AudioClassification,
    AudioEvent,
    BoundingBox,
    FaceTrack,
    SceneBoundary,
    SceneDescription,
)
from videopython.base.transcription import Transcription, TranscriptionSegment
from videopython.base.video import Video


class _FakeAudioToText:
    def __init__(self, **_kwargs):
        pass

    def transcribe(self, _media):
        return Transcription(
            segments=[
                TranscriptionSegment(start=0.0, end=1.0, text="first line", words=[]),
                TranscriptionSegment(start=1.0, end=2.0, text="second line", words=[]),
                TranscriptionSegment(start=2.0, end=4.0, text="third line", words=[]),
            ]
        )


class _FakeSceneDetector:
    def __init__(self, **_kwargs):
        pass

    def detect_streaming(self, _path):
        return [
            SceneBoundary(start=0.0, end=2.0, start_frame=0, end_frame=20),
            SceneBoundary(start=2.0, end=4.0, start_frame=20, end_frame=40),
        ]

    def detect(self, _video):
        return self.detect_streaming(None)


class _FakeSceneVLM:
    def __init__(self, **_kwargs):
        pass

    def analyze_scene(self, frames):
        return SceneDescription(
            caption=f"scene_with_{len(frames)}_frames",
            subjects=["test"],
            shot_type="medium",
        )


class _FailingSceneVLM:
    def __init__(self, **_kwargs):
        pass

    def analyze_scene(self, _frames):
        raise RuntimeError("scene vlm failure")


class _FakeAudioClassifier:
    def __init__(self, **_kwargs):
        pass

    def classify(self, _media):
        return AudioClassification(
            events=[AudioEvent(start=0.1, end=0.7, label="Speech", confidence=0.9)],
            clip_predictions={"Speech": 0.9},
        )


class _FakeFaceTracker:
    def __init__(self, **_kwargs):
        pass

    def track_shot(self, frames, frame_indices=None):
        if frame_indices is None:
            frame_indices = list(range(len(frames)))
        return [
            FaceTrack(
                track_id=1,
                frame_indices=list(frame_indices),
                boxes=[BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2) for _ in frame_indices],
                confidences=[0.9 for _ in frame_indices],
            )
        ]


def _patch_scene_first_analyzers(monkeypatch: pytest.MonkeyPatch, *, failing_vlm: bool = False) -> None:
    monkeypatch.setattr(_stages, "AudioToText", _FakeAudioToText)
    monkeypatch.setattr(_stages, "SemanticSceneDetector", _FakeSceneDetector)
    monkeypatch.setattr(_analyzer, "SceneVLM", _FailingSceneVLM if failing_vlm else _FakeSceneVLM)
    monkeypatch.setattr(_analyzer, "AudioClassifier", _FakeAudioClassifier)
    monkeypatch.setattr(_analyzer, "FaceTracker", _FakeFaceTracker)


def _video_4s() -> Video:
    frames = np.zeros((40, 6, 6, 3), dtype=np.uint8)
    return Video.from_frames(frames, fps=10)


def _video_30s() -> Video:
    frames = np.zeros((300, 6, 6, 3), dtype=np.uint8)
    return Video.from_frames(frames, fps=10)


def test_scene_analysis_sample_roundtrip_dict() -> None:
    sample = va.SceneAnalysisSample(
        scene_index=0,
        start_second=0.0,
        end_second=2.0,
        start_frame=0,
        end_frame=20,
        scene_description=SceneDescription(
            caption="two people talking",
            subjects=["person", "person"],
            shot_type="medium",
        ),
        audio_classification=AudioClassification(
            events=[AudioEvent(start=0.1, end=0.9, label="Speech", confidence=0.8)],
            clip_predictions={"Speech": 0.8},
        ),
        faces=[
            FaceTrack(
                track_id=1,
                frame_indices=[0, 1],
                boxes=[
                    BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
                    BoundingBox(x=0.12, y=0.11, width=0.2, height=0.2),
                ],
                confidences=[0.95, 0.93],
            )
        ],
    )

    restored = va.SceneAnalysisSample.from_dict(sample.to_dict())
    assert restored == sample


def test_config_defaults_and_rejects_unknown_ids() -> None:
    restored = va.VideoAnalysisConfig.from_dict({})
    assert restored.enabled_analyzers == set(va.ALL_ANALYZER_IDS)

    with pytest.raises(ValueError, match="Unknown analyzer ids"):
        va.VideoAnalysisConfig(enabled_analyzers={"unknown"})


def test_scene_first_full_run_outputs_scene_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scene_first_analyzers(monkeypatch)
    analysis = va.VideoAnalyzer(config=va.VideoAnalysisConfig()).analyze(_video_4s())

    assert analysis.scenes is not None
    assert len(analysis.scenes.samples) == 2
    assert analysis.audio is not None
    assert analysis.audio.transcription is not None
    assert [segment.text for segment in analysis.audio.transcription.segments] == [
        "first line",
        "second line",
        "third line",
    ]
    description = analysis.scenes.samples[0].scene_description
    assert description is not None
    assert description.caption.startswith("scene_with_")
    assert description.subjects == ["test"]
    assert description.shot_type == "medium"
    assert analysis.scenes.samples[0].audio_classification is not None
    assert analysis.scenes.samples[0].faces is not None
    assert not hasattr(analysis, "frames")
    payload = analysis.to_dict()
    assert "frames" not in payload
    assert "temporal" not in payload


def test_disabled_analyzers_do_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scene_first_analyzers(monkeypatch)
    config = va.VideoAnalysisConfig(enabled_analyzers={va.SEMANTIC_SCENE_DETECTOR})
    analysis = va.VideoAnalyzer(config=config).analyze(_video_4s())

    assert analysis.audio is None
    assert analysis.scenes is not None
    assert analysis.scenes.samples
    assert analysis.scenes.samples[0].scene_description is None
    assert analysis.scenes.samples[0].audio_classification is None
    assert analysis.scenes.samples[0].faces is None


def test_scene_payload_survives_vlm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scene_first_analyzers(monkeypatch, failing_vlm=True)
    config = va.VideoAnalysisConfig(enabled_analyzers={va.SEMANTIC_SCENE_DETECTOR, va.SCENE_VLM})
    analysis = va.VideoAnalyzer(config=config).analyze(_video_4s())

    assert analysis.scenes is not None
    assert len(analysis.scenes.samples) == 2
    assert analysis.scenes.samples[0].scene_description is None


def test_scene_vlm_produces_structured_output_with_scaled_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scene_first_analyzers(monkeypatch)
    config = va.VideoAnalysisConfig(enabled_analyzers={va.SCENE_VLM})
    analysis = va.VideoAnalyzer(config=config).analyze(_video_30s())

    assert analysis.scenes is not None
    assert len(analysis.scenes.samples) == 1
    description = analysis.scenes.samples[0].scene_description
    assert description is not None
    # 30s scene at sampling=medium: ceil(3 * ln(30/5 + 1)) = 6 frames; phash dedup
    # is a no-op on identical zero frames so the deduped list has 1 frame.
    assert description.caption == "scene_with_1_frames"


def test_low_sampling_reduces_frame_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scene_first_analyzers(monkeypatch)
    config = va.VideoAnalysisConfig(enabled_analyzers={va.SCENE_VLM})
    analyzer = va.VideoAnalyzer(config=config, sampling="low")
    assert analyzer.sampling == "low"
    assert analyzer._sampling_profile.max_frames == 8


def test_unknown_sampling_preset_rejected() -> None:
    with pytest.raises(ValueError, match="sampling must be one of"):
        va.VideoAnalyzer(sampling="ultra")  # type: ignore[arg-type]


def test_run_info_roundtrip_includes_stage_timings() -> None:
    info = va.AnalysisRunInfo(
        created_at="2026-05-03T00:00:00Z",
        mode="path",
        library_version="0.26.10",
        stage_durations_seconds={"whisper": 1.5, "scene_detection": 0.3},
        total_duration_seconds=2.7,
    )
    restored = va.AnalysisRunInfo.from_dict(info.to_dict())
    assert restored == info


def test_analyze_populates_stage_durations_for_parallel_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scene_first_analyzers(monkeypatch)
    analysis = va.VideoAnalyzer(config=va.VideoAnalysisConfig()).analyze(_video_4s())

    timings = analysis.run_info.stage_durations_seconds
    assert "whisper_and_scene_detection_parallel" in timings
    assert "whisper" in timings
    assert "scene_detection" in timings
    assert "scene_analysis" in timings
    assert "scene_vlm" in timings
    assert "audio_classification" in timings
    assert "face_tracker" in timings
    assert all(v > 0.0 for v in timings.values())
    assert analysis.run_info.total_duration_seconds is not None
    assert analysis.run_info.total_duration_seconds > 0.0


def test_analyze_records_sequential_stages_when_only_one_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scene_first_analyzers(monkeypatch)
    config = va.VideoAnalysisConfig(enabled_analyzers={va.SEMANTIC_SCENE_DETECTOR})
    analysis = va.VideoAnalyzer(config=config).analyze(_video_4s())

    timings = analysis.run_info.stage_durations_seconds
    assert "scene_detection" in timings
    assert "whisper" not in timings
    assert "whisper_and_scene_detection_parallel" not in timings
