"""Scene-first VideoAnalysis tests."""

from __future__ import annotations

import numpy as np
import pytest

import videopython.ai.video_analysis as va
from videopython.base.description import (
    AudioClassification,
    AudioEvent,
    SceneBoundary,
)
from videopython.base.text.transcription import Transcription, TranscriptionSegment
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
        return f"scene_with_{len(frames)}_frames"


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


def _patch_scene_first_analyzers(monkeypatch: pytest.MonkeyPatch, *, failing_vlm: bool = False) -> None:
    monkeypatch.setattr(va, "AudioToText", _FakeAudioToText)
    monkeypatch.setattr(va, "SemanticSceneDetector", _FakeSceneDetector)
    monkeypatch.setattr(va, "SceneVLM", _FailingSceneVLM if failing_vlm else _FakeSceneVLM)
    monkeypatch.setattr(va, "AudioClassifier", _FakeAudioClassifier)


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
        caption="two people talking",
        audio_classification=AudioClassification(
            events=[AudioEvent(start=0.1, end=0.9, label="Speech", confidence=0.8)],
            clip_predictions={"Speech": 0.8},
        ),
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
    assert analysis.scenes.samples[0].caption is not None
    assert analysis.scenes.samples[0].audio_classification is not None
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
    assert analysis.scenes.samples[0].caption is None
    assert analysis.scenes.samples[0].audio_classification is None


def test_scene_payload_survives_vlm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scene_first_analyzers(monkeypatch, failing_vlm=True)
    config = va.VideoAnalysisConfig(enabled_analyzers={va.SEMANTIC_SCENE_DETECTOR, va.SCENE_VLM})
    analysis = va.VideoAnalyzer(config=config).analyze(_video_4s())

    assert analysis.scenes is not None
    assert len(analysis.scenes.samples) == 2
    assert analysis.scenes.samples[0].caption is None


def test_scene_vlm_produces_single_caption_with_scaled_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scene_first_analyzers(monkeypatch)
    config = va.VideoAnalysisConfig(enabled_analyzers={va.SCENE_VLM})
    analysis = va.VideoAnalyzer(config=config).analyze(_video_30s())

    assert analysis.scenes is not None
    assert len(analysis.scenes.samples) == 1
    # 30s scene: ceil(3 * ln(30/5 + 1)) = 6 frames
    assert analysis.scenes.samples[0].caption == "scene_with_6_frames"


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


def test_run_info_from_dict_tolerates_legacy_payload_without_timings() -> None:
    legacy = {
        "created_at": "2026-01-01T00:00:00Z",
        "mode": "path",
        "library_version": "0.26.8",
    }
    restored = va.AnalysisRunInfo.from_dict(legacy)
    assert restored.stage_durations_seconds == {}
    assert restored.total_duration_seconds is None


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
