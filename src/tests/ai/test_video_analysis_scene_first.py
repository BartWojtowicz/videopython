"""Scene-first VideoAnalysis tests."""

from __future__ import annotations

import numpy as np
import pytest

import videopython.ai.video_analysis as va
from videopython.base.description import (
    AudioClassification,
    AudioEvent,
    DetectedAction,
    DetectedObject,
    SceneBoundary,
)
from videopython.base.text.transcription import Transcription, TranscriptionSegment
from videopython.base.video import Video


class _FakeSceneResult:
    def __init__(self, caption: str):
        self.caption = caption
        self.objects = [DetectedObject(label="person", confidence=0.9)]
        self.text = ["SCORE", "score"]


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
        return _FakeSceneResult(caption=f"scene_with_{len(frames)}_frames")


class _FailingSceneVLM:
    def __init__(self, **_kwargs):
        pass

    def analyze_scene(self, _frames):
        raise RuntimeError("scene vlm failure")


class _FakeActionRecognizer:
    def __init__(self, **_kwargs):
        pass

    def recognize_path(self, _path, top_k=5, start_second=None, end_second=None):
        del top_k
        return [
            DetectedAction(
                label="run",
                confidence=0.8,
                start_time=start_second,
                end_time=end_second,
            )
        ]

    def recognize(self, _video):
        return [DetectedAction(label="run", confidence=0.8, start_time=0.2, end_time=0.8)]


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
    monkeypatch.setattr(va, "ActionRecognizer", _FakeActionRecognizer)
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
        visual_segments=[
            va.SceneVisualSegment(
                start_second=0.0,
                end_second=2.0,
                caption="two people talking",
                objects=[DetectedObject(label="person", confidence=0.9)],
                text=["EXIT"],
            )
        ],
        actions=[DetectedAction(label="talking", confidence=0.8, start_time=0.2, end_time=1.5)],
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
    assert analysis.scenes.samples[0].visual_segments
    assert analysis.scenes.samples[0].visual_segments[0].caption is not None
    assert analysis.scenes.samples[0].visual_segments[0].objects
    assert analysis.scenes.samples[0].visual_segments[0].text == ["SCORE"]
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
    assert analysis.scenes.samples[0].visual_segments == []
    assert not analysis.scenes.samples[0].actions
    assert analysis.scenes.samples[0].audio_classification is None


def test_scene_payload_survives_vlm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scene_first_analyzers(monkeypatch, failing_vlm=True)
    config = va.VideoAnalysisConfig(enabled_analyzers={va.SEMANTIC_SCENE_DETECTOR, va.SCENE_VLM})
    analysis = va.VideoAnalyzer(config=config).analyze(_video_4s())

    assert analysis.scenes is not None
    assert len(analysis.scenes.samples) == 2
    assert analysis.scenes.samples[0].visual_segments == []


def test_scene_vlm_outputs_chunk_level_visual_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scene_first_analyzers(monkeypatch)
    config = va.VideoAnalysisConfig(enabled_analyzers={va.SCENE_VLM})
    analysis = va.VideoAnalyzer(config=config).analyze(_video_30s())

    assert analysis.scenes is not None
    assert len(analysis.scenes.samples) == 1
    segments = analysis.scenes.samples[0].visual_segments
    assert len(segments) == 3
    assert [(item.start_second, item.end_second) for item in segments] == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 30.0),
    ]
