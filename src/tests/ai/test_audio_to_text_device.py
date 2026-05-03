"""Unit tests for AudioToText device selection and VAD-gated transcription."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import videopython.ai.understanding.audio as audio_mod
from videopython.base.audio import Audio, AudioMetadata


def test_audio_to_text_disables_mps_auto_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"mps_allowed": True}

    def fake_select_device(_requested, mps_allowed=False):
        called["mps_allowed"] = mps_allowed
        return "cpu"

    monkeypatch.setattr(audio_mod, "select_device", fake_select_device)

    transcriber = audio_mod.AudioToText(model_name="small", device=None)

    assert called["mps_allowed"] is False
    assert transcriber.device == "cpu"


class _FakeWhisperModel:
    """Stub Whisper model that records transcribe() calls without loading weights."""

    def __init__(self, n_mels: int = 80) -> None:
        self.dims = type("Dims", (), {"n_mels": n_mels})()
        self.device = "cpu"
        self.transcribe_calls: list[dict[str, Any]] = []
        self.detect_language_calls: list[Any] = []

    def transcribe(self, **kwargs: Any) -> dict[str, Any]:
        self.transcribe_calls.append(kwargs)
        return {"segments": [], "language": kwargs.get("language", "en")}

    def detect_language(self, mel: Any) -> tuple[Any, dict[str, float]]:
        self.detect_language_calls.append(mel)
        # Highest-prob language wins; tests can override via instance attribute.
        return (None, {"ja": 0.9, "en": 0.1})


@pytest.fixture
def cpu_transcriber(monkeypatch: pytest.MonkeyPatch) -> audio_mod.AudioToText:
    """AudioToText pinned to CPU, no real models loaded."""
    monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
    return audio_mod.AudioToText(model_name="small", device=None)


def _short_audio(duration_s: float = 1.0, sample_rate: int = 16000) -> Audio:
    """Low-amplitude noise so ``Audio.is_silent`` is False and the call reaches
    ``_transcribe_local``. Content doesn't matter — VAD/Whisper are stubbed."""
    rng = np.random.default_rng(0)
    samples = int(duration_s * sample_rate)
    data = (rng.standard_normal(samples) * 1e-3).astype(np.float32)
    metadata = AudioMetadata(
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
        duration_seconds=duration_s,
        frame_count=samples,
    )
    return Audio(data, metadata)


class TestVadShortCircuit:
    """When VAD finds no speech, return empty Transcription without calling Whisper."""

    def test_empty_voiced_spans_skips_whisper(
        self, cpu_transcriber: audio_mod.AudioToText, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_model = _FakeWhisperModel()

        def fake_init_local(self: Any) -> None:
            self._model = fake_model

        def fake_run_vad(self: Any, _audio: Audio) -> list[tuple[float, float]]:
            return []

        monkeypatch.setattr(audio_mod.AudioToText, "_init_local", fake_init_local)
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", fake_run_vad)

        result = cpu_transcriber.transcribe(_short_audio())

        assert result.segments == []
        assert fake_model.transcribe_calls == []
        assert fake_model.detect_language_calls == []


class TestLanguageDetectionThreading:
    """Detected language must be passed into Whisper's transcribe() call."""

    def test_plain_branch_passes_language(
        self, cpu_transcriber: audio_mod.AudioToText, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_model = _FakeWhisperModel()

        def fake_init_local(self: Any) -> None:
            self._model = fake_model

        def fake_run_vad(self: Any, _audio: Audio) -> list[tuple[float, float]]:
            return [(0.0, 1.0)]

        def fake_detect_language(self: Any, _audio: Audio, _spans: list[tuple[float, float]]) -> str:
            return "ja"

        monkeypatch.setattr(audio_mod.AudioToText, "_init_local", fake_init_local)
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", fake_run_vad)
        monkeypatch.setattr(audio_mod.AudioToText, "_detect_language", fake_detect_language)

        cpu_transcriber.transcribe(_short_audio())

        assert len(fake_model.transcribe_calls) == 1
        assert fake_model.transcribe_calls[0]["language"] == "ja"
        assert fake_model.transcribe_calls[0]["word_timestamps"] is True

    def test_diarization_branch_passes_language(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(enable_diarization=True, device=None)

        fake_model = _FakeWhisperModel()

        def fake_init_local(self: Any) -> None:
            self._model = fake_model

        def fake_init_diarization(self: Any) -> None:
            # Stand-in pipeline: returns an object with empty exclusive_speaker_diarization.
            class _EmptyAnnotation:
                def itertracks(self, yield_label: bool = True):
                    return iter([])

            class _DiarOutput:
                exclusive_speaker_diarization = _EmptyAnnotation()

            self._diarization_pipeline = lambda _payload: _DiarOutput()

        def fake_run_vad(self: Any, _audio: Audio) -> list[tuple[float, float]]:
            return [(0.0, 1.0)]

        def fake_detect_language(self: Any, _audio: Audio, _spans: list[tuple[float, float]]) -> str:
            return "ja"

        monkeypatch.setattr(audio_mod.AudioToText, "_init_local", fake_init_local)
        monkeypatch.setattr(audio_mod.AudioToText, "_init_diarization", fake_init_diarization)
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", fake_run_vad)
        monkeypatch.setattr(audio_mod.AudioToText, "_detect_language", fake_detect_language)

        transcriber.transcribe(_short_audio())

        assert len(fake_model.transcribe_calls) == 1
        assert fake_model.transcribe_calls[0]["language"] == "ja"


class TestVadDisabled:
    """enable_vad=False must reproduce pre-change behaviour exactly: no VAD run,
    no language= kwarg, no empty-audio short-circuit."""

    def test_no_vad_init_and_no_language_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(enable_vad=False, device=None)

        fake_model = _FakeWhisperModel()
        vad_init_calls = {"count": 0}

        def fake_init_local(self: Any) -> None:
            self._model = fake_model

        def fake_init_vad(self: Any) -> None:
            vad_init_calls["count"] += 1

        def fail_if_called(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("VAD helpers must not run when enable_vad=False")

        monkeypatch.setattr(audio_mod.AudioToText, "_init_local", fake_init_local)
        monkeypatch.setattr(audio_mod.AudioToText, "_init_vad", fake_init_vad)
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", fail_if_called)
        monkeypatch.setattr(audio_mod.AudioToText, "_detect_language", fail_if_called)

        transcriber.transcribe(_short_audio())

        assert vad_init_calls["count"] == 0
        assert len(fake_model.transcribe_calls) == 1
        assert "language" not in fake_model.transcribe_calls[0]


class TestDetectLanguageWindow:
    """_detect_language must build the mel from at most 30s of voiced audio."""

    def test_concatenates_and_caps_at_30s(
        self, cpu_transcriber: audio_mod.AudioToText, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import whisper

        # 60s of silence (16 kHz mono). _run_vad returns spans whose total
        # duration exceeds 30s; _detect_language must cap concatenation at
        # whisper.audio.N_SAMPLES (= 30s * 16000).
        sample_rate = 16000
        audio = Audio.silence(duration=60.0, sample_rate=sample_rate, channels=1)

        fake_model = _FakeWhisperModel(n_mels=80)
        cpu_transcriber._model = fake_model

        captured: dict[str, Any] = {}

        def stub_pad_or_trim(array: Any, length: int = whisper.audio.N_SAMPLES) -> Any:
            captured["pre_pad_samples"] = array.shape[0]
            captured["pad_target"] = length
            # Return a zero tensor of target length so the downstream mel stub
            # sees the post-pad shape.
            import torch

            return torch.zeros(length, dtype=array.dtype)

        def stub_log_mel(audio_tensor: Any, n_mels: int = 80) -> Any:
            captured["mel_input_samples"] = audio_tensor.shape[0]
            captured["n_mels_passed"] = n_mels
            import torch

            # Whisper's encoder consumes (n_mels, n_frames). Shape is
            # incidental for this stub; we only verify the call.
            return torch.zeros((n_mels, 3000), dtype=torch.float32)

        monkeypatch.setattr(whisper.audio, "pad_or_trim", stub_pad_or_trim)
        monkeypatch.setattr(whisper.audio, "log_mel_spectrogram", stub_log_mel)

        # Ten spans of 5s = 50s of "voiced" content; expect cap at 30s.
        spans = [(float(i * 5), float(i * 5 + 5)) for i in range(10)]
        result = cpu_transcriber._detect_language(audio, spans)

        # Concatenation must be capped at N_SAMPLES before pad_or_trim runs.
        assert captured["pre_pad_samples"] == whisper.audio.N_SAMPLES
        assert captured["pad_target"] == whisper.audio.N_SAMPLES
        assert captured["mel_input_samples"] == whisper.audio.N_SAMPLES
        assert captured["n_mels_passed"] == 80
        assert len(fake_model.detect_language_calls) == 1
        assert result == "ja"
