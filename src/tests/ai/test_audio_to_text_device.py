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


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> _FakeWhisperModel:
    """Patch ``_init_local`` so it installs a stub Whisper model. Returns the
    stub so tests can inspect ``transcribe_calls`` / ``detect_language_calls``."""
    fake_model = _FakeWhisperModel()

    def fake_init_local(self: Any) -> None:
        self._model = fake_model

    monkeypatch.setattr(audio_mod.AudioToText, "_init_local", fake_init_local)
    return fake_model


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
        self,
        cpu_transcriber: audio_mod.AudioToText,
        fake_whisper: _FakeWhisperModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", lambda self, _a: [])

        result = cpu_transcriber.transcribe(_short_audio())

        assert result.segments == []
        assert fake_whisper.transcribe_calls == []
        assert fake_whisper.detect_language_calls == []


class TestLanguageDetectionThreading:
    """Detected language must be passed into Whisper's transcribe() call."""

    def test_plain_branch_passes_language(
        self,
        cpu_transcriber: audio_mod.AudioToText,
        fake_whisper: _FakeWhisperModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", lambda self, _a: [(0.0, 1.0)])
        monkeypatch.setattr(audio_mod.AudioToText, "_detect_language", lambda self, _a, _s: "ja")

        cpu_transcriber.transcribe(_short_audio())

        assert len(fake_whisper.transcribe_calls) == 1
        assert fake_whisper.transcribe_calls[0]["language"] == "ja"
        assert fake_whisper.transcribe_calls[0]["word_timestamps"] is True

    def test_diarization_branch_passes_language(
        self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(enable_diarization=True, device=None)

        def fake_init_diarization(self: Any) -> None:
            # Stand-in pipeline: returns an object with empty exclusive_speaker_diarization.
            class _EmptyAnnotation:
                def itertracks(self, yield_label: bool = True):
                    return iter([])

            class _DiarOutput:
                exclusive_speaker_diarization = _EmptyAnnotation()

            self._diarization_pipeline = lambda _payload: _DiarOutput()

        monkeypatch.setattr(audio_mod.AudioToText, "_init_diarization", fake_init_diarization)
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", lambda self, _a: [(0.0, 1.0)])
        monkeypatch.setattr(audio_mod.AudioToText, "_detect_language", lambda self, _a, _s: "ja")

        transcriber.transcribe(_short_audio())

        assert len(fake_whisper.transcribe_calls) == 1
        assert fake_whisper.transcribe_calls[0]["language"] == "ja"


class TestVadDisabled:
    """enable_vad=False must reproduce pre-change behaviour: no VAD run, and
    Whisper is invoked with language=None (its auto-detect default)."""

    def test_no_vad_init_and_language_is_none(
        self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(enable_vad=False, device=None)

        def fail_if_called(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("VAD helpers must not run when enable_vad=False")

        monkeypatch.setattr(audio_mod.AudioToText, "_init_vad", fail_if_called)
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", fail_if_called)
        monkeypatch.setattr(audio_mod.AudioToText, "_detect_language", fail_if_called)

        transcriber.transcribe(_short_audio())

        assert len(fake_whisper.transcribe_calls) == 1
        assert fake_whisper.transcribe_calls[0]["language"] is None


class TestAntiHallucinationKwargs:
    """The three Whisper anti-hallucination kwargs must reach both transcribe()
    call sites (plain and diarization branches), with new-default values when
    the caller doesn't override and with overridden values when they do."""

    EXPECTED_DEFAULTS = {
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
    }

    def test_defaults_forwarded_plain_branch(
        self,
        cpu_transcriber: audio_mod.AudioToText,
        fake_whisper: _FakeWhisperModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", lambda self, _a: [(0.0, 1.0)])
        monkeypatch.setattr(audio_mod.AudioToText, "_detect_language", lambda self, _a, _s: "ja")

        cpu_transcriber.transcribe(_short_audio())

        call = fake_whisper.transcribe_calls[0]
        for key, expected in self.EXPECTED_DEFAULTS.items():
            assert call[key] == expected, f"{key}: expected {expected!r}, got {call[key]!r}"

    def test_defaults_forwarded_diarization_branch(
        self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(enable_diarization=True, device=None)

        def fake_init_diarization(self: Any) -> None:
            class _EmptyAnnotation:
                def itertracks(self, yield_label: bool = True):
                    return iter([])

            class _DiarOutput:
                exclusive_speaker_diarization = _EmptyAnnotation()

            self._diarization_pipeline = lambda _payload: _DiarOutput()

        monkeypatch.setattr(audio_mod.AudioToText, "_init_diarization", fake_init_diarization)
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", lambda self, _a: [(0.0, 1.0)])
        monkeypatch.setattr(audio_mod.AudioToText, "_detect_language", lambda self, _a, _s: "ja")

        transcriber.transcribe(_short_audio())

        call = fake_whisper.transcribe_calls[0]
        for key, expected in self.EXPECTED_DEFAULTS.items():
            assert call[key] == expected

    def test_overrides_forwarded(self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(
            condition_on_previous_text=True,
            no_speech_threshold=0.85,
            logprob_threshold=-0.5,
            device=None,
        )
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", lambda self, _a: [(0.0, 1.0)])
        monkeypatch.setattr(audio_mod.AudioToText, "_detect_language", lambda self, _a, _s: "ja")

        transcriber.transcribe(_short_audio())

        call = fake_whisper.transcribe_calls[0]
        assert call["condition_on_previous_text"] is True
        assert call["no_speech_threshold"] == 0.85
        assert call["logprob_threshold"] == -0.5

    def test_logprob_threshold_none_is_forwarded(
        self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Whisper accepts None to disable the gate; the helper must not drop it.
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(logprob_threshold=None, device=None)
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", lambda self, _a: [(0.0, 1.0)])
        monkeypatch.setattr(audio_mod.AudioToText, "_detect_language", lambda self, _a, _s: "ja")

        transcriber.transcribe(_short_audio())

        call = fake_whisper.transcribe_calls[0]
        assert "logprob_threshold" in call
        assert call["logprob_threshold"] is None

    def test_kwargs_forwarded_with_vad_disabled(
        self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(enable_vad=False, device=None)

        transcriber.transcribe(_short_audio())

        call = fake_whisper.transcribe_calls[0]
        for key, expected in self.EXPECTED_DEFAULTS.items():
            assert call[key] == expected


class TestProcessTranscriptionConfidenceFields:
    """Whisper's per-segment confidence signals (avg_logprob, no_speech_prob,
    compression_ratio) must flow through _process_transcription_result onto
    TranscriptionSegment so downstream quality heuristics can use them."""

    def test_confidence_fields_populated_when_present(self, cpu_transcriber: audio_mod.AudioToText) -> None:
        raw_result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "words": [{"word": "hello", "start": 0.0, "end": 1.0}],
                    "avg_logprob": -0.8,
                    "no_speech_prob": 0.1,
                    "compression_ratio": 2.1,
                }
            ],
        }

        transcription = cpu_transcriber._process_transcription_result(raw_result)

        assert len(transcription.segments) == 1
        seg = transcription.segments[0]
        assert seg.avg_logprob == -0.8
        assert seg.no_speech_prob == 0.1
        assert seg.compression_ratio == 2.1

    def test_confidence_fields_none_when_missing(self, cpu_transcriber: audio_mod.AudioToText) -> None:
        raw_result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "words": [],
                }
            ],
        }

        transcription = cpu_transcriber._process_transcription_result(raw_result)

        seg = transcription.segments[0]
        assert seg.avg_logprob is None
        assert seg.no_speech_prob is None
        assert seg.compression_ratio is None


class TestAttachConfidenceByOverlap:
    """_attach_confidence_by_overlap re-attaches Whisper's per-segment
    confidence onto a diarization-rebuilt segment list by max-overlap match."""

    @staticmethod
    def _seg(start: float, end: float, *, avg_logprob=None, no_speech_prob=None, compression_ratio=None):
        from videopython.base.text.transcription import TranscriptionSegment

        return TranscriptionSegment(
            start=start,
            end=end,
            text="x",
            words=[],
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            compression_ratio=compression_ratio,
        )

    def test_target_inside_one_source_inherits_its_confidence(self) -> None:
        sources = [self._seg(0.0, 10.0, avg_logprob=-0.5, no_speech_prob=0.1, compression_ratio=1.8)]
        targets = [self._seg(2.0, 7.0)]

        audio_mod._attach_confidence_by_overlap(targets, sources)

        assert targets[0].avg_logprob == -0.5
        assert targets[0].no_speech_prob == 0.1
        assert targets[0].compression_ratio == 1.8

    def test_target_spanning_two_sources_picks_max_overlap(self) -> None:
        # source A covers [0, 4), source B covers [4, 10). Target is [3, 8) —
        # 1s overlap with A, 4s overlap with B → B wins.
        sources = [
            self._seg(0.0, 4.0, avg_logprob=-0.3),
            self._seg(4.0, 10.0, avg_logprob=-1.5),
        ]
        targets = [self._seg(3.0, 8.0)]

        audio_mod._attach_confidence_by_overlap(targets, sources)

        assert targets[0].avg_logprob == -1.5

    def test_target_with_no_overlap_left_untouched(self) -> None:
        sources = [self._seg(0.0, 5.0, avg_logprob=-0.5)]
        targets = [self._seg(10.0, 15.0)]

        audio_mod._attach_confidence_by_overlap(targets, sources)

        assert targets[0].avg_logprob is None
        assert targets[0].no_speech_prob is None
        assert targets[0].compression_ratio is None

    def test_diarization_split_each_half_inherits_parent(self) -> None:
        """Realistic case: one Whisper segment, diarization splits it across two
        speakers. Both halves should inherit the parent's confidence."""
        sources = [self._seg(0.0, 10.0, avg_logprob=-0.7, no_speech_prob=0.05)]
        targets = [self._seg(0.0, 4.5), self._seg(4.5, 10.0)]

        audio_mod._attach_confidence_by_overlap(targets, sources)

        for tgt in targets:
            assert tgt.avg_logprob == -0.7
            assert tgt.no_speech_prob == 0.05


class TestDiarizationCarriesConfidence:
    """End-to-end check: _transcribe_with_diarization must surface
    per-segment confidence on the rebuilt segments. Without M2.0 these
    fields are None on every diarized run."""

    def test_diarized_segments_inherit_whisper_confidence(
        self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(enable_diarization=True, device=None)

        # Whisper produces one 10s segment with healthy confidence and
        # word-level timings.
        def fake_transcribe(**kwargs: Any) -> dict[str, Any]:
            fake_whisper.transcribe_calls.append(kwargs)
            return {
                "language": "ja",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 10.0,
                        "text": "alpha beta",
                        "words": [
                            {"word": "alpha", "start": 0.0, "end": 4.0},
                            {"word": "beta", "start": 4.0, "end": 10.0},
                        ],
                        "avg_logprob": -0.6,
                        "no_speech_prob": 0.08,
                        "compression_ratio": 1.9,
                    }
                ],
            }

        fake_whisper.transcribe = fake_transcribe  # type: ignore[method-assign]

        # Diarization splits the segment into two speakers at t=4.0.
        def fake_init_diarization(self: Any) -> None:
            class _Annotation:
                def itertracks(self, yield_label: bool = True):
                    Turn = type("Turn", (), {})
                    a = Turn()
                    a.start, a.end = 0.0, 4.0
                    b = Turn()
                    b.start, b.end = 4.0, 10.0
                    return iter([(a, None, "SPEAKER_00"), (b, None, "SPEAKER_01")])

            class _DiarOutput:
                exclusive_speaker_diarization = _Annotation()

            self._diarization_pipeline = lambda _payload: _DiarOutput()

        monkeypatch.setattr(audio_mod.AudioToText, "_init_diarization", fake_init_diarization)
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", lambda self, _a: [(0.0, 10.0)])
        monkeypatch.setattr(audio_mod.AudioToText, "_detect_language", lambda self, _a, _s: "ja")

        # Use a 10s audio so VAD/Whisper see real timing; content is irrelevant.
        result = transcriber.transcribe(_short_audio(duration_s=10.0))

        # Diarization should have produced two segments (one per speaker).
        assert len(result.segments) == 2
        speakers = {s.speaker for s in result.segments}
        assert speakers == {"SPEAKER_00", "SPEAKER_01"}

        # Both halves overlap the single Whisper segment, so both inherit its
        # confidence — proves M2's confidence-aware prompting has signal on
        # the diarized path.
        for seg in result.segments:
            assert seg.avg_logprob == -0.6
            assert seg.no_speech_prob == 0.08
            assert seg.compression_ratio == 1.9


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


class TestVocabularyBiasing:
    """``vocabulary`` becomes Whisper's ``initial_prompt`` when set, and is
    silently absent when not. Constructor default + per-call override both
    work; normalization (dedup, strip, casing) is applied."""

    @staticmethod
    def _vad_and_lang(monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub VAD + language detection so transcribe() reaches the model."""
        monkeypatch.setattr(audio_mod.AudioToText, "_run_vad", lambda self, _a: [(0.0, 1.0)])
        monkeypatch.setattr(audio_mod.AudioToText, "_detect_language", lambda self, _a, _s: "en")

    def test_no_vocabulary_omits_initial_prompt(
        self,
        cpu_transcriber: audio_mod.AudioToText,
        fake_whisper: _FakeWhisperModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Default constructor has no vocabulary; the kwargs dict must not
        # carry an ``initial_prompt`` key — preserves byte-identity with
        # pre-M1 call shape.
        self._vad_and_lang(monkeypatch)

        cpu_transcriber.transcribe(_short_audio())

        assert "initial_prompt" not in fake_whisper.transcribe_calls[0]

    def test_constructor_vocabulary_injects_prompt(
        self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._vad_and_lang(monkeypatch)
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(vocabulary=["Klarna", "Allegro", "InPost"], device=None)

        transcriber.transcribe(_short_audio())

        prompt = fake_whisper.transcribe_calls[0]["initial_prompt"]
        # All three terms appear, casing preserved.
        assert "Klarna" in prompt
        assert "Allegro" in prompt
        assert "InPost" in prompt

    def test_per_call_vocabulary_overrides_constructor(
        self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._vad_and_lang(monkeypatch)
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(vocabulary=["Klarna"], device=None)

        transcriber.transcribe(_short_audio(), vocabulary=["Pyszne", "Wolt"])

        prompt = fake_whisper.transcribe_calls[0]["initial_prompt"]
        assert "Klarna" not in prompt
        assert "Pyszne" in prompt
        assert "Wolt" in prompt

    def test_per_call_empty_list_overrides_constructor_to_no_prompt(
        self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Per-call ``[]`` is an explicit "ignore the instance default" — the
        # only escape hatch when one shared transcriber serves callers who
        # don't want biasing.
        self._vad_and_lang(monkeypatch)
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(vocabulary=["Klarna"], device=None)

        transcriber.transcribe(_short_audio(), vocabulary=[])

        assert "initial_prompt" not in fake_whisper.transcribe_calls[0]

    def test_diarization_branch_carries_initial_prompt(
        self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both transcribe paths funnel through ``_transcribe_kwargs`` so
        # diarization must see the same ``initial_prompt``.
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(vocabulary=["Klarna"], enable_diarization=True, device=None)

        def fake_init_diarization(self: Any) -> None:
            class _EmptyAnnotation:
                def itertracks(self, yield_label: bool = True):
                    return iter([])

            class _DiarOutput:
                exclusive_speaker_diarization = _EmptyAnnotation()

            self._diarization_pipeline = lambda _payload: _DiarOutput()

        monkeypatch.setattr(audio_mod.AudioToText, "_init_diarization", fake_init_diarization)
        self._vad_and_lang(monkeypatch)

        transcriber.transcribe(_short_audio())

        assert "Klarna" in fake_whisper.transcribe_calls[0]["initial_prompt"]

    def test_normalization_dedup_strip_preserve_casing(
        self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._vad_and_lang(monkeypatch)
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(
            vocabulary=["  Klarna  ", "klarna", "KLARNA", "", "Allegro"],
            device=None,
        )

        # Constructor-level normalization: 5 inputs collapse to 2 unique.
        assert transcriber.vocabulary == ["Klarna", "Allegro"]

        transcriber.transcribe(_short_audio())
        prompt = fake_whisper.transcribe_calls[0]["initial_prompt"]
        # Original casing of the *first occurrence* survives; lowercase
        # duplicates are dropped.
        assert "Klarna" in prompt
        assert "klarna," not in prompt and "KLARNA" not in prompt
        assert "Allegro" in prompt

    def test_whitespace_only_vocabulary_omits_prompt(
        self, fake_whisper: _FakeWhisperModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._vad_and_lang(monkeypatch)
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        transcriber = audio_mod.AudioToText(vocabulary=["", "   ", "\t"], device=None)

        transcriber.transcribe(_short_audio())

        assert "initial_prompt" not in fake_whisper.transcribe_calls[0]

    def test_token_budget_truncation_warns(
        self,
        fake_whisper: _FakeWhisperModel,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # 500 unique multi-syllable names trivially exceeds Whisper's 224-
        # token initial_prompt window. The builder must trim from the tail
        # and emit one warning.
        import logging

        self._vad_and_lang(monkeypatch)
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        big_vocab = [f"Brandname{i:04d}" for i in range(500)]
        transcriber = audio_mod.AudioToText(vocabulary=big_vocab, device=None)

        with caplog.at_level(logging.WARNING, logger=audio_mod.logger.name):
            transcriber.transcribe(_short_audio())

        prompt = fake_whisper.transcribe_calls[0]["initial_prompt"]
        # Verify the actually-emitted prompt fits the budget.
        import whisper.tokenizer

        tok = whisper.tokenizer.get_tokenizer(multilingual=True, task="transcribe")
        assert len(tok.encode(prompt)) <= audio_mod._INITIAL_PROMPT_TOKEN_BUDGET

        # Front of the list is preserved; tail is dropped.
        assert "Brandname0000" in prompt
        assert "Brandname0499" not in prompt

        warning_lines = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("vocabulary truncated" in r.message for r in warning_lines)

    def test_non_list_vocabulary_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        with pytest.raises(TypeError, match="vocabulary"):
            audio_mod.AudioToText(vocabulary="Klarna", device=None)  # type: ignore[arg-type]

    def test_non_str_vocabulary_item_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(audio_mod, "select_device", lambda _r, mps_allowed=False: "cpu")
        with pytest.raises(TypeError, match="vocabulary items"):
            audio_mod.AudioToText(vocabulary=["Klarna", 42], device=None)  # type: ignore[list-item]
