from __future__ import annotations

import numpy as np
import pytest

from videopython.ai.dubbing._checkpoint import DubbingCheckpoint
from videopython.ai.dubbing.models import TranslatedSegment
from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
from videopython.ai.dubbing.timing import TimingSynchronizer
from videopython.audio import Audio, AudioMetadata
from videopython.base.transcription import Transcription, TranscriptionSegment, TranscriptionWord


def _source() -> tuple[Audio, TranscriptionSegment]:
    sample_rate = 8000
    data = np.zeros(sample_rate * 2, dtype=np.float32)
    audio = Audio(data, AudioMetadata(sample_rate, 1, 2, 2.0, len(data)))
    words = [TranscriptionWord(start=0.0, end=1.0, word="Hello", speaker="speaker_0")]
    segment = TranscriptionSegment.from_words(words, speaker="speaker_0")
    return audio, segment


def test_checkpoint_rejects_different_run(tmp_path) -> None:
    DubbingCheckpoint(tmp_path, {"version": 1, "target": "pl"})

    with pytest.raises(ValueError, match="does not match"):
        DubbingCheckpoint(tmp_path, {"version": 1, "target": "es"})


def test_pipeline_resumes_translation_and_speech(tmp_path, monkeypatch) -> None:
    source_audio, segment = _source()
    transcription = Transcription(segments=[segment], language="en")
    checkpoint = DubbingCheckpoint(tmp_path, {"version": 1})
    calls = {"translate": 0, "tts": 0}

    def translate(self, _transcription, source_lang, target_lang, _progress):
        calls["translate"] += 1
        translated = TranslatedSegment(
            original_segment=segment,
            translated_text="Cześć",
            source_lang=source_lang,
            target_lang=target_lang,
            source_segment_index=0,
        )
        return [translated], []

    def synthesize(self, **_kwargs):
        calls["tts"] += 1
        return Audio.create_silent(0.5, sample_rate=8000, stereo=False)

    monkeypatch.setattr(LocalDubbingPipeline, "_translate", translate)
    monkeypatch.setattr(LocalDubbingPipeline, "_tts_segment_audio", synthesize)

    first = LocalDubbingPipeline().process(
        source_audio=source_audio,
        target_lang="pl",
        transcription=transcription,
        preserve_background=False,
        voice_clone=False,
        checkpoint=checkpoint,
    )
    progress: list[str] = []
    second = LocalDubbingPipeline().process(
        source_audio=source_audio,
        target_lang="pl",
        preserve_background=False,
        voice_clone=False,
        checkpoint=checkpoint,
        progress_callback=lambda stage, _fraction: progress.append(stage),
    )

    assert calls == {"translate": 1, "tts": 1}
    assert first.translated_segments == second.translated_segments
    assert second.source_transcription == transcription
    assert (tmp_path / "translation.json").exists()
    assert checkpoint.speech_path([0]).exists()
    assert "Using checkpointed transcription" in progress
    assert "Using checkpointed translation" in progress


def test_checkpointed_assembly_matches_in_memory_assembly(tmp_path) -> None:
    sample_rate = 8000
    audio = Audio(
        np.full(800, 0.25, dtype=np.float32),
        AudioMetadata(sample_rate, 1, 2, 0.1, 800),
    )
    paths = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for path in paths:
        audio.save(path)

    loaded = [Audio.from_path(path) for path in paths]
    synchronizer = TimingSynchronizer()
    synchronized, expected_adjustments = synchronizer.synchronize_segments(loaded, [0.1, 0.1])
    expected = synchronizer.assemble_with_timing(synchronized, [0.0, 0.2], 0.4)
    actual, actual_adjustments = synchronizer.synchronize_files_and_assemble(paths, [0.1, 0.1], [0.0, 0.2], 0.4)

    np.testing.assert_allclose(actual.data, expected.data)
    assert actual_adjustments == expected_adjustments
