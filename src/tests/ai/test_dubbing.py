"""Tests for video dubbing functionality."""

import numpy as np
import pytest

from videopython.ai.dubbing.models import DubbingResult, SeparatedAudio, TranslatedSegment
from videopython.ai.dubbing.timing import TimingSynchronizer
from videopython.base.audio import Audio, AudioMetadata
from videopython.base.text.transcription import TranscriptionSegment, TranscriptionWord


@pytest.fixture
def sample_audio():
    """Create a sample audio for testing."""
    sample_rate = 24000
    duration = 2.0
    frame_count = int(sample_rate * duration)
    # Create a simple sine wave
    t = np.linspace(0, duration, frame_count, dtype=np.float32)
    data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    metadata = AudioMetadata(
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
        duration_seconds=duration,
        frame_count=frame_count,
    )
    return Audio(data, metadata)


@pytest.fixture
def sample_segment():
    """Create a sample transcription segment."""
    words = [
        TranscriptionWord(start=0.0, end=0.5, word="Hello"),
        TranscriptionWord(start=0.5, end=1.0, word="world"),
    ]
    return TranscriptionSegment(
        start=0.0,
        end=1.0,
        text="Hello world",
        words=words,
        speaker="speaker_0",
    )


class TestTranslatedSegment:
    """Tests for TranslatedSegment data model."""

    def test_creation(self, sample_segment):
        """Test creating a TranslatedSegment."""
        segment = TranslatedSegment(
            original_segment=sample_segment,
            translated_text="Hola mundo",
            source_lang="en",
            target_lang="es",
        )

        assert segment.translated_text == "Hola mundo"
        assert segment.source_lang == "en"
        assert segment.target_lang == "es"
        assert segment.original_text == "Hello world"
        assert segment.start == 0.0
        assert segment.end == 1.0
        assert segment.duration == 1.0
        assert segment.speaker == "speaker_0"

    def test_explicit_timing(self, sample_segment):
        """Test that explicit timing overrides segment timing."""
        segment = TranslatedSegment(
            original_segment=sample_segment,
            translated_text="Hola mundo",
            source_lang="en",
            target_lang="es",
            start=2.0,
            end=3.0,
        )

        assert segment.start == 2.0
        assert segment.end == 3.0
        assert segment.duration == 1.0

    def test_explicit_speaker(self, sample_segment):
        """Test that explicit speaker overrides segment speaker."""
        segment = TranslatedSegment(
            original_segment=sample_segment,
            translated_text="Hola mundo",
            source_lang="en",
            target_lang="es",
            speaker="speaker_1",
        )

        assert segment.speaker == "speaker_1"


class TestSeparatedAudio:
    """Tests for SeparatedAudio data model."""

    def test_creation(self, sample_audio):
        """Test creating a SeparatedAudio."""
        separated = SeparatedAudio(
            vocals=sample_audio,
            background=sample_audio,
            original=sample_audio,
        )

        assert separated.vocals == sample_audio
        assert separated.background == sample_audio
        assert separated.original == sample_audio
        assert not separated.has_detailed_separation

    def test_detailed_separation(self, sample_audio):
        """Test SeparatedAudio with detailed separation."""
        separated = SeparatedAudio(
            vocals=sample_audio,
            background=sample_audio,
            original=sample_audio,
            music=sample_audio,
            effects=sample_audio,
        )

        assert separated.has_detailed_separation


class TestTimingSynchronizer:
    """Tests for TimingSynchronizer."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        sync = TimingSynchronizer()

        assert sync.min_speed == 0.8
        assert sync.max_speed == 1.3
        assert sync.gap_threshold == 0.1

    def test_initialization_custom(self):
        """Test custom initialization values."""
        sync = TimingSynchronizer(min_speed=0.5, max_speed=2.0, gap_threshold=0.2)

        assert sync.min_speed == 0.5
        assert sync.max_speed == 2.0
        assert sync.gap_threshold == 0.2

    def test_initialization_invalid(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="min_speed must be positive"):
            TimingSynchronizer(min_speed=0)

        with pytest.raises(ValueError, match="max_speed must be greater than min_speed"):
            TimingSynchronizer(min_speed=1.0, max_speed=0.5)

    def test_synchronize_segment_no_change(self, sample_audio):
        """Test synchronizing when no change is needed."""
        sync = TimingSynchronizer()
        target_duration = sample_audio.metadata.duration_seconds

        result, adjustment = sync.synchronize_segment(sample_audio, target_duration)

        assert abs(result.metadata.duration_seconds - target_duration) < 0.1
        assert adjustment.speed_factor == 1.0
        assert not adjustment.was_truncated

    def test_synchronize_segment_speed_up(self, sample_audio):
        """Test synchronizing by speeding up audio."""
        sync = TimingSynchronizer()
        target_duration = sample_audio.metadata.duration_seconds / 1.2  # Need 20% speedup

        result, adjustment = sync.synchronize_segment(sample_audio, target_duration)

        # Result should be shorter
        assert result.metadata.duration_seconds <= target_duration + 0.1
        assert adjustment.speed_factor > 1.0
        assert adjustment.original_duration > adjustment.actual_duration

    def test_synchronize_segment_slow_down(self, sample_audio):
        """Test synchronizing by slowing down audio."""
        sync = TimingSynchronizer()
        target_duration = sample_audio.metadata.duration_seconds * 1.2  # Need 20% slowdown

        result, adjustment = sync.synchronize_segment(sample_audio, target_duration, segment_index=5)

        # Result should be longer (but may still be shorter than target if at min_speed limit)
        assert adjustment.speed_factor < 1.0
        assert adjustment.segment_index == 5

    def test_synchronize_segment_truncation(self, sample_audio):
        """Test that audio is truncated when even max speed isn't enough."""
        sync = TimingSynchronizer(max_speed=1.3)
        target_duration = sample_audio.metadata.duration_seconds / 2.0  # Need 50% shorter

        result, adjustment = sync.synchronize_segment(sample_audio, target_duration)

        # Should be truncated to target duration
        assert abs(result.metadata.duration_seconds - target_duration) < 0.1
        assert adjustment.was_truncated

    def test_synchronize_segments(self, sample_audio):
        """Test synchronizing multiple segments."""
        sync = TimingSynchronizer()
        audio_segments = [sample_audio, sample_audio]
        target_durations = [1.5, 2.5]

        results, adjustments = sync.synchronize_segments(audio_segments, target_durations)

        assert len(results) == 2
        assert len(adjustments) == 2
        assert adjustments[0].segment_index == 0
        assert adjustments[1].segment_index == 1

    def test_synchronize_segments_length_mismatch(self, sample_audio):
        """Test that mismatched lengths raise an error."""
        sync = TimingSynchronizer()
        audio_segments = [sample_audio, sample_audio]
        target_durations = [1.5]

        with pytest.raises(ValueError, match="Length mismatch"):
            sync.synchronize_segments(audio_segments, target_durations)

    def test_assemble_with_timing(self, sample_audio):
        """Test assembling segments with timing."""
        sync = TimingSynchronizer()
        # Create shorter segments
        short_audio = sample_audio.slice(0, 0.5)
        segments = [short_audio, short_audio]
        start_times = [0.0, 2.0]
        total_duration = 3.0

        result = sync.assemble_with_timing(segments, start_times, total_duration)

        assert result.metadata.duration_seconds >= total_duration - 0.1
        assert result.metadata.channels == 1

    def test_assemble_with_timing_length_mismatch(self, sample_audio):
        """Test that mismatched lengths raise an error."""
        sync = TimingSynchronizer()
        segments = [sample_audio]
        start_times = [0.0, 1.0]

        with pytest.raises(ValueError, match="Length mismatch"):
            sync.assemble_with_timing(segments, start_times, 5.0)

    def test_assemble_with_timing_invalid_start(self, sample_audio):
        """Test that negative start times raise an error."""
        sync = TimingSynchronizer()
        segments = [sample_audio]
        start_times = [-1.0]

        with pytest.raises(ValueError, match="Invalid start time"):
            sync.assemble_with_timing(segments, start_times, 5.0)

    def test_check_overlaps_no_overlaps(self):
        """Test check_overlaps with non-overlapping segments."""
        sync = TimingSynchronizer()
        start_times = [0.0, 2.0, 4.0]
        durations = [1.0, 1.0, 1.0]

        overlaps = sync.check_overlaps(start_times, durations)

        assert len(overlaps) == 0

    def test_check_overlaps_with_overlaps(self):
        """Test check_overlaps with overlapping segments."""
        sync = TimingSynchronizer()
        start_times = [0.0, 1.5, 3.0]
        durations = [2.0, 2.0, 1.0]  # First overlaps with second

        overlaps = sync.check_overlaps(start_times, durations)

        assert len(overlaps) >= 1
        # First pair should overlap
        first_overlap = overlaps[0]
        assert first_overlap[0] == 0  # First segment index
        assert first_overlap[1] == 1  # Second segment index
        assert first_overlap[2] > 0  # Overlap duration


class TestDubbingResult:
    """Tests for DubbingResult data model."""

    def test_creation(self, sample_audio, sample_segment):
        """Test creating a DubbingResult."""
        from videopython.base.text.transcription import Transcription

        translated = TranslatedSegment(
            original_segment=sample_segment,
            translated_text="Hola mundo",
            source_lang="en",
            target_lang="es",
        )

        result = DubbingResult(
            dubbed_audio=sample_audio,
            translated_segments=[translated],
            source_transcription=Transcription(segments=[sample_segment]),
            source_lang="en",
            target_lang="es",
        )

        assert result.num_segments == 1
        assert result.total_duration == sample_audio.metadata.duration_seconds
        assert result.source_lang == "en"
        assert result.target_lang == "es"

    def test_get_segments_by_speaker(self, sample_audio, sample_segment):
        """Test grouping segments by speaker."""
        from videopython.base.text.transcription import Transcription

        # Create segments with different speakers
        segment1 = TranslatedSegment(
            original_segment=sample_segment,
            translated_text="Hola",
            source_lang="en",
            target_lang="es",
            speaker="speaker_0",
        )
        segment2 = TranslatedSegment(
            original_segment=sample_segment,
            translated_text="Mundo",
            source_lang="en",
            target_lang="es",
            speaker="speaker_1",
        )
        segment3 = TranslatedSegment(
            original_segment=sample_segment,
            translated_text="Adios",
            source_lang="en",
            target_lang="es",
            speaker="speaker_0",
        )

        result = DubbingResult(
            dubbed_audio=sample_audio,
            translated_segments=[segment1, segment2, segment3],
            source_transcription=Transcription(segments=[sample_segment]),
            source_lang="en",
            target_lang="es",
        )

        by_speaker = result.get_segments_by_speaker()

        assert len(by_speaker) == 2
        assert len(by_speaker["speaker_0"]) == 2
        assert len(by_speaker["speaker_1"]) == 1


class TestVideoDubber:
    """Tests for VideoDubber class."""

    def test_initialization_default(self):
        """Test default initialization."""
        from videopython.ai.dubbing import VideoDubber

        dubber = VideoDubber()

        assert dubber.device is None
        assert dubber._local_pipeline is None

    def test_initialization_with_device(self):
        """Test initialization with explicit device."""
        from videopython.ai.dubbing import VideoDubber

        dubber = VideoDubber(device="cpu")

        assert dubber.device == "cpu"

    def test_get_supported_languages(self):
        """Test getting supported languages."""
        from videopython.ai.dubbing import VideoDubber

        languages = VideoDubber.get_supported_languages()

        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages
        assert languages["en"] == "English"
        assert languages["es"] == "Spanish"


class TestTextTranslator:
    """Tests for TextTranslator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        from videopython.ai.generation.translation import TextTranslator

        translator = TextTranslator()

        assert translator.model_name is None

    def test_initialization_with_model_name(self):
        """Test initialization with explicit model name."""
        from videopython.ai.generation.translation import TextTranslator

        translator = TextTranslator(model_name="Helsinki-NLP/opus-mt-en-es")

        assert translator.model_name == "Helsinki-NLP/opus-mt-en-es"

    def test_get_supported_languages(self):
        """Test getting supported languages."""
        from videopython.ai.generation.translation import TextTranslator

        languages = TextTranslator.get_supported_languages()

        assert "en" in languages
        assert "es" in languages
        assert languages["en"] == "English"

    def test_translate_empty_text(self):
        """Test that empty text is returned as-is."""
        from videopython.ai.generation.translation import TextTranslator

        translator = TextTranslator()

        result = translator.translate("", target_lang="es")

        assert result == ""

    def test_translate_segments(self, sample_segment):
        """Test translate_segments structure (without actual API call)."""
        from videopython.ai.generation.translation import TextTranslator

        # Create translator but don't call translate (would require API)
        translator = TextTranslator()

        # Just verify the method exists and has correct signature
        assert hasattr(translator, "translate_segments")
        assert callable(translator.translate_segments)


class TestAudioSeparator:
    """Tests for AudioSeparator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        from videopython.ai.understanding.separation import AudioSeparator

        separator = AudioSeparator()

        assert separator.model_name == "htdemucs"

    def test_initialization_custom_model(self):
        """Test custom model initialization."""
        from videopython.ai.understanding.separation import AudioSeparator

        separator = AudioSeparator(model_name="htdemucs_ft")

        assert separator.model_name == "htdemucs_ft"

    def test_initialization_invalid_model(self):
        """Test that invalid model raises error."""
        from videopython.ai.understanding.separation import AudioSeparator

        with pytest.raises(ValueError, match="not supported"):
            AudioSeparator(model_name="invalid_model")


class TestUnloadMethods:
    """Tests for unload() on each model class used in dubbing.

    Each class must clear its cached model attributes so that the next call
    re-initializes lazily. Used by low-memory dubbing to free VRAM between
    pipeline stages.
    """

    def test_audio_separator_unload(self):
        from videopython.ai.understanding.separation import AudioSeparator

        separator = AudioSeparator()
        separator._model = object()  # stand-in for a loaded model
        separator.unload()
        assert separator._model is None

    def test_audio_separator_unload_is_idempotent(self):
        from videopython.ai.understanding.separation import AudioSeparator

        separator = AudioSeparator()
        separator.unload()
        separator.unload()
        assert separator._model is None

    def test_text_translator_unload(self):
        from videopython.ai.generation.translation import TextTranslator

        translator = TextTranslator()
        translator._model = object()
        translator._tokenizer = object()
        translator._current_lang_pair = ("en", "es")
        translator.unload()
        assert translator._model is None
        assert translator._tokenizer is None
        assert translator._current_lang_pair is None

    def test_text_to_speech_unload(self):
        from videopython.ai.generation.audio import TextToSpeech

        tts = TextToSpeech()
        tts._model = object()
        tts.unload()
        assert tts._model is None

    def test_audio_to_text_unload(self, monkeypatch):
        import videopython.ai.understanding.audio as audio_mod

        monkeypatch.setattr(audio_mod, "select_device", lambda _requested, mps_allowed=False: "cpu")

        transcriber = audio_mod.AudioToText()
        transcriber._model = object()
        transcriber._diarization_pipeline = object()
        transcriber.unload()
        assert transcriber._model is None
        assert transcriber._diarization_pipeline is None


class TestLocalDubbingPipelineLowMemory:
    """Tests for low_memory mode plumbing in LocalDubbingPipeline."""

    def test_default_is_not_low_memory(self):
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        pipeline = LocalDubbingPipeline()
        assert pipeline.low_memory is False

    def test_low_memory_flag_stored(self):
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        pipeline = LocalDubbingPipeline(low_memory=True)
        assert pipeline.low_memory is True

    def test_maybe_unload_noop_when_low_memory_disabled(self):
        """With low_memory=False, _maybe_unload must not call unload()."""
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        pipeline = LocalDubbingPipeline(low_memory=False)

        class FakeModel:
            def __init__(self):
                self.unload_calls = 0

            def unload(self):
                self.unload_calls += 1

        fake = FakeModel()
        pipeline._transcriber = fake
        pipeline._maybe_unload("_transcriber")
        assert fake.unload_calls == 0

    def test_maybe_unload_calls_unload_when_enabled(self):
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        pipeline = LocalDubbingPipeline(low_memory=True)

        class FakeModel:
            def __init__(self):
                self.unload_calls = 0

            def unload(self):
                self.unload_calls += 1

        fake = FakeModel()
        pipeline._translator = fake
        pipeline._maybe_unload("_translator")
        assert fake.unload_calls == 1

    def test_maybe_unload_noop_when_component_is_none(self):
        """If a stage was never initialized (e.g. caller provided transcription),
        _maybe_unload must not raise."""
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        pipeline = LocalDubbingPipeline(low_memory=True)
        assert pipeline._transcriber is None
        pipeline._maybe_unload("_transcriber")  # should not raise


class TestVideoDubberLowMemory:
    """Tests for low_memory plumbing from VideoDubber to LocalDubbingPipeline."""

    def test_default_low_memory_false(self):
        from videopython.ai.dubbing import VideoDubber

        dubber = VideoDubber()
        assert dubber.low_memory is False

    def test_low_memory_propagated_to_pipeline(self):
        from videopython.ai.dubbing import VideoDubber

        dubber = VideoDubber(low_memory=True)
        assert dubber.low_memory is True

        dubber._init_local_pipeline()
        assert dubber._local_pipeline.low_memory is True


class TestWhisperModelSelection:
    """Tests for whisper_model plumbing through VideoDubber and the pipeline."""

    def test_pipeline_default_whisper_model(self):
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        pipeline = LocalDubbingPipeline()
        assert pipeline.whisper_model == "small"

    def test_pipeline_whisper_model_stored(self):
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        pipeline = LocalDubbingPipeline(whisper_model="turbo")
        assert pipeline.whisper_model == "turbo"

    def test_dubber_default_whisper_model(self):
        from videopython.ai.dubbing import VideoDubber

        dubber = VideoDubber()
        assert dubber.whisper_model == "small"

    def test_dubber_whisper_model_propagated_to_pipeline(self):
        from videopython.ai.dubbing import VideoDubber

        dubber = VideoDubber(whisper_model="large")
        dubber._init_local_pipeline()
        assert dubber._local_pipeline.whisper_model == "large"

    def test_init_transcriber_uses_whisper_model(self, monkeypatch):
        """_init_transcriber must pass whisper_model to AudioToText."""
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
        from videopython.ai.understanding import audio as audio_mod

        captured: dict[str, object] = {}

        class FakeAudioToText:
            def __init__(self, model_name, device, enable_diarization):
                captured["model_name"] = model_name
                captured["device"] = device
                captured["enable_diarization"] = enable_diarization

        monkeypatch.setattr(audio_mod, "AudioToText", FakeAudioToText)

        pipeline = LocalDubbingPipeline(whisper_model="medium")
        pipeline._init_transcriber(enable_diarization=True)

        assert captured == {"model_name": "medium", "device": None, "enable_diarization": True}


class TestReplaceAudioStream:
    """Tests for the ffmpeg audio-stream replacement helper."""

    def test_missing_video_raises(self, tmp_path):
        from videopython.ai.dubbing.remux import replace_audio_stream

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")

        with pytest.raises(FileNotFoundError, match="Video file not found"):
            replace_audio_stream(tmp_path / "missing.mp4", audio, tmp_path / "out.mp4")

    def test_missing_audio_raises(self, tmp_path):
        from videopython.ai.dubbing.remux import replace_audio_stream

        video = tmp_path / "v.mp4"
        video.write_bytes(b"")

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            replace_audio_stream(video, tmp_path / "missing.wav", tmp_path / "out.mp4")

    def test_ffmpeg_failure_raises_remux_error(self, tmp_path, monkeypatch):
        """Non-zero ffmpeg exit code is wrapped in RemuxError with stderr."""
        import subprocess as sp

        import videopython.ai.dubbing.remux as remux_mod

        video = tmp_path / "v.mp4"
        video.write_bytes(b"not a real video")
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"not a real wav")

        class FakeResult:
            returncode = 1
            stderr = b"simulated ffmpeg error"

        def fake_run(cmd, capture_output):
            return FakeResult()

        monkeypatch.setattr(sp, "run", fake_run)
        monkeypatch.setattr(remux_mod.subprocess, "run", fake_run)

        with pytest.raises(remux_mod.RemuxError, match="simulated ffmpeg error"):
            remux_mod.replace_audio_stream(video, audio, tmp_path / "out.mp4")

    def test_ffmpeg_command_structure(self, tmp_path, monkeypatch):
        """ffmpeg is invoked with stream-copy video and mapped audio."""
        import videopython.ai.dubbing.remux as remux_mod

        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"fake")
        out = tmp_path / "out.mp4"

        captured: dict = {}

        class FakeResult:
            returncode = 0
            stderr = b""

        def fake_run(cmd, capture_output):
            captured["cmd"] = cmd
            return FakeResult()

        monkeypatch.setattr(remux_mod.subprocess, "run", fake_run)

        remux_mod.replace_audio_stream(video, audio, out)

        cmd = captured["cmd"]
        assert cmd[0] == "ffmpeg"
        assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
        assert "-map" in cmd
        # Two -map args: video from input 0, audio from input 1
        map_indices = [i for i, v in enumerate(cmd) if v == "-map"]
        assert len(map_indices) == 2
        assert cmd[map_indices[0] + 1] == "0:v:0"
        assert cmd[map_indices[1] + 1] == "1:a:0"
        assert "-shortest" in cmd
        assert cmd[-1] == str(out)


class TestVideoDubberDubFile:
    """Tests for VideoDubber.dub_file path-based entry point."""

    def test_missing_input_raises(self, tmp_path):
        from videopython.ai.dubbing import VideoDubber

        dubber = VideoDubber()

        with pytest.raises(FileNotFoundError, match="Input video not found"):
            dubber.dub_file(
                input_path=tmp_path / "missing.mp4",
                output_path=tmp_path / "out.mp4",
                target_lang="es",
            )

    def test_dub_file_orchestration(self, tmp_path, sample_audio, sample_segment, monkeypatch):
        """dub_file extracts audio, runs pipeline, saves dubbed audio, and remuxes.

        Mocks Audio.from_path (to avoid ffmpeg), the pipeline (to avoid models),
        Audio.save (to avoid encoding), and replace_audio_stream (to avoid ffmpeg).
        Verifies the call sequence and argument wiring.
        """
        from pathlib import Path

        import videopython.ai.dubbing.dubber as dubber_mod
        import videopython.ai.dubbing.remux as remux_mod
        from videopython.ai.dubbing import VideoDubber
        from videopython.ai.dubbing.models import DubbingResult, TranslatedSegment
        from videopython.base.audio import audio as audio_mod
        from videopython.base.text.transcription import Transcription

        input_path = tmp_path / "in.mp4"
        input_path.write_bytes(b"fake mp4 bytes")
        output_path = tmp_path / "out.mp4"

        calls: list[tuple[str, dict]] = []

        def fake_from_path(cls, file_path):
            calls.append(("from_path", {"file_path": str(file_path)}))
            return sample_audio

        monkeypatch.setattr(audio_mod.Audio, "from_path", classmethod(fake_from_path))

        translated = TranslatedSegment(
            original_segment=sample_segment,
            translated_text="Hola",
            source_lang="en",
            target_lang="es",
        )
        fake_result = DubbingResult(
            dubbed_audio=sample_audio,
            translated_segments=[translated],
            source_transcription=Transcription(segments=[sample_segment]),
            source_lang="en",
            target_lang="es",
        )

        class FakePipeline:
            low_memory = False

            def process(self, **kwargs):
                calls.append(("process", kwargs))
                return fake_result

        def fake_init(self):
            self._local_pipeline = FakePipeline()

        monkeypatch.setattr(VideoDubber, "_init_local_pipeline", fake_init)

        def fake_save(self, file_path, format=None):
            calls.append(("save", {"file_path": str(file_path)}))
            Path(file_path).write_bytes(b"fake wav")

        monkeypatch.setattr(audio_mod.Audio, "save", fake_save)

        def fake_replace(video_path, audio_path, output_path, **kwargs):
            calls.append(
                (
                    "replace",
                    {
                        "video_path": str(video_path),
                        "audio_path": str(audio_path),
                        "output_path": str(output_path),
                    },
                )
            )

        monkeypatch.setattr(remux_mod, "replace_audio_stream", fake_replace)
        monkeypatch.setattr(dubber_mod, "tempfile", __import__("tempfile"))

        dubber = VideoDubber()
        result = dubber.dub_file(
            input_path=input_path,
            output_path=output_path,
            target_lang="es",
        )

        names = [c[0] for c in calls]
        assert names == ["from_path", "process", "save", "replace"]
        assert calls[0][1]["file_path"] == str(input_path)
        assert calls[1][1]["source_audio"] is sample_audio
        assert calls[1][1]["target_lang"] == "es"
        assert calls[3][1]["video_path"] == str(input_path)
        assert calls[3][1]["output_path"] == str(output_path)
        assert calls[2][1]["file_path"] == calls[3][1]["audio_path"]
        assert result is fake_result

    def test_dub_file_cleans_up_temp_audio_on_failure(self, tmp_path, sample_audio, sample_segment, monkeypatch):
        """Temp wav is deleted even if replace_audio_stream raises."""
        from pathlib import Path as _Path

        import videopython.ai.dubbing.remux as remux_mod
        from videopython.ai.dubbing import VideoDubber
        from videopython.ai.dubbing.models import DubbingResult, TranslatedSegment
        from videopython.base.audio import audio as audio_mod
        from videopython.base.text.transcription import Transcription

        input_path = tmp_path / "in.mp4"
        input_path.write_bytes(b"fake")

        temp_paths: list[_Path] = []

        monkeypatch.setattr(audio_mod.Audio, "from_path", classmethod(lambda cls, p: sample_audio))

        translated = TranslatedSegment(
            original_segment=sample_segment,
            translated_text="Hola",
            source_lang="en",
            target_lang="es",
        )
        fake_result = DubbingResult(
            dubbed_audio=sample_audio,
            translated_segments=[translated],
            source_transcription=Transcription(segments=[sample_segment]),
            source_lang="en",
            target_lang="es",
        )

        class FakePipeline:
            low_memory = False

            def process(self, **kwargs):
                return fake_result

        def fake_init(self):
            self._local_pipeline = FakePipeline()

        monkeypatch.setattr(VideoDubber, "_init_local_pipeline", fake_init)

        def fake_save(self, file_path, format=None):
            p = _Path(file_path)
            temp_paths.append(p)
            p.write_bytes(b"fake wav")

        monkeypatch.setattr(audio_mod.Audio, "save", fake_save)

        def failing_replace(**kwargs):
            raise remux_mod.RemuxError("boom")

        monkeypatch.setattr(remux_mod, "replace_audio_stream", failing_replace)

        dubber = VideoDubber()
        with pytest.raises(remux_mod.RemuxError, match="boom"):
            dubber.dub_file(
                input_path=input_path,
                output_path=tmp_path / "out.mp4",
                target_lang="es",
            )

        assert len(temp_paths) == 1
        assert not temp_paths[0].exists()
