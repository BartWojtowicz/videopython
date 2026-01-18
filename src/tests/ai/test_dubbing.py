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

        assert dubber.backend == "local"
        assert dubber.translation_backend == "openai"
        assert dubber.tts_backend == "local"

    def test_initialization_elevenlabs(self):
        """Test ElevenLabs backend initialization."""
        from videopython.ai.dubbing import VideoDubber

        dubber = VideoDubber(backend="elevenlabs")

        assert dubber.backend == "elevenlabs"

    def test_initialization_invalid_backend(self):
        """Test that invalid backend raises error."""
        from videopython.ai.backends import UnsupportedBackendError
        from videopython.ai.dubbing import VideoDubber

        with pytest.raises(UnsupportedBackendError):
            VideoDubber(backend="invalid")

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

        # Default falls back to "local" when no config is set
        assert translator.backend == "local"

    def test_initialization_local(self):
        """Test local backend initialization."""
        from videopython.ai.generation.translation import TextTranslator

        translator = TextTranslator(backend="local")

        assert translator.backend == "local"

    def test_initialization_invalid_backend(self):
        """Test that invalid backend raises error."""
        from videopython.ai.backends import UnsupportedBackendError
        from videopython.ai.generation.translation import TextTranslator

        with pytest.raises(UnsupportedBackendError):
            TextTranslator(backend="invalid")

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

        translator = TextTranslator(backend="local")

        result = translator.translate("", target_lang="es")

        assert result == ""

    def test_translate_segments(self, sample_segment):
        """Test translate_segments structure (without actual API call)."""
        from videopython.ai.generation.translation import TextTranslator

        # Create translator but don't call translate (would require API)
        translator = TextTranslator(backend="local")

        # Just verify the method exists and has correct signature
        assert hasattr(translator, "translate_segments")
        assert callable(translator.translate_segments)


class TestAudioSeparator:
    """Tests for AudioSeparator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        from videopython.ai.understanding.separation import AudioSeparator

        separator = AudioSeparator()

        assert separator.backend == "local"
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

    def test_initialization_invalid_backend(self):
        """Test that invalid backend raises error."""
        from videopython.ai.backends import UnsupportedBackendError
        from videopython.ai.understanding.separation import AudioSeparator

        with pytest.raises(UnsupportedBackendError):
            AudioSeparator(backend="invalid")
