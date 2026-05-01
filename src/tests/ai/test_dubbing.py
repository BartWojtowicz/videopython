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

    def test_assemble_with_timing_extends_past_total_duration(self, sample_audio):
        """Segments running past total_duration extend the output buffer.

        Mirrors the previous Audio.overlay behavior so dubbed speech that
        slightly overruns the source duration is preserved instead of
        silently truncated.
        """
        sync = TimingSynchronizer()
        # sample_audio is 2.0s; place it starting at 4.5s with total 5.0s
        result = sync.assemble_with_timing([sample_audio], [4.5], total_duration=5.0)

        assert result.metadata.duration_seconds >= 6.5 - 0.01
        # Energy should be in the placed region, not at the start
        sr = sample_audio.metadata.sample_rate
        head_rms = float(np.sqrt(np.mean(result.data[: sr // 2] ** 2)))
        tail_rms = float(np.sqrt(np.mean(result.data[int(4.5 * sr) : int(5.0 * sr)] ** 2)))
        assert tail_rms > head_rms

    def test_assemble_with_timing_no_unnecessary_normalization(self, sample_audio):
        """Non-overlapping segments below 1.0 amplitude are not rescaled.

        Single-pass assembler peak-guards once at the end. For typical dubs
        with non-overlapping segments this should be a no-op; the segment
        amplitude must come through unchanged.
        """
        sync = TimingSynchronizer()
        segments = [sample_audio.slice(0, 0.5), sample_audio.slice(0, 0.5)]
        start_times = [0.0, 1.0]
        result = sync.assemble_with_timing(segments, start_times, total_duration=2.0)

        peak = float(np.max(np.abs(result.data)))
        # sample_audio peaks at 0.5; with no overlap, output peak should match
        assert abs(peak - 0.5) < 0.01

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

    def test_is_translatable_text_filters_punctuation_and_short_text(self):
        """Filter accepts real text and rejects empty/punctuation/single-char Whisper noise."""
        from videopython.ai.generation.translation import _is_translatable_text

        assert _is_translatable_text("Hello world") is True
        assert _is_translatable_text("Hi") is True
        assert _is_translatable_text("¡Sí!") is True

        assert _is_translatable_text("") is False
        assert _is_translatable_text(" ") is False
        assert _is_translatable_text(" .") is False
        assert _is_translatable_text("...") is False
        assert _is_translatable_text("?") is False
        assert _is_translatable_text("♪") is False
        # Single-letter Whisper segment — too short to be reliable signal.
        assert _is_translatable_text("a") is False
        assert _is_translatable_text(" a ") is False

    def test_translate_segments_skips_punctuation_segments(self, monkeypatch):
        """Punctuation-only segments get translated_text="" without hitting the model."""
        from videopython.ai.generation.translation import TextTranslator

        translator = TextTranslator()

        translate_calls: list[list[str]] = []

        def fake_translate_batch(self, texts, target_lang, source_lang=None):
            translate_calls.append(list(texts))
            return [f"[{t}]" for t in texts]

        monkeypatch.setattr(TextTranslator, "translate_batch", fake_translate_batch)

        words_real = [TranscriptionWord(start=0.0, end=1.0, word="hello")]
        words_dot = [TranscriptionWord(start=1.0, end=2.0, word=".")]
        segments = [
            TranscriptionSegment(start=0.0, end=1.0, text="hello there", words=words_real),
            TranscriptionSegment(start=1.0, end=2.0, text=" .", words=words_dot),
            TranscriptionSegment(start=2.0, end=3.0, text="...", words=words_dot),
            TranscriptionSegment(start=3.0, end=4.0, text="how are you", words=words_real),
        ]

        result = translator.translate_segments(segments, target_lang="es", source_lang="en")

        # Only the two real segments are sent to the model.
        assert len(translate_calls) == 1
        assert translate_calls[0] == ["hello there", "how are you"]

        # Output preserves all four segments, with empty text for the filtered ones.
        assert len(result) == 4
        assert result[0].translated_text == "[hello there]"
        assert result[1].translated_text == ""
        assert result[2].translated_text == ""
        assert result[3].translated_text == "[how are you]"

        # Timing/speaker fields preserved on the skipped segments.
        assert result[1].start == 1.0
        assert result[1].end == 2.0


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


class TestMergeRegions:
    """Tests for the _merge_regions helper that prepares speech regions for Demucs."""

    def test_empty_returns_empty(self):
        from videopython.ai.understanding.separation import _merge_regions

        assert _merge_regions([], audio_duration=10.0) == []

    def test_single_region_is_padded_and_clamped(self):
        from videopython.ai.understanding.separation import _merge_regions

        out = _merge_regions([(2.0, 4.0)], audio_duration=10.0, pad=0.5)
        assert out == [(1.5, 4.5)]

    def test_pad_clamps_to_audio_bounds(self):
        from videopython.ai.understanding.separation import _merge_regions

        out = _merge_regions([(0.1, 9.8)], audio_duration=10.0, pad=0.5)
        # left clamped to 0.0; right clamped to 10.0
        assert out == [(0.0, 10.0)]

    def test_overlapping_regions_merge(self):
        from videopython.ai.understanding.separation import _merge_regions

        out = _merge_regions([(0.0, 3.0), (2.0, 5.0)], audio_duration=10.0, pad=0.0)
        assert out == [(0.0, 5.0)]

    def test_close_regions_merge_within_gap(self):
        from videopython.ai.understanding.separation import _merge_regions

        # gaps of 0.8s after padding are merged (default merge_gap=1.0)
        out = _merge_regions([(0.0, 2.0), (2.8, 5.0)], audio_duration=10.0, pad=0.0, merge_gap=1.0)
        assert out == [(0.0, 5.0)]

    def test_distant_regions_stay_separate(self):
        from videopython.ai.understanding.separation import _merge_regions

        out = _merge_regions([(0.0, 2.0), (8.0, 10.0)], audio_duration=10.0, pad=0.0)
        assert out == [(0.0, 2.0), (8.0, 10.0)]

    def test_unsorted_input_is_handled(self):
        from videopython.ai.understanding.separation import _merge_regions

        out = _merge_regions([(8.0, 9.0), (1.0, 2.0)], audio_duration=10.0, pad=0.0)
        assert out == [(1.0, 2.0), (8.0, 9.0)]

    def test_zero_length_regions_dropped(self):
        from videopython.ai.understanding.separation import _merge_regions

        out = _merge_regions([(2.0, 2.0), (3.0, 4.0)], audio_duration=10.0, pad=0.0)
        assert out == [(3.0, 4.0)]


class TestSeparateRegions:
    """Tests for AudioSeparator.separate_regions."""

    @pytest.fixture
    def long_stereo_audio(self):
        """A 10-second, 24 kHz stereo audio. Distinct values per second so we
        can verify which regions were touched by the fake separator."""
        sample_rate = 24000
        duration = 10.0
        frame_count = int(sample_rate * duration)
        # Per-second tag in left channel, ramp in right.
        left = np.repeat(np.arange(10, dtype=np.float32) * 0.05, sample_rate)[:frame_count]
        right = np.linspace(-0.5, 0.5, frame_count, dtype=np.float32)
        data = np.column_stack([left, right])
        metadata = AudioMetadata(
            sample_rate=sample_rate,
            channels=2,
            sample_width=2,
            duration_seconds=duration,
            frame_count=frame_count,
        )
        return Audio(data, metadata)

    def test_no_regions_returns_passthrough(self, long_stereo_audio, monkeypatch):
        """No speech regions => silent vocals, original as background. No Demucs call."""
        from videopython.ai.dubbing.models import SeparatedAudio
        from videopython.ai.understanding.separation import AudioSeparator

        separator = AudioSeparator()

        def fail_if_called(self, audio):
            raise AssertionError("Demucs should not run when there are no regions")

        monkeypatch.setattr(AudioSeparator, "_separate_local", fail_if_called)

        result = separator.separate_regions(long_stereo_audio, regions=[])

        assert isinstance(result, SeparatedAudio)
        assert result.vocals.metadata.duration_seconds == long_stereo_audio.metadata.duration_seconds
        assert np.allclose(result.vocals.data, 0.0)
        # Background is the original audio, untouched.
        assert np.array_equal(result.background.data, long_stereo_audio.data)

    def test_full_coverage_falls_back_to_full_separation(self, long_stereo_audio, monkeypatch):
        """When regions cover >= threshold of audio, full-track separate() is called."""
        from videopython.ai.understanding.separation import AudioSeparator

        separator = AudioSeparator()
        sentinel_calls: list[float] = []

        def fake_separate_local(self, audio):
            sentinel_calls.append(audio.metadata.duration_seconds)
            silent = np.zeros_like(audio.data, dtype=np.float32)
            return SeparatedAudio(
                vocals=Audio(silent, audio.metadata),
                background=Audio(audio.data.astype(np.float32), audio.metadata),
                original=audio,
                music=None,
                effects=None,
            )

        monkeypatch.setattr(AudioSeparator, "_separate_local", fake_separate_local)

        # Regions cover 9.5/10s = 95%, above 0.9 threshold => one full-track call.
        regions = [(0.0, 9.5)]
        separator.separate_regions(long_stereo_audio, regions, full_separation_threshold=0.9)

        assert len(sentinel_calls) == 1
        assert sentinel_calls[0] == long_stereo_audio.metadata.duration_seconds

    def test_partial_coverage_runs_per_region(self, long_stereo_audio, monkeypatch):
        """Partial speech => Demucs runs once per region; non-speech gaps stay original."""
        from videopython.ai.understanding.separation import AudioSeparator

        separator = AudioSeparator()
        chunk_durations: list[float] = []

        def fake_separate_local(self, audio):
            chunk_durations.append(audio.metadata.duration_seconds)
            # "Separated" chunk: vocals = constant 0.7, background = -0.3
            n = len(audio.data)
            vocals = np.full((n, 2), 0.7, dtype=np.float32)
            bg = np.full((n, 2), -0.3, dtype=np.float32)
            return SeparatedAudio(
                vocals=Audio(vocals, audio.metadata),
                background=Audio(bg, audio.metadata),
                original=audio,
                music=None,
                effects=None,
            )

        monkeypatch.setattr(AudioSeparator, "_separate_local", fake_separate_local)

        # Two distant regions, padded inside _merge_regions before this call;
        # we pass pre-merged regions directly.
        regions = [(1.0, 2.0), (6.0, 7.0)]
        result = separator.separate_regions(long_stereo_audio, regions, full_separation_threshold=0.9)

        # Two Demucs calls, one per region.
        assert len(chunk_durations) == 2
        for d in chunk_durations:
            assert abs(d - 1.0) < 0.01

        sr = long_stereo_audio.metadata.sample_rate
        # Vocals: zero outside regions, 0.7 inside.
        assert np.allclose(result.vocals.data[: int(1.0 * sr)], 0.0)
        assert np.allclose(result.vocals.data[int(1.0 * sr) : int(2.0 * sr)], 0.7)
        assert np.allclose(result.vocals.data[int(2.0 * sr) : int(6.0 * sr)], 0.0)
        assert np.allclose(result.vocals.data[int(6.0 * sr) : int(7.0 * sr)], 0.7)
        assert np.allclose(result.vocals.data[int(7.0 * sr) :], 0.0)

        # Background: original audio in gaps, -0.3 inside the regions.
        assert np.array_equal(
            result.background.data[: int(1.0 * sr)],
            long_stereo_audio.data[: int(1.0 * sr)],
        )
        assert np.allclose(result.background.data[int(1.0 * sr) : int(2.0 * sr)], -0.3)
        assert np.array_equal(
            result.background.data[int(2.0 * sr) : int(6.0 * sr)],
            long_stereo_audio.data[int(2.0 * sr) : int(6.0 * sr)],
        )
        assert np.allclose(result.background.data[int(6.0 * sr) : int(7.0 * sr)], -0.3)

    def test_mono_input_produces_stereo_output(self, monkeypatch):
        """Mono input is upmixed; output matches full-track contract (always stereo)."""
        from videopython.ai.understanding.separation import AudioSeparator

        sample_rate = 24000
        duration = 4.0
        frame_count = int(sample_rate * duration)
        data = np.full(frame_count, 0.2, dtype=np.float32)
        metadata = AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=duration,
            frame_count=frame_count,
        )
        mono = Audio(data, metadata)

        def fake_separate_local(self, audio):
            # Real _separate_local always returns stereo regardless of input.
            n = audio.metadata.frame_count
            zeros = np.zeros((n, 2), dtype=np.float32)
            stereo_meta = AudioMetadata(
                sample_rate=audio.metadata.sample_rate,
                channels=2,
                sample_width=audio.metadata.sample_width,
                duration_seconds=audio.metadata.duration_seconds,
                frame_count=n,
            )
            return SeparatedAudio(
                vocals=Audio(zeros, stereo_meta),
                background=Audio(zeros.copy(), stereo_meta),
                original=audio,
                music=None,
                effects=None,
            )

        monkeypatch.setattr(AudioSeparator, "_separate_local", fake_separate_local)

        separator = AudioSeparator()
        result = separator.separate_regions(mono, regions=[(1.0, 2.0)], full_separation_threshold=0.9)

        assert result.vocals.metadata.channels == 2
        assert result.background.metadata.channels == 2


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


class TestDiarizeTranscription:
    """Tests for AudioToText.diarize_transcription standalone diarization helper."""

    def _make_transcription_with_words(self):
        from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord

        # Two segments, each with multiple words — passes the word-level-timing check.
        seg1_words = [
            TranscriptionWord(start=0.0, end=0.5, word="Hello"),
            TranscriptionWord(start=0.5, end=1.0, word="world"),
        ]
        seg2_words = [
            TranscriptionWord(start=2.0, end=2.5, word="Goodbye"),
            TranscriptionWord(start=2.5, end=3.0, word="now"),
        ]
        segments = [
            TranscriptionSegment(start=0.0, end=1.0, text="Hello world", words=seg1_words),
            TranscriptionSegment(start=2.0, end=3.0, text="Goodbye now", words=seg2_words),
        ]
        return Transcription(segments=segments, language="en")

    def test_rejects_transcription_without_word_level_timings(self, sample_audio):
        """SRT-loaded transcriptions (one synthetic word per segment) must be rejected."""
        from videopython.ai.understanding.audio import AudioToText
        from videopython.base.text.transcription import Transcription

        srt = "1\n00:00:00,000 --> 00:00:01,000\nHello world\n\n"
        transcription = Transcription.from_srt(srt)

        transcriber = AudioToText()
        with pytest.raises(ValueError, match="word-level timings"):
            transcriber.diarize_transcription(sample_audio, transcription)

    def test_rejects_empty_transcription(self, sample_audio):
        from videopython.ai.understanding.audio import AudioToText
        from videopython.base.text.transcription import Transcription

        transcriber = AudioToText()
        with pytest.raises(ValueError, match="no words"):
            transcriber.diarize_transcription(sample_audio, Transcription(segments=[]))

    def test_runs_pyannote_and_attaches_speakers(self, sample_audio, monkeypatch):
        """Diarize-only path must run pyannote, overlay speakers, and return a new Transcription."""
        from videopython.ai.understanding.audio import AudioToText

        transcription = self._make_transcription_with_words()
        assert not transcription.speakers  # precondition

        class FakePipeline:
            def __init__(self):
                self.calls = 0

            def __call__(self, payload):
                self.calls += 1
                assert "waveform" in payload and "sample_rate" in payload
                return "fake-diarization-result"

        fake_pipeline = FakePipeline()

        def fake_init_diarization(self):
            self._diarization_pipeline = fake_pipeline

        captured: dict = {}

        @staticmethod
        def fake_assign_speakers(words, diarization_result):
            captured["diarization_result"] = diarization_result
            captured["n_words"] = len(words)
            # Alternate speakers across the supplied words to exercise rebuilding.
            return [
                TranscriptionWord(start=w.start, end=w.end, word=w.word, speaker=f"S{i % 2}")
                for i, w in enumerate(words)
            ]

        monkeypatch.setattr(AudioToText, "_init_diarization", fake_init_diarization)
        monkeypatch.setattr(AudioToText, "_assign_speakers_to_words", fake_assign_speakers)

        transcriber = AudioToText()
        result = transcriber.diarize_transcription(sample_audio, transcription)

        assert fake_pipeline.calls == 1
        assert captured["diarization_result"] == "fake-diarization-result"
        assert captured["n_words"] == 4
        assert result.speakers == {"S0", "S1"}
        assert result.language == "en"


class TestPipelineSuppliedTranscriptionDiarization:
    """Tests for diarization-on-supplied-transcription plumbing in LocalDubbingPipeline."""

    def _make_transcription(self, *, with_speakers: bool):
        from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord

        words1 = [
            TranscriptionWord(start=0.0, end=0.5, word="Hello"),
            TranscriptionWord(start=0.5, end=1.0, word="world"),
        ]
        words2 = [
            TranscriptionWord(start=2.0, end=2.5, word="Goodbye"),
            TranscriptionWord(start=2.5, end=3.0, word="now"),
        ]
        speaker = "SPEAKER_00" if with_speakers else None
        segments = [
            TranscriptionSegment(start=0.0, end=1.0, text="Hello world", words=words1, speaker=speaker),
            TranscriptionSegment(start=2.0, end=3.0, text="Goodbye now", words=words2, speaker=speaker),
        ]
        return Transcription(segments=segments, language="en")

    def _install_fakes(self, pipeline, monkeypatch):
        """Replace heavy stages with no-op fakes so process() can run end-to-end."""
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        # Skip separation, translation, TTS, and synchronization; we only care
        # about how the pipeline handles the supplied transcription.
        def fake_init_translator(self):
            class FakeTranslator:
                def translate_segments(self, segments, target_lang, source_lang):
                    from videopython.ai.dubbing.models import TranslatedSegment

                    return [
                        TranslatedSegment(
                            original_segment=s,
                            translated_text=s.text,
                            source_lang=source_lang,
                            target_lang=target_lang,
                        )
                        for s in segments
                    ]

            self._translator = FakeTranslator()

        def fake_init_tts(self, language="en"):
            class FakeTTS:
                def generate_audio(self, text, voice_sample=None, voice_sample_path=None):
                    sample_rate = 24000
                    duration = 0.2
                    frame_count = int(sample_rate * duration)
                    data = np.zeros(frame_count, dtype=np.float32)
                    metadata = AudioMetadata(
                        sample_rate=sample_rate,
                        channels=1,
                        sample_width=2,
                        duration_seconds=duration,
                        frame_count=frame_count,
                    )
                    return Audio(data, metadata)

            self._tts = FakeTTS()

        monkeypatch.setattr(LocalDubbingPipeline, "_init_translator", fake_init_translator)
        monkeypatch.setattr(LocalDubbingPipeline, "_init_tts", fake_init_tts)

    def test_supplied_with_speakers_skips_diarization(self, sample_audio, monkeypatch):
        """A pre-diarized transcription must be passed through unchanged."""
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        pipeline = LocalDubbingPipeline()
        self._install_fakes(pipeline, monkeypatch)

        transcription = self._make_transcription(with_speakers=True)

        # If diarize_transcription is called, fail loudly.
        def must_not_be_called(*args, **kwargs):
            raise AssertionError("diarize_transcription should not run for pre-diarized input")

        # Stash a sentinel transcriber so _maybe_unload doesn't NPE.
        class FakeTranscriber:
            diarize_transcription = staticmethod(must_not_be_called)

            def unload(self):
                pass

        # _init_transcriber would replace this with a real model — patch it out.
        monkeypatch.setattr(
            LocalDubbingPipeline,
            "_init_transcriber",
            lambda self, enable_diarization=False: setattr(self, "_transcriber", FakeTranscriber()),
        )

        result = pipeline.process(
            source_audio=sample_audio,
            target_lang="es",
            preserve_background=False,
            voice_clone=False,
            enable_diarization=True,  # must be ignored
            transcription=transcription,
        )

        assert result.source_transcription.speakers == {"SPEAKER_00"}

    def test_supplied_no_speakers_with_diarization_runs_diarize(self, sample_audio, monkeypatch):
        """No-speaker transcription + enable_diarization=True must trigger diarize_transcription."""
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
        from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord

        pipeline = LocalDubbingPipeline()
        self._install_fakes(pipeline, monkeypatch)

        captured: dict = {}

        class FakeTranscriber:
            def diarize_transcription(self, audio, transcription):
                captured["called"] = True
                # Return a fresh transcription with speakers attached.
                new_words = [
                    TranscriptionWord(start=0.0, end=0.5, word="Hello", speaker="A"),
                    TranscriptionWord(start=0.5, end=1.0, word="world", speaker="A"),
                    TranscriptionWord(start=2.0, end=2.5, word="Goodbye", speaker="B"),
                    TranscriptionWord(start=2.5, end=3.0, word="now", speaker="B"),
                ]
                segs = [
                    TranscriptionSegment(start=0.0, end=1.0, text="Hello world", words=new_words[:2], speaker="A"),
                    TranscriptionSegment(start=2.0, end=3.0, text="Goodbye now", words=new_words[2:], speaker="B"),
                ]
                return Transcription(segments=segs, language=transcription.language)

            def unload(self):
                pass

        monkeypatch.setattr(
            LocalDubbingPipeline,
            "_init_transcriber",
            lambda self, enable_diarization=False: setattr(self, "_transcriber", FakeTranscriber()),
        )

        transcription = self._make_transcription(with_speakers=False)
        assert not transcription.speakers

        result = pipeline.process(
            source_audio=sample_audio,
            target_lang="es",
            preserve_background=False,
            voice_clone=False,
            enable_diarization=True,
            transcription=transcription,
        )

        assert captured.get("called") is True
        assert result.source_transcription.speakers == {"A", "B"}

    def test_supplied_no_speakers_without_diarization_uses_as_is(self, sample_audio, monkeypatch):
        """No-speaker transcription + enable_diarization=False must NOT call diarize_transcription."""
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        pipeline = LocalDubbingPipeline()
        self._install_fakes(pipeline, monkeypatch)

        def must_not_be_called(*args, **kwargs):
            raise AssertionError("diarize_transcription should not run when enable_diarization=False")

        class FakeTranscriber:
            diarize_transcription = staticmethod(must_not_be_called)

            def unload(self):
                pass

        monkeypatch.setattr(
            LocalDubbingPipeline,
            "_init_transcriber",
            lambda self, enable_diarization=False: setattr(self, "_transcriber", FakeTranscriber()),
        )

        transcription = self._make_transcription(with_speakers=False)
        result = pipeline.process(
            source_audio=sample_audio,
            target_lang="es",
            preserve_background=False,
            voice_clone=False,
            enable_diarization=False,
            transcription=transcription,
        )

        assert not result.source_transcription.speakers


class TestVoiceSampleCache:
    """Pipeline encodes each speaker's voice sample to a temp WAV exactly once."""

    def _make_two_speaker_transcription(self):
        from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord

        segs = [
            TranscriptionSegment(
                start=0.0,
                end=1.0,
                text="hello there",
                words=[
                    TranscriptionWord(start=0.0, end=0.5, word="hello", speaker="A"),
                    TranscriptionWord(start=0.5, end=1.0, word="there", speaker="A"),
                ],
                speaker="A",
            ),
            TranscriptionSegment(
                start=1.0,
                end=2.0,
                text="hi back",
                words=[
                    TranscriptionWord(start=1.0, end=1.5, word="hi", speaker="A"),
                    TranscriptionWord(start=1.5, end=2.0, word="back", speaker="A"),
                ],
                speaker="A",
            ),
            TranscriptionSegment(
                start=2.0,
                end=3.5,
                text="and another",
                words=[
                    TranscriptionWord(start=2.0, end=2.5, word="and", speaker="B"),
                    TranscriptionWord(start=2.5, end=3.5, word="another", speaker="B"),
                ],
                speaker="B",
            ),
        ]
        return Transcription(segments=segs, language="en")

    def test_voice_samples_encoded_once_per_speaker(self, sample_audio, monkeypatch):
        """Three segments across two speakers => exactly two Audio.save calls."""
        from videopython.ai.dubbing.models import TranslatedSegment
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
        from videopython.base.audio import audio as audio_mod

        save_count = {"n": 0}
        tts_calls: list[dict] = []

        original_save = audio_mod.Audio.save

        def counting_save(self, file_path, format=None):
            save_count["n"] += 1
            from pathlib import Path as _Path

            _Path(file_path).write_bytes(b"fake wav")

        monkeypatch.setattr(audio_mod.Audio, "save", counting_save)

        # Make voice-sample extraction return one sample per speaker so the
        # cache has something to encode. _extract_voice_samples relies on
        # segment durations >= min_duration=3s, which our short fixture lacks;
        # short-circuit to a deterministic per-speaker map.
        def fake_extract(self, audio, transcription, min_duration=3.0, max_duration=10.0):
            return {"A": sample_audio, "B": sample_audio}

        monkeypatch.setattr(LocalDubbingPipeline, "_extract_voice_samples", fake_extract)

        def fake_init_translator(self):
            class FakeTranslator:
                def translate_segments(self, segments, target_lang, source_lang):
                    return [
                        TranslatedSegment(
                            original_segment=s,
                            translated_text=s.text,
                            source_lang=source_lang,
                            target_lang=target_lang,
                        )
                        for s in segments
                    ]

            self._translator = FakeTranslator()

        def fake_init_tts(self, language="en"):
            class FakeTTS:
                def generate_audio(self, text, voice_sample=None, voice_sample_path=None):
                    tts_calls.append({"voice_sample_path": voice_sample_path, "voice_sample": voice_sample})
                    sample_rate = 24000
                    duration = 0.2
                    frame_count = int(sample_rate * duration)
                    data = np.zeros(frame_count, dtype=np.float32)
                    metadata = AudioMetadata(
                        sample_rate=sample_rate,
                        channels=1,
                        sample_width=2,
                        duration_seconds=duration,
                        frame_count=frame_count,
                    )
                    return Audio(data, metadata)

            self._tts = FakeTTS()

        monkeypatch.setattr(LocalDubbingPipeline, "_init_translator", fake_init_translator)
        monkeypatch.setattr(LocalDubbingPipeline, "_init_tts", fake_init_tts)

        # Skip the timing synchronizer — it would call real ffmpeg via
        # time_stretch on the fake-saved WAVs and fail. We're only checking
        # encode counts and TTS argument plumbing here.
        class FakeSynchronizer:
            def synchronize_segments(self, segments, durations):
                return segments, []

            def assemble_with_timing(self, segments, start_times, total_duration):
                return segments[0] if segments else sample_audio

        monkeypatch.setattr(
            LocalDubbingPipeline,
            "_init_synchronizer",
            lambda self: setattr(self, "_synchronizer", FakeSynchronizer()),
        )

        pipeline = LocalDubbingPipeline()
        pipeline.process(
            source_audio=sample_audio,
            target_lang="es",
            preserve_background=False,
            voice_clone=True,
            enable_diarization=False,
            transcription=self._make_two_speaker_transcription(),
        )

        # Restore so cleanup paths don't accidentally use the fake.
        monkeypatch.setattr(audio_mod.Audio, "save", original_save)

        # Two speakers => two encodes, regardless of segment count.
        assert save_count["n"] == 2

        # Every TTS call should receive a path, not an Audio object.
        assert len(tts_calls) == 3
        for call in tts_calls:
            assert call["voice_sample_path"] is not None
            assert call["voice_sample"] is None

        # Speaker A reuses the same path across both segments.
        a_paths = {str(tts_calls[0]["voice_sample_path"]), str(tts_calls[1]["voice_sample_path"])}
        assert len(a_paths) == 1
        # Speaker B uses a different path.
        assert str(tts_calls[2]["voice_sample_path"]) != next(iter(a_paths))

    def test_voice_clone_disabled_skips_cache(self, sample_audio, monkeypatch):
        """voice_clone=False => no voice-sample encodes, TTS gets no path."""
        from videopython.ai.dubbing.models import TranslatedSegment
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
        from videopython.base.audio import audio as audio_mod

        save_count = {"n": 0}
        tts_calls: list[dict] = []

        def counting_save(self, file_path, format=None):
            save_count["n"] += 1

        monkeypatch.setattr(audio_mod.Audio, "save", counting_save)

        def fake_init_translator(self):
            class FakeTranslator:
                def translate_segments(self, segments, target_lang, source_lang):
                    return [
                        TranslatedSegment(
                            original_segment=s,
                            translated_text=s.text,
                            source_lang=source_lang,
                            target_lang=target_lang,
                        )
                        for s in segments
                    ]

            self._translator = FakeTranslator()

        def fake_init_tts(self, language="en"):
            class FakeTTS:
                def generate_audio(self, text, voice_sample=None, voice_sample_path=None):
                    tts_calls.append({"voice_sample_path": voice_sample_path})
                    sample_rate = 24000
                    duration = 0.2
                    frame_count = int(sample_rate * duration)
                    data = np.zeros(frame_count, dtype=np.float32)
                    metadata = AudioMetadata(
                        sample_rate=sample_rate,
                        channels=1,
                        sample_width=2,
                        duration_seconds=duration,
                        frame_count=frame_count,
                    )
                    return Audio(data, metadata)

            self._tts = FakeTTS()

        monkeypatch.setattr(LocalDubbingPipeline, "_init_translator", fake_init_translator)
        monkeypatch.setattr(LocalDubbingPipeline, "_init_tts", fake_init_tts)

        class FakeSynchronizer:
            def synchronize_segments(self, segments, durations):
                return segments, []

            def assemble_with_timing(self, segments, start_times, total_duration):
                return segments[0] if segments else sample_audio

        monkeypatch.setattr(
            LocalDubbingPipeline,
            "_init_synchronizer",
            lambda self: setattr(self, "_synchronizer", FakeSynchronizer()),
        )

        pipeline = LocalDubbingPipeline()
        pipeline.process(
            source_audio=sample_audio,
            target_lang="es",
            preserve_background=False,
            voice_clone=False,
            enable_diarization=False,
            transcription=self._make_two_speaker_transcription(),
        )

        assert save_count["n"] == 0
        for call in tts_calls:
            assert call["voice_sample_path"] is None


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


class TestReplaceAudioStreamFromAudio:
    """Tests for the streaming-audio variant of replace_audio_stream."""

    def test_missing_video_raises(self, tmp_path, sample_audio):
        from videopython.ai.dubbing.remux import replace_audio_stream_from_audio

        with pytest.raises(FileNotFoundError, match="Video file not found"):
            replace_audio_stream_from_audio(tmp_path / "missing.mp4", sample_audio, tmp_path / "out.mp4")

    def test_ffmpeg_failure_raises_remux_error(self, tmp_path, sample_audio, monkeypatch):
        """Non-zero ffmpeg exit code is wrapped in RemuxError with stderr."""
        import videopython.ai.dubbing.remux as remux_mod

        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")

        class FakeProcess:
            returncode = 1

            def communicate(self, _stdin):
                return b"", b"simulated ffmpeg error"

        def fake_popen(cmd, stdin=None, stderr=None):
            return FakeProcess()

        monkeypatch.setattr(remux_mod.subprocess, "Popen", fake_popen)

        with pytest.raises(remux_mod.RemuxError, match="simulated ffmpeg error"):
            remux_mod.replace_audio_stream_from_audio(video, sample_audio, tmp_path / "out.mp4")

    def test_ffmpeg_command_pipes_audio_via_stdin(self, tmp_path, sample_audio, monkeypatch):
        """Video is read from disk; audio is piped as WAV via stdin."""
        import videopython.ai.dubbing.remux as remux_mod

        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")
        out = tmp_path / "out.mp4"

        captured: dict = {}

        class FakeProcess:
            returncode = 0

            def communicate(self, stdin_bytes):
                captured["stdin_bytes"] = stdin_bytes
                return b"", b""

        def fake_popen(cmd, stdin=None, stderr=None):
            captured["cmd"] = cmd
            captured["stdin_kwarg"] = stdin
            return FakeProcess()

        monkeypatch.setattr(remux_mod.subprocess, "Popen", fake_popen)

        remux_mod.replace_audio_stream_from_audio(video, sample_audio, out)

        cmd = captured["cmd"]
        assert cmd[0] == "ffmpeg"
        # First input is the video file; second is "-" (stdin).
        i_indices = [i for i, v in enumerate(cmd) if v == "-i"]
        assert len(i_indices) == 2
        assert cmd[i_indices[0] + 1] == str(video)
        assert cmd[i_indices[1] + 1] == "-"
        # Stdin is requested.
        assert captured["stdin_kwarg"] is not None
        # Map: video from input 0, audio from input 1.
        map_indices = [i for i, v in enumerate(cmd) if v == "-map"]
        assert len(map_indices) == 2
        assert cmd[map_indices[0] + 1] == "0:v:0"
        assert cmd[map_indices[1] + 1] == "1:a:0"
        assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
        assert "-shortest" in cmd
        assert cmd[-1] == str(out)
        # WAV bytes were written: header "RIFF" then "WAVE".
        assert captured["stdin_bytes"][:4] == b"RIFF"
        assert captured["stdin_bytes"][8:12] == b"WAVE"


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
        """dub_file extracts audio, runs pipeline, and streams dubbed audio to ffmpeg.

        The dubbed Audio is piped directly to ffmpeg via
        ``replace_audio_stream_from_audio`` — there is no temp WAV on disk.
        Verifies the call sequence and that the in-memory Audio object is
        forwarded to the remux helper.
        """
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

        def fake_replace(video_path, audio, output_path, **kwargs):
            calls.append(
                (
                    "replace",
                    {
                        "video_path": str(video_path),
                        "audio": audio,
                        "output_path": str(output_path),
                    },
                )
            )

        monkeypatch.setattr(remux_mod, "replace_audio_stream_from_audio", fake_replace)

        dubber = VideoDubber()
        result = dubber.dub_file(
            input_path=input_path,
            output_path=output_path,
            target_lang="es",
        )

        names = [c[0] for c in calls]
        assert names == ["from_path", "process", "replace"]
        assert calls[0][1]["file_path"] == str(input_path)
        assert calls[1][1]["source_audio"] is sample_audio
        assert calls[1][1]["target_lang"] == "es"
        assert calls[2][1]["video_path"] == str(input_path)
        assert calls[2][1]["output_path"] == str(output_path)
        # Dubbed audio is forwarded by reference, not via a temp file.
        assert calls[2][1]["audio"] is sample_audio
        assert result is fake_result

    def test_dub_file_propagates_remux_error(self, tmp_path, sample_audio, sample_segment, monkeypatch):
        """RemuxError from the streaming helper propagates out of dub_file."""
        import videopython.ai.dubbing.remux as remux_mod
        from videopython.ai.dubbing import VideoDubber
        from videopython.ai.dubbing.models import DubbingResult, TranslatedSegment
        from videopython.base.audio import audio as audio_mod
        from videopython.base.text.transcription import Transcription

        input_path = tmp_path / "in.mp4"
        input_path.write_bytes(b"fake")

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

        def failing_replace(**kwargs):
            raise remux_mod.RemuxError("boom")

        monkeypatch.setattr(remux_mod, "replace_audio_stream_from_audio", failing_replace)

        dubber = VideoDubber()
        with pytest.raises(remux_mod.RemuxError, match="boom"):
            dubber.dub_file(
                input_path=input_path,
                output_path=tmp_path / "out.mp4",
                target_lang="es",
            )


class TestPeakMatch:
    """Tests for the _peak_match helper used to align dubbed loudness to source."""

    def _make_audio(self, peak: float, duration: float = 1.0, sample_rate: int = 24000) -> Audio:
        frame_count = int(duration * sample_rate)
        data = np.full(frame_count, peak, dtype=np.float32)
        metadata = AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=duration,
            frame_count=frame_count,
        )
        return Audio(data, metadata)

    def test_scales_to_match_reference_peak(self):
        from videopython.ai.dubbing.pipeline import _peak_match

        target = self._make_audio(0.3)
        reference = self._make_audio(0.9)

        out = _peak_match(target, reference)

        assert abs(float(np.max(np.abs(out.data))) - 0.9) < 1e-5
        # Original target buffer must not be mutated.
        assert abs(float(np.max(np.abs(target.data))) - 0.3) < 1e-5

    def test_silent_target_is_returned_as_is(self):
        from videopython.ai.dubbing.pipeline import _peak_match

        target = self._make_audio(0.0)
        reference = self._make_audio(0.5)

        out = _peak_match(target, reference)
        assert out is target

    def test_silent_reference_is_no_op(self):
        from videopython.ai.dubbing.pipeline import _peak_match

        target = self._make_audio(0.4)
        reference = self._make_audio(0.0)

        out = _peak_match(target, reference)
        assert out is target

    def test_near_unit_scale_is_no_op(self):
        """Tiny scale factors skip allocation — keeps the common 'already matched' path cheap."""
        from videopython.ai.dubbing.pipeline import _peak_match

        target = self._make_audio(0.5001)
        reference = self._make_audio(0.5)

        out = _peak_match(target, reference)
        assert out is target


class TestPipelineSpeechRegionGating:
    """Pipeline calls separate_regions with merged transcription regions, not separate()."""

    def test_separate_regions_called_with_transcription_segments(self, sample_audio, monkeypatch):
        """Pipeline derives speech regions from transcription and forwards to separate_regions."""
        from videopython.ai.dubbing.models import SeparatedAudio
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
        from videopython.ai.understanding.separation import AudioSeparator
        from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord

        captured: dict = {}

        def fake_separate_regions(self, audio, regions, **kwargs):
            captured["regions"] = list(regions)
            silent = np.zeros_like(audio.data, dtype=np.float32)
            stereo_meta = AudioMetadata(
                sample_rate=audio.metadata.sample_rate,
                channels=audio.metadata.channels,
                sample_width=audio.metadata.sample_width,
                duration_seconds=audio.metadata.duration_seconds,
                frame_count=audio.metadata.frame_count,
            )
            return SeparatedAudio(
                vocals=Audio(silent, stereo_meta),
                background=Audio(audio.data.astype(np.float32), stereo_meta),
                original=audio,
                music=None,
                effects=None,
            )

        def fail_separate(self, audio):
            raise AssertionError("separate() should not be called when transcription is available")

        monkeypatch.setattr(AudioSeparator, "separate_regions", fake_separate_regions)
        monkeypatch.setattr(AudioSeparator, "separate", fail_separate)
        monkeypatch.setattr(
            LocalDubbingPipeline,
            "_init_separator",
            lambda self: setattr(self, "_separator", AudioSeparator()),
        )

        # Translator and TTS as fakes so we don't touch real models.
        def fake_init_translator(self):
            class FakeTranslator:
                def translate_segments(self, segments, target_lang, source_lang):
                    return [
                        TranslatedSegment(
                            original_segment=s,
                            translated_text=s.text,
                            source_lang=source_lang,
                            target_lang=target_lang,
                        )
                        for s in segments
                    ]

            self._translator = FakeTranslator()

        def fake_init_tts(self, language="en"):
            class FakeTTS:
                def generate_audio(self, text, voice_sample=None, voice_sample_path=None):
                    sr = 24000
                    n = int(sr * 0.2)
                    data = np.zeros(n, dtype=np.float32)
                    return Audio(
                        data,
                        AudioMetadata(
                            sample_rate=sr,
                            channels=1,
                            sample_width=2,
                            duration_seconds=0.2,
                            frame_count=n,
                        ),
                    )

            self._tts = FakeTTS()

        monkeypatch.setattr(LocalDubbingPipeline, "_init_translator", fake_init_translator)
        monkeypatch.setattr(LocalDubbingPipeline, "_init_tts", fake_init_tts)

        # Two distant speech segments inside a 10s source.
        words = [TranscriptionWord(start=1.0, end=2.0, word="hello there")]
        words2 = [TranscriptionWord(start=7.0, end=8.0, word="goodbye now")]
        segs = [
            TranscriptionSegment(start=1.0, end=2.0, text="hello there", words=words),
            TranscriptionSegment(start=7.0, end=8.0, text="goodbye now", words=words2),
        ]
        transcription = Transcription(segments=segs, language="en")

        # Build a 10-second source audio so segment timings fall inside it.
        sr = 24000
        long_data = np.zeros(int(10 * sr), dtype=np.float32)
        long_audio = Audio(
            long_data,
            AudioMetadata(
                sample_rate=sr,
                channels=1,
                sample_width=2,
                duration_seconds=10.0,
                frame_count=len(long_data),
            ),
        )

        pipeline = LocalDubbingPipeline()
        pipeline.process(
            source_audio=long_audio,
            target_lang="es",
            preserve_background=True,
            voice_clone=False,
            enable_diarization=False,
            transcription=transcription,
        )

        # Default _merge_regions: pad=0.5, merge_gap=1.0. The two regions are
        # 5s apart so they don't merge; each is padded by 0.5 on each side.
        assert captured["regions"] == [(0.5, 2.5), (6.5, 8.5)]


class TestPipelineEmptyTranslationSkipped:
    """Pipeline skips TTS for empty/punctuation-translated segments."""

    def test_empty_translated_text_is_not_tts_called(self, sample_audio, monkeypatch):
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
        from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord

        tts_calls: list[str] = []

        def fake_init_translator(self):
            class FakeTranslator:
                def translate_segments(self, segments, target_lang, source_lang):
                    # Mimic the real filter: punctuation-only -> "".
                    out = []
                    for s in segments:
                        text = s.text
                        translated = "[X]" if any(c.isalnum() for c in text) and len(text.strip()) >= 2 else ""
                        out.append(
                            TranslatedSegment(
                                original_segment=s,
                                translated_text=translated,
                                source_lang=source_lang,
                                target_lang=target_lang,
                            )
                        )
                    return out

            self._translator = FakeTranslator()

        def fake_init_tts(self, language="en"):
            class FakeTTS:
                def generate_audio(self, text, voice_sample=None, voice_sample_path=None):
                    tts_calls.append(text)
                    sr = 24000
                    n = int(sr * 0.2)
                    return Audio(
                        np.zeros(n, dtype=np.float32),
                        AudioMetadata(
                            sample_rate=sr,
                            channels=1,
                            sample_width=2,
                            duration_seconds=0.2,
                            frame_count=n,
                        ),
                    )

            self._tts = FakeTTS()

        monkeypatch.setattr(LocalDubbingPipeline, "_init_translator", fake_init_translator)
        monkeypatch.setattr(LocalDubbingPipeline, "_init_tts", fake_init_tts)

        words = [TranscriptionWord(start=0.0, end=1.0, word="hello")]
        segs = [
            TranscriptionSegment(start=0.0, end=1.0, text="hello there", words=words),
            TranscriptionSegment(start=1.0, end=2.0, text=" .", words=words),
            TranscriptionSegment(start=2.0, end=3.0, text="how are you", words=words),
        ]
        transcription = Transcription(segments=segs, language="en")

        pipeline = LocalDubbingPipeline()
        pipeline.process(
            source_audio=sample_audio,
            target_lang="es",
            preserve_background=False,
            voice_clone=False,
            enable_diarization=False,
            transcription=transcription,
        )

        # Only two TTS calls; the punctuation-only segment was skipped.
        assert tts_calls == ["[X]", "[X]"]
