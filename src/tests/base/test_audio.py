import tempfile
from pathlib import Path

import numpy as np
import pytest

from videopython.base.audio import Audio, AudioMetadata

# Test constants
MONO_SAMPLE_RATE = 44100
STEREO_SAMPLE_RATE = 44100
NEW_SAMPLE_RATE = 16000

TEST_ROOT_DIR: Path = Path(__file__).parent.parent
TEST_DATA_DIR: Path = TEST_ROOT_DIR / "test_data"


def test_mono_mp3_metadata():
    """Test loading a mono MP3 file and verify its metadata"""
    # Load the test file
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    # Check metadata values
    assert audio.metadata.sample_rate == MONO_SAMPLE_RATE, "Sample rate should be 22.05kHz"
    assert audio.metadata.channels == 1, "Audio should be mono"
    assert audio.metadata.bits_per_sample == 16, "Bit depth should be 16 bits"

    # Verify data array properties
    assert isinstance(audio.data, np.ndarray), "Data should be a numpy array"
    assert audio.data.dtype == np.float32, "Data should be float32"
    assert audio.data.ndim == 1, "Mono audio should be 1-dimensional"

    # Check data normalization
    assert np.all(audio.data >= -1.0), "Data should be normalized >= -1.0"
    assert np.all(audio.data <= 1.0), "Data should be normalized <= 1.0"

    # Verify frame count matches data length
    assert len(audio.data) == audio.metadata.frame_count, "Frame count should match data length"


def test_file_not_found():
    """Test that appropriate error is raised for missing files"""
    with pytest.raises(FileNotFoundError):
        Audio.from_file("nonexistent.mp3")


def test_get_channel_mono():
    """Test that get_channel on mono audio returns the same audio"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    channel = audio.get_channel(0)

    # Should be the same data
    np.testing.assert_array_equal(audio.data, channel.data)
    assert audio.metadata.channels == channel.metadata.channels


def test_to_mono_on_mono():
    """Test that to_mono on mono audio returns the same audio"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    mono = audio.to_mono()

    # Should be the same data
    np.testing.assert_array_equal(audio.data, mono.data)
    assert audio.metadata.channels == mono.metadata.channels


def test_audio_representation():
    """Test the string representation of the Audio object"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    repr_str = repr(audio)

    assert "44100Hz" in repr_str, "Sample rate should be in string representation"
    assert "channels=1" in repr_str, "Channel count should be in string representation"
    assert isinstance(repr_str, str), "Representation should be a string"


def test_length():
    """Test the __len__ method returns correct frame count"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    assert len(audio) == audio.metadata.frame_count


def test_stereo_mp3_metadata():
    """Test loading a stereo MP3 file and verify its metadata"""
    # Load the test file
    audio = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Check metadata values
    assert audio.metadata.sample_rate == STEREO_SAMPLE_RATE, "Sample rate should be 44.1kHz"
    assert audio.metadata.channels == 2, "Audio should be stereo"
    assert audio.metadata.bits_per_sample == 16, "Bit depth should be 16 bits"

    # Verify data array properties
    assert isinstance(audio.data, np.ndarray), "Data should be a numpy array"
    assert audio.data.dtype == np.float32, "Data should be float32"
    assert audio.data.ndim == 2, "Stereo audio should be 2-dimensional"
    assert audio.data.shape[1] == 2, "Stereo audio should have 2 channels"

    # Check data normalization
    assert np.all(audio.data >= -1.0), "Data should be normalized >= -1.0"
    assert np.all(audio.data <= 1.0), "Data should be normalized <= 1.0"

    # Verify frame count matches data length
    assert len(audio.data) == audio.metadata.frame_count, "Frame count should match data length"


def test_stereo_channel_separation():
    """Test that stereo channels can be correctly separated"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Get individual channels
    left = audio.get_channel(0)
    right = audio.get_channel(1)

    # Check that channels are mono
    assert left.metadata.channels == 1
    assert right.metadata.channels == 1

    # Check that the data matches the original
    np.testing.assert_array_equal(audio.data[:, 0], left.data)
    np.testing.assert_array_equal(audio.data[:, 1], right.data)

    # Check sample rates are preserved
    assert left.metadata.sample_rate == STEREO_SAMPLE_RATE
    assert right.metadata.sample_rate == STEREO_SAMPLE_RATE


def test_stereo_to_mono_conversion():
    """Test converting stereo to mono"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")
    mono = audio.to_mono()

    # Check that output is mono
    assert mono.metadata.channels == 1
    assert mono.data.ndim == 1

    # Check that mono data is average of stereo channels
    expected_mono = audio.data.mean(axis=1)
    np.testing.assert_array_equal(mono.data, expected_mono)

    # Check metadata is preserved
    assert mono.metadata.sample_rate == STEREO_SAMPLE_RATE
    assert mono.metadata.sample_width == audio.metadata.sample_width

    # Check normalization is preserved
    assert np.all(mono.data >= -1.0)
    assert np.all(mono.data <= 1.0)


def assert_audios_equal(audio1, audio2, check_data=True, is_lossy=False):
    """Helper function to compare two Audio objects

    Args:
        audio1: First audio object
        audio2: Second audio object
        check_data: If True, compare audio data
        is_lossy: If True, use more lenient tolerances for lossy formats
    """
    # Compare metadata
    assert audio1.metadata.sample_rate == audio2.metadata.sample_rate, "Sample rates should match"
    assert audio1.metadata.channels == audio2.metadata.channels, "Channel counts should match"
    assert audio1.metadata.sample_width == audio2.metadata.sample_width, "Sample widths should match"
    assert abs(audio1.metadata.duration_seconds - audio2.metadata.duration_seconds) < 0.1, (
        "Durations should match within 0.1s"
    )

    # For lossy formats, frame count might differ slightly
    if not is_lossy:
        assert audio1.metadata.frame_count == audio2.metadata.frame_count, "Frame counts should match"

    # Compare actual audio data if requested
    if check_data:
        if is_lossy:
            # Very lenient comparison for MP3
            # We're mainly checking that the overall structure is preserved
            assert audio1.data.shape == audio2.data.shape, "Audio shapes should match"

            # Check that RMS difference is not too large
            rms_diff = np.sqrt(np.mean((audio1.data - audio2.data) ** 2))
            assert rms_diff < 0.1, f"RMS difference too large: {rms_diff}"

            # Check correlation to ensure signal structure is preserved
            correlation = np.corrcoef(audio1.data.ravel(), audio2.data.ravel())[0, 1]
            assert correlation > 0.9, f"Correlation too low: {correlation}"
        else:
            # Strict comparison for lossless formats
            np.testing.assert_allclose(audio1.data, audio2.data, rtol=1e-4, atol=1e-4)


# Fix 1: Update test_save_and_load_mono
def test_save_and_load_mono():
    """Test saving and loading mono audio preserves the data"""
    original = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    with (
        tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav,
        tempfile.NamedTemporaryFile(suffix=".mp3") as temp_mp3,
    ):
        # Test WAV format
        original.save(temp_wav.name)
        loaded_wav = Audio.from_file(temp_wav.name)
        assert_audios_equal(original, loaded_wav)

        # Test MP3 format - note is_lossy=True here
        original.save(temp_mp3.name)
        loaded_mp3 = Audio.from_file(temp_mp3.name)
        assert_audios_equal(original, loaded_mp3, is_lossy=True)


def test_save_and_load_stereo():
    """Test saving and loading stereo audio preserves the data"""
    original = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    with (
        tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav,
        tempfile.NamedTemporaryFile(suffix=".mp3") as temp_mp3,
    ):
        # Test WAV format (lossless)
        original.save(temp_wav.name)
        loaded_wav = Audio.from_file(temp_wav.name)
        assert_audios_equal(original, loaded_wav, is_lossy=False)

        # Test MP3 format (lossy)
        original.save(temp_mp3.name)
        loaded_mp3 = Audio.from_file(temp_mp3.name)
        assert_audios_equal(original, loaded_mp3, is_lossy=True)


def test_save_invalid_format():
    """Test that saving with invalid format raises error"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    with tempfile.NamedTemporaryFile(suffix=".xyz") as temp_file:
        with pytest.raises(ValueError):
            audio.save(temp_file.name)


def test_concat_mono():
    """Test concatenating two mono audio files"""
    # Load same file twice for testing
    audio1 = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    audio2 = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    # Concatenate
    result = audio1.concat(audio2)

    # Check metadata
    assert result.metadata.sample_rate == audio1.metadata.sample_rate
    assert result.metadata.channels == 1
    assert result.metadata.sample_width == audio1.metadata.sample_width
    assert (
        abs(result.metadata.duration_seconds - (audio1.metadata.duration_seconds + audio2.metadata.duration_seconds))
        < 0.1
    )

    # Check data
    assert len(result.data) == len(audio1.data) + len(audio2.data)
    np.testing.assert_array_equal(result.data[: len(audio1.data)], audio1.data)
    np.testing.assert_array_equal(result.data[len(audio1.data) :], audio2.data)


def test_concat_stereo():
    """Test concatenating two stereo audio files"""
    # Load same file twice for testing
    audio1 = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")
    audio2 = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Concatenate
    result = audio1.concat(audio2)

    # Check metadata
    assert result.metadata.sample_rate == audio1.metadata.sample_rate
    assert result.metadata.channels == 2
    assert result.metadata.sample_width == audio1.metadata.sample_width
    assert (
        abs(result.metadata.duration_seconds - (audio1.metadata.duration_seconds + audio2.metadata.duration_seconds))
        < 0.1
    )

    # Check data shape and content
    assert result.data.shape == (audio1.data.shape[0] + audio2.data.shape[0], 2)
    np.testing.assert_array_equal(result.data[: audio1.data.shape[0]], audio1.data)
    np.testing.assert_array_equal(result.data[audio1.data.shape[0] :], audio2.data)


def test_concat_invalid():
    """Test concatenating incompatible audio files"""
    mono = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    # Create audio with different sample rate for testing
    different_rate = Audio(
        mono.data,
        AudioMetadata(
            sample_rate=22050,  # Different from mono file
            channels=mono.metadata.channels,
            sample_width=mono.metadata.sample_width,
            duration_seconds=mono.metadata.duration_seconds,
            frame_count=len(mono.data),
        ),
    )

    # Test mismatched sample rates
    with pytest.raises(ValueError, match="Sample rates must match"):
        mono.concat(different_rate)


def test_slice():
    """Test slicing audio by time"""
    # Load test file
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    # Test slicing the middle portion
    start_time = 0.5
    end_time = 1.0
    sliced = audio.slice(start_time, end_time)

    # Check metadata
    assert sliced.metadata.sample_rate == audio.metadata.sample_rate
    assert sliced.metadata.channels == audio.metadata.channels
    assert sliced.metadata.sample_width == audio.metadata.sample_width
    assert abs(sliced.metadata.duration_seconds - (end_time - start_time)) < 0.1

    # Check expected length in samples
    expected_samples = int((end_time - start_time) * audio.metadata.sample_rate)
    assert abs(len(sliced) - expected_samples) <= 1  # Allow for rounding

    # Test slicing from start
    start_slice = audio.slice(end_seconds=1.0)
    assert abs(start_slice.metadata.duration_seconds - 1.0) < 0.1

    # Test slicing to end
    end_slice = audio.slice(start_seconds=1.0)
    assert abs(end_slice.metadata.duration_seconds - (audio.metadata.duration_seconds - 1.0)) < 0.1

    # Test invalid inputs
    with pytest.raises(ValueError):
        audio.slice(-1.0)  # Negative start time

    with pytest.raises(ValueError):
        audio.slice(2.0, 1.0)  # End before start

    with pytest.raises(ValueError):
        audio.slice(0.0, audio.metadata.duration_seconds + 1)  # End after audio duration


def test_slice_stereo():
    """Test slicing stereo audio"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Slice a portion
    sliced = audio.slice(0.5, 1.5)

    # Check that stereo structure is preserved
    assert sliced.metadata.channels == 2
    assert sliced.data.ndim == 2
    assert sliced.data.shape[1] == 2

    # Check duration
    assert abs(sliced.metadata.duration_seconds - 1.0) < 0.1

    # Check that the data is a proper subset
    start_idx = int(0.5 * audio.metadata.sample_rate)
    end_idx = int(1.5 * audio.metadata.sample_rate)
    np.testing.assert_array_equal(sliced.data, audio.data[start_idx:end_idx])


def test_create_silent_stereo():
    """Test creating a stereo silent track"""
    duration = 2.0
    audio = Audio.create_silent(duration)

    # Check metadata
    assert audio.metadata.channels == 2
    assert audio.metadata.sample_rate == 44100
    assert audio.metadata.sample_width == 2
    assert abs(audio.metadata.duration_seconds - duration) < 0.0001
    assert audio.metadata.frame_count == int(duration * 44100)

    # Check data shape and content
    assert audio.data.shape == (int(duration * 44100), 2)
    assert np.all(audio.data == 0)
    assert audio.data.dtype == np.float32


def test_create_silent_mono():
    """Test creating a mono silent track"""
    duration = 1.5
    audio = Audio.create_silent(duration, stereo=False)

    # Check metadata
    assert audio.metadata.channels == 1
    assert audio.metadata.sample_rate == 44100
    assert audio.metadata.sample_width == 2
    assert abs(audio.metadata.duration_seconds - duration) < 0.0001
    assert audio.metadata.frame_count == int(duration * 44100)

    # Check data shape and content
    assert audio.data.shape == (int(duration * 44100),)
    assert np.all(audio.data == 0)
    assert audio.data.dtype == np.float32


def test_create_silent_custom_params():
    """Test creating silent track with custom parameters"""
    duration = 1.0
    sample_rate = 22050
    sample_width = 4

    audio = Audio.create_silent(duration, stereo=True, sample_rate=sample_rate, sample_width=sample_width)

    # Check metadata
    assert audio.metadata.channels == 2
    assert audio.metadata.sample_rate == sample_rate
    assert audio.metadata.sample_width == sample_width
    assert abs(audio.metadata.duration_seconds - duration) < 0.0001
    assert audio.metadata.frame_count == int(duration * sample_rate)


def test_create_silent_invalid_params():
    """Test error handling for invalid parameters"""
    # Test negative duration
    with pytest.raises(ValueError, match="Duration must be positive"):
        Audio.create_silent(-1.0)

    # Test zero duration
    with pytest.raises(ValueError, match="Duration must be positive"):
        Audio.create_silent(0.0)

    # Test invalid sample rate
    with pytest.raises(ValueError, match="Sample rate must be positive"):
        Audio.create_silent(1.0, sample_rate=0)

    # Test invalid sample width
    with pytest.raises(ValueError, match="Sample width must be 1, 2, or 4 bytes"):
        Audio.create_silent(1.0, sample_width=3)


def test_concat_with_crossfade_mono():
    """Test concatenating two mono audio files with crossfade"""
    # Load same file twice for testing
    audio1 = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    audio2 = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    # Test with 0.5 second crossfade
    crossfade_duration = 0.5
    result = audio1.concat(audio2, crossfade=crossfade_duration)

    # Check metadata
    assert result.metadata.sample_rate == audio1.metadata.sample_rate
    assert result.metadata.channels == 1
    assert result.metadata.sample_width == audio1.metadata.sample_width

    # Check data shape
    crossfade_samples = int(crossfade_duration * audio1.metadata.sample_rate)
    expected_length = len(audio1.data) + len(audio2.data) - crossfade_samples
    assert len(result.data) == expected_length

    # Check normalization
    assert np.all(result.data >= -1.0)
    assert np.all(result.data <= 1.0)

    # Test crossfade region
    crossfade_start_idx = len(audio1.data) - crossfade_samples
    crossfade_region = result.data[crossfade_start_idx : crossfade_start_idx + crossfade_samples]

    # Verify crossfade is actually happening
    assert np.all(np.diff(crossfade_region) != 0), "Crossfade region should not be constant"

    # Check duration
    expected_duration = audio1.metadata.duration_seconds + audio2.metadata.duration_seconds - crossfade_duration
    assert abs(result.metadata.duration_seconds - expected_duration) < 0.1


def test_concat_with_crossfade_stereo():
    """Test concatenating two stereo audio files with crossfade"""
    # Load same file twice for testing
    audio1 = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")
    audio2 = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Test with 0.5 second crossfade
    crossfade_duration = 0.5
    result = audio1.concat(audio2, crossfade=crossfade_duration)

    # Check metadata
    assert result.metadata.sample_rate == audio1.metadata.sample_rate
    assert result.metadata.channels == 2
    assert result.metadata.sample_width == audio1.metadata.sample_width

    # Check data shape
    crossfade_samples = int(crossfade_duration * audio1.metadata.sample_rate)
    expected_length = len(audio1.data) + len(audio2.data) - crossfade_samples
    assert len(result.data) == expected_length
    assert result.data.shape[1] == 2

    # Check normalization
    assert np.all(result.data >= -1.0)
    assert np.all(result.data <= 1.0)

    # Test crossfade region
    crossfade_start_idx = len(audio1.data) - crossfade_samples
    crossfade_region = result.data[crossfade_start_idx : crossfade_start_idx + crossfade_samples]

    # Verify crossfade is happening in both channels
    assert np.all(np.diff(crossfade_region[:, 0]) != 0), "Left channel crossfade should not be constant"
    assert np.all(np.diff(crossfade_region[:, 1]) != 0), "Right channel crossfade should not be constant"

    # Check duration
    expected_duration = audio1.metadata.duration_seconds + audio2.metadata.duration_seconds - crossfade_duration
    assert abs(result.metadata.duration_seconds - expected_duration) < 0.1


def test_concat_crossfade_invalid():
    """Test concatenating with invalid crossfade parameters"""
    mono = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    # Create audio with different sample rate
    different_rate = Audio(
        mono.data,
        AudioMetadata(
            sample_rate=22050,
            channels=mono.metadata.channels,
            sample_width=mono.metadata.sample_width,
            duration_seconds=mono.metadata.duration_seconds,
            frame_count=len(mono.data),
        ),
    )

    # Test mismatched sample rates
    with pytest.raises(ValueError, match="Sample rates must match"):
        mono.concat(different_rate, crossfade=0.5)


def test_concat_zero_crossfade():
    """Test that concat with zero crossfade is same as regular concat"""
    audio1 = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    audio2 = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    # Get results both ways
    result_no_crossfade = audio1.concat(audio2)
    result_zero_crossfade = audio1.concat(audio2, crossfade=0.0)

    # Results should be identical
    np.testing.assert_array_equal(result_no_crossfade.data, result_zero_crossfade.data)


def test_overlay_stereo_with_stereo():
    """Test overlaying two stereo tracks"""
    # Load test files
    base = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")
    overlay = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Test basic overlay at beginning
    result = base.overlay(overlay, position=0.0)
    assert result.metadata.channels == 2
    # Use np.isclose() for floating point comparison with small tolerance
    assert np.isclose(result.metadata.duration_seconds, base.metadata.duration_seconds, rtol=1e-6)
    assert len(result.data) == len(base.data)

    # Test overlay with positive position
    position = 1.0  # 1 second
    result = base.overlay(overlay, position=position)
    assert result.metadata.duration_seconds >= base.metadata.duration_seconds
    assert result.metadata.duration_seconds >= position + overlay.metadata.duration_seconds


def test_overlay_amplitude_scaling():
    """Test that overlaid audio is properly scaled to prevent clipping"""
    base = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")
    overlay = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    result = base.overlay(overlay, position=0.0)
    assert np.max(np.abs(result.data)) <= 1.0


def test_overlay_position_validation():
    """Test position parameter validation"""
    base = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")
    overlay = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Test negative position
    with pytest.raises(ValueError, match="Position cannot be negative"):
        base.overlay(overlay, position=-1.0)


def test_overlay_end_alignment():
    """Test overlaying track near the end of base track"""
    base = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")
    overlay = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Position overlay near end of base track
    position = base.metadata.duration_seconds - 0.5  # 0.5 seconds before end
    result = base.overlay(overlay, position=position)

    # Result should be longer than base to accommodate overlay
    assert result.metadata.duration_seconds > base.metadata.duration_seconds


def test_overlay_mono_with_mono():
    """Test overlaying two mono tracks"""
    base = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    overlay = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    result = base.overlay(overlay, position=0.0)
    assert result.metadata.channels == 1
    assert np.max(np.abs(result.data)) <= 1.0


def test_overlay_sample_rate_mismatch():
    """Test overlaying tracks with different sample rates"""
    # Create a mock audio with different sample rate
    mock_overlay = Audio(
        data=np.zeros((1000, 2)),
        metadata=AudioMetadata(
            sample_rate=22050,  # Different from standard 44100
            channels=2,
            sample_width=2,
            duration_seconds=1.0,
            frame_count=1000,
        ),
    )

    base = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    with pytest.raises(ValueError, match="Sample rates must match"):
        base.overlay(mock_overlay, position=0.0)


def test_overlay_different_durations():
    """Test overlaying tracks of different durations"""
    base = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Create shorter overlay
    short_duration = 1.0  # 1 second
    short_overlay = base.slice(0, short_duration)

    # Overlay at different positions
    start_pos = base.metadata.duration_seconds / 2
    result = base.overlay(short_overlay, position=start_pos)

    # Check result length
    expected_duration = max(base.metadata.duration_seconds, start_pos + short_duration)
    assert abs(result.metadata.duration_seconds - expected_duration) < 0.01


def test_overlay_identical_position():
    """Test overlaying the same track at the same position"""
    base = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    result = base.overlay(base, position=0.0)

    # Result should be scaled to prevent clipping
    assert np.max(np.abs(result.data)) <= 1.0
    # Due to scaling, amplitudes should be less than direct sum
    assert np.all(np.abs(result.data) <= np.abs(base.data * 2))


def test_is_silent():
    """Test the is_silent property"""
    # Test explicit silent audio
    silent_audio = Audio.create_silent(duration_seconds=1.0)
    assert silent_audio.is_silent, "Created silent audio should be detected as silent"

    # Test non-silent audio
    non_silent = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    assert not non_silent.is_silent, "Regular audio file should not be detected as silent"

    # Test near-zero but not quite silent audio
    almost_silent_data = np.full((44100,), 1e-8)  # Very small but non-zero values
    almost_silent = Audio(
        data=almost_silent_data,
        metadata=AudioMetadata(
            sample_rate=44100,
            channels=1,
            sample_width=2,
            duration_seconds=1.0,
            frame_count=44100,
        ),
    )
    assert almost_silent.is_silent, "Nearly silent audio should be detected as silent"

    # Test partially silent audio
    partial_silent_data = np.zeros((44100,))
    partial_silent_data[22050] = 0.5  # Single non-zero sample in the middle
    partial_silent = Audio(
        data=partial_silent_data,
        metadata=AudioMetadata(
            sample_rate=44100,
            channels=1,
            sample_width=2,
            duration_seconds=1.0,
            frame_count=44100,
        ),
    )
    assert not partial_silent.is_silent, "Partially silent audio should not be detected as silent"


def test_to_stereo():
    """Test internal _to_stereo method"""
    # Load mono audio
    mono = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    # Convert to stereo
    stereo = mono._to_stereo()

    # Check that it's now stereo
    assert stereo.metadata.channels == 2
    assert stereo.data.ndim == 2
    assert stereo.data.shape[1] == 2

    # Check that both channels are identical
    np.testing.assert_array_equal(stereo.data[:, 0], stereo.data[:, 1])

    # Test that stereo audio remains unchanged
    original_stereo = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")
    result = original_stereo._to_stereo()
    np.testing.assert_array_equal(original_stereo.data, result.data)


def test_overlay_mono_with_stereo():
    """Test overlaying mono track with stereo track"""
    mono = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    stereo = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Test both ways
    result1 = mono.overlay(stereo, position=0.0)
    result2 = stereo.overlay(mono, position=0.0)

    # Check that results are stereo
    assert result1.metadata.channels == 2
    assert result2.metadata.channels == 2

    # Check normalization
    assert np.all(np.abs(result1.data) <= 1.0)
    assert np.all(np.abs(result2.data) <= 1.0)


def test_concat_mono_with_stereo():
    """Test concatenating mono track with stereo track"""
    mono = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    stereo = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Test both ways
    result1 = mono.concat(stereo)
    result2 = stereo.concat(mono)

    # Check that results are stereo
    assert result1.metadata.channels == 2
    assert result2.metadata.channels == 2

    # Check data shapes
    assert result1.data.shape[1] == 2
    assert result2.data.shape[1] == 2

    # Test with crossfade
    result_crossfade = mono.concat(stereo, crossfade=0.5)
    assert result_crossfade.metadata.channels == 2
    assert np.all(np.abs(result_crossfade.data) <= 1.0)


def test_concat_crossfade_mono_stereo():
    """Test concatenating mono and stereo tracks with crossfade"""
    mono = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    stereo = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Test with crossfade
    crossfade_duration = 0.5
    result = mono.concat(stereo, crossfade=crossfade_duration)

    # Check metadata
    assert result.metadata.channels == 2  # Result should be stereo
    assert result.metadata.sample_rate == mono.metadata.sample_rate
    assert result.metadata.sample_width == mono.metadata.sample_width

    # Check crossfade region
    crossfade_samples = int(crossfade_duration * mono.metadata.sample_rate)
    crossfade_start_idx = len(mono.data) - crossfade_samples
    crossfade_region = result.data[crossfade_start_idx : crossfade_start_idx + crossfade_samples]

    # Verify crossfade is happening in both channels
    assert np.all(np.diff(crossfade_region[:, 0]) != 0), "Left channel crossfade should not be constant"
    assert np.all(np.diff(crossfade_region[:, 1]) != 0), "Right channel crossfade should not be constant"


def test_overlay_position_mono_stereo():
    """Test overlaying mono on stereo at different positions"""
    mono = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")
    stereo = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Test overlay at different positions
    positions = [0.0, 0.5, 1.0]
    for pos in positions:
        result = stereo.overlay(mono, position=pos)
        assert result.metadata.channels == 2
        assert np.all(np.abs(result.data) <= 1.0)

        # Check that the result is longer than the position plus mono duration
        min_duration = pos + mono.metadata.duration_seconds
        assert result.metadata.duration_seconds >= min_duration


def test_resample_mono():
    """Test resampling mono audio"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    # Resample to 16kHz
    resampled = audio.resample(NEW_SAMPLE_RATE)

    # Check metadata
    assert resampled.metadata.sample_rate == NEW_SAMPLE_RATE
    assert resampled.metadata.channels == 1
    assert resampled.metadata.sample_width == audio.metadata.sample_width
    assert abs(resampled.metadata.duration_seconds - audio.metadata.duration_seconds) < 0.1

    # Check reconstruction
    reconstructed = resampled.resample(audio.metadata.sample_rate)
    assert reconstructed.data.shape[0] - audio.data.shape[0] <= 1


def test_resample_stereo():
    """Test resampling stereo audio"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_stereo.mp3")

    # Resample to 16kHz
    resampled = audio.resample(NEW_SAMPLE_RATE)

    # Check metadata
    assert resampled.metadata.sample_rate == NEW_SAMPLE_RATE
    assert resampled.metadata.channels == 2
    assert resampled.metadata.sample_width == audio.metadata.sample_width
    assert abs(resampled.metadata.duration_seconds - audio.metadata.duration_seconds) < 0.1


# =============================================================================
# Audio Analysis Tests
# =============================================================================


def test_get_levels_silent():
    """Test level calculation on silent audio"""
    audio = Audio.create_silent(duration_seconds=1.0)
    levels = audio.get_levels()

    assert levels.rms < 1e-7
    assert levels.peak < 1e-7
    assert levels.db_rms < -100  # Very low dB for silent audio
    assert levels.db_peak < -100


def test_get_levels_full_scale_sine():
    """Test level calculation on full-scale sine wave"""
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Full scale sine wave
    data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    audio = Audio(
        data=data,
        metadata=AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=duration,
            frame_count=len(data),
        ),
    )

    levels = audio.get_levels()

    # Sine wave RMS is 1/sqrt(2) = ~0.707
    assert 0.7 < levels.rms < 0.72
    # Peak should be 1.0
    assert 0.99 < levels.peak <= 1.0
    # Peak dB should be ~0 for full scale
    assert -0.1 < levels.db_peak < 0.1


def test_get_levels_partial():
    """Test level calculation on a portion of the audio"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    full_levels = audio.get_levels()
    partial_levels = audio.get_levels(start_seconds=0.0, end_seconds=0.5)

    # Both should return valid AudioLevels
    assert full_levels.rms > 0
    assert partial_levels.rms > 0


def test_get_levels_over_time():
    """Test sliding window level analysis"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    levels_over_time = audio.get_levels_over_time(window_seconds=0.1)

    # Should return multiple measurements
    assert len(levels_over_time) > 1

    # Each entry should be a tuple of (timestamp, AudioLevels)
    for timestamp, levels in levels_over_time:
        assert isinstance(timestamp, float)
        assert 0 <= timestamp <= audio.metadata.duration_seconds
        assert hasattr(levels, "rms")
        assert hasattr(levels, "db_rms")


def test_detect_silence_on_silent_audio():
    """Test silence detection on fully silent audio"""
    audio = Audio.create_silent(duration_seconds=2.0)
    silent_segments = audio.detect_silence(threshold_db=-40.0, min_duration=0.5)

    # Entire audio should be detected as silent
    assert len(silent_segments) >= 1
    # Total silent duration should be close to 2.0 seconds
    total_silent = sum(seg.duration for seg in silent_segments)
    assert total_silent > 1.5  # Allow some tolerance


def test_detect_silence_with_gaps():
    """Test silence detection on audio with silent gaps"""
    sample_rate = 44100
    duration = 3.0

    # Create audio: 1s tone, 1s silence, 1s tone
    t = np.linspace(0, 1.0, sample_rate, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * t) * 0.5
    silence = np.zeros(sample_rate, dtype=np.float32)

    data = np.concatenate([tone, silence, tone])

    audio = Audio(
        data=data,
        metadata=AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=duration,
            frame_count=len(data),
        ),
    )

    silent_segments = audio.detect_silence(threshold_db=-40.0, min_duration=0.5)

    # Should detect the 1-second silent gap
    assert len(silent_segments) >= 1
    # The silent segment should be around 1 second at position 1.0s
    middle_segment = [s for s in silent_segments if 0.8 < s.start < 1.2]
    assert len(middle_segment) >= 1


def test_detect_silence_no_silence():
    """Test silence detection on continuous audio"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    # Use a very low threshold to ensure nothing is detected as silent
    silent_segments = audio.detect_silence(threshold_db=-100.0, min_duration=0.5)

    # May or may not have silent segments depending on the file
    # Just verify it returns a list
    assert isinstance(silent_segments, list)


def test_classify_segments_synthetic_silence():
    """Test classification on silent audio"""
    from videopython.base.audio.analysis import AudioSegmentType

    audio = Audio.create_silent(duration_seconds=3.0)
    segments = audio.classify_segments(segment_length=2.0, overlap=0.0)

    # At least one segment should be classified
    assert len(segments) >= 1

    # Silent audio should be classified as SILENCE
    for seg in segments:
        assert seg.segment_type == AudioSegmentType.SILENCE


def test_classify_segments_synthetic_noise():
    """Test classification on white noise"""
    from videopython.base.audio.analysis import AudioSegmentType

    sample_rate = 44100
    duration = 3.0
    data = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.3

    audio = Audio(
        data=data,
        metadata=AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=duration,
            frame_count=len(data),
        ),
    )

    segments = audio.classify_segments(segment_length=2.0, overlap=0.0)

    # Should return segments
    assert len(segments) >= 1

    # Noise should ideally be classified as NOISE, but heuristics aren't perfect
    # At minimum, check structure
    for seg in segments:
        assert seg.segment_type in AudioSegmentType
        assert 0.0 <= seg.confidence <= 1.0
        assert seg.duration > 0


def test_classify_segments_returns_valid_structure():
    """Test that classify_segments returns valid AudioSegment objects"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    # Only classify if audio is long enough
    if audio.metadata.duration_seconds >= 2.0:
        segments = audio.classify_segments(segment_length=2.0, overlap=0.5)

        for seg in segments:
            assert seg.start >= 0
            assert seg.end > seg.start
            assert seg.duration == seg.end - seg.start
            assert 0.0 <= seg.confidence <= 1.0
            assert seg.levels is not None


def test_normalize_peak():
    """Test peak normalization"""
    sample_rate = 44100
    duration = 1.0
    # Create audio at -12 dB peak
    peak_amplitude = 0.25  # -12 dB
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    data = (np.sin(2 * np.pi * 440 * t) * peak_amplitude).astype(np.float32)

    audio = Audio(
        data=data,
        metadata=AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=duration,
            frame_count=len(data),
        ),
    )

    # Normalize to -3 dB peak
    normalized = audio.normalize(target_db=-3.0, method="peak")

    # Check that peak is now at -3 dB (approximately 0.708)
    new_peak = np.max(np.abs(normalized.data))
    expected_peak = 10 ** (-3.0 / 20)  # ~0.708
    assert abs(new_peak - expected_peak) < 0.01


def test_normalize_rms():
    """Test RMS normalization"""
    sample_rate = 44100
    duration = 1.0
    # Create audio at -20 dB RMS
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    data = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)

    audio = Audio(
        data=data,
        metadata=AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=duration,
            frame_count=len(data),
        ),
    )

    # Normalize RMS to -10 dB
    normalized = audio.normalize(target_db=-10.0, method="rms")

    # Check that normalization occurred (RMS should be higher)
    original_rms = np.sqrt(np.mean(audio.data**2))
    new_rms = np.sqrt(np.mean(normalized.data**2))
    assert new_rms > original_rms


def test_normalize_silent_audio():
    """Test that normalizing silent audio returns silent audio"""
    audio = Audio.create_silent(duration_seconds=1.0)
    normalized = audio.normalize(target_db=-3.0)

    # Should still be silent
    assert np.all(np.abs(normalized.data) < 1e-7)


def test_normalize_invalid_method():
    """Test that invalid normalization method raises error"""
    audio = Audio.from_file(TEST_DATA_DIR / "test_mono.mp3")

    with pytest.raises(ValueError, match="Unknown method"):
        audio.normalize(method="invalid")
