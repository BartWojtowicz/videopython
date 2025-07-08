import numpy as np
import pytest

from videopython.base.text.overlay import TranscriptionOverlay
from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video

from .test_config import TEST_FONT_PATH


@pytest.fixture(scope="session")
def dummy_video():
    """Create a 10-second video with black frames."""
    return Video.from_image(np.zeros((360, 640, 3), dtype=np.uint8), fps=30, length_seconds=10.0)


@pytest.fixture(scope="session")
def dummy_transcription():
    """Create a dummy transcription with sample segments and words."""
    words_segment1 = [
        TranscriptionWord(start=0.0, end=0.5, word="Hello"),
        TranscriptionWord(start=0.5, end=1.0, word="world"),
        TranscriptionWord(start=1.0, end=1.2, word="this"),
        TranscriptionWord(start=1.2, end=1.4, word="is"),
        TranscriptionWord(start=1.4, end=1.8, word="test"),
    ]

    words_segment2 = [
        TranscriptionWord(start=2.0, end=2.3, word="Second"),
        TranscriptionWord(start=2.3, end=2.8, word="segment"),
        TranscriptionWord(start=2.8, end=3.0, word="of"),
        TranscriptionWord(start=3.0, end=3.5, word="transcription"),
    ]

    words_segment3 = [
        TranscriptionWord(start=5.0, end=5.4, word="Final"),
        TranscriptionWord(start=5.4, end=5.8, word="words"),
        TranscriptionWord(start=5.8, end=6.2, word="in"),
        TranscriptionWord(start=6.2, end=6.8, word="video"),
    ]

    segments = [
        TranscriptionSegment(start=0.0, end=1.8, text="Hello world this is test", words=words_segment1),
        TranscriptionSegment(start=2.0, end=3.5, text="Second segment of transcription", words=words_segment2),
        TranscriptionSegment(start=5.0, end=6.8, text="Final words in video", words=words_segment3),
    ]

    return Transcription(segments=segments)


def test_transcription_overlay_initialization(dummy_transcription):
    """Test that TranscriptionOverlay can be initialized with default parameters."""
    overlay = TranscriptionOverlay(font_filename=TEST_FONT_PATH)

    assert overlay.font_filename == TEST_FONT_PATH


def test_transcription_overlay_apply_basic(dummy_video, dummy_transcription):
    """Test basic application of TranscriptionOverlay to a video."""
    overlay = TranscriptionOverlay(font_filename=TEST_FONT_PATH, font_size=20)

    result_video = overlay.apply(video=dummy_video, transcription=dummy_transcription)

    # Check that the result is a Video object
    assert isinstance(result_video, Video)

    # Check that the video dimensions are preserved
    assert result_video.frame_shape == dummy_video.frame_shape
    assert result_video.fps == dummy_video.fps
    assert len(result_video.frames) == len(dummy_video.frames)

    # Check that audio is preserved
    assert result_video.audio == dummy_video.audio


def test_transcription_overlay_with_custom_settings(dummy_video, dummy_transcription):
    """Test TranscriptionOverlay with custom text styling parameters."""
    overlay = TranscriptionOverlay(
        font_filename=TEST_FONT_PATH,
        font_size=30,
        text_color=(255, 0, 0),  # Red text
        background_color=(0, 0, 0, 200),  # Black background with transparency
        highlight_color=(0, 255, 0),  # Green highlight
        position=(0.5, 0.8),
        box_width=0.6,
    )

    result_video = overlay.apply(video=dummy_video, transcription=dummy_transcription)

    # Check that the overlay was applied (video should still have same basic properties)
    assert isinstance(result_video, Video)
    assert result_video.frame_shape == dummy_video.frame_shape
    assert len(result_video.frames) == len(dummy_video.frames)


def test_transcription_overlay_no_background(dummy_video, dummy_transcription):
    """Test TranscriptionOverlay with no background color."""
    overlay = TranscriptionOverlay(font_filename=TEST_FONT_PATH, background_color=None)

    result_video = overlay.apply(video=dummy_video, transcription=dummy_transcription)

    assert isinstance(result_video, Video)
    assert len(result_video.frames) == len(dummy_video.frames)


def test_transcription_overlay_frame_content_changes(dummy_video, dummy_transcription):
    """Test that TranscriptionOverlay actually modifies frame content during active segments."""
    overlay = TranscriptionOverlay(font_filename=TEST_FONT_PATH)

    result_video = overlay.apply(video=dummy_video, transcription=dummy_transcription)

    # Frame at timestamp 1.0 should have text overlay (within first segment 0.0-1.8)
    frame_at_1s = result_video.frames[30]  # 30fps * 1s = frame 30
    original_frame_at_1s = dummy_video.frames[30]

    # The frames should be different (overlay was applied)
    assert not np.array_equal(frame_at_1s, original_frame_at_1s)

    # Frame at timestamp 4.0 should be unchanged (no active segment)
    frame_at_4s = result_video.frames[120]  # 30fps * 4s = frame 120
    original_frame_at_4s = dummy_video.frames[120]

    # The frames should be identical (no overlay applied)
    assert np.array_equal(frame_at_4s, original_frame_at_4s)


def test_transcription_with_offset(dummy_transcription):
    """Test that with_offset method correctly offsets all timings."""
    offset_time = 2.5
    offset_transcription = dummy_transcription.offset(offset_time)

    # Check that we have the same number of segments
    assert len(offset_transcription.segments) == len(dummy_transcription.segments)

    # Check that all segment timings are offset correctly
    for original_segment, offset_segment in zip(dummy_transcription.segments, offset_transcription.segments):
        assert offset_segment.start == original_segment.start + offset_time
        assert offset_segment.end == original_segment.end + offset_time
        assert offset_segment.text == original_segment.text

        # Check that word timings are also offset correctly
        assert len(offset_segment.words) == len(original_segment.words)
        for original_word, offset_word in zip(original_segment.words, offset_segment.words):
            assert offset_word.start == original_word.start + offset_time
            assert offset_word.end == original_word.end + offset_time
            assert offset_word.word == original_word.word

    # Verify specific timing examples
    first_segment = offset_transcription.segments[0]
    assert first_segment.start == 0.0 + offset_time
    assert first_segment.end == 1.8 + offset_time
    assert first_segment.words[0].start == 0.0 + offset_time
    assert first_segment.words[0].end == 0.5 + offset_time


def test_standardize_segments_by_num_words(dummy_transcription):
    """Test standardize_segments with num_words parameter."""
    # Standardize to 2 words per segment
    result = dummy_transcription.standardize_segments(num_words=2)

    # Should have 7 segments total (13 words / 2 = 6.5, rounded up to 7)
    assert len(result.segments) == 7

    # Check first segment (first 2 words)
    first_segment = result.segments[0]
    assert first_segment.text == "Hello world"
    assert first_segment.start == 0.0
    assert first_segment.end == 1.0
    assert len(first_segment.words) == 2
    assert first_segment.words[0].word == "Hello"
    assert first_segment.words[1].word == "world"

    # Check second segment (next 2 words)
    second_segment = result.segments[1]
    assert second_segment.text == "this is"
    assert second_segment.start == 1.0
    assert second_segment.end == 1.4
    assert len(second_segment.words) == 2

    # Check last segment (remaining 1 word)
    last_segment = result.segments[-1]
    assert last_segment.text == "video"
    assert len(last_segment.words) == 1
    assert last_segment.words[0].word == "video"


def test_standardize_segments_by_time(dummy_transcription):
    """Test standardize_segments with time parameter."""
    # Standardize to 1.5 second segments
    result = dummy_transcription.standardize_segments(time=1.5)

    # Should have multiple segments based on time constraint
    assert len(result.segments) > 1

    # Check that each segment respects the time constraint
    for segment in result.segments:
        duration = segment.end - segment.start
        assert duration <= 1.5

    # Check first segment should contain words that fit in 1.5 seconds
    first_segment = result.segments[0]
    assert first_segment.start == 0.0
    assert first_segment.end <= 1.5

    # Verify words are properly grouped - "is" ends at 1.4 so it fits in 1.5s
    expected_words = ["Hello", "world", "this", "is"]  # These should fit in 1.5s (ends at 1.4)
    actual_words = [w.word for w in first_segment.words]
    assert actual_words == expected_words


def test_standardize_segments_by_time_large_constraint(dummy_transcription):
    """Test standardize_segments with time constraint larger than any segment."""
    # Use a large time constraint that should fit multiple original segments
    result = dummy_transcription.standardize_segments(time=10.0)

    # Should result in fewer segments than original
    assert len(result.segments) <= len(dummy_transcription.segments)

    # All words should be preserved
    all_original_words = []
    for segment in dummy_transcription.segments:
        all_original_words.extend([w.word for w in segment.words])

    all_result_words = []
    for segment in result.segments:
        all_result_words.extend([w.word for w in segment.words])

    assert all_result_words == all_original_words


def test_standardize_segments_invalid_parameters():
    """Test standardize_segments with invalid parameters."""
    transcription = Transcription(segments=[])

    # Test providing both parameters
    with pytest.raises(ValueError, match="Exactly one of 'time' or 'num_words' must be provided"):
        transcription.standardize_segments(time=1.0, num_words=2)

    # Test providing neither parameter
    with pytest.raises(ValueError, match="Exactly one of 'time' or 'num_words' must be provided"):
        transcription.standardize_segments()

    # Test negative time
    with pytest.raises(ValueError, match="Time must be positive"):
        transcription.standardize_segments(time=-1.0)

    # Test zero time
    with pytest.raises(ValueError, match="Time must be positive"):
        transcription.standardize_segments(time=0.0)

    # Test negative num_words
    with pytest.raises(ValueError, match="Number of words must be positive"):
        transcription.standardize_segments(num_words=-1)

    # Test zero num_words
    with pytest.raises(ValueError, match="Number of words must be positive"):
        transcription.standardize_segments(num_words=0)


def test_standardize_segments_empty_transcription():
    """Test standardize_segments with empty transcription."""
    empty_transcription = Transcription(segments=[])

    # Should return empty transcription for both time and num_words
    result_time = empty_transcription.standardize_segments(time=1.0)
    assert len(result_time.segments) == 0

    result_words = empty_transcription.standardize_segments(num_words=2)
    assert len(result_words.segments) == 0


def test_standardize_segments_single_word():
    """Test standardize_segments with single word transcription."""
    single_word = TranscriptionWord(start=0.0, end=1.0, word="test")
    single_segment = TranscriptionSegment(start=0.0, end=1.0, text="test", words=[single_word])
    transcription = Transcription(segments=[single_segment])

    # Test with num_words=1
    result = transcription.standardize_segments(num_words=1)
    assert len(result.segments) == 1
    assert result.segments[0].text == "test"
    assert len(result.segments[0].words) == 1

    # Test with num_words=2 (should still have 1 segment with 1 word)
    result = transcription.standardize_segments(num_words=2)
    assert len(result.segments) == 1
    assert result.segments[0].text == "test"
    assert len(result.segments[0].words) == 1


def test_standardize_segments_preserves_word_timing(dummy_transcription):
    """Test that standardize_segments preserves individual word timings."""
    result = dummy_transcription.standardize_segments(num_words=3)

    # Check that word timings are preserved exactly
    all_original_words = []
    for segment in dummy_transcription.segments:
        all_original_words.extend(segment.words)

    all_result_words = []
    for segment in result.segments:
        all_result_words.extend(segment.words)

    # Compare each word's timing
    for orig_word, result_word in zip(all_original_words, all_result_words):
        assert orig_word.start == result_word.start
        assert orig_word.end == result_word.end
        assert orig_word.word == result_word.word
