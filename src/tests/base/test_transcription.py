import numpy as np
import pytest

from tests.test_config import TEST_FONT_PATH
from videopython.base.text.overlay import TranscriptionOverlay
from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video


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


def test_transcription_initialization_with_words():
    """Test Transcription initialization with words parameter for speaker diarization."""
    words = [
        TranscriptionWord(start=0.0, end=0.5, word="Hello", speaker="SPEAKER_00"),
        TranscriptionWord(start=0.5, end=1.0, word="world", speaker="SPEAKER_00"),
        TranscriptionWord(start=1.0, end=1.5, word="Hi", speaker="SPEAKER_01"),
        TranscriptionWord(start=1.5, end=2.0, word="there", speaker="SPEAKER_01"),
        TranscriptionWord(start=2.0, end=2.5, word="Back", speaker="SPEAKER_00"),
        TranscriptionWord(start=2.5, end=3.0, word="again", speaker="SPEAKER_00"),
    ]

    transcription = Transcription(words=words)

    # Should have 3 segments (SPEAKER_00, SPEAKER_01, SPEAKER_00)
    assert len(transcription.segments) == 3

    # First segment: SPEAKER_00 says "Hello world"
    assert transcription.segments[0].speaker == "SPEAKER_00"
    assert transcription.segments[0].text == "Hello world"
    assert transcription.segments[0].start == 0.0
    assert transcription.segments[0].end == 1.0
    assert len(transcription.segments[0].words) == 2

    # Second segment: SPEAKER_01 says "Hi there"
    assert transcription.segments[1].speaker == "SPEAKER_01"
    assert transcription.segments[1].text == "Hi there"
    assert transcription.segments[1].start == 1.0
    assert transcription.segments[1].end == 2.0
    assert len(transcription.segments[1].words) == 2

    # Third segment: SPEAKER_00 says "Back again"
    assert transcription.segments[2].speaker == "SPEAKER_00"
    assert transcription.segments[2].text == "Back again"
    assert transcription.segments[2].start == 2.0
    assert transcription.segments[2].end == 3.0
    assert len(transcription.segments[2].words) == 2

    # Check speakers set
    assert transcription.speakers == {"SPEAKER_00", "SPEAKER_01"}


def test_transcription_initialization_invalid_parameters():
    """Test that Transcription raises error when given both or neither parameters."""
    words = [TranscriptionWord(start=0.0, end=1.0, word="test", speaker="SPEAKER_00")]
    segments = [TranscriptionSegment(start=0.0, end=1.0, text="test", words=words, speaker="SPEAKER_00")]

    # Test providing both parameters
    with pytest.raises(ValueError, match="Exactly one of 'segments' or 'words' must be provided"):
        Transcription(segments=segments, words=words)

    # Test providing neither parameter
    with pytest.raises(ValueError, match="Exactly one of 'segments' or 'words' must be provided"):
        Transcription()


def test_speaker_stats():
    """Test speaker_stats method calculates speaking time percentages correctly."""
    words = [
        TranscriptionWord(start=0.0, end=1.0, word="Hello", speaker="SPEAKER_00"),  # 1.0s
        TranscriptionWord(start=1.0, end=2.0, word="world", speaker="SPEAKER_00"),  # 1.0s
        TranscriptionWord(start=2.0, end=2.5, word="Hi", speaker="SPEAKER_01"),  # 0.5s
        TranscriptionWord(start=2.5, end=3.0, word="there", speaker="SPEAKER_01"),  # 0.5s
    ]

    transcription = Transcription(words=words)
    stats = transcription.speaker_stats()

    # SPEAKER_00: 2.0s out of 3.0s total = 66.67%
    # SPEAKER_01: 1.0s out of 3.0s total = 33.33%
    assert len(stats) == 2
    assert abs(stats["SPEAKER_00"] - 2.0 / 3.0) < 0.001
    assert abs(stats["SPEAKER_01"] - 1.0 / 3.0) < 0.001


def test_speaker_stats_equal_distribution():
    """Test speaker_stats with equal speaking time."""
    words = [
        TranscriptionWord(start=0.0, end=1.0, word="First", speaker="SPEAKER_00"),
        TranscriptionWord(start=1.0, end=2.0, word="Second", speaker="SPEAKER_01"),
    ]

    transcription = Transcription(words=words)
    stats = transcription.speaker_stats()

    assert stats["SPEAKER_00"] == 0.5
    assert stats["SPEAKER_01"] == 0.5


def test_speaker_stats_empty_transcription():
    """Test speaker_stats with empty transcription."""
    transcription = Transcription(segments=[])
    stats = transcription.speaker_stats()

    assert stats == {}


def test_offset_preserves_speaker_information():
    """Test that offset method preserves speaker information."""
    words = [
        TranscriptionWord(start=0.0, end=0.5, word="Hello", speaker="SPEAKER_00"),
        TranscriptionWord(start=0.5, end=1.0, word="world", speaker="SPEAKER_00"),
        TranscriptionWord(start=1.0, end=1.5, word="Hi", speaker="SPEAKER_01"),
    ]

    transcription = Transcription(words=words)
    offset_time = 2.0
    offset_transcription = transcription.offset(offset_time)

    # Check that speakers are preserved
    assert offset_transcription.speakers == transcription.speakers

    # Check that segment speaker information is preserved
    for orig_seg, offset_seg in zip(transcription.segments, offset_transcription.segments):
        assert offset_seg.speaker == orig_seg.speaker
        assert offset_seg.start == orig_seg.start + offset_time
        assert offset_seg.end == orig_seg.end + offset_time

        # Check that word speaker information is preserved
        for orig_word, offset_word in zip(orig_seg.words, offset_seg.words):
            assert offset_word.speaker == orig_word.speaker
            assert offset_word.start == orig_word.start + offset_time
            assert offset_word.end == orig_word.end + offset_time


def test_transcription_with_none_speakers():
    """Test Transcription with words that have None as speaker (no diarization)."""
    words = [
        TranscriptionWord(start=0.0, end=0.5, word="Hello", speaker=None),
        TranscriptionWord(start=0.5, end=1.0, word="world", speaker=None),
    ]

    transcription = Transcription(words=words)

    # Should have 1 segment since all words have the same speaker (None)
    assert len(transcription.segments) == 1
    assert transcription.segments[0].speaker is None
    assert transcription.segments[0].text == "Hello world"

    # Speakers set should be empty (None is filtered out)
    assert transcription.speakers == set()

    # speaker_stats should return empty dict
    stats = transcription.speaker_stats()
    assert stats == {}


def test_slice_basic(dummy_transcription):
    """Test basic slicing of transcription by time range."""
    # Slice to get only the second segment (2.0-3.5)
    sliced = dummy_transcription.slice(1.9, 4.0)

    assert sliced is not None
    assert len(sliced.segments) == 1
    assert sliced.segments[0].text == "Second segment of transcription"


def test_slice_word_level_granularity():
    """Test that slicing works at word-level granularity."""
    words = [
        TranscriptionWord(start=0.0, end=1.0, word="first"),
        TranscriptionWord(start=1.0, end=2.0, word="second"),
        TranscriptionWord(start=2.0, end=3.0, word="third"),
        TranscriptionWord(start=3.0, end=4.0, word="fourth"),
    ]
    segment = TranscriptionSegment(start=0.0, end=4.0, text="first second third fourth", words=words)
    transcription = Transcription(segments=[segment])

    # Slice to get only "second" and "third" (1.0-3.0)
    sliced = transcription.slice(1.0, 3.0)

    assert sliced is not None
    assert len(sliced.segments) == 1
    assert sliced.segments[0].text == "second third"
    assert len(sliced.segments[0].words) == 2
    assert sliced.segments[0].words[0].word == "second"
    assert sliced.segments[0].words[1].word == "third"


def test_slice_partial_word_overlap():
    """Test that words partially overlapping with time range are included."""
    words = [
        TranscriptionWord(start=0.0, end=2.0, word="first"),  # Partially overlaps with 1.0-3.0
        TranscriptionWord(start=2.0, end=4.0, word="second"),  # Partially overlaps with 1.0-3.0
    ]
    segment = TranscriptionSegment(start=0.0, end=4.0, text="first second", words=words)
    transcription = Transcription(segments=[segment])

    # Both words overlap with 1.0-3.0
    sliced = transcription.slice(1.0, 3.0)

    assert sliced is not None
    assert len(sliced.segments) == 1
    assert sliced.segments[0].text == "first second"
    assert len(sliced.segments[0].words) == 2


def test_slice_no_overlap():
    """Test slicing when no words overlap with time range."""
    words = [
        TranscriptionWord(start=0.0, end=1.0, word="first"),
        TranscriptionWord(start=1.0, end=2.0, word="second"),
    ]
    segment = TranscriptionSegment(start=0.0, end=2.0, text="first second", words=words)
    transcription = Transcription(segments=[segment])

    # No words in 5.0-10.0 range
    sliced = transcription.slice(5.0, 10.0)

    assert sliced is None


def test_slice_invalid_range():
    """Test slicing with invalid time range."""
    words = [TranscriptionWord(start=0.0, end=1.0, word="test")]
    segment = TranscriptionSegment(start=0.0, end=1.0, text="test", words=words)
    transcription = Transcription(segments=[segment])

    # start >= end should return None
    assert transcription.slice(5.0, 5.0) is None
    assert transcription.slice(5.0, 3.0) is None


def test_slice_preserves_speaker_info():
    """Test that slicing preserves speaker information."""
    words = [
        TranscriptionWord(start=0.0, end=1.0, word="hello", speaker="SPEAKER_00"),
        TranscriptionWord(start=1.0, end=2.0, word="world", speaker="SPEAKER_00"),
        TranscriptionWord(start=2.0, end=3.0, word="hi", speaker="SPEAKER_01"),
        TranscriptionWord(start=3.0, end=4.0, word="there", speaker="SPEAKER_01"),
    ]
    transcription = Transcription(words=words)

    # Slice to get words from both speakers
    sliced = transcription.slice(0.5, 3.5)

    assert sliced is not None
    assert len(sliced.segments) == 2  # Should have 2 segments (one per speaker)
    assert sliced.segments[0].speaker == "SPEAKER_00"
    assert sliced.segments[0].text == "hello world"
    assert sliced.segments[1].speaker == "SPEAKER_01"
    assert sliced.segments[1].text == "hi there"


def test_slice_across_multiple_segments(dummy_transcription):
    """Test slicing across multiple original segments."""
    # Slice from 0.5 to 6.0 should include parts of all three segments
    sliced = dummy_transcription.slice(0.5, 6.0)

    assert sliced is not None
    # All words from 0.5 to 6.0 should be included
    all_words = []
    for seg in sliced.segments:
        all_words.extend([w.word for w in seg.words])

    # Should include words from all three original segments that overlap
    assert "world" in all_words  # From first segment
    assert "Second" in all_words  # From second segment
    assert "Final" in all_words  # From third segment
