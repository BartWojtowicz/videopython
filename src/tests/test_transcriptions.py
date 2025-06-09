import numpy as np
import pytest

from videopython.base.transcription import Transcription, TranscriptionOverlay, TranscriptionSegment, TranscriptionWord
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
    overlay = TranscriptionOverlay(transcription=dummy_transcription, font_filename=TEST_FONT_PATH)

    assert overlay.transcription == dummy_transcription
    assert overlay.font_filename == TEST_FONT_PATH
    assert overlay.font_size == 24
    assert overlay.text_color == (255, 255, 255)


def test_transcription_overlay_apply_basic(dummy_video, dummy_transcription):
    """Test basic application of TranscriptionOverlay to a video."""
    overlay = TranscriptionOverlay(transcription=dummy_transcription, font_filename=TEST_FONT_PATH, font_size=20)

    result_video = overlay.apply(dummy_video)
    result_video.save()

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
        transcription=dummy_transcription,
        font_filename=TEST_FONT_PATH,
        font_size=30,
        text_color=(255, 0, 0),  # Red text
        background_color=(0, 0, 0, 200),  # Black background with transparency
        highlight_color=(0, 255, 0),  # Green highlight
        position=(0.5, 0.8),
        box_width=0.6,
    )

    result_video = overlay.apply(dummy_video)

    # Check that the overlay was applied (video should still have same basic properties)
    assert isinstance(result_video, Video)
    assert result_video.frame_shape == dummy_video.frame_shape
    assert len(result_video.frames) == len(dummy_video.frames)


def test_transcription_overlay_no_background(dummy_video, dummy_transcription):
    """Test TranscriptionOverlay with no background color."""
    overlay = TranscriptionOverlay(
        transcription=dummy_transcription, font_filename=TEST_FONT_PATH, background_color=None
    )

    result_video = overlay.apply(dummy_video)

    assert isinstance(result_video, Video)
    assert len(result_video.frames) == len(dummy_video.frames)


def test_transcription_overlay_frame_content_changes(dummy_video, dummy_transcription):
    """Test that TranscriptionOverlay actually modifies frame content during active segments."""
    overlay = TranscriptionOverlay(transcription=dummy_transcription, font_filename=TEST_FONT_PATH)

    result_video = overlay.apply(dummy_video)

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
