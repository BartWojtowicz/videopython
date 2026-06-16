from typing import Any

import numpy as np
import pytest

from tests.test_config import TEST_FONT_PATH
from videopython.base.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video
from videopython.editing import VideoEdit
from videopython.editing.transcription_overlay import TranscriptionOverlay

# Eager VideoEdit/TranscriptionOverlay.apply() is gone (0.44.0); subtitles only
# exist on the streaming-to-file engine. These overlay tests therefore render a
# real `add_subtitles` plan via `VideoEdit.run_to_file` and read the mp4 back.
# This file lives in tests/base/, which has no editing `render` fixture, so the
# helpers below build the run_to_file + Video.from_path round-trip inline. The
# x264/AAC round-trip is lossy, so anything checked on decoded frames uses
# mean-abs-diff tolerances, never exact equality.

# Subtitle pixels drawn against the black source clear this fraction of pixels
# changed by >50; codec bleed on a truly inactive frame stays well below it.
_ACTIVE_PIXEL_FRACTION = 0.005


@pytest.fixture(scope="session")
def black_source(tmp_path_factory):
    """Path to an 8s black source video on disk (read-only, shared).

    `SegmentConfig.source` is a file path, so the in-memory black clip the old
    eager tests used must be materialized once to disk for plans to cut from.
    A pure-black source keeps subtitle pixels a clean, isolated signal against
    near-zero re-encode noise.
    """
    src = tmp_path_factory.mktemp("black_src") / "black.mp4"
    Video.from_image(np.zeros((360, 640, 3), dtype=np.uint8), fps=30, length_seconds=8.0).save(src)
    return str(src)


def _subtitles_plan(source: str, operation: dict[str, Any], *, end: float = 6.0) -> VideoEdit:
    """A single-segment plan cutting [0, end) from `source` with one add_subtitles op."""
    return VideoEdit.model_validate(
        {"segments": [{"source": source, "start": 0.0, "end": end, "operations": [operation]}]}
    )


def _plain_plan(source: str, *, end: float = 6.0) -> VideoEdit:
    """The same cut with no operations -- the A/B reference for frame diffs."""
    return VideoEdit.model_validate({"segments": [{"source": source, "start": 0.0, "end": end}]})


def _render(plan: VideoEdit, tmp_path, name: str, *, transcription: Transcription | None = None) -> Video:
    """Run a plan to file and load it back as a Video (the only execution engine)."""
    context = {"transcription": transcription} if transcription is not None else None
    out = plan.run_to_file(tmp_path / name, context=context)
    return Video.from_path(str(out))


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


@pytest.fixture(scope="session")
def early_cue_transcription():
    """One cue at 0.0-1.8s, then a long silent tail.

    The frame-content tests need an inactive sampling point (~4s) that is far
    enough from every cue that x264 inter-frame bleed from a nearby subtitle
    does not leak into it. A single early cue gives an active frame at ~1s and a
    provably blank frame at ~4s.
    """
    words = [
        TranscriptionWord(start=0.0, end=0.5, word="Hello"),
        TranscriptionWord(start=0.5, end=1.0, word="world"),
        TranscriptionWord(start=1.0, end=1.2, word="this"),
        TranscriptionWord(start=1.2, end=1.4, word="is"),
        TranscriptionWord(start=1.4, end=1.8, word="test"),
    ]
    return Transcription(
        segments=[TranscriptionSegment(start=0.0, end=1.8, text="Hello world this is test", words=words)]
    )


def test_transcription_overlay_initialization(dummy_transcription):
    """Test that TranscriptionOverlay can be initialized with default parameters."""
    overlay = TranscriptionOverlay(font_filename=TEST_FONT_PATH)

    assert overlay.font_filename == TEST_FONT_PATH


def test_transcription_overlay_render_preserves_shape_and_audio(black_source, dummy_transcription, tmp_path):
    """A subtitled render keeps dimensions, fps, frame count, and the audio track.

    add_subtitles is a video-only libass filter, so the streamed output must
    match a plain render of the same cut everywhere except the burned pixels.
    """
    op = {"op": "add_subtitles", "font_scale": 0.1, "font_filename": TEST_FONT_PATH}
    subs = _render(_subtitles_plan(black_source, op), tmp_path, "subs.mp4", transcription=dummy_transcription)
    plain = _render(_plain_plan(black_source), tmp_path, "plain.mp4")

    subs_meta, plain_meta = subs.metadata, plain.metadata
    assert (subs_meta.width, subs_meta.height) == (plain_meta.width, plain_meta.height)
    assert subs_meta.fps == plain_meta.fps
    assert subs_meta.frame_count == plain_meta.frame_count

    # Audio passes through untouched (video-only filter).
    assert subs.audio.metadata.duration_seconds == plain.audio.metadata.duration_seconds
    assert subs.audio.metadata.sample_rate == plain.audio.metadata.sample_rate


def test_transcription_overlay_with_custom_settings(black_source, dummy_transcription, tmp_path):
    """Custom styling fields still render end-to-end and preserve dimensions."""
    op = {
        "op": "add_subtitles",
        "font_filename": TEST_FONT_PATH,
        "font_size": 30,
        "text_color": [255, 0, 0],  # Red text
        "background_color": [0, 0, 0, 200],  # Black background with transparency
        "highlight_color": [0, 255, 0],  # Green highlight
        "position": [0.5, 0.8],
        "box_width": 0.6,
    }
    subs = _render(_subtitles_plan(black_source, op), tmp_path, "subs.mp4", transcription=dummy_transcription)
    plain = _render(_plain_plan(black_source), tmp_path, "plain.mp4")

    assert subs.frame_shape == plain.frame_shape
    assert len(subs.frames) == len(plain.frames)


def test_transcription_overlay_no_background(black_source, dummy_transcription, tmp_path):
    """A subtitle render with the background disabled still produces a full clip."""
    op = {"op": "add_subtitles", "font_filename": TEST_FONT_PATH, "background_color": None}
    subs = _render(_subtitles_plan(black_source, op), tmp_path, "subs.mp4", transcription=dummy_transcription)
    plain = _render(_plain_plan(black_source), tmp_path, "plain.mp4")

    assert len(subs.frames) == len(plain.frames)


def test_transcription_overlay_frame_content_changes(black_source, early_cue_transcription, tmp_path):
    """Active-segment frames gain subtitle pixels; an inactive frame does not.

    The eager path could assert exact equality on in-memory black frames; the
    streaming engine round-trips through x264, so this compares the subtitled
    render against a plain render of the same cut with mean-abs-diff tolerances.
    """
    op = {"op": "add_subtitles", "font_scale": 0.1, "font_filename": TEST_FONT_PATH}
    subs = _render(_subtitles_plan(black_source, op), tmp_path, "subs.mp4", transcription=early_cue_transcription)
    plain = _render(_plain_plan(black_source), tmp_path, "plain.mp4")

    fps = subs.fps
    active = subs.frames[round(1.0 * fps)].astype(int) - plain.frames[round(1.0 * fps)].astype(int)
    inactive = subs.frames[round(4.0 * fps)].astype(int) - plain.frames[round(4.0 * fps)].astype(int)

    # Active frame (cue 0.0-1.8s): a meaningful fraction of pixels lit up.
    assert (np.abs(active) > 50).mean() > _ACTIVE_PIXEL_FRACTION, "no subtitle pixels drawn on the active frame"
    # Inactive frame (~4s, far from any cue): unchanged within codec noise.
    assert np.abs(inactive).mean() < 1.0


def test_transcription_overlay_defaults_font_when_none(black_source, early_cue_transcription, tmp_path):
    """font_filename is optional: None renders with the bundled default font."""
    overlay = TranscriptionOverlay(font_size=20)
    assert overlay.font_filename is None  # construction default preserved

    # No font_filename in the op -> the bundled default font is used. A larger
    # font_scale keeps the rendered pixels a robust signal through x264.
    op = {"op": "add_subtitles", "font_scale": 0.14}
    subs = _render(_subtitles_plan(black_source, op), tmp_path, "subs.mp4", transcription=early_cue_transcription)
    plain = _render(_plain_plan(black_source), tmp_path, "plain.mp4")

    fps = subs.fps
    active = subs.frames[round(1.0 * fps)].astype(int) - plain.frames[round(1.0 * fps)].astype(int)
    inactive = subs.frames[round(4.0 * fps)].astype(int) - plain.frames[round(4.0 * fps)].astype(int)

    # Active segment (0.0-1.8): frame at 1s must be modified despite no font path.
    assert (np.abs(active) > 50).mean() > _ACTIVE_PIXEL_FRACTION, "default font drew no subtitle pixels"
    # Inactive region (~4s): unchanged within codec noise.
    assert np.abs(inactive).mean() < 1.0


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


def test_standardize_segments_preserves_speaker_by_time():
    """Test that standardize_segments(time=...) preserves speaker info and splits on speaker change."""
    words = [
        TranscriptionWord(start=0.0, end=0.5, word="Hello", speaker="SPEAKER_00"),
        TranscriptionWord(start=0.5, end=1.0, word="world", speaker="SPEAKER_00"),
        TranscriptionWord(start=1.0, end=1.5, word="Hi", speaker="SPEAKER_01"),
        TranscriptionWord(start=1.5, end=2.0, word="there", speaker="SPEAKER_01"),
        TranscriptionWord(start=2.0, end=2.5, word="Back", speaker="SPEAKER_00"),
    ]
    transcription = Transcription(words=words)

    # Large time window -- should still split on speaker changes
    result = transcription.standardize_segments(time=10.0)

    assert len(result.segments) == 3
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[0].text == "Hello world"
    assert result.segments[1].speaker == "SPEAKER_01"
    assert result.segments[1].text == "Hi there"
    assert result.segments[2].speaker == "SPEAKER_00"
    assert result.segments[2].text == "Back"


def test_standardize_segments_preserves_speaker_by_num_words():
    """Test that standardize_segments(num_words=...) preserves speaker info and splits on speaker change."""
    words = [
        TranscriptionWord(start=0.0, end=0.5, word="Hello", speaker="SPEAKER_00"),
        TranscriptionWord(start=0.5, end=1.0, word="world", speaker="SPEAKER_00"),
        TranscriptionWord(start=1.0, end=1.5, word="Hi", speaker="SPEAKER_01"),
        TranscriptionWord(start=1.5, end=2.0, word="there", speaker="SPEAKER_01"),
    ]
    transcription = Transcription(words=words)

    # num_words=10 is larger than total words, but speaker change should still split
    result = transcription.standardize_segments(num_words=10)

    assert len(result.segments) == 2
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[0].text == "Hello world"
    assert result.segments[1].speaker == "SPEAKER_01"
    assert result.segments[1].text == "Hi there"


def test_standardize_segments_speaker_change_forces_break():
    """Test that speaker change forces a segment break even within time/word limits."""
    words = [
        TranscriptionWord(start=0.0, end=0.3, word="One", speaker="SPEAKER_00"),
        TranscriptionWord(start=0.3, end=0.5, word="two", speaker="SPEAKER_01"),
        TranscriptionWord(start=0.5, end=0.8, word="three", speaker="SPEAKER_00"),
    ]
    transcription = Transcription(words=words)

    # Time: all within 1 second, but 3 speakers changes -> 3 segments
    result_time = transcription.standardize_segments(time=5.0)
    assert len(result_time.segments) == 3
    assert [s.speaker for s in result_time.segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]

    # Words: num_words=10 but still 3 segments due to speaker changes
    result_words = transcription.standardize_segments(num_words=10)
    assert len(result_words.segments) == 3
    assert [s.speaker for s in result_words.segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_standardize_segments_no_speaker():
    """Test that standardize_segments works correctly with None speakers (no diarization)."""
    words = [
        TranscriptionWord(start=0.0, end=0.5, word="Hello"),
        TranscriptionWord(start=0.5, end=1.0, word="world"),
        TranscriptionWord(start=1.0, end=1.5, word="foo"),
        TranscriptionWord(start=1.5, end=2.0, word="bar"),
    ]
    segment = TranscriptionSegment(start=0.0, end=2.0, text="Hello world foo bar", words=words)
    transcription = Transcription(segments=[segment])

    # All None speakers -- should group normally by time
    result = transcription.standardize_segments(time=1.5)
    assert len(result.segments) == 2
    assert result.segments[0].text == "Hello world foo"
    assert result.segments[0].speaker is None
    assert result.segments[1].text == "bar"
    assert result.segments[1].speaker is None

    # Group normally by word count
    result = transcription.standardize_segments(num_words=2)
    assert len(result.segments) == 2
    assert result.segments[0].text == "Hello world"
    assert result.segments[1].text == "foo bar"


def _lc_words(*pairs: tuple[str, float, float]) -> list[TranscriptionWord]:
    return [TranscriptionWord(start=s, end=e, word=w) for w, s, e in pairs]


def test_capitalize_sentences_basic():
    """First word is capitalized; mid-sentence words are left alone."""
    seg = TranscriptionSegment(
        start=0.0,
        end=1.5,
        text="hello world there",
        words=_lc_words(("hello", 0.0, 0.5), ("world", 0.5, 1.0), ("there", 1.0, 1.5)),
    )
    result = Transcription(segments=[seg]).capitalize_sentences()

    assert [w.word for w in result.segments[0].words] == ["Hello", "world", "there"]
    assert result.segments[0].text == "Hello world there"


def test_capitalize_sentences_after_terminators():
    """Words following '.', '!' or '?' start new sentences and are capitalized."""
    seg = TranscriptionSegment(
        start=0.0,
        end=3.0,
        text="hi there. how are you? great! thanks",
        words=_lc_words(
            ("hi", 0.0, 0.3),
            ("there.", 0.3, 0.6),
            ("how", 0.6, 0.9),
            ("are", 0.9, 1.2),
            ("you?", 1.2, 1.5),
            ("great!", 1.5, 1.8),
            ("thanks", 1.8, 2.1),
        ),
    )
    result = Transcription(segments=[seg]).capitalize_sentences()

    # "you?" stays lowercase: it is mid-sentence ("How are you?"), only the
    # word after a terminator starts a new sentence.
    assert [w.word for w in result.segments[0].words] == [
        "Hi",
        "there.",
        "How",
        "are",
        "you?",
        "Great!",
        "Thanks",
    ]


def test_capitalize_sentences_spans_segments():
    """Sentence state carries across segment boundaries (no false capitalization)."""
    seg1 = TranscriptionSegment(
        start=0.0, end=1.0, text="this is", words=_lc_words(("this", 0.0, 0.5), ("is", 0.5, 1.0))
    )
    seg2 = TranscriptionSegment(
        start=1.0, end=2.0, text="one sentence.", words=_lc_words(("one", 1.0, 1.5), ("sentence.", 1.5, 2.0))
    )
    seg3 = TranscriptionSegment(
        start=2.0, end=3.0, text="next one", words=_lc_words(("next", 2.0, 2.5), ("one", 2.5, 3.0))
    )
    result = Transcription(segments=[seg1, seg2, seg3]).capitalize_sentences()

    assert [w.word for w in result.segments[0].words] == ["This", "is"]
    assert [w.word for w in result.segments[1].words] == ["one", "sentence."]
    assert [w.word for w in result.segments[2].words] == ["Next", "one"]


def test_capitalize_sentences_preserves_existing_caps():
    """Acronyms and proper nouns are not lower-cased."""
    seg = TranscriptionSegment(
        start=0.0,
        end=1.5,
        text="the NASA team launched IT",
        words=_lc_words(
            ("the", 0.0, 0.3),
            ("NASA", 0.3, 0.6),
            ("team", 0.6, 0.9),
            ("launched", 0.9, 1.2),
            ("IT", 1.2, 1.5),
        ),
    )
    result = Transcription(segments=[seg]).capitalize_sentences()

    assert [w.word for w in result.segments[0].words] == ["The", "NASA", "team", "launched", "IT"]


def test_capitalize_sentences_leading_non_alpha_token():
    """A non-alphabetic token at a sentence start does not consume the capitalization."""
    seg = TranscriptionSegment(
        start=0.0,
        end=1.5,
        text='"..." really? — yes',
        words=_lc_words(('"..."', 0.0, 0.3), ("really?", 0.3, 0.6), ("—", 0.6, 0.9), ("yes", 0.9, 1.2)),
    )
    result = Transcription(segments=[seg]).capitalize_sentences()

    # First alphabetic token gets capitalized; "—" after "?" does not absorb it.
    assert [w.word for w in result.segments[0].words] == ['"..."', "Really?", "—", "Yes"]


def test_capitalize_sentences_trailing_wrappers():
    """Closing quotes/brackets after a terminator still end the sentence."""
    seg = TranscriptionSegment(
        start=0.0,
        end=1.5,
        text='he said "go!" now',
        words=_lc_words(("he", 0.0, 0.3), ("said", 0.3, 0.6), ('"go!"', 0.6, 0.9), ("now", 0.9, 1.2)),
    )
    result = Transcription(segments=[seg]).capitalize_sentences()

    assert [w.word for w in result.segments[0].words] == ["He", "said", '"go!"', "Now"]


def test_capitalize_sentences_preserves_timing_speaker_language():
    """Timing, speaker, and language are carried through unchanged."""
    seg = TranscriptionSegment(
        start=0.0,
        end=1.0,
        text="hello world",
        words=[
            TranscriptionWord(start=0.0, end=0.5, word="hello", speaker="SPEAKER_00"),
            TranscriptionWord(start=0.5, end=1.0, word="world", speaker="SPEAKER_00"),
        ],
        speaker="SPEAKER_00",
    )
    result = Transcription(segments=[seg], language="en").capitalize_sentences()

    out = result.segments[0]
    assert result.language == "en"
    assert out.start == 0.0 and out.end == 1.0
    assert out.speaker == "SPEAKER_00"
    assert [(w.start, w.end, w.speaker) for w in out.words] == [
        (0.0, 0.5, "SPEAKER_00"),
        (0.5, 1.0, "SPEAKER_00"),
    ]


def test_capitalize_sentences_does_not_mutate_original():
    """The source Transcription is left untouched (returns a new object)."""
    seg = TranscriptionSegment(
        start=0.0, end=1.0, text="hello world", words=_lc_words(("hello", 0.0, 0.5), ("world", 0.5, 1.0))
    )
    original = Transcription(segments=[seg])
    original.capitalize_sentences()

    assert original.segments[0].words[0].word == "hello"
    assert original.segments[0].text == "hello world"


def test_capitalize_sentences_empty():
    """Empty transcription stays empty."""
    result = Transcription(segments=[], language="fr").capitalize_sentences()
    assert result.segments == []
    assert result.language == "fr"


def test_chunk_segments_splits_within_segment():
    """A segment is split into <= max_words cues with group-local timings."""
    seg = TranscriptionSegment(
        start=0.0,
        end=3.0,
        text="one two three four five six",
        words=_lc_words(
            ("one", 0.0, 0.5),
            ("two", 0.5, 1.0),
            ("three", 1.0, 1.5),
            ("four", 1.5, 2.0),
            ("five", 2.0, 2.5),
            ("six", 2.5, 3.0),
        ),
    )
    result = Transcription(segments=[seg]).chunk_segments(max_words=4)

    assert len(result.segments) == 2
    assert [s.text for s in result.segments] == ["one two three four", "five six"]
    assert (result.segments[0].start, result.segments[0].end) == (0.0, 2.0)
    assert (result.segments[1].start, result.segments[1].end) == (2.0, 3.0)
    assert all(len(s.words) <= 4 for s in result.segments)


def test_chunk_segments_preserves_silence_gaps():
    """Words are never merged across segments, so pauses are kept intact."""
    seg1 = TranscriptionSegment(
        start=0.0, end=1.0, text="hello world", words=_lc_words(("hello", 0.0, 0.5), ("world", 0.5, 1.0))
    )
    # 4s of silence before the next segment.
    seg2 = TranscriptionSegment(
        start=5.0, end=6.0, text="goodbye now", words=_lc_words(("goodbye", 5.0, 5.5), ("now", 5.5, 6.0))
    )
    result = Transcription(segments=[seg1, seg2]).chunk_segments(max_words=5)

    # Both segments fit under the limit and stay separate -- no cue spans 1.0..5.0.
    assert len(result.segments) == 2
    assert (result.segments[0].start, result.segments[0].end) == (0.0, 1.0)
    assert (result.segments[1].start, result.segments[1].end) == (5.0, 6.0)


def test_chunk_segments_preserves_speaker_confidence_language():
    """Speaker, confidence fields, and language are carried through."""
    seg = TranscriptionSegment(
        start=0.0,
        end=1.0,
        text="a b c",
        words=[
            TranscriptionWord(start=0.0, end=0.3, word="a", speaker="S0"),
            TranscriptionWord(start=0.3, end=0.6, word="b", speaker="S0"),
            TranscriptionWord(start=0.6, end=1.0, word="c", speaker="S0"),
        ],
        speaker="S0",
        avg_logprob=-0.25,
        no_speech_prob=0.01,
        compression_ratio=1.4,
    )
    result = Transcription(segments=[seg], language="pl").chunk_segments(max_words=2)

    assert result.language == "pl"
    assert len(result.segments) == 2
    for out in result.segments:
        assert out.speaker == "S0"
        assert out.avg_logprob == -0.25
        assert out.no_speech_prob == 0.01
        assert out.compression_ratio == 1.4
        assert all(w.speaker == "S0" for w in out.words)


def test_chunk_segments_invalid_max_words():
    """Non-positive max_words is rejected."""
    transcription = Transcription(
        segments=[TranscriptionSegment(start=0.0, end=0.5, text="x", words=_lc_words(("x", 0.0, 0.5)))]
    )
    with pytest.raises(ValueError, match="max_words must be positive"):
        transcription.chunk_segments(max_words=0)
    with pytest.raises(ValueError, match="max_words must be positive"):
        transcription.chunk_segments(max_words=-3)


def test_chunk_segments_empty_and_no_mutation():
    """Empty transcription stays empty; the source is not mutated."""
    assert Transcription(segments=[], language="en").chunk_segments(max_words=3).segments == []

    seg = TranscriptionSegment(
        start=0.0, end=1.0, text="keep me", words=_lc_words(("keep", 0.0, 0.5), ("me", 0.5, 1.0))
    )
    original = Transcription(segments=[seg])
    original.chunk_segments(max_words=1)
    assert len(original.segments) == 1
    assert original.segments[0].text == "keep me"


def test_chunk_segments_empty_words_segment_is_fresh_copy():
    """A words-less segment is passed through as a new object, not aliased."""
    seg = TranscriptionSegment(start=0.0, end=1.0, text="(music)", words=[], speaker="S0")
    src = Transcription(segments=[seg])
    out = src.chunk_segments(max_words=3)

    assert len(out.segments) == 1
    assert out.segments[0] is not seg
    assert out.segments[0].words is not seg.words
    assert out.segments[0].text == "(music)" and out.segments[0].speaker == "S0"


def test_segment_from_words_derives_span_and_carries_metadata():
    """from_words derives start/end/text and passes confidence through."""
    words = _lc_words(("a", 0.5, 1.0), ("b", 1.0, 2.0))
    seg = TranscriptionSegment.from_words(
        words, speaker="S0", avg_logprob=-0.3, no_speech_prob=0.02, compression_ratio=1.5
    )

    assert (seg.start, seg.end, seg.text) == (0.5, 2.0, "a b")
    assert (seg.speaker, seg.avg_logprob, seg.no_speech_prob, seg.compression_ratio) == ("S0", -0.3, 0.02, 1.5)
    # Word list is copied, not aliased.
    assert seg.words == words and seg.words is not words


def test_segment_from_words_defaults_metadata_to_none_and_rejects_empty():
    seg = TranscriptionSegment.from_words(_lc_words(("x", 0.0, 0.1)))
    assert seg.speaker is None
    assert seg.avg_logprob is None and seg.no_speech_prob is None and seg.compression_ratio is None

    with pytest.raises(ValueError, match="from_words requires a non-empty word list"):
        TranscriptionSegment.from_words([])


def test_offset_preserves_confidence_fields():
    """Regression: a pure timing shift must not drop avg_logprob/etc."""
    seg = TranscriptionSegment(
        start=0.0,
        end=1.0,
        text="hi",
        words=[TranscriptionWord(start=0.0, end=1.0, word="hi", speaker="S0")],
        speaker="S0",
        avg_logprob=-0.7,
        no_speech_prob=0.05,
        compression_ratio=1.8,
    )
    out = Transcription(segments=[seg], language="en").offset(2.0).segments[0]

    assert (out.start, out.end) == (2.0, 3.0)
    assert out.words[0].start == 2.0 and out.words[0].end == 3.0
    assert (out.speaker, out.avg_logprob, out.no_speech_prob, out.compression_ratio) == ("S0", -0.7, 0.05, 1.8)


def test_overlay_normalization_defaults():
    """New overlay knobs default to sensible subtitle behavior."""
    overlay = TranscriptionOverlay(font_filename=TEST_FONT_PATH)
    assert overlay.max_words_per_cue == 5
    assert overlay.capitalize is True


def test_overlay_normalizes_long_lowercase_segment():
    """Defaults chunk a long lowercase segment into cues and capitalize sentence starts.

    The end-to-end render is dropped: the underlying transforms
    (``chunk_segments`` / ``capitalize_sentences``) are exercised by their own
    unit tests above, so this asserts the overlay's shared cue transform
    (``_transform``) directly -- the exact data the libass compile path feeds.
    """
    words = _lc_words(
        ("the", 0.0, 0.2),
        ("quick", 0.2, 0.4),
        ("brown", 0.4, 0.6),
        ("fox", 0.6, 0.8),
        ("jumps", 0.8, 1.0),
        ("over", 1.0, 1.2),
        ("the", 1.2, 1.4),
        ("lazy", 1.4, 1.6),
        ("dog", 1.6, 1.8),
    )
    transcription = Transcription(
        segments=[
            TranscriptionSegment(start=0.0, end=1.8, text="the quick brown fox jumps over the lazy dog", words=words)
        ]
    )
    overlay = TranscriptionOverlay(font_filename=TEST_FONT_PATH, font_size=20)  # defaults: 5 words/cue, capitalize

    result = overlay._transform(transcription)

    # Chunked to <= 5 words/cue and the first word capitalized.
    assert [s.text for s in result.segments] == ["The quick brown fox jumps", "over the lazy dog"]
    assert all(len(s.words) <= overlay.max_words_per_cue for s in result.segments)


def test_overlay_normalization_can_be_disabled(dummy_transcription):
    """Disabling both knobs makes the cue transform an identity (source segmentation/casing kept)."""
    overlay = TranscriptionOverlay(font_filename=TEST_FONT_PATH, font_size=20, max_words_per_cue=None, capitalize=False)

    result = overlay._transform(dummy_transcription)

    # No re-chunking and no re-casing: segments pass through verbatim.
    assert [s.text for s in result.segments] == [s.text for s in dummy_transcription.segments]
    assert [[w.word for w in s.words] for s in result.segments] == [
        [w.word for w in s.words] for s in dummy_transcription.segments
    ]


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


def test_language_stored_on_transcription():
    """Test that language field is stored and defaults to None."""
    words = [TranscriptionWord(start=0.0, end=1.0, word="test")]
    segment = TranscriptionSegment(start=0.0, end=1.0, text="test", words=words)

    # Default is None
    t = Transcription(segments=[segment])
    assert t.language is None

    # Explicit language
    t = Transcription(segments=[segment], language="pl")
    assert t.language == "pl"

    # Via words path
    t = Transcription(words=[TranscriptionWord(start=0.0, end=1.0, word="test")], language="en")
    assert t.language == "en"


def test_language_propagated_by_offset():
    """Test that offset() carries language forward."""
    words = [TranscriptionWord(start=0.0, end=1.0, word="test")]
    segment = TranscriptionSegment(start=0.0, end=1.0, text="test", words=words)
    t = Transcription(segments=[segment], language="de")

    assert t.offset(1.0).language == "de"


def test_language_propagated_by_standardize_segments():
    """Test that standardize_segments() carries language forward."""
    words = [TranscriptionWord(start=0.0, end=1.0, word="test")]
    segment = TranscriptionSegment(start=0.0, end=1.0, text="test", words=words)
    t = Transcription(segments=[segment], language="fr")

    assert t.standardize_segments(num_words=1).language == "fr"
    assert t.standardize_segments(time=5.0).language == "fr"


def test_language_propagated_by_slice():
    """Test that slice() carries language forward."""
    words = [TranscriptionWord(start=0.0, end=1.0, word="test")]
    segment = TranscriptionSegment(start=0.0, end=1.0, text="test", words=words)
    t = Transcription(segments=[segment], language="ja")

    sliced = t.slice(0.0, 2.0)
    assert sliced is not None
    assert sliced.language == "ja"


class TestTranscriptionSerialization:
    """Tests for to_dict/from_dict serialization methods."""

    def test_transcription_word_roundtrip(self):
        """Test TranscriptionWord serialization roundtrip."""
        word = TranscriptionWord(start=0.0, end=0.5, word="hello", speaker="SPEAKER_00")
        data = word.to_dict()
        restored = TranscriptionWord.from_dict(data)

        assert restored.start == word.start
        assert restored.end == word.end
        assert restored.word == word.word
        assert restored.speaker == word.speaker

    def test_transcription_word_without_speaker(self):
        """Test TranscriptionWord serialization without speaker."""
        word = TranscriptionWord(start=1.0, end=1.5, word="world")
        data = word.to_dict()
        restored = TranscriptionWord.from_dict(data)

        assert restored.word == word.word
        assert restored.speaker is None

    def test_transcription_segment_roundtrip(self):
        """Test TranscriptionSegment serialization roundtrip."""
        words = [
            TranscriptionWord(start=0.0, end=0.5, word="hello", speaker="SPEAKER_00"),
            TranscriptionWord(start=0.5, end=1.0, word="world", speaker="SPEAKER_00"),
        ]
        segment = TranscriptionSegment(
            start=0.0,
            end=1.0,
            text="hello world",
            words=words,
            speaker="SPEAKER_00",
        )
        data = segment.to_dict()
        restored = TranscriptionSegment.from_dict(data)

        assert restored.start == segment.start
        assert restored.end == segment.end
        assert restored.text == segment.text
        assert restored.speaker == segment.speaker
        assert len(restored.words) == 2
        assert restored.words[0].word == "hello"
        assert restored.words[1].word == "world"

    def test_transcription_roundtrip(self, dummy_transcription):
        """Test Transcription serialization roundtrip."""
        data = dummy_transcription.to_dict()
        restored = Transcription.from_dict(data)

        assert len(restored.segments) == len(dummy_transcription.segments)
        for orig_seg, rest_seg in zip(dummy_transcription.segments, restored.segments):
            assert rest_seg.start == orig_seg.start
            assert rest_seg.end == orig_seg.end
            assert rest_seg.text == orig_seg.text
            assert len(rest_seg.words) == len(orig_seg.words)

    def test_transcription_with_speakers_roundtrip(self):
        """Test Transcription with speaker info serialization roundtrip."""
        words = [
            TranscriptionWord(start=0.0, end=0.5, word="Hello", speaker="SPEAKER_00"),
            TranscriptionWord(start=0.5, end=1.0, word="world", speaker="SPEAKER_00"),
            TranscriptionWord(start=1.0, end=1.5, word="Hi", speaker="SPEAKER_01"),
        ]
        transcription = Transcription(words=words)

        data = transcription.to_dict()
        restored = Transcription.from_dict(data)

        assert len(restored.segments) == len(transcription.segments)
        assert restored.segments[0].speaker == "SPEAKER_00"
        assert restored.segments[1].speaker == "SPEAKER_01"

    def test_empty_transcription_roundtrip(self):
        """Test empty Transcription serialization roundtrip."""
        transcription = Transcription(segments=[])
        data = transcription.to_dict()
        restored = Transcription.from_dict(data)

        assert len(restored.segments) == 0

    def test_language_roundtrip(self):
        """Test that language survives serialization roundtrip."""
        words = [TranscriptionWord(start=0.0, end=1.0, word="test")]
        segment = TranscriptionSegment(start=0.0, end=1.0, text="test", words=words)
        transcription = Transcription(segments=[segment], language="pl")

        data = transcription.to_dict()
        assert data["language"] == "pl"

        restored = Transcription.from_dict(data)
        assert restored.language == "pl"

    def test_language_missing_from_dict(self):
        """Test backwards compatibility: from_dict with no language key."""
        data = {"segments": []}
        restored = Transcription.from_dict(data)
        assert restored.language is None

    def test_segment_confidence_fields_roundtrip(self):
        """avg_logprob/no_speech_prob/compression_ratio survive to_dict/from_dict."""
        words = [TranscriptionWord(start=0.0, end=1.0, word="hi")]
        segment = TranscriptionSegment(
            start=0.0,
            end=1.0,
            text="hi",
            words=words,
            speaker="SPEAKER_00",
            avg_logprob=-0.7,
            no_speech_prob=0.05,
            compression_ratio=1.8,
        )

        data = segment.to_dict()
        assert data["avg_logprob"] == -0.7
        assert data["no_speech_prob"] == 0.05
        assert data["compression_ratio"] == 1.8

        restored = TranscriptionSegment.from_dict(data)
        assert restored.avg_logprob == -0.7
        assert restored.no_speech_prob == 0.05
        assert restored.compression_ratio == 1.8

    def test_segment_confidence_fields_default_none(self):
        """Segments built without confidence fields serialize None and round-trip."""
        words = [TranscriptionWord(start=0.0, end=1.0, word="hi")]
        segment = TranscriptionSegment(start=0.0, end=1.0, text="hi", words=words)

        data = segment.to_dict()
        assert data["avg_logprob"] is None
        assert data["no_speech_prob"] is None
        assert data["compression_ratio"] is None

        restored = TranscriptionSegment.from_dict(data)
        assert restored.avg_logprob is None
        assert restored.no_speech_prob is None
        assert restored.compression_ratio is None

    def test_segment_from_dict_back_compat_without_confidence_keys(self):
        """Old persisted JSON without the new keys must still load (back-compat)."""
        data = {
            "start": 0.0,
            "end": 1.0,
            "text": "hi",
            "words": [{"start": 0.0, "end": 1.0, "word": "hi", "speaker": None}],
            "speaker": None,
        }

        restored = TranscriptionSegment.from_dict(data)
        assert restored.text == "hi"
        assert restored.avg_logprob is None
        assert restored.no_speech_prob is None
        assert restored.compression_ratio is None


class TestTranscriptionSrt:
    """Tests for SRT export and import methods."""

    def test_to_srt_basic(self, dummy_transcription):
        """Test basic SRT export with multiple segments."""
        srt = dummy_transcription.to_srt()

        assert "1\n00:00:00,000 --> 00:00:01,800\nHello world this is test" in srt
        assert "2\n00:00:02,000 --> 00:00:03,500\nSecond segment of transcription" in srt
        assert "3\n00:00:05,000 --> 00:00:06,800\nFinal words in video" in srt

    def test_to_srt_empty(self):
        """Test SRT export with empty transcription."""
        transcription = Transcription(segments=[])
        assert transcription.to_srt() == ""

    def test_to_srt_timestamp_formatting(self):
        """Test that timestamps over an hour are formatted correctly."""
        words = [TranscriptionWord(start=3723.456, end=3725.789, word="late")]
        segment = TranscriptionSegment(start=3723.456, end=3725.789, text="late", words=words)
        transcription = Transcription(segments=[segment])
        srt = transcription.to_srt()

        assert "01:02:03,456 --> 01:02:05,789" in srt

    def test_save_srt(self, dummy_transcription, tmp_path):
        """Test saving SRT to a file."""
        output_path = tmp_path / "output.srt"
        dummy_transcription.save_srt(output_path)

        content = output_path.read_text(encoding="utf-8")
        assert content == dummy_transcription.to_srt()

    def test_from_srt_basic(self):
        """Test parsing a basic SRT string."""
        srt = "1\n00:00:00,000 --> 00:00:01,800\nHello world\n\n2\n00:00:02,000 --> 00:00:03,500\nSecond line\n"
        transcription = Transcription.from_srt(srt)

        assert len(transcription.segments) == 2
        assert transcription.segments[0].text == "Hello world"
        assert transcription.segments[0].start == 0.0
        assert transcription.segments[0].end == 1.8
        assert transcription.segments[1].text == "Second line"
        assert transcription.segments[1].start == 2.0
        assert transcription.segments[1].end == 3.5

    def test_from_srt_empty(self):
        """Test parsing an empty SRT string."""
        transcription = Transcription.from_srt("")
        assert len(transcription.segments) == 0

    def test_from_srt_hour_timestamps(self):
        """Test parsing SRT with timestamps over an hour."""
        srt = "1\n01:02:03,456 --> 01:02:05,789\nlate\n"
        transcription = Transcription.from_srt(srt)

        assert len(transcription.segments) == 1
        assert abs(transcription.segments[0].start - 3723.456) < 0.001
        assert abs(transcription.segments[0].end - 3725.789) < 0.001

    def test_from_srt_multiline_text(self):
        """Test parsing SRT blocks with multi-line text."""
        srt = "1\n00:00:00,000 --> 00:00:02,000\nFirst line\nSecond line\n"
        transcription = Transcription.from_srt(srt)

        assert len(transcription.segments) == 1
        assert transcription.segments[0].text == "First line\nSecond line"

    def test_srt_roundtrip(self, dummy_transcription):
        """Test that to_srt -> from_srt preserves segment text and timing."""
        srt = dummy_transcription.to_srt()
        restored = Transcription.from_srt(srt)

        assert len(restored.segments) == len(dummy_transcription.segments)
        for orig, rest in zip(dummy_transcription.segments, restored.segments):
            assert rest.text == orig.text
            assert abs(rest.start - orig.start) < 0.001
            assert abs(rest.end - orig.end) < 0.001
