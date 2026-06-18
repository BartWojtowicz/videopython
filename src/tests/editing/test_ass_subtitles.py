"""Tests for the libass subtitle renderer (ASS compilation + filter path)."""

from __future__ import annotations

import glob
import tempfile
from typing import Any

import numpy as np
import pytest
from PIL import ImageFont

from tests.test_config import SMALL_VIDEO_PATH
from videopython.base._ffmpeg import escape_filter_value
from videopython.base.exceptions import PlanValidationError
from videopython.base.fonts import BUNDLED_FONT_FAMILIES, BUNDLED_FONTS, bundled_fonts_dir
from videopython.base.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video
from videopython.editing import StreamingClass, VideoEdit
from videopython.editing._ass import (
    AnchorPoint,
    AssLook,
    _ass_color,
    _ass_color_tag,
    _ass_time,
    _word_state_intervals,
    build_ass,
)
from videopython.editing.operation import FilterCtx
from videopython.editing.transcription_overlay import TranscriptionOverlay


def _words(spec: list[tuple[str, float, float]]) -> list[TranscriptionWord]:
    return [TranscriptionWord(word=w, start=s, end=e) for w, s, e in spec]


def _transcription() -> Transcription:
    """Three words at 4.5-7.5s on the source timeline (segment cut is [4, 8])."""
    return Transcription(
        segments=[
            TranscriptionSegment(
                text="hello streaming world",
                start=4.4,
                end=7.6,
                words=_words([("hello", 4.5, 5.2), ("streaming", 5.5, 6.4), ("world", 6.8, 7.5)]),
            )
        ],
        language="en",
    )


def _local_transcription() -> Transcription:
    """Segment-local twin of :func:`_transcription` (already re-based)."""
    sliced = _transcription().slice(4.0, 8.0)
    assert sliced is not None
    return sliced.offset(-4.0)


def _look(**overrides: Any) -> AssLook:
    defaults: dict[str, Any] = dict(
        font_family="DejaVu Sans",
        bold=False,
        font_px=40,
        text_color=(255, 235, 59),
        highlight_color=(76, 175, 80),
        outline_px=2,
        background=(0, 0, 0, 100),
        background_padding=15,
        highlight_size_multiplier=1.2,
        position=(0.5, 0.82),
        anchor=AnchorPoint.CENTER,
        box_width=0.6,
    )
    defaults.update(overrides)
    return AssLook(**defaults)


def _plan(operations: list[dict[str, Any]]) -> VideoEdit:
    return VideoEdit.model_validate(
        {"segments": [{"source": SMALL_VIDEO_PATH, "start": 4.0, "end": 8.0, "operations": operations}]}
    )


SUBTITLES = {"op": "add_subtitles", "font_scale": 0.1}


class TestAssPrimitives:
    def test_time_format_is_centisecond_single_digit_hour(self):
        assert _ass_time(0.0) == "0:00:00.00"
        assert _ass_time(3661.456) == "1:01:01.46"
        assert _ass_time(59.999) == "0:01:00.00"
        assert _ass_time(-1.0) == "0:00:00.00"

    def test_color_is_aabbggrr_with_inverted_alpha(self):
        assert _ass_color((255, 235, 59)) == "&H003BEBFF"
        assert _ass_color((0, 0, 0), opacity=100) == "&H9B000000"

    def test_color_tag_is_bbggrr(self):
        assert _ass_color_tag((76, 175, 80)) == "&H50AF4C&"

    def test_escape_filter_value(self):
        assert escape_filter_value("/tmp/a.ass") == "'/tmp/a.ass'"
        assert escape_filter_value("/tmp/a:b'c.ass") == r"'/tmp/a\:b\'c.ass'"
        assert escape_filter_value("C:\\x") == r"'C\:\\x'"

    def test_word_state_intervals_tile_exactly(self):
        intervals = _word_state_intervals(0.4, 3.5, [(0.5, 1.2), (1.5, 2.4), (2.8, 3.5)])

        assert intervals[0] == (40, 50, None)
        assert intervals[-1][1] == 350
        for (_, prev_end, _), (next_start, _, _) in zip(intervals, intervals[1:]):
            assert prev_end == next_start
        assert [hl for _, _, hl in intervals] == [None, 0, None, 1, None, 2]

    def test_word_state_intervals_drop_zero_length_and_clamp_overlap(self):
        intervals = _word_state_intervals(0.0, 1.0, [(0.0, 0.6), (0.5, 1.0)])

        assert intervals == [(0, 60, 0), (60, 100, 1)]


class TestBuildAss:
    def test_document_structure_and_style(self):
        doc = build_ass(_local_transcription(), width=800, height=500, look=_look())

        assert "PlayResX: 800" in doc and "PlayResY: 500" in doc
        assert "ScaledBorderAndShadow: yes" in doc
        style = next(line for line in doc.splitlines() if line.startswith("Style: Default,"))
        # BOXED look: yellow text, BorderStyle 4 box (black, alpha 155), padding 15 via Shadow.
        assert ",&H003BEBFF," in style and ",&H9B000000," in style
        assert ",4,2,15,5," in style  # BorderStyle, Outline, Shadow, Alignment
        assert style.startswith("Style: Default,DejaVu Sans,40,")

    def test_events_tile_cue_with_highlight_overrides(self):
        doc = build_ass(_local_transcription(), width=800, height=500, look=_look())
        events = [line for line in doc.splitlines() if line.startswith("Dialogue:")]

        assert len(events) == 5  # 3 highlighted words + 2 inter-word gaps
        assert all("\\an5\\pos(400.0,410.0)" in e for e in events)
        highlighted = [e for e in events if "\\1c&H50AF4C&" in e]
        assert len(highlighted) == 3
        assert all("\\fscx120\\fscy120" in e and "{\\r}" in e for e in highlighted)

    def test_no_background_uses_border_style_one(self):
        doc = build_ass(_local_transcription(), width=800, height=500, look=_look(background=None))
        style = next(line for line in doc.splitlines() if line.startswith("Style: Default,"))

        assert ",1,2,0,5," in style  # BorderStyle 1, Shadow 0

    def test_window_clips_events(self):
        doc = build_ass(_local_transcription(), width=800, height=500, look=_look(), window=(0.0, 1.0))
        events = [line for line in doc.splitlines() if line.startswith("Dialogue:")]

        # Cue runs 0.5-3.5 locally; clipped to [0, 1.0) only "hello" (0.5-1.0) survives.
        assert len(events) == 1
        assert "0:00:00.50,0:00:01.00" in events[0]

    def test_ass_markup_in_words_is_neutralized(self):
        tr = Transcription(
            segments=[TranscriptionSegment(text="a", start=0.0, end=1.0, words=_words([("{\\pos(0,0)}x", 0.0, 1.0)]))],
            language="en",
        )
        doc = build_ass(tr, width=100, height=100, look=_look())

        assert "{\\pos(0,0)}x" not in doc.split("[Events]")[1].replace("{\\an5", "")
        assert "(/pos(0,0))x" in doc

    def test_bold_flag(self):
        doc = build_ass(_local_transcription(), width=800, height=500, look=_look(bold=True))
        style = next(line for line in doc.splitlines() if line.startswith("Style: Default,"))
        assert ",-1,0,0,0,100,100," in style


class TestFontFamilies:
    def test_bundled_families_match_font_name_tables(self):
        """BUNDLED_FONT_FAMILIES pins the family names libass matches on."""
        fonts_dir = bundled_fonts_dir()
        for name, (family, bold) in BUNDLED_FONT_FAMILIES.items():
            filename = BUNDLED_FONTS[name] if name is not None else "DejaVuSans.ttf"
            actual_family, actual_face = ImageFont.truetype(str(fonts_dir / filename), 16).getname()
            assert actual_family == family, f"{filename}: {actual_family!r} != {family!r}"
            assert ("bold" in (actual_face or "").lower()) == bold, filename

    def test_fonts_dir_exists_and_holds_ttfs(self):
        d = bundled_fonts_dir()
        assert d.is_dir()
        assert (d / "DejaVuSans.ttf").exists()


class TestLibassCompilation:
    def test_libass_is_the_only_renderer(self):
        """Pinned post-sign-off (June 2026): the python renderer is gone."""
        assert "renderer" not in TranscriptionOverlay.model_fields
        assert TranscriptionOverlay().compiles_to_filter

    def test_libass_compiles_and_registers_temp_file(self):
        op = TranscriptionOverlay()
        ctx = FilterCtx(width=800, height=500, fps=24, context={"transcription": _local_transcription()})
        expr = op.to_ffmpeg_filter(ctx)

        assert op.compiles_to_filter
        assert expr is not None and expr.startswith("subtitles=filename='")
        assert "fontsdir=" in expr
        (ass_path,) = ctx.owned_files
        try:
            assert ass_path.exists()
            assert "Dialogue:" in ass_path.read_text(encoding="utf-8")
        finally:
            ass_path.unlink(missing_ok=True)

    def test_missing_context_raises_at_compile(self):
        op = TranscriptionOverlay()
        ctx = FilterCtx(width=800, height=500, fps=24)
        with pytest.raises(ValueError, match="requires transcription data"):
            op.to_ffmpeg_filter(ctx)
        assert ctx.owned_files == []


class TestLibassStreamability:
    def test_libass_classifies_as_filter(self):
        report = _plan([SUBTITLES]).streamability()
        (entry,) = report.entries
        assert entry.streaming_class is StreamingClass.FILTER

    def test_transform_after_libass_subtitles_streams(self):
        """The ordering win: subtitles joins the vf chain, so a following transform does too."""
        report = _plan([SUBTITLES, {"op": "crop", "width": 400, "height": 300}]).streamability()
        assert [e.streaming_class for e in report.entries] == [StreamingClass.FILTER, StreamingClass.FILTER]
        assert report.streamable

    def test_subtitles_after_frame_effect_classify_as_encode_stage_filter(self):
        """Subtitles following frame effects land in the encode-stage chain -- still a filter."""
        report = _plan([{"op": "fade", "mode": "in", "duration": 0.5}, SUBTITLES]).streamability()
        assert [e.streaming_class for e in report.entries] == [
            StreamingClass.FRAME_EFFECT,
            StreamingClass.FILTER,
        ]
        assert report.streamable

    def test_frame_effect_after_encode_stage_subtitles_is_unstreamable(self):
        report = _plan(
            [
                {"op": "fade", "mode": "in", "duration": 0.5},
                SUBTITLES,
                {"op": "glitch"},
            ]
        ).streamability()
        trailing = report.entries[2]
        assert trailing.streaming_class is StreamingClass.UNSTREAMABLE
        assert trailing.reason is not None and "encode-stage" in trailing.reason
        assert not report.streamable


class TestLibassExecution:
    def test_streams_and_draws_subtitles(self, tmp_path):
        before = set(glob.glob(tempfile.gettempdir() + "/*.ass"))
        out = _plan([SUBTITLES]).run_to_file(tmp_path / "subs.mp4", context={"transcription": _transcription()})
        base = _plan([]).run_to_file(tmp_path / "base.mp4")
        assert set(glob.glob(tempfile.gettempdir() + "/*.ass")) == before

        subs = Video.from_path(str(out)).frames
        plain = Video.from_path(str(base)).frames
        assert len(subs) == len(plain)
        fps = len(plain) / 4.0
        active = round(1.7 * fps)  # mid "streaming" word
        quiet = round(0.1 * fps)  # before the first word
        active_diff = np.abs(subs[active].astype(int) - plain[active].astype(int))
        assert (active_diff > 50).mean() > 0.005, "no subtitle pixels drawn"
        assert np.abs(subs[quiet].astype(int) - plain[quiet].astype(int)).mean() < 2.0

    def test_transform_after_libass_subtitles_streams_end_to_end(self, tmp_path):
        plan = _plan([SUBTITLES, {"op": "crop", "width": 400, "height": 300}])
        out = plan.run_to_file(tmp_path / "out.mp4", context={"transcription": _transcription()})
        meta = Video.from_path(str(out))
        assert meta.frames.shape[2] == 400 and meta.frames.shape[1] == 300

    def test_rejected_plan_leaks_no_ass_file(self, tmp_path):
        """A plan rejected for an unstreamable op must not leak temp files."""
        before = set(glob.glob(tempfile.gettempdir() + "/*.ass"))
        # Unstreamable: a frame effect (glitch) after the encode-stage subtitles
        # filter. The plan still carries add_subtitles, so this guards that a
        # rejected plan does not leak the .ass temp file.
        plan = _plan([{"op": "fade", "mode": "in", "duration": 0.5}, SUBTITLES, {"op": "glitch"}])
        with pytest.raises(PlanValidationError, match="cannot stream"):
            plan.run_to_file(tmp_path / "out.mp4", context={"transcription": _transcription()})
        assert set(glob.glob(tempfile.gettempdir() + "/*.ass")) == before

    def test_subtitles_after_frame_effect_stream_via_encode_stage(self, tmp_path):
        """[fade, add_subtitles] streams: the filter burns at encode time."""
        plan = _plan([{"op": "fade", "mode": "in", "duration": 1.0}, SUBTITLES])
        out = plan.run_to_file(tmp_path / "out.mp4", context={"transcription": _transcription()})
        base = _plan([{"op": "fade", "mode": "in", "duration": 1.0}]).run_to_file(tmp_path / "base.mp4")

        subs = Video.from_path(str(out)).frames
        plain = Video.from_path(str(base)).frames
        fps = len(plain) / 4.0
        active = round(1.7 * fps)  # mid "streaming" word, past the fade
        active_diff = np.abs(subs[active].astype(int) - plain[active].astype(int))
        assert (active_diff > 50).mean() > 0.005, "no subtitle pixels drawn on the encode-stage path"
        # The fade still applies: the first frame is (near) black on both.
        assert subs[0].mean() < 5 and plain[0].mean() < 5

    def test_missing_context_raises_before_decode(self, tmp_path):
        with pytest.raises(ValueError, match="requires transcription"):
            _plan([SUBTITLES]).run_to_file(tmp_path / "out.mp4")
