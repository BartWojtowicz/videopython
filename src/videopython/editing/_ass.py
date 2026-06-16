"""ASS (Advanced SubStation Alpha) generation for libass subtitle burn-in.

Compiles a segment-local, word-level :class:`Transcription` plus the resolved
``add_subtitles`` look into an ``.ass`` document that ffmpeg's ``subtitles=``
filter (libass) renders natively at decode time -- zero per-frame Python.

The word-level highlight (active word in the highlight color, enlarged by the
size multiplier) cannot be expressed with ASS karaoke ``\\k`` tags -- those are
two-state (upcoming vs spoken-and-active) while this look is three-state
(active word only). Instead each cue is emitted as a sequence of word-state
``Dialogue`` events tiling the cue's time range: one event per active-word
interval with inline ``\\1c``/``\\fscx``/``\\fscy`` overrides on that word, and
plain events for the gaps between words. Event boundaries are rounded to ASS's
centisecond precision *before* deriving the intervals, so adjacent events tile
exactly (no double-draw overlap, no blank gaps).

Placement uses ``{\\an..\\pos(x,y)}`` on every event -- the exact box-anchor
semantics of the Python renderer (a box anchored at a normalized position) --
rather than margin arithmetic; ``MarginL/R`` are still set so libass wraps at
the same ``box_width`` the Python renderer uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from videopython.base.transcription import Transcription

__all__ = ["AnchorPoint", "AssLook", "build_ass"]

RGBColor = tuple[int, int, int]
RGBAColor = tuple[int, int, int, int]


class AnchorPoint(str, Enum):
    """Which point of the subtitle box sits at the configured position."""

    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    TOP_RIGHT = "top-right"
    CENTER_LEFT = "center-left"
    CENTER = "center"
    CENTER_RIGHT = "center-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"


# AnchorPoint -> ASS \an numpad code (7-8-9 top, 4-5-6 middle, 1-2-3 bottom).
_ANCHOR_TO_AN: dict[AnchorPoint, int] = {
    AnchorPoint.TOP_LEFT: 7,
    AnchorPoint.TOP_CENTER: 8,
    AnchorPoint.TOP_RIGHT: 9,
    AnchorPoint.CENTER_LEFT: 4,
    AnchorPoint.CENTER: 5,
    AnchorPoint.CENTER_RIGHT: 6,
    AnchorPoint.BOTTOM_LEFT: 1,
    AnchorPoint.BOTTOM_CENTER: 2,
    AnchorPoint.BOTTOM_RIGHT: 3,
}


@dataclass(frozen=True)
class AssLook:
    """The resolved ``add_subtitles`` look, expressed in ASS terms.

    ``font_family`` is the family name inside the font file (libass matches by
    name table, not filename); ``bold`` selects the bold face. ``background``
    of ``None`` means no box (BorderStyle 1); otherwise the RGBA box uses
    libass's BorderStyle 4 (whole-event background, burn-in only) with
    ``background_padding`` as the box padding.
    """

    font_family: str
    bold: bool
    font_px: int
    text_color: RGBColor
    highlight_color: RGBColor
    outline_px: int
    background: RGBAColor | None
    background_padding: int
    highlight_size_multiplier: float
    position: tuple[float, float]
    anchor: AnchorPoint
    box_width: float


def _ass_time(seconds: float) -> str:
    """``H:MM:SS.cc`` -- ASS timestamps are centisecond-precision only."""
    cs = max(0, round(seconds * 100))
    hours, rem = divmod(cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, centis = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_color(rgb: RGBColor, opacity: int = 255) -> str:
    """RGB + opacity (0-255, 255=opaque) -> ``&HAABBGGRR`` (ASS alpha: 00=opaque)."""
    r, g, b = rgb
    alpha = 255 - max(0, min(255, opacity))
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _ass_color_tag(rgb: RGBColor) -> str:
    """RGB -> the inline override form ``&HBBGGRR&`` (no alpha, trailing ``&``)."""
    r, g, b = rgb
    return f"&H{b:02X}{g:02X}{r:02X}&"


def _escape_text(text: str) -> str:
    """Neutralize ASS markup in transcribed text.

    ``{``/``}`` delimit override blocks and ``\\`` introduces escapes/breaks;
    ASS has no escape sequence for them, so they are substituted.
    """
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def _word_state_intervals(
    start: float, end: float, word_spans: list[tuple[float, float]]
) -> list[tuple[int, int, int | None]]:
    """Tile ``[start, end)`` into centisecond intervals keyed by active word.

    Returns ``(start_cs, end_cs, highlight_index)`` tuples; ``None`` means no
    word is active (the gap before/between/after words). Boundaries are
    rounded to centiseconds first and durations derived from the rounded
    boundaries, so rounding error cannot accumulate into overlaps or gaps.
    Zero-length intervals are dropped.
    """
    cue_start = round(start * 100)
    cue_end = round(end * 100)
    intervals: list[tuple[int, int, int | None]] = []
    cursor = cue_start
    for i, (w_start, w_end) in enumerate(word_spans):
        if cursor >= cue_end:
            break
        ws = min(cue_end, max(cursor, round(w_start * 100)))
        we = min(cue_end, round(w_end * 100))
        if ws > cursor:
            intervals.append((cursor, ws, None))
            cursor = ws
        if we > cursor:
            intervals.append((cursor, we, i))
            cursor = we
    if cue_end > cursor:
        intervals.append((cursor, cue_end, None))
    return intervals


def build_ass(
    transcription: Transcription,
    *,
    width: int,
    height: int,
    look: AssLook,
    window: tuple[float | None, float | None] | None = None,
) -> str:
    """Compile a segment-local transcription into a complete ASS document.

    ``transcription`` must already be on the timeline of the frames the
    ``subtitles=`` filter will see (the plan builder re-bases context onto the
    cut segment, and the ``-ss``-before-``-i`` decode resets PTS to zero, so
    segment-local is exactly right). ``window`` clips events to the effect's
    active range -- the ``subtitles`` filter has no timeline (``enable=``)
    support, so windowing happens here, at compile time.

    ``PlayResX/Y`` are set to the frame size entering the filter, so one
    script pixel is one video pixel and ``font_px`` carries over 1:1.
    """
    win_start = window[0] if window is not None else None
    win_stop = window[1] if window is not None else None

    primary = _ass_color(look.text_color)
    outline_colour = _ass_color((0, 0, 0))
    if look.background is not None:
        border_style = 4  # libass extension: whole-event background box (burn-in only)
        back_colour = _ass_color(look.background[:3], look.background[3])
        shadow = look.background_padding  # BorderStyle 4 reads box padding from Shadow
    else:
        border_style = 1
        back_colour = _ass_color((0, 0, 0))
        shadow = 0

    an = _ANCHOR_TO_AN[look.anchor]
    pos_x = look.position[0] * width
    pos_y = look.position[1] * height
    # Margins do not place positioned events, but they still define the wrap
    # width (PlayResX - MarginL - MarginR), which is how box_width carries over.
    margin_h = max(0, round((1.0 - look.box_width) / 2.0 * width))

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: Default,{look.font_family},{look.font_px},{primary},{primary},{outline_colour},"
            f"{back_colour},{-1 if look.bold else 0},0,0,0,100,100,0,0,{border_style},{look.outline_px},"
            f"{shadow},{an},{margin_h},{margin_h},0,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    scale = round(look.highlight_size_multiplier * 100)
    highlight = _ass_color_tag(look.highlight_color)
    placement = f"\\an{an}\\pos({pos_x:.1f},{pos_y:.1f})"

    for segment in transcription.segments:
        words = [_escape_text(w.word) for w in segment.words]
        if not any(w.strip() for w in words):
            continue
        start, end = segment.start, segment.end
        if win_start is not None:
            start = max(start, win_start)
        if win_stop is not None:
            end = min(end, win_stop)
        if end <= start:
            continue
        spans = [(w.start, w.end) for w in segment.words]
        for start_cs, end_cs, hl in _word_state_intervals(start, end, spans):
            if hl is None:
                text = " ".join(words)
            else:
                marked = f"{{\\1c{highlight}\\fscx{scale}\\fscy{scale}}}{words[hl]}{{\\r}}"
                text = " ".join([*words[:hl], marked, *words[hl + 1 :]])
            lines.append(
                f"Dialogue: 0,{_ass_time(start_cs / 100)},{_ass_time(end_cs / 100)},Default,,0,0,0,,"
                f"{{{placement}}}{text}"
            )

    return "\n".join(lines) + "\n"
