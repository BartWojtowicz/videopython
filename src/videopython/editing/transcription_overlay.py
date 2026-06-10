"""Burned-in word-level subtitles, compiled to ASS and rendered by libass.

The ``add_subtitles`` op consumes its transcription when the plan compiles:
the cue transforms (``max_words_per_cue`` chunking, sentence capitalization)
and the resolved look are baked into an ASS document, and ffmpeg's
``subtitles=`` filter (libass) burns it in. One pixel path everywhere:

* the streaming path emits one ``-vf`` entry -- decode-stage normally, or
  encode-stage when the op follows frame effects in plan order, so
  ``[fade, add_subtitles]`` keeps streaming in plan order;
* the eager path (:meth:`TranscriptionOverlay.apply`, used by
  ``VideoEdit.run``) pipes the in-memory frames through the same filter.

Geometry is resolution-relative (``font_scale``/``region``) and libass wraps
long cues within the box instead of failing, so there is no fit validation:
``predict_metadata`` is identity.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from PIL import ImageFont
from pydantic import Field

from videopython.base import _ffmpeg
from videopython.base.fonts import BUNDLED_FONT_FAMILIES, bundled_fonts_dir, load_font
from videopython.base.transcription import Transcription
from videopython.base.video import Video
from videopython.editing._ass import AnchorPoint, AssLook, build_ass, escape_filter_value
from videopython.editing.operation import Effect, FilterCtx

__all__ = ["TranscriptionOverlay", "SubtitleStyle", "SubtitleRegion"]

# Sentinel for ``background_color``: ``None`` already means "no background",
# so it cannot double as "derive from the style preset".
_AUTO: Literal["auto"] = "auto"

RGBColor = tuple[int, int, int]
RGBAColor = tuple[int, int, int, int]

_MISSING_CONTEXT_ERROR = (
    "TranscriptionOverlay requires transcription data. "
    "Pass it via VideoEdit.run(context={'transcription': ...}) or directly to apply()."
)


class SubtitleStyle(str, Enum):
    """Named look bundling colors / border / background / highlight.

    Lets a caller express intent ("boxed", "outline", ...) instead of a
    dozen individual numbers.
    """

    BOXED = "boxed"
    OUTLINE = "outline"
    CLEAN = "clean"
    KARAOKE = "karaoke"


class SubtitleRegion(str, Enum):
    """Vertical safe-area band the subtitle box is centered in."""

    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class _StyleParams:
    text_color: RGBColor
    highlight_color: RGBColor
    border: int
    background_color: RGBAColor | None
    background_padding: int
    highlight_size_multiplier: float


_STYLE_PRESETS: dict[SubtitleStyle, _StyleParams] = {
    SubtitleStyle.BOXED: _StyleParams((255, 235, 59), (76, 175, 80), 2, (0, 0, 0, 100), 15, 1.2),
    SubtitleStyle.OUTLINE: _StyleParams((255, 255, 255), (255, 235, 59), 4, None, 0, 1.15),
    SubtitleStyle.CLEAN: _StyleParams((255, 255, 255), (76, 175, 80), 2, None, 0, 1.1),
    SubtitleStyle.KARAOKE: _StyleParams((255, 255, 255), (255, 90, 95), 3, (0, 0, 0, 120), 18, 1.25),
}

# region -> normalized (x, y) center of the box; the anchor stays CENTER so
# the box is centered on this point. Chosen to sit inside a conventional safe
# area.
_REGION_POSITION: dict[SubtitleRegion, tuple[float, float]] = {
    SubtitleRegion.TOP: (0.5, 0.18),
    SubtitleRegion.CENTER: (0.5, 0.5),
    SubtitleRegion.BOTTOM: (0.5, 0.82),
}


class TranscriptionOverlay(Effect):
    """Renders animated word-by-word subtitles with the current word highlighted.

    Each word lights up in the highlight color (enlarged by the size
    multiplier) as it is spoken, based on transcription timestamps. Requires a
    word-level transcription, which the runner supplies via the
    ``requires=("transcription",)`` declaration -- re-based onto the segment's
    local timeline and delivered at plan-compile time through
    :class:`FilterCtx`; the op compiles to a libass ``subtitles=`` filter
    (:attr:`compiles_to_filter`), so subtitled edits run on the O(1)-memory
    streaming path at native speed. The eager path pipes frames through the
    same filter, so both paths share one pixel implementation.
    """

    op: Literal["add_subtitles"] = "add_subtitles"
    streamable: ClassVar[bool] = True
    requires: ClassVar[tuple[str, ...]] = ("transcription",)

    # ---- primary, resolution-independent surface ----
    style: SubtitleStyle = Field(
        SubtitleStyle.BOXED,
        description='Look preset bundling colors/border/background/highlight: "boxed", "outline", "clean", "karaoke".',
    )
    region: SubtitleRegion = Field(
        SubtitleRegion.BOTTOM,
        description='Vertical placement band: "top", "center", or "bottom" of the frame.',
    )
    font_scale: float = Field(
        0.055,
        gt=0.0,
        le=0.5,
        description=(
            "Base font height as a fraction of frame height (resolution-independent; the recommended "
            "way to size subtitles). Long cues wrap within the box."
        ),
    )
    max_words_per_cue: int | None = Field(
        5,
        ge=1,
        description=(
            "Maximum words shown on screen at once. Each transcription segment is re-chunked into "
            "cues of at most this many words, without bridging the silence gaps between segments, so "
            "subtitles stay readable and don't linger over pauses. None preserves the source "
            "transcription's segmentation."
        ),
    )
    capitalize: bool = Field(
        True,
        description=(
            "Capitalize the first letter of each sentence (first word, and words after '.', '!', '?'). "
            "Fixes lowercase sentence starts from word-level speech-to-text. Set False to render text "
            "exactly as transcribed."
        ),
    )
    font: Literal["anton", "bebas-neue", "lato-bold", "poppins-bold"] | None = Field(
        None,
        description=(
            "Bundled font for subtitles, or null for the default. "
            "'poppins-bold': clean geometric sans, general purpose. "
            "'lato-bold': humanist sans, very readable. "
            "'anton': tall condensed display, ideal for short-form vertical. "
            "'bebas-neue': bold condensed display, dramatic alternative."
        ),
    )
    font_filename: str | None = Field(
        None,
        description=(
            "Advanced override: path to a .ttf font file for subtitle text. Takes precedence over `font`; "
            "None for the bundled default font."
        ),
        json_schema_extra={"llm_hidden": True},
    )
    # ---- advanced overrides: None => derive from style/region/font_scale ----
    font_size: int | None = Field(
        None,
        ge=1,
        description=(
            "Advanced override: absolute base font size in pixels. Leave None to derive from "
            "`font_scale` (recommended -- resolution-independent)."
        ),
    )
    font_border_size: int | None = Field(
        None, ge=0, description="Advanced override for outline thickness in px. None takes it from `style`."
    )
    text_color: RGBColor | None = Field(
        None, description="Advanced override for default text color [R, G, B] (0-255). None takes it from `style`."
    )
    background_color: RGBAColor | None | Literal["auto"] = Field(
        _AUTO,
        description=(
            'Advanced override for the box background [R, G, B, A] (0-255). "auto" takes it from `style`; '
            "null explicitly disables the background."
        ),
    )
    background_padding: int | None = Field(
        None, ge=0, description="Advanced override: px between text and background edge. None takes it from `style`."
    )
    highlight_color: RGBColor | None = Field(
        None, description="Advanced override for the spoken-word color [R, G, B]. None takes it from `style`."
    )
    highlight_size_multiplier: float | None = Field(
        None, gt=0, description="Advanced override: scale factor for the highlighted word. None takes it from `style`."
    )
    position: tuple[float, float] | None = Field(
        None,
        description="Advanced override: box center as normalized (x, y). None derives it from `region`.",
    )
    box_width: float | None = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Advanced override: box width as a fraction of frame width in (0, 1]. None uses 0.6.",
    )
    anchor: AnchorPoint | None = Field(
        None, description="Advanced override: which point of the box sits at the position. None uses center."
    )

    # ------------------------------------------------------------- resolution

    @property
    def compiles_to_filter(self) -> bool:
        return True

    def _style_params(self) -> _StyleParams:
        """Effective look: the ``style`` preset overlaid by any explicit overrides."""
        p = _STYLE_PRESETS[self.style]
        bg = p.background_color if self.background_color == _AUTO else self.background_color
        return _StyleParams(
            text_color=self.text_color or p.text_color,
            highlight_color=self.highlight_color or p.highlight_color,
            border=self.font_border_size if self.font_border_size is not None else p.border,
            background_color=bg,
            background_padding=(
                self.background_padding if self.background_padding is not None else p.background_padding
            ),
            highlight_size_multiplier=(
                self.highlight_size_multiplier
                if self.highlight_size_multiplier is not None
                else p.highlight_size_multiplier
            ),
        )

    def _transform(self, transcription: Transcription) -> Transcription:
        """Apply the cue transforms every render path MUST share."""
        if self.max_words_per_cue is not None:
            transcription = transcription.chunk_segments(self.max_words_per_cue)
        if self.capitalize:
            transcription = transcription.capitalize_sentences()
        return transcription

    def _ass_font(self) -> tuple[str, bool, Path]:
        """``(family, bold, fontsdir)`` for libass font matching.

        libass matches by the family name inside the font file, not the
        filename, so a ``font_filename`` override is probed for its name-table
        family (falling back to the bundled default on an unreadable file --
        the same never-hard-fail policy as ``load_font``).
        """
        if self.font_filename:
            path = Path(self.font_filename)
            try:
                family, face = ImageFont.truetype(str(path), 16).getname()
                return family, "bold" in (face or "").lower(), path.parent
            except (OSError, ValueError):
                pass
        family, bold = BUNDLED_FONT_FAMILIES.get(self.font, BUNDLED_FONT_FAMILIES[None])
        return family, bold, bundled_fonts_dir()

    def _ass_look(self, height: int) -> AssLook:
        """Resolve every override-or-preset field into the ASS look."""
        sp = self._style_params()
        font_px = self.font_size if self.font_size is not None else max(1, round(self.font_scale * height))
        # libass interprets Fontsize as the GDI cell height (ascender +
        # descender) while PIL sizes the em square; scale by the font's
        # cell/em ratio so font_scale keeps its historical apparent size
        # (1.5x divergence for a tall display font like Anton without this).
        metrics_font = load_font(self.font_filename or self.font, 100)
        if isinstance(metrics_font, ImageFont.FreeTypeFont):
            ascent, descent = metrics_font.getmetrics()
            font_px = max(1, round(font_px * (ascent + descent) / 100))
        family, bold, _ = self._ass_font()
        return AssLook(
            font_family=family,
            bold=bold,
            font_px=font_px,
            text_color=sp.text_color,
            highlight_color=sp.highlight_color,
            outline_px=sp.border,
            background=sp.background_color,
            background_padding=sp.background_padding,
            highlight_size_multiplier=sp.highlight_size_multiplier,
            position=self.position if self.position is not None else _REGION_POSITION[self.region],
            anchor=self.anchor if self.anchor is not None else AnchorPoint.CENTER,
            box_width=self.box_width if self.box_width is not None else 0.6,
        )

    def _compile_ass(self, transcription: Transcription, width: int, height: int) -> str:
        """The full ASS document for ``transcription`` at the given frame size.

        ``transcription`` timestamps must be local to the timeline the frames
        come from (the plan builder re-bases context onto the cut segment; the
        eager path receives the video-local transcription directly). The
        ``window`` is applied by clipping event times -- the ``subtitles``
        filter has no timeline support.
        """
        window = (self.window.start, self.window.stop) if self.window is not None else None
        return build_ass(
            self._transform(transcription),
            width=width,
            height=height,
            look=self._ass_look(height),
            window=window,
        )

    def _write_ass(self, document: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".ass", delete=False, encoding="utf-8")
        try:
            tmp.write(document)
        finally:
            tmp.close()
        return Path(tmp.name)

    def _filter_expr(self, ass_path: Path) -> str:
        _, _, fonts_dir = self._ass_font()
        return f"subtitles=filename={escape_filter_value(str(ass_path))}:fontsdir={escape_filter_value(str(fonts_dir))}"

    # ------------------------------------------------------------- execution

    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None:
        """Compile to a libass ``subtitles=`` filter entry.

        Consumes the segment-local transcription from ``ctx.context`` at plan
        compile time: writes a temp ``.ass`` (registered on ``ctx.owned_files``
        for the runner to delete after streaming) and emits one ``-vf`` entry.
        A missing transcription raises the op's clear context error here --
        before any decode -- mirroring the eager path.
        """
        transcription = ctx.context.get("transcription")
        if not isinstance(transcription, Transcription):
            raise ValueError(_MISSING_CONTEXT_ERROR)
        ass_path = self._write_ass(self._compile_ass(transcription, ctx.width, ctx.height))
        ctx.owned_files.append(ass_path)
        return self._filter_expr(ass_path)

    def apply(
        self,
        video: Video,
        transcription: Transcription | None = None,
        **_context: Any,
    ) -> Video:
        """Eager render: pipe the in-memory frames through the same filter.

        One ffmpeg rawvideo->``subtitles=``->rawvideo roundtrip, so eager and
        streaming output come from the same renderer. The rawvideo pipe's
        timestamps start at zero, matching the video-local transcription the
        caller passes (``VideoEdit.run`` re-bases per segment). Frames are
        replaced in place, like every effect; audio is untouched.
        """
        if transcription is None:
            raise ValueError(_MISSING_CONTEXT_ERROR)
        n_frames, height, width = video.frames.shape[:3]
        ass_path = self._write_ass(self._compile_ass(transcription, width, height))
        try:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(video.fps),
                "-i",
                "pipe:0",
                "-vf",
                self._filter_expr(ass_path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
            out = _ffmpeg.run(cmd, stdin=video.frames.tobytes())
        finally:
            ass_path.unlink(missing_ok=True)
        frames = np.frombuffer(out, dtype=np.uint8).reshape(-1, height, width, 3)
        if frames.shape[0] != n_frames:
            raise RuntimeError(
                f"subtitles filter changed the frame count from {n_frames} to {frames.shape[0]}; "
                "effects must preserve shape and frame count."
            )
        video.frames = frames.copy()  # frombuffer is read-only; effects own writable frames
        return video
