"""Subtitle overlay effect.

``TranscriptionOverlay`` is an :class:`Effect` that renders animated
word-by-word subtitles onto a :class:`Video` using a word-level
:class:`Transcription`. Rendering is delegated to ``ImageText`` from
the sibling module.

The public surface is intentionally small and *resolution-independent*:
pick a ``style`` preset, a ``region``, and a ``font_scale`` (fraction of
frame height). The legacy absolute fields (``font_size``, explicit colors,
``position``/``anchor``/``box_width``/``margin``) remain as optional advanced
overrides for back-compat -- left unset they are derived from the presets, so
an authored plan cannot encode a resolution-specific value that overflows the
real (post-transform) frame.

Fit safety is layered:

* one routine (:meth:`_resolve_layout`) decides the final font size for both
  the dry-run and the render, so they can never disagree;
* it auto-shrinks the font within a legible band and clamps the box inside
  the frame (graceful render fit);
* only a cue that cannot fit even at the minimum legible size is an error,
  and that error is raised by :meth:`predict_metadata` at
  ``VideoEdit.validate()`` time -- before any frame/GPU work -- never
  mid-render.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal

import numpy as np
from PIL import Image
from pydantic import Field
from tqdm import tqdm

from videopython.base.image_text import AnchorPoint, ImageText, TextAlign
from videopython.base.transcription import Transcription, TranscriptionSegment
from videopython.base.video import Video, VideoMetadata
from videopython.editing.operation import Effect

__all__ = ["TranscriptionOverlay", "SubtitleStyle", "SubtitleRegion"]

logger = logging.getLogger(__name__)

# Sentinel for ``background_color``: ``None`` already means "no background",
# so it cannot double as "derive from the style preset".
_AUTO: Literal["auto"] = "auto"

RGBColor = tuple[int, int, int]
RGBAColor = tuple[int, int, int, int]


class SubtitleStyle(str, Enum):
    """Named look bundling colors / border / background / highlight.

    Lets a caller express intent ("boxed", "outline", ...) instead of a
    dozen individual numbers. ``BOXED`` reproduces the historical defaults
    exactly, so upgrading without changing fields is visually a no-op except
    for the now resolution-relative font size.
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
    # Exactly the pre-redesign defaults.
    SubtitleStyle.BOXED: _StyleParams((255, 235, 59), (76, 175, 80), 2, (0, 0, 0, 100), 15, 1.2),
    SubtitleStyle.OUTLINE: _StyleParams((255, 255, 255), (255, 235, 59), 4, None, 0, 1.15),
    SubtitleStyle.CLEAN: _StyleParams((255, 255, 255), (76, 175, 80), 2, None, 0, 1.1),
    SubtitleStyle.KARAOKE: _StyleParams((255, 255, 255), (255, 90, 95), 3, (0, 0, 0, 120), 18, 1.25),
}

# region -> normalized (x, y) center of the box; anchor stays CENTER so the
# box is centered on this point. Chosen to sit inside a conventional safe area.
_REGION_POSITION: dict[SubtitleRegion, tuple[float, float]] = {
    SubtitleRegion.TOP: (0.5, 0.18),
    SubtitleRegion.CENTER: (0.5, 0.5),
    SubtitleRegion.BOTTOM: (0.5, 0.82),
}


@dataclass(frozen=True)
class _CueBox:
    """Absolute, frame-clamped placement of one cue's text box."""

    x: int
    y: int
    box_w: int
    height: int
    fits: bool


@dataclass(frozen=True)
class _ResolvedConfig:
    """Every override-or-preset field resolved to a concrete value once.

    Deterministic from the model fields, so the dry-run and the render
    derive identical geometry and look (parity).
    """

    position: tuple[float, float]
    anchor: AnchorPoint
    box_width: float
    text_align: TextAlign
    margin: int | tuple[int, int, int, int]
    style: _StyleParams


@dataclass(frozen=True)
class _SubtitleLayout:
    """Outcome of resolving the overlay against a concrete frame size.

    ``segments`` are post-transform cues (``chunk`` + ``capitalize``) and
    ``config`` is the resolved geometry/look -- both shared by render and
    dry-run so they measure and draw identical boxes. ``font_px`` is the
    single font size both paths use. ``fits`` is False with a populated
    ``error`` only when a cue cannot fit even at the minimum legible size.
    """

    segments: list[TranscriptionSegment]
    config: _ResolvedConfig
    font_px: int
    fits: bool
    error: str | None


class TranscriptionOverlay(Effect):
    """Renders animated word-by-word subtitles with the current word highlighted.

    Each word lights up in the highlight color as it is spoken, based on
    transcription timestamps. Requires a word-level transcription, which the
    runner supplies via the ``requires=("transcription",)`` declaration.

    Geometry is resolution-relative by default (``font_scale``/``region``), so
    a plan validated by ``VideoEdit.validate()`` that passes will also render;
    a plan that cannot fit fails fast in :meth:`predict_metadata` instead of
    crashing mid-render after expensive upstream ops.
    """

    op: Literal["add_subtitles"] = "add_subtitles"
    streamable: ClassVar[bool] = False
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
            "way to size subtitles). Auto-shrinks toward `min_font_scale` if a cue would overflow."
        ),
    )
    min_font_scale: float = Field(
        0.030,
        gt=0.0,
        le=0.5,
        description=(
            "Lower bound for auto-fit shrinking, as a fraction of frame height. A cue that cannot fit "
            "even at this size is a validation error rather than an illegible render."
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
    font_filename: str | None = Field(
        None,
        description="Path to a .ttf font file for rendering subtitle text, or None for the bundled default font.",
    )
    highlight_bold_font: str | None = Field(
        None, description="Path to a bold .ttf font for the highlighted word, or None to use the regular font."
    )

    # ---- advanced overrides: None => derive from style/region/font_scale ----
    font_size: int | None = Field(
        None,
        ge=1,
        description=(
            "Advanced override: absolute base font size in pixels. Leave None to derive from "
            "`font_scale` (recommended -- resolution-independent and overflow-safe)."
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
    text_align: TextAlign | None = Field(
        None, description='Advanced override: text alignment within the box. None uses "center".'
    )
    anchor: AnchorPoint | None = Field(
        None, description="Advanced override: which point of the box sits at the position. None uses center."
    )
    margin: int | tuple[int, int, int, int] | None = Field(
        None,
        description="Advanced override: space around the box in px (or [top, right, bottom, left]). None uses 20.",
    )

    # ------------------------------------------------------------- resolution

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

    def _resolve_config(self) -> _ResolvedConfig:
        """Resolve every override-or-preset field to a concrete value once."""
        return _ResolvedConfig(
            position=self.position if self.position is not None else _REGION_POSITION[self.region],
            anchor=self.anchor if self.anchor is not None else AnchorPoint.CENTER,
            box_width=self.box_width if self.box_width is not None else 0.6,
            text_align=self.text_align if self.text_align is not None else TextAlign.CENTER,
            margin=self.margin if self.margin is not None else 20,
            style=self._style_params(),
        )

    def _transform(self, transcription: Transcription) -> Transcription:
        """Apply the cue transforms render and dry-run MUST share."""
        if self.max_words_per_cue is not None:
            transcription = transcription.chunk_segments(self.max_words_per_cue)
        if self.capitalize:
            transcription = transcription.capitalize_sentences()
        return transcription

    def _place_cue(self, img_text: ImageText, text: str, font_px: int, cfg: _ResolvedConfig) -> _CueBox | None:
        """Measure ``text`` at ``font_px`` and clamp its box inside the margins.

        Returns ``None`` for a degenerate (whitespace-only) cue. ``fits`` is
        False when the box is larger than the drawable area even after
        clamping -- i.e. shrinking the font is the only remedy. Used by both
        the fit search and the renderer, so they never diverge. Margin math
        comes from ``ImageText.available_region`` (one source of truth with
        ``measure_text_box``).
        """
        rect = img_text.measure_text_box(
            text=text,
            font_filename=self.font_filename,
            xy=cfg.position,
            box_width=cfg.box_width,
            font_size=font_px,
            anchor=cfg.anchor,
            margin=cfg.margin,
        )
        if rect.height == 0:
            return None
        box_w = int(rect.width)
        box_h = rect.height
        left, top, avail_w, avail_h = img_text.available_region(cfg.margin)
        fits = box_w <= avail_w and box_h <= avail_h
        x = min(max(int(round(rect.x)), left), left + avail_w - box_w)
        y = min(max(int(round(rect.y)), top), top + avail_h - box_h)
        return _CueBox(x=x, y=y, box_w=box_w, height=box_h, fits=fits)

    def _resolve_layout(self, width: int, height: int, transcription: Transcription) -> _SubtitleLayout:
        """Single source of truth for config + font size + fit (render & dry-run)."""
        segments = self._transform(transcription).segments
        cues = [s for s in segments if s.text.strip()]
        cfg = self._resolve_config()

        desired = self.font_size if self.font_size is not None else max(1, round(self.font_scale * height))
        floor = max(1, round(self.min_font_scale * height))
        # Never search above the desired size nor below the legible floor --
        # but if the user pinned a font_size below the floor, honor it.
        lo = min(desired, floor)

        img_text = ImageText(image_size=(height, width), background=(0, 0, 0, 0))

        def first_unfit(font_px: int) -> TranscriptionSegment | None:
            for cue in cues:
                box = self._place_cue(img_text, cue.text, font_px, cfg)
                if box is not None and not box.fits:
                    return cue
            return None

        # "fits" is monotonic in font size (a larger font never fits where a
        # smaller one didn't -- box width is font-independent and box height
        # is non-decreasing in font size), so binary-search the largest fit.
        if first_unfit(desired) is None:
            return _SubtitleLayout(segments, cfg, desired, True, None)
        offender = first_unfit(lo)
        if offender is not None:
            error = (
                f"Subtitle cue {offender.text!r} cannot fit in a {width}x{height} frame even at the "
                f"minimum font size ({lo}px, min_font_scale={self.min_font_scale}). Lower min_font_scale, "
                f"reduce max_words_per_cue, widen box_width, or render at a larger resolution."
            )
            return _SubtitleLayout(segments, cfg, lo, False, error)
        low, high = lo, desired  # invariant: fits at low, not at high
        while high - low > 1:
            mid = (low + high) // 2
            if first_unfit(mid) is None:
                low = mid
            else:
                high = mid
        return _SubtitleLayout(segments, cfg, low, True, None)

    # ------------------------------------------------------------- timeline

    def _get_active_segment(self, transcription: Transcription, timestamp: float) -> TranscriptionSegment | None:
        for segment in transcription.segments:
            if segment.start <= timestamp <= segment.end:
                return segment
        return None

    def _get_active_word_index(self, segment: TranscriptionSegment, timestamp: float) -> int | None:
        for i, word in enumerate(segment.words):
            if word.start <= timestamp <= word.end:
                return i
        return None

    def _create_text_overlay(
        self,
        video_shape: tuple[int, int, int],
        segment: TranscriptionSegment,
        highlight_word_index: int | None,
        layout: _SubtitleLayout,
        cache: dict[tuple[str, int | None], np.ndarray],
    ) -> np.ndarray:
        height, width = video_shape[:2]
        cache_key = (segment.text, highlight_word_index)
        if cache_key in cache:
            return cache[cache_key]

        cfg = layout.config
        img_text = ImageText(image_size=(height, width), background=(0, 0, 0, 0))
        box = self._place_cue(img_text, segment.text, layout.font_px, cfg)
        if box is not None:
            sp = cfg.style
            # Absolute, pre-clamped placement (anchor=TOP_LEFT, explicit px box,
            # margin already applied) -- the same numbers _resolve_layout used,
            # so a layout that validated cannot raise OutOfBoundsError here.
            img_text.write_text_box(
                text=segment.text,
                font_filename=self.font_filename,
                xy=(box.x, box.y),
                box_width=box.box_w,
                font_size=layout.font_px,
                font_border_size=sp.border,
                text_color=sp.text_color,
                background_color=sp.background_color,
                background_padding=sp.background_padding,
                place=cfg.text_align,
                anchor=AnchorPoint.TOP_LEFT,
                margin=0,
                words=[w.word for w in segment.words],
                highlight_word_index=highlight_word_index,
                highlight_color=sp.highlight_color,
                highlight_size_multiplier=sp.highlight_size_multiplier,
                highlight_bold_font=self.highlight_bold_font,
            )

        overlay_image = img_text.img_array
        cache[cache_key] = overlay_image
        return overlay_image

    def apply(  # type: ignore[override]
        self,
        video: Video,
        transcription: Transcription | None = None,
    ) -> Video:
        if transcription is None:
            raise ValueError(
                "TranscriptionOverlay requires transcription data. "
                "Pass it via VideoEdit.run(context={'transcription': ...}) or directly to apply()."
            )

        height, width = video.frame_shape[:2]
        layout = self._resolve_layout(width, height, transcription)
        if not layout.fits:
            # Should be unreachable when the plan went through validate(); kept
            # as defense in depth so a direct apply() still fails clearly
            # rather than crashing mid-render in ImageText.
            raise ValueError(layout.error)

        # Per-call memo of rendered overlays, keyed by (cue text, highlighted
        # word). Local rather than instance state so the model stays stateless
        # and re-entrant -- a reused instance can render differently sized
        # videos without serving a stale-resolution overlay.
        cache: dict[tuple[str, int | None], np.ndarray] = {}
        transformed = Transcription(segments=layout.segments, language=transcription.language)

        logger.info("Applying transcription overlay (font %dpx)...", layout.font_px)
        new_frames = []
        for frame_index, frame in enumerate(tqdm(video.frames, desc="Transcription overlay")):
            timestamp = frame_index / video.fps
            active_segment = self._get_active_segment(transformed, timestamp)
            if active_segment is None:
                new_frames.append(frame)
                continue
            highlight_word_index = self._get_active_word_index(active_segment, timestamp)
            text_overlay = self._create_text_overlay(
                video.frame_shape, active_segment, highlight_word_index, layout, cache
            )
            new_frames.append(self._apply_overlay_to_frame(frame, text_overlay))

        new_video = Video.from_frames(np.array(new_frames), fps=video.fps)
        new_video.audio = video.audio
        return new_video

    def predict_metadata(
        self,
        meta: VideoMetadata,
        transcription: Transcription | None = None,
        **_context: Any,
    ) -> VideoMetadata:
        """Identity for metadata (shape/count preserved) -- but fail fast here
        if the resolved subtitles cannot fit the predicted frame.

        This is the backstop that closes the validate/run gap: ``VideoEdit``
        runs it during the dry-run, so an un-fittable plan is rejected before
        any frame/GPU work, symmetric with the timing/dimension checks. Mirrors
        ``SilenceRemoval``: with no ``transcription`` in the validate context
        the layout cannot be checked, so this is a no-op identity (the same
        conditional guarantee as time re-basing).
        """
        if transcription is None:
            return meta
        layout = self._resolve_layout(meta.width, meta.height, transcription)
        if not layout.fits:
            raise ValueError(layout.error)
        return meta

    def _apply_overlay_to_frame(self, frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        frame_pil = Image.fromarray(frame)
        overlay_pil = Image.fromarray(overlay)
        frame_pil.paste(overlay_pil, (0, 0), overlay_pil)
        return np.array(frame_pil)
