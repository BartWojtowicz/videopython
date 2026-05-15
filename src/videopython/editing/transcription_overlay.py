"""Subtitle overlay effect.

``TranscriptionOverlay`` is an :class:`Effect` that renders animated
word-by-word subtitles onto a :class:`Video` using a word-level
:class:`Transcription`. Rendering is delegated to ``ImageText`` from
the sibling module.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Literal

import numpy as np
from PIL import Image
from pydantic import Field, PrivateAttr
from tqdm import tqdm

from videopython.base.image_text import AnchorPoint, ImageText, TextAlign
from videopython.base.transcription import Transcription, TranscriptionSegment
from videopython.base.video import Video
from videopython.editing.operation import Effect

__all__ = ["TranscriptionOverlay"]

logger = logging.getLogger(__name__)


class TranscriptionOverlay(Effect):
    """Renders animated word-by-word subtitles with the current word highlighted.

    Each word lights up in the highlight color as it is spoken, based on
    transcription timestamps. Requires a word-level transcription, which the
    runner supplies via the ``requires=("transcription",)`` declaration.
    """

    op: Literal["add_subtitles"] = "add_subtitles"
    streamable: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ("transcription",)

    font_filename: str | None = Field(
        None,
        description="Path to a .ttf font file for rendering subtitle text, or None for the bundled default font.",
    )
    font_size: int = Field(40, ge=1, description="Base font size in pixels.")
    font_border_size: int = Field(
        2, ge=0, description="Outline thickness around each character in pixels. 0 = no outline."
    )
    text_color: tuple[int, int, int] = Field((255, 235, 59), description="Default text color as [R, G, B], each 0-255.")
    background_color: tuple[int, int, int, int] | None = Field(
        (0, 0, 0, 100),
        description="Subtitle box background as [R, G, B, A] (0-255), or None to disable the background.",
    )
    background_padding: int = Field(15, ge=0, description="Pixels of space between text and background edge.")
    position: tuple[float, float] = Field(
        (0.5, 0.7),
        description="Text box center as normalized (x, y). (0, 0) = top-left, (1, 1) = bottom-right.",
    )
    box_width: float = Field(
        0.6, gt=0.0, le=1.0, description="Width of the text box as a fraction of frame width, in (0, 1]."
    )
    text_align: TextAlign = Field(
        TextAlign.CENTER, description='Text alignment within the box: "left", "right", or "center".'
    )
    anchor: AnchorPoint = Field(
        AnchorPoint.CENTER, description="Which point of the text box sits at the position coordinate."
    )
    margin: int | tuple[int, int, int, int] = Field(
        20,
        description="Space around the text box in pixels, or a [top, right, bottom, left] tuple of per-side values.",
    )
    highlight_color: tuple[int, int, int] = Field(
        (76, 175, 80), description="Color for the currently spoken word as [R, G, B]."
    )
    highlight_size_multiplier: float = Field(
        1.2, gt=0, description="Scale factor for the highlighted word. 1.0 = same size, 1.2 = 20% larger."
    )
    highlight_bold_font: str | None = Field(
        None, description="Path to a bold .ttf font for the highlighted word, or None to use the regular font."
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

    _overlay_cache: dict[tuple[str, int | None], np.ndarray] = PrivateAttr(default_factory=dict)

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
    ) -> np.ndarray:
        height, width = video_shape[:2]
        cache_key = (segment.text, highlight_word_index)
        if cache_key in self._overlay_cache:
            return self._overlay_cache[cache_key]

        img_text = ImageText(image_size=(height, width), background=(0, 0, 0, 0))
        img_text.write_text_box(
            text=segment.text,
            font_filename=self.font_filename,
            xy=self.position,
            box_width=self.box_width,
            font_size=self.font_size,
            font_border_size=self.font_border_size,
            text_color=self.text_color,
            background_color=self.background_color,
            background_padding=self.background_padding,
            place=self.text_align,
            anchor=self.anchor,
            margin=self.margin,
            words=[w.word for w in segment.words],
            highlight_word_index=highlight_word_index,
            highlight_color=self.highlight_color,
            highlight_size_multiplier=self.highlight_size_multiplier,
            highlight_bold_font=self.highlight_bold_font,
        )

        overlay_image = img_text.img_array
        self._overlay_cache[cache_key] = overlay_image
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

        if self.max_words_per_cue is not None:
            transcription = transcription.chunk_segments(self.max_words_per_cue)
        if self.capitalize:
            transcription = transcription.capitalize_sentences()

        logger.info("Applying transcription overlay...")
        new_frames = []
        for frame_index, frame in enumerate(tqdm(video.frames, desc="Transcription overlay")):
            timestamp = frame_index / video.fps
            active_segment = self._get_active_segment(transcription, timestamp)
            if active_segment is None:
                new_frames.append(frame)
                continue
            highlight_word_index = self._get_active_word_index(active_segment, timestamp)
            text_overlay = self._create_text_overlay(video.frame_shape, active_segment, highlight_word_index)
            new_frames.append(self._apply_overlay_to_frame(frame, text_overlay))

        new_video = Video.from_frames(np.array(new_frames), fps=video.fps)
        new_video.audio = video.audio
        return new_video

    def _apply_overlay_to_frame(self, frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        frame_pil = Image.fromarray(frame)
        overlay_pil = Image.fromarray(overlay)
        frame_pil.paste(overlay_pil, (0, 0), overlay_pil)
        return np.array(frame_pil)
