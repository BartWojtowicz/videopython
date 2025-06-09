from dataclasses import dataclass
from typing import Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from videopython.base.video import Video
from videopython.utils.text import AnchorPoint, ImageText, MarginType, PositionType, RGBAColor, RGBColor, TextAlign


@dataclass
class TranscriptionWord:
    start: float
    end: float
    word: str


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    words: list[TranscriptionWord]


@dataclass
class Transcription:
    segments: list[TranscriptionSegment]


class TranscriptionOverlay:
    def __init__(
        self,
        transcription: Transcription,
        font_filename: str,
        font_size: int = 24,
        text_color: RGBColor = (255, 255, 255),
        background_color: RGBAColor | None = (0, 0, 0, 180),
        background_padding: int = 15,
        position: PositionType = (0.5, 0.9),
        box_width: Union[int, float] = 0.8,
        text_align: TextAlign = TextAlign.CENTER,
        anchor: AnchorPoint = AnchorPoint.BOTTOM_CENTER,
        margin: MarginType = 20,
        highlight_color: RGBColor = (255, 255, 0),
        highlight_size_multiplier: float = 1.2,
        highlight_bold_font: str | None = None,
    ):
        """
        Initialize TranscriptionOverlay effect.

        Args:
            transcription: Transcription object containing segments and words
            font_filename: Path to font file for text rendering
            font_size: Base font size for text
            text_color: RGB color for normal text
            background_color: RGBA background color (None for no background)
            background_padding: Padding around text background
            position: Position of text box (relative 0-1 or absolute pixels)
            box_width: Width of text box (relative 0-1 or absolute pixels)
            text_align: Text alignment within box
            anchor: Anchor point for text positioning
            margin: Margin around text box
            highlight_color: RGB color for highlighted words
            highlight_size_multiplier: Size multiplier for highlighted words
            highlight_bold_font: Optional bold font for highlighting
        """
        self.transcription = transcription
        self.font_filename = font_filename
        self.font_size = font_size
        self.text_color = text_color
        self.background_color = background_color
        self.background_padding = background_padding
        self.position = position
        self.box_width = box_width
        self.text_align = text_align
        self.anchor = anchor
        self.margin = margin
        self.highlight_color = highlight_color
        self.highlight_size_multiplier = highlight_size_multiplier
        self.highlight_bold_font = highlight_bold_font

        # Cache for text overlays to avoid regenerating identical frames
        self._overlay_cache: dict[tuple[str, int | None], np.ndarray] = {}

    def _get_active_segment(self, timestamp: float) -> TranscriptionSegment | None:
        """Get the transcription segment active at the given timestamp."""
        for segment in self.transcription.segments:
            if segment.start <= timestamp <= segment.end:
                return segment
        return None

    def _get_active_word_index(self, segment: TranscriptionSegment, timestamp: float) -> int | None:
        """Get the index of the word being spoken at the given timestamp within a segment."""
        for i, word in enumerate(segment.words):
            if word.start <= timestamp <= word.end:
                return i
        return None

    def _create_text_overlay(
        self, video_shape: tuple[int, int, int], segment: TranscriptionSegment, highlight_word_index: int | None
    ) -> np.ndarray:
        """Create a text overlay image for the given segment and highlight."""
        # Use video frame dimensions for overlay
        height, width = video_shape[:2]

        # Create cache key based on segment text and highlight
        cache_key = (segment.text, highlight_word_index)
        if cache_key in self._overlay_cache:
            return self._overlay_cache[cache_key]

        # Create ImageText with video dimensions
        img_text = ImageText(image_size=(width, height), background=(0, 0, 0, 0))

        # Write text with highlighting
        img_text.write_text_box(
            text=segment.text,
            font_filename=self.font_filename,
            xy=self.position,
            box_width=self.box_width,
            font_size=self.font_size,
            text_color=self.text_color,
            background_color=self.background_color,
            background_padding=self.background_padding,
            place=self.text_align,
            anchor=self.anchor,
            margin=self.margin,
            highlight_word_index=highlight_word_index,
            highlight_color=self.highlight_color,
            highlight_size_multiplier=self.highlight_size_multiplier,
            highlight_bold_font=self.highlight_bold_font,
        )

        overlay_image = img_text.img_array

        # Cache the overlay
        self._overlay_cache[cache_key] = overlay_image

        return overlay_image

    def apply(self, video: Video) -> Video:
        """Apply transcription overlay to video frames."""
        print("Applying transcription overlay...")

        new_frames = []

        for frame_idx, frame in enumerate(tqdm(video.frames)):
            # Calculate timestamp for this frame
            timestamp = frame_idx / video.fps

            # Get active segment at this timestamp
            active_segment = self._get_active_segment(timestamp)

            if active_segment is None:
                # No active transcription, keep original frame
                new_frames.append(frame)
                continue

            # Get active word index for highlighting
            highlight_word_index = self._get_active_word_index(active_segment, timestamp)

            # Create text overlay
            text_overlay = self._create_text_overlay(video.frame_shape, active_segment, highlight_word_index)

            # Apply overlay to frame
            overlaid_frame = self._apply_overlay_to_frame(frame, text_overlay)
            new_frames.append(overlaid_frame)

        # Create new video with overlaid frames
        new_video = Video.from_frames(np.array(new_frames), fps=video.fps)
        new_video.audio = video.audio  # Preserve audio

        return new_video

    def _apply_overlay_to_frame(self, frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Apply a text overlay to a single frame."""

        # Convert frame to PIL Image
        frame_pil = Image.fromarray(frame)

        # Convert overlay to PIL Image
        overlay_pil = Image.fromarray(overlay)

        # Paste overlay onto frame using alpha channel
        frame_pil.paste(overlay_pil, (0, 0), overlay_pil)

        return np.array(frame_pil)
