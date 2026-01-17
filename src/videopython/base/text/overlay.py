"""
Beware, the code below was heavily "vibe-coded".

The main purpose of this file are 2 classes:
1. `ImageText` class for creating RGBA image with rendered subtitles
2. `TranscriptionOverlay` class, which takes the `Transcription` and `Video` objects and overlays subtitles on `Video`.
"""

from enum import Enum
from typing import TypeAlias

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from videopython.base.exceptions import OutOfBoundsError
from videopython.base.text.transcription import Transcription, TranscriptionSegment
from videopython.base.video import Video

__all__ = ["TranscriptionOverlay", "ImageText", "TextAlign", "AnchorPoint"]

# Type aliases for clarity
MarginType: TypeAlias = int | tuple[int, int, int, int]
RGBColor: TypeAlias = tuple[int, int, int]
RGBAColor: TypeAlias = tuple[int, int, int, int]
PositionType: TypeAlias = tuple[int, int] | tuple[float, float]

# Text highlight styling constants
DEFAULT_HIGHLIGHT_SIZE_MULTIPLIER = 1.5  # Make highlighted words 50% larger


# Text alignment enum
class TextAlign(str, Enum):
    """Defines text alignment options for positioning within containers."""

    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


class AnchorPoint(str, Enum):
    """Defines anchor points for positioning text elements."""

    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    TOP_RIGHT = "top-right"
    CENTER_LEFT = "center-left"
    CENTER = "center"
    CENTER_RIGHT = "center-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"

    # Group anchor points by their horizontal position
    @classmethod
    def left_anchors(cls) -> tuple["AnchorPoint", ...]:
        return (cls.TOP_LEFT, cls.CENTER_LEFT, cls.BOTTOM_LEFT)

    @classmethod
    def center_anchors(cls) -> tuple["AnchorPoint", ...]:
        return (cls.TOP_CENTER, cls.CENTER, cls.BOTTOM_CENTER)

    @classmethod
    def right_anchors(cls) -> tuple["AnchorPoint", ...]:
        return (cls.TOP_RIGHT, cls.CENTER_RIGHT, cls.BOTTOM_RIGHT)

    # Group anchor points by their vertical position
    @classmethod
    def top_anchors(cls) -> tuple["AnchorPoint", ...]:
        return (cls.TOP_LEFT, cls.TOP_CENTER, cls.TOP_RIGHT)

    @classmethod
    def middle_anchors(cls) -> tuple["AnchorPoint", ...]:
        return (cls.CENTER_LEFT, cls.CENTER, cls.CENTER_RIGHT)

    @classmethod
    def bottom_anchors(cls) -> tuple["AnchorPoint", ...]:
        return (cls.BOTTOM_LEFT, cls.BOTTOM_CENTER, cls.BOTTOM_RIGHT)


class ImageText:
    def __init__(
        self,
        image_size: tuple[int, int] = (1920, 1080),  # (height, width) - NumPy convention
        mode: str = "RGBA",
        background: RGBAColor = (0, 0, 0, 0),  # Transparent background
    ):
        """
        Initialize an image for text rendering.

        Args:
            image_size: Dimensions of the image (height, width) - NumPy convention
            mode: Image mode (RGB, RGBA, etc.)
            background: Background color with alpha channel

        Raises:
            ValueError: If image_size dimensions are not positive
        """
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("Image dimensions must be positive")

        if len(background) != 4:
            raise ValueError("Background color must be RGBA (4 values)")

        self.image_size = image_size  # Stored as (height, width)
        # PIL uses (width, height), so we reverse for Image.new
        self.image = Image.new(mode, (image_size[1], image_size[0]), color=background)
        self._draw = ImageDraw.Draw(self.image)
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}  # Cache for font objects

    @property
    def img_array(self) -> np.ndarray:
        """Convert the PIL Image to a numpy array."""
        return np.array(self.image)

    def save(self, filename: str) -> None:
        """Save the image to a file."""
        if not filename:
            raise ValueError("Filename cannot be empty")
        self.image.save(filename)

    def _fit_font_width(self, text: str, font: str, max_width: int) -> int:
        """
        Find the maximum font size where the text width is less than or equal to max_width.

        Args:
            text: The text to measure
            font: Path to the font file
            max_width: Maximum allowed width in pixels

        Returns:
            The maximum font size that fits within max_width

        Raises:
            ValueError: If text is empty or max_width is too small for any font size
        """
        if not text:
            return 1  # Default to minimum size for empty text

        if max_width <= 0:
            raise ValueError("Maximum width must be positive")

        font_size = 1
        text_width = self.get_text_dimensions(font, font_size, text)[0]
        while text_width < max_width:
            font_size += 1
            text_width = self.get_text_dimensions(font, font_size, text)[0]
        max_font_size = font_size - 1
        if max_font_size < 1:
            raise ValueError(f"Max width {max_width} is too small for any font size!")
        return max_font_size

    def _fit_font_height(self, text: str, font: str, max_height: int) -> int:
        """
        Find the maximum font size where the text height is less than or equal to max_height.

        Args:
            text: The text to measure
            font: Path to the font file
            max_height: Maximum allowed height in pixels

        Returns:
            The maximum font size that fits within max_height

        Raises:
            ValueError: If text is empty or max_height is too small for any font size
        """
        if not text:
            return 1  # Default to minimum size for empty text

        if max_height <= 0:
            raise ValueError("Maximum height must be positive")

        font_size = 1
        text_height = self.get_text_dimensions(font, font_size, text)[1]
        while text_height < max_height:
            font_size += 1
            text_height = self.get_text_dimensions(font, font_size, text)[1]
        max_font_size = font_size - 1
        if max_font_size < 1:
            raise ValueError(f"Max height {max_height} is too small for any font size!")
        return max_font_size

    def _get_font_size(
        self,
        text: str,
        font: str,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> int:
        """
        Get maximum font size for text to fit within given dimensions.

        Args:
            text: The text to fit
            font: Path to the font file
            max_width: Maximum allowed width in pixels
            max_height: Maximum allowed height in pixels

        Returns:
            The maximum font size that fits within constraints

        Raises:
            ValueError: If neither max_width nor max_height is provided, or text is empty
        """
        if not text:
            raise ValueError("Text cannot be empty")

        if max_width is None and max_height is None:
            raise ValueError("You need to pass max_width or max_height")

        if max_width is not None and max_width <= 0:
            raise ValueError("Maximum width must be positive")

        if max_height is not None and max_height <= 0:
            raise ValueError("Maximum height must be positive")

        width_font_size = self._fit_font_width(text, font, max_width) if max_width is not None else None
        height_font_size = self._fit_font_height(text, font, max_height) if max_height is not None else None

        sizes = [size for size in [width_font_size, height_font_size] if size is not None]
        if not sizes:
            raise ValueError("No valid font size could be calculated")

        return min(sizes)

    def _process_margin(self, margin: MarginType) -> tuple[int, int, int, int]:
        """
        Process the margin parameter into individual top, right, bottom, left values.

        Args:
            margin: A single int for all sides, or a tuple of 4 values for each side

        Returns:
            Tuple of (top, right, bottom, left) margin values

        Raises:
            ValueError: If margin tuple doesn't have exactly 4 values
        """
        if isinstance(margin, int):
            if margin < 0:
                raise ValueError("Margin cannot be negative")
            return margin, margin, margin, margin
        elif isinstance(margin, tuple) and len(margin) == 4:
            if any(m < 0 for m in margin):
                raise ValueError("Margin values cannot be negative")
            return margin
        else:
            raise ValueError("Margin must be an int or a tuple of 4 ints")

    def _convert_position(
        self, position: PositionType, margin_top: int, margin_left: int, available_width: int, available_height: int
    ) -> tuple[float, float]:
        """
        Convert a position from relative (0-1) to absolute pixels.

        Args:
            position: Position as (x, y) coordinates, either as pixels or relative (0-1)
            margin_top: Top margin in pixels
            margin_left: Left margin in pixels
            available_width: Available width considering margins
            available_height: Available height considering margins

        Returns:
            Position in absolute pixel coordinates (might still be float)
        """
        x_pos, y_pos = position

        # Convert relative position (0-1) to absolute pixels
        if isinstance(x_pos, float) and 0 <= x_pos <= 1:
            x_pos = margin_left + x_pos * available_width
        if isinstance(y_pos, float) and 0 <= y_pos <= 1:
            y_pos = margin_top + y_pos * available_height

        return x_pos, y_pos

    def _calculate_position(
        self,
        text_size: tuple[int, int],
        position: PositionType,
        anchor: AnchorPoint = AnchorPoint.TOP_LEFT,
        margin: MarginType = 0,
    ) -> tuple[int, int]:
        """
        Calculate the absolute position based on anchor point, relative positioning and margins.

        Args:
            text_size: Width and height of the text in pixels
            position: Either absolute coordinates (int) or relative to frame size (float 0-1)
            anchor: Which part of the text to anchor at the position
            margin: Margin in pixels (single value or [top, right, bottom, left])

        Returns:
            Absolute x, y coordinates for text placement

        Raises:
            ValueError: If position or margin values are invalid
        """
        if not isinstance(text_size, tuple) or len(text_size) != 2:
            raise ValueError("text_size must be a tuple of (width, height)")

        text_width, text_height = text_size

        # Process margins
        margin_top, margin_right, margin_bottom, margin_left = self._process_margin(margin)

        # Calculate available area considering margins
        available_width = self.image_size[1] - margin_left - margin_right
        available_height = self.image_size[0] - margin_top - margin_bottom

        # Convert relative position to absolute if needed
        x_pos, y_pos = self._convert_position(position, margin_top, margin_left, available_width, available_height)

        # Apply margin to absolute position when using 0,0 as starting point
        if x_pos == 0 and anchor in AnchorPoint.left_anchors():
            x_pos = margin_left
        if y_pos == 0 and anchor in AnchorPoint.top_anchors():
            y_pos = margin_top

        # Adjust position based on anchor point
        if anchor in AnchorPoint.center_anchors():
            x_pos -= text_width // 2
        elif anchor in AnchorPoint.right_anchors():
            x_pos -= text_width

        if anchor in AnchorPoint.middle_anchors():
            y_pos -= text_height // 2
        elif anchor in AnchorPoint.bottom_anchors():
            y_pos -= text_height

        return int(x_pos), int(y_pos)

    def write_text(
        self,
        text: str,
        font_filename: str,
        xy: PositionType,
        font_size: int | None = 11,
        font_border_size: int = 0,
        color: RGBColor = (0, 0, 0),
        max_width: int | None = None,
        max_height: int | None = None,
        anchor: AnchorPoint = AnchorPoint.TOP_LEFT,
        margin: MarginType = 0,
    ) -> tuple[int, int]:
        """
        Write text to the image with advanced positioning options.

        Args:
            text: Text to be written
            font_filename: Path to the font file
            xy: Position (x,y) either as absolute pixels (int) or relative to frame (float 0-1)
            font_size: Size of the font in points, or None to auto-calculate
            font_border_size: Size of border around text in pixels (0 for no border)
            color: RGB color of the text
            max_width: Maximum width for auto font sizing
            max_height: Maximum height for auto font sizing
            anchor: Which part of the text to anchor at the position
            margin: Margin in pixels (single value or [top, right, bottom, left])

        Returns:
            Dimensions of the rendered text (width, height)

        Raises:
            ValueError: If text is empty or font parameters are invalid
            OutOfBoundsError: If the text would be rendered outside the image bounds
        """
        if not text:
            raise ValueError("Text cannot be empty")

        if not font_filename:
            raise ValueError("Font filename cannot be empty")

        if font_size is not None and font_size <= 0:
            raise ValueError("Font size must be positive")

        if font_border_size < 0:
            raise ValueError("Font border size cannot be negative")

        if font_size is None and (max_width is None or max_height is None):
            raise ValueError("Must set either `font_size`, or both `max_width` and `max_height`!")
        elif font_size is None:
            font_size = self._get_font_size(text, font_filename, max_width, max_height)

        # Get or create the font object (with caching)
        font = self._get_font(font_filename, font_size)
        text_dimensions = self.get_text_dimensions(font_filename, font_size, text)

        # Calculate the position based on anchor point and margins
        x, y = self._calculate_position(text_dimensions, xy, anchor, margin)

        # Verify text will fit within bounds
        if x < 0 or y < 0 or x + text_dimensions[0] > self.image_size[1] or y + text_dimensions[1] > self.image_size[0]:
            raise OutOfBoundsError(f"Text with size {text_dimensions} at position ({x}, {y}) is out of bounds!")

        # Draw border if requested
        if font_border_size > 0:
            # Draw text border by drawing text in multiple positions around the main text
            for border_x in range(-font_border_size, font_border_size + 1):
                for border_y in range(-font_border_size, font_border_size + 1):
                    if border_x != 0 or border_y != 0:  # Skip the center position
                        self._draw.text((x + border_x, y + border_y), text, font=font, fill=(0, 0, 0))

        # Draw the main text on top
        self._draw.text((x, y), text, font=font, fill=color)
        return text_dimensions

    def _get_font(self, font_filename: str, font_size: int) -> ImageFont.FreeTypeFont:
        """
        Get a font object, using cache if available.

        Args:
            font_filename: Path to the font file
            font_size: Size of the font in points

        Returns:
            Font object for rendering text
        """
        key = (font_filename, font_size)
        if key not in self._font_cache:
            try:
                self._font_cache[key] = ImageFont.truetype(font_filename, font_size)
            except (OSError, IOError) as e:
                raise ValueError(f"Error loading font '{font_filename}': {str(e)}")
        return self._font_cache[key]

    def get_text_dimensions(self, font_filename: str, font_size: int, text: str) -> tuple[int, int]:
        """
        Return dimensions (width, height) of the rendered text.

        Args:
            font_filename: Path to the font file
            font_size: Size of the font in points
            text: Text to measure

        Returns:
            Tuple of (width, height) for the rendered text

        Raises:
            ValueError: If font parameters are invalid or text is empty
        """
        if not text:
            return (0, 0)  # Empty text has no dimensions

        if font_size <= 0:
            raise ValueError("Font size must be positive")

        font = self._get_font(font_filename, font_size)
        try:
            bbox = font.getbbox(text)
            if bbox is None:
                return (0, 0)  # Handle case where getbbox returns None
            return bbox[2:] if len(bbox) >= 4 else (0, 0)
        except Exception as e:
            raise ValueError(f"Error measuring text: {str(e)}")

    def _get_font_baseline_offset(
        self, base_font_filename: str, base_font_size: int, highlight_font_filename: str, highlight_font_size: int
    ) -> int:
        """
        Calculate the vertical offset needed to align baselines of different fonts and sizes.

        Args:
            base_font_filename: Path to the base font file
            base_font_size: Font size of normal text
            highlight_font_filename: Path to the highlight font file
            highlight_font_size: Font size of highlighted text

        Returns:
            Vertical offset in pixels to align highlighted text baseline with normal text baseline
        """
        base_font = self._get_font(base_font_filename, base_font_size)
        highlight_font = self._get_font(highlight_font_filename, highlight_font_size)

        # Use a reference character to get baseline metrics
        # We use 'A' as it's a good reference for ascender height
        ref_char = "A"

        # Get bounding boxes for the reference character
        base_bbox = base_font.getbbox(ref_char)
        highlight_bbox = highlight_font.getbbox(ref_char)

        if base_bbox is None or highlight_bbox is None:
            return 0  # Fallback if bbox calculation fails

        # The baseline offset is the difference in the top of the bounding box
        # since getbbox returns (left, top, right, bottom) where top is negative for ascenders
        base_ascent = -base_bbox[1]  # Distance from baseline to top of character
        highlight_ascent = -highlight_bbox[1]  # Distance from baseline to top of character

        # Calculate the offset needed to align baselines
        # If highlighted text has a larger ascent, we need to move it down
        baseline_offset = highlight_ascent - base_ascent

        return baseline_offset

    def _split_lines_by_width(
        self,
        text: str,
        font_filename: str,
        font_size: int,
        box_width: int,
    ) -> list[str]:
        """
        Split the text into lines that fit within the specified width.

        Args:
            text: Text to split into lines
            font_filename: Path to the font file
            font_size: Size of the font in points
            box_width: Maximum width for each line in pixels

        Returns:
            List of text lines that fit within box_width

        Raises:
            ValueError: If font parameters are invalid or box_width is too small
        """
        if not text:
            return []  # Empty text produces no lines

        if box_width <= 0:
            raise ValueError("Box width must be positive")

        if font_size <= 0:
            raise ValueError("Font size must be positive")

        words = text.split()
        if not words:
            return []  # No words means no lines

        # Handle single-word case efficiently
        if len(words) == 1:
            return [text]

        split_lines: list[list[str]] = []
        current_line: list[str] = []

        for word in words:
            # If current line is empty and this word is too long for box_width,
            # we'll have to split the word itself (not implemented)
            if not current_line and self.get_text_dimensions(font_filename, font_size, word)[0] > box_width:
                # Just add the word anyway, it'll overflow but we can't do better without splitting words
                split_lines.append([word])
                continue

            # Try adding the word to current line
            new_line = " ".join(current_line + [word]) if current_line else word
            size = self.get_text_dimensions(font_filename, font_size, new_line)
            if size[0] <= box_width:
                current_line.append(word)
            else:
                # This word doesn't fit, start new line
                if current_line:  # Only if we have a current line to add
                    split_lines.append(current_line)
                current_line = [word]

        # Add the last line if it has content
        if current_line:
            split_lines.append(current_line)

        # Join the words in each line with spaces
        lines = [" ".join(line) for line in split_lines]
        return lines

    def write_text_box(
        self,
        text: str,
        font_filename: str,
        xy: PositionType,
        box_width: int | float | None = None,
        font_size: int = 11,
        font_border_size: int = 0,
        text_color: RGBColor = (0, 0, 0),
        background_color: RGBAColor | None = None,
        background_padding: int = 0,
        place: TextAlign = TextAlign.LEFT,
        anchor: AnchorPoint = AnchorPoint.TOP_LEFT,
        margin: MarginType = 0,
        words: list[str] | None = None,
        highlight_word_index: int | None = None,
        highlight_color: RGBColor | None = None,
        highlight_size_multiplier: float = DEFAULT_HIGHLIGHT_SIZE_MULTIPLIER,
        highlight_bold_font: str | None = None,
    ) -> tuple[int, int]:
        """
        Write text in a box with advanced positioning and alignment options.

        Args:
            text: Text to be written inside the box
            font_filename: Path to the font file
            xy: Position (x,y) either as absolute pixels (int) or relative to frame (float 0-1)
            box_width: Width of the box in pixels (int) or relative to frame width (float 0-1)
            font_size: Font size in points
            font_border_size: Size of border around text in pixels (0 for no border)
            text_color: RGB color of the text
            background_color: If set, adds background color to the text box. Expects RGBA values.
            background_padding: Number of padding pixels to add when adding text background color
            place: Text alignment within the box (TextAlign.LEFT, TextAlign.RIGHT, TextAlign.CENTER)
            anchor: Which part of the text box to anchor at the position
            margin: Margin in pixels (single value or [top, right, bottom, left])
            words: All words occuring in text, helpful for highlighting.
            highlight_word_index: Index of word to highlight (0-based, None to disable highlighting)
            highlight_color: RGB color for the highlighted word (defaults to text_color if None)
            highlight_size_multiplier: Font size multiplier for highlighted word
            highlight_bold_font: Path to bold font file for highlighted word (defaults to font_filename if None)

        Returns:
            Coordinates of the lower-right corner of the written text box (x, y)

        Raises:
            ValueError: If text is empty or parameters are invalid
            OutOfBoundsError: If text box would be outside image bounds
        """
        if not text:
            raise ValueError("Text cannot be empty")

        if not font_filename:
            raise ValueError("Font filename cannot be empty")

        if font_size <= 0:
            raise ValueError("Font size must be positive")

        if background_padding < 0:
            raise ValueError("Background padding cannot be negative")

        if font_border_size < 0:
            raise ValueError("Font border size cannot be negative")

        # Validate highlighting parameters
        if highlight_word_index is not None:
            if not words:
                words = text.split()
            if highlight_word_index < 0 or highlight_word_index >= len(words):
                raise ValueError(
                    f"highlight_word_index {highlight_word_index} out of range for text with {len(words)} words"
                )

        if highlight_size_multiplier <= 0:
            raise ValueError("highlight_size_multiplier must be positive")

        # Set default highlight color if not provided
        if highlight_word_index is not None and highlight_color is None:
            highlight_color = text_color

        # Process margins to determine available area
        margin_top, margin_right, margin_bottom, margin_left = self._process_margin(margin)
        available_width = self.image_size[1] - margin_left - margin_right
        available_height = self.image_size[0] - margin_top - margin_bottom

        # Handle relative box width
        if box_width is None:
            box_width = available_width
        elif isinstance(box_width, float) and 0 < box_width <= 1:
            box_width = int(available_width * box_width)
        elif isinstance(box_width, int) and box_width <= 0:
            raise ValueError("Box width must be positive")

        # Calculate initial position based on margin and anchor before splitting text
        x_pos, y_pos = self._convert_position(xy, margin_top, margin_left, available_width, available_height)

        # Split text into lines that fit within box_width
        lines = self._split_lines_by_width(text, font_filename, font_size, int(box_width))

        # Calculate total height of all lines
        lines_height = sum([self.get_text_dimensions(font_filename, font_size, line)[1] for line in lines])
        if lines_height == 0:
            # If we have no valid lines or zero height, return the position
            return (int(x_pos), int(y_pos))

        # Final position calculation based on anchor point
        if anchor in AnchorPoint.center_anchors():
            x_pos -= box_width // 2
        elif anchor in AnchorPoint.right_anchors():
            x_pos -= box_width

        if anchor in AnchorPoint.middle_anchors():
            y_pos -= lines_height // 2
        elif anchor in AnchorPoint.bottom_anchors():
            y_pos -= lines_height

        # Verify box will fit within bounds
        if (
            x_pos < 0
            or y_pos < 0
            or x_pos + box_width > self.image_size[1]
            or y_pos + lines_height > self.image_size[0]
        ):
            raise OutOfBoundsError(
                f"Text box with size ({box_width}x{lines_height}) at position ({x_pos}, {y_pos}) is out of bounds!"
            )

        # Write lines
        current_text_height = y_pos
        word_index_offset = 0  # Track global word index across lines
        for line in lines:
            line_dimensions = self.get_text_dimensions(font_filename, font_size, line)

            # Calculate horizontal position based on alignment
            if place == TextAlign.LEFT:
                x_left = x_pos
            elif place == TextAlign.RIGHT:
                x_left = x_pos + box_width - line_dimensions[0]
            elif place == TextAlign.CENTER:
                x_left = int(x_pos + ((box_width - line_dimensions[0]) / 2))
            else:
                valid_places = [e.value for e in TextAlign]
                raise ValueError(f"Place '{place}' is not supported. Must be one of: {', '.join(valid_places)}")

            # Check if highlighting is needed for this line
            if highlight_word_index is not None:
                line_words = line.split()
                line_start_word_index = word_index_offset
                line_end_word_index = word_index_offset + len(line_words) - 1

                # Check if the highlighted word is in this line
                if line_start_word_index <= highlight_word_index <= line_end_word_index:
                    self._write_line_with_highlight(
                        line=line,
                        font_filename=font_filename,
                        font_size=font_size,
                        font_border_size=font_border_size,
                        text_color=text_color,
                        highlight_color=highlight_color or (255, 255, 255),
                        highlight_size_multiplier=highlight_size_multiplier,
                        highlight_word_local_index=highlight_word_index - line_start_word_index,
                        highlight_bold_font=highlight_bold_font,
                        x_left=int(x_left),
                        y_top=int(current_text_height),
                    )
                else:
                    # Write normal line without highlighting
                    self.write_text(
                        text=line,
                        font_filename=font_filename,
                        xy=(x_left, current_text_height),
                        font_size=font_size,
                        font_border_size=font_border_size,
                        color=text_color,
                    )

                word_index_offset += len(line_words)
            else:
                # Write normal line without highlighting
                self.write_text(
                    text=line,
                    font_filename=font_filename,
                    xy=(x_left, current_text_height),
                    font_size=font_size,
                    font_border_size=font_border_size,
                    color=text_color,
                )

            # Increment vertical position for next line
            current_text_height += line_dimensions[1]

        # Add background color for the text if specified
        if background_color is not None:
            if len(background_color) != 4:
                raise ValueError(f"Text background color {background_color} must be RGBA (4 values)!")

            img = self.img_array

            # Find bounding rectangle for written text
            # Skip if the box is empty
            if y_pos >= current_text_height or x_pos >= x_pos + box_width:
                return (int(x_pos + box_width), int(current_text_height))

            # Get the slice of the image containing the text box
            box_slice = img[int(y_pos) : int(current_text_height), int(x_pos) : int(x_pos + box_width)]
            if box_slice.size == 0:  # Empty slice
                return (int(x_pos + box_width), int(current_text_height))

            # Create mask of non-zero pixels (text)
            text_mask = np.any(box_slice != 0, axis=2).astype(np.uint8)
            if not isinstance(text_mask, np.ndarray):
                raise TypeError(f"The returned text mask is of type {type(text_mask)}, but it should be numpy array!")

            # If no text pixels found, return without background
            if not np.any(text_mask):
                return (int(x_pos + box_width), int(current_text_height))

            # Find the smallest rectangle containing text
            try:
                xmin, xmax, ymin, ymax = self._find_smallest_bounding_rect(text_mask)
            except ValueError:
                # _find_smallest_bounding_rect raises ValueError for empty mask
                xmin, xmax, ymin, ymax = 0, box_slice.shape[1] - 1, 0, box_slice.shape[0] - 1

            # Get global bounding box position
            xmin = int(xmin + x_pos - background_padding)
            xmax = int(xmax + x_pos + background_padding)
            ymin = int(ymin + y_pos - background_padding)
            ymax = int(ymax + y_pos + background_padding)

            # Make sure we are inside image bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(xmax, self.image_size[1])
            ymax = min(ymax, self.image_size[0])

            # Skip if bounding box is invalid
            if xmin >= xmax or ymin >= ymax:
                return (int(x_pos + box_width), int(current_text_height))

            # Slice the bounding box and find text mask
            bbox_slice = img[ymin:ymax, xmin:xmax]
            if bbox_slice.size == 0:  # Empty slice
                return (int(x_pos + box_width), int(current_text_height))

            bbox_text_mask = np.any(bbox_slice != 0, axis=2).astype(np.uint8)

            # Add background color outside of text
            bbox_slice[~bbox_text_mask.astype(bool)] = background_color

            # Handle semi-transparent pixels for smooth text blending
            text_slice = bbox_slice[bbox_text_mask.astype(bool)]
            if text_slice.size > 0:
                text_background = text_slice[:, :3] * (np.expand_dims(text_slice[:, -1], axis=1) / 255)
                color_background = (1 - (np.expand_dims(text_slice[:, -1], axis=1) / 255)) * background_color
                faded_background = text_background[:, :3] + color_background[:, :3]
                text_slice[:, :3] = faded_background
                text_slice[:, -1] = 255  # Full opacity
                bbox_slice[bbox_text_mask.astype(bool)] = text_slice

            # Update the image with the background color
            self.image = Image.fromarray(img)

        return (int(x_pos + box_width), int(current_text_height))

    def _write_line_with_highlight(
        self,
        line: str,
        font_filename: str,
        font_size: int,
        font_border_size: int,
        text_color: RGBColor,
        highlight_color: RGBColor,
        highlight_size_multiplier: float,
        highlight_word_local_index: int,
        highlight_bold_font: str | None,
        x_left: int,
        y_top: int,
    ) -> None:
        """
        Write a line of text with one word highlighted using word-by-word rendering with baseline alignment.

        Args:
            line: The text line to render
            font_filename: Path to the font file
            font_size: Base font size in points
            font_border_size: Size of border around text in pixels (0 for no border)
            text_color: RGB color for normal text
            highlight_color: RGB color for highlighted word
            highlight_size_multiplier: Font size multiplier for highlighted word
            highlight_word_local_index: Index of word to highlight within this line (0-based)
            highlight_bold_font: Path to bold font file for highlighted word (defaults to font_filename if None)
            x_left: Left x position for the line
            y_top: Top y position for the line
        """
        # Split line into words
        words = line.split()
        if highlight_word_local_index >= len(words):
            return  # Safety check

        # Calculate highlighted font size and determine font files
        highlight_font_size = int(font_size * highlight_size_multiplier)
        highlight_font_file = highlight_bold_font if highlight_bold_font is not None else font_filename

        # Calculate baseline offset for highlighted words (using the appropriate font files)
        baseline_offset = self._get_font_baseline_offset(
            font_filename, font_size, highlight_font_file, highlight_font_size
        )

        # Render words one by one with proper spacing
        current_x = x_left

        for i, word in enumerate(words):
            # Determine if this is the highlighted word
            is_highlighted = i == highlight_word_local_index

            # Choose font file, size, and color based on highlighting
            word_font_file = highlight_font_file if is_highlighted else font_filename
            word_font_size = highlight_font_size if is_highlighted else font_size
            word_color = highlight_color if is_highlighted else text_color

            # Calculate y position with baseline alignment
            word_y = y_top
            if is_highlighted:
                word_y += baseline_offset

            # Render the word
            self.write_text(
                text=word,
                font_filename=word_font_file,
                xy=(current_x, word_y),
                font_size=word_font_size,
                font_border_size=font_border_size,
                color=word_color,
            )

            # Calculate the width of this word for spacing
            word_width = self.get_text_dimensions(word_font_file, word_font_size, word)[0]

            # Update current_x for next word (add word width plus space)
            current_x += word_width

            # Add space between words (except after the last word)
            if i < len(words) - 1:
                space_width = self.get_text_dimensions(font_filename, font_size, " ")[0]
                current_x += space_width

    def _find_smallest_bounding_rect(self, mask: np.ndarray) -> tuple[int, int, int, int]:
        """
        Find the smallest bounding rectangle containing non-zero values in the mask.

        Args:
            mask: 2D numpy array with non-zero values representing pixels of interest

        Returns:
            Tuple of (xmin, xmax, ymin, ymax) coordinates

        Raises:
            ValueError: If mask is empty or has no non-zero values
        """
        if mask.size == 0:
            raise ValueError("Mask is empty")

        # Check if mask has any non-zero values
        if not np.any(mask):
            raise ValueError("Mask has no non-zero values")

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        # Find indices of first and last True values
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        # Handle empty results
        if len(row_indices) == 0 or len(col_indices) == 0:
            raise ValueError("No bounding rectangle found")

        ymin, ymax = row_indices[[0, -1]]
        xmin, xmax = col_indices[[0, -1]]

        return xmin, xmax, ymin, ymax


class TranscriptionOverlay:
    def __init__(
        self,
        font_filename: str,
        font_size: int = 40,
        font_border_size: int = 2,
        text_color: RGBColor = (255, 235, 59),
        background_color: RGBAColor | None = (0, 0, 0, 100),
        background_padding: int = 15,
        position: PositionType = (0.5, 0.7),
        box_width: int | float = 0.6,
        text_align: TextAlign = TextAlign.CENTER,
        anchor: AnchorPoint = AnchorPoint.CENTER,
        margin: MarginType = 20,
        highlight_color: RGBColor = (76, 175, 80),
        highlight_size_multiplier: float = 1.2,
        highlight_bold_font: str | None = None,
    ):
        """
        Initialize TranscriptionOverlay effect.

        Args:
            font_filename: Path to font file for text rendering
            font_size: Base font size for text
            text_color: RGB color for normal text
            font_border_size: Size of border around text in pixels (0 for no border)
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
        self.font_filename = font_filename
        self.font_size = font_size
        self.text_color = text_color
        self.font_border_size = font_border_size
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

    def _get_active_segment(self, transcription: Transcription, timestamp: float) -> TranscriptionSegment | None:
        """Get the transcription segment active at the given timestamp."""
        for segment in transcription.segments:
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
        img_text = ImageText(image_size=(height, width), background=(0, 0, 0, 0))

        # Write text with highlighting
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

        # Cache the overlay
        self._overlay_cache[cache_key] = overlay_image

        return overlay_image

    def apply(self, video: Video, transcription: Transcription) -> Video:
        """Apply transcription overlay to video frames."""
        print("Applying transcription overlay...")

        new_frames = []

        for frame_index, frame in enumerate(tqdm(video.frames)):
            # Calculate timestamp for this frame
            timestamp = frame_index / video.fps

            # Get active segment at this timestamp
            active_segment = self._get_active_segment(transcription, timestamp)

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
