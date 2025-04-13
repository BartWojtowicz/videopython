from enum import Enum
from typing import TypeAlias, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from videopython.base.exceptions import OutOfBoundsError

# Type aliases for clarity
MarginType: TypeAlias = Union[int, tuple[int, int, int, int]]
RGBColor: TypeAlias = tuple[int, int, int]
RGBAColor: TypeAlias = tuple[int, int, int, int]
PositionType: TypeAlias = Union[tuple[int, int], tuple[float, float]]


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
        image_size: tuple[int, int] = (1080, 1920),  # (width, height)
        mode: str = "RGBA",
        background: RGBAColor = (0, 0, 0, 0),  # Transparent background
    ):
        """
        Initialize an image for text rendering.

        Args:
            image_size: Dimensions of the image (width, height)
            mode: Image mode (RGB, RGBA, etc.)
            background: Background color with alpha channel

        Raises:
            ValueError: If image_size dimensions are not positive
        """
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("Image dimensions must be positive")

        if len(background) != 4:
            raise ValueError("Background color must be RGBA (4 values)")

        self.image_size = image_size
        self.image = Image.new(mode, image_size, color=background)
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
        available_width = self.image_size[0] - margin_left - margin_right
        available_height = self.image_size[1] - margin_top - margin_bottom

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
        if x < 0 or y < 0 or x + text_dimensions[0] > self.image_size[0] or y + text_dimensions[1] > self.image_size[1]:
            raise OutOfBoundsError(f"Text with size {text_dimensions} at position ({x}, {y}) is out of bounds!")

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
        box_width: Union[int, float] | None = None,
        font_size: int = 11,
        text_color: RGBColor = (0, 0, 0),
        background_color: RGBAColor | None = None,
        background_padding: int = 0,
        place: TextAlign = TextAlign.LEFT,
        anchor: AnchorPoint = AnchorPoint.TOP_LEFT,
        margin: MarginType = 0,
    ) -> tuple[int, int]:
        """
        Write text in a box with advanced positioning and alignment options.

        Args:
            text: Text to be written inside the box
            font_filename: Path to the font file
            xy: Position (x,y) either as absolute pixels (int) or relative to frame (float 0-1)
            box_width: Width of the box in pixels (int) or relative to frame width (float 0-1)
            font_size: Font size in points
            text_color: RGB color of the text
            background_color: If set, adds background color to the text box. Expects RGBA values.
            background_padding: Number of padding pixels to add when adding text background color
            place: Text alignment within the box (TextAlign.LEFT, TextAlign.RIGHT, TextAlign.CENTER)
            anchor: Which part of the text box to anchor at the position
            margin: Margin in pixels (single value or [top, right, bottom, left])

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

        # Process margins to determine available area
        margin_top, margin_right, margin_bottom, margin_left = self._process_margin(margin)
        available_width = self.image_size[0] - margin_left - margin_right
        available_height = self.image_size[1] - margin_top - margin_bottom

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
            or x_pos + box_width > self.image_size[0]
            or y_pos + lines_height > self.image_size[1]
        ):
            raise OutOfBoundsError(
                f"Text box with size ({box_width}x{lines_height}) at position ({x_pos}, {y_pos}) is out of bounds!"
            )

        # Write lines
        current_text_height = y_pos
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

            # Write the line
            self.write_text(
                text=line,
                font_filename=font_filename,
                xy=(x_left, current_text_height),
                font_size=font_size,
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
            except Exception:
                # If bounding rectangle calculation fails, use the whole box
                xmin, xmax, ymin, ymax = 0, box_slice.shape[1] - 1, 0, box_slice.shape[0] - 1

            # Get global bounding box position
            xmin = int(xmin + x_pos - background_padding)
            xmax = int(xmax + x_pos + background_padding)
            ymin = int(ymin + y_pos - background_padding)
            ymax = int(ymax + y_pos + background_padding)

            # Make sure we are inside image bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(xmax, self.image_size[0])
            ymax = min(ymax, self.image_size[1])

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
