from enum import Enum
from typing import Literal, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from videopython.base.exceptions import OutOfBoundsError


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


class ImageText:
    def __init__(
        self,
        image_size: tuple[int, int] = (1080, 1920),  # (width, height)
        mode: str = "RGBA",
        background: tuple[int, int, int, int] = (0, 0, 0, 0),  # Transparent background
    ):
        self.image_size = image_size
        self.image = Image.new(mode, image_size, color=background)
        self._draw = ImageDraw.Draw(self.image)

    @property
    def img_array(self) -> np.ndarray:
        return np.array(self.image)

    def save(self, filename: str) -> None:
        self.image.save(filename)

    def _fit_font_width(self, text: str, font: str, max_width: int) -> int:
        """Find the maximum font size where the text width is less than or equal to max_width."""
        font_size = 1
        text_width = self.get_text_size(font, font_size, text)[0]
        while text_width < max_width:
            font_size += 1
            text_width = self.get_text_size(font, font_size, text)[0]
        max_font_size = font_size - 1
        if max_font_size < 1:
            raise ValueError(f"Max height {max_width} is too small for any font size!")
        return max_font_size

    def _fit_font_height(self, text: str, font: str, max_height: int) -> int:
        """Find the maximum font size where the text height is less than or equal to max_height."""
        font_size = 1
        text_height = self.get_text_size(font, font_size, text)[1]
        while text_height < max_height:
            font_size += 1
            text_height = self.get_text_size(font, font_size, text)[1]
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
        """Get maximum font size for `text` to fill in the `max_width` and `max_height`."""
        if max_width is None and max_height is None:
            raise ValueError("You need to pass max_width or max_height")
        if max_width is not None:
            width_font_size = self._fit_font_width(text, font, max_width)
        else:
            width_font_size = None
        if max_height is not None:
            height_font_size = self._fit_font_height(text, font, max_height)
        else:
            height_font_size = None
        return min([size for size in [width_font_size, height_font_size] if size is not None])

    def _calculate_position(
        self,
        text_size: tuple[int, int],
        position: Union[tuple[int, int], tuple[float, float]],
        anchor: AnchorPoint = AnchorPoint.TOP_LEFT,
        margin: Union[int, tuple[int, int, int, int]] = 0,
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
        """
        text_width, text_height = text_size

        # Process margins
        if isinstance(margin, int):
            margin_top = margin_right = margin_bottom = margin_left = margin
        else:
            margin_top, margin_right, margin_bottom, margin_left = margin

        # Calculate available area considering margins
        available_width = self.image_size[0] - margin_left - margin_right
        available_height = self.image_size[1] - margin_top - margin_bottom

        # Convert relative position to absolute if needed
        x_pos, y_pos = position
        if isinstance(x_pos, float) and 0 <= x_pos <= 1:
            x_pos = int(margin_left + x_pos * available_width)
        if isinstance(y_pos, float) and 0 <= y_pos <= 1:
            y_pos = int(margin_top + y_pos * available_height)

        # Apply margin to absolute position when using 0,0 as starting point
        if x_pos == 0 and anchor in (AnchorPoint.TOP_LEFT, AnchorPoint.CENTER_LEFT, AnchorPoint.BOTTOM_LEFT):
            x_pos = margin_left
        if y_pos == 0 and anchor in (AnchorPoint.TOP_LEFT, AnchorPoint.TOP_CENTER, AnchorPoint.TOP_RIGHT):
            y_pos = margin_top

        # Adjust position based on anchor point
        if anchor in (AnchorPoint.TOP_CENTER, AnchorPoint.CENTER, AnchorPoint.BOTTOM_CENTER):
            x_pos -= text_width // 2
        elif anchor in (AnchorPoint.TOP_RIGHT, AnchorPoint.CENTER_RIGHT, AnchorPoint.BOTTOM_RIGHT):
            x_pos -= text_width

        if anchor in (AnchorPoint.CENTER_LEFT, AnchorPoint.CENTER, AnchorPoint.CENTER_RIGHT):
            y_pos -= text_height // 2
        elif anchor in (AnchorPoint.BOTTOM_LEFT, AnchorPoint.BOTTOM_CENTER, AnchorPoint.BOTTOM_RIGHT):
            y_pos -= text_height

        return int(x_pos), int(y_pos)

    def write_text(
        self,
        text: str,
        font_filename: str,
        xy: Union[tuple[int, int], tuple[float, float]],
        font_size: int | None = 11,
        color: tuple[int, int, int] = (0, 0, 0),
        max_width: int | None = None,
        max_height: int | None = None,
        anchor: AnchorPoint = AnchorPoint.TOP_LEFT,
        margin: Union[int, tuple[int, int, int, int]] = 0,
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
            Actual size of the rendered text (width, height)
        """
        if font_size is None and (max_width is None or max_height is None):
            raise ValueError("Must set either `font_size`, or both `max_width` and `max_height`!")
        elif font_size is None:
            font_size = self._get_font_size(text, font_filename, max_width, max_height)

        font = ImageFont.truetype(font_filename, font_size)
        text_size = self.get_text_size(font_filename, font_size, text)

        # Calculate the position based on anchor point and margins
        x, y = self._calculate_position(text_size, xy, anchor, margin)

        # Verify text will fit within bounds
        if x < 0 or y < 0 or x + text_size[0] > self.image_size[0] or y + text_size[1] > self.image_size[1]:
            raise OutOfBoundsError(f"Text with size {text_size} at position ({x}, {y}) is out of bounds!")

        self._draw.text((x, y), text, font=font, fill=color)
        return text_size

    def get_text_size(self, font_filename: str, font_size: int, text: str) -> tuple[int, int]:
        """Return bounding box size of the rendered `text` with `font_filename` and `font_size`."""
        font = ImageFont.truetype(font_filename, font_size)
        return font.getbbox(text)[2:]

    def _split_lines_by_width(
        self,
        text: str,
        font_filename: str,
        font_size: int,
        box_width: int,
    ) -> list[str]:
        """Split the `text` into lines of maximum `box_width`."""
        words = text.split()
        split_lines: list[list[str]] = []
        current_line: list[str] = []
        for word in words:
            new_line = " ".join(current_line + [word])
            size = self.get_text_size(font_filename, font_size, new_line)
            if size[0] <= box_width:
                current_line.append(word)
            else:
                split_lines.append(current_line)
                current_line = [word]
        if current_line:
            split_lines.append(current_line)
        lines = [" ".join(line) for line in split_lines]
        return lines

    def write_text_box(
        self,
        text: str,
        font_filename: str,
        xy: Union[tuple[int, int], tuple[float, float]],
        box_width: Optional[Union[int, float]] = None,
        font_size: int = 11,
        text_color: tuple[int, int, int] = (0, 0, 0),
        background_color: None | tuple[int, int, int, int] = None,
        background_padding: int = 0,
        place: Literal["left", "right", "center"] = "left",
        anchor: AnchorPoint = AnchorPoint.TOP_LEFT,
        margin: Union[int, tuple[int, int, int, int]] = 0,
    ) -> tuple[int, int]:
        """Write text in box with advanced positioning options.

        Args:
            text: Text to be written inside the box.
            font_filename: Path to the font file.
            xy: Position (x,y) either as absolute pixels (int) or relative to frame (float 0-1)
            box_width: Width of the box in pixels (int) or relative to frame width (float 0-1)
            font_size: Font size.
            text_color: RGB color of the text.
            background_color: If set, adds background color to the text box. Expects RGBA values.
            background_padding: Number of padding pixels to add when adding text background color.
            place: Strategy for justifying the text inside the container box. Defaults to "left".
            anchor: Which part of the text box to anchor at the position
            margin: Margin in pixels (single value or [top, right, bottom, left])

        Returns:
            Lower-right corner of the written text box.
        """
        # Process margins to determine available area
        if isinstance(margin, int):
            margin_top = margin_right = margin_bottom = margin_left = margin
        else:
            margin_top, margin_right, margin_bottom, margin_left = margin

        available_width = self.image_size[0] - margin_left - margin_right

        # Handle relative box width
        if box_width is None:
            box_width = available_width
        elif isinstance(box_width, float) and 0 < box_width <= 1:
            box_width = int(available_width * box_width)

        # Calculate initial position based on anchor and margins before splitting text
        # We'll need to adjust later based on actual text box dimensions
        x_pos, y_pos = xy
        if isinstance(x_pos, float) and 0 <= x_pos <= 1:
            x_pos = int(margin_left + x_pos * available_width)
        if isinstance(y_pos, float) and 0 <= y_pos <= 1:
            y_pos = int(margin_top + y_pos * (self.image_size[1] - margin_top - margin_bottom))

        # Split text into lines
        lines = self._split_lines_by_width(text, font_filename, font_size, int(box_width))
        lines_height = sum([self.get_text_size(font_filename, font_size, line)[1] for line in lines])

        # Final position calculation based on anchor point
        if anchor in (AnchorPoint.TOP_CENTER, AnchorPoint.CENTER, AnchorPoint.BOTTOM_CENTER):
            x_pos -= box_width // 2
        elif anchor in (AnchorPoint.TOP_RIGHT, AnchorPoint.CENTER_RIGHT, AnchorPoint.BOTTOM_RIGHT):
            x_pos -= box_width

        if anchor in (AnchorPoint.CENTER_LEFT, AnchorPoint.CENTER, AnchorPoint.CENTER_RIGHT):
            y_pos -= lines_height // 2
        elif anchor in (AnchorPoint.BOTTOM_LEFT, AnchorPoint.BOTTOM_CENTER, AnchorPoint.BOTTOM_RIGHT):
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
            line_size = self.get_text_size(font_filename, font_size, line)
            # Write line text into the image based on horizontal alignment
            if place == "left":
                self.write_text(
                    text=line,
                    font_filename=font_filename,
                    xy=(x_pos, current_text_height),
                    font_size=font_size,
                    color=text_color,
                )
            elif place == "right":
                x_left = x_pos + box_width - line_size[0]
                self.write_text(
                    text=line,
                    font_filename=font_filename,
                    xy=(x_left, current_text_height),
                    font_size=font_size,
                    color=text_color,
                )
            elif place == "center":
                x_left = int(x_pos + ((box_width - line_size[0]) / 2))
                self.write_text(
                    text=line,
                    font_filename=font_filename,
                    xy=(x_left, current_text_height),
                    font_size=font_size,
                    color=text_color,
                )
            else:
                raise ValueError(f"Place {place} is not supported. Use one of: `left`, `right` or `center`!")
            # Increment text height
            current_text_height += line_size[1]

        # Add background color for the text if set
        if background_color is not None:
            if len(background_color) != 4:
                raise ValueError(f"Text background color {background_color} must be RGBA!")
            img = self.img_array
            # Find bounding rectangle for written text
            box_slice = img[int(y_pos):int(current_text_height), int(x_pos):int(x_pos + box_width)]
            text_mask = np.any(box_slice != 0, axis=2).astype(np.uint8)
            if not isinstance(text_mask, np.ndarray):
                raise TypeError(f"The returned text mask is of type {type(text_mask)}, but it should be numpy array!")
            xmin, xmax, ymin, ymax = self._find_smallest_bounding_rect(text_mask)
            # Get global bounding box position
            xmin = int(xmin + x_pos - background_padding)
            xmax = int(xmax + x_pos + background_padding)
            ymin = int(ymin + y_pos - background_padding)
            ymax = int(ymax + y_pos + background_padding)
            # Make sure we are inside image, cut to image if not
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(xmax, self.image_size[0])
            ymax = min(ymax, self.image_size[1])
            # Slice the bounding box and find text mask
            bbox_slice = img[ymin:ymax, xmin:xmax]
            bbox_text_mask = np.any(bbox_slice != 0, axis=2).astype(np.uint8)
            # Add background color outside of text
            bbox_slice[~bbox_text_mask.astype(bool)] = background_color
            # Blur nicely with semi-transparent pixels from the font
            text_slice = bbox_slice[bbox_text_mask.astype(bool)]
            text_background = text_slice[:, :3] * (np.expand_dims(text_slice[:, -1], axis=1) / 255)
            color_background = (1 - (np.expand_dims(text_slice[:, -1], axis=1) / 255)) * background_color
            faded_background = text_background[:, :3] + color_background[:, :3]
            text_slice[:, :3] = faded_background
            text_slice[:, -1] = 255
            bbox_slice[bbox_text_mask.astype(bool)] = text_slice
            # Set image with the background color
            self.image = Image.fromarray(img)

        return (int(x_pos + box_width), int(current_text_height))

    def _find_smallest_bounding_rect(self, mask: np.ndarray) -> tuple[int, int, int, int]:
        """Find the smallest bounding rectangle for the mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax
