from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from videopython.exceptions import OutOfBoundsError


class ImageText(object):
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
    def img(self) -> np.ndarray:
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
        if max_height is not None:
            height_font_size = self._fit_font_height(text, font, max_height)
        return min([size for size in [width_font_size, height_font_size] if size is not None])

    def write_text(
        self,
        text: str,
        font_filename: str,
        xy: tuple[int, int],
        font_size: int | None = 11,
        color: tuple[int, int, int] = (0, 0, 0),
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> tuple[int, int]:
        x, y = xy
        if font_size is None and (max_width is None or max_height is None):
            raise ValueError(f"Must set either `font_size`, or both `max_width` and `max_height`!")
        elif font_size is None:
            font_size = self._get_font_size(text, font_filename, max_width, max_height)
        text_size = self.get_text_size(font_filename, font_size, text)
        if (text_size[0] + x > self.image_size[0]) or (text_size[1] + y > self.image_size[1]):
            raise OutOfBoundsError(f"Font size `{font_size}` is too big, text won't fit!")
        font = ImageFont.truetype(font_filename, font_size)
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
        xy: tuple[int, int],
        box_width: int,
        font_size: int = 11,
        color: tuple[int, int, int] = (0, 0, 0),
        place: Literal["left", "right", "center"] = "left",
    ) -> tuple[int, int]:
        x, y = xy
        if x + box_width > self.image_size[0]:
            raise OutOfBoundsError(f"Box width {box_width} is too big for the image width {self.image_size[0]}!")
        lines = self._split_lines_by_width(text, font_filename, font_size, box_width)
        lines_height = sum([self.get_text_size(font_filename, font_size, line)[1] for line in lines])
        if y + lines_height > self.image_size[1]:
            available_space = self.image_size[1] - y
            raise OutOfBoundsError(f"Text height {lines_height} is too big for the available space {available_space}!")
        # Write lines
        current_text_height = y
        for line in lines:
            line_size = self.get_text_size(font_filename, font_size, line)
            # Write line text into the image
            if place == "left":
                self.write_text(
                    text=line,
                    font_filename=font_filename,
                    xy=(0, current_text_height),
                    font_size=font_size,
                    color=color,
                )
            elif place == "right":
                x_left = x + box_width - line_size[0]
                self.write_text(
                    text=line,
                    font_filename=font_filename,
                    xy=(x_left, current_text_height),
                    font_size=font_size,
                    color=color,
                )
            elif place == "center":
                x_left = int(x + ((box_width - line_size[0]) / 2))
                self.write_text(
                    text=line,
                    font_filename=font_filename,
                    xy=(x_left, current_text_height),
                    font_size=font_size,
                    color=color,
                )
            else:
                raise ValueError(f"Place {place} is not supported. Use one of: `left`, `right` or `center`!")
            # Increment text height
            current_text_height += line_size[1]
        return (x, current_text_height)