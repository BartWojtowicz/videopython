from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from videopython.base.exceptions import OutOfBoundsError
from videopython.base.video import Video


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
        text_color: tuple[int, int, int] = (0, 0, 0),
        background_color: None | tuple[int, int, int, int] = None,
        background_padding: int = 0,
        place: Literal["left", "right", "center"] = "left",
    ) -> tuple[int, int]:
        """Write text in box described by upper-left corner and maxium width of the box.

        Args:
            text: Text to be written inside the box.
            font_filename: Path to the font file.
            xy: X and Y coordinates describing upper-left of the box containing the text.
            box_width: Pixel width of the box containing the text.
            font_size: Font size.
            text_color: RGB color of the text.
            background_color: If set, adds background color to the text box. Expects RGBA values.
            background_padding: Number of padding pixels to add when adding text background color.
            place: Strategy for justifying the text inside the container box. Defaults to "left".

        Returns:
            Lower-left corner of the written text box.
        """
        x, y = xy
        lines = self._split_lines_by_width(text, font_filename, font_size, box_width)
        # Run checks to see if the text will fit
        if x + box_width > self.image_size[0]:
            raise OutOfBoundsError(f"Box width {box_width} is too big for the image width {self.image_size[0]}!")
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
                    xy=(x, current_text_height),
                    font_size=font_size,
                    color=text_color,
                )
            elif place == "right":
                x_left = x + box_width - line_size[0]
                self.write_text(
                    text=line,
                    font_filename=font_filename,
                    xy=(x_left, current_text_height),
                    font_size=font_size,
                    color=text_color,
                )
            elif place == "center":
                x_left = int(x + ((box_width - line_size[0]) / 2))
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
            box_slice = img[y:current_text_height, x : x + box_width]
            text_mask = np.any(box_slice != 0, axis=2).astype(np.uint8)
            if not isinstance(text_mask, np.ndarray):
                raise TypeError(
                    f"The returned text mask is of type {type(text_mask)}, " "but it should be numpy array!"
                )
            xmin, xmax, ymin, ymax = self._find_smallest_bounding_rect(text_mask)
            # Get global bounding box position
            xmin += x - background_padding
            xmax += x + background_padding
            ymin += y - background_padding
            ymax += y + background_padding
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
        return (x, current_text_height)

    def _find_smallest_bounding_rect(self, mask: np.ndarray) -> tuple[int, int, int, int]:
        """Find the smallest bounding rectangle for the mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax


class SlideOverImage:
    def __init__(
        self,
        direction: Literal["left", "right"],
        video_shape: tuple[int, int] = (1080, 1920),
        fps: float = 24.0,
        length_seconds: float = 1.0,
    ) -> None:
        self.direction = direction
        self.video_width, self.video_height = video_shape
        self.fps = fps
        self.length_seconds = length_seconds

    def apply(self, image: np.ndarray) -> Video:
        image = self._resize(image)
        max_offset = image.shape[1] - self.video_width
        frame_count = round(self.fps * self.length_seconds)

        deltas = np.linspace(0, max_offset, frame_count)
        frames = []

        for delta in deltas:
            if self.direction == "right":
                frame = image[:, round(delta) : round(delta) + self.video_width]
            elif self.direction == "left":
                frame = image[:, image.shape[1] - round(delta) - self.video_width : image.shape[1] - round(delta)]
            frames.append(frame)

        return Video.from_frames(frames=np.stack(frames, axis=0), fps=self.fps)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        resize_factor = image.shape[0] / self.video_height
        resize_dims = (round(image.shape[1] / resize_factor), round(image.shape[0] / resize_factor))  # width, height
        image = cv2.resize(image, resize_dims)
        if self.video_height > image.shape[0] or self.video_width > image.shape[1]:
            raise ValueError(
                f"Image `{image.shape}` is too small for the video frame `({self.video_width}, {self.video_height})`!"
            )
        return image
