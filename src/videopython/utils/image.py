from typing import Literal

import cv2
import numpy as np

from videopython.base.video import Video


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
