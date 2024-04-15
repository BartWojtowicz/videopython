from abc import ABC, abstractmethod
from typing import Literal, final

import cv2
import numpy as np
from tqdm import tqdm

from videopython.base.video import Video


class Effect(ABC):
    """Abstract class for effect on frames of video.

    The effect must not change the number of frames and the shape of the frames.
    """

    @final
    def apply(self, video: Video) -> Video:
        original_shape = video.video_shape
        video_with_effect = self._apply(video)
        if not video_with_effect.video_shape == original_shape:
            raise RuntimeError("The effect must not change the number of frames and the shape of the frames!")
        return video_with_effect

    @abstractmethod
    def _apply(self, video: Video) -> Video:
        pass


class FullImageOverlay(Effect):
    def __init__(self, overlay_image: np.ndarray, alpha: float | None = None):
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError("Alpha must be in range [0, 1]!")
        elif not (overlay_image.ndim == 3 and overlay_image.shape[-1] in [3, 4]):
            raise ValueError("Only RGB and RGBA images are supported as an overlay!")
        elif alpha is None:
            alpha = 1.0

        if overlay_image.shape[-1] == 3:
            overlay_image = np.dstack([overlay_image, np.full(overlay_image.shape[:2], 255, dtype=np.uint8)])
        overlay_image[:, :, 3] = overlay_image[:, :, 3] * alpha

        self._overlay_alpha = (overlay_image[:, :, 3] / 255.0)[:, :, np.newaxis]
        self._base_transparency = 1 - self._overlay_alpha

        self.overlay = overlay_image[:, :, :3] * self._overlay_alpha

    def _overlay(self, img: np.ndarray) -> np.ndarray:
        return self.overlay + (img * self._base_transparency)

    def _apply(self, video: Video) -> Video:
        if not video.frame_shape == self.overlay.shape:
            raise ValueError(
                f"Mismatch of overlay shape `{self.overlay.shape}` with video shape: `{video.frame_shape}`!"
            )
        print("Overlaying video...")
        video.frames = np.array([self._overlay(frame) for frame in tqdm(video.frames)], dtype=np.uint8)
        return video


class Blur(Effect):
    def __init__(
        self,
        mode: Literal["constant", "ascending", "descending"],
        iterations: int,
        kernel_size: tuple[int, int] = (5, 5),
    ):
        if iterations < 1:
            raise ValueError("Iterations must be at least 1!")
        self.mode = mode
        self.iterations = iterations
        self.kernel_size = kernel_size

    def _apply(self, video: Video) -> Video:
        n_frames = len(video.frames)
        new_frames = []
        if self.mode == "constant":
            for frame in video.frames:
                blurred_frame = frame
                for _ in range(self.iterations):
                    blurred_frame = cv2.GaussianBlur(blurred_frame, self.kernel_size, 0)
                new_frames.append(blurred_frame)
        elif self.mode == "ascending":
            for i, frame in tqdm(enumerate(video.frames)):
                frame_iterations = max(1, round((i / n_frames) * self.iterations))
                blurred_frame = frame
                for _ in range(frame_iterations):
                    blurred_frame = cv2.GaussianBlur(blurred_frame, self.kernel_size, 0)
                new_frames.append(blurred_frame)
        elif self.mode == "descending":
            for i, frame in tqdm(enumerate(video.frames)):
                frame_iterations = max(round(((n_frames - i) / n_frames) * self.iterations), 1)
                blurred_frame = frame
                for _ in range(frame_iterations):
                    blurred_frame = cv2.GaussianBlur(blurred_frame, self.kernel_size, 0)
                new_frames.append(blurred_frame)
        else:
            raise ValueError(f"Unknown mode: `{self.mode}`.")
        video.frames = np.asarray(new_frames)
        return video


class Zoom(Effect):
    def __init__(self, zoom_factor: float, mode: Literal["in", "out"]):
        if zoom_factor <= 1:
            raise ValueError("Zoom factor must be greater than 1!")
        self.zoom_factor = zoom_factor
        self.mode = mode

    def _apply(self, video: Video) -> Video:
        n_frames = len(video.frames)
        new_frames = []

        width = video.metadata.width
        height = video.metadata.height
        crop_sizes_w, crop_sizes_h = np.linspace(width // self.zoom_factor, width, n_frames), np.linspace(
            height // self.zoom_factor, height, n_frames
        )

        if self.mode == "in":
            for frame, w, h in tqdm(zip(video.frames, reversed(crop_sizes_w), reversed(crop_sizes_h))):

                x = width / 2 - w / 2
                y = height / 2 - h / 2

                cropped_frame = frame[round(y) : round(y + h), round(x) : round(x + w)]
                zoomed_frame = cv2.resize(cropped_frame, (width, height))
                new_frames.append(zoomed_frame)
        elif self.mode == "out":
            for frame, w, h in tqdm(zip(video.frames, crop_sizes_w, crop_sizes_h)):
                x = width / 2 - w / 2
                y = height / 2 - h / 2

                cropped_frame = frame[round(y) : round(y + h), round(x) : round(x + w)]
                zoomed_frame = cv2.resize(cropped_frame, (width, height))
                new_frames.append(zoomed_frame)
        else:
            raise ValueError(f"Unknown mode: `{self.mode}`.")
        video.frames = np.asarray(new_frames)
        return video
