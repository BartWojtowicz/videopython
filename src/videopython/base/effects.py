from abc import ABC, abstractmethod
from typing import final

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
