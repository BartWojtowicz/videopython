import math
from abc import ABC, abstractmethod
from typing import final

import numpy as np

from videopython.base.effects import Blur
from videopython.base.exceptions import IncompatibleVideoError, InsufficientDurationError
from videopython.base.video import Video

__all__ = [
    "Transition",
    "InstantTransition",
    "FadeTransition",
    "BlurTransition",
]


class Transition(ABC):
    """Abstract class for Transitions on Videos.

    To build a new transition, you need to implement the `_apply`
    abstractmethod.
    """

    @final
    def apply(self, videos: tuple[Video, Video]) -> Video:
        if not videos[0].metadata.can_be_merged_with(videos[1].metadata):
            raise IncompatibleVideoError("Videos have incompatible metadata and cannot be merged")
        return self._apply(videos)

    @abstractmethod
    def _apply(self, videos: tuple[Video, Video]) -> Video:
        pass


class InstantTransition(Transition):
    """Instant cut without any transition."""

    def _apply(self, videos: tuple[Video, Video]) -> Video:
        return videos[0] + videos[1]


class FadeTransition(Transition):
    """Fade transition. Each video must last at least half of effect time."""

    def __init__(self, effect_time_seconds: float):
        self.effect_time_seconds = effect_time_seconds

    def fade(self, frames1, frames2):
        if len(frames1) != len(frames2):
            raise ValueError(f"Frame sequences must have equal length, got {len(frames1)} and {len(frames2)}")
        t = len(frames1)
        # Calculate transitioned frames using weighted average
        transitioned_frames = (
            frames1 * (t - np.arange(t) - 1)[:, np.newaxis, np.newaxis, np.newaxis]
            + frames2 * np.arange(t)[:, np.newaxis, np.newaxis, np.newaxis]
        ) / (t - 1)

        return transitioned_frames.astype(np.uint8)

    def _apply(self, videos: tuple[Video, Video]) -> Video:
        video_fps = videos[0].fps
        for video in videos:
            if video.total_seconds < self.effect_time_seconds:
                raise InsufficientDurationError("Not enough space to make transition!")

        effect_time_fps = math.floor(self.effect_time_seconds * video_fps)
        transition = self.fade(videos[0].frames[-effect_time_fps:], videos[1].frames[:effect_time_fps])

        faded_videos = Video.from_frames(
            np.r_[
                "0,2",
                videos[0].frames[:-effect_time_fps],
                transition,
                videos[1].frames[effect_time_fps:],
            ],
            fps=video_fps,
        )
        faded_videos.audio = videos[0].audio.concat(videos[1].audio, crossfade=(effect_time_fps / video_fps))
        return faded_videos


class BlurTransition(Transition):
    def __init__(
        self, effect_time_seconds: float = 1.5, blur_iterations: int = 400, blur_kernel_size: tuple[int, int] = (11, 11)
    ):
        self.effect_time_seconds = effect_time_seconds
        self.blur_iterations = blur_iterations
        self.blur_kernel_size = blur_kernel_size

    def _apply(self, videos: tuple[Video, Video]) -> Video:
        video_fps = videos[0].fps
        for video in videos:
            if video.total_seconds < self.effect_time_seconds:
                raise InsufficientDurationError("Not enough space to make transition!")

        effect_time_fps = math.floor(self.effect_time_seconds * video_fps)

        # Create frame-only videos for blur effect (avoids audio slicing issues)
        end_frames = Video.from_frames(videos[0].frames[-effect_time_fps:], fps=video_fps)
        start_frames = Video.from_frames(videos[1].frames[:effect_time_fps], fps=video_fps)

        ascending_blur = Blur("ascending", self.blur_iterations, self.blur_kernel_size)
        descending_blur = Blur("descending", self.blur_iterations, self.blur_kernel_size)
        transition = ascending_blur.apply(end_frames) + descending_blur.apply(start_frames)

        blurred_videos = Video.from_frames(
            np.r_[
                "0,2",
                videos[0].frames[:-effect_time_fps],
                transition.frames,
                videos[1].frames[effect_time_fps:],
            ],
            fps=video_fps,
        )
        blurred_videos.audio = videos[0].audio.concat(videos[1].audio)
        return blurred_videos
