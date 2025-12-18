from typing import Literal

import numpy as np

from videopython.base.transforms import ResampleFPS, Resize
from videopython.base.video import Video


class StackVideos:
    """Stacks two videos together horizontally or vertically."""

    def __init__(self, mode: Literal["horizontal", "vertical"]) -> None:
        """Initialize video stacker.

        Args:
            mode: Stack direction, either "horizontal" or "vertical".
        """
        self.mode = mode

    def _validate(self, video1: Video, video2: Video) -> tuple[Video, Video]:
        video1, video2 = self._align_shapes(video1, video2)
        video1, video2 = self._align_fps(video1, video2)
        video1, video2 = self._align_duration(video1, video2)
        return video1, video2

    def _align_fps(self, video1: Video, video2: Video) -> tuple[Video, Video]:
        if video1.fps > video2.fps:
            video1 = ResampleFPS(fps=video2.fps).apply(video1)
        elif video1.fps < video2.fps:
            video2 = ResampleFPS(fps=video1.fps).apply(video2)
        return (video1, video2)

    def _align_shapes(self, video1: Video, video2: Video) -> tuple[Video, Video]:
        if self.mode == "horizontal":
            video2 = Resize(height=video1.metadata.height).apply(video2)
        elif self.mode == "vertical":
            video2 = Resize(width=video1.metadata.width).apply(video2)
        return (video1, video2)

    def _align_duration(self, video1: Video, video2: Video) -> tuple[Video, Video]:
        if len(video1.frames) > len(video2.frames):
            video1 = video1[: len(video2.frames)]
        elif len(video1.frames) < len(video2.frames):
            video2 = video2[: len(video1.frames)]
        return (video1, video2)

    def apply(self, videos: tuple[Video, Video]) -> Video:
        """Stack two videos together, aligning FPS, dimensions, and duration.

        Args:
            videos: Tuple of two videos to stack.

        Returns:
            Stacked video with overlaid audio.
        """
        videos = self._validate(*videos)
        axis = 1 if self.mode == "vertical" else 2
        new_frames = np.concatenate((videos[0].frames, videos[1].frames), axis=axis)
        new_audio = videos[0].audio.overlay(videos[1].audio)
        return Video(frames=new_frames, fps=videos[0].fps, audio=new_audio)
