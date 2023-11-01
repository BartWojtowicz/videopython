import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import final

import numpy as np

from videopython.base.video import Video


class Transitions(Enum):
    NULL = "null"
    FADE = "fade"


class Transition(ABC):
    """Abstract class for Transitions on Videos.
    
    To build a new transition, you need to implement the `_apply`
    abstractmethod.
    """

    @final
    def apply(self, videos: tuple[Video, Video], **kwargs) -> Video:
        assert videos[0].metadata.can_be_merged_with(videos[1].metadata)
        return self._apply(videos, **kwargs)

    @abstractmethod
    def _apply(self, videos: tuple[Video, Video], **kwargs) -> Video:
        pass


class InstantTransition(Transition):
    """Instant cut without any transition."""

    def _apply(self, videos: Video) -> Video:
        return videos[0].concatenate(videos[1], axis=0)


class FadeTransition(Transition):
    """Fade transition. Each video must last at least half of effect time."""

    def __init__(self, effect_time_seconds: float):
        self.effect_time_seconds = effect_time_seconds

    def fade(self, frames1, frames2):
        assert len(frames1) == len(frames2)
        t = len(frames1)
        linspace = np.linspace(0, 1, t).reshape(t, 1, 1, 1).astype(np.float16)
        transitioned_frames = frames1.astype(
            np.float16) * (np.ones_like(linspace) - linspace) + frames2.astype(
                np.float16) * linspace
        return transitioned_frames.astype(np.uint8)

    def _apply(self, videos: tuple[Video, Video]) -> Video:
        video_fps = videos[0].fps
        for video in videos:
            if video.total_seconds < self.effect_time_seconds:
                raise RuntimeError("Not enough space to make transition!")

        effect_time_fps = math.floor(self.effect_time_seconds * video_fps)
        transition = self.fade(videos[0].frames[-effect_time_fps:],
                               videos[1].frames[:effect_time_fps])
        transition_video = Video.from_frames(transition, video_fps)
        videos[0].frames = videos[0].frames[:-effect_time_fps]
        videos[1].frames = videos[1].frames[effect_time_fps:]
        return videos[0] + transition_video + videos[1]
