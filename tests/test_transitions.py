import numpy as np
import pytest

from videopython.base.transitions import FadeTransition, InstantTransition
from videopython.base import Video


def test_fade_transition_length(short_video, long_video):
    transition_short = FadeTransition(effect_time_seconds=1)
    transition_long = FadeTransition(effect_time_seconds=3)

    # Apply the transition to the videos
    result_short = transition_short.apply((short_video, short_video))
    result_long = transition_long.apply((long_video, long_video))

    assert short_video.frames.shape[0] == result_short.frames.shape[0]
    assert long_video.frames.shape[0] == result_long.frames.shape[0]


def test_fade_correctness():
    all_black_video = Video.from_frames(np.zeros((255, 10, 10, 3), dtype=np.uint8), fps=255)
    all_white_video = Video.from_frames(np.full((255, 10, 10, 3), 255, dtype=np.uint8), fps=255)

    transition = FadeTransition(effect_time_seconds=1)
    result = transition.apply((all_black_video, all_white_video))
    assert result.frames[0].sum() == 0
    assert result.frames[127].sum() == 127 * 10 * 10 * 3
    assert result.frames[-1].sum() == 255 * 3 * 10 * 10


def test_instant_transition(short_video):
    result = InstantTransition().apply((short_video, short_video))

    assert result.frames.shape[0] == 2 * short_video.frames.shape[0]
    assert np.all(result.frames == (short_video + short_video).frames)
