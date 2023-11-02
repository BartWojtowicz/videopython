import numpy as np
import pytest

from videopython.base.transitions import FadeTransition
from videopython.base import Video

from conftest import short_video, long_video


def test_fade_transition_length(short_video, long_video):
    transition_short = FadeTransition(effect_time_seconds=1)
    transition_long = FadeTransition(effect_time_seconds=5)

    # Apply the transition to the videos
    result_short = transition_short.apply((short_video, short_video))
    result_long = transition_long.apply((long_video, long_video))

    assert short_video.frames.shape[0] == result_short.frames.shape[0]
    assert long_video.frames.shape[0] == result_long.frames.shape[0]
