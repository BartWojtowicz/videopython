import numpy as np

from videopython.base.transitions import FadeTransition, InstantTransition
from videopython.base.video import Video


def test_fade_transition_length(small_video):
    org_length = small_video.frames.shape[0]
    transition_short = FadeTransition(effect_time_seconds=1)
    result_short = transition_short.apply((small_video, small_video.copy()))

    assert result_short.frames.shape[0] == 2 * org_length - 1 * small_video.fps


def test_fade_correctness():
    all_black_video = Video.from_frames(np.zeros((255, 10, 10, 3), dtype=np.uint8), fps=255)
    all_white_video = Video.from_frames(np.full((255, 10, 10, 3), 255, dtype=np.uint8), fps=255)

    transition = FadeTransition(effect_time_seconds=1)
    result = transition.apply((all_black_video, all_white_video))
    assert result.frames[0].sum() == 0
    assert result.frames[127].sum() == 127 * 10 * 10 * 3
    assert result.frames[-1].sum() == 255 * 3 * 10 * 10


def test_instant_transition(small_video):
    result = InstantTransition().apply((small_video, small_video))

    assert result.frames.shape[0] == 2 * small_video.frames.shape[0]
    assert np.all(result.frames == (small_video + small_video).frames)
