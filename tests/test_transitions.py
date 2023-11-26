import numpy as np

from videopython.base.transitions import FadeTransition, InstantTransition
from videopython.base.video import Video


def test_fade_transition_length(black_frames_video):
    org_length = black_frames_video.frames.shape[0]
    transition_short = FadeTransition(effect_time_seconds=1)
    result_short = transition_short.apply((black_frames_video, black_frames_video.copy()))

    assert result_short.frames.shape[0] == 2 * org_length - 1 * black_frames_video.fps


def test_fade_correctness():
    all_black_video = Video.from_frames(np.zeros((255, 10, 10, 3), dtype=np.uint8), fps=255)
    all_white_video = Video.from_frames(np.full((255, 10, 10, 3), 255, dtype=np.uint8), fps=255)

    transition = FadeTransition(effect_time_seconds=1)
    result = transition.apply((all_black_video, all_white_video))
    assert result.frames[0].sum() == 0
    assert result.frames[127].sum() == 127 * 10 * 10 * 3
    assert result.frames[-1].sum() == 255 * 3 * 10 * 10


def test_instant_transition(black_frames_video):
    result = InstantTransition().apply((black_frames_video, black_frames_video))

    assert result.frames.shape[0] == 2 * black_frames_video.frames.shape[0]
    assert np.all(result.frames == (black_frames_video + black_frames_video).frames)
