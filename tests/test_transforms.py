import numpy as np
import pytest

from videopython.base.transforms import CutFrames, CutSeconds


@pytest.mark.parametrize("start, end", [(0,100), (100,101), (100, 120)])
def test_cut_frames(start, end, small_video):

    cut_frames = CutFrames(start_frame=start, end_frame=end)
    start_frame = small_video.frames[start].copy()
    transformed = cut_frames.apply(small_video)
    assert len(transformed.frames) == (end - start)
    assert np.all(transformed.frames[0] == start_frame)

@pytest.mark.parametrize("start, end", [(0,0.5), (0,1), (0.5, 1.5)])
def test_cut_seconds(start, end, small_video):
    cut_seconds = CutSeconds(start_second=start, end_second=end)
    start_frame = small_video.frames[round(start * small_video.fps)].copy()
    transformed = cut_seconds.apply(small_video)
    assert len(transformed.frames) == round((end - start) * small_video.fps)
    assert np.all(transformed.frames[0] == start_frame)
