import cv2
import numpy as np
import pytest

from videopython.base.transforms import CutFrames, CutSeconds, Resize


@pytest.mark.parametrize("start, end", [(0, 100), (100, 101), (100, 120)])
def test_cut_frames(start, end, small_video):
    cut_frames = CutFrames(start_frame=start, end_frame=end)
    start_frame = small_video.frames[start].copy()
    transformed = cut_frames.apply(small_video)
    assert len(transformed.frames) == (end - start)
    assert np.all(transformed.frames[0] == start_frame)


@pytest.mark.parametrize("start, end", [(0, 0.5), (0, 1), (0.5, 1.5)])
def test_cut_seconds(start, end, small_video):
    cut_seconds = CutSeconds(start_second=start, end_second=end)
    start_frame = small_video.frames[round(start * small_video.fps)].copy()
    transformed = cut_seconds.apply(small_video)
    assert len(transformed.frames) == round((end - start) * small_video.fps)
    assert np.all(transformed.frames[0] == start_frame)


@pytest.mark.parametrize(
    "height,width",
    [
        (
            40,
            60,
        ),
        (
            500,
            700,
        ),
    ],
)
def test_video_resize(height, width, small_video):
    """Tests Video.resize."""

    resample = Resize(new_height=height, new_width=width)
    video = resample.apply(small_video)

    assert video.frames.shape[1:3] == (height, width)
    assert np.all(
        video.frames[0]
        == cv2.resize(
            small_video.frames[0],
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    )
    assert np.all(
        video.frames[-1]
        == cv2.resize(
            small_video.frames[-1],
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    )
    assert np.all(
        video.frames[len(video.frames) // 2]
        == cv2.resize(
            small_video.frames[len(small_video.frames) // 2],
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    )
