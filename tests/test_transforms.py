import cv2
import numpy as np
import pytest

from videopython.base.transforms import CutFrames, CutSeconds, Resize


@pytest.mark.parametrize("start, end", [(0, 100), (100, 101), (100, 120)])
def test_cut_frames(start, end, black_frames_video):
    cut_frames = CutFrames(start_frame=start, end_frame=end)
    start_frame = black_frames_video.frames[start].copy()
    transformed = cut_frames.apply(black_frames_video)
    assert len(transformed.frames) == (end - start)
    assert np.all(transformed.frames[0] == start_frame)


@pytest.mark.parametrize("start, end", [(0, 0.5), (0, 1), (0.5, 1.5)])
def test_cut_seconds(start, end, black_frames_video):
    cut_seconds = CutSeconds(start_second=start, end_second=end)
    start_frame = black_frames_video.frames[round(start * black_frames_video.fps)].copy()
    transformed = cut_seconds.apply(black_frames_video)
    assert len(transformed.frames) == round((end - start) * black_frames_video.fps)
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
def test_video_resize(height, width, black_frames_video):
    """Tests Video.resize."""

    resample = Resize(new_height=height, new_width=width)
    video = resample.apply(black_frames_video)

    assert video.frames.shape[1:3] == (height, width)
    assert np.all(
        video.frames[0]
        == cv2.resize(
            black_frames_video.frames[0],
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    )
    assert np.all(
        video.frames[-1]
        == cv2.resize(
            black_frames_video.frames[-1],
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    )
    assert np.all(
        video.frames[len(video.frames) // 2]
        == cv2.resize(
            black_frames_video.frames[len(black_frames_video.frames) // 2],
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    )
