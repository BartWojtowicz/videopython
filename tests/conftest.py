import pytest
import numpy as np

from videopython.base import Video


@pytest.fixture
def video_form_path(video_path: str):
    return Video.from_video(video_path)


@pytest.fixture
def video_from_frames(frames: np, ndarray, fps: int):
    return Video.from_frames(frames, fps)
