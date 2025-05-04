import numpy as np
import pytest

from videopython.base.video import Video

from .test_config import (
    BIG_VIDEO_PATH,
    SMALL_VIDEO_PATH,
)


@pytest.fixture(scope="session")
def big_video():
    return Video.from_path(BIG_VIDEO_PATH)


@pytest.fixture(scope="session")
def small_video():
    return Video.from_path(SMALL_VIDEO_PATH)


@pytest.fixture(scope="session")
def black_frames_test_video():
    return Video.from_image(np.zeros((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
