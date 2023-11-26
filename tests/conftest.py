import numpy as np
import pytest

from videopython.base.video import Video
from videopython.project_config import LocationConfig


@pytest.fixture
def black_frames_video():
    return Video.from_path(str(LocationConfig.test_videos_dir / "fast_benchmark.mp4"))


@pytest.fixture
def big_video():
    return Video.from_path(str(LocationConfig.test_videos_dir / "slow_benchmark.mp4"))


@pytest.fixture
def black_frames_video():
    return Video.from_image(np.zeros((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
