import numpy as np
import pytest

from videopython.base import Video
from videopython.project_config import LocationConfig


@pytest.fixture
def short_video():
    return Video.from_path(str(LocationConfig.test_videos_dir / "fast_benchmark.mp4"))


@pytest.fixture
def long_video():
    return Video.from_path(str(LocationConfig.test_videos_dir / "slow_benchmark.mp4"))
