import pytest

from videopython.base.video import Video
from videopython.project_config import LocationConfig


@pytest.fixture
def small_video():
    return Video.from_path(str(LocationConfig.test_videos_dir / "fast_benchmark.mp4"))


@pytest.fixture
def big_video():
    return Video.from_path(str(LocationConfig.test_videos_dir / "slow_benchmark.mp4"))
