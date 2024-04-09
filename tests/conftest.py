from pathlib import Path

import numpy as np
import pytest

from videopython.base.video import Video

TEST_ROOT_DIR: Path = Path(__file__).parent
TEST_DATA_DIR: Path = TEST_ROOT_DIR / "test_data"


@pytest.fixture
def small_video():
    return Video.from_path(str(TEST_DATA_DIR / "fast_benchmark.mp4"))


@pytest.fixture
def big_video():
    return Video.from_path(str(TEST_DATA_DIR / "slow_benchmark.mp4"))


@pytest.fixture
def black_frames_video():
    return Video.from_image(np.zeros((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)


@pytest.fixture
def test_font_path():
    return str(TEST_DATA_DIR / "test_font.ttf")
