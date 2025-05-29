import numpy as np
import pytest

from videopython.base.combine import StackVideos
from videopython.base.video import Video


@pytest.mark.parametrize("mode", ["horizontal", "vertical"])
def test_combine_same_shape(mode):
    v1 = Video.from_image(np.zeros((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
    v2 = Video.from_image(255 * np.ones((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
    StackVideos(mode=mode).apply(videos=(v1, v2))


@pytest.mark.parametrize("mode", ["horizontal", "vertical"])
def test_combine_different_dimensions(mode):
    v1 = Video.from_image(np.zeros((300, 400, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
    v2 = Video.from_image(255 * np.ones((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
    StackVideos(mode=mode).apply(videos=(v1, v2))


@pytest.mark.parametrize("mode", ["horizontal", "vertical"])
def test_combine_different_fps(mode):
    v1 = Video.from_image(np.zeros((100, 100, 3), dtype=np.uint8), fps=30, length_seconds=5.0)
    v2 = Video.from_image(255 * np.ones((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
    StackVideos(mode=mode).apply(videos=(v1, v2))


@pytest.mark.parametrize("mode", ["horizontal", "vertical"])
def test_combine_different_duration(mode):
    v1 = Video.from_image(np.zeros((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=2.0)
    v2 = Video.from_image(255 * np.ones((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
    StackVideos(mode=mode).apply(videos=(v1, v2))


@pytest.mark.parametrize("mode", ["horizontal", "vertical"])
def test_combine_different_duration_and_fps(mode):
    v1 = Video.from_image(np.zeros((100, 100, 3), dtype=np.uint8), fps=30, length_seconds=2.0)
    v2 = Video.from_image(255 * np.ones((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
    StackVideos(mode=mode).apply(videos=(v1, v2))
