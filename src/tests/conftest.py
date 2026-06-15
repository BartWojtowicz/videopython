"""Pytest configuration.

Test structure:
- tests/base/ - No AI dependencies, runs in CI
- tests/ai/ - Requires AI extras, mostly runs in CI

AI test markers:
- @pytest.mark.requires_model_download - Tests that download models 100MB+
  (e.g., YOLOv8-face, Qwen3.5, PANNs). These are skipped in CI.

CI runs: uv run pytest src/tests/ai -m "not requires_model_download"
Local runs: uv run pytest src/tests/ai -v (all tests including model downloads)
"""

import numpy as np
import pytest

from tests.test_config import (
    BIG_VIDEO_PATH,
    SMALL_VIDEO_PATH,
)
from videopython.base.video import Video


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "requires_model_download: marks tests that download heavy AI models "
        "(YOLOv8-face, Qwen3.5, PANNs, Whisper) - skipped in CI",
    )


def pytest_collection_modifyitems(config, items):
    """Enforce that base tests don't import from videopython.ai."""
    for item in items:
        # Only check tests in the base/ subdirectory
        if "/base/" not in str(item.fspath):
            continue

        module = item.module
        if module is None:
            continue

        for name, obj in vars(module).items():
            if hasattr(obj, "__module__") and obj.__module__ and obj.__module__.startswith("videopython.ai"):
                pytest.fail(
                    f"Test {item.nodeid} imports from videopython.ai ({obj.__module__}). "
                    f"Move this test to tests/ai/ or remove the import."
                )


# Load each sample video once per session, but hand every test a fresh copy:
# many tests mutate the Video in place (``.apply`` reassigns ``video.frames``),
# so a shared session-scoped instance leaks state across tests (a transforms
# test could shrink it to 0 frames and break a later effect test). The
# session loader keeps disk I/O once; the per-test copy keeps tests isolated.
@pytest.fixture(scope="session")
def _big_video_source():
    return Video.from_path(BIG_VIDEO_PATH)


@pytest.fixture
def big_video(_big_video_source):
    return _big_video_source.copy()


@pytest.fixture(scope="session")
def _small_video_source():
    return Video.from_path(SMALL_VIDEO_PATH)


@pytest.fixture
def small_video(_small_video_source):
    return _small_video_source.copy()


@pytest.fixture
def black_frames_test_video():
    return Video.from_image(np.zeros((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
