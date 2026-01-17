"""Pytest configuration.

Test structure:
- tests/base/ - No AI dependencies, runs in CI
- tests/ai/ - Requires AI extras, mostly runs in CI

AI test markers:
- @pytest.mark.requires_model_download - Tests that download models 100MB+
  (e.g., EasyOCR). These are skipped in CI.

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
        "requires_model_download: marks tests that download heavy AI models (YOLO, EasyOCR, PANNs) - skipped in CI",
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


@pytest.fixture(scope="session")
def big_video():
    return Video.from_path(BIG_VIDEO_PATH)


@pytest.fixture(scope="session")
def small_video():
    return Video.from_path(SMALL_VIDEO_PATH)


@pytest.fixture(scope="session")
def black_frames_test_video():
    return Video.from_image(np.zeros((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
