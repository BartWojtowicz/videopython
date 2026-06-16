"""Fixtures shared across editing tests."""

import pytest

from videopython.base.video import Video


@pytest.fixture
def render(tmp_path):
    """Render a VideoEdit plan to a file and load it back as a ``Video``.

    Streaming-to-file is the only execution engine, so a test that needs a
    ``Video`` object runs the plan to disk and reads it back. Keyword arguments
    (e.g. ``context=``) are forwarded to ``run_to_file``. Pass distinct
    ``name=`` values when rendering more than one plan in a single test.
    """

    def _render(plan, *, name="out.mp4", **kwargs):
        out = plan.run_to_file(tmp_path / name, **kwargs)
        return Video.from_path(str(out))

    return _render
