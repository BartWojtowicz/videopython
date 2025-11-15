import pytest

from videopython.base.description import Scene


class TestScene:
    """Tests for Scene dataclass."""

    def test_scene_creation(self):
        """Test basic scene creation."""
        scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=120)
        assert scene.start == 0.0
        assert scene.end == 5.0
        assert scene.start_frame == 0
        assert scene.end_frame == 120

    def test_scene_duration(self):
        """Test scene duration calculation."""
        scene = Scene(start=2.5, end=7.8, start_frame=60, end_frame=187)
        assert scene.duration == pytest.approx(5.3)

    def test_scene_frame_count(self):
        """Test scene frame count calculation."""
        scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=120)
        assert scene.frame_count == 120

    def test_get_frame_indices_single(self):
        """Test getting a single frame index."""
        scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=100)
        indices = scene.get_frame_indices(num_samples=1)
        assert len(indices) == 1
        assert indices[0] == 50  # Middle frame

    def test_get_frame_indices_multiple(self):
        """Test getting multiple evenly distributed frame indices."""
        scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=100)
        indices = scene.get_frame_indices(num_samples=3)
        assert len(indices) == 3
        assert indices[0] == 0
        assert indices[-1] == 99

    def test_get_frame_indices_invalid(self):
        """Test that invalid num_samples raises ValueError."""
        scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=100)
        with pytest.raises(ValueError):
            scene.get_frame_indices(num_samples=0)
        with pytest.raises(ValueError):
            scene.get_frame_indices(num_samples=-1)
