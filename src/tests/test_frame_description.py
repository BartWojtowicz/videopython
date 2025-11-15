from videopython.base.frames import FrameDescription
from videopython.base.scene_description import SceneDescription
from videopython.base.scenes import Scene


class TestFrameDescription:
    """Tests for FrameDescription dataclass."""

    def test_frame_description_creation(self):
        """Test basic frame description creation."""
        fd = FrameDescription(frame_index=42, timestamp=1.75, description="A dog playing in a park")
        assert fd.frame_index == 42
        assert fd.timestamp == 1.75
        assert fd.description == "A dog playing in a park"


class TestSceneDescription:
    """Tests for SceneDescription dataclass."""

    def test_scene_description_creation(self):
        """Test basic scene description creation."""
        scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=120)
        frame_descriptions = [
            FrameDescription(frame_index=0, timestamp=0.0, description="Scene start"),
            FrameDescription(frame_index=60, timestamp=2.5, description="Scene middle"),
            FrameDescription(frame_index=119, timestamp=4.96, description="Scene end"),
        ]
        sd = SceneDescription(scene=scene, frame_descriptions=frame_descriptions)

        assert sd.scene == scene
        assert len(sd.frame_descriptions) == 3
        assert sd.num_frames_described == 3

    def test_get_description_summary(self):
        """Test getting summary of all descriptions."""
        scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=120)
        frame_descriptions = [
            FrameDescription(frame_index=0, timestamp=0.0, description="A red car."),
            FrameDescription(frame_index=60, timestamp=2.5, description="The car drives away."),
        ]
        sd = SceneDescription(scene=scene, frame_descriptions=frame_descriptions)

        summary = sd.get_description_summary()
        assert summary == "A red car. The car drives away."
