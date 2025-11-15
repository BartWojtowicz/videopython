import numpy as np
import pytest
from PIL import Image

from videopython.ai.understanding.image import ImageToText
from videopython.base.description import FrameDescription, Scene
from videopython.base.video import Video


class TestImageToText:
    """Tests for ImageToText class."""

    @pytest.fixture
    def sample_video(self):
        """Create a sample video for testing."""
        # Create 100 frames at 24 fps (~4 seconds)
        frames = np.random.randint(0, 255, (100, 200, 200, 3), dtype=np.uint8)
        return Video.from_frames(frames, fps=24.0)

    def test_image_to_text_initialization_cpu(self):
        """Test ImageToText initialization on CPU."""
        model = ImageToText(device="cpu")
        assert model.device == "cpu"
        assert model.model is not None
        assert model.processor is not None

    def test_describe_image_numpy(self):
        """Test describing an image from numpy array."""
        model = ImageToText(device="cpu")
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        description = model.describe_image(image)
        assert isinstance(description, str)
        assert len(description) > 0

    def test_describe_image_pil(self):
        """Test describing an image from PIL Image."""
        model = ImageToText(device="cpu")
        image = Image.new("RGB", (200, 200), color="red")

        description = model.describe_image(image)
        assert isinstance(description, str)
        assert len(description) > 0

    def test_describe_image_with_prompt(self):
        """Test describing an image with a prompt."""
        model = ImageToText(device="cpu")
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        description = model.describe_image(image, prompt="Describe the colors in this image")
        assert isinstance(description, str)
        assert len(description) > 0

    def test_describe_frame(self, sample_video):
        """Test describing a single frame from a video."""
        model = ImageToText(device="cpu")
        frame_desc = model.describe_frame(sample_video, frame_index=50)

        assert isinstance(frame_desc, FrameDescription)
        assert frame_desc.frame_index == 50
        assert frame_desc.timestamp == pytest.approx(50 / 24.0)
        assert isinstance(frame_desc.description, str)
        assert len(frame_desc.description) > 0

    def test_describe_frame_out_of_bounds(self, sample_video):
        """Test that out of bounds frame index raises error."""
        model = ImageToText(device="cpu")

        with pytest.raises(ValueError):
            model.describe_frame(sample_video, frame_index=1000)

        with pytest.raises(ValueError):
            model.describe_frame(sample_video, frame_index=-1)

    def test_describe_frames(self, sample_video):
        """Test describing multiple frames."""
        model = ImageToText(device="cpu")
        frame_indices = [0, 25, 50, 75]

        descriptions = model.describe_frames(sample_video, frame_indices)

        assert len(descriptions) == 4
        for i, idx in enumerate(frame_indices):
            assert descriptions[i].frame_index == idx
            assert descriptions[i].timestamp == pytest.approx(idx / 24.0)
            assert isinstance(descriptions[i].description, str)

    def test_describe_scene_default_fps(self, sample_video):
        """Test describing a scene with default 1 fps sampling."""
        scene = Scene(start=0.0, end=2.0, start_frame=0, end_frame=48)
        model = ImageToText(device="cpu")

        descriptions = model.describe_scene(sample_video, scene)

        # At 1 fps sampling, with 24 fps video, we sample every 24 frames
        # Scene is 48 frames, so we should get 2 descriptions (frame 0 and 24)
        assert len(descriptions) == 2
        assert descriptions[0].frame_index == 0
        assert descriptions[1].frame_index == 24

    def test_describe_scene_custom_fps(self, sample_video):
        """Test describing a scene with custom fps sampling."""
        scene = Scene(start=0.0, end=2.0, start_frame=0, end_frame=48)
        model = ImageToText(device="cpu")

        # Sample at 2 fps (every 12 frames with 24 fps video)
        descriptions = model.describe_scene(sample_video, scene, frames_per_second=2.0)

        # Should get frames: 0, 12, 24, 36
        assert len(descriptions) == 4
        assert descriptions[0].frame_index == 0
        assert descriptions[1].frame_index == 12
        assert descriptions[2].frame_index == 24
        assert descriptions[3].frame_index == 36

    def test_describe_scene_invalid_fps(self, sample_video):
        """Test that invalid fps raises error."""
        scene = Scene(start=0.0, end=2.0, start_frame=0, end_frame=48)
        model = ImageToText(device="cpu")

        with pytest.raises(ValueError):
            model.describe_scene(sample_video, scene, frames_per_second=0)

        with pytest.raises(ValueError):
            model.describe_scene(sample_video, scene, frames_per_second=-1)

    def test_describe_scene_single_frame(self, sample_video):
        """Test describing a very short scene with only one frame."""
        scene = Scene(start=0.0, end=0.04, start_frame=0, end_frame=1)
        model = ImageToText(device="cpu")

        descriptions = model.describe_scene(sample_video, scene)

        # Even with a single-frame scene, we should get at least one description
        assert len(descriptions) >= 1
        assert descriptions[0].frame_index == 0
