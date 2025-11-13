import numpy as np
import pytest

from videopython.base.scenes import Scene
from videopython.base.video import Video


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


class TestSceneDetector:
    """Tests for SceneDetector class."""

    def test_detector_initialization(self):
        """Test detector initialization with valid parameters."""
        from videopython.ai.understanding.scenes import SceneDetector

        detector = SceneDetector(threshold=0.5, min_scene_length=1.0)
        assert detector.threshold == 0.5
        assert detector.min_scene_length == 1.0

    def test_detector_invalid_threshold(self):
        """Test that invalid threshold raises ValueError."""
        from videopython.ai.understanding.scenes import SceneDetector

        with pytest.raises(ValueError):
            SceneDetector(threshold=-0.1)
        with pytest.raises(ValueError):
            SceneDetector(threshold=1.5)

    def test_detector_invalid_min_scene_length(self):
        """Test that invalid min_scene_length raises ValueError."""
        from videopython.ai.understanding.scenes import SceneDetector

        with pytest.raises(ValueError):
            SceneDetector(min_scene_length=-1.0)

    def test_detect_single_frame_video(self):
        """Test scene detection on a single-frame video."""
        from videopython.ai.understanding.scenes import SceneDetector

        # Create a single-frame video
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        video = Video.from_frames(frame[np.newaxis, :, :, :], fps=24.0)

        detector = SceneDetector()
        scenes = detector.detect(video)

        assert len(scenes) == 1
        assert scenes[0].start == 0.0
        assert scenes[0].start_frame == 0
        assert scenes[0].end_frame == 1

    @pytest.mark.skip(reason="Video.from_frames doesn't handle empty videos")
    def test_detect_empty_video(self):
        """Test scene detection on an empty video."""
        from videopython.ai.understanding.scenes import SceneDetector

        # Create an empty video
        frames = np.empty((0, 100, 100, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=24.0)

        detector = SceneDetector()
        scenes = detector.detect(video)

        assert len(scenes) == 0

    def test_detect_uniform_video(self):
        """Test scene detection on a video with uniform frames (no scene changes)."""
        from videopython.ai.understanding.scenes import SceneDetector

        # Create a video with identical frames
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frames = np.repeat(frame[np.newaxis, :, :, :], 100, axis=0)
        video = Video.from_frames(frames, fps=24.0)

        detector = SceneDetector(threshold=0.3)
        scenes = detector.detect(video)

        # Should detect a single scene
        assert len(scenes) == 1
        assert scenes[0].start_frame == 0
        assert scenes[0].end_frame == 100

    def test_detect_two_scene_video(self):
        """Test scene detection on a video with two distinct scenes."""
        from videopython.ai.understanding.scenes import SceneDetector

        # Create first scene (red frames)
        scene1_frames = np.zeros((50, 100, 100, 3), dtype=np.uint8)
        scene1_frames[:, :, :, 0] = 255  # Red channel

        # Create second scene (blue frames)
        scene2_frames = np.zeros((50, 100, 100, 3), dtype=np.uint8)
        scene2_frames[:, :, :, 2] = 255  # Blue channel

        # Combine scenes
        frames = np.concatenate([scene1_frames, scene2_frames], axis=0)
        video = Video.from_frames(frames, fps=24.0)

        detector = SceneDetector(threshold=0.3, min_scene_length=0.1)
        scenes = detector.detect(video)

        # Should detect two scenes
        assert len(scenes) == 2
        assert scenes[0].start_frame == 0
        assert scenes[1].start_frame == 50
        assert scenes[1].end_frame == 100

    def test_merge_short_scenes(self):
        """Test that short scenes are merged."""
        from videopython.ai.understanding.scenes import SceneDetector

        # Create frames with rapid changes (each frame different color)
        frames = []
        for i in range(100):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[:, :, i % 3] = (i * 50) % 256
            frames.append(frame)
        frames = np.array(frames)
        video = Video.from_frames(frames, fps=24.0)

        # Use high threshold to detect many scenes, then merge short ones
        detector = SceneDetector(threshold=0.1, min_scene_length=1.0)
        scenes = detector.detect(video)

        # All scenes should be at least min_scene_length or be the last scene
        for i, scene in enumerate(scenes[:-1]):  # Check all but last
            assert scene.duration >= 1.0 or i == len(scenes) - 1

    def test_histogram_difference_identical_frames(self):
        """Test histogram difference for identical frames."""
        from videopython.ai.understanding.scenes import SceneDetector

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detector = SceneDetector()

        diff = detector._calculate_histogram_difference(frame, frame)
        assert diff < 0.01  # Should be very close to 0

    def test_histogram_difference_different_frames(self):
        """Test histogram difference for completely different frames."""
        from videopython.ai.understanding.scenes import SceneDetector

        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame1[:, :, 0] = 255  # Red

        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[:, :, 2] = 255  # Blue

        detector = SceneDetector()
        diff = detector._calculate_histogram_difference(frame1, frame2)

        # Should be noticeably different
        assert diff > 0.3
