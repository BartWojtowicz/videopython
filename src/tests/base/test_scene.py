"""Tests for scene detection in base module."""

import numpy as np
import pytest

from tests.test_config import SMALL_VIDEO_PATH
from videopython.base.audio import Audio
from videopython.base.scene import SceneDetector
from videopython.base.video import Video


def _create_video(frames: np.ndarray, fps: float = 24.0) -> Video:
    """Helper to create a Video from numpy frames."""
    duration = len(frames) / fps
    if duration <= 0:
        duration = 0.1  # Minimum duration for empty video case
    audio = Audio.create_silent(duration_seconds=duration)
    return Video(frames=frames, fps=fps, audio=audio)


def _create_solid_frame(color: tuple[int, int, int], height: int = 100, width: int = 100) -> np.ndarray:
    """Create a solid color frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


class TestSceneDetectorInit:
    """Tests for SceneDetector initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        detector = SceneDetector()
        assert detector.threshold == 0.3
        assert detector.min_scene_length == 0.5

    def test_custom_values(self):
        """Test custom initialization values."""
        detector = SceneDetector(threshold=0.5, min_scene_length=1.0)
        assert detector.threshold == 0.5
        assert detector.min_scene_length == 1.0

    def test_invalid_threshold_high(self):
        """Test that threshold > 1.0 raises error."""
        with pytest.raises(ValueError, match="threshold must be between"):
            SceneDetector(threshold=1.5)

    def test_invalid_threshold_low(self):
        """Test that threshold < 0.0 raises error."""
        with pytest.raises(ValueError, match="threshold must be between"):
            SceneDetector(threshold=-0.1)

    def test_invalid_min_scene_length(self):
        """Test that negative min_scene_length raises error."""
        with pytest.raises(ValueError, match="min_scene_length must be non-negative"):
            SceneDetector(min_scene_length=-1.0)


class TestSceneDetection:
    """Tests for scene detection functionality."""

    def test_empty_video(self):
        """Test detection on empty video."""
        frames = np.zeros((0, 100, 100, 3), dtype=np.uint8)
        video = _create_video(frames)

        detector = SceneDetector()
        scenes = detector.detect(video)

        assert scenes == []

    def test_single_frame(self):
        """Test detection on single-frame video."""
        frame = _create_solid_frame((255, 0, 0))
        frames = np.array([frame])
        video = _create_video(frames, fps=24.0)

        detector = SceneDetector()
        scenes = detector.detect(video)

        assert len(scenes) == 1
        assert scenes[0].start == 0.0
        assert scenes[0].start_frame == 0
        assert scenes[0].end_frame == 1

    def test_no_scene_change(self):
        """Test detection on video with no scene changes."""
        frame = _create_solid_frame((255, 0, 0))
        frames = np.array([frame] * 48)  # 2 seconds at 24fps
        video = _create_video(frames, fps=24.0)

        detector = SceneDetector()
        scenes = detector.detect(video)

        # Should be one continuous scene
        assert len(scenes) == 1
        assert scenes[0].start == 0.0
        assert scenes[0].start_frame == 0
        assert scenes[0].end_frame == 48

    def test_clear_cut(self):
        """Test detection of a clear visual cut."""
        # Red frame
        red_frame = _create_solid_frame((255, 0, 0))
        # Blue frame
        blue_frame = _create_solid_frame((0, 0, 255))

        # 1 second red, then 1 second blue
        red_frames = [red_frame] * 24
        blue_frames = [blue_frame] * 24
        frames = np.array(red_frames + blue_frames)
        video = _create_video(frames, fps=24.0)

        detector = SceneDetector(threshold=0.3, min_scene_length=0.0)
        scenes = detector.detect(video)

        # Should detect two scenes
        assert len(scenes) == 2
        assert scenes[0].start_frame == 0
        assert scenes[1].start_frame == 24

    def test_scene_timestamps(self):
        """Test that scene timestamps are calculated correctly."""
        frame = _create_solid_frame((255, 0, 0))
        frames = np.array([frame] * 48)  # 2 seconds at 24fps
        video = _create_video(frames, fps=24.0)

        detector = SceneDetector()
        scenes = detector.detect(video)

        assert len(scenes) == 1
        assert scenes[0].start == 0.0
        assert abs(scenes[0].end - 2.0) < 0.01  # 2 seconds
        assert abs(scenes[0].duration - 2.0) < 0.01


class TestSceneMerging:
    """Tests for short scene merging."""

    def test_merge_short_scenes(self):
        """Test that short scenes are merged."""
        red = _create_solid_frame((255, 0, 0))
        green = _create_solid_frame((0, 255, 0))
        blue = _create_solid_frame((0, 0, 255))

        # Create: 1s red, 0.2s green (short), 1s blue
        fps = 24.0
        red_frames = [red] * 24  # 1 second
        green_frames = [green] * 5  # ~0.2 seconds
        blue_frames = [blue] * 24  # 1 second

        frames = np.array(red_frames + green_frames + blue_frames)
        video = _create_video(frames, fps=fps)

        # With min_scene_length=0.5, the green scene should be merged
        detector = SceneDetector(threshold=0.3, min_scene_length=0.5)
        scenes = detector.detect(video)

        # Should have 2 scenes (green merged with something)
        assert len(scenes) == 2

    def test_no_merge_when_disabled(self):
        """Test that scenes aren't merged when min_scene_length=0."""
        red = _create_solid_frame((255, 0, 0))
        blue = _create_solid_frame((0, 0, 255))

        # Create alternating frames
        frames = np.array([red, blue, red, blue] * 12)  # 48 frames
        video = _create_video(frames, fps=24.0)

        detector = SceneDetector(threshold=0.3, min_scene_length=0.0)
        scenes = detector.detect(video)

        # Should detect many scene changes
        assert len(scenes) > 2


class TestHistogramDifference:
    """Tests for histogram difference calculation."""

    def test_identical_frames(self):
        """Test that identical frames have zero difference."""
        frame = _create_solid_frame((128, 128, 128))
        detector = SceneDetector()

        diff = detector._calculate_histogram_difference(frame, frame)

        assert diff < 0.01  # Should be very close to 0

    def test_completely_different_frames(self):
        """Test that completely different frames have high difference."""
        white = _create_solid_frame((255, 255, 255))
        black = _create_solid_frame((0, 0, 0))
        detector = SceneDetector()

        diff = detector._calculate_histogram_difference(white, black)

        # Histogram comparison gives moderate difference for white/black
        # because both have low saturation (grayscale)
        assert diff > 0.3  # Should be noticeably different


class TestSceneBoundaryProperties:
    """Test that returned SceneBoundary objects have correct properties."""

    def test_scene_boundary_fields(self):
        """Test that SceneBoundary objects have all required fields."""
        frame = _create_solid_frame((255, 0, 0))
        frames = np.array([frame] * 48)
        video = _create_video(frames, fps=24.0)

        detector = SceneDetector()
        scenes = detector.detect(video)

        assert len(scenes) == 1
        scene = scenes[0]

        # Check all fields are present
        assert hasattr(scene, "start")
        assert hasattr(scene, "end")
        assert hasattr(scene, "start_frame")
        assert hasattr(scene, "end_frame")
        assert hasattr(scene, "duration")

        # Check values make sense
        assert scene.start >= 0
        assert scene.end > scene.start
        assert scene.start_frame >= 0
        assert scene.end_frame > scene.start_frame
        assert scene.duration > 0


class TestStreamingSceneDetection:
    """Tests for streaming scene detection methods."""

    def test_detect_streaming_basic(self):
        """Test that streaming detection works on real video file."""
        detector = SceneDetector(threshold=0.3, min_scene_length=0.5)
        scenes = detector.detect_streaming(SMALL_VIDEO_PATH)

        # Should detect at least one scene
        assert len(scenes) >= 1
        # First scene should start at 0
        assert scenes[0].start == 0.0
        assert scenes[0].start_frame == 0

    def test_detect_streaming_matches_detect(self):
        """Test that streaming detection produces similar results as batch."""
        # Load video fully for comparison
        video = Video.from_path(SMALL_VIDEO_PATH)

        detector = SceneDetector(threshold=0.3, min_scene_length=0.0)

        # Compare batch vs streaming
        batch_scenes = detector.detect(video)
        streaming_scenes = detector.detect_streaming(SMALL_VIDEO_PATH)

        # Should have similar number of scenes (may differ slightly due to frame count)
        assert abs(len(batch_scenes) - len(streaming_scenes)) <= 1

        # First scene boundaries should match
        if batch_scenes and streaming_scenes:
            assert batch_scenes[0].start_frame == streaming_scenes[0].start_frame

    def test_detect_from_path_classmethod(self):
        """Test convenience class method."""
        scenes = SceneDetector.detect_from_path(SMALL_VIDEO_PATH, threshold=0.3)

        assert len(scenes) >= 1
        assert scenes[0].start == 0.0
        assert scenes[0].start_frame == 0

    def test_detect_from_path_custom_params(self):
        """Test class method with custom parameters."""
        scenes_sensitive = SceneDetector.detect_from_path(SMALL_VIDEO_PATH, threshold=0.1, min_scene_length=0.0)
        scenes_less_sensitive = SceneDetector.detect_from_path(SMALL_VIDEO_PATH, threshold=0.5, min_scene_length=0.5)

        # More sensitive threshold should detect more or equal scenes
        assert len(scenes_sensitive) >= len(scenes_less_sensitive)

    def test_detect_streaming_scene_properties(self):
        """Test that streaming scenes have correct properties."""
        detector = SceneDetector(threshold=0.3)
        scenes = detector.detect_streaming(SMALL_VIDEO_PATH)

        for scene in scenes:
            # Check all fields are present and valid
            assert hasattr(scene, "start")
            assert hasattr(scene, "end")
            assert hasattr(scene, "start_frame")
            assert hasattr(scene, "end_frame")
            assert hasattr(scene, "duration")

            assert scene.start >= 0
            assert scene.end > scene.start
            assert scene.start_frame >= 0
            assert scene.end_frame > scene.start_frame
            assert scene.duration > 0

    def test_detect_streaming_file_not_found(self):
        """Test that non-existent file raises error."""
        detector = SceneDetector()
        with pytest.raises(FileNotFoundError):
            detector.detect_streaming("/nonexistent/path/video.mp4")

    def test_detect_from_path_file_not_found(self):
        """Test that class method raises for non-existent file."""
        with pytest.raises(FileNotFoundError):
            SceneDetector.detect_from_path("/nonexistent/path/video.mp4")
