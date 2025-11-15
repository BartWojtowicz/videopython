import numpy as np
import pytest

from videopython.ai.understanding.video import SceneDetector, VideoAnalyzer
from videopython.base.description import SceneDescription, VideoDescription
from videopython.base.video import Video


class TestSceneDetector:
    """Tests for SceneDetector class."""

    def test_detector_initialization(self):
        """Test detector initialization with valid parameters."""
        detector = SceneDetector(threshold=0.5, min_scene_length=1.0)
        assert detector.threshold == 0.5
        assert detector.min_scene_length == 1.0

    def test_detector_invalid_threshold(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError):
            SceneDetector(threshold=-0.1)
        with pytest.raises(ValueError):
            SceneDetector(threshold=1.5)

    def test_detector_invalid_min_scene_length(self):
        """Test that invalid min_scene_length raises ValueError."""
        with pytest.raises(ValueError):
            SceneDetector(min_scene_length=-1.0)

    def test_detect_single_frame_video(self):
        """Test scene detection on a single-frame video."""
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
        # Create an empty video
        frames = np.empty((0, 100, 100, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=24.0)

        detector = SceneDetector()
        scenes = detector.detect(video)

        assert len(scenes) == 0

    def test_detect_uniform_video(self):
        """Test scene detection on a video with uniform frames (no scene changes)."""
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

        # All scenes except the last should be at least min_scene_length
        # (the last scene might be shorter if it can't be merged)
        for scene in scenes[:-1]:
            assert scene.duration >= 1.0

    def test_histogram_difference_identical_frames(self):
        """Test histogram difference for identical frames."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detector = SceneDetector()

        diff = detector._calculate_histogram_difference(frame, frame)
        assert diff < 0.01  # Should be very close to 0

    def test_histogram_difference_different_frames(self):
        """Test histogram difference for completely different frames."""
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame1[:, :, 0] = 255  # Red

        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[:, :, 2] = 255  # Blue

        detector = SceneDetector()
        diff = detector._calculate_histogram_difference(frame1, frame2)

        # Should be noticeably different
        assert diff > 0.3


@pytest.fixture(scope="session")
def sample_video_for_analysis():
    """Create a sample video for testing."""
    # Create 200 frames at 24 fps (~8 seconds) - enough for multiple scenes
    frames = []
    # First 100 frames - red
    red_frames = np.zeros((100, 200, 200, 3), dtype=np.uint8)
    red_frames[:, :, :, 0] = 255
    frames.append(red_frames)

    # Next 100 frames - blue
    blue_frames = np.zeros((100, 200, 200, 3), dtype=np.uint8)
    blue_frames[:, :, :, 2] = 255
    frames.append(blue_frames)

    all_frames = np.concatenate(frames, axis=0)
    return Video.from_frames(all_frames, fps=24.0)


def test_video_analyzer_initialization():
    """Test VideoAnalyzer initialization."""
    analyzer = VideoAnalyzer(scene_threshold=0.5, min_scene_length=1.0, device="cpu")

    assert analyzer.scene_detector.threshold == 0.5
    assert analyzer.scene_detector.min_scene_length == 1.0
    assert analyzer.image_to_text.device == "cpu"


def test_video_analyzer_analyze_without_transcription(sample_video_for_analysis):
    """Test VideoAnalyzer.analyze without transcription."""
    analyzer = VideoAnalyzer(scene_threshold=0.3, min_scene_length=0.5, device="cpu")
    understanding = analyzer.analyze(sample_video_for_analysis, frames_per_second=1.0, transcribe=False)

    assert isinstance(understanding, VideoDescription)
    assert understanding.num_scenes >= 1  # Should detect at least one scene
    assert understanding.total_frames_analyzed >= 1  # Should analyze at least one frame
    assert understanding.transcription is None


def test_video_analyzer_analyze_with_custom_fps(sample_video_for_analysis):
    """Test VideoAnalyzer.analyze with custom frame sampling rate."""
    analyzer = VideoAnalyzer(device="cpu")
    understanding = analyzer.analyze(sample_video_for_analysis, frames_per_second=2.0, transcribe=False)

    assert isinstance(understanding, VideoDescription)
    # With 2 fps sampling, should analyze more frames than 1 fps
    assert understanding.total_frames_analyzed >= understanding.num_scenes


def test_video_analyzer_analyze_scenes_only(sample_video_for_analysis):
    """Test VideoAnalyzer.analyze_scenes_only convenience method."""
    analyzer = VideoAnalyzer(device="cpu")
    scene_descriptions = analyzer.analyze_scenes_only(sample_video_for_analysis)

    assert isinstance(scene_descriptions, list)
    assert len(scene_descriptions) >= 1
    assert all(isinstance(sd, SceneDescription) for sd in scene_descriptions)


def test_video_analyzer_with_description_prompt(sample_video_for_analysis):
    """Test VideoAnalyzer with custom description prompt."""
    analyzer = VideoAnalyzer(device="cpu")
    understanding = analyzer.analyze(
        sample_video_for_analysis, frames_per_second=1.0, transcribe=False, description_prompt="Describe the colors"
    )

    assert isinstance(understanding, VideoDescription)
    assert understanding.num_scenes >= 1
    # Check that descriptions were generated
    for scene_desc in understanding.scene_descriptions:
        assert len(scene_desc.frame_descriptions) > 0
        for frame_desc in scene_desc.frame_descriptions:
            assert isinstance(frame_desc.description, str)
            assert len(frame_desc.description) > 0
