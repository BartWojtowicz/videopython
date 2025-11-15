import numpy as np
import pytest

from videopython.ai.understanding.video import VideoAnalyzer
from videopython.base.scene_description import SceneDescription
from videopython.base.video import Video
from videopython.base.video_description import VideoDescription


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
