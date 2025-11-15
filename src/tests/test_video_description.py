import numpy as np
import pytest

from videopython.base.frames import FrameDescription
from videopython.base.scene_description import SceneDescription
from videopython.base.scenes import Scene
from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord
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


def test_video_description_creation():
    """Test basic VideoDescription creation."""
    scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=120)
    frame_desc = FrameDescription(frame_index=0, timestamp=0.0, description="Test scene")
    scene_desc = SceneDescription(scene=scene, frame_descriptions=[frame_desc])

    understanding = VideoDescription(scene_descriptions=[scene_desc], transcription=None)

    assert understanding.num_scenes == 1
    assert understanding.total_frames_analyzed == 1
    assert understanding.transcription is None


def test_video_description_with_transcription():
    """Test VideoDescription with transcription."""
    scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=120)
    frame_desc = FrameDescription(frame_index=0, timestamp=0.0, description="Test scene")
    scene_desc = SceneDescription(scene=scene, frame_descriptions=[frame_desc])

    word = TranscriptionWord(start=0.0, end=1.0, word="test")
    segment = TranscriptionSegment(start=0.0, end=1.0, text="test", words=[word])
    transcription = Transcription(segments=[segment])

    understanding = VideoDescription(scene_descriptions=[scene_desc], transcription=transcription)

    assert understanding.num_scenes == 1
    assert understanding.transcription is not None
    assert len(understanding.transcription.segments) == 1


def test_video_description_multiple_scenes():
    """Test VideoDescription with multiple scenes."""
    scene1 = Scene(start=0.0, end=2.0, start_frame=0, end_frame=48)
    frame_desc1 = FrameDescription(frame_index=0, timestamp=0.0, description="Scene 1")
    scene_desc1 = SceneDescription(scene=scene1, frame_descriptions=[frame_desc1])

    scene2 = Scene(start=2.0, end=4.0, start_frame=48, end_frame=96)
    frame_desc2 = FrameDescription(frame_index=48, timestamp=2.0, description="Scene 2")
    scene_desc2 = SceneDescription(scene=scene2, frame_descriptions=[frame_desc2])

    understanding = VideoDescription(scene_descriptions=[scene_desc1, scene_desc2])

    assert understanding.num_scenes == 2
    assert understanding.total_frames_analyzed == 2


def test_get_scene_summary():
    """Test getting summary of a specific scene."""
    scene = Scene(start=1.5, end=3.0, start_frame=36, end_frame=72)
    frame_desc = FrameDescription(frame_index=36, timestamp=1.5, description="A red car drives by.")
    scene_desc = SceneDescription(scene=scene, frame_descriptions=[frame_desc])

    understanding = VideoDescription(scene_descriptions=[scene_desc])

    summary = understanding.get_scene_summary(0)
    assert "Scene 1" in summary
    assert "1.50s" in summary
    assert "3.00s" in summary
    assert "A red car drives by." in summary


def test_get_scene_summary_invalid_index():
    """Test that invalid scene index raises error."""
    scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=120)
    frame_desc = FrameDescription(frame_index=0, timestamp=0.0, description="Test")
    scene_desc = SceneDescription(scene=scene, frame_descriptions=[frame_desc])

    understanding = VideoDescription(scene_descriptions=[scene_desc])

    with pytest.raises(ValueError):
        understanding.get_scene_summary(1)

    with pytest.raises(ValueError):
        understanding.get_scene_summary(-1)


def test_get_full_summary_without_transcription():
    """Test getting full summary without transcription."""
    scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=120)
    frame_desc = FrameDescription(frame_index=0, timestamp=0.0, description="Test scene")
    scene_desc = SceneDescription(scene=scene, frame_descriptions=[frame_desc])

    understanding = VideoDescription(scene_descriptions=[scene_desc])

    summary = understanding.get_full_summary()
    assert "Video Analysis" in summary
    assert "1 scenes" in summary
    assert "1 frames analyzed" in summary
    assert "Test scene" in summary
    assert "Transcription:" not in summary


def test_get_full_summary_with_transcription():
    """Test getting full summary with transcription."""
    scene = Scene(start=0.0, end=5.0, start_frame=0, end_frame=120)
    frame_desc = FrameDescription(frame_index=0, timestamp=0.0, description="Test scene")
    scene_desc = SceneDescription(scene=scene, frame_descriptions=[frame_desc])

    word = TranscriptionWord(start=0.0, end=1.0, word="hello")
    segment = TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])
    transcription = Transcription(segments=[segment])

    understanding = VideoDescription(scene_descriptions=[scene_desc], transcription=transcription)

    summary = understanding.get_full_summary()
    assert "Video Analysis" in summary
    assert "Transcription:" in summary
    assert "hello" in summary


def test_video_analyzer_initialization():
    """Test VideoAnalyzer initialization."""
    from videopython.ai.understanding.video import VideoAnalyzer

    analyzer = VideoAnalyzer(scene_threshold=0.5, min_scene_length=1.0, device="cpu")

    assert analyzer.scene_detector.threshold == 0.5
    assert analyzer.scene_detector.min_scene_length == 1.0
    assert analyzer.image_to_text.device == "cpu"


def test_video_analyzer_analyze_without_transcription(sample_video_for_analysis):
    """Test VideoAnalyzer.analyze without transcription."""
    from videopython.ai.understanding.video import VideoAnalyzer

    analyzer = VideoAnalyzer(scene_threshold=0.3, min_scene_length=0.5, device="cpu")
    understanding = analyzer.analyze(sample_video_for_analysis, frames_per_second=1.0, transcribe=False)

    assert isinstance(understanding, VideoDescription)
    assert understanding.num_scenes >= 1  # Should detect at least one scene
    assert understanding.total_frames_analyzed >= 1  # Should analyze at least one frame
    assert understanding.transcription is None


def test_video_analyzer_analyze_with_custom_fps(sample_video_for_analysis):
    """Test VideoAnalyzer.analyze with custom frame sampling rate."""
    from videopython.ai.understanding.video import VideoAnalyzer

    analyzer = VideoAnalyzer(device="cpu")
    understanding = analyzer.analyze(sample_video_for_analysis, frames_per_second=2.0, transcribe=False)

    assert isinstance(understanding, VideoDescription)
    # With 2 fps sampling, should analyze more frames than 1 fps
    assert understanding.total_frames_analyzed >= understanding.num_scenes


def test_video_analyzer_analyze_scenes_only(sample_video_for_analysis):
    """Test VideoAnalyzer.analyze_scenes_only convenience method."""
    from videopython.ai.understanding.video import VideoAnalyzer

    analyzer = VideoAnalyzer(device="cpu")
    scene_descriptions = analyzer.analyze_scenes_only(sample_video_for_analysis)

    assert isinstance(scene_descriptions, list)
    assert len(scene_descriptions) >= 1
    assert all(isinstance(sd, SceneDescription) for sd in scene_descriptions)


def test_video_analyzer_with_description_prompt(sample_video_for_analysis):
    """Test VideoAnalyzer with custom description prompt."""
    from videopython.ai.understanding.video import VideoAnalyzer

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
