import pytest

from videopython.base.frames import FrameDescription
from videopython.base.scene_description import SceneDescription
from videopython.base.scenes import Scene
from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video_description import VideoDescription


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
