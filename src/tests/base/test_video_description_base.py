import pytest

from videopython.base.description import FrameDescription, SceneDescription, VideoDescription
from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord


def test_video_description_creation():
    """Test basic VideoDescription creation."""
    frame_desc = FrameDescription(frame_index=0, timestamp=0.0, description="Test scene")
    scene_desc = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120, frame_descriptions=[frame_desc])

    understanding = VideoDescription(scene_descriptions=[scene_desc], transcription=None)

    assert understanding.num_scenes == 1
    assert understanding.total_frames_analyzed == 1
    assert understanding.transcription is None


def test_video_description_with_transcription():
    """Test VideoDescription with transcription."""
    frame_desc = FrameDescription(frame_index=0, timestamp=0.0, description="Test scene")
    scene_desc = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120, frame_descriptions=[frame_desc])

    word = TranscriptionWord(start=0.0, end=1.0, word="test")
    segment = TranscriptionSegment(start=0.0, end=1.0, text="test", words=[word])
    transcription = Transcription(segments=[segment])

    understanding = VideoDescription(scene_descriptions=[scene_desc], transcription=transcription)

    assert understanding.num_scenes == 1
    assert understanding.transcription is not None
    assert len(understanding.transcription.segments) == 1


def test_video_description_multiple_scenes():
    """Test VideoDescription with multiple scenes."""
    frame_desc1 = FrameDescription(frame_index=0, timestamp=0.0, description="Scene 1")
    scene_desc1 = SceneDescription(start=0.0, end=2.0, start_frame=0, end_frame=48, frame_descriptions=[frame_desc1])

    frame_desc2 = FrameDescription(frame_index=48, timestamp=2.0, description="Scene 2")
    scene_desc2 = SceneDescription(start=2.0, end=4.0, start_frame=48, end_frame=96, frame_descriptions=[frame_desc2])

    understanding = VideoDescription(scene_descriptions=[scene_desc1, scene_desc2])

    assert understanding.num_scenes == 2
    assert understanding.total_frames_analyzed == 2


def test_get_scene_summary():
    """Test getting summary of a specific scene."""
    frame_desc = FrameDescription(frame_index=36, timestamp=1.5, description="A red car drives by.")
    scene_desc = SceneDescription(start=1.5, end=3.0, start_frame=36, end_frame=72, frame_descriptions=[frame_desc])

    understanding = VideoDescription(scene_descriptions=[scene_desc])

    summary = understanding.get_scene_summary(0)
    assert "Scene 1" in summary
    assert "1.50s" in summary
    assert "3.00s" in summary
    assert "A red car drives by." in summary


def test_get_scene_summary_invalid_index():
    """Test that invalid scene index raises error."""
    frame_desc = FrameDescription(frame_index=0, timestamp=0.0, description="Test")
    scene_desc = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120, frame_descriptions=[frame_desc])

    understanding = VideoDescription(scene_descriptions=[scene_desc])

    with pytest.raises(ValueError):
        understanding.get_scene_summary(1)

    with pytest.raises(ValueError):
        understanding.get_scene_summary(-1)


def test_get_full_summary_without_transcription():
    """Test getting full summary without transcription."""
    frame_desc = FrameDescription(frame_index=0, timestamp=0.0, description="Test scene")
    scene_desc = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120, frame_descriptions=[frame_desc])

    understanding = VideoDescription(scene_descriptions=[scene_desc])

    summary = understanding.get_full_summary()
    assert "Video Analysis" in summary
    assert "1 scenes" in summary
    assert "1 frames analyzed" in summary
    assert "Test scene" in summary
    assert "Transcription:" not in summary


def test_get_full_summary_with_transcription():
    """Test getting full summary with transcription."""
    frame_desc = FrameDescription(frame_index=0, timestamp=0.0, description="Test scene")
    scene_desc = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120, frame_descriptions=[frame_desc])

    word = TranscriptionWord(start=0.0, end=1.0, word="hello")
    segment = TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])
    transcription = Transcription(segments=[segment])

    understanding = VideoDescription(scene_descriptions=[scene_desc], transcription=transcription)

    summary = understanding.get_full_summary()
    assert "Video Analysis" in summary
    assert "Full Transcription:" in summary
    assert "hello" in summary


def test_distribute_transcription():
    """Test distributing transcription to scenes."""
    # Create two scenes
    frame_desc1 = FrameDescription(frame_index=0, timestamp=0.0, description="Scene 1")
    scene_desc1 = SceneDescription(start=0.0, end=2.0, start_frame=0, end_frame=48, frame_descriptions=[frame_desc1])

    frame_desc2 = FrameDescription(frame_index=48, timestamp=2.0, description="Scene 2")
    scene_desc2 = SceneDescription(start=2.0, end=4.0, start_frame=48, end_frame=96, frame_descriptions=[frame_desc2])

    # Create transcription with segments in different time ranges
    word1 = TranscriptionWord(start=0.5, end=1.0, word="first")
    segment1 = TranscriptionSegment(start=0.5, end=1.0, text="first", words=[word1])

    word2 = TranscriptionWord(start=2.5, end=3.0, word="second")
    segment2 = TranscriptionSegment(start=2.5, end=3.0, text="second", words=[word2])

    transcription = Transcription(segments=[segment1, segment2])

    understanding = VideoDescription(scene_descriptions=[scene_desc1, scene_desc2], transcription=transcription)

    # Initially, scenes have no transcription
    assert scene_desc1.transcription is None
    assert scene_desc2.transcription is None

    # Distribute transcription
    understanding.distribute_transcription()

    # Now each scene should have its relevant transcription
    assert scene_desc1.transcription is not None
    assert len(scene_desc1.transcription.segments) == 1
    assert scene_desc1.transcription.segments[0].text == "first"

    assert scene_desc2.transcription is not None
    assert len(scene_desc2.transcription.segments) == 1
    assert scene_desc2.transcription.segments[0].text == "second"


def test_get_scene_summary_with_scene_transcription():
    """Test that scene summary includes scene-level transcription."""
    word = TranscriptionWord(start=1.5, end=2.0, word="hello")
    segment = TranscriptionSegment(start=1.5, end=2.0, text="hello", words=[word])
    scene_transcription = Transcription(segments=[segment])

    frame_desc = FrameDescription(frame_index=36, timestamp=1.5, description="A person speaking.")
    scene_desc = SceneDescription(
        start=1.5,
        end=3.0,
        start_frame=36,
        end_frame=72,
        frame_descriptions=[frame_desc],
        transcription=scene_transcription,
    )

    understanding = VideoDescription(scene_descriptions=[scene_desc])

    summary = understanding.get_scene_summary(0)
    assert "A person speaking." in summary
    assert "[Speech: hello]" in summary


def test_video_description_roundtrip():
    """Test VideoDescription serialization roundtrip."""
    frame_desc1 = FrameDescription(frame_index=0, timestamp=0.0, description="Scene 1 start")
    frame_desc2 = FrameDescription(frame_index=60, timestamp=2.5, description="Scene 1 end")
    scene_desc1 = SceneDescription(
        start=0.0,
        end=5.0,
        start_frame=0,
        end_frame=120,
        frame_descriptions=[frame_desc1, frame_desc2],
        summary="First scene",
    )

    frame_desc3 = FrameDescription(frame_index=120, timestamp=5.0, description="Scene 2")
    scene_desc2 = SceneDescription(
        start=5.0,
        end=10.0,
        start_frame=120,
        end_frame=240,
        frame_descriptions=[frame_desc3],
    )

    word = TranscriptionWord(start=0.0, end=1.0, word="hello")
    segment = TranscriptionSegment(start=0.0, end=1.0, text="hello", words=[word])
    transcription = Transcription(segments=[segment])

    understanding = VideoDescription(
        scene_descriptions=[scene_desc1, scene_desc2],
        transcription=transcription,
    )

    data = understanding.to_dict()
    restored = VideoDescription.from_dict(data)

    assert restored.num_scenes == 2
    assert restored.total_frames_analyzed == 3
    assert restored.transcription is not None
    assert restored.transcription.segments[0].text == "hello"
    assert restored.scene_descriptions[0].summary == "First scene"
    assert restored.scene_descriptions[0].frame_descriptions[0].description == "Scene 1 start"


def test_video_description_minimal_roundtrip():
    """Test VideoDescription serialization with minimal data."""
    scene_desc = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120)
    understanding = VideoDescription(scene_descriptions=[scene_desc])

    data = understanding.to_dict()
    restored = VideoDescription.from_dict(data)

    assert restored.num_scenes == 1
    assert restored.transcription is None
    assert len(restored.scene_descriptions[0].frame_descriptions) == 0


def test_scene_description_key_frame_roundtrip():
    """Test SceneDescription key frame serialization roundtrip."""
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    scene_desc = SceneDescription(
        start=0.0,
        end=5.0,
        start_frame=0,
        end_frame=120,
        key_frame=jpeg_bytes,
        key_frame_timestamp=2.5,
    )

    data = scene_desc.to_dict()
    assert data["key_frame_base64"] is not None
    assert data["key_frame_timestamp"] == 2.5

    restored = SceneDescription.from_dict(data)
    assert restored.key_frame == jpeg_bytes
    assert restored.key_frame_timestamp == 2.5


def test_scene_description_key_frame_none_roundtrip():
    """Test SceneDescription serialization with no key frame."""
    scene_desc = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120)

    data = scene_desc.to_dict()
    assert data["key_frame_base64"] is None
    assert data["key_frame_timestamp"] is None

    restored = SceneDescription.from_dict(data)
    assert restored.key_frame is None
    assert restored.key_frame_timestamp is None
