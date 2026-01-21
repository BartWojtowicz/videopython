import pytest

from videopython.base.description import (
    BoundingBox,
    DetectedFace,
    DetectedObject,
    FrameDescription,
    MotionInfo,
    SceneDescription,
)
from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord

# Note: Transcription.slice() tests are in test_transcription.py


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_bounding_box_creation(self):
        """Test basic bounding box creation."""
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        assert bbox.x == 0.1
        assert bbox.y == 0.2
        assert bbox.width == 0.3
        assert bbox.height == 0.4

    def test_bounding_box_center(self):
        """Test center property calculation."""
        bbox = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5)
        assert bbox.center == (0.25, 0.25)

        bbox2 = BoundingBox(x=0.2, y=0.3, width=0.4, height=0.2)
        assert bbox2.center == (0.4, 0.4)

    def test_bounding_box_area(self):
        """Test area property calculation."""
        bbox = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5)
        assert bbox.area == 0.25

        bbox2 = BoundingBox(x=0.1, y=0.1, width=0.2, height=0.3)
        assert bbox2.area == pytest.approx(0.06)


class TestDetectedObject:
    """Tests for DetectedObject dataclass."""

    def test_detected_object_creation(self):
        """Test basic detected object creation."""
        obj = DetectedObject(label="person", confidence=0.95)
        assert obj.label == "person"
        assert obj.confidence == 0.95
        assert obj.bounding_box is None

    def test_detected_object_with_bounding_box(self):
        """Test detected object with bounding box."""
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        obj = DetectedObject(label="car", confidence=0.87, bounding_box=bbox)
        assert obj.label == "car"
        assert obj.confidence == 0.87
        assert obj.bounding_box is not None
        assert obj.bounding_box.x == 0.1


class TestFrameDescription:
    """Tests for FrameDescription dataclass."""

    def test_frame_description_creation(self):
        """Test basic frame description creation."""
        fd = FrameDescription(frame_index=42, timestamp=1.75, description="A dog playing in a park")
        assert fd.frame_index == 42
        assert fd.timestamp == 1.75
        assert fd.description == "A dog playing in a park"
        # Optional fields should default to None
        assert fd.detected_objects is None
        assert fd.detected_text is None
        assert fd.detected_faces is None
        assert fd.shot_type is None
        assert fd.motion is None

    def test_frame_description_with_detected_objects(self):
        """Test frame description with detected objects."""
        objects = [
            DetectedObject(label="person", confidence=0.95),
            DetectedObject(label="dog", confidence=0.88, bounding_box=BoundingBox(0.5, 0.5, 0.2, 0.3)),
        ]
        fd = FrameDescription(
            frame_index=10,
            timestamp=0.5,
            description="A person with a dog",
            detected_objects=objects,
        )
        assert fd.detected_objects is not None
        assert len(fd.detected_objects) == 2
        assert fd.detected_objects[0].label == "person"
        assert fd.detected_objects[1].label == "dog"
        assert fd.detected_objects[1].bounding_box is not None

    def test_frame_description_with_ocr(self):
        """Test frame description with OCR text."""
        fd = FrameDescription(
            frame_index=20,
            timestamp=1.0,
            description="A sign on a building",
            detected_text=["STOP", "Main Street"],
        )
        assert fd.detected_text is not None
        assert len(fd.detected_text) == 2
        assert "STOP" in fd.detected_text

    def test_frame_description_with_faces(self):
        """Test frame description with detected faces."""
        faces = [
            DetectedFace(bounding_box=BoundingBox(0.1, 0.1, 0.2, 0.2), confidence=0.95),
            DetectedFace(bounding_box=BoundingBox(0.5, 0.3, 0.15, 0.18), confidence=0.88),
            DetectedFace(),  # Face without bounding box (e.g., from cloud backend)
        ]
        fd = FrameDescription(
            frame_index=30,
            timestamp=1.5,
            description="A group of people",
            detected_faces=faces,
        )
        assert fd.detected_faces is not None
        assert len(fd.detected_faces) == 3
        assert fd.detected_faces[0].bounding_box is not None
        assert fd.detected_faces[0].confidence == 0.95
        assert fd.detected_faces[2].bounding_box is None  # Cloud backend face

    def test_frame_description_with_shot_analysis(self):
        """Test frame description with shot type and motion info."""
        motion = MotionInfo(motion_type="static", magnitude=0.05, raw_magnitude=1.2)
        fd = FrameDescription(
            frame_index=40,
            timestamp=2.0,
            description="Close-up of a face",
            shot_type="close-up",
            motion=motion,
        )
        assert fd.shot_type == "close-up"
        assert fd.motion is not None
        assert fd.motion.motion_type == "static"
        assert fd.motion.is_static

    def test_frame_description_fully_populated(self):
        """Test frame description with all fields populated."""
        objects = [DetectedObject(label="car", confidence=0.92)]
        faces = [DetectedFace(bounding_box=BoundingBox(0.3, 0.2, 0.1, 0.15))]
        motion = MotionInfo(motion_type="pan", magnitude=0.4, raw_magnitude=8.5)
        fd = FrameDescription(
            frame_index=50,
            timestamp=2.5,
            description="A car on a highway",
            detected_objects=objects,
            detected_text=["EXIT 42"],
            detected_faces=faces,
            shot_type="wide",
            motion=motion,
        )
        assert fd.frame_index == 50
        assert fd.detected_objects is not None
        assert fd.detected_text == ["EXIT 42"]
        assert fd.detected_faces is not None
        assert len(fd.detected_faces) == 1
        assert fd.shot_type == "wide"
        assert fd.motion is not None
        assert fd.motion.motion_type == "pan"


class TestSceneDescription:
    """Tests for SceneDescription dataclass."""

    def test_scene_description_creation(self):
        """Test basic scene description creation."""
        frame_descriptions = [
            FrameDescription(frame_index=0, timestamp=0.0, description="Scene start"),
            FrameDescription(frame_index=60, timestamp=2.5, description="Scene middle"),
            FrameDescription(frame_index=119, timestamp=4.96, description="Scene end"),
        ]
        sd = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120, frame_descriptions=frame_descriptions)

        assert sd.start == 0.0
        assert sd.end == 5.0
        assert sd.start_frame == 0
        assert sd.end_frame == 120
        assert len(sd.frame_descriptions) == 3
        assert sd.num_frames_described == 3

    def test_duration_property(self):
        """Test duration calculation."""
        sd = SceneDescription(start=1.5, end=4.0, start_frame=36, end_frame=96)
        assert sd.duration == 2.5

    def test_frame_count_property(self):
        """Test frame count calculation."""
        sd = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120)
        assert sd.frame_count == 120

    def test_get_frame_indices(self):
        """Test getting evenly distributed frame indices."""
        sd = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=100)

        # Single sample should return middle frame
        indices = sd.get_frame_indices(num_samples=1)
        assert indices == [50]

        # Multiple samples should be evenly spaced
        indices = sd.get_frame_indices(num_samples=3)
        assert len(indices) == 3
        assert indices[0] == 0  # start
        assert indices[-1] == 99  # end (approximately)

    def test_get_frame_indices_invalid(self):
        """Test that invalid num_samples raises error."""
        sd = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120)
        with pytest.raises(ValueError):
            sd.get_frame_indices(num_samples=0)

    def test_get_description_summary(self):
        """Test getting summary of all descriptions."""
        frame_descriptions = [
            FrameDescription(frame_index=0, timestamp=0.0, description="A red car."),
            FrameDescription(frame_index=60, timestamp=2.5, description="The car drives away."),
        ]
        sd = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120, frame_descriptions=frame_descriptions)

        summary = sd.get_description_summary()
        assert summary == "A red car. The car drives away."

    def test_get_transcription_text(self):
        """Test getting transcription text for a scene."""
        word1 = TranscriptionWord(start=0.0, end=0.5, word="hello")
        word2 = TranscriptionWord(start=0.5, end=1.0, word="world")
        segment = TranscriptionSegment(start=0.0, end=1.0, text="hello world", words=[word1, word2])
        transcription = Transcription(segments=[segment])

        sd = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120, transcription=transcription)

        assert sd.get_transcription_text() == "hello world"

    def test_get_transcription_text_empty(self):
        """Test getting transcription text when no transcription exists."""
        sd = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120)
        assert sd.get_transcription_text() == ""


class TestSerialization:
    """Tests for to_dict/from_dict serialization methods."""

    def test_bounding_box_roundtrip(self):
        """Test BoundingBox serialization roundtrip."""
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        data = bbox.to_dict()
        restored = BoundingBox.from_dict(data)

        assert restored.x == bbox.x
        assert restored.y == bbox.y
        assert restored.width == bbox.width
        assert restored.height == bbox.height

    def test_detected_object_roundtrip(self):
        """Test DetectedObject serialization roundtrip."""
        obj = DetectedObject(
            label="person",
            confidence=0.95,
            bounding_box=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
        )
        data = obj.to_dict()
        restored = DetectedObject.from_dict(data)

        assert restored.label == obj.label
        assert restored.confidence == obj.confidence
        assert restored.bounding_box is not None
        assert restored.bounding_box.x == obj.bounding_box.x

    def test_detected_object_without_bbox(self):
        """Test DetectedObject serialization without bounding box."""
        obj = DetectedObject(label="car", confidence=0.8)
        data = obj.to_dict()
        restored = DetectedObject.from_dict(data)

        assert restored.label == obj.label
        assert restored.bounding_box is None

    def test_detected_face_roundtrip(self):
        """Test DetectedFace serialization roundtrip."""
        face = DetectedFace(
            bounding_box=BoundingBox(x=0.2, y=0.3, width=0.1, height=0.15),
            confidence=0.92,
        )
        data = face.to_dict()
        restored = DetectedFace.from_dict(data)

        assert restored.confidence == face.confidence
        assert restored.bounding_box is not None
        assert restored.bounding_box.width == face.bounding_box.width

    def test_motion_info_roundtrip(self):
        """Test MotionInfo serialization roundtrip."""
        motion = MotionInfo(motion_type="pan", magnitude=0.5, raw_magnitude=10.0)
        data = motion.to_dict()
        restored = MotionInfo.from_dict(data)

        assert restored.motion_type == motion.motion_type
        assert restored.magnitude == motion.magnitude
        assert restored.raw_magnitude == motion.raw_magnitude

    def test_frame_description_roundtrip(self):
        """Test FrameDescription serialization roundtrip."""
        fd = FrameDescription(
            frame_index=42,
            timestamp=1.75,
            description="A person walking",
            detected_objects=[
                DetectedObject(label="person", confidence=0.9),
                DetectedObject(label="dog", confidence=0.85, bounding_box=BoundingBox(0.5, 0.5, 0.2, 0.3)),
            ],
            detected_text=["STOP", "Main St"],
            detected_faces=[DetectedFace(bounding_box=BoundingBox(0.1, 0.1, 0.2, 0.2))],
            shot_type="medium",
            motion=MotionInfo(motion_type="static", magnitude=0.1, raw_magnitude=2.0),
        )
        data = fd.to_dict()
        restored = FrameDescription.from_dict(data)

        assert restored.frame_index == fd.frame_index
        assert restored.timestamp == fd.timestamp
        assert restored.description == fd.description
        assert restored.detected_text == fd.detected_text
        assert restored.shot_type == fd.shot_type
        assert len(restored.detected_objects) == 2
        assert restored.detected_objects[0].label == "person"
        assert len(restored.detected_faces) == 1
        assert restored.motion.motion_type == "static"

    def test_scene_description_roundtrip(self):
        """Test SceneDescription serialization roundtrip."""
        frame_descriptions = [
            FrameDescription(frame_index=0, timestamp=0.0, description="Scene start"),
            FrameDescription(frame_index=60, timestamp=2.5, description="Scene middle"),
        ]

        word = TranscriptionWord(start=0.0, end=0.5, word="hello")
        segment = TranscriptionSegment(start=0.0, end=0.5, text="hello", words=[word])
        transcription = Transcription(segments=[segment])

        sd = SceneDescription(
            start=0.0,
            end=5.0,
            start_frame=0,
            end_frame=120,
            frame_descriptions=frame_descriptions,
            transcription=transcription,
            summary="A short scene",
            scene_type="dialogue",
            detected_entities=["person", "car"],
            dominant_colors=[(255, 0, 0), (0, 255, 0)],
            avg_motion_magnitude=0.3,
            dominant_motion_type="pan",
        )
        data = sd.to_dict()
        restored = SceneDescription.from_dict(data)

        assert restored.start == sd.start
        assert restored.end == sd.end
        assert restored.start_frame == sd.start_frame
        assert restored.end_frame == sd.end_frame
        assert len(restored.frame_descriptions) == 2
        assert restored.frame_descriptions[0].description == "Scene start"
        assert restored.transcription is not None
        assert restored.transcription.segments[0].text == "hello"
        assert restored.summary == sd.summary
        assert restored.scene_type == sd.scene_type
        assert restored.detected_entities == sd.detected_entities
        assert restored.dominant_colors == [(255, 0, 0), (0, 255, 0)]
        assert restored.avg_motion_magnitude == sd.avg_motion_magnitude
        assert restored.dominant_motion_type == sd.dominant_motion_type

    def test_scene_description_minimal_roundtrip(self):
        """Test SceneDescription serialization with minimal fields."""
        sd = SceneDescription(start=0.0, end=5.0, start_frame=0, end_frame=120)
        data = sd.to_dict()
        restored = SceneDescription.from_dict(data)

        assert restored.start == sd.start
        assert restored.end == sd.end
        assert restored.transcription is None
        assert restored.summary is None
        assert restored.detected_entities is None
