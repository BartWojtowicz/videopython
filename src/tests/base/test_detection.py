"""Tests for detection dataclasses in videopython.base."""

import pytest

from videopython.base.description import AudioClassification, AudioEvent, BoundingBox, DetectedObject


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_bounding_box_creation(self):
        """Test BoundingBox can be created with valid values."""
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        assert bbox.x == 0.1
        assert bbox.y == 0.2
        assert bbox.width == 0.3
        assert bbox.height == 0.4

    def test_bounding_box_center(self):
        """Test BoundingBox center calculation."""
        bbox = BoundingBox(x=0.1, y=0.2, width=0.2, height=0.4)
        cx, cy = bbox.center
        assert cx == pytest.approx(0.2)  # 0.1 + 0.2/2
        assert cy == pytest.approx(0.4)  # 0.2 + 0.4/2

    def test_bounding_box_area(self):
        """Test BoundingBox area calculation."""
        bbox = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.4)
        assert bbox.area == pytest.approx(0.2)  # 0.5 * 0.4


class TestDetectedObject:
    """Tests for DetectedObject dataclass."""

    def test_detected_object_creation(self):
        """Test DetectedObject can be created."""
        obj = DetectedObject(label="person", confidence=0.95)
        assert obj.label == "person"
        assert obj.confidence == 0.95
        assert obj.bounding_box is None

    def test_detected_object_with_bbox(self):
        """Test DetectedObject with bounding box."""
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        obj = DetectedObject(label="car", confidence=0.8, bounding_box=bbox)
        assert obj.label == "car"
        assert obj.bounding_box is not None
        assert obj.bounding_box.x == 0.1


class TestAudioClassification:
    """Tests for AudioClassification serialization helpers."""

    def test_roundtrip_dict_serialization(self):
        """AudioClassification should roundtrip via to_dict/from_dict."""
        original = AudioClassification(
            events=[
                AudioEvent(start=0.0, end=1.2, label="Speech", confidence=0.93),
                AudioEvent(start=1.3, end=2.0, label="Music", confidence=0.78),
            ],
            clip_predictions={"Speech": 0.93, "Music": 0.78},
        )

        restored = AudioClassification.from_dict(original.to_dict())
        assert restored == original
