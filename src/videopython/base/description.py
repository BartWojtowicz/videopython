from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "BoundingBox",
    "DetectedObject",
    "DetectedFace",
    "AudioEvent",
    "AudioClassification",
    "MotionInfo",
    "SceneBoundary",
    "DetectedAction",
]


@dataclass
class DetectedAction:
    """An action/activity detected in a video segment.

    Attributes:
        label: Name of the detected action (e.g., "walking", "running", "dancing")
        confidence: Detection confidence score between 0 and 1
        start_frame: Start frame index of the action
        end_frame: End frame index of the action (exclusive)
        start_time: Start time in seconds
        end_time: End time in seconds
    """

    label: str
    confidence: float
    start_frame: int | None = None
    end_frame: int | None = None
    start_time: float | None = None
    end_time: float | None = None

    @property
    def duration(self) -> float | None:
        """Duration of the action in seconds."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DetectedAction":
        """Create DetectedAction from dictionary."""
        return cls(
            label=data["label"],
            confidence=data["confidence"],
            start_frame=data.get("start_frame"),
            end_frame=data.get("end_frame"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
        )


@dataclass
class SceneBoundary:
    """Timing information for a detected scene.

    A lightweight structure representing scene boundaries detected by SceneDetector.
    This is a backbone type - higher-level scene analysis belongs in orchestration packages.

    Attributes:
        start: Scene start time in seconds
        end: Scene end time in seconds
        start_frame: Index of the first frame in this scene
        end_frame: Index of the last frame in this scene (exclusive)
    """

    start: float
    end: float
    start_frame: int
    end_frame: int

    @property
    def duration(self) -> float:
        """Duration of the scene in seconds."""
        return self.end - self.start

    @property
    def frame_count(self) -> int:
        """Number of frames in this scene."""
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start": self.start,
            "end": self.end,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneBoundary":
        """Create SceneBoundary from dictionary."""
        return cls(
            start=data["start"],
            end=data["end"],
            start_frame=data["start_frame"],
            end_frame=data["end_frame"],
        )


@dataclass
class BoundingBox:
    """A bounding box for detected objects in an image.

    Coordinates are normalized to [0, 1] range relative to image dimensions.

    Attributes:
        x: Left edge of the box (0 = left edge of image)
        y: Top edge of the box (0 = top edge of image)
        width: Width of the box
        height: Height of the box
    """

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        """Area of the bounding box (normalized)."""
        return self.width * self.height

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict) -> BoundingBox:
        """Create BoundingBox from dictionary."""
        return cls(x=data["x"], y=data["y"], width=data["width"], height=data["height"])


@dataclass
class DetectedObject:
    """An object detected in a video frame.

    Attributes:
        label: Name/class of the detected object (e.g., "person", "car", "dog")
        confidence: Detection confidence score between 0 and 1
        bounding_box: Optional bounding box location of the object
    """

    label: str
    confidence: float
    bounding_box: BoundingBox | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DetectedObject:
        """Create DetectedObject from dictionary."""
        return cls(
            label=data["label"],
            confidence=data["confidence"],
            bounding_box=BoundingBox.from_dict(data["bounding_box"]) if data.get("bounding_box") else None,
        )


@dataclass
class DetectedFace:
    """A face detected in a video frame.

    Attributes:
        bounding_box: Bounding box location of the face (normalized 0-1 coordinates).
            May be None for cloud backends that only return face counts.
        confidence: Detection confidence score between 0 and 1
    """

    bounding_box: BoundingBox | None = None
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float] | None:
        """Center point of the face bounding box, or None if no bounding box."""
        return self.bounding_box.center if self.bounding_box else None

    @property
    def area(self) -> float | None:
        """Area of the face bounding box (normalized), or None if no bounding box."""
        return self.bounding_box.area if self.bounding_box else None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DetectedFace:
        """Create DetectedFace from dictionary."""
        return cls(
            bounding_box=BoundingBox.from_dict(data["bounding_box"]) if data.get("bounding_box") else None,
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class AudioEvent:
    """A detected audio event with timestamp.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        label: Name of the detected sound (e.g., "Music", "Speech", "Dog bark")
        confidence: Detection confidence score between 0 and 1
    """

    start: float
    end: float
    label: str
    confidence: float

    @property
    def duration(self) -> float:
        """Duration of the audio event in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AudioEvent:
        """Create AudioEvent from dictionary."""
        return cls(
            start=data["start"],
            end=data["end"],
            label=data["label"],
            confidence=data["confidence"],
        )


@dataclass
class AudioClassification:
    """Complete audio classification results.

    Attributes:
        events: List of detected audio events with timestamps
        clip_predictions: Overall class probabilities for the entire audio clip
    """

    events: list[AudioEvent]
    clip_predictions: dict[str, float] = field(default_factory=dict)


@dataclass
class MotionInfo:
    """Motion characteristics between consecutive frames.

    Attributes:
        motion_type: Classification of camera/scene motion
            - "static": No significant motion
            - "pan": Horizontal camera movement
            - "tilt": Vertical camera movement
            - "zoom": Camera zoom in/out
            - "complex": Mixed or irregular motion
        magnitude: Normalized motion magnitude (0.0 = no motion, 1.0 = high motion)
        raw_magnitude: Raw optical flow magnitude (pixels/frame)
    """

    motion_type: str
    magnitude: float
    raw_magnitude: float

    @property
    def is_static(self) -> bool:
        """Check if this frame has no significant motion."""
        return self.motion_type == "static"

    @property
    def is_dynamic(self) -> bool:
        """Check if this frame has significant motion."""
        return self.motion_type != "static"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "motion_type": self.motion_type,
            "magnitude": self.magnitude,
            "raw_magnitude": self.raw_magnitude,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MotionInfo:
        """Create MotionInfo from dictionary."""
        return cls(
            motion_type=data["motion_type"],
            magnitude=data["magnitude"],
            raw_magnitude=data["raw_magnitude"],
        )
