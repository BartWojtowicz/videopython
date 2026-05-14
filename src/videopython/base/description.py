from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BoundingBox",
    "DetectedObject",
    "DetectedFace",
    "DetectedText",
    "AudioEvent",
    "AudioClassification",
    "MotionInfo",
    "SceneBoundary",
    "SceneDescription",
    "FaceTrack",
]


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


class BoundingBox(BaseModel):
    """A bounding box for detected objects or crop regions in an image.

    Coordinates are normalized to ``[0, 1]`` relative to image dimensions.
    Promoted to a Pydantic model so it can be embedded directly into
    ``Operation`` fields (e.g. ``KenBurns.start_region``) and validated /
    serialised as part of an op's JSON wire format.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(description="Left edge of the box, 0=left of the image.")
    y: float = Field(description="Top edge of the box, 0=top of the image.")
    width: float = Field(description="Width of the box, normalized to image width.")
    height: float = Field(description="Height of the box, normalized to image height.")

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        """Area of the bounding box (normalized)."""
        return self.width * self.height

    def to_dict(self) -> dict:
        """Backwards-compat alias for ``model_dump()``."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> BoundingBox:
        """Backwards-compat alias for ``model_validate(data)``."""
        return cls.model_validate(data)


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
class DetectedText:
    """Text detected in a video frame.

    Attributes:
        text: OCR text content
        confidence: Detection confidence score between 0 and 1
        bounding_box: Optional normalized bounding box for the text region
    """

    text: str
    confidence: float
    bounding_box: BoundingBox | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DetectedText":
        """Create DetectedText from dictionary."""
        return cls(
            text=data["text"],
            confidence=data["confidence"],
            bounding_box=BoundingBox.from_dict(data["bounding_box"]) if data.get("bounding_box") else None,
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

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "events": [event.to_dict() for event in self.events],
            "clip_predictions": self.clip_predictions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AudioClassification":
        """Create AudioClassification from dictionary."""
        return cls(
            events=[AudioEvent.from_dict(event) for event in data.get("events", [])],
            clip_predictions={k: float(v) for k, v in data.get("clip_predictions", {}).items()},
        )


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


@dataclass
class SceneDescription:
    """Structured visual scene description from the SceneVLM.

    The v1 schema is intentionally narrow (caption + subjects + shot_type).
    Wider schemas drop JSON parse rate on small models without eval data
    to defend the cost. Fields are added in v2 as parse-rate measurements
    justify them; closed enums first, open lists last.

    Attributes:
        caption: One-sentence summary of the scene.
        subjects: Open list of named subjects visible in the frames.
        shot_type: Closed enum framing the camera distance, or None
            when JSON parsing fell back to raw text.
    """

    caption: str
    subjects: list[str] = field(default_factory=list)
    shot_type: str | None = None

    def to_dict(self) -> dict:
        return {
            "caption": self.caption,
            "subjects": list(self.subjects),
            "shot_type": self.shot_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneDescription":
        return cls(
            caption=str(data["caption"]),
            subjects=[str(s) for s in data.get("subjects", [])],
            shot_type=data.get("shot_type"),
        )


@dataclass
class FaceTrack:
    """A face tracked across consecutive frames within a single shot.

    Tracks are produced by IoU association — no embedding re-id, so a
    track does not survive across shot/scene boundaries. ``frame_indices``
    and ``boxes`` are parallel lists of equal length.

    Attributes:
        track_id: Stable id within the shot the track was produced in.
            Not globally unique across scenes.
        frame_indices: Source-video frame indices for each detection.
        boxes: Per-frame bounding boxes (normalized 0-1 coords).
        confidences: Per-frame detection confidence in [0, 1].
    """

    track_id: int
    frame_indices: list[int]
    boxes: list[BoundingBox]
    confidences: list[float] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Number of frames in this track."""
        return len(self.frame_indices)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "frame_indices": list(self.frame_indices),
            "boxes": [box.to_dict() for box in self.boxes],
            "confidences": list(self.confidences),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FaceTrack":
        return cls(
            track_id=int(data["track_id"]),
            frame_indices=[int(i) for i in data.get("frame_indices", [])],
            boxes=[BoundingBox.from_dict(b) for b in data.get("boxes", [])],
            confidences=[float(c) for c in data.get("confidences", [])],
        )
