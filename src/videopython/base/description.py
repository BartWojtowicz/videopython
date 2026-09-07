from __future__ import annotations

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


class SceneBoundary(BaseModel):
    """Timing information for a detected scene.

    A lightweight structure representing scene boundaries returned by
    scene detectors (e.g. ``videopython.ai.SemanticSceneDetector``). This
    is a backbone type — higher-level scene analysis lives in orchestration
    packages.

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


class BoundingBox(BaseModel):
    """A bounding box for detected objects or crop regions in an image.

    Coordinates are normalized to ``[0, 1]`` relative to image dimensions.
    It can be embedded directly into ``Operation`` fields (for example,
    ``KenBurns.start_region``) and validated as part of an operation's JSON
    wire format.
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


class DetectedObject(BaseModel):
    """An object detected in a video frame.

    Attributes:
        label: Name/class of the detected object (e.g., "person", "car", "dog")
        confidence: Detection confidence score between 0 and 1
        bounding_box: Optional bounding box location of the object
    """

    label: str
    confidence: float
    bounding_box: BoundingBox | None = None


class DetectedFace(BaseModel):
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


class DetectedText(BaseModel):
    """Text detected in a video frame.

    Attributes:
        text: OCR text content
        confidence: Detection confidence score between 0 and 1
        bounding_box: Optional normalized bounding box for the text region
    """

    text: str
    confidence: float
    bounding_box: BoundingBox | None = None


class AudioEvent(BaseModel):
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


class AudioClassification(BaseModel):
    """Complete audio classification results.

    Attributes:
        events: List of detected audio events with timestamps
        clip_predictions: Overall class probabilities for the entire audio clip
    """

    events: list[AudioEvent]
    clip_predictions: dict[str, float] = Field(default_factory=dict)


class MotionInfo(BaseModel):
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


class SceneDescription(BaseModel):
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
    subjects: list[str] = Field(default_factory=list)
    shot_type: str | None = None


class FaceTrack(BaseModel):
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
    confidences: list[float] = Field(default_factory=list)

    @property
    def length(self) -> int:
        """Number of frames in this track."""
        return len(self.frame_indices)
