from __future__ import annotations

import base64
from dataclasses import dataclass, field

from videopython.base.text.transcription import Transcription

__all__ = [
    "FrameDescription",
    "SceneDescription",
    "VideoDescription",
    "ColorHistogram",
    "BoundingBox",
    "DetectedObject",
    "DetectedFace",
    "AudioEvent",
    "AudioClassification",
    "MotionInfo",
    "DetectedAction",
]


@dataclass
class ColorHistogram:
    """Color features extracted from a video frame.

    Attributes:
        dominant_colors: Top N dominant colors in RGB format (0-255)
        avg_hue: Average hue value (0-180 in OpenCV HSV)
        avg_saturation: Average saturation value (0-255)
        avg_value: Average value/brightness (0-255)
    """

    dominant_colors: list[tuple[int, int, int]]
    avg_hue: float
    avg_saturation: float
    avg_value: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "dominant_colors": [list(c) for c in self.dominant_colors],
            "avg_hue": self.avg_hue,
            "avg_saturation": self.avg_saturation,
            "avg_value": self.avg_value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ColorHistogram:
        """Create ColorHistogram from dictionary."""
        return cls(
            dominant_colors=[tuple(c) for c in data["dominant_colors"]],  # type: ignore[misc]
            avg_hue=data["avg_hue"],
            avg_saturation=data["avg_saturation"],
            avg_value=data["avg_value"],
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
    def from_dict(cls, data: dict) -> DetectedAction:
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
class FrameDescription:
    """Represents a description of a video frame.

    Attributes:
        frame_index: Index of the frame in the video
        timestamp: Time in seconds when this frame appears
        description: Text description of what's in the frame
        color_histogram: Optional color features extracted from the frame
        detected_objects: Optional list of objects detected in the frame
        detected_text: Optional list of text strings found via OCR
        detected_faces: Optional list of faces detected in the frame with bounding boxes
        shot_type: Optional shot classification (e.g., "close-up", "medium", "wide")
        motion: Optional motion info with type and magnitude (includes camera motion type)
    """

    frame_index: int
    timestamp: float
    description: str
    color_histogram: ColorHistogram | None = None
    detected_objects: list[DetectedObject] | None = None
    detected_text: list[str] | None = None
    detected_faces: list[DetectedFace] | None = None
    shot_type: str | None = None
    motion: MotionInfo | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "description": self.description,
            "color_histogram": self.color_histogram.to_dict() if self.color_histogram else None,
            "detected_objects": [obj.to_dict() for obj in self.detected_objects] if self.detected_objects else None,
            "detected_text": self.detected_text,
            "detected_faces": [face.to_dict() for face in self.detected_faces] if self.detected_faces else None,
            "shot_type": self.shot_type,
            "motion": self.motion.to_dict() if self.motion else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FrameDescription:
        """Create FrameDescription from dictionary."""
        return cls(
            frame_index=data["frame_index"],
            timestamp=data["timestamp"],
            description=data["description"],
            color_histogram=ColorHistogram.from_dict(data["color_histogram"]) if data.get("color_histogram") else None,
            detected_objects=[DetectedObject.from_dict(obj) for obj in data["detected_objects"]]
            if data.get("detected_objects")
            else None,
            detected_text=data.get("detected_text"),
            detected_faces=[DetectedFace.from_dict(face) for face in data["detected_faces"]]
            if data.get("detected_faces")
            else None,
            shot_type=data.get("shot_type"),
            motion=MotionInfo.from_dict(data["motion"]) if data.get("motion") else None,
        )


@dataclass
class SceneDescription:
    """A self-contained description of a video scene.

    A scene is a continuous segment of video where the visual content remains relatively consistent,
    bounded by scene changes or transitions. This class combines timing information with
    visual analysis, transcription, and other metadata.

    Attributes:
        start: Scene start time in seconds
        end: Scene end time in seconds
        start_frame: Index of the first frame in this scene
        end_frame: Index of the last frame in this scene (exclusive)
        frame_descriptions: List of descriptions for frames sampled from this scene
        transcription: Optional transcription of speech within this scene
        summary: Optional LLM-generated summary of the scene
        scene_type: Optional classification (e.g., "dialogue", "action", "transition")
        detected_entities: Optional list of entities/objects detected in the scene
        dominant_colors: Optional dominant colors aggregated across the scene
        audio_events: Optional list of audio events detected in this scene
        avg_motion_magnitude: Optional average motion magnitude across the scene (0.0-1.0)
        dominant_motion_type: Optional most common motion type in the scene
    """

    start: float
    end: float
    start_frame: int
    end_frame: int
    frame_descriptions: list[FrameDescription] = field(default_factory=list)
    transcription: Transcription | None = None
    summary: str | None = None
    scene_type: str | None = None
    detected_entities: list[str] | None = None
    dominant_colors: list[tuple[int, int, int]] | None = None
    audio_events: list[AudioEvent] | None = None
    avg_motion_magnitude: float | None = None
    dominant_motion_type: str | None = None
    detected_actions: list[DetectedAction] | None = None
    key_frame: bytes | None = None
    key_frame_timestamp: float | None = None

    @property
    def duration(self) -> float:
        """Duration of the scene in seconds."""
        return self.end - self.start

    @property
    def frame_count(self) -> int:
        """Number of frames in this scene."""
        return self.end_frame - self.start_frame

    @property
    def num_frames_described(self) -> int:
        """Number of frames that were described in this scene."""
        return len(self.frame_descriptions)

    def get_frame_indices(self, num_samples: int = 3) -> list[int]:
        """Get evenly distributed frame indices from this scene.

        Args:
            num_samples: Number of frames to sample from the scene

        Returns:
            List of frame indices evenly distributed throughout the scene
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        if num_samples == 1:
            # Return middle frame
            return [self.start_frame + self.frame_count // 2]

        # Get evenly spaced frames including start and end
        step = (self.end_frame - self.start_frame - 1) / (num_samples - 1)
        return [int(self.start_frame + i * step) for i in range(num_samples)]

    def get_description_summary(self) -> str:
        """Get a summary of all frame descriptions concatenated.

        Returns:
            Single string with all frame descriptions joined
        """
        return " ".join([fd.description for fd in self.frame_descriptions])

    def get_transcription_text(self) -> str:
        """Get the full transcription text for this scene.

        Returns:
            Concatenated transcription text, or empty string if no transcription
        """
        if not self.transcription or not self.transcription.segments:
            return ""
        return " ".join(segment.text for segment in self.transcription.segments)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start": self.start,
            "end": self.end,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_descriptions": [fd.to_dict() for fd in self.frame_descriptions],
            "transcription": self.transcription.to_dict() if self.transcription else None,
            "summary": self.summary,
            "scene_type": self.scene_type,
            "detected_entities": self.detected_entities,
            "dominant_colors": [list(c) for c in self.dominant_colors] if self.dominant_colors else None,
            "audio_events": [ae.to_dict() for ae in self.audio_events] if self.audio_events else None,
            "avg_motion_magnitude": self.avg_motion_magnitude,
            "dominant_motion_type": self.dominant_motion_type,
            "detected_actions": [da.to_dict() for da in self.detected_actions] if self.detected_actions else None,
            "key_frame_base64": base64.b64encode(self.key_frame).decode("utf-8") if self.key_frame else None,
            "key_frame_timestamp": self.key_frame_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SceneDescription:
        """Create SceneDescription from dictionary."""
        return cls(
            start=data["start"],
            end=data["end"],
            start_frame=data["start_frame"],
            end_frame=data["end_frame"],
            frame_descriptions=[FrameDescription.from_dict(fd) for fd in data.get("frame_descriptions", [])],
            transcription=Transcription.from_dict(data["transcription"]) if data.get("transcription") else None,
            summary=data.get("summary"),
            scene_type=data.get("scene_type"),
            detected_entities=data.get("detected_entities"),
            dominant_colors=[tuple(c) for c in data["dominant_colors"]] if data.get("dominant_colors") else None,  # type: ignore[misc]
            audio_events=[AudioEvent.from_dict(ae) for ae in data["audio_events"]]
            if data.get("audio_events")
            else None,
            avg_motion_magnitude=data.get("avg_motion_magnitude"),
            dominant_motion_type=data.get("dominant_motion_type"),
            detected_actions=[DetectedAction.from_dict(da) for da in data["detected_actions"]]
            if data.get("detected_actions")
            else None,
            key_frame=base64.b64decode(data["key_frame_base64"]) if data.get("key_frame_base64") else None,
            key_frame_timestamp=data.get("key_frame_timestamp"),
        )


@dataclass
class VideoDescription:
    """Complete understanding of a video including visual and audio analysis.

    Attributes:
        scene_descriptions: List of scene descriptions with frame analysis and per-scene transcription
        transcription: Optional full audio transcription for the entire video
    """

    scene_descriptions: list[SceneDescription]
    transcription: Transcription | None = None

    @property
    def num_scenes(self) -> int:
        """Number of scenes detected in the video."""
        return len(self.scene_descriptions)

    @property
    def total_frames_analyzed(self) -> int:
        """Total number of frames analyzed across all scenes."""
        return sum(sd.num_frames_described for sd in self.scene_descriptions)

    def distribute_transcription(self) -> None:
        """Distribute the video-level transcription to each scene.

        Slices the full transcription at word-level granularity and assigns
        relevant words/segments to each SceneDescription based on time overlap.
        Modifies scene_descriptions in place.
        """
        if not self.transcription:
            return

        for sd in self.scene_descriptions:
            sd.transcription = self.transcription.slice(sd.start, sd.end)

    def get_scene_summary(self, scene_index: int) -> str:
        """Get a text summary of a specific scene.

        Args:
            scene_index: Index of the scene to summarize

        Returns:
            Text summary of the scene including timing, descriptions, and transcription
        """
        if scene_index < 0 or scene_index >= len(self.scene_descriptions):
            raise ValueError(f"scene_index {scene_index} out of bounds (0-{len(self.scene_descriptions) - 1})")

        sd = self.scene_descriptions[scene_index]
        summary = f"Scene {scene_index + 1} ({sd.start:.2f}s - {sd.end:.2f}s, {sd.duration:.2f}s): "
        summary += sd.get_description_summary()

        # Include scene-level transcription if available
        scene_transcript = sd.get_transcription_text()
        if scene_transcript:
            summary += f" [Speech: {scene_transcript}]"

        return summary

    def get_full_summary(self) -> str:
        """Get a complete text summary of the entire video.

        Returns:
            Multi-line string with scene summaries and optional transcription
        """
        lines = [f"Video Analysis - {self.num_scenes} scenes, {self.total_frames_analyzed} frames analyzed\n"]

        for i in range(len(self.scene_descriptions)):
            lines.append(self.get_scene_summary(i))

        if self.transcription and self.transcription.segments:
            lines.append("\nFull Transcription:")
            for segment in self.transcription.segments:
                lines.append(f"  [{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scene_descriptions": [sd.to_dict() for sd in self.scene_descriptions],
            "transcription": self.transcription.to_dict() if self.transcription else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VideoDescription:
        """Create VideoDescription from dictionary."""
        return cls(
            scene_descriptions=[SceneDescription.from_dict(sd) for sd in data["scene_descriptions"]],
            transcription=Transcription.from_dict(data["transcription"]) if data.get("transcription") else None,
        )
