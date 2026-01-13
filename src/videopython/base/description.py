from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from videopython.base.text.transcription import Transcription

__all__ = [
    "FrameDescription",
    "SceneDescription",
    "VideoDescription",
    "ColorHistogram",
    "BoundingBox",
    "DetectedObject",
    "AudioEvent",
    "AudioClassification",
    "MotionInfo",
]


@dataclass
class ColorHistogram:
    """Color features extracted from a video frame.

    Attributes:
        dominant_colors: Top N dominant colors in RGB format (0-255)
        avg_hue: Average hue value (0-180 in OpenCV HSV)
        avg_saturation: Average saturation value (0-255)
        avg_value: Average value/brightness (0-255)
        hsv_histogram: Optional full HSV histogram for advanced analysis
    """

    dominant_colors: list[tuple[int, int, int]]
    avg_hue: float
    avg_saturation: float
    avg_value: float
    hsv_histogram: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


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
        detected_faces: Optional count of faces detected in the frame
        shot_type: Optional shot classification (e.g., "close-up", "medium", "wide")
        camera_motion: Optional camera motion type (e.g., "static", "pan", "tilt", "zoom")
        motion: Optional motion info with type and magnitude
    """

    frame_index: int
    timestamp: float
    description: str
    color_histogram: ColorHistogram | None = None
    detected_objects: list[DetectedObject] | None = None
    detected_text: list[str] | None = None
    detected_faces: int | None = None
    shot_type: str | None = None
    camera_motion: str | None = None
    motion: MotionInfo | None = None


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
