from videopython._exceptions import (
    AudioError,
    AudioLoadError,
    TransformError,
    VideoError,
    VideoLoadError,
    VideoMetadataError,
    VideoPythonError,
)

from .description import (
    AudioClassification,
    AudioEvent,
    BoundingBox,
    DetectedFace,
    DetectedObject,
    DetectedText,
    FaceTrack,
    MotionInfo,
    SceneBoundary,
    SceneDescription,
)
from .draw_detections import DetectionStyle, class_color, draw_detections
from .transcription import Transcription, TranscriptionSegment, TranscriptionWord
from .video import FrameIterator, Video, VideoMetadata

__all__ = [
    # Core
    "Video",
    "VideoMetadata",
    "FrameIterator",
    # Exceptions
    "VideoPythonError",
    "VideoError",
    "VideoLoadError",
    "VideoMetadataError",
    "AudioError",
    "AudioLoadError",
    "TransformError",
    # Detection overlay renderer (AI-free)
    "draw_detections",
    "DetectionStyle",
    "class_color",
    # Transcription result models
    "Transcription",
    "TranscriptionSegment",
    "TranscriptionWord",
    # Detection / scene / motion result types (consumed by ai/, editing/)
    "BoundingBox",
    "DetectedObject",
    "DetectedFace",
    "DetectedText",
    "FaceTrack",
    "AudioEvent",
    "AudioClassification",
    "MotionInfo",
    "SceneBoundary",
    "SceneDescription",
]
