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
from .exceptions import (
    AudioError,
    AudioLoadError,
    OutOfBoundsError,
    TextRenderError,
    TransformError,
    VideoError,
    VideoLoadError,
    VideoMetadataError,
    VideoPythonError,
)
from .image_text import AnchorPoint, ImageText, TextAlign
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
    "TextRenderError",
    "OutOfBoundsError",
    # Text rendering primitives
    "ImageText",
    "TextAlign",
    "AnchorPoint",
    # Transcription data classes
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
