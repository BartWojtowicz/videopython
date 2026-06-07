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
from .exceptions import (
    AudioError,
    AudioLoadError,
    OutOfBoundsError,
    PlanError,
    PlanErrorCode,
    PlanRepair,
    PlanValidationError,
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
    # Structured plan validation / repair
    "PlanError",
    "PlanErrorCode",
    "PlanValidationError",
    "PlanRepair",
    # Text rendering primitives
    "ImageText",
    "TextAlign",
    "AnchorPoint",
    # Detection overlay renderer (AI-free)
    "draw_detections",
    "DetectionStyle",
    "class_color",
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
