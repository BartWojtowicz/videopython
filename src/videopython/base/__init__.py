from .audio import Audio, AudioMetadata
from .description import (
    AudioClassification,
    AudioEvent,
    BoundingBox,
    DetectedAction,
    DetectedFace,
    DetectedObject,
    MotionInfo,
    SceneBoundary,
)
from .edit import EffectApplication, SegmentConfig, VideoEdit
from .effects import Blur, ColorGrading, Effect, FullImageOverlay, KenBurns, Vignette, Zoom
from .exceptions import (
    AudioError,
    AudioLoadError,
    IncompatibleVideoError,
    InsufficientDurationError,
    OutOfBoundsError,
    TextRenderError,
    TransformError,
    VideoError,
    VideoLoadError,
    VideoMetadataError,
    VideoPythonError,
)
from .progress import configure, set_progress, set_verbose
from .registry import (
    OperationCategory,
    OperationSpec,
    ParamSpec,
    get_operation_spec,
    get_operation_specs,
    get_specs_by_category,
    get_specs_by_tag,
    register,
    spec_from_class,
)
from .scene import SceneDetector
from .text import (
    AnchorPoint,
    ImageText,
    TextAlign,
    Transcription,
    TranscriptionOverlay,
    TranscriptionSegment,
    TranscriptionWord,
)
from .transforms import (
    Crop,
    CropMode,
    CutFrames,
    CutSeconds,
    PictureInPicture,
    ResampleFPS,
    Resize,
    SpeedChange,
    Transformation,
)
from .transitions import BlurTransition, FadeTransition, InstantTransition, Transition
from .video import FrameIterator, Video, VideoMetadata

__all__ = [
    # Editing
    "VideoEdit",
    "SegmentConfig",
    "EffectApplication",
    # Core
    "Video",
    "VideoMetadata",
    "FrameIterator",
    # Audio
    "Audio",
    "AudioMetadata",
    # Exceptions
    "VideoPythonError",
    "VideoError",
    "VideoLoadError",
    "VideoMetadataError",
    "AudioError",
    "AudioLoadError",
    "TransformError",
    "InsufficientDurationError",
    "IncompatibleVideoError",
    "TextRenderError",
    "OutOfBoundsError",
    # Transforms
    "Transformation",
    "CutFrames",
    "CutSeconds",
    "Resize",
    "ResampleFPS",
    "Crop",
    "CropMode",
    "SpeedChange",
    "PictureInPicture",
    # Transitions
    "Transition",
    "InstantTransition",
    "FadeTransition",
    "BlurTransition",
    # Effects
    "Effect",
    "FullImageOverlay",
    "Blur",
    "Zoom",
    "ColorGrading",
    "Vignette",
    "KenBurns",
    # Text & Transcription
    "Transcription",
    "TranscriptionSegment",
    "TranscriptionWord",
    "TranscriptionOverlay",
    "ImageText",
    "TextAlign",
    "AnchorPoint",
    # Scene Detection
    "SceneDetector",
    "SceneBoundary",
    # Detection types
    "BoundingBox",
    "DetectedObject",
    "DetectedFace",
    "DetectedAction",
    "AudioEvent",
    "AudioClassification",
    "MotionInfo",
    # Registry
    "OperationCategory",
    "OperationSpec",
    "ParamSpec",
    "get_operation_specs",
    "get_operation_spec",
    "get_specs_by_category",
    "get_specs_by_tag",
    "register",
    "spec_from_class",
    # Configuration
    "configure",
    "set_verbose",
    "set_progress",
]
