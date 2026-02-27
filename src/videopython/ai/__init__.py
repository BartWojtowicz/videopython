from videopython.ai import registry as _ai_registry  # noqa: F401

from .generation import ImageToVideo, TextToImage, TextToMusic, TextToSpeech, TextToVideo
from .swapping import ObjectSwapper
from .transforms import FaceTracker, FaceTrackingCrop, SplitScreenComposite
from .understanding import (
    ActionRecognizer,
    AudioClassifier,
    AudioToText,
    CameraMotionDetector,
    FaceDetector,
    ImageToText,
    MotionAnalyzer,
    ObjectDetector,
    SemanticSceneDetector,
    TextDetector,
)
from .video_analysis import VideoAnalysis, VideoAnalysisConfig, VideoAnalyzer

__all__ = [
    # Generation
    "TextToVideo",
    "ImageToVideo",
    "TextToImage",
    "TextToSpeech",
    "TextToMusic",
    # Understanding
    "AudioToText",
    "AudioClassifier",
    "ImageToText",
    # Detection
    "ObjectDetector",
    "FaceDetector",
    "TextDetector",
    "CameraMotionDetector",
    # Motion
    "MotionAnalyzer",
    # Temporal
    "ActionRecognizer",
    "SemanticSceneDetector",
    # Transforms (AI-powered)
    "FaceTracker",
    "FaceTrackingCrop",
    "SplitScreenComposite",
    # Swapping
    "ObjectSwapper",
    # Video analysis
    "VideoAnalysis",
    "VideoAnalysisConfig",
    "VideoAnalyzer",
]
