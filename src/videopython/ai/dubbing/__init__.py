"""Video dubbing functionality with multi-backend support."""

from videopython.ai.dubbing.dubber import VideoDubber
from videopython.ai.dubbing.models import DubbingResult, RevoiceResult, SeparatedAudio, TranslatedSegment
from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
from videopython.ai.dubbing.timing import TimingSynchronizer

__all__ = [
    "VideoDubber",
    "DubbingResult",
    "RevoiceResult",
    "TranslatedSegment",
    "SeparatedAudio",
    "LocalDubbingPipeline",
    "TimingSynchronizer",
]
