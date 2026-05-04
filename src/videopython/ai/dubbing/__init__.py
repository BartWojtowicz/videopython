"""Local video dubbing functionality."""

from videopython.ai.dubbing.dubber import VideoDubber
from videopython.ai.dubbing.models import DubbingResult, RevoiceResult, SeparatedAudio, TranslatedSegment
from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
from videopython.ai.dubbing.quality import GarbageTranscriptError, TranscriptQuality, assess_transcript
from videopython.ai.dubbing.timing import TimingSynchronizer

__all__ = [
    "VideoDubber",
    "DubbingResult",
    "RevoiceResult",
    "TranslatedSegment",
    "SeparatedAudio",
    "LocalDubbingPipeline",
    "TimingSynchronizer",
    "GarbageTranscriptError",
    "TranscriptQuality",
    "assess_transcript",
]
