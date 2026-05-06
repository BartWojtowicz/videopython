"""Local video dubbing functionality."""

from videopython.ai.dubbing.cache import DubCache, dub_cache_clear
from videopython.ai.dubbing.dubber import VideoDubber
from videopython.ai.dubbing.models import (
    DubbingResult,
    Expressiveness,
    RevoiceResult,
    SeparatedAudio,
    TranslatedSegment,
)
from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
from videopython.ai.dubbing.quality import GarbageTranscriptError, TranscriptQuality, assess_transcript
from videopython.ai.dubbing.timing import TimingSynchronizer
from videopython.ai.generation.translation import UnsupportedLanguageError

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
    "UnsupportedLanguageError",
    "DubCache",
    "dub_cache_clear",
    "Expressiveness",
]
