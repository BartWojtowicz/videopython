"""Local video dubbing functionality, lazily re-exported (PEP 562).

The pipeline's translation backends (``MarianTranslator``/``Qwen3Translator``)
top-level import transformers/llama-cpp/torch, so eagerly chaining ``pipeline``
here would drag the ``[dub]`` translation stack in just to read
``DubbingConfig`` (pydantic-only). Lazy re-exports keep the lightweight
config/model symbols importable without that; only ``LocalDubbingPipeline`` /
``VideoDubber`` access loads the heavy chain. The ``TYPE_CHECKING`` block keeps
symbols visible to mypy and IDEs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from videopython.ai._optional import lazy_exports

if TYPE_CHECKING:
    # Redundant aliases = intentional re-exports (visible to mypy/IDE, not
    # flagged by ruff). Runtime resolution is lazy via __getattr__ below.
    from videopython.ai.dubbing.config import DubbingConfig as DubbingConfig
    from videopython.ai.dubbing.dubber import VideoDubber as VideoDubber
    from videopython.ai.dubbing.models import DubbingResult as DubbingResult
    from videopython.ai.dubbing.models import Expressiveness as Expressiveness
    from videopython.ai.dubbing.models import RevoiceResult as RevoiceResult
    from videopython.ai.dubbing.models import SeparatedAudio as SeparatedAudio
    from videopython.ai.dubbing.models import TranslatedSegment as TranslatedSegment
    from videopython.ai.dubbing.pipeline import LocalDubbingPipeline as LocalDubbingPipeline
    from videopython.ai.dubbing.quality import GarbageTranscriptError as GarbageTranscriptError
    from videopython.ai.dubbing.quality import TranscriptQuality as TranscriptQuality
    from videopython.ai.dubbing.quality import assess_transcript as assess_transcript
    from videopython.ai.dubbing.timing import TimingSynchronizer as TimingSynchronizer
    from videopython.ai.generation.translation import UnsupportedLanguageError as UnsupportedLanguageError

# Public symbol -> fully-qualified module that defines it.
_exports: dict[str, str] = {
    "DubbingConfig": "videopython.ai.dubbing.config",
    "VideoDubber": "videopython.ai.dubbing.dubber",
    "DubbingResult": "videopython.ai.dubbing.models",
    "RevoiceResult": "videopython.ai.dubbing.models",
    "TranslatedSegment": "videopython.ai.dubbing.models",
    "SeparatedAudio": "videopython.ai.dubbing.models",
    "Expressiveness": "videopython.ai.dubbing.models",
    "LocalDubbingPipeline": "videopython.ai.dubbing.pipeline",
    "TimingSynchronizer": "videopython.ai.dubbing.timing",
    "GarbageTranscriptError": "videopython.ai.dubbing.quality",
    "TranscriptQuality": "videopython.ai.dubbing.quality",
    "assess_transcript": "videopython.ai.dubbing.quality",
    "UnsupportedLanguageError": "videopython.ai.generation.translation",
}

__all__ = list(_exports)

__getattr__, __dir__ = lazy_exports(__name__, _exports)
