"""Local generation models, lazily re-exported (PEP 562).

Each symbol's backing leaf module is imported only on first access, so
``from videopython.ai.generation import TextToImage`` does not pull in
``audio`` (chatterbox/musicgen) or ``video`` (diffusers/Wan2.2). The
``TYPE_CHECKING`` block keeps the symbols visible to mypy and IDEs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from videopython.ai._optional import lazy_exports

if TYPE_CHECKING:
    # Redundant aliases = intentional re-exports (visible to mypy/IDE, not
    # flagged by ruff). Runtime resolution is lazy via __getattr__ below.
    from .audio import TextToMusic as TextToMusic
    from .audio import TextToSpeech as TextToSpeech
    from .image import TextToImage as TextToImage
    from .video import ImageToVideo as ImageToVideo
    from .video import TextToVideo as TextToVideo

_exports: dict[str, str] = {
    "TextToSpeech": ".audio",
    "TextToMusic": ".audio",
    "TextToImage": ".image",
    "ImageToVideo": ".video",
    "TextToVideo": ".video",
}

__all__ = list(_exports)

__getattr__, __dir__ = lazy_exports(__name__, _exports)
