"""Local generation models, lazily re-exported (PEP 562).

Each symbol's backing leaf module is imported only on first access, so
``from videopython.ai.generation import TextToImage`` does not pull in
``audio`` (chatterbox/musicgen) or ``video`` (diffusers/CogVideoX). The
``TYPE_CHECKING`` block keeps the symbols visible to mypy and IDEs.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Redundant aliases = intentional re-exports (visible to mypy/IDE, not
    # flagged by ruff). Runtime resolution is lazy via __getattr__ below.
    from .audio import TextToMusic as TextToMusic
    from .audio import TextToSpeech as TextToSpeech
    from .image import TextToImage as TextToImage
    from .video import ImageToVideo as ImageToVideo
    from .video import TextToVideo as TextToVideo

_SYMBOL_MODULES: dict[str, str] = {
    "TextToSpeech": ".audio",
    "TextToMusic": ".audio",
    "TextToImage": ".image",
    "ImageToVideo": ".video",
    "TextToVideo": ".video",
}

__all__ = list(_SYMBOL_MODULES)


def __getattr__(name: str) -> object:
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
