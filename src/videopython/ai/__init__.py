"""AI-powered generation, understanding, dubbing, and analysis.

Every public symbol is re-exported lazily via PEP 562 ``__getattr__`` so that
``import videopython.ai`` (or ``from videopython.ai import X``) loads ONLY the
leaf module backing the requested symbol — not every sibling. This is what
makes the granular extras (``asr``, ``vision``, ``tts``, ``generation``,
``dub``, ...) usable: with only ``[asr]`` installed, ``from videopython.ai
import AudioToText`` works, while touching ``TextToSpeech`` raises a clear
``[tts]``-pointing ``ImportError`` at attribute access instead of failing the
whole package import.

The ``TYPE_CHECKING`` block below keeps the eager imports visible to mypy and
IDEs (static autocompletion / re-export checking) without executing them at
runtime.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Redundant aliases mark these as intentional re-exports (mypy/IDE see the
    # symbols; ruff doesn't flag them as unused). Runtime resolution is lazy via
    # __getattr__ below.
    from .effects import ObjectDetectionOverlay as ObjectDetectionOverlay
    from .generation import ImageToVideo as ImageToVideo
    from .generation import TextToImage as TextToImage
    from .generation import TextToMusic as TextToMusic
    from .generation import TextToSpeech as TextToSpeech
    from .generation import TextToVideo as TextToVideo
    from .transforms import FaceTrackingCrop as FaceTrackingCrop
    from .understanding import AudioClassifier as AudioClassifier
    from .understanding import AudioToText as AudioToText
    from .understanding import FaceTracker as FaceTracker
    from .understanding import ObjectDetector as ObjectDetector
    from .understanding import SceneVLM as SceneVLM
    from .understanding import SemanticSceneDetector as SemanticSceneDetector
    from .video_analysis import VideoAnalysis as VideoAnalysis
    from .video_analysis import VideoAnalysisConfig as VideoAnalysisConfig
    from .video_analysis import VideoAnalyzer as VideoAnalyzer

# Public symbol -> submodule (relative to this package) that defines it. The
# submodule's own (lazy) __init__ resolves the symbol; we only import the
# submodule, never its siblings.
_SYMBOL_MODULES: dict[str, str] = {
    # Generation
    "TextToVideo": ".generation",
    "ImageToVideo": ".generation",
    "TextToImage": ".generation",
    "TextToSpeech": ".generation",
    "TextToMusic": ".generation",
    # Understanding
    "AudioToText": ".understanding",
    "AudioClassifier": ".understanding",
    "FaceTracker": ".understanding",
    "ObjectDetector": ".understanding",
    "SceneVLM": ".understanding",
    "SemanticSceneDetector": ".understanding",
    # Transforms (AI-powered)
    "FaceTrackingCrop": ".transforms",
    # Effects (AI-powered)
    "ObjectDetectionOverlay": ".effects",
    # Video analysis
    "VideoAnalysis": ".video_analysis",
    "VideoAnalysisConfig": ".video_analysis",
    "VideoAnalyzer": ".video_analysis",
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
