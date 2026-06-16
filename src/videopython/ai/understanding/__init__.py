"""Local understanding models, lazily re-exported (PEP 562).

Each symbol's backing leaf module is imported only on first access, so
``from videopython.ai.understanding import ObjectDetector`` (``[vision]``)
does not pull in ``audio`` (whisper/pyannote — ``[asr]``). The
``TYPE_CHECKING`` block keeps the symbols visible to mypy and IDEs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from videopython.ai._optional import lazy_exports

if TYPE_CHECKING:
    # Redundant aliases = intentional re-exports (visible to mypy/IDE, not
    # flagged by ruff). Runtime resolution is lazy via __getattr__ below.
    from .audio import AudioClassifier as AudioClassifier
    from .audio import AudioToText as AudioToText
    from .faces import FaceTracker as FaceTracker
    from .image import SceneVLM as SceneVLM
    from .objects import ObjectDetector as ObjectDetector
    from .temporal import SemanticSceneDetector as SemanticSceneDetector

_exports: dict[str, str] = {
    "AudioToText": ".audio",
    "AudioClassifier": ".audio",
    "FaceTracker": ".faces",
    "ObjectDetector": ".objects",
    "SceneVLM": ".image",
    "SemanticSceneDetector": ".temporal",
}

__all__ = list(_exports)

__getattr__, __dir__ = lazy_exports(__name__, _exports)
