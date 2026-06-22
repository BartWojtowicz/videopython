"""AI-powered generation, understanding, dubbing, and analysis.

Every public symbol is re-exported lazily via PEP 562 ``__getattr__`` so that
``import videopython.ai`` (or ``from videopython.ai import X``) loads ONLY the
leaf module backing the requested symbol — not every sibling. All AI capabilities
ship in the single ``[ai]`` extra; the laziness keeps ``import videopython`` (and
importing one leaf class) light by deferring the heavy ML imports (torch /
transformers / diffusers / ultralytics) until a symbol is actually used. When
``[ai]`` is not installed, touching a symbol raises a clear
``pip install 'videopython[ai]'``-pointing ``ImportError`` at attribute access
instead of failing the whole package import.

The ``TYPE_CHECKING`` block below keeps the eager imports visible to mypy and
IDEs (static autocompletion / re-export checking) without executing them at
runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from videopython.ai._optional import lazy_exports

if TYPE_CHECKING:
    # Redundant aliases mark these as intentional re-exports (mypy/IDE see the
    # symbols; ruff doesn't flag them as unused). Runtime resolution is lazy via
    # __getattr__ below.
    from .auto_edit import AutoEditError as AutoEditError
    from .auto_edit import AutoEditor as AutoEditor
    from .auto_edit import EditCatalog as EditCatalog
    from .auto_edit import EditPlan as EditPlan
    from .auto_edit import OllamaVisionLLM as OllamaVisionLLM
    from .auto_edit import PlannerError as PlannerError
    from .auto_edit import StructuredVisionLLM as StructuredVisionLLM
    from .auto_edit import build_catalog as build_catalog
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
_exports: dict[str, str] = {
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
    # Auto-editing (LLM-authored edits)
    "AutoEditor": ".auto_edit",
    "AutoEditError": ".auto_edit",
    "OllamaVisionLLM": ".auto_edit",
    "StructuredVisionLLM": ".auto_edit",
    "PlannerError": ".auto_edit",
    "EditCatalog": ".auto_edit",
    "EditPlan": ".auto_edit",
    "build_catalog": ".auto_edit",
}

__all__ = list(_exports)

__getattr__, __dir__ = lazy_exports(__name__, _exports)
