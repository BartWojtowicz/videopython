"""Scene-first video analysis: VideoAnalyzer plus its result models.

Unlike the leaf packages (understanding / generation / dubbing) this uses eager
imports: it is a thin orchestration package whose submodules are always loaded
together when analysis runs, and the heavy ML deps it pulls in (via
``understanding``) are themselves still lazily imported at first use.
"""

from __future__ import annotations

from .analyzer import VideoAnalyzer
from .models import (
    ALL_ANALYZER_IDS,
    AUDIO_CLASSIFIER,
    AUDIO_TO_TEXT,
    FACE_TRACKER,
    SCENE_VLM,
    SEMANTIC_SCENE_DETECTOR,
    AnalysisRunInfo,
    AudioAnalysisSection,
    GeoMetadata,
    SceneAnalysisSample,
    SceneAnalysisSection,
    VideoAnalysis,
    VideoAnalysisConfig,
    VideoAnalysisSource,
)

__all__ = [
    "ALL_ANALYZER_IDS",
    "AUDIO_CLASSIFIER",
    "AUDIO_TO_TEXT",
    "AnalysisRunInfo",
    "AudioAnalysisSection",
    "FACE_TRACKER",
    "GeoMetadata",
    "SCENE_VLM",
    "SEMANTIC_SCENE_DETECTOR",
    "SceneAnalysisSample",
    "SceneAnalysisSection",
    "VideoAnalysis",
    "VideoAnalysisConfig",
    "VideoAnalysisSource",
    "VideoAnalyzer",
]
