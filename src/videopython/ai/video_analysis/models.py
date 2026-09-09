"""Result models and configuration for VideoAnalyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from videopython.base.description import AudioClassification, FaceTrack, SceneDescription
from videopython.base.transcription import Transcription

from ._identity import source_digest

__all__ = [
    "ALL_ANALYZER_IDS",
    "AUDIO_CLASSIFIER",
    "AUDIO_TO_TEXT",
    "AnalyzerOutcome",
    "AnalysisRunInfo",
    "AnalysisProvenance",
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
]

_AnalyzerId = Literal[
    "audio_to_text",
    "audio_classifier",
    "semantic_scene_detector",
    "scene_vlm",
    "face_tracker",
]

AUDIO_TO_TEXT: _AnalyzerId = "audio_to_text"
AUDIO_CLASSIFIER: _AnalyzerId = "audio_classifier"
SEMANTIC_SCENE_DETECTOR: _AnalyzerId = "semantic_scene_detector"
SCENE_VLM: _AnalyzerId = "scene_vlm"
FACE_TRACKER: _AnalyzerId = "face_tracker"

ALL_ANALYZER_IDS: tuple[_AnalyzerId, ...] = (
    AUDIO_TO_TEXT,
    AUDIO_CLASSIFIER,
    SEMANTIC_SCENE_DETECTOR,
    SCENE_VLM,
    FACE_TRACKER,
)


class GeoMetadata(BaseModel):
    """Optional geolocation metadata attached to a video container."""

    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    source: str | None = None


class VideoAnalysisSource(BaseModel):
    """Source-level metadata for the analyzed video."""

    title: str | None = None
    path: str | None = None
    filename: str | None = None
    duration: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    creation_time: str | None = None
    geo: GeoMetadata | None = None
    raw_tags: dict[str, str] | None = None


class AnalyzerOutcome(BaseModel):
    """Completion state for one configured analyzer."""

    model_config = ConfigDict(extra="forbid")

    analyzer: _AnalyzerId
    status: Literal["completed", "skipped", "failed"]
    reason: Literal["disabled", "initialization_failed", "execution_failed"] | None


class AnalysisRunInfo(BaseModel):
    """Runtime/provenance metadata for a full analysis run.

    ``analyzer_outcomes`` records whether each configured analyzer completed,
    was disabled, or failed during initialization or execution.
    ``stage_durations_seconds`` is populated by the analyzer with per-stage
    wall-clock times (whisper, scene_detection, scene_analysis, scene_vlm,
    audio_classification, and -- when both run together --
    whisper_and_scene_detection_parallel). Consumers can persist or aggregate
    these to track pipeline performance over time.
    """

    created_at: str
    mode: str
    library_version: str | None = None
    analyzer_outcomes: list[AnalyzerOutcome]
    stage_durations_seconds: dict[str, float] = Field(default_factory=dict)
    total_duration_seconds: float | None = None


class VideoAnalysisConfig(BaseModel):
    """Execution config for scene-first analysis runs.

    ``analyzer_params`` lets you forward keyword arguments to each predictor
    constructor keyed by analyzer id.  For example::

        VideoAnalysisConfig(
            analyzer_params={
                "audio_to_text": {"model_name": "large"},
                "scene_vlm": {"model": "qwen3.6:27b"},
            }
        )
    """

    enabled_analyzers: set[str] = Field(default_factory=lambda: {str(analyzer) for analyzer in ALL_ANALYZER_IDS})
    analyzer_params: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_unknown_analyzer_ids(self) -> VideoAnalysisConfig:
        unknown_enabled = sorted(set(self.enabled_analyzers) - set(ALL_ANALYZER_IDS))
        if unknown_enabled:
            raise ValueError(f"Unknown analyzer ids in enabled_analyzers: {unknown_enabled}")
        unknown_params = sorted(set(self.analyzer_params) - set(ALL_ANALYZER_IDS))
        if unknown_params:
            raise ValueError(f"Unknown analyzer ids in analyzer_params: {unknown_params}")
        return self

    def get_params(self, analyzer_id: str) -> dict[str, Any]:
        """Return kwargs dict for the given analyzer, defaulting to empty."""
        return dict(self.analyzer_params.get(analyzer_id, {}))

    @classmethod
    def for_profile(cls, profile: str, *, faces: bool = True) -> VideoAnalysisConfig:
        """Config for an analysis profile: 'full' (all analyzers) or 'editing' (catalog-only, no audio classifier)."""
        if profile == "full":
            return cls()
        if profile == "editing":
            enabled = {SEMANTIC_SCENE_DETECTOR, SCENE_VLM, AUDIO_TO_TEXT}
            if faces:
                enabled.add(FACE_TRACKER)
            return cls(enabled_analyzers=enabled)
        raise ValueError(f"Unknown profile: {profile!r} (expected 'full' or 'editing')")


class AudioAnalysisSection(BaseModel):
    """Audio understanding outputs."""

    transcription: Transcription | None = None


class SceneAnalysisSample(BaseModel):
    """Flat scene payload with all per-scene analyzer outputs.

    ``scene_description`` carries the structured SceneVLM output (caption +
    subjects + shot type). ``faces`` is one list of tracks **per scene**
    (not per frame); each ``FaceTrack`` carries its own per-frame
    trajectory internally.
    """

    scene_index: int
    start_second: float
    end_second: float
    start_frame: int | None = None
    end_frame: int | None = None
    scene_description: SceneDescription | None = None
    audio_classification: AudioClassification | None = None
    faces: list[FaceTrack] | None = None


class SceneAnalysisSection(BaseModel):
    """Scene-centric visual/temporal/audio understanding output."""

    samples: list[SceneAnalysisSample] = Field(default_factory=list)


class AnalysisProvenance(BaseModel):
    """Source identity and model revisions recorded during analysis."""

    model_config = ConfigDict(extra="forbid")

    format_version: Literal[1]
    source_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    sampling: Literal["low", "medium", "high"]
    models: dict[str, dict[str, str | None] | None]


class VideoAnalysis(BaseModel):
    """Serializable aggregate scene-first analysis result for one video."""

    source: VideoAnalysisSource
    provenance: AnalysisProvenance
    config: VideoAnalysisConfig
    run_info: AnalysisRunInfo
    audio: AudioAnalysisSection | None = None
    scenes: SceneAnalysisSection | None = None

    def verify_source(self) -> Path:
        """Check the recorded file digest and return its resolved path without inference."""
        if self.source.path is None or self.provenance.source_sha256 is None:
            raise ValueError("Analysis has no verified file identity; analyze the source path again")
        path = Path(self.source.path).resolve()
        if source_digest(path) != self.provenance.source_sha256:
            raise ValueError("Source content differs from the saved analysis")
        return path

    def save(self, path: str | Path, *, indent: int | None = 2) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(self.model_dump_json(indent=indent), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> VideoAnalysis:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
