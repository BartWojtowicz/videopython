"""Result models and configuration for VideoAnalyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, model_validator

from videopython.base.description import AudioClassification, FaceTrack, SceneDescription
from videopython.base.transcription import Transcription

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
]

AUDIO_TO_TEXT = "audio_to_text"
AUDIO_CLASSIFIER = "audio_classifier"
SEMANTIC_SCENE_DETECTOR = "semantic_scene_detector"
SCENE_VLM = "scene_vlm"
FACE_TRACKER = "face_tracker"

ALL_ANALYZER_IDS: tuple[str, ...] = (
    AUDIO_TO_TEXT,
    AUDIO_CLASSIFIER,
    SEMANTIC_SCENE_DETECTOR,
    SCENE_VLM,
    FACE_TRACKER,
)


# Nested-type bridges: Transcription, SceneDescription, AudioClassification,
# and FaceTrack still live in videopython.base as plain dataclasses with
# hand-rolled to_dict/from_dict. Until they migrate to BaseModel, we
# interop at the field boundary so the wire format stays identical.
def _build_validator(cls: Any) -> Any:
    def _validate(value: Any) -> Any:
        if value is None or isinstance(value, cls):
            return value
        return cls.from_dict(value)

    return _validate


def _serialize(value: Any) -> Any:
    return value.to_dict() if value is not None else None


_TranscriptionField = Annotated[
    Transcription,
    BeforeValidator(_build_validator(Transcription)),
    PlainSerializer(_serialize, return_type=dict, when_used="always"),
]
_SceneDescriptionField = Annotated[
    SceneDescription,
    BeforeValidator(_build_validator(SceneDescription)),
    PlainSerializer(_serialize, return_type=dict, when_used="always"),
]
_AudioClassificationField = Annotated[
    AudioClassification,
    BeforeValidator(_build_validator(AudioClassification)),
    PlainSerializer(_serialize, return_type=dict, when_used="always"),
]


def _validate_face_tracks(value: Any) -> Any:
    if value is None:
        return None
    return [item if isinstance(item, FaceTrack) else FaceTrack.from_dict(item) for item in value]


def _serialize_face_tracks(value: Any) -> Any:
    if value is None:
        return None
    return [track.to_dict() for track in value]


_FaceTracksField = Annotated[
    list[FaceTrack],
    BeforeValidator(_validate_face_tracks),
    PlainSerializer(_serialize_face_tracks, return_type=list, when_used="always"),
]


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


class AnalysisRunInfo(BaseModel):
    """Runtime/provenance metadata for a full analysis run.

    ``stage_durations_seconds`` is populated by the analyzer with per-stage
    wall-clock times (whisper, scene_detection, scene_analysis, scene_vlm,
    audio_classification, and -- when both run together --
    whisper_and_scene_detection_parallel). Consumers can persist or aggregate
    these to track pipeline performance over time.
    """

    created_at: str
    mode: str
    library_version: str | None = None
    stage_durations_seconds: dict[str, float] = Field(default_factory=dict)
    total_duration_seconds: float | None = None


class VideoAnalysisConfig(BaseModel):
    """Execution config for scene-first analysis runs.

    ``analyzer_params`` lets you forward keyword arguments to each predictor
    constructor keyed by analyzer id.  For example::

        VideoAnalysisConfig(
            analyzer_params={
                "audio_to_text": {"model_name": "large"},
                "scene_vlm": {"model_size": "9b"},
            }
        )
    """

    enabled_analyzers: set[str] = Field(default_factory=lambda: set(ALL_ANALYZER_IDS))
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
    def rich_understanding_preset(cls) -> VideoAnalysisConfig:
        """Backward-compatible alias for the default config."""
        return cls()


class AudioAnalysisSection(BaseModel):
    """Audio understanding outputs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transcription: _TranscriptionField | None = None


class SceneAnalysisSample(BaseModel):
    """Flat scene payload with all per-scene analyzer outputs.

    ``scene_description`` carries the structured SceneVLM output (caption +
    subjects + shot type). ``faces`` is one list of tracks **per scene**
    (not per frame); each ``FaceTrack`` carries its own per-frame
    trajectory internally.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scene_index: int
    start_second: float
    end_second: float
    start_frame: int | None = None
    end_frame: int | None = None
    scene_description: _SceneDescriptionField | None = None
    audio_classification: _AudioClassificationField | None = None
    faces: _FaceTracksField | None = None


class SceneAnalysisSection(BaseModel):
    """Scene-centric visual/temporal/audio understanding output."""

    samples: list[SceneAnalysisSample] = Field(default_factory=list)


class VideoAnalysis(BaseModel):
    """Serializable aggregate scene-first analysis result for one video."""

    source: VideoAnalysisSource
    config: VideoAnalysisConfig
    run_info: AnalysisRunInfo
    audio: AudioAnalysisSection | None = None
    scenes: SceneAnalysisSection | None = None

    def save(self, path: str | Path, *, indent: int | None = 2) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(self.model_dump_json(indent=indent), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> VideoAnalysis:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
