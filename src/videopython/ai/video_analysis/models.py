"""Result models and configuration dataclasses for VideoAnalyzer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass
class GeoMetadata:
    """Optional geolocation metadata attached to a video container."""

    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeoMetadata":
        return cls(
            latitude=data.get("latitude"),
            longitude=data.get("longitude"),
            altitude=data.get("altitude"),
            source=data.get("source"),
        )


@dataclass
class VideoAnalysisSource:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "path": self.path,
            "filename": self.filename,
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "creation_time": self.creation_time,
            "geo": self.geo.to_dict() if self.geo else None,
            "raw_tags": self.raw_tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoAnalysisSource":
        geo_data = data.get("geo")
        return cls(
            title=data.get("title"),
            path=data.get("path"),
            filename=data.get("filename"),
            duration=data.get("duration"),
            fps=data.get("fps"),
            width=data.get("width"),
            height=data.get("height"),
            frame_count=data.get("frame_count"),
            creation_time=data.get("creation_time"),
            geo=GeoMetadata.from_dict(geo_data) if geo_data else None,
            raw_tags=data.get("raw_tags"),
        )


@dataclass
class AnalysisRunInfo:
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
    stage_durations_seconds: dict[str, float] = field(default_factory=dict)
    total_duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "mode": self.mode,
            "library_version": self.library_version,
            "stage_durations_seconds": dict(self.stage_durations_seconds),
            "total_duration_seconds": self.total_duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisRunInfo":
        return cls(
            created_at=data["created_at"],
            mode=data["mode"],
            library_version=data.get("library_version"),
            stage_durations_seconds={str(k): float(v) for k, v in data["stage_durations_seconds"].items()},
            total_duration_seconds=data["total_duration_seconds"],
        )


@dataclass
class VideoAnalysisConfig:
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

    enabled_analyzers: set[str] = field(default_factory=lambda: set(ALL_ANALYZER_IDS))
    analyzer_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown_enabled = sorted(set(self.enabled_analyzers) - set(ALL_ANALYZER_IDS))
        if unknown_enabled:
            raise ValueError(f"Unknown analyzer ids in enabled_analyzers: {unknown_enabled}")
        unknown_params = sorted(set(self.analyzer_params) - set(ALL_ANALYZER_IDS))
        if unknown_params:
            raise ValueError(f"Unknown analyzer ids in analyzer_params: {unknown_params}")

    def get_params(self, analyzer_id: str) -> dict[str, Any]:
        """Return kwargs dict for the given analyzer, defaulting to empty."""
        return dict(self.analyzer_params.get(analyzer_id, {}))

    @classmethod
    def rich_understanding_preset(cls) -> "VideoAnalysisConfig":
        """Backward-compatible alias for the default config."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enabled_analyzers": sorted(self.enabled_analyzers),
        }
        if self.analyzer_params:
            result["analyzer_params"] = self.analyzer_params
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoAnalysisConfig":
        enabled_raw = data.get("enabled_analyzers")
        return cls(
            enabled_analyzers=set(enabled_raw) if enabled_raw is not None else set(ALL_ANALYZER_IDS),
            analyzer_params=data.get("analyzer_params", {}),
        )


@dataclass
class AudioAnalysisSection:
    """Audio understanding outputs."""

    transcription: Transcription | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcription": self.transcription.to_dict() if self.transcription else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioAnalysisSection":
        transcription_data = data.get("transcription")
        return cls(transcription=Transcription.from_dict(transcription_data) if transcription_data else None)


@dataclass
class SceneAnalysisSample:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_index": self.scene_index,
            "start_second": self.start_second,
            "end_second": self.end_second,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "scene_description": self.scene_description.to_dict() if self.scene_description else None,
            "audio_classification": self.audio_classification.to_dict() if self.audio_classification else None,
            "faces": [track.to_dict() for track in self.faces] if self.faces is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneAnalysisSample":
        faces_raw = data.get("faces")
        return cls(
            scene_index=int(data["scene_index"]),
            start_second=float(data["start_second"]),
            end_second=float(data["end_second"]),
            start_frame=data.get("start_frame"),
            end_frame=data.get("end_frame"),
            scene_description=(
                SceneDescription.from_dict(data["scene_description"]) if data.get("scene_description") else None
            ),
            audio_classification=(
                AudioClassification.from_dict(data["audio_classification"])
                if data.get("audio_classification")
                else None
            ),
            faces=[FaceTrack.from_dict(item) for item in faces_raw] if faces_raw is not None else None,
        )


@dataclass
class SceneAnalysisSection:
    """Scene-centric visual/temporal/audio understanding output."""

    samples: list[SceneAnalysisSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneAnalysisSection":
        return cls(samples=[SceneAnalysisSample.from_dict(item) for item in data.get("samples", [])])


@dataclass
class VideoAnalysis:
    """Serializable aggregate scene-first analysis result for one video."""

    source: VideoAnalysisSource
    config: VideoAnalysisConfig
    run_info: AnalysisRunInfo
    audio: AudioAnalysisSection | None = None
    scenes: SceneAnalysisSection | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "config": self.config.to_dict(),
            "run_info": self.run_info.to_dict(),
            "audio": self.audio.to_dict() if self.audio else None,
            "scenes": self.scenes.to_dict() if self.scenes else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoAnalysis":
        audio_data = data.get("audio")
        scenes_data = data.get("scenes")
        return cls(
            source=VideoAnalysisSource.from_dict(data["source"]),
            config=VideoAnalysisConfig.from_dict(data["config"]),
            run_info=AnalysisRunInfo.from_dict(data["run_info"]),
            audio=AudioAnalysisSection.from_dict(audio_data) if audio_data else None,
            scenes=SceneAnalysisSection.from_dict(scenes_data) if scenes_data else None,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "VideoAnalysis":
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path, *, indent: int | None = 2) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VideoAnalysis":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
