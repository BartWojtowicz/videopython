from __future__ import annotations

import json
import re
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

from videopython.ai.understanding import (
    ActionRecognizer,
    AudioClassifier,
    AudioToText,
    CameraMotionDetector,
    CombinedFrameAnalyzer,
    FaceDetector,
    ImageToText,
    MotionAnalyzer,
    ObjectDetector,
    SemanticSceneDetector,
    ShotTypeClassifier,
    TextDetector,
)
from videopython.ai.understanding.detection import CombinedFrameAnalysis
from videopython.base.audio import Audio
from videopython.base.description import (
    AudioClassification,
    DetectedAction,
    DetectedFace,
    DetectedObject,
    MotionInfo,
    SceneBoundary,
)
from videopython.base.text.transcription import Transcription
from videopython.base.video import FrameIterator, Video, VideoMetadata, extract_frames_at_indices

__all__ = ["VideoAnalysis", "VideoAnalysisConfig", "VideoAnalyzer"]

AUDIO_TO_TEXT = "audio_to_text"
AUDIO_CLASSIFIER = "audio_classifier"
SEMANTIC_SCENE_DETECTOR = "semantic_scene_detector"
ACTION_RECOGNIZER = "action_recognizer"
MOTION_ANALYZER = "motion_analyzer"
CAMERA_MOTION_DETECTOR = "camera_motion_detector"
COMBINED_FRAME_ANALYZER = "combined_frame_analyzer"
OBJECT_DETECTOR = "object_detector"
FACE_DETECTOR = "face_detector"
TEXT_DETECTOR = "text_detector"
SHOT_TYPE_CLASSIFIER = "shot_type_classifier"
IMAGE_TO_TEXT = "image_to_text"

ALL_ANALYZER_IDS: tuple[str, ...] = (
    AUDIO_TO_TEXT,
    AUDIO_CLASSIFIER,
    SEMANTIC_SCENE_DETECTOR,
    ACTION_RECOGNIZER,
    MOTION_ANALYZER,
    CAMERA_MOTION_DETECTOR,
    COMBINED_FRAME_ANALYZER,
    OBJECT_DETECTOR,
    FACE_DETECTOR,
    TEXT_DETECTOR,
    SHOT_TYPE_CLASSIFIER,
    IMAGE_TO_TEXT,
)

_OVERLAPPING_FRAME_ANALYZERS: tuple[str, ...] = (
    OBJECT_DETECTOR,
    TEXT_DETECTOR,
    SHOT_TYPE_CLASSIFIER,
)

_CREATION_TIME_TAG_KEYS: tuple[str, ...] = (
    "creation_time",
    "com.apple.quicktime.creationdate",
    "date",
)

_GEO_TAG_KEYS: tuple[str, ...] = (
    "com.apple.quicktime.location.iso6709",
    "location",
    "location-eng",
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
class AnalysisStepStatus:
    """Execution status of one analyzer step."""

    status: str
    duration_seconds: float | None = None
    error: str | None = None
    warning: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "warning": self.warning,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisStepStatus":
        return cls(
            status=data["status"],
            duration_seconds=data.get("duration_seconds"),
            error=data.get("error"),
            warning=data.get("warning"),
            details=data.get("details", {}),
        )


@dataclass
class AnalysisRunInfo:
    """Runtime/provenance metadata for a full analysis run."""

    created_at: str
    mode: str
    library_version: str | None = None
    elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "mode": self.mode,
            "library_version": self.library_version,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisRunInfo":
        return cls(
            created_at=data["created_at"],
            mode=data["mode"],
            library_version=data.get("library_version"),
            elapsed_seconds=data.get("elapsed_seconds"),
        )


@dataclass
class VideoAnalysisConfig:
    """Serializable execution plan for VideoAnalyzer."""

    enabled_analyzers: set[str] = field(
        default_factory=lambda: {
            AUDIO_TO_TEXT,
            AUDIO_CLASSIFIER,
            SEMANTIC_SCENE_DETECTOR,
            ACTION_RECOGNIZER,
            MOTION_ANALYZER,
            CAMERA_MOTION_DETECTOR,
            COMBINED_FRAME_ANALYZER,
            OBJECT_DETECTOR,
            FACE_DETECTOR,
            TEXT_DETECTOR,
            SHOT_TYPE_CLASSIFIER,
        }
    )
    optional_analyzers: set[str] = field(default_factory=lambda: {SHOT_TYPE_CLASSIFIER, COMBINED_FRAME_ANALYZER})
    analyzer_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    backend_overrides: dict[str, str] = field(default_factory=dict)
    frame_sampling_mode: str = "hybrid"
    frames_per_second: float = 0.5
    max_frames: int | None = 120
    include_scene_boundaries: bool = True
    scene_representative_offset: float = 0.5
    camera_motion_stride: int = 1
    frame_chunk_size: int = 16
    max_memory_mb: int | None = None
    best_effort: bool = True
    fail_fast: bool = False
    prefer_combined_frame_analyzer: bool = True
    include_geo: bool = True
    redact_geo: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled_analyzers": sorted(self.enabled_analyzers),
            "optional_analyzers": sorted(self.optional_analyzers),
            "analyzer_params": self.analyzer_params,
            "backend_overrides": self.backend_overrides,
            "frame_sampling_mode": self.frame_sampling_mode,
            "frames_per_second": self.frames_per_second,
            "max_frames": self.max_frames,
            "include_scene_boundaries": self.include_scene_boundaries,
            "scene_representative_offset": self.scene_representative_offset,
            "camera_motion_stride": self.camera_motion_stride,
            "frame_chunk_size": self.frame_chunk_size,
            "max_memory_mb": self.max_memory_mb,
            "best_effort": self.best_effort,
            "fail_fast": self.fail_fast,
            "prefer_combined_frame_analyzer": self.prefer_combined_frame_analyzer,
            "include_geo": self.include_geo,
            "redact_geo": self.redact_geo,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoAnalysisConfig":
        defaults = cls()
        enabled_raw = data.get("enabled_analyzers")
        optional_raw = data.get("optional_analyzers")
        return cls(
            enabled_analyzers=set(enabled_raw) if enabled_raw is not None else defaults.enabled_analyzers,
            optional_analyzers=set(optional_raw) if optional_raw is not None else defaults.optional_analyzers,
            analyzer_params=data.get("analyzer_params", {}),
            backend_overrides=data.get("backend_overrides", {}),
            frame_sampling_mode=data.get("frame_sampling_mode", "hybrid"),
            frames_per_second=float(data.get("frames_per_second", 0.5)),
            max_frames=data.get("max_frames", 120),
            include_scene_boundaries=bool(data.get("include_scene_boundaries", True)),
            scene_representative_offset=float(data.get("scene_representative_offset", 0.5)),
            camera_motion_stride=int(data.get("camera_motion_stride", 1)),
            frame_chunk_size=int(data.get("frame_chunk_size", 16)),
            max_memory_mb=data.get("max_memory_mb"),
            best_effort=bool(data.get("best_effort", True)),
            fail_fast=bool(data.get("fail_fast", False)),
            prefer_combined_frame_analyzer=bool(data.get("prefer_combined_frame_analyzer", True)),
            include_geo=bool(data.get("include_geo", True)),
            redact_geo=bool(data.get("redact_geo", False)),
        )


@dataclass
class AudioAnalysisSection:
    """Audio understanding outputs."""

    transcription: Transcription | None = None
    classification: AudioClassification | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcription": self.transcription.to_dict() if self.transcription else None,
            "classification": self.classification.to_dict() if self.classification else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioAnalysisSection":
        transcription_data = data.get("transcription")
        classification_data = data.get("classification")
        return cls(
            transcription=Transcription.from_dict(transcription_data) if transcription_data else None,
            classification=AudioClassification.from_dict(classification_data) if classification_data else None,
        )


@dataclass
class TemporalAnalysisSection:
    """Scene and action outputs."""

    scenes: list[SceneBoundary] = field(default_factory=list)
    actions: list[DetectedAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenes": [scene.to_dict() for scene in self.scenes],
            "actions": [action.to_dict() for action in self.actions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalAnalysisSection":
        return cls(
            scenes=[SceneBoundary.from_dict(item) for item in data.get("scenes", [])],
            actions=[DetectedAction.from_dict(item) for item in data.get("actions", [])],
        )


@dataclass
class MotionTimelineSample:
    """Timestamped motion sample."""

    timestamp: float
    motion: MotionInfo

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp": self.timestamp, "motion": self.motion.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionTimelineSample":
        return cls(timestamp=float(data["timestamp"]), motion=MotionInfo.from_dict(data["motion"]))


@dataclass
class CameraMotionSample:
    """Camera motion label over a sampled frame window."""

    start: float
    end: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "label": self.label}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraMotionSample":
        return cls(start=float(data["start"]), end=float(data["end"]), label=data["label"])


@dataclass
class MotionAnalysisSection:
    """Motion outputs."""

    video_motion: list[MotionInfo] = field(default_factory=list)
    motion_timeline: list[MotionTimelineSample] = field(default_factory=list)
    camera_motion_samples: list[CameraMotionSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_motion": [item.to_dict() for item in self.video_motion],
            "motion_timeline": [item.to_dict() for item in self.motion_timeline],
            "camera_motion_samples": [item.to_dict() for item in self.camera_motion_samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionAnalysisSection":
        return cls(
            video_motion=[MotionInfo.from_dict(item) for item in data.get("video_motion", [])],
            motion_timeline=[MotionTimelineSample.from_dict(item) for item in data.get("motion_timeline", [])],
            camera_motion_samples=[
                CameraMotionSample.from_dict(item) for item in data.get("camera_motion_samples", [])
            ],
        )


@dataclass
class FrameSamplingReport:
    """Sampling strategy and selected frame references."""

    mode: str
    frames_per_second: float | None
    max_frames: int | None
    sampled_indices: list[int] = field(default_factory=list)
    sampled_timestamps: list[float] = field(default_factory=list)
    access_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "frames_per_second": self.frames_per_second,
            "max_frames": self.max_frames,
            "sampled_indices": self.sampled_indices,
            "sampled_timestamps": self.sampled_timestamps,
            "access_mode": self.access_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameSamplingReport":
        return cls(
            mode=data["mode"],
            frames_per_second=data.get("frames_per_second"),
            max_frames=data.get("max_frames"),
            sampled_indices=[int(v) for v in data.get("sampled_indices", [])],
            sampled_timestamps=[float(v) for v in data.get("sampled_timestamps", [])],
            access_mode=data.get("access_mode"),
        )


@dataclass
class FrameAnalysisSample:
    """Aggregated per-frame analysis result for one sampled timestamp."""

    timestamp: float
    frame_index: int | None = None
    image_caption: str | None = None
    objects: list[DetectedObject] | None = None
    faces: list[DetectedFace] | None = None
    text: list[str] | None = None
    shot_type: str | None = None
    combined: CombinedFrameAnalysis | None = None
    step_results: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "image_caption": self.image_caption,
            "objects": [item.to_dict() for item in self.objects] if self.objects is not None else None,
            "faces": [item.to_dict() for item in self.faces] if self.faces is not None else None,
            "text": self.text,
            "shot_type": self.shot_type,
            "combined": self.combined.to_dict() if self.combined else None,
            "step_results": self.step_results,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameAnalysisSample":
        objects = data.get("objects")
        faces = data.get("faces")
        combined = data.get("combined")
        return cls(
            timestamp=float(data["timestamp"]),
            frame_index=data.get("frame_index"),
            image_caption=data.get("image_caption"),
            objects=[DetectedObject.from_dict(item) for item in objects] if objects is not None else None,
            faces=[DetectedFace.from_dict(item) for item in faces] if faces is not None else None,
            text=data.get("text"),
            shot_type=data.get("shot_type"),
            combined=CombinedFrameAnalysis.from_dict(combined) if combined else None,
            step_results=data.get("step_results"),
        )


@dataclass
class FrameAnalysisSection:
    """Frame-level sampling and outputs."""

    sampling: FrameSamplingReport
    samples: list[FrameAnalysisSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampling": self.sampling.to_dict(),
            "samples": [item.to_dict() for item in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameAnalysisSection":
        return cls(
            sampling=FrameSamplingReport.from_dict(data["sampling"]),
            samples=[FrameAnalysisSample.from_dict(item) for item in data.get("samples", [])],
        )


@dataclass
class VideoAnalysis:
    """Serializable aggregate analysis result for one video."""

    source: VideoAnalysisSource
    config: VideoAnalysisConfig
    run_info: AnalysisRunInfo
    steps: dict[str, AnalysisStepStatus] = field(default_factory=dict)
    audio: AudioAnalysisSection | None = None
    temporal: TemporalAnalysisSection | None = None
    motion: MotionAnalysisSection | None = None
    frames: FrameAnalysisSection | None = None
    summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "config": self.config.to_dict(),
            "run_info": self.run_info.to_dict(),
            "steps": {name: step.to_dict() for name, step in self.steps.items()},
            "audio": self.audio.to_dict() if self.audio else None,
            "temporal": self.temporal.to_dict() if self.temporal else None,
            "motion": self.motion.to_dict() if self.motion else None,
            "frames": self.frames.to_dict() if self.frames else None,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoAnalysis":
        audio_data = data.get("audio")
        temporal_data = data.get("temporal")
        motion_data = data.get("motion")
        frames_data = data.get("frames")
        return cls(
            source=VideoAnalysisSource.from_dict(data["source"]),
            config=VideoAnalysisConfig.from_dict(data["config"]),
            run_info=AnalysisRunInfo.from_dict(data["run_info"]),
            steps={name: AnalysisStepStatus.from_dict(item) for name, item in data.get("steps", {}).items()},
            audio=AudioAnalysisSection.from_dict(audio_data) if audio_data else None,
            temporal=TemporalAnalysisSection.from_dict(temporal_data) if temporal_data else None,
            motion=MotionAnalysisSection.from_dict(motion_data) if motion_data else None,
            frames=FrameAnalysisSection.from_dict(frames_data) if frames_data else None,
            summary=data.get("summary"),
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


class VideoAnalyzer:
    """Orchestrates understanding analyzers and builds `VideoAnalysis` output."""

    def __init__(self, config: VideoAnalysisConfig | None = None):
        self.config = config or VideoAnalysisConfig()

    def analyze_path(self, path: str | Path) -> VideoAnalysis:
        """Analyze a video path with bounded frame memory usage."""
        path_obj = Path(path)
        metadata = VideoMetadata.from_path(path_obj)
        source = self._build_source_from_path(path_obj, metadata)
        return self._analyze(video=None, source_path=path_obj, metadata=metadata, source=source)

    def analyze(self, video: Video, *, source_path: str | Path | None = None) -> VideoAnalysis:
        """Analyze an in-memory Video object."""
        metadata = VideoMetadata.from_video(video)
        source = self._build_source_from_video(video=video, source_path=source_path, metadata=metadata)
        return self._analyze(
            video=video,
            source_path=Path(source_path) if source_path else None,
            metadata=metadata,
            source=source,
        )

    def _analyze(
        self,
        *,
        video: Video | None,
        source_path: Path | None,
        metadata: VideoMetadata,
        source: VideoAnalysisSource,
    ) -> VideoAnalysis:
        mode = "path" if source_path is not None else "video"
        if source_path is None and video is None:
            raise ValueError("Either `source_path` or `video` must be provided")
        started = time.perf_counter()
        steps: dict[str, AnalysisStepStatus] = {}

        for analyzer_id in ALL_ANALYZER_IDS:
            if analyzer_id not in self.config.enabled_analyzers:
                steps[analyzer_id] = AnalysisStepStatus(status="skipped", warning="Disabled in config")

        run_info = AnalysisRunInfo(
            created_at=_utc_now_iso(),
            mode=mode,
            library_version=_library_version(),
            elapsed_seconds=None,
        )

        audio_section = AudioAnalysisSection()
        temporal_section = TemporalAnalysisSection()
        motion_section = MotionAnalysisSection()

        audio_cache: Audio | None = None

        def get_path_audio() -> Audio:
            nonlocal audio_cache
            if audio_cache is None:
                if source_path is None:
                    raise RuntimeError("Path audio requested for in-memory analysis without source path")
                audio_cache = Audio.from_path(source_path)
            return audio_cache

        if AUDIO_TO_TEXT in self.config.enabled_analyzers:
            audio_input: Audio | Video
            if source_path is not None:
                audio_input = get_path_audio()
            else:
                assert video is not None
                audio_input = video
            transcription = self._run_step(
                steps,
                AUDIO_TO_TEXT,
                lambda: AudioToText(**self._analyzer_kwargs(AUDIO_TO_TEXT)).transcribe(audio_input),
                optional=AUDIO_TO_TEXT in self.config.optional_analyzers,
            )
            if transcription is not None:
                audio_section.transcription = transcription

        if AUDIO_CLASSIFIER in self.config.enabled_analyzers:
            classifier_input: Audio | Video
            if source_path is not None:
                classifier_input = get_path_audio()
            else:
                assert video is not None
                classifier_input = video
            classification = self._run_step(
                steps,
                AUDIO_CLASSIFIER,
                lambda: AudioClassifier(**self._analyzer_kwargs(AUDIO_CLASSIFIER)).classify(classifier_input),
                optional=AUDIO_CLASSIFIER in self.config.optional_analyzers,
            )
            if classification is not None:
                audio_section.classification = classification

        scenes: list[SceneBoundary] = []
        if SEMANTIC_SCENE_DETECTOR in self.config.enabled_analyzers:
            scene_detector = SemanticSceneDetector(**self._analyzer_kwargs(SEMANTIC_SCENE_DETECTOR))
            scenes_result = self._run_step(
                steps,
                SEMANTIC_SCENE_DETECTOR,
                lambda: scene_detector.detect_streaming(source_path)
                if source_path is not None
                else scene_detector.detect(_require_video(video)),
                optional=SEMANTIC_SCENE_DETECTOR in self.config.optional_analyzers,
            )
            if scenes_result is not None:
                scenes = scenes_result
                temporal_section.scenes = scenes_result

        if ACTION_RECOGNIZER in self.config.enabled_analyzers:
            action_recognizer = ActionRecognizer(**self._analyzer_kwargs(ACTION_RECOGNIZER))
            actions_result = self._run_step(
                steps,
                ACTION_RECOGNIZER,
                lambda: action_recognizer.recognize_path(source_path)
                if source_path is not None
                else action_recognizer.recognize(_require_video(video)),
                optional=ACTION_RECOGNIZER in self.config.optional_analyzers,
            )
            if actions_result is not None:
                temporal_section.actions = actions_result

        if MOTION_ANALYZER in self.config.enabled_analyzers:
            motion_analyzer = MotionAnalyzer(**self._analyzer_kwargs(MOTION_ANALYZER))
            if source_path is not None:
                motion_timeline = self._run_step(
                    steps,
                    MOTION_ANALYZER,
                    lambda: motion_analyzer.analyze_video_path(
                        source_path,
                        frames_per_second=max(float(self.config.frames_per_second), 0.1),
                    ),
                    optional=MOTION_ANALYZER in self.config.optional_analyzers,
                )
                if motion_timeline is not None:
                    motion_section.motion_timeline = [
                        MotionTimelineSample(timestamp=float(ts), motion=motion) for ts, motion in motion_timeline
                    ]
                    motion_section.video_motion = [sample.motion for sample in motion_section.motion_timeline]
            else:
                motion_result = self._run_step(
                    steps,
                    MOTION_ANALYZER,
                    lambda: motion_analyzer.analyze_video(_require_video(video)),
                    optional=MOTION_ANALYZER in self.config.optional_analyzers,
                )
                if motion_result is not None:
                    motion_section.video_motion = motion_result

        frame_indices = self._plan_frame_indices(metadata=metadata, scenes=scenes)
        sampling = FrameSamplingReport(
            mode=self.config.frame_sampling_mode,
            frames_per_second=self.config.frames_per_second,
            max_frames=self.config.max_frames,
            sampled_indices=frame_indices,
            sampled_timestamps=[round(idx / metadata.fps, 6) for idx in frame_indices],
            access_mode=None,
        )

        frame_steps_runtime: dict[str, dict[str, Any]] = {}
        camera_samples: list[CameraMotionSample] = []
        frame_samples: list[FrameAnalysisSample] = []
        frame_work_ids = (
            COMBINED_FRAME_ANALYZER,
            OBJECT_DETECTOR,
            FACE_DETECTOR,
            TEXT_DETECTOR,
            SHOT_TYPE_CLASSIFIER,
            IMAGE_TO_TEXT,
            CAMERA_MOTION_DETECTOR,
        )
        frame_work_enabled = any(step_id in self.config.enabled_analyzers for step_id in frame_work_ids)

        if frame_indices and frame_work_enabled:
            frame_steps_runtime = self._initialize_frame_steps(steps)
            if frame_steps_runtime:
                access_mode = self._choose_frame_access_mode(len(frame_indices), metadata.frame_count)
                sampling.access_mode = access_mode
                if source_path is not None:
                    frame_samples, camera_samples = self._process_path_samples(
                        path=source_path,
                        metadata=metadata,
                        frame_indices=frame_indices,
                        frame_steps_runtime=frame_steps_runtime,
                        steps=steps,
                    )
                else:
                    frame_samples, camera_samples = self._process_video_samples(
                        video=video,
                        frame_indices=frame_indices,
                        frame_steps_runtime=frame_steps_runtime,
                        steps=steps,
                    )
                self._finalize_frame_steps(frame_steps_runtime=frame_steps_runtime, steps=steps)
        elif frame_work_enabled:
            for step_id in frame_work_ids:
                if step_id in self.config.enabled_analyzers and step_id not in steps:
                    steps[step_id] = AnalysisStepStatus(status="skipped", warning="No frames sampled")

        if camera_samples:
            motion_section.camera_motion_samples = camera_samples

        frames_section = FrameAnalysisSection(sampling=sampling, samples=frame_samples)

        run_info.elapsed_seconds = time.perf_counter() - started

        summary = {
            "scene_count": len(temporal_section.scenes),
            "action_count": len(temporal_section.actions),
            "frame_sample_count": len(frame_samples),
            "audio_events_count": len(audio_section.classification.events) if audio_section.classification else 0,
        }

        return VideoAnalysis(
            source=source,
            config=self.config,
            run_info=run_info,
            steps=steps,
            audio=audio_section if (audio_section.transcription or audio_section.classification) else None,
            temporal=temporal_section if (temporal_section.scenes or temporal_section.actions) else None,
            motion=motion_section
            if (motion_section.video_motion or motion_section.motion_timeline or motion_section.camera_motion_samples)
            else None,
            frames=frames_section,
            summary=summary,
        )

    def _run_step(
        self,
        steps: dict[str, AnalysisStepStatus],
        step_id: str,
        func: Any,
        *,
        optional: bool,
    ) -> Any:
        started = time.perf_counter()
        try:
            result = func()
        except Exception as exc:
            duration = time.perf_counter() - started
            status = "skipped" if optional else "failed"
            steps[step_id] = AnalysisStepStatus(status=status, duration_seconds=duration, error=str(exc))
            if self._should_raise(optional=optional):
                raise
            return None

        steps[step_id] = AnalysisStepStatus(status="succeeded", duration_seconds=time.perf_counter() - started)
        return result

    def _should_raise(self, *, optional: bool) -> bool:
        if optional:
            return False
        return self.config.fail_fast or (not self.config.best_effort)

    def _analyzer_kwargs(self, analyzer_id: str) -> dict[str, Any]:
        kwargs = dict(self.config.analyzer_params.get(analyzer_id, {}))
        if "backend" not in kwargs and analyzer_id in self.config.backend_overrides:
            kwargs["backend"] = self.config.backend_overrides[analyzer_id]
        return kwargs

    def _initialize_frame_steps(self, steps: dict[str, AnalysisStepStatus]) -> dict[str, dict[str, Any]]:
        runtime: dict[str, dict[str, Any]] = {}

        if COMBINED_FRAME_ANALYZER in self.config.enabled_analyzers:
            combined_optional = COMBINED_FRAME_ANALYZER in self.config.optional_analyzers
            analyzer = self._create_analyzer(
                CombinedFrameAnalyzer,
                analyzer_id=COMBINED_FRAME_ANALYZER,
                steps=steps,
                optional=combined_optional,
            )
            if analyzer is not None:
                runtime[COMBINED_FRAME_ANALYZER] = self._frame_runtime(analyzer, optional=combined_optional)

        suppress_overlapping = self.config.prefer_combined_frame_analyzer and COMBINED_FRAME_ANALYZER in runtime

        for analyzer_id, analyzer_cls in (
            (OBJECT_DETECTOR, ObjectDetector),
            (TEXT_DETECTOR, TextDetector),
            (SHOT_TYPE_CLASSIFIER, ShotTypeClassifier),
        ):
            if analyzer_id not in self.config.enabled_analyzers:
                continue
            if suppress_overlapping:
                steps[analyzer_id] = AnalysisStepStatus(
                    status="skipped",
                    warning=f"Suppressed by {COMBINED_FRAME_ANALYZER} precedence",
                )
                continue

            optional = analyzer_id in self.config.optional_analyzers
            analyzer = self._create_analyzer(analyzer_cls, analyzer_id=analyzer_id, steps=steps, optional=optional)
            if analyzer is not None:
                runtime[analyzer_id] = self._frame_runtime(analyzer, optional=optional)

        for analyzer_id, analyzer_cls2 in (
            (FACE_DETECTOR, FaceDetector),
            (IMAGE_TO_TEXT, ImageToText),
            (CAMERA_MOTION_DETECTOR, CameraMotionDetector),
        ):
            if analyzer_id not in self.config.enabled_analyzers:
                continue
            optional = analyzer_id in self.config.optional_analyzers
            analyzer = self._create_analyzer(analyzer_cls2, analyzer_id=analyzer_id, steps=steps, optional=optional)
            if analyzer is not None:
                runtime[analyzer_id] = self._frame_runtime(analyzer, optional=optional)

        return runtime

    def _frame_runtime(self, analyzer: Any, *, optional: bool) -> dict[str, Any]:
        return {
            "analyzer": analyzer,
            "optional": optional,
            "started": time.perf_counter(),
            "processed": 0,
            "error": None,
        }

    def _create_analyzer(
        self,
        analyzer_cls: Any,
        *,
        analyzer_id: str,
        steps: dict[str, AnalysisStepStatus],
        optional: bool,
    ) -> Any | None:
        started = time.perf_counter()
        try:
            analyzer = analyzer_cls(**self._analyzer_kwargs(analyzer_id))
        except Exception as exc:
            duration = time.perf_counter() - started
            status = "skipped" if optional else "failed"
            steps[analyzer_id] = AnalysisStepStatus(status=status, duration_seconds=duration, error=str(exc))
            if self._should_raise(optional=optional):
                raise
            return None
        return analyzer

    def _finalize_frame_steps(
        self,
        *,
        frame_steps_runtime: dict[str, dict[str, Any]],
        steps: dict[str, AnalysisStepStatus],
    ) -> None:
        for analyzer_id, runtime in frame_steps_runtime.items():
            duration = time.perf_counter() - runtime["started"]
            error = runtime["error"]
            optional = bool(runtime["optional"])
            processed = int(runtime["processed"])
            if error is not None:
                status = "skipped" if optional else "failed"
                steps[analyzer_id] = AnalysisStepStatus(
                    status=status,
                    duration_seconds=duration,
                    error=error,
                    details={"processed_samples": processed},
                )
            else:
                steps[analyzer_id] = AnalysisStepStatus(
                    status="succeeded",
                    duration_seconds=duration,
                    details={"processed_samples": processed},
                )

    def _process_path_samples(
        self,
        *,
        path: Path,
        metadata: VideoMetadata,
        frame_indices: list[int],
        frame_steps_runtime: dict[str, dict[str, Any]],
        steps: dict[str, AnalysisStepStatus],
    ) -> tuple[list[FrameAnalysisSample], list[CameraMotionSample]]:
        sample_count = len(frame_indices)
        density = sample_count / max(metadata.frame_count, 1)
        use_streaming = density >= 0.20

        if use_streaming:
            return self._process_path_samples_streaming(
                path=path,
                metadata=metadata,
                frame_indices=frame_indices,
                frame_steps_runtime=frame_steps_runtime,
                steps=steps,
            )
        return self._process_path_samples_chunked(
            path=path,
            metadata=metadata,
            frame_indices=frame_indices,
            frame_steps_runtime=frame_steps_runtime,
            steps=steps,
        )

    def _process_path_samples_streaming(
        self,
        *,
        path: Path,
        metadata: VideoMetadata,
        frame_indices: list[int],
        frame_steps_runtime: dict[str, dict[str, Any]],
        steps: dict[str, AnalysisStepStatus],
    ) -> tuple[list[FrameAnalysisSample], list[CameraMotionSample]]:
        samples: list[FrameAnalysisSample] = []
        camera_samples: list[CameraMotionSample] = []
        if not frame_indices:
            return samples, camera_samples

        target_indices = sorted(frame_indices)
        target_pos = 0

        start_second = target_indices[0] / metadata.fps
        end_second = min(metadata.total_seconds, (target_indices[-1] + 1) / metadata.fps)

        camera_window: deque[tuple[int, float, np.ndarray]] = deque(maxlen=max(self.config.camera_motion_stride, 1) + 1)

        with FrameIterator(path, start_second=start_second, end_second=end_second) as iterator:
            for frame_idx, frame in iterator:
                while target_pos < len(target_indices) and target_indices[target_pos] < frame_idx:
                    target_pos += 1
                if target_pos >= len(target_indices):
                    break
                if frame_idx != target_indices[target_pos]:
                    continue

                sample, camera_sample = self._analyze_sampled_frame(
                    frame=frame,
                    frame_index=frame_idx,
                    fps=metadata.fps,
                    frame_steps_runtime=frame_steps_runtime,
                    camera_window=camera_window,
                    steps=steps,
                )
                samples.append(sample)
                if camera_sample is not None:
                    camera_samples.append(camera_sample)

                target_pos += 1
                if target_pos >= len(target_indices):
                    break

        return samples, camera_samples

    def _process_path_samples_chunked(
        self,
        *,
        path: Path,
        metadata: VideoMetadata,
        frame_indices: list[int],
        frame_steps_runtime: dict[str, dict[str, Any]],
        steps: dict[str, AnalysisStepStatus],
    ) -> tuple[list[FrameAnalysisSample], list[CameraMotionSample]]:
        samples: list[FrameAnalysisSample] = []
        camera_samples: list[CameraMotionSample] = []
        camera_window: deque[tuple[int, float, np.ndarray]] = deque(maxlen=max(self.config.camera_motion_stride, 1) + 1)

        chunk_size = max(1, int(self.config.frame_chunk_size))
        for chunk_start in range(0, len(frame_indices), chunk_size):
            chunk_indices = frame_indices[chunk_start : chunk_start + chunk_size]
            chunk_frames = extract_frames_at_indices(path, chunk_indices)
            for i, frame in enumerate(chunk_frames):
                frame_idx = chunk_indices[i]
                sample, camera_sample = self._analyze_sampled_frame(
                    frame=frame,
                    frame_index=frame_idx,
                    fps=metadata.fps,
                    frame_steps_runtime=frame_steps_runtime,
                    camera_window=camera_window,
                    steps=steps,
                )
                samples.append(sample)
                if camera_sample is not None:
                    camera_samples.append(camera_sample)

        return samples, camera_samples

    def _process_video_samples(
        self,
        *,
        video: Video | None,
        frame_indices: list[int],
        frame_steps_runtime: dict[str, dict[str, Any]],
        steps: dict[str, AnalysisStepStatus],
    ) -> tuple[list[FrameAnalysisSample], list[CameraMotionSample]]:
        if video is None:
            return [], []

        samples: list[FrameAnalysisSample] = []
        camera_samples: list[CameraMotionSample] = []
        camera_window: deque[tuple[int, float, np.ndarray]] = deque(maxlen=max(self.config.camera_motion_stride, 1) + 1)

        for frame_idx in frame_indices:
            sample, camera_sample = self._analyze_sampled_frame(
                frame=video.frames[frame_idx],
                frame_index=frame_idx,
                fps=video.fps,
                frame_steps_runtime=frame_steps_runtime,
                camera_window=camera_window,
                steps=steps,
            )
            samples.append(sample)
            if camera_sample is not None:
                camera_samples.append(camera_sample)

        return samples, camera_samples

    def _analyze_sampled_frame(
        self,
        *,
        frame: np.ndarray,
        frame_index: int,
        fps: float,
        frame_steps_runtime: dict[str, dict[str, Any]],
        camera_window: deque[tuple[int, float, np.ndarray]],
        steps: dict[str, AnalysisStepStatus],
    ) -> tuple[FrameAnalysisSample, CameraMotionSample | None]:
        timestamp = round(frame_index / fps, 6)
        sample = FrameAnalysisSample(
            timestamp=timestamp,
            frame_index=frame_index,
            objects=[],
            faces=[],
            text=[],
        )
        step_results: dict[str, str] = {}

        for analyzer_id in (
            COMBINED_FRAME_ANALYZER,
            OBJECT_DETECTOR,
            FACE_DETECTOR,
            TEXT_DETECTOR,
            SHOT_TYPE_CLASSIFIER,
            IMAGE_TO_TEXT,
        ):
            runtime = frame_steps_runtime.get(analyzer_id)
            if runtime is None or runtime["error"] is not None:
                continue

            analyzer = runtime["analyzer"]
            try:
                if analyzer_id == COMBINED_FRAME_ANALYZER:
                    combined = analyzer.analyze(frame)
                    sample.combined = combined
                    sample.objects = combined.detected_objects
                    sample.text = combined.detected_text
                    sample.shot_type = combined.shot_type
                elif analyzer_id == OBJECT_DETECTOR:
                    sample.objects = analyzer.detect(frame)
                elif analyzer_id == FACE_DETECTOR:
                    sample.faces = analyzer.detect(frame)
                elif analyzer_id == TEXT_DETECTOR:
                    sample.text = analyzer.detect(frame)
                elif analyzer_id == SHOT_TYPE_CLASSIFIER:
                    sample.shot_type = analyzer.classify(frame)
                elif analyzer_id == IMAGE_TO_TEXT:
                    sample.image_caption = analyzer.describe_image(frame)

                runtime["processed"] += 1
                step_results[analyzer_id] = "ok"
            except Exception as exc:
                runtime["error"] = str(exc)
                step_results[analyzer_id] = "error"
                if self._should_raise(optional=bool(runtime["optional"])):
                    raise

        camera_sample: CameraMotionSample | None = None
        camera_runtime = frame_steps_runtime.get(CAMERA_MOTION_DETECTOR)
        if camera_runtime is not None and camera_runtime["error"] is None:
            camera_window.append((frame_index, timestamp, frame))
            stride = max(int(self.config.camera_motion_stride), 1)
            if len(camera_window) >= stride + 1:
                first_idx, first_timestamp, first_frame = camera_window[0]
                _, last_timestamp, last_frame = camera_window[-1]
                try:
                    label = camera_runtime["analyzer"].detect(first_frame, last_frame)
                    camera_runtime["processed"] += 1
                    step_results[CAMERA_MOTION_DETECTOR] = "ok"
                    camera_sample = CameraMotionSample(start=first_timestamp, end=last_timestamp, label=label)
                except Exception as exc:
                    camera_runtime["error"] = str(exc)
                    step_results[CAMERA_MOTION_DETECTOR] = "error"
                    if self._should_raise(optional=bool(camera_runtime["optional"])):
                        raise

        sample.objects = sample.objects if sample.objects is not None else []
        sample.faces = sample.faces if sample.faces is not None else []
        sample.text = sample.text if sample.text is not None else []

        # Keep payload compact.
        if step_results:
            sample.step_results = step_results

        return sample, camera_sample

    def _choose_frame_access_mode(self, sampled_frames: int, total_frames: int) -> str:
        if total_frames <= 0:
            return "chunked"
        density = sampled_frames / total_frames
        return "streaming" if density >= 0.20 else "chunked"

    def _plan_frame_indices(self, *, metadata: VideoMetadata, scenes: list[SceneBoundary]) -> list[int]:
        if metadata.frame_count <= 0:
            return []

        mode = self.config.frame_sampling_mode
        sampled: set[int] = set()

        if mode in {"uniform", "hybrid"}:
            fps = max(float(self.config.frames_per_second), 0.0)
            if fps > 0:
                interval = max(int(round(metadata.fps / fps)), 1)
                sampled.update(range(0, metadata.frame_count, interval))

        if mode in {"scene_boundary", "scene_representative", "hybrid"} and scenes:
            if self.config.include_scene_boundaries:
                sampled.update(max(0, min(metadata.frame_count - 1, scene.start_frame)) for scene in scenes)

            if mode in {"scene_representative", "hybrid"}:
                offset = min(max(self.config.scene_representative_offset, 0.0), 1.0)
                for scene in scenes:
                    span = max(scene.end_frame - scene.start_frame, 1)
                    representative = scene.start_frame + int(round(offset * (span - 1)))
                    sampled.add(max(0, min(metadata.frame_count - 1, representative)))

        if not sampled:
            sampled.add(0)

        ordered = sorted(sampled)
        max_frames = self.config.max_frames
        if max_frames is not None and len(ordered) > max_frames:
            indices = np.linspace(0, len(ordered) - 1, max_frames, dtype=int)
            ordered = [ordered[i] for i in indices]

        return ordered

    def _build_source_from_path(self, path: Path, metadata: VideoMetadata) -> VideoAnalysisSource:
        tags = self._extract_source_tags(path)
        creation_time = _normalize_creation_time(next((tags[k] for k in _CREATION_TIME_TAG_KEYS if k in tags), None))

        geo: GeoMetadata | None = None
        if self.config.include_geo and not self.config.redact_geo:
            geo = _parse_geo_metadata(tags)

        title = tags.get("title") or path.stem

        return VideoAnalysisSource(
            title=title,
            path=str(path),
            filename=path.name,
            duration=metadata.total_seconds,
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
            frame_count=metadata.frame_count,
            creation_time=creation_time,
            geo=geo,
            raw_tags=tags or None,
        )

    def _build_source_from_video(
        self,
        *,
        video: Video,
        source_path: str | Path | None,
        metadata: VideoMetadata,
    ) -> VideoAnalysisSource:
        path_obj = Path(source_path) if source_path is not None else None
        tags = self._extract_source_tags(path_obj) if path_obj else {}
        creation_time = _normalize_creation_time(next((tags[k] for k in _CREATION_TIME_TAG_KEYS if k in tags), None))

        geo: GeoMetadata | None = None
        if self.config.include_geo and not self.config.redact_geo:
            geo = _parse_geo_metadata(tags)

        title = tags.get("title") if tags else None
        if title is None and path_obj is not None:
            title = path_obj.stem

        return VideoAnalysisSource(
            title=title,
            path=str(path_obj) if path_obj else None,
            filename=path_obj.name if path_obj else None,
            duration=video.total_seconds,
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
            frame_count=metadata.frame_count,
            creation_time=creation_time,
            geo=geo,
            raw_tags=tags or None,
        )

    def _extract_source_tags(self, path: Path | None) -> dict[str, str]:
        if path is None:
            return {}

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags:stream_tags",
            "-of",
            "json",
            str(path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            payload = json.loads(result.stdout)
        except Exception:
            return {}

        tags: dict[str, str] = {}

        format_tags = payload.get("format", {}).get("tags", {})
        if isinstance(format_tags, dict):
            tags.update({str(k).lower(): str(v) for k, v in format_tags.items()})

        for stream in payload.get("streams", []):
            stream_tags = stream.get("tags", {})
            if not isinstance(stream_tags, dict):
                continue
            for key, value in stream_tags.items():
                lowered = str(key).lower()
                tags.setdefault(lowered, str(value))

        return tags


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _library_version() -> str | None:
    try:
        return importlib_metadata.version("videopython")
    except importlib_metadata.PackageNotFoundError:
        return None


def _require_video(video: Video | None) -> Video:
    if video is None:
        raise ValueError("Video input is required for in-memory analysis")
    return video


def _normalize_creation_time(value: str | None) -> str | None:
    if value is None:
        return None

    raw = value.strip()
    if not raw:
        return None

    # Common ffprobe timezone suffix.
    candidate = raw.replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return raw

    if parsed.tzinfo is None:
        return parsed.isoformat()

    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_geo_metadata(tags: dict[str, str]) -> GeoMetadata | None:
    for key in _GEO_TAG_KEYS:
        value = tags.get(key)
        if not value:
            continue
        geo = _parse_iso6709_or_pair(value)
        if geo is not None:
            geo.source = key
            return geo
    return None


def _parse_iso6709_or_pair(value: str) -> GeoMetadata | None:
    # ISO 6709, commonly used by QuickTime, e.g. +37.3317-122.0307+005.0/
    iso6709_match = re.match(
        r"^\s*([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?/?\s*$",
        value,
    )
    if iso6709_match:
        lat = float(iso6709_match.group(1))
        lon = float(iso6709_match.group(2))
        alt = float(iso6709_match.group(3)) if iso6709_match.group(3) is not None else None
        return GeoMetadata(latitude=lat, longitude=lon, altitude=alt)

    # Generic "lat,lon" fallback.
    pair_match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(?:\s*,\s*(-?\d+(?:\.\d+)?))?\s*$", value)
    if pair_match:
        lat = float(pair_match.group(1))
        lon = float(pair_match.group(2))
        alt = float(pair_match.group(3)) if pair_match.group(3) is not None else None
        return GeoMetadata(latitude=lat, longitude=lon, altitude=alt)

    return None
