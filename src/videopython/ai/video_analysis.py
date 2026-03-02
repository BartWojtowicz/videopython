from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from videopython.ai.understanding import (
    ActionRecognizer,
    AudioClassifier,
    AudioToText,
    SceneVLM,
    SemanticSceneDetector,
)
from videopython.ai.understanding.image import DEFAULT_SCENE_VLM_MODEL_SIZE, SCENE_VLM_MODEL_IDS
from videopython.base.audio import Audio
from videopython.base.description import (
    AudioClassification,
    AudioEvent,
    DetectedAction,
    DetectedObject,
    SceneBoundary,
)
from videopython.base.text.transcription import Transcription
from videopython.base.video import Video, VideoMetadata, extract_frames_at_times

__all__ = ["VideoAnalysis", "VideoAnalysisConfig", "VideoAnalyzer"]

AUDIO_TO_TEXT = "audio_to_text"
AUDIO_CLASSIFIER = "audio_classifier"
SEMANTIC_SCENE_DETECTOR = "semantic_scene_detector"
ACTION_RECOGNIZER = "action_recognizer"
SCENE_VLM = "scene_vlm"

ALL_ANALYZER_IDS: tuple[str, ...] = (
    AUDIO_TO_TEXT,
    AUDIO_CLASSIFIER,
    SEMANTIC_SCENE_DETECTOR,
    ACTION_RECOGNIZER,
    SCENE_VLM,
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

# Hard-coded scene VLM defaults by design (not user config).
_SCENE_VLM_MODEL_SIZE = DEFAULT_SCENE_VLM_MODEL_SIZE
_SCENE_VLM_MAX_SEGMENT_SECONDS = 10.0
_SCENE_VLM_FRAMES_PER_SEGMENT = 2


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
    """Runtime/provenance metadata for a full analysis run."""

    created_at: str
    mode: str
    library_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "mode": self.mode,
            "library_version": self.library_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisRunInfo":
        return cls(
            created_at=data["created_at"],
            mode=data["mode"],
            library_version=data.get("library_version"),
        )


@dataclass
class VideoAnalysisConfig:
    """Minimal execution config for scene-first analysis runs."""

    enabled_analyzers: set[str] = field(default_factory=lambda: set(ALL_ANALYZER_IDS))

    def __post_init__(self) -> None:
        unknown_enabled = sorted(set(self.enabled_analyzers) - set(ALL_ANALYZER_IDS))
        if unknown_enabled:
            raise ValueError(f"Unknown analyzer ids in enabled_analyzers: {unknown_enabled}")

    @classmethod
    def rich_understanding_preset(cls) -> "VideoAnalysisConfig":
        """Backward-compatible alias for the default config."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled_analyzers": sorted(self.enabled_analyzers),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoAnalysisConfig":
        enabled_raw = data.get("enabled_analyzers")
        return cls(enabled_analyzers=set(enabled_raw) if enabled_raw is not None else set(ALL_ANALYZER_IDS))


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
class SceneVisualSegment:
    """Chunk-level visual understanding output inside one scene."""

    start_second: float
    end_second: float
    caption: str | None = None
    objects: list[DetectedObject] = field(default_factory=list)
    text: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_second": self.start_second,
            "end_second": self.end_second,
            "caption": self.caption,
            "objects": [obj.to_dict() for obj in self.objects],
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneVisualSegment":
        return cls(
            start_second=float(data["start_second"]),
            end_second=float(data["end_second"]),
            caption=data.get("caption"),
            objects=[DetectedObject.from_dict(item) for item in data.get("objects", [])],
            text=[str(item) for item in data.get("text", [])],
        )


@dataclass
class SceneAnalysisSample:
    """Flat scene payload with all per-scene analyzer outputs."""

    scene_index: int
    start_second: float
    end_second: float
    start_frame: int | None = None
    end_frame: int | None = None
    visual_segments: list[SceneVisualSegment] = field(default_factory=list)
    actions: list[DetectedAction] = field(default_factory=list)
    audio_classification: AudioClassification | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_index": self.scene_index,
            "start_second": self.start_second,
            "end_second": self.end_second,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "visual_segments": [segment.to_dict() for segment in self.visual_segments],
            "actions": [action.to_dict() for action in self.actions],
            "audio_classification": self.audio_classification.to_dict() if self.audio_classification else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneAnalysisSample":
        return cls(
            scene_index=int(data["scene_index"]),
            start_second=float(data["start_second"]),
            end_second=float(data["end_second"]),
            start_frame=data.get("start_frame"),
            end_frame=data.get("end_frame"),
            visual_segments=[SceneVisualSegment.from_dict(item) for item in data.get("visual_segments", [])],
            actions=[DetectedAction.from_dict(item) for item in data.get("actions", [])],
            audio_classification=(
                AudioClassification.from_dict(data["audio_classification"])
                if data.get("audio_classification")
                else None
            ),
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


class VideoAnalyzer:
    """Orchestrates scene-first analyzers and builds `VideoAnalysis` output."""

    def __init__(self, config: VideoAnalysisConfig | None = None):
        self.config = config or VideoAnalysisConfig()

    def analyze_path(self, path: str | Path) -> VideoAnalysis:
        """Analyze a video path in scene-first mode."""
        path_obj = Path(path)
        metadata = VideoMetadata.from_path(path_obj)
        source = self._build_source(
            metadata=metadata,
            path_obj=path_obj,
            duration_seconds=metadata.total_seconds,
            title_fallback=path_obj.stem,
        )
        return self._analyze(video=None, source_path=path_obj, metadata=metadata, source=source)

    def analyze(self, video: Video, *, source_path: str | Path | None = None) -> VideoAnalysis:
        """Analyze an in-memory `Video` object."""
        path_obj = Path(source_path) if source_path else None
        metadata = VideoMetadata.from_video(video)
        source = self._build_source(
            metadata=metadata,
            path_obj=path_obj,
            duration_seconds=video.total_seconds,
            title_fallback=path_obj.stem if path_obj is not None else None,
        )
        return self._analyze(
            video=video,
            source_path=path_obj,
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

        enabled = self.config.enabled_analyzers

        run_info = AnalysisRunInfo(
            created_at=_utc_now_iso(),
            mode=mode,
            library_version=_library_version(),
        )

        transcription: Transcription | None = None
        if AUDIO_TO_TEXT in enabled:
            try:
                transcription = AudioToText().transcribe(
                    Audio.from_path(source_path) if source_path is not None else _require_video(video)
                )
            except Exception:
                transcription = None

        scenes = self._default_scene_boundaries(metadata)
        if SEMANTIC_SCENE_DETECTOR in enabled:
            detected: list[SceneBoundary] | None = None
            try:
                scene_detector = SemanticSceneDetector()
                detected = (
                    scene_detector.detect_streaming(source_path)
                    if source_path is not None
                    else scene_detector.detect(_require_video(video))
                )
            except Exception:
                detected = None
            if detected is not None:
                scenes = self._normalize_scene_boundaries(detected, metadata)

        if not scenes:
            scenes = self._default_scene_boundaries(metadata)

        scene_section = self._analyze_scenes(
            source_path=source_path,
            video=video,
            metadata=metadata,
            scenes=scenes,
        )

        audio_section = AudioAnalysisSection(transcription=transcription) if transcription is not None else None

        return VideoAnalysis(
            source=source,
            config=self.config,
            run_info=run_info,
            audio=audio_section,
            scenes=scene_section if scene_section.samples else None,
        )

    def _analyze_scenes(
        self,
        *,
        source_path: Path | None,
        video: Video | None,
        metadata: VideoMetadata,
        scenes: list[SceneBoundary],
    ) -> SceneAnalysisSection:
        enabled = self.config.enabled_analyzers

        try:
            scene_vlm = (
                SceneVLM(model_name=SCENE_VLM_MODEL_IDS[_SCENE_VLM_MODEL_SIZE]) if SCENE_VLM in enabled else None
            )
        except Exception:
            scene_vlm = None

        try:
            action_recognizer = ActionRecognizer() if ACTION_RECOGNIZER in enabled else None
        except Exception:
            action_recognizer = None

        try:
            audio_classifier = AudioClassifier() if AUDIO_CLASSIFIER in enabled else None
        except Exception:
            audio_classifier = None

        path_audio: Audio | None = None
        if audio_classifier is not None and source_path is not None:
            try:
                path_audio = Audio.from_path(source_path)
            except Exception:
                path_audio = None

        samples: list[SceneAnalysisSample] = []
        for index, scene in enumerate(scenes):
            sample = SceneAnalysisSample(
                scene_index=index,
                start_second=float(scene.start),
                end_second=float(scene.end),
                start_frame=int(scene.start_frame),
                end_frame=int(scene.end_frame),
            )

            scene_clip: Video | None = None
            needs_scene_clip = (action_recognizer is not None and source_path is None) or (
                audio_classifier is not None and path_audio is None
            )
            if needs_scene_clip:
                try:
                    scene_clip = self._load_scene_video_clip(
                        source_path=source_path,
                        video=video,
                        start_second=scene.start,
                        end_second=scene.end,
                    )
                except Exception:
                    scene_clip = None

            if scene_vlm is not None:
                try:
                    sample.visual_segments = self._run_scene_vlm(
                        scene_vlm=scene_vlm,
                        source_path=source_path,
                        video=video,
                        metadata=metadata,
                        start_second=scene.start,
                        end_second=scene.end,
                    )
                except Exception:
                    pass

            if action_recognizer is not None:
                try:
                    sample.actions = self._run_scene_actions(
                        action_recognizer=action_recognizer,
                        source_path=source_path,
                        scene_clip=scene_clip,
                        scene_start=scene.start,
                        scene_end=scene.end,
                        start_frame_offset=scene.start_frame,
                    )
                except Exception:
                    pass

            if audio_classifier is not None:
                try:
                    sample.audio_classification = self._run_scene_audio_classification(
                        audio_classifier=audio_classifier,
                        path_audio=path_audio,
                        scene_clip=scene_clip,
                        scene_start=scene.start,
                        scene_end=scene.end,
                    )
                except Exception:
                    pass

            samples.append(sample)

        return SceneAnalysisSection(samples=samples)

    def _run_scene_vlm(
        self,
        *,
        scene_vlm: SceneVLM,
        source_path: Path | None,
        video: Video | None,
        metadata: VideoMetadata,
        start_second: float,
        end_second: float,
    ) -> list[SceneVisualSegment]:
        segments: list[SceneVisualSegment] = []
        for window_start, window_end in self._scene_vlm_windows(start_second, end_second):
            frames = self._sample_scene_frames(
                source_path=source_path,
                video=video,
                metadata=metadata,
                start_second=window_start,
                end_second=window_end,
            )
            if not frames:
                continue

            result = scene_vlm.analyze_scene(frames)
            caption = _normalize_text(getattr(result, "caption", None))

            objects_by_label: dict[str, DetectedObject] = {}
            for obj in getattr(result, "objects", []) or []:
                key = _normalize_text(obj.label).lower()
                if not key:
                    continue
                existing = objects_by_label.get(key)
                if existing is None or float(obj.confidence) > float(existing.confidence):
                    objects_by_label[key] = DetectedObject(
                        label=_normalize_text(obj.label) or obj.label,
                        confidence=float(obj.confidence),
                        bounding_box=obj.bounding_box,
                    )

            text_tokens: list[str] = []
            seen_text: set[str] = set()
            for token in getattr(result, "text", []) or []:
                normalized = _normalize_text(token)
                if not normalized:
                    continue
                lowered = normalized.lower()
                if lowered in seen_text:
                    continue
                seen_text.add(lowered)
                text_tokens.append(normalized)

            objects = sorted(objects_by_label.values(), key=lambda item: float(item.confidence), reverse=True)
            segments.append(
                SceneVisualSegment(
                    start_second=round(window_start, 6),
                    end_second=round(window_end, 6),
                    caption=caption or None,
                    objects=objects,
                    text=text_tokens,
                )
            )
        return segments

    def _run_scene_actions(
        self,
        *,
        action_recognizer: ActionRecognizer,
        source_path: Path | None,
        scene_clip: Video | None,
        scene_start: float,
        scene_end: float,
        start_frame_offset: int | None,
    ) -> list[DetectedAction]:
        if scene_end <= scene_start:
            return []

        if source_path is not None:
            return action_recognizer.recognize_path(
                source_path,
                start_second=scene_start,
                end_second=scene_end,
            )

        if scene_clip is None:
            return []

        clip_actions = action_recognizer.recognize(scene_clip)
        offset = int(start_frame_offset or 0)
        for action in clip_actions:
            if action.start_frame is not None:
                action.start_frame = action.start_frame + offset
            if action.end_frame is not None:
                action.end_frame = action.end_frame + offset
            if action.start_time is not None:
                action.start_time = scene_start + action.start_time
            if action.end_time is not None:
                action.end_time = scene_start + action.end_time
        return clip_actions

    def _run_scene_audio_classification(
        self,
        *,
        audio_classifier: AudioClassifier,
        path_audio: Audio | None,
        scene_clip: Video | None,
        scene_start: float,
        scene_end: float,
    ) -> AudioClassification | None:
        if scene_end <= scene_start:
            return None

        if path_audio is not None:
            scene_media: Audio | Video = path_audio.slice(start_seconds=scene_start, end_seconds=scene_end)
        elif scene_clip is not None:
            scene_media = scene_clip
        else:
            return None

        classification = audio_classifier.classify(scene_media)
        offset_events = [
            AudioEvent(
                start=scene_start + event.start,
                end=scene_start + event.end,
                label=event.label,
                confidence=event.confidence,
            )
            for event in classification.events
        ]
        return AudioClassification(events=offset_events, clip_predictions=classification.clip_predictions)

    def _scene_vlm_windows(self, start_second: float, end_second: float) -> list[tuple[float, float]]:
        if end_second <= start_second:
            return []

        windows: list[tuple[float, float]] = []
        cursor = float(start_second)
        max_window = float(_SCENE_VLM_MAX_SEGMENT_SECONDS)
        while cursor < end_second:
            window_end = min(end_second, cursor + max_window)
            windows.append((cursor, window_end))
            cursor = window_end
        return windows

    def _sample_scene_frames(
        self,
        *,
        source_path: Path | None,
        video: Video | None,
        metadata: VideoMetadata,
        start_second: float,
        end_second: float,
    ) -> list[np.ndarray | Image.Image]:
        timestamps = self._sample_timestamps(start_second=start_second, end_second=end_second)
        if not timestamps:
            return []

        if source_path is not None:
            sampled_frames: list[np.ndarray | Image.Image] = []
            sampled_frames.extend(extract_frames_at_times(source_path, timestamps))
            return sampled_frames

        current_video = _require_video(video)
        max_frame = max(len(current_video.frames) - 1, 0)
        indices = [max(0, min(max_frame, int(ts * metadata.fps))) for ts in timestamps]
        in_memory_frames: list[np.ndarray | Image.Image] = []
        in_memory_frames.extend(current_video.frames[idx] for idx in indices)
        return in_memory_frames

    def _sample_timestamps(self, *, start_second: float, end_second: float) -> list[float]:
        duration = max(0.0, end_second - start_second)
        if duration <= 0.0:
            return []

        frame_count = max(1, int(_SCENE_VLM_FRAMES_PER_SEGMENT))
        if frame_count == 1:
            return [start_second + (duration * 0.5)]

        # Center samples inside the interval so we avoid exact boundaries.
        step = duration / float(frame_count + 1)
        timestamps = [start_second + (step * (idx + 1)) for idx in range(frame_count)]
        epsilon = 1e-3
        return [min(end_second - epsilon, max(start_second + epsilon, ts)) for ts in timestamps]

    def _load_scene_video_clip(
        self,
        *,
        source_path: Path | None,
        video: Video | None,
        start_second: float,
        end_second: float,
    ) -> Video | None:
        if end_second <= start_second:
            return None
        if source_path is not None:
            return Video.from_path(str(source_path), start_second=start_second, end_second=end_second)
        return _require_video(video).cut(start_second, end_second)

    def _default_scene_boundaries(self, metadata: VideoMetadata) -> list[SceneBoundary]:
        if metadata.total_seconds <= 0 or metadata.frame_count <= 0:
            return []
        return [
            SceneBoundary(
                start=0.0,
                end=float(metadata.total_seconds),
                start_frame=0,
                end_frame=int(metadata.frame_count),
            )
        ]

    def _normalize_scene_boundaries(self, scenes: list[SceneBoundary], metadata: VideoMetadata) -> list[SceneBoundary]:
        normalized: list[SceneBoundary] = []
        max_time = float(metadata.total_seconds)
        max_frame = int(metadata.frame_count)

        for item in scenes:
            start = max(0.0, min(max_time, float(item.start)))
            end = max(0.0, min(max_time, float(item.end)))
            if end <= start:
                continue

            start_frame = int(item.start_frame)
            end_frame = int(item.end_frame)
            start_frame = max(0, min(max_frame, start_frame))
            end_frame = max(0, min(max_frame, end_frame))
            if end_frame <= start_frame:
                start_frame = int(round(start * metadata.fps))
                end_frame = max(start_frame + 1, int(round(end * metadata.fps)))
                start_frame = max(0, min(max_frame, start_frame))
                end_frame = max(0, min(max_frame, end_frame))
                if end_frame <= start_frame:
                    continue

            normalized.append(
                SceneBoundary(
                    start=round(start, 6),
                    end=round(end, 6),
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

        normalized.sort(key=lambda scene: (scene.start, scene.end))
        return normalized

    def _build_source(
        self,
        *,
        metadata: VideoMetadata,
        path_obj: Path | None,
        duration_seconds: float,
        title_fallback: str | None,
    ) -> VideoAnalysisSource:
        tags = self._extract_source_tags(path_obj) if path_obj else {}
        creation_time = _normalize_creation_time(
            next((tags[key] for key in _CREATION_TIME_TAG_KEYS if key in tags), None)
        )
        geo = _parse_geo_metadata(tags)
        title = tags.get("title") or title_fallback

        return VideoAnalysisSource(
            title=title,
            path=str(path_obj) if path_obj else None,
            filename=path_obj.name if path_obj else None,
            duration=duration_seconds,
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
    iso6709_match = re.match(
        r"^\s*([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?/?\s*$",
        value,
    )
    if iso6709_match:
        lat = float(iso6709_match.group(1))
        lon = float(iso6709_match.group(2))
        alt = float(iso6709_match.group(3)) if iso6709_match.group(3) is not None else None
        return GeoMetadata(latitude=lat, longitude=lon, altitude=alt)

    pair_match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(?:\s*,\s*(-?\d+(?:\.\d+)?))?\s*$", value)
    if pair_match:
        lat = float(pair_match.group(1))
        lon = float(pair_match.group(2))
        alt = float(pair_match.group(3)) if pair_match.group(3) is not None else None
        return GeoMetadata(latitude=lat, longitude=lon, altitude=alt)

    return None


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()
