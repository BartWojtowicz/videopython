from __future__ import annotations

import gc
import json
import logging
import math
import re
import subprocess
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from PIL import Image

from videopython.ai.understanding import (
    AudioClassifier,
    AudioToText,
    SceneVLM,
    SemanticSceneDetector,
)
from videopython.base.audio import Audio
from videopython.base.description import (
    AudioClassification,
    AudioEvent,
    SceneBoundary,
)
from videopython.base.text.transcription import Transcription
from videopython.base.video import Video, VideoMetadata, extract_frames_at_times

__all__ = ["VideoAnalysis", "VideoAnalysisConfig", "VideoAnalyzer"]

logger = logging.getLogger(__name__)

AUDIO_TO_TEXT = "audio_to_text"
AUDIO_CLASSIFIER = "audio_classifier"
SEMANTIC_SCENE_DETECTOR = "semantic_scene_detector"
SCENE_VLM = "scene_vlm"

ALL_ANALYZER_IDS: tuple[str, ...] = (
    AUDIO_TO_TEXT,
    AUDIO_CLASSIFIER,
    SEMANTIC_SCENE_DETECTOR,
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

_SCENE_VLM_FRAME_SCALE = 3.0  # controls log curve steepness for frame sampling
_SCENE_VLM_FRAME_BASE = 5.0  # seconds per unit in log formula
_SCENE_VLM_MAX_FRAMES = 30  # hard cap on frames per scene
_SCENE_VLM_GROUP_THRESHOLD = 10.0  # seconds; adjacent scenes shorter than this get merged for one VLM call


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
                "scene_vlm": {"model_size": "2b"},
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
    """Flat scene payload with all per-scene analyzer outputs."""

    scene_index: int
    start_second: float
    end_second: float
    start_frame: int | None = None
    end_frame: int | None = None
    caption: str | None = None
    audio_classification: AudioClassification | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_index": self.scene_index,
            "start_second": self.start_second,
            "end_second": self.end_second,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "caption": self.caption,
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
            caption=data.get("caption"),
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

        t_analysis_start = time.perf_counter()

        run_whisper = AUDIO_TO_TEXT in enabled
        run_scene_det = SEMANTIC_SCENE_DETECTOR in enabled

        transcription: Transcription | None = None
        detected: list[SceneBoundary] | None = None

        # Whisper and TransNetV2 operate on independent data (audio vs video
        # frames) and both fit comfortably in GPU memory together. Run them
        # concurrently via threads -- the GIL is released during GPU compute
        # and ffmpeg I/O so real parallelism is achieved.
        #
        # SceneVLM is loaded *after* Whisper/TransNetV2 finish (not concurrently)
        # because transformers' from_pretrained(torch_dtype="auto") mutates the
        # process-global torch.get_default_dtype() during model construction,
        # which corrupts Whisper's model weights if they're initialized at the
        # same time.
        if run_whisper and run_scene_det:
            transcription, detected = self._run_whisper_and_scene_detection(
                source_path=source_path, video=video, run_info=run_info
            )
        else:
            if run_whisper:
                with _record_stage(run_info, "whisper"):
                    transcription = self._run_whisper(source_path=source_path, video=video)

            if run_scene_det:
                with _record_stage(run_info, "scene_detection"):
                    detected = self._run_scene_detection(source_path=source_path, video=video)

        if run_scene_det:
            self._reset_transnetv2_torch_state()

        # Whisper and TransNetV2 are done -- free their GPU memory before
        # loading SceneVLM (~9GB). Python GC doesn't guarantee immediate
        # cleanup, so force it and release the CUDA cache.
        if run_whisper or run_scene_det:
            gc.collect()
            self._release_gpu_cache()

        scenes = self._default_scene_boundaries(metadata)
        if detected is not None:
            scenes = self._normalize_scene_boundaries(detected, metadata)

        if not scenes:
            scenes = self._default_scene_boundaries(metadata)

        with _record_stage(run_info, "scene_analysis"):
            scene_section = self._analyze_scenes(
                source_path=source_path,
                video=video,
                metadata=metadata,
                scenes=scenes,
                preloaded_scene_vlm=None,
                run_info=run_info,
            )

        audio_section = AudioAnalysisSection(transcription=transcription) if transcription is not None else None

        run_info.total_duration_seconds = time.perf_counter() - t_analysis_start
        logger.info("Total analysis completed in %.2fs", run_info.total_duration_seconds)
        return VideoAnalysis(
            source=source,
            config=self.config,
            run_info=run_info,
            audio=audio_section,
            scenes=scene_section if scene_section.samples else None,
        )

    def _run_whisper(self, *, source_path: Path | None, video: Video | None) -> Transcription | None:
        try:
            return AudioToText(**self.config.get_params(AUDIO_TO_TEXT)).transcribe(
                Audio.from_path(source_path) if source_path is not None else _require_video(video)
            )
        except Exception:
            logger.warning("AudioToText failed, skipping transcription", exc_info=True)
            return None

    def _run_scene_detection(self, *, source_path: Path | None, video: Video | None) -> list[SceneBoundary] | None:
        try:
            scene_detector = SemanticSceneDetector(**self.config.get_params(SEMANTIC_SCENE_DETECTOR))
            return (
                scene_detector.detect_streaming(source_path)
                if source_path is not None
                else scene_detector.detect(_require_video(video))
            )
        except Exception:
            logger.warning("SemanticSceneDetector failed, using default scene boundaries", exc_info=True)
            return None

    def _run_whisper_and_scene_detection(
        self, *, source_path: Path | None, video: Video | None, run_info: AnalysisRunInfo
    ) -> tuple[Transcription | None, list[SceneBoundary] | None]:
        with _record_stage(run_info, "whisper_and_scene_detection_parallel"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                whisper_future = pool.submit(
                    _run_with_stage, run_info, "whisper", self._run_whisper, source_path=source_path, video=video
                )
                scene_future = pool.submit(
                    _run_with_stage,
                    run_info,
                    "scene_detection",
                    self._run_scene_detection,
                    source_path=source_path,
                    video=video,
                )
                transcription = whisper_future.result()
                detected = scene_future.result()

        return transcription, detected

    @staticmethod
    def _reset_transnetv2_torch_state() -> None:
        """Reset global torch state that TransNetV2 sets during init.

        TransNetV2 sets torch.use_deterministic_algorithms(True) and
        cudnn.benchmark=False globally. Reset to defaults so subsequent
        models (especially SceneVLM's convolution-heavy ViT) can use cuDNN
        autotuner and non-deterministic kernels for better throughput.
        """
        try:
            import torch

            torch.use_deterministic_algorithms(False)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
        except ImportError:
            pass

    @staticmethod
    def _release_gpu_cache() -> None:
        """Force-release unused GPU memory back to the CUDA allocator."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _analyze_scenes(
        self,
        *,
        source_path: Path | None,
        video: Video | None,
        metadata: VideoMetadata,
        scenes: list[SceneBoundary],
        run_info: AnalysisRunInfo,
        preloaded_scene_vlm: SceneVLM | None = None,
    ) -> SceneAnalysisSection:
        enabled = self.config.enabled_analyzers

        scene_vlm: SceneVLM | None
        if preloaded_scene_vlm is not None:
            scene_vlm = preloaded_scene_vlm
        else:
            try:
                scene_vlm = SceneVLM(**self.config.get_params(SCENE_VLM)) if SCENE_VLM in enabled else None
            except Exception:
                logger.warning("Failed to initialize SceneVLM, skipping visual understanding", exc_info=True)
                scene_vlm = None

        try:
            audio_classifier = (
                AudioClassifier(**self.config.get_params(AUDIO_CLASSIFIER)) if AUDIO_CLASSIFIER in enabled else None
            )
        except Exception:
            logger.warning("Failed to initialize AudioClassifier, skipping audio classification", exc_info=True)
            audio_classifier = None

        path_audio: Audio | None = None
        if audio_classifier is not None and source_path is not None:
            try:
                path_audio = Audio.from_path(source_path)
            except Exception:
                logger.warning(
                    "Failed to load audio from path, audio classification will use clip fallback", exc_info=True
                )
                path_audio = None

        # -- Batched SceneVLM: collect all timestamps, extract frames once, run one forward pass --
        captions: list[str | None] = [None] * len(scenes)
        if scene_vlm is not None:
            with _record_stage(run_info, "scene_vlm"):
                try:
                    captions = self._run_scene_vlm_batched(
                        scene_vlm=scene_vlm,
                        source_path=source_path,
                        video=video,
                        metadata=metadata,
                        scenes=scenes,
                    )
                except Exception:
                    logger.warning("Batched SceneVLM failed, skipping visual understanding", exc_info=True)

        samples: list[SceneAnalysisSample] = []
        audio_ctx = _record_stage(run_info, "audio_classification") if audio_classifier is not None else nullcontext()
        with audio_ctx:
            for index, scene in enumerate(scenes):
                sample = SceneAnalysisSample(
                    scene_index=index,
                    start_second=float(scene.start),
                    end_second=float(scene.end),
                    start_frame=int(scene.start_frame),
                    end_frame=int(scene.end_frame),
                    caption=captions[index],
                )

                if audio_classifier is not None:
                    try:
                        scene_clip: Video | None = None
                        if path_audio is None:
                            try:
                                scene_clip = self._load_scene_video_clip(
                                    source_path=source_path,
                                    video=video,
                                    start_second=scene.start,
                                    end_second=scene.end,
                                )
                            except Exception:
                                scene_clip = None
                        sample.audio_classification = self._run_scene_audio_classification(
                            audio_classifier=audio_classifier,
                            path_audio=path_audio,
                            scene_clip=scene_clip,
                            scene_start=scene.start,
                            scene_end=scene.end,
                        )
                    except Exception:
                        logger.warning(
                            "AudioClassifier failed for scene %d (%.1f-%.1fs)",
                            index,
                            scene.start,
                            scene.end,
                            exc_info=True,
                        )

                samples.append(sample)

        return SceneAnalysisSection(samples=samples)

    def _run_scene_vlm_batched(
        self,
        *,
        scene_vlm: SceneVLM,
        source_path: Path | None,
        video: Video | None,
        metadata: VideoMetadata,
        scenes: list[SceneBoundary],
    ) -> list[str | None]:
        """Extract frames for all scenes in one ffmpeg call, then caption each group.

        Adjacent short scenes (< _SCENE_VLM_GROUP_THRESHOLD seconds) are merged
        into a single VLM call to reduce per-call overhead.
        """
        # Group adjacent short scenes to reduce VLM call count.
        # Each group is a list of scene indices that share one VLM call.
        groups: list[list[int]] = []
        current_group: list[int] = []
        current_group_duration = 0.0
        for i, scene in enumerate(scenes):
            dur = max(0.0, scene.end - scene.start)
            if current_group and current_group_duration + dur > _SCENE_VLM_GROUP_THRESHOLD:
                groups.append(current_group)
                current_group = [i]
                current_group_duration = dur
            else:
                current_group.append(i)
                current_group_duration += dur
        if current_group:
            groups.append(current_group)

        # Compute timestamps for each group (treating merged scenes as one span)
        group_timestamps: list[list[float]] = []
        all_timestamps: list[float] = []
        for group in groups:
            span_start = scenes[group[0]].start
            span_end = scenes[group[-1]].end
            duration = max(0.0, span_end - span_start)
            frame_count = min(
                _SCENE_VLM_MAX_FRAMES,
                max(1, math.ceil(_SCENE_VLM_FRAME_SCALE * math.log(duration / _SCENE_VLM_FRAME_BASE + 1))),
            )
            timestamps = self._sample_timestamps(start_second=span_start, end_second=span_end, frame_count=frame_count)
            group_timestamps.append(timestamps)
            all_timestamps.extend(timestamps)

        if not all_timestamps:
            return [None] * len(scenes)

        # Extract all frames in a single ffmpeg call
        if source_path is not None:
            all_frames_array = extract_frames_at_times(source_path, all_timestamps)
            all_frames: list[np.ndarray | Image.Image] = list(all_frames_array)
        else:
            current_video = _require_video(video)
            max_frame = max(len(current_video.frames) - 1, 0)
            indices = [max(0, min(max_frame, int(ts * metadata.fps))) for ts in all_timestamps]
            all_frames = [current_video.frames[idx] for idx in indices]

        # Caption each group and assign to all scenes in that group
        captions: list[str | None] = [None] * len(scenes)
        offset = 0
        for group, timestamps in zip(groups, group_timestamps):
            frame_count = len(timestamps)
            group_frames = all_frames[offset : offset + frame_count]
            offset += frame_count
            if not group_frames:
                continue
            caption: str | None = None
            try:
                caption = scene_vlm.analyze_scene(group_frames) or None
            except Exception:
                logger.warning(
                    "SceneVLM failed for scenes %d-%d (%.1f-%.1fs)",
                    group[0],
                    group[-1],
                    scenes[group[0]].start,
                    scenes[group[-1]].end,
                    exc_info=True,
                )
                caption = None
            for i in group:
                captions[i] = caption
        logger.info("SceneVLM: %d groups from %d scenes", len(groups), len(scenes))
        return captions

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

    @staticmethod
    def _sample_timestamps(*, start_second: float, end_second: float, frame_count: int) -> list[float]:
        duration = max(0.0, end_second - start_second)
        if duration <= 0.0:
            return []

        count = max(1, frame_count)
        if count == 1:
            return [start_second + (duration * 0.5)]

        # Center samples inside the interval so we avoid exact boundaries.
        step = duration / float(count + 1)
        timestamps = [start_second + (step * (idx + 1)) for idx in range(count)]
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


@contextmanager
def _record_stage(run_info: AnalysisRunInfo, stage: str) -> Iterator[None]:
    """Time a block, write the elapsed seconds into ``run_info``, and log it."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        run_info.stage_durations_seconds[stage] = elapsed
        logger.info("%s completed in %.2fs", stage, elapsed)


_T = TypeVar("_T")


def _run_with_stage(run_info: AnalysisRunInfo, stage: str, fn: Callable[..., _T], /, **kwargs: Any) -> _T:
    """Call ``fn(**kwargs)`` inside ``_record_stage``. Use with ``ThreadPoolExecutor.submit``."""
    with _record_stage(run_info, stage):
        return fn(**kwargs)


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
