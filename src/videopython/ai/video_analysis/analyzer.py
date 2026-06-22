"""VideoAnalyzer: orchestrates scene-first analyzers and builds VideoAnalysis output."""

from __future__ import annotations

import gc
import logging
import time
from contextlib import nullcontext
from pathlib import Path

from videopython.ai.understanding import AudioClassifier, SceneVLM
from videopython.ai.understanding.faces import FaceShotTracker
from videopython.audio import Audio
from videopython.base.description import SceneBoundary, SceneDescription
from videopython.base.video import Video, VideoMetadata

from . import detectors, source_metadata
from .models import (
    AUDIO_CLASSIFIER,
    AUDIO_TO_TEXT,
    FACE_TRACKER,
    SCENE_VLM,
    SEMANTIC_SCENE_DETECTOR,
    AnalysisRunInfo,
    AudioAnalysisSection,
    SceneAnalysisSample,
    SceneAnalysisSection,
    VideoAnalysis,
    VideoAnalysisConfig,
    VideoAnalysisSource,
)
from .sampling import DEFAULT_SAMPLING_PRESET, SAMPLING_PRESETS, SamplingPreset

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Orchestrates scene-first analyzers and builds `VideoAnalysis` output.

    ``sampling`` controls how aggressively the SceneVLM samples frames per
    scene. ``low`` is a fast preview pass for long videos, ``high`` keeps
    talking-head depth, ``medium`` is the previous default. The preset
    tunes the per-scene frame cap, the log-curve scale/base used to size
    short scenes, and the threshold below which adjacent scenes get
    merged into one VLM call.

    ``sampling`` and the SceneVLM ``tier`` are orthogonal: small models
    can't make use of dense sampling, but the user owns that tradeoff.
    """

    def __init__(
        self,
        config: VideoAnalysisConfig | None = None,
        *,
        sampling: SamplingPreset = DEFAULT_SAMPLING_PRESET,
    ):
        if sampling not in SAMPLING_PRESETS:
            supported = ", ".join(SAMPLING_PRESETS)
            raise ValueError(f"sampling must be one of: {supported}")
        self.config = config or VideoAnalysisConfig()
        self.sampling: SamplingPreset = sampling
        self._sampling_profile = SAMPLING_PRESETS[sampling]

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
            created_at=detectors.utc_now_iso(),
            mode=mode,
            library_version=detectors.library_version(),
        )

        t_analysis_start = time.perf_counter()

        run_whisper = AUDIO_TO_TEXT in enabled
        run_scene_det = SEMANTIC_SCENE_DETECTOR in enabled

        transcription = None
        detected: list[SceneBoundary] | None = None

        # SceneVLM is loaded *after* Whisper/TransNetV2 finish (not concurrently)
        # because transformers' from_pretrained(torch_dtype="auto") mutates the
        # process-global torch.get_default_dtype() during model construction,
        # which corrupts Whisper's model weights if they're initialized at the
        # same time.
        if run_whisper and run_scene_det:
            transcription, detected = detectors.run_whisper_and_scene_detection(
                config=self.config, source_path=source_path, video=video, run_info=run_info
            )
        else:
            if run_whisper:
                with detectors.record_stage(run_info, "whisper"):
                    transcription = detectors.run_whisper(config=self.config, source_path=source_path, video=video)

            if run_scene_det:
                with detectors.record_stage(run_info, "scene_detection"):
                    detected = detectors.run_scene_detection(config=self.config, source_path=source_path, video=video)

        if run_scene_det:
            detectors.reset_transnetv2_torch_state()

        # Whisper and TransNetV2 are done -- free their GPU memory before
        # loading SceneVLM (~9GB). Python GC doesn't guarantee immediate
        # cleanup, so force it and release the CUDA cache.
        if run_whisper or run_scene_det:
            gc.collect()
            detectors.release_gpu_cache()

        scenes = self._default_scene_boundaries(metadata)
        if detected is not None:
            scenes = self._normalize_scene_boundaries(detected, metadata)

        if not scenes:
            scenes = self._default_scene_boundaries(metadata)

        with detectors.record_stage(run_info, "scene_analysis"):
            scene_section = self._analyze_scenes(
                source_path=source_path,
                video=video,
                metadata=metadata,
                scenes=scenes,
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

    def _analyze_scenes(
        self,
        *,
        source_path: Path | None,
        video: Video | None,
        metadata: VideoMetadata,
        scenes: list[SceneBoundary],
        run_info: AnalysisRunInfo,
    ) -> SceneAnalysisSection:
        enabled = self.config.enabled_analyzers

        # Best-effort init: a missing extra / model-load failure degrades that
        # analyzer to "skipped" rather than aborting the whole run.
        scene_vlm = (
            source_metadata.try_init(lambda: SceneVLM(**self.config.get_params(SCENE_VLM)), "SceneVLM")
            if SCENE_VLM in enabled
            else None
        )
        audio_classifier = (
            source_metadata.try_init(
                lambda: AudioClassifier(**self.config.get_params(AUDIO_CLASSIFIER)), "AudioClassifier"
            )
            if AUDIO_CLASSIFIER in enabled
            else None
        )
        face_tracker = (
            source_metadata.try_init(lambda: FaceShotTracker(**self.config.get_params(FACE_TRACKER)), "FaceShotTracker")
            if FACE_TRACKER in enabled
            else None
        )

        path_audio: Audio | None = None
        if audio_classifier is not None and source_path is not None:
            try:
                path_audio = Audio.from_path(source_path)
            except (OSError, RuntimeError, ValueError):
                logger.warning(
                    "Failed to load audio from path, audio classification will use clip fallback",
                    exc_info=True,
                )
                path_audio = None

        descriptions: list[SceneDescription | None] = [None] * len(scenes)
        if scene_vlm is not None:
            with detectors.record_stage(run_info, "scene_vlm"):
                try:
                    descriptions = detectors.run_scene_vlm_batched(
                        scene_vlm=scene_vlm,
                        profile=self._sampling_profile,
                        sampling=self.sampling,
                        source_path=source_path,
                        video=video,
                        metadata=metadata,
                        scenes=scenes,
                    )
                except (IndexError, OSError, RuntimeError, ValueError):
                    logger.warning("Batched SceneVLM failed, skipping visual understanding", exc_info=True)

        samples: list[SceneAnalysisSample] = []
        audio_ctx = (
            detectors.record_stage(run_info, "audio_classification") if audio_classifier is not None else nullcontext()
        )
        face_ctx = detectors.record_stage(run_info, "face_tracker") if face_tracker is not None else nullcontext()
        with audio_ctx, face_ctx:
            for index, scene in enumerate(scenes):
                sample = SceneAnalysisSample(
                    scene_index=index,
                    start_second=float(scene.start),
                    end_second=float(scene.end),
                    start_frame=int(scene.start_frame),
                    end_frame=int(scene.end_frame),
                    scene_description=descriptions[index],
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
                            except (OSError, RuntimeError, ValueError):
                                scene_clip = None
                        sample.audio_classification = detectors.run_scene_audio_classification(
                            audio_classifier=audio_classifier,
                            path_audio=path_audio,
                            scene_clip=scene_clip,
                            scene_start=scene.start,
                            scene_end=scene.end,
                        )
                    except (OSError, RuntimeError, ValueError):
                        logger.warning(
                            "AudioClassifier failed for scene %d (%.1f-%.1fs)",
                            index,
                            scene.start,
                            scene.end,
                            exc_info=True,
                        )

                if face_tracker is not None:
                    try:
                        sample.faces = detectors.run_scene_face_tracker(
                            face_tracker=face_tracker,
                            source_path=source_path,
                            video=video,
                            metadata=metadata,
                            scene=scene,
                        )
                    except (IndexError, OSError, RuntimeError, ValueError):
                        logger.warning(
                            "FaceShotTracker failed for scene %d (%.1f-%.1fs)",
                            index,
                            scene.start,
                            scene.end,
                            exc_info=True,
                        )

                samples.append(sample)

        return SceneAnalysisSection(samples=samples)

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
        v = detectors.require_video(video)
        return v[round(start_second * v.fps) : round(end_second * v.fps)]

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
        tags = source_metadata.extract_source_tags(path_obj) if path_obj else {}
        creation_time = source_metadata.creation_time_from_tags(tags)
        geo = source_metadata.parse_geo_metadata(tags)
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
