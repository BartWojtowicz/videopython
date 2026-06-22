"""Stage runners and stage-recording utilities for VideoAnalyzer."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from PIL import Image

from videopython.ai.understanding import AudioToText, SemanticSceneDetector
from videopython.audio import Audio
from videopython.base.description import (
    AudioClassification,
    AudioEvent,
    FaceTrack,
    SceneBoundary,
    SceneDescription,
)
from videopython.base.transcription import Transcription
from videopython.base.video import Video, VideoMetadata, extract_frames_at_times

from .models import (
    AUDIO_TO_TEXT,
    SEMANTIC_SCENE_DETECTOR,
    AnalysisRunInfo,
    VideoAnalysisConfig,
)
from .sampling import (
    FACE_TRACK_MAX_FRAMES_PER_SHOT,
    FACE_TRACK_SAMPLE_PERIOD_SECONDS,
    PHASH_DEDUP_DISTANCE,
    SamplingPreset,
    _SamplingProfile,
    phash_dedup_frames,
    sample_timestamps,
)

if TYPE_CHECKING:
    from videopython.ai.understanding import AudioClassifier, SceneVLM
    from videopython.ai.understanding.faces import FaceShotTracker

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def library_version() -> str | None:
    try:
        return importlib_metadata.version("videopython")
    except importlib_metadata.PackageNotFoundError:
        return None


def require_video(video: Video | None) -> Video:
    if video is None:
        raise ValueError("Video input is required for in-memory analysis")
    return video


@contextmanager
def record_stage(run_info: AnalysisRunInfo, stage: str) -> Iterator[None]:
    """Time a block, write the elapsed seconds into ``run_info``, and log it."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        run_info.stage_durations_seconds[stage] = elapsed
        logger.info("%s completed in %.2fs", stage, elapsed)


def run_with_stage(run_info: AnalysisRunInfo, stage: str, fn: Callable[..., _T], /, **kwargs: Any) -> _T:
    """Call ``fn(**kwargs)`` inside ``record_stage``. Use with ``ThreadPoolExecutor.submit``."""
    with record_stage(run_info, stage):
        return fn(**kwargs)


def reset_transnetv2_torch_state() -> None:
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


def release_gpu_cache() -> None:
    """Force-release unused GPU memory back to the CUDA allocator."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_whisper(
    *,
    config: VideoAnalysisConfig,
    source_path: Path | None,
    video: Video | None,
) -> Transcription | None:
    try:
        return AudioToText(**config.get_params(AUDIO_TO_TEXT)).transcribe(
            Audio.from_path(source_path) if source_path is not None else require_video(video)
        )
    except (ImportError, OSError, RuntimeError, ValueError):
        logger.warning("AudioToText failed, skipping transcription", exc_info=True)
        return None


def run_scene_detection(
    *,
    config: VideoAnalysisConfig,
    source_path: Path | None,
    video: Video | None,
) -> list[SceneBoundary] | None:
    try:
        scene_detector = SemanticSceneDetector(**config.get_params(SEMANTIC_SCENE_DETECTOR))
        return (
            scene_detector.detect_streaming(source_path)
            if source_path is not None
            else scene_detector.detect(require_video(video))
        )
    except (ImportError, OSError, RuntimeError, ValueError):
        logger.warning("SemanticSceneDetector failed, using default scene boundaries", exc_info=True)
        return None


def run_whisper_and_scene_detection(
    *,
    config: VideoAnalysisConfig,
    source_path: Path | None,
    video: Video | None,
    run_info: AnalysisRunInfo,
) -> tuple[Transcription | None, list[SceneBoundary] | None]:
    # Whisper and TransNetV2 operate on independent data (audio vs video
    # frames) and both fit comfortably in GPU memory together. Run them
    # concurrently via threads -- the GIL is released during GPU compute
    # and ffmpeg I/O so real parallelism is achieved.
    with record_stage(run_info, "whisper_and_scene_detection_parallel"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            whisper_future = pool.submit(
                run_with_stage,
                run_info,
                "whisper",
                run_whisper,
                config=config,
                source_path=source_path,
                video=video,
            )
            scene_future = pool.submit(
                run_with_stage,
                run_info,
                "scene_detection",
                run_scene_detection,
                config=config,
                source_path=source_path,
                video=video,
            )
            transcription = whisper_future.result()
            detected = scene_future.result()

    return transcription, detected


def run_scene_vlm_batched(
    *,
    scene_vlm: SceneVLM,
    profile: _SamplingProfile,
    sampling: SamplingPreset,
    source_path: Path | None,
    video: Video | None,
    metadata: VideoMetadata,
    scenes: list[SceneBoundary],
) -> list[SceneDescription | None]:
    """Extract frames for all scenes in one ffmpeg call, then describe each group.

    Adjacent short scenes (whose total duration would stay under the
    sampling preset's ``group_threshold_seconds``) are merged into a
    single VLM call to reduce per-call overhead. Within each group,
    sampled frames are deduped by perceptual hash so static
    talking-head shots collapse to 1-2 frames and free budget elsewhere.
    """
    # Group adjacent short scenes to reduce VLM call count.
    # Each group is a list of scene indices that share one VLM call.
    groups: list[list[int]] = []
    current_group: list[int] = []
    current_group_duration = 0.0
    for i, scene in enumerate(scenes):
        dur = max(0.0, scene.end - scene.start)
        if current_group and current_group_duration + dur > profile.group_threshold_seconds:
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
            profile.max_frames,
            max(1, math.ceil(profile.frame_scale * math.log(duration / profile.frame_base + 1))),
        )
        timestamps = sample_timestamps(start_second=span_start, end_second=span_end, frame_count=frame_count)
        group_timestamps.append(timestamps)
        all_timestamps.extend(timestamps)

    if not all_timestamps:
        return [None] * len(scenes)

    # Extract all frames in a single ffmpeg call
    if source_path is not None:
        all_frames_array = extract_frames_at_times(source_path, all_timestamps)
        all_frames: list[np.ndarray | Image.Image] = list(all_frames_array)
    else:
        current_video = require_video(video)
        max_frame = max(len(current_video.frames) - 1, 0)
        indices = [max(0, min(max_frame, int(ts * metadata.fps))) for ts in all_timestamps]
        all_frames = [current_video.frames[idx] for idx in indices]

    # Caption each group and assign to all scenes in that group
    descriptions: list[SceneDescription | None] = [None] * len(scenes)
    offset = 0
    total_in = 0
    total_out = 0
    for group, timestamps in zip(groups, group_timestamps):
        frame_count = len(timestamps)
        group_frames = all_frames[offset : offset + frame_count]
        offset += frame_count
        if not group_frames:
            continue
        deduped = phash_dedup_frames(group_frames, max_distance=PHASH_DEDUP_DISTANCE)
        total_in += len(group_frames)
        total_out += len(deduped)
        description: SceneDescription | None = None
        try:
            description = scene_vlm.analyze_scene(deduped)
        except (IndexError, OSError, RuntimeError, ValueError):
            logger.warning(
                "SceneVLM failed for scenes %d-%d (%.1f-%.1fs)",
                group[0],
                group[-1],
                scenes[group[0]].start,
                scenes[group[-1]].end,
                exc_info=True,
            )
            description = None
        for i in group:
            descriptions[i] = description
    logger.info(
        "SceneVLM: %d groups from %d scenes (sampling=%s, frames %d -> %d after phash dedup)",
        len(groups),
        len(scenes),
        sampling,
        total_in,
        total_out,
    )
    return descriptions


def run_scene_audio_classification(
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


def run_scene_face_tracker(
    *,
    face_tracker: FaceShotTracker,
    source_path: Path | None,
    video: Video | None,
    metadata: VideoMetadata,
    scene: SceneBoundary,
) -> list[FaceTrack] | None:
    """Sample frames inside a scene and run per-shot IoU face tracking."""
    duration = max(0.0, scene.end - scene.start)
    if duration <= 0.0:
        return None

    sample_count = max(1, min(FACE_TRACK_MAX_FRAMES_PER_SHOT, int(duration / FACE_TRACK_SAMPLE_PERIOD_SECONDS)))
    timestamps = sample_timestamps(start_second=scene.start, end_second=scene.end, frame_count=sample_count)
    if not timestamps:
        return None

    frame_indices = [int(round(ts * metadata.fps)) for ts in timestamps]

    if source_path is not None:
        frames_array = extract_frames_at_times(source_path, timestamps)
        frames: list[np.ndarray] = list(frames_array)
    else:
        current_video = require_video(video)
        max_frame = max(len(current_video.frames) - 1, 0)
        frames = [current_video.frames[max(0, min(max_frame, idx))] for idx in frame_indices]

    tracks = face_tracker.track_shot(frames, frame_indices=frame_indices)
    return tracks if tracks else None
