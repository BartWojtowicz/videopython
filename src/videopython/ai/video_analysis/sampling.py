"""Frame sampling profile and helpers for VideoAnalyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from PIL import Image

__all__ = [
    "DEFAULT_SAMPLING_PRESET",
    "FACE_TRACK_MAX_FRAMES_PER_SHOT",
    "FACE_TRACK_SAMPLE_PERIOD_SECONDS",
    "PHASH_DEDUP_DISTANCE",
    "SAMPLING_PRESETS",
    "SamplingPreset",
    "phash_dedup_frames",
    "sample_timestamps",
]

SamplingPreset = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class _SamplingProfile:
    """Per-scene frame budget knobs for SceneVLM sampling.

    Replaces the four ``_SCENE_VLM_*`` module constants. The log-curve
    formula stays:

        frames = clip(1, max_frames, ceil(scale * log(duration/base + 1)))

    so smaller budgets reach their cap on shorter scenes by tuning
    ``scale`` and ``base`` together.
    """

    frame_scale: float
    frame_base: float
    max_frames: int
    group_threshold_seconds: float


SAMPLING_PRESETS: dict[SamplingPreset, _SamplingProfile] = {
    "low": _SamplingProfile(frame_scale=2.0, frame_base=8.0, max_frames=8, group_threshold_seconds=20.0),
    "medium": _SamplingProfile(frame_scale=3.0, frame_base=5.0, max_frames=30, group_threshold_seconds=10.0),
    "high": _SamplingProfile(frame_scale=4.0, frame_base=3.0, max_frames=60, group_threshold_seconds=4.0),
}
DEFAULT_SAMPLING_PRESET: SamplingPreset = "medium"

# Hamming distance threshold for perceptual-hash dedup of sampled frames
# inside one VLM group. 4 is the conventional cutoff for imagehash; same
# pattern as the M1 voice-sample thresholds (constant, not user-facing).
PHASH_DEDUP_DISTANCE = 4

# Default per-shot face-tracking cadence. The analyzer samples one frame
# per ``FACE_TRACK_SAMPLE_PERIOD_SECONDS`` of scene duration -- enough
# to bind a track to a subject for downstream consumers (M6 lip-sync
# refines this with every-frame tracking). Module-level constant; same
# pattern as the phash distance.
FACE_TRACK_SAMPLE_PERIOD_SECONDS = 0.5
FACE_TRACK_MAX_FRAMES_PER_SHOT = 60


def sample_timestamps(*, start_second: float, end_second: float, frame_count: int) -> list[float]:
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


def phash_dedup_frames(
    frames: list[np.ndarray | Image.Image],
    *,
    max_distance: int,
) -> list[np.ndarray | Image.Image]:
    """Drop near-duplicate frames inside a single VLM group via perceptual hash.

    Static talking-head shots collapse to 1-2 frames; action shots keep
    their budget. The first frame is always retained; subsequent frames
    are kept only if their hash differs from every kept frame by more
    than ``max_distance`` Hamming bits. **At least one frame survives**
    even when every frame collides.
    """
    if len(frames) <= 1:
        return list(frames)

    from videopython.ai._optional import require

    imagehash = require("imagehash", "ai", feature="scene-frame dedup")

    kept: list[np.ndarray | Image.Image] = []
    kept_hashes: list[Any] = []
    for frame in frames:
        pil = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        h = imagehash.phash(pil)
        if any((h - prev) <= max_distance for prev in kept_hashes):
            continue
        kept.append(frame)
        kept_hashes.append(h)

    # Hard floor: never let dedup empty a group; it would silently lose
    # the scene's caption budget.
    if not kept:
        return [frames[0]]
    return kept
