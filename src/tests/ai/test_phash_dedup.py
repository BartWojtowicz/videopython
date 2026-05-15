"""Unit tests for the SceneVLM frame-dedup helper."""

from __future__ import annotations

import numpy as np
from PIL import Image

from videopython.ai.video_analysis.sampling import phash_dedup_frames


def _solid(value: int) -> np.ndarray | Image.Image:
    return np.full((16, 16, 3), value, dtype=np.uint8)


def test_dedup_collapses_identical_frames_to_one() -> None:
    frames: list[np.ndarray | Image.Image] = [_solid(0) for _ in range(5)]
    kept = phash_dedup_frames(frames, max_distance=4)
    assert len(kept) == 1


def test_dedup_keeps_visually_different_frames() -> None:
    rng = np.random.default_rng(seed=42)
    frames: list[np.ndarray | Image.Image] = [
        rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8),
        rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8),
    ]
    kept = phash_dedup_frames(frames, max_distance=4)
    assert len(kept) == 2


def test_dedup_floor_keeps_at_least_one_frame() -> None:
    frames: list[np.ndarray | Image.Image] = [_solid(7) for _ in range(3)]
    kept = phash_dedup_frames(frames, max_distance=64)
    assert len(kept) >= 1


def test_dedup_passthrough_for_single_frame_input() -> None:
    frames: list[np.ndarray | Image.Image] = [_solid(0)]
    assert phash_dedup_frames(frames, max_distance=4) is not frames  # returns a fresh list
    assert len(phash_dedup_frames(frames, max_distance=4)) == 1
