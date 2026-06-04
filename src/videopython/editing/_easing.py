"""Shared easing curves for time-based effects.

Each function maps normalized progress ``t`` in ``[0, 1]`` to an eased value in
``[0, 1]``. Functions are vectorized -- pass a numpy array (e.g. an envelope of
per-frame progress) and get an array back. Used by effects that animate a
parameter over their window (``KenBurns``, ``PunchIn``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

EasingMode = Literal["linear", "ease_in", "ease_out", "ease_in_out"]


def ease_in(t: np.ndarray) -> np.ndarray:
    """Quadratic ease-in: starts slow, accelerates."""
    return t * t


def ease_out(t: np.ndarray) -> np.ndarray:
    """Quadratic ease-out: starts fast, decelerates."""
    return 1.0 - (1.0 - t) * (1.0 - t)


def ease_in_out(t: np.ndarray) -> np.ndarray:
    """Quadratic ease-in-out: slow at both ends, fastest in the middle."""
    return np.where(t < 0.5, 2.0 * t * t, 1.0 - 2.0 * (1.0 - t) * (1.0 - t))


def ease(t: np.ndarray, mode: EasingMode) -> np.ndarray:
    """Apply the easing curve named by ``mode`` to ``t``."""
    if mode == "linear":
        return t
    if mode == "ease_in":
        return ease_in(t)
    if mode == "ease_out":
        return ease_out(t)
    if mode == "ease_in_out":
        return ease_in_out(t)
    raise ValueError(f"Unknown easing mode: {mode!r}")
