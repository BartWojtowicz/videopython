"""Pure helpers for video dimension math.

Centralises the libx264+yuv420p even-dimension constraint and the
"round to even" calculations that previously lived (with subtly
different semantics) in ``base/video.py``, ``ai/transforms.py``, and
``base/transforms.py``.
"""

from __future__ import annotations

from typing import Literal

RoundMode = Literal["nearest", "floor"]


def round_to_even(value: int | float, *, mode: RoundMode = "nearest", min_value: int = 2) -> int:
    """Round a dimension to an even integer.

    Args:
        value: Source dimension (pixels). Floats are accepted for scale math.
        mode: ``"nearest"`` rounds to the closest even integer; ``"floor"``
            rounds down to the next even integer (used by H.264-aware
            crop math that must never enlarge the source region).
        min_value: Lower bound on the result. ``2`` matches the historical
            ``_round_dimension_to_even`` behaviour; pass ``0`` to mirror
            the legacy ``_make_even`` (which never clamped).
    """
    v = float(value)
    if mode == "nearest":
        rounded = int(round(v / 2.0) * 2)
    else:
        rounded = int(v) - (int(v) % 2)
    return max(min_value, rounded)


def require_even(width: int, height: int) -> None:
    """Guard for libx264+yuv420p output, which rejects odd dimensions.

    Raises:
        ValueError: If either dimension is odd.
    """
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError(
            "libx264 with yuv420p requires even frame dimensions. "
            f"Got {width}x{height}. Resize, crop, or pad to even width and height before saving."
        )
