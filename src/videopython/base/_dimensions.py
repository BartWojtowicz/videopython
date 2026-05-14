"""Pure helpers for video dimension math.

Centralises the libx264+yuv420p even-dimension constraint and the
two "round to even" calculations that previously lived (with subtly
different semantics) in ``base/video.py``, ``ai/transforms.py``, and
``editing/transforms.py``.
"""

from __future__ import annotations


def round_to_even(value: int | float) -> int:
    """Round a dimension to the nearest even integer (minimum 2).

    Use this when computing a target dimension from a ratio or scale
    factor and either direction (up or down) is acceptable.
    """
    return max(2, int(round(float(value) / 2.0) * 2))


def floor_to_even(value: int | float) -> int:
    """Round a dimension down to the next even integer (minimum 2).

    Use this when the result must not exceed the source region — e.g.
    cropping, where rounding up would read past the frame edge.
    """
    v = int(value)
    return max(2, v - (v % 2))


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
