"""Pure, AI-free renderer for object-detection overlays.

Draws labelled bounding boxes onto a frame from a list of
:class:`~videopython.base.description.DetectedObject`. This module has **no AI
dependencies** -- it is the single source of truth for how detections look, so
it can be unit-tested with synthetic detections and reused by any detector. The
AI side (``videopython.ai``) only produces the ``DetectedObject`` list and calls
:func:`draw_detections`.

Visuals: a resolution-scaled box stroke plus a label chip filled in the box's
own colour (so chip and box read as one unit) with anti-aliased text. Colours
are deterministic per class via :func:`class_color`, so the same class is the
same colour in every frame and across runs.
"""

from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from videopython.base.description import DetectedObject
from videopython.base.fonts import load_font

__all__ = ["DetectionStyle", "class_color", "draw_detections"]

# Hand-picked Material-palette hues for common COCO classes so busy scenes read
# clearly. Any class not listed gets a deterministic colour from ``class_color``.
_RESERVED_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (76, 175, 80),  # green
    "bicycle": (0, 188, 212),  # cyan
    "car": (33, 150, 243),  # blue
    "motorcycle": (156, 39, 176),  # purple
    "bus": (255, 193, 7),  # amber
    "truck": (255, 87, 34),  # deep orange
    "cat": (233, 30, 99),  # pink
    "dog": (255, 152, 0),  # orange
}


def class_color(label: str) -> tuple[int, int, int]:
    """Deterministic RGB colour for a class label.

    Common COCO classes get a reserved Material hue; everything else maps
    ``md5(label) -> HSV hue`` at fixed saturation/value. ``md5`` (not the
    salted built-in ``hash``) is used so colours are stable across processes
    and test runs.
    """
    reserved = _RESERVED_COLORS.get(label)
    if reserved is not None:
        return reserved
    digest = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16)
    hue = (digest % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


@dataclass(frozen=True)
class DetectionStyle:
    """Styling for :func:`draw_detections`.

    Lengths expressed as a fraction of the frame's longer side are
    resolution-independent: the same style reads consistently at 1080p and 4k.
    """

    box_color: tuple[int, int, int] | None = None
    """Fixed ``(R, G, B)`` for every box, or ``None`` for per-class colours."""
    line_thickness: float = 0.003
    """Box stroke width as a fraction of ``max(height, width)`` (~3px at 1080p)."""
    show_confidence: bool = True
    """Append the confidence as a whole-number percent to each label."""
    label_font_size: float = 0.022
    """Label text height as a fraction of ``max(height, width)`` (~24px at 1080p)."""
    label_text_color: tuple[int, int, int] = (255, 255, 255)
    """Colour of the label text drawn on the chip."""
    label_bg_alpha: int = 200
    """Opacity (0-255) of the label chip background."""
    min_confidence: float = 0.0
    """Detections below this confidence are skipped."""
    font: str | None = None
    """Bundled font name or path; ``None`` uses the default font."""


def draw_detections(
    frame: np.ndarray,
    detections: list[DetectedObject],
    style: DetectionStyle = DetectionStyle(),
) -> np.ndarray:
    """Return a copy of ``frame`` with ``detections`` drawn as labelled boxes.

    Shape-preserving: the result is the same ``(H, W, 3)`` ``uint8`` array. An
    empty ``detections`` list (or one filtered out by ``min_confidence``) is a
    no-op that returns ``frame`` unchanged. Boxes are clamped to the frame, so
    off-frame coordinates clip cleanly instead of raising. Label chips flip
    inside the box when they would overflow the top edge and clamp horizontally
    so they never leave the frame.

    Args:
        frame: Source frame as ``(H, W, 3)`` ``uint8`` (RGB).
        detections: Objects to draw; each uses its normalized ``bounding_box``.
        style: Visual styling (colours, stroke width, label options).

    Returns:
        A new ``(H, W, 3)`` ``uint8`` frame with the overlays composited on.
    """
    if not detections:
        return frame

    h, w = frame.shape[:2]
    scale = max(h, w)
    thickness = max(1, round(style.line_thickness * scale))
    font_px = max(8, round(style.label_font_size * scale))
    font = load_font(style.font, font_px)

    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    drew_any = False
    for det in detections:
        box = det.bounding_box
        if box is None or det.confidence < style.min_confidence:
            continue
        drew_any = True
        color = style.box_color or class_color(det.label)

        x0 = max(0, min(w - 1, int(box.x * w)))
        y0 = max(0, min(h - 1, int(box.y * h)))
        x1 = max(0, min(w - 1, int((box.x + box.width) * w)))
        y1 = max(0, min(h - 1, int((box.y + box.height) * h)))
        draw.rectangle((x0, y0, x1, y1), outline=(*color, 255), width=thickness)

        text = det.label.title()
        if style.show_confidence:
            text = f"{text} {det.confidence * 100:.0f}%"

        tb = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(2, thickness)
        chip_w, chip_h = text_w + 2 * pad, text_h + 2 * pad

        # Flip the chip inside the box when it would overflow the top edge,
        # and clamp horizontally so it never leaves the frame.
        chip_y = y0 - chip_h if y0 - chip_h >= 0 else y0
        chip_x = max(0, min(x0, w - chip_w))
        draw.rectangle(
            (chip_x, chip_y, chip_x + chip_w, chip_y + chip_h),
            fill=(*color, style.label_bg_alpha),
        )
        draw.text(
            (chip_x + pad - tb[0], chip_y + pad - tb[1]),
            text,
            font=font,
            fill=(*style.label_text_color, 255),
        )

    if not drew_any:
        return frame

    out = Image.fromarray(frame).convert("RGBA")
    out.alpha_composite(canvas)
    return np.array(out.convert("RGB"), dtype=np.uint8)
