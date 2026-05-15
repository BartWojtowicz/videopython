"""Bundled default font and graceful font loading.

Text operations let callers omit a font path. This module provides a
reliable resolution chain so rendering never hard-fails on a missing or
unreadable font:

1. The explicit ``font_filename`` if given and loadable.
2. The bundled DejaVu Sans (broad Unicode coverage).
3. PIL's built-in font (always available).
"""

from __future__ import annotations

from importlib.resources import as_file, files

from PIL import ImageFont

__all__ = ["DEFAULT_FONT_FILENAME", "load_font"]

DEFAULT_FONT_FILENAME = "DejaVuSans.ttf"


def _try_truetype(path: str, font_size: int) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(path, font_size)
    except (OSError, ValueError):
        return None


def load_font(font_filename: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font, falling back gracefully when one is unavailable.

    Resolution order: the given ``font_filename`` -> the bundled DejaVu
    Sans -> PIL's built-in bitmap font. Never raises for a missing or
    unreadable font, so callers may pass ``None`` to mean "use the
    default".

    Args:
        font_filename: Path to a ``.ttf``/``.otf`` file, or ``None``.
        font_size: Font size in points.

    Returns:
        A loaded PIL font object.
    """
    if font_filename:
        font = _try_truetype(font_filename, font_size)
        if font is not None:
            return font

    try:
        with as_file(files(__package__).joinpath(DEFAULT_FONT_FILENAME)) as bundled:
            font = _try_truetype(str(bundled), font_size)
            if font is not None:
                return font
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    return ImageFont.load_default(font_size)
