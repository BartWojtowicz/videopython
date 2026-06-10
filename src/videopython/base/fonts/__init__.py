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
from pathlib import Path

from PIL import ImageFont

__all__ = [
    "BUNDLED_FONTS",
    "BUNDLED_FONT_FAMILIES",
    "DEFAULT_FONT_FILENAME",
    "FONT_NAMES",
    "bundled_fonts_dir",
    "load_font",
]

DEFAULT_FONT_FILENAME = "DejaVuSans.ttf"

# Registered short NAME -> bundled filename. The names are stable identifiers an
# LLM can emit (saved plans round-trip on these); ``load_font`` resolves a name
# to its bundled file before the path/DejaVu/PIL fallback chain.
BUNDLED_FONTS: dict[str, str] = {
    "poppins-bold": "Poppins-Bold.ttf",
    "lato-bold": "Lato-Bold.ttf",
    "anton": "Anton-Regular.ttf",
    "bebas-neue": "BebasNeue-Regular.ttf",
}

FONT_NAMES: list[str] = sorted(BUNDLED_FONTS)

# Registered NAME (or None for the default) -> (family name, is-bold-face),
# exactly as declared in each file's name table. Renderers that match fonts by
# family name -- libass via the ffmpeg ``subtitles=`` filter's ``fontsdir`` --
# need these, not the filenames. Verified against the bundled .ttf files
# (PIL ``ImageFont.getname()``); a test pins them.
BUNDLED_FONT_FAMILIES: dict[str | None, tuple[str, bool]] = {
    None: ("DejaVu Sans", False),
    "poppins-bold": ("Poppins", True),
    "lato-bold": ("Lato", True),
    "anton": ("Anton", False),
    "bebas-neue": ("Bebas Neue", False),
}


def bundled_fonts_dir() -> Path:
    """Filesystem directory holding the bundled .ttf files.

    For ffmpeg/libass ``fontsdir`` wiring, which needs a real directory path.
    Standard (non-zip) installs ship the package unpacked, so the resources
    root is a plain directory.
    """
    return Path(str(files(__package__)))


def _try_truetype(path: str, font_size: int) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(path, font_size)
    except (OSError, ValueError):
        return None


def load_font(font_filename: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font, falling back gracefully when one is unavailable.

    Resolution order: a registered bundled NAME (``BUNDLED_FONTS``) -> the
    given ``font_filename`` -> the bundled DejaVu Sans -> PIL's built-in
    bitmap font. Never raises for a missing, unknown, or unreadable font, so
    callers may pass ``None`` (or an unrecognized name) to mean "use the
    default".

    Args:
        font_filename: A registered bundled font name, a path to a
            ``.ttf``/``.otf`` file, or ``None``.
        font_size: Font size in points.

    Returns:
        A loaded PIL font object.
    """
    if font_filename:
        bundled_filename = BUNDLED_FONTS.get(font_filename)
        if bundled_filename is not None:
            try:
                with as_file(files(__package__).joinpath(bundled_filename)) as bundled:
                    font = _try_truetype(str(bundled), font_size)
                    if font is not None:
                        return font
            except (FileNotFoundError, ModuleNotFoundError):
                pass
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
