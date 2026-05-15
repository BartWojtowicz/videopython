"""Tests for graceful font loading (videopython.base.fonts)."""

from importlib.resources import files

from PIL import ImageFont

from tests.test_config import TEST_FONT_PATH
from videopython.base.fonts import DEFAULT_FONT_FILENAME, load_font


def test_bundled_default_font_is_packaged():
    bundled = files("videopython.base.fonts").joinpath(DEFAULT_FONT_FILENAME)
    assert bundled.is_file()


def test_load_font_explicit_path():
    font = load_font(TEST_FONT_PATH, 32)
    assert isinstance(font, ImageFont.FreeTypeFont)


def test_load_font_none_uses_bundled_default():
    font = load_font(None, 32)
    # Bundled DejaVu Sans loads as a real TrueType font, not the PIL bitmap fallback.
    assert isinstance(font, ImageFont.FreeTypeFont)
    assert "DejaVu" in font.getname()[0]


def test_load_font_bad_path_falls_back_without_raising():
    font = load_font("/nonexistent/totally-not-a-font.ttf", 32)
    assert isinstance(font, ImageFont.FreeTypeFont)
    assert "DejaVu" in font.getname()[0]


def test_load_font_empty_string_falls_back():
    font = load_font("", 24)
    assert isinstance(font, ImageFont.FreeTypeFont)
