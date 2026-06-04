"""Tests for graceful font loading (videopython.base.fonts)."""

from importlib.resources import files

from PIL import ImageFont

from tests.test_config import TEST_FONT_PATH
from videopython.base.fonts import BUNDLED_FONTS, DEFAULT_FONT_FILENAME, FONT_NAMES, load_font
from videopython.editing.effects import TextOverlay
from videopython.editing.transcription_overlay import TranscriptionOverlay


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


def test_font_names_are_the_four_bundled_names():
    assert FONT_NAMES == ["anton", "bebas-neue", "lato-bold", "poppins-bold"]
    assert FONT_NAMES == sorted(BUNDLED_FONTS)


def test_load_font_registered_name_resolves_to_bundled_font():
    font = load_font("poppins-bold", 32)
    # A registered NAME resolves to its bundled TrueType file, not DejaVu.
    assert isinstance(font, ImageFont.FreeTypeFont)
    assert "DejaVu" not in font.getname()[0]


def test_load_font_unknown_name_falls_back_without_raising():
    font = load_font("not-a-registered-font", 32)
    assert isinstance(font, ImageFont.FreeTypeFont)
    assert "DejaVu" in font.getname()[0]


def test_text_overlay_font_name_round_trips_as_name():
    overlay = TextOverlay(text="hi", font="anton")
    data = overlay.model_dump(mode="json")
    assert data["font"] == "anton"
    assert TextOverlay.model_validate(data).font == "anton"


def test_add_subtitles_font_name_round_trips_as_name():
    op = TranscriptionOverlay(font="anton")
    data = op.model_dump(mode="json")
    assert data["font"] == "anton"
    assert TranscriptionOverlay.model_validate(data).font == "anton"
