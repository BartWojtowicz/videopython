"""Tests for SceneVLM plain-text caption output."""

from __future__ import annotations

import videopython.ai.understanding.image as image_mod


def test_analyze_scene_returns_str() -> None:
    vlm = image_mod.SceneVLM()
    assert vlm.analyze_scene.__annotations__.get("return") == "str"


def test_analyze_frame_returns_str() -> None:
    vlm = image_mod.SceneVLM()
    assert vlm.analyze_frame.__annotations__.get("return") == "str"
