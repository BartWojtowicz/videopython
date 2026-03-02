"""Parsing/validation tests for SceneVLM structured outputs."""

from __future__ import annotations

import json

import videopython.ai.understanding.image as image_mod


def test_scene_vlm_parse_response_accepts_valid_schema() -> None:
    raw = json.dumps(
        {
            "caption": "  A fight scene  ",
            "primary_action": "  punching  ",
            "confidence": 1.4,
            "objects": [
                {"label": " fighter ", "confidence": 0.9},
                {"label": " ", "confidence": 0.8},
            ],
            "text": [" SCORE ", "score", ""],
            "extra": "ignored",
        }
    )

    result = image_mod.SceneVLM()._parse_response(raw)

    assert result.caption == "A fight scene"
    assert result.primary_action == "punching"
    assert result.confidence == 1.0
    assert len(result.objects) == 1
    assert result.objects[0].label == "fighter"
    assert result.text == ["SCORE"]


def test_scene_vlm_parse_response_rejects_truncated_json() -> None:
    raw = '{ "caption": "A fight scene", "objects": [ { "label": "fighter", "confidence"'

    result = image_mod.SceneVLM()._parse_response(raw)

    assert result.caption == "No scene description"
    assert result.objects == []
    assert result.text == []


def test_scene_vlm_parse_response_rejects_wrong_schema() -> None:
    raw = json.dumps(
        {
            "caption": "A fight scene",
            "objects": "not-a-list",
            "text": [],
        }
    )

    result = image_mod.SceneVLM()._parse_response(raw)

    assert result.caption == "No scene description"
    assert result.objects == []
    assert result.text == []
