"""Tests for SceneVLM structured output and JSON parse fallback."""

from __future__ import annotations

import videopython.ai.understanding.image as image_mod
from videopython.base.description import SceneDescription


def test_analyze_scene_returns_scene_description() -> None:
    vlm = image_mod.SceneVLM()
    assert vlm.analyze_scene.__annotations__.get("return") == "SceneDescription"


def test_analyze_frame_returns_scene_description() -> None:
    vlm = image_mod.SceneVLM()
    assert vlm.analyze_frame.__annotations__.get("return") == "SceneDescription"


def test_try_parse_scene_json_clean_object() -> None:
    raw = '{"caption": "Two people talk.", "subjects": ["a", "b"], "shot_type": "medium"}'
    parsed = image_mod._try_parse_scene_json(raw)
    assert parsed == SceneDescription(caption="Two people talk.", subjects=["a", "b"], shot_type="medium")


def test_try_parse_scene_json_extracts_block_from_prose() -> None:
    raw = 'Sure, here is the JSON:\n{"caption": "A cat.", "subjects": ["cat"], "shot_type": "close-up"}\nDone.'
    parsed = image_mod._try_parse_scene_json(raw)
    assert parsed is not None
    assert parsed.caption == "A cat."
    assert parsed.subjects == ["cat"]
    assert parsed.shot_type == "close-up"


def test_try_parse_scene_json_drops_invalid_shot_type() -> None:
    raw = '{"caption": "x", "subjects": [], "shot_type": "bird-eye"}'
    parsed = image_mod._try_parse_scene_json(raw)
    assert parsed is not None
    assert parsed.shot_type is None


def test_try_parse_scene_json_returns_none_on_garbage() -> None:
    assert image_mod._try_parse_scene_json("nothing to see here") is None
    assert image_mod._try_parse_scene_json("") is None


def test_try_parse_scene_json_requires_caption() -> None:
    raw = '{"subjects": ["a"], "shot_type": "wide"}'
    assert image_mod._try_parse_scene_json(raw) is None


def test_structured_output_falls_back_to_raw_text(monkeypatch) -> None:
    """When both attempts produce unparseable output, surface raw text as caption."""
    vlm = image_mod.SceneVLM(model_size="4b")
    monkeypatch.setattr(vlm, "_generate_one", lambda images, prompt: "I cannot do JSON, sorry.")
    description = vlm.analyze_scene([_dummy_pil_image()])
    assert description.caption == "I cannot do JSON, sorry."
    assert description.subjects == []
    assert description.shot_type is None


def test_structured_output_succeeds_first_try(monkeypatch) -> None:
    vlm = image_mod.SceneVLM(model_size="4b")
    monkeypatch.setattr(
        vlm,
        "_generate_one",
        lambda images, prompt: '{"caption": "A wide vista.", "subjects": ["mountain"], "shot_type": "wide"}',
    )
    description = vlm.analyze_scene([_dummy_pil_image()])
    assert description.caption == "A wide vista."
    assert description.subjects == ["mountain"]
    assert description.shot_type == "wide"


def _dummy_pil_image():
    from PIL import Image

    return Image.new("RGB", (8, 8), color=(0, 0, 0))
