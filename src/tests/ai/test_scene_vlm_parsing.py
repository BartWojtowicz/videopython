"""Tests for the Ollama-backed SceneVLM (injected fake client; no server/ollama)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from videopython.ai.understanding.image import SceneVLM
from videopython.base.description import SceneDescription


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def chat(self, *, model: str, messages: list[Any], format: Any, options: dict[str, Any]) -> SimpleNamespace:
        self.calls.append({"model": model, "messages": messages, "format": format, "options": options})
        return SimpleNamespace(message=SimpleNamespace(content=self.content))


def _vlm_with(content: str) -> tuple[SceneVLM, _FakeClient]:
    vlm = SceneVLM(model="m")
    fake = _FakeClient(content)
    vlm._client._client = fake  # inject inside the shared OllamaStructuredClient
    return vlm, fake


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_analyze_scene_returns_scene_description() -> None:
    vlm, fake = _vlm_with(json.dumps({"caption": "a cat sleeps", "subjects": ["cat"], "shot_type": "close-up"}))
    desc = vlm.analyze_scene([_frame(), _frame()])

    assert isinstance(desc, SceneDescription)
    assert desc.caption == "a cat sleeps"
    assert desc.subjects == ["cat"]
    assert desc.shot_type == "close-up"
    call = fake.calls[0]
    assert len(call["messages"][1]["images"]) == 2  # both frames sent
    assert call["format"]["properties"]["shot_type"]["enum"]  # schema-constrained


def test_analyze_frame_delegates() -> None:
    vlm, _ = _vlm_with(json.dumps({"caption": "x", "subjects": [], "shot_type": "wide"}))
    desc = vlm.analyze_frame(_frame())
    assert desc.caption == "x"
    assert desc.shot_type == "wide"


def test_invalid_shot_type_becomes_none() -> None:
    vlm, _ = _vlm_with(json.dumps({"caption": "x", "subjects": [], "shot_type": "bogus"}))
    assert vlm.analyze_scene([_frame()]).shot_type is None


def test_empty_images_raises() -> None:
    vlm, _ = _vlm_with("{}")
    with pytest.raises(ValueError, match="frame"):
        vlm.analyze_scene([])
