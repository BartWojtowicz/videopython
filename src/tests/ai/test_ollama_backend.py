"""Tests for OllamaVisionLLM with an injected fake client (no server, no ollama package)."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from videopython.ai._ollama import _encode_png_b64
from videopython.ai.auto_edit import ImagePart, OllamaVisionLLM, Part, PlannerError, TextPart


class _FakeClient:
    """Records chat() kwargs and returns a fixed ChatResponse-shaped object."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def chat(self, *, model: str, messages: list[Any], format: Any, options: dict[str, Any]) -> SimpleNamespace:
        self.calls.append({"model": model, "messages": messages, "format": format, "options": options})
        return SimpleNamespace(message=SimpleNamespace(content=self.content))


def _inject(backend: OllamaVisionLLM, fake: _FakeClient) -> None:
    # OllamaVisionLLM composes an OllamaStructuredClient whose inner ollama client is _client.
    backend._client._client = fake


def test_generate_json_builds_messages_and_parses() -> None:
    backend = OllamaVisionLLM(model="m")
    fake = _FakeClient('{"segments": [{"scene_id": "x"}]}')
    _inject(backend, fake)
    schema = {"type": "object", "properties": {}}
    parts: list[Part] = [TextPart("brief"), ImagePart(image=np.zeros((4, 4, 3), dtype=np.uint8), label="x")]

    out = backend.generate_json(system="sys", parts=parts, schema=schema)

    assert out == {"segments": [{"scene_id": "x"}]}
    call = fake.calls[0]
    assert call["model"] == "m"
    assert call["format"] == schema  # schema-constrained decode
    assert call["options"]["temperature"] == 0.0
    sys_msg, user_msg = call["messages"]
    assert sys_msg == {"role": "system", "content": "sys"}
    assert "brief" in user_msg["content"]
    assert len(user_msg["images"]) == 1


def test_no_images_omits_images_key() -> None:
    backend = OllamaVisionLLM()
    fake = _FakeClient("{}")
    _inject(backend, fake)
    backend.generate_json(system="s", parts=[TextPart("only text")], schema={})
    assert "images" not in fake.calls[0]["messages"][1]


def test_non_json_raises_planner_error() -> None:
    backend = OllamaVisionLLM()
    _inject(backend, _FakeClient("I cannot help with that."))
    with pytest.raises(PlannerError):
        backend.generate_json(system="s", parts=[TextPart("t")], schema={})


def test_encode_png_b64_roundtrips() -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[0, 0] = [255, 0, 0]
    decoded = Image.open(io.BytesIO(base64.b64decode(_encode_png_b64(frame))))
    assert decoded.size == (6, 4)  # PIL size is (width, height)
