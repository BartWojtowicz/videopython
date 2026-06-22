"""Shared structured-generation client over a local Ollama server."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import cv2
import numpy as np
from PIL import Image

from videopython.ai._optional import require


class OllamaError(RuntimeError):
    """Ollama returned unusable output (non-JSON or an unexpected shape)."""


class OllamaStructuredClient:
    """Generate schema-constrained JSON from text + optional images via Ollama.

    Shared by the auto-edit planner, scene captioner, and translator. The model
    must be served by a local Ollama daemon and support structured-output
    ``format`` (and vision, when images are passed); ``options`` are extra Ollama
    generation options merged over ``temperature=0``.
    """

    def __init__(self, model: str, *, host: str | None = None, options: dict[str, Any] | None = None) -> None:
        self.model = model
        self.host = host
        self.options: dict[str, Any] = {"temperature": 0.0, **(options or {})}
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            ollama = require("ollama", feature="Ollama")
            self._client = ollama.Client(host=self.host)
        return self._client

    def generate_json(
        self,
        *,
        system: str,
        text: str,
        schema: dict[str, Any],
        images: list[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Return the parsed JSON object Ollama generates under ``schema``."""
        user: dict[str, Any] = {"role": "user", "content": text}
        if images:
            user["images"] = [_encode_png_b64(image) for image in images]
        messages = [{"role": "system", "content": system}, user]
        response = self._get_client().chat(model=self.model, messages=messages, format=schema, options=self.options)
        content = response.message.content
        try:
            data = json.loads(content)
        except (ValueError, TypeError) as exc:
            raise OllamaError(f"Ollama returned non-JSON output: {content!r}") from exc
        if not isinstance(data, dict):
            raise OllamaError(f"Ollama returned a non-object JSON value: {type(data).__name__}")
        return data

    def unload(self) -> None:
        self._client = None


# Used only on the MCP keyframe path (videopython.mcp). SceneVLM captioning and the local
# planner deliberately encode full-resolution frames via _encode_png_b64.
KEYFRAME_MAX_DIM = 768  # bound a keyframe's longest side before PNG-encoding for the MCP payload


def _downscale(frame: np.ndarray, max_dim: int = KEYFRAME_MAX_DIM) -> np.ndarray:
    """Shrink an RGB frame so its longest side is at most ``max_dim`` (aspect preserved; never upscales)."""
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def _encode_png_b64(frame: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
