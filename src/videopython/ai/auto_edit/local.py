"""A local, model-agnostic planner backed by an Ollama server."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np
from PIL import Image

from videopython.ai._optional import require

from .backend import ImagePart, Part, PlannerError, TextPart

DEFAULT_OLLAMA_MODEL = "llama3.2-vision"


class OllamaVisionLLM:
    """A StructuredVisionLLM backed by a local Ollama server (any pulled vision model).

    ``model`` is the Ollama tag to use (``ollama pull <model>`` first); ``options``
    are extra Ollama generation options merged over ``temperature=0``.
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        *,
        host: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.host = host
        self.options: dict[str, Any] = {"temperature": 0.0, **(options or {})}
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            ollama = require("ollama", "ai", feature="OllamaVisionLLM")
            self._client = ollama.Client(host=self.host)
        return self._client

    def generate_json(self, *, system: str, parts: list[Part], schema: dict[str, Any]) -> dict[str, Any]:
        text = "\n\n".join(part.text for part in parts if isinstance(part, TextPart))
        images = [_encode_png_b64(part.image) for part in parts if isinstance(part, ImagePart)]
        user: dict[str, Any] = {"role": "user", "content": text}
        if images:
            user["images"] = images
        messages = [{"role": "system", "content": system}, user]
        # Ollama's structured output: the schema constrains the decode to valid JSON.
        response = self._get_client().chat(model=self.model, messages=messages, format=schema, options=self.options)
        content = response.message.content
        try:
            return json.loads(content)
        except (ValueError, TypeError) as exc:
            raise PlannerError(f"Ollama returned non-JSON output: {content!r}") from exc


def _encode_png_b64(frame: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
