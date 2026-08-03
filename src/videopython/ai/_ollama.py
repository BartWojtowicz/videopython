"""Shared structured-generation client over a local Ollama server."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from videopython.ai._optional import require
from videopython.ai.errors import AiError
from videopython.ai.keyframe import encode_png_b64


class OllamaError(AiError, RuntimeError):
    """Ollama returned unusable output (non-JSON or an unexpected shape)."""


class OllamaStructuredClient:
    """Generate schema-constrained JSON from text + optional images via Ollama.

    Shared by the auto-edit planner, scene captioner, and translator. The model
    must be served by a local Ollama daemon and support structured-output
    ``format`` (and vision, when images are passed); ``options`` are extra Ollama
    generation options merged over ``temperature=0``.

    Reasoning models emit their chain-of-thought *before* the schema-constrained
    answer, and that thinking counts against ``num_predict``. On a reasoning model
    (the default ``qwen3.6:27b`` is one) a translation call spends its entire token
    budget thinking, stops on ``length``, and returns empty content. None of these
    callers want the chain-of-thought, so thinking is disabled on models that
    support it.
    """

    def __init__(self, model: str, *, host: str | None = None, options: dict[str, Any] | None = None) -> None:
        self.model = model
        self.host = host
        self.options: dict[str, Any] = {"temperature": 0.0, **(options or {})}
        self._client: Any = None
        self._thinking_capable: bool | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            ollama = require("ollama", feature="Ollama")
            self._client = ollama.Client(host=self.host)
        return self._client

    def _supports_thinking(self) -> bool:
        """Whether the model advertises Ollama's ``thinking`` capability (cached).

        Passing ``think`` to a model that has no thinking capability is an error, so
        this is checked rather than assumed.
        """
        if self._thinking_capable is None:
            capabilities = self._get_client().show(self.model).capabilities or []
            self._thinking_capable = "thinking" in capabilities
        return self._thinking_capable

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
            user["images"] = [encode_png_b64(image) for image in images]
        messages = [{"role": "system", "content": system}, user]
        kwargs: dict[str, Any] = {}
        if self._supports_thinking():
            kwargs["think"] = False
        response = self._get_client().chat(
            model=self.model, messages=messages, format=schema, options=self.options, **kwargs
        )
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
