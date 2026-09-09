"""Shared structured-generation client over a local Ollama server."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np

from videopython.ai._optional import require
from videopython.ai.errors import AiError
from videopython.ai.keyframe import encode_png_b64

logger = logging.getLogger(__name__)


class OllamaError(AiError, RuntimeError):
    """Ollama returned unusable output (non-JSON or an unexpected shape)."""


# Ollama's own default context window is 4096 tokens, and an oversized request
# *fails* (``exceed_context_size_error``) instead of being truncated -- so a call
# carrying images has to size the window before sending. A vision model spends
# roughly this many tokens per image. Deliberately generous: underestimating
# fails the request outright, overestimating only costs KV-cache memory.
_TOKENS_PER_IMAGE = 1024
# Headroom for the system prompt, the user text, and the generated answer.
_TEXT_TOKEN_ALLOWANCE = 2048
# Never ask for less than Ollama's own default.
_MIN_NUM_CTX = 4096
_CONTEXT_OVERFLOW_RE = re.compile(r"request \((\d+) tokens\) exceeds the available context size")


def _round_num_ctx(token_count: int) -> int:
    return max(_MIN_NUM_CTX, 1 << (token_count - 1).bit_length())


def _num_ctx_for_images(image_count: int) -> int:
    """Estimate the initial context window for ``image_count`` images.

    Rounded up to a power of two so a run over scenes with differing frame counts
    reuses a handful of window sizes instead of a new one per call -- ``num_ctx``
    is a runner-level setting, so varying it every call risks reloading the model
    between scenes.
    """
    needed = image_count * _TOKENS_PER_IMAGE + _TEXT_TOKEN_ALLOWANCE
    return _round_num_ctx(needed)


def _num_ctx_after_overflow(error: str, current: int) -> int | None:
    """Return a larger context after Ollama reports a context overflow."""
    match = _CONTEXT_OVERFLOW_RE.search(error)
    if match:
        return _round_num_ctx(int(match.group(1)) + _TEXT_TOKEN_ALLOWANCE)
    if "exceed_context_size_error" in error:
        return current * 2
    return None


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

    Image requests start with a context estimate based on the image count. If a
    model uses more visual tokens, the request retries once with the token count
    reported by Ollama. An explicit ``num_ctx`` in ``options`` always wins.
    """

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> None:
        self.model = model
        self.host = host
        self.keep_alive = keep_alive
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
        options = self.options
        image_count = len(images) if images else 0
        auto_num_ctx = image_count > 0 and "num_ctx" not in options
        if auto_num_ctx:
            options = {**options, "num_ctx": _num_ctx_for_images(image_count)}
        kwargs: dict[str, Any] = {}
        if self._supports_thinking():
            kwargs["think"] = False
        if self.keep_alive is not None:
            kwargs["keep_alive"] = self.keep_alive
        client = self._get_client()
        try:
            response = client.chat(model=self.model, messages=messages, format=schema, options=options, **kwargs)
        except Exception as exc:
            retry_num_ctx = _num_ctx_after_overflow(str(exc), options["num_ctx"]) if auto_num_ctx else None
            if retry_num_ctx is None:
                raise
            options = {**options, "num_ctx": retry_num_ctx}
            response = client.chat(model=self.model, messages=messages, format=schema, options=options, **kwargs)
        if getattr(response, "done_reason", None) == "length":
            raise OllamaError("Ollama exhausted its output budget; refusing an incomplete response")
        content = response.message.content
        try:
            data = json.loads(content)
        except (ValueError, TypeError) as exc:
            raise OllamaError(f"Ollama returned non-JSON output: {content!r}") from exc
        if not isinstance(data, dict):
            raise OllamaError(f"Ollama returned a non-object JSON value: {type(data).__name__}")
        return data

    def model_provenance(self) -> dict[str, str | None] | None:
        """Resolve the used model tag to a server digest, without inference."""
        if self._client is None:
            return None
        try:
            models = self._client.list().models
        except Exception:
            return None
        for model in models:
            if model.model in (self.model, f"{self.model}:latest"):
                return {model.model: model.digest}
        return None

    def unload(self) -> None:
        try:
            if self._client is not None and self.keep_alive is not None:
                # Release explicit residency before the low-memory pipeline
                # loads its next GPU model. Merely dropping the HTTP client
                # leaves the Ollama runner resident on the server.
                self._client.generate(model=self.model, keep_alive=0)
        except Exception as exc:
            logger.warning("Could not unload Ollama model %s; server memory may remain allocated: %s", self.model, exc)
        finally:
            self._client = None
