"""A local, model-agnostic planner backed by an Ollama server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from videopython.ai._ollama import OllamaError, OllamaStructuredClient

from .backend import PlannerError

if TYPE_CHECKING:
    import numpy as np

DEFAULT_OLLAMA_MODEL = "gemma3:27b"


class OllamaVisionLLM:
    """A StructuredVisionLLM backed by a local Ollama server.

    The model must be vision-capable (it is sent keyframes) AND support Ollama's
    structured-output ``format`` (the EditPlan schema constrains the decode). Not
    every model supports schema conditioning -- ``gemma3:27b`` is verified working;
    some builds (e.g. certain MLX ones) fail it. ``ollama pull <model>`` first;
    ``options`` are extra generation options merged over ``temperature=0``.

    Thin wrapper over the shared :class:`OllamaStructuredClient`: its only job is
    to translate :class:`OllamaError` into the :class:`PlannerError` the editor
    retries on.
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        *,
        host: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self._client = OllamaStructuredClient(model=model, host=host, options=options)

    def generate_json(
        self, *, system: str, text: str, images: list[np.ndarray] | None, schema: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            return self._client.generate_json(system=system, text=text, schema=schema, images=images or None)
        except OllamaError as exc:
            raise PlannerError(str(exc)) from exc
