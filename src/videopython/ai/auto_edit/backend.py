"""The SDK-free seam between the editor and a structured-vision model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


class PlannerError(RuntimeError):
    """A backend produced unusable output; the editor retries (infra errors should propagate instead)."""


@runtime_checkable
class StructuredVisionLLM(Protocol):
    """Returns schema-shaped JSON from a system prompt + text + optional keyframes.

    The signature mirrors
    :meth:`videopython.ai._ollama.OllamaStructuredClient.generate_json`, so any
    structured-generation client satisfies it structurally. Implementations
    raise :class:`PlannerError` on unusable output (the editor retries those);
    infra errors should propagate so they are not silently retried.
    """

    def generate_json(
        self, *, system: str, text: str, images: list[np.ndarray] | None, schema: dict[str, Any]
    ) -> dict[str, Any]: ...
