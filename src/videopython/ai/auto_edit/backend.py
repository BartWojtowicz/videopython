"""The SDK-free seam between the editor and a structured-vision model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class TextPart:
    """A text chunk in a planner prompt."""

    text: str


@dataclass
class ImagePart:
    """A keyframe in a planner prompt: an RGB (H, W, 3) uint8 array."""

    image: np.ndarray
    label: str | None = None


Part = TextPart | ImagePart


class PlannerError(RuntimeError):
    """A backend produced unusable output; the editor retries (infra errors should propagate instead)."""


@runtime_checkable
class StructuredVisionLLM(Protocol):
    """Returns schema-shaped JSON from interleaved text + images; raises PlannerError on bad output."""

    def generate_json(self, *, system: str, parts: list[Part], schema: dict[str, Any]) -> dict[str, Any]: ...
