"""Scene description via a local Ollama vision model."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from videopython.ai._ollama import OllamaStructuredClient
from videopython.ai._predictor import ManagedPredictor
from videopython.base.description import SceneDescription

DEFAULT_SCENE_VLM_MODEL = "qwen3.6:27b"

_SHOT_TYPES = ("wide", "medium", "close-up", "extreme close-up", "establishing", "other")

_SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "subjects": {"type": "array", "items": {"type": "string"}},
        "shot_type": {"type": "string", "enum": list(_SHOT_TYPES)},
    },
    "required": ["caption", "subjects", "shot_type"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You describe the visual content of video frames. Return a JSON object with: "
    "caption (one concise sentence), subjects (lowercase noun phrases of the main "
    "on-screen subjects), and shot_type (one of "
    "wide / medium / close-up / extreme close-up / establishing / other)."
)
_USER_PROMPT = "Describe these frames."


class SceneVLM(ManagedPredictor):
    """Generates structured scene descriptions with a local Ollama vision model.

    The model must be vision-capable and support Ollama's structured-output
    ``format``; ``ollama pull <model>`` first. ``options`` are extra Ollama
    generation options merged over ``temperature=0``.
    """

    def __init__(
        self,
        model: str = DEFAULT_SCENE_VLM_MODEL,
        *,
        host: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self._client = OllamaStructuredClient(model=model, host=host, options=options)

    def analyze_frame(self, image: np.ndarray | Image.Image, prompt: str | None = None) -> SceneDescription:
        """Analyze one frame and return a structured scene description."""
        return self.analyze_scene([image], prompt=prompt)

    def analyze_scene(self, images: list[np.ndarray | Image.Image], prompt: str | None = None) -> SceneDescription:
        """Analyze a scene's frames and return a structured description."""
        if not images:
            raise ValueError("`images` must contain at least one frame")
        frames = [_to_rgb_array(image) for image in images]
        data = self._client.generate_json(
            system=_SYSTEM_PROMPT, text=prompt or _USER_PROMPT, schema=_SCENE_SCHEMA, images=frames
        )
        shot_type = data.get("shot_type")
        return SceneDescription(
            caption=str(data.get("caption", "")),
            subjects=[str(s) for s in data.get("subjects", [])],
            shot_type=shot_type if shot_type in _SHOT_TYPES else None,
        )

    def unload(self) -> None:
        self._client.unload()


def _to_rgb_array(image: np.ndarray | Image.Image) -> np.ndarray:
    return image if isinstance(image, np.ndarray) else np.asarray(image.convert("RGB"))
