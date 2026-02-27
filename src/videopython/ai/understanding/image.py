"""Image understanding using local models."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from videopython.ai._device import select_device


class ImageToText:
    """Generates text descriptions of images using BLIP."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._processor: Any = None
        self._model: Any = None

    def _init_local(self) -> None:
        """Initialize local BLIP model."""
        from transformers.models.blip import BlipForConditionalGeneration, BlipProcessor

        # MPS is intentionally disabled here due to worse BLIP performance/compatibility.
        device = select_device(self.device, mps_allowed=False)

        model_name = "Salesforce/blip-image-captioning-large"
        self._processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self._model = BlipForConditionalGeneration.from_pretrained(model_name)
        self._model.to(device)
        self.device = device

    def describe_image(
        self,
        image: np.ndarray | Image.Image,
        prompt: str | None = None,
    ) -> str:
        """Generate a text description of an image."""
        if self._model is None:
            self._init_local()

        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        inputs = self._processor(pil_image, prompt, return_tensors="pt").to(self.device)
        output = self._model.generate(**inputs, max_new_tokens=50)
        return self._processor.decode(output[0], skip_special_tokens=True)
