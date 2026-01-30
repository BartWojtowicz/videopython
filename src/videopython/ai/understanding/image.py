"""Image understanding with multi-backend support."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from videopython.ai.backends import ImageToTextBackend, UnsupportedBackendError, get_api_key
from videopython.ai.config import get_default_backend


class ImageToText:
    """Generates text descriptions of images."""

    SUPPORTED_BACKENDS: list[str] = ["local", "openai", "gemini"]

    def __init__(
        self,
        backend: ImageToTextBackend | None = None,
        device: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize image-to-text model.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            device: Device for local backend ('cuda' or 'cpu').
            api_key: API key for cloud backends. If None, reads from environment.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("image_to_text")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: ImageToTextBackend = resolved_backend  # type: ignore[assignment]
        self.device = device
        self.api_key = api_key

        self._processor: Any = None
        self._model: Any = None

    def _init_local(self) -> None:
        """Initialize local BLIP model."""
        import torch
        from transformers.models.blip import BlipForConditionalGeneration, BlipProcessor

        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                # MPS is slower than CPU for BLIP (~2x slower in benchmarks),
                # so we default to CPU for best performance
                device = "cpu"

        model_name = "Salesforce/blip-image-captioning-large"
        self._processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self._model = BlipForConditionalGeneration.from_pretrained(model_name)
        self._model.to(device)

        self.device = device

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _describe_local(
        self,
        image: np.ndarray | Image.Image,
        prompt: str | None,
    ) -> str:
        """Generate description using local BLIP model."""
        if self._model is None:
            self._init_local()

        # Convert numpy array to PIL Image if needed
        pil_image = image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)

        inputs = self._processor(pil_image, prompt, return_tensors="pt").to(self.device)
        output = self._model.generate(**inputs, max_new_tokens=50)
        return self._processor.decode(output[0], skip_special_tokens=True)

    def _describe_openai(
        self,
        image: np.ndarray | Image.Image,
        prompt: str | None,
    ) -> str:
        """Generate description using OpenAI GPT-4o."""
        from openai import OpenAI

        api_key = get_api_key("openai", self.api_key)
        client = OpenAI(api_key=api_key)

        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_base64 = self._image_to_base64(image)

        system_prompt = "You are an image analysis assistant. Describe images concisely."
        user_prompt = prompt or "Describe this image in 1-2 sentences."

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                    ],
                },
            ],
            max_tokens=100,
        )

        return response.choices[0].message.content or ""

    def _describe_gemini(
        self,
        image: np.ndarray | Image.Image,
        prompt: str | None,
    ) -> str:
        """Generate description using Google Gemini."""
        import google.generativeai as genai

        api_key = get_api_key("gemini", self.api_key)
        genai.configure(api_key=api_key)

        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        model = genai.GenerativeModel("gemini-2.0-flash")
        user_prompt = prompt or "Describe this image in 1-2 sentences."

        response = model.generate_content([user_prompt, image])
        return response.text

    def describe_image(
        self,
        image: np.ndarray | Image.Image,
        prompt: str | None = None,
    ) -> str:
        """Generate a text description of an image.

        Args:
            image: Image as numpy array (H, W, 3) in RGB format or PIL Image.
            prompt: Optional text prompt to guide the description.

        Returns:
            Text description of the image.
        """
        if self.backend == "local":
            return self._describe_local(image, prompt)
        elif self.backend == "openai":
            return self._describe_openai(image, prompt)
        elif self.backend == "gemini":
            return self._describe_gemini(image, prompt)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)
