"""Image generation with multi-backend support."""

from __future__ import annotations

import asyncio
import io
from typing import Any

from PIL import Image

from videopython.ai.backends import TextToImageBackend, UnsupportedBackendError, get_api_key
from videopython.ai.config import get_default_backend


class TextToImage:
    """Generates images from text descriptions."""

    SUPPORTED_BACKENDS: list[str] = ["local", "openai"]

    def __init__(
        self,
        backend: TextToImageBackend | None = None,
        api_key: str | None = None,
    ):
        """Initialize text-to-image generator.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            api_key: API key for cloud backends. If None, reads from environment.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("text_to_image")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: TextToImageBackend = resolved_backend  # type: ignore[assignment]
        self.api_key = api_key
        self._pipeline: Any = None

    def _init_local(self) -> None:
        """Initialize local diffusion pipeline."""
        import torch
        from diffusers import DiffusionPipeline

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but local TextToImage requires CUDA.")

        model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        self._pipeline = DiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        self._pipeline.to("cuda")

    async def _generate_local(self, prompt: str) -> Image.Image:
        """Generate image using local diffusion model."""
        if self._pipeline is None:
            await asyncio.to_thread(self._init_local)

        def _run_pipeline() -> Image.Image:
            return self._pipeline(prompt=prompt).images[0]

        return await asyncio.to_thread(_run_pipeline)

    async def _generate_openai(self, prompt: str, size: str) -> Image.Image:
        """Generate image using OpenAI DALL-E."""
        import httpx
        from openai import AsyncOpenAI

        api_key = get_api_key("openai", self.api_key)
        client = AsyncOpenAI(api_key=api_key)

        response = await client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,  # type: ignore
            quality="hd",
            n=1,
        )

        image_url = response.data[0].url
        if image_url is None:
            raise RuntimeError("OpenAI returned no image URL")

        # Download the image
        async with httpx.AsyncClient() as http_client:
            image_response = await http_client.get(image_url, timeout=60.0)
            image_response.raise_for_status()

            return Image.open(io.BytesIO(image_response.content))

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
    ) -> Image.Image:
        """Generate image from text prompt.

        Args:
            prompt: Text description of desired image.
            size: Image size (OpenAI backend only). Options: "1024x1024", "1792x1024", "1024x1792".

        Returns:
            Generated PIL Image.
        """
        if self.backend == "local":
            return await self._generate_local(prompt)
        elif self.backend == "openai":
            return await self._generate_openai(prompt, size)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)
