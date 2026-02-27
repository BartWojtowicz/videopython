"""Image generation using local diffusion models."""

from __future__ import annotations

from typing import Any

from PIL import Image

from videopython.ai._device import log_device_initialization, select_device


class TextToImage:
    """Generates images from text descriptions using local models."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._pipeline: Any = None

    def _init_local(self) -> None:
        """Initialize local diffusion pipeline."""
        import torch
        from diffusers import DiffusionPipeline

        requested_device = self.device
        device = select_device(self.device, mps_allowed=True)
        dtype = torch.float16 if device == "cuda" else torch.float32
        variant = "fp16" if device == "cuda" else None

        model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        self._pipeline = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            variant=variant,
            use_safetensors=True,
        )
        self._pipeline.to(device)
        self.device = device
        log_device_initialization(
            "TextToImage",
            requested_device=requested_device,
            resolved_device=device,
        )

        if device == "mps":
            self._pipeline.enable_attention_slicing()

    def generate_image(self, prompt: str) -> Image.Image:
        """Generate an image from a text prompt."""
        if self._pipeline is None:
            self._init_local()
        return self._pipeline(prompt=prompt).images[0]
