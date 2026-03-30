"""Video generation using local diffusion models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from videopython.ai._device import log_device_initialization, select_device
from videopython.base.video import Video

if TYPE_CHECKING:
    from PIL.Image import Image


def _get_torch_device_and_dtype(device: str | None) -> tuple[str, Any]:
    """Get the best available torch device and dtype for CogVideoX."""
    import torch

    selected_device = select_device(device, mps_allowed=False)
    if selected_device == "cuda":
        return selected_device, torch.bfloat16
    return selected_device, torch.float32


class TextToVideo:
    """Generates videos from text descriptions using local diffusion models."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._pipeline: Any = None
        self._device: str | None = None

    def _init_local(self) -> None:
        from diffusers import CogVideoXPipeline

        requested_device = self.device
        self._device, dtype = _get_torch_device_and_dtype(self.device)

        model_name = "THUDM/CogVideoX1.5-5B"
        self._pipeline = CogVideoXPipeline.from_pretrained(model_name, torch_dtype=dtype)
        self._pipeline.to(self._device)
        self.device = self._device
        log_device_initialization(
            "TextToVideo",
            requested_device=requested_device,
            resolved_device=self._device,
        )

    def generate_video(
        self,
        prompt: str,
        num_steps: int = 50,
        num_frames: int = 81,
        guidance_scale: float = 6.0,
    ) -> Video:
        """Generate video from text prompt."""
        import torch

        if self._pipeline is None:
            self._init_local()

        video_frames = self._pipeline(
            prompt=prompt,
            num_inference_steps=num_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=self._device).manual_seed(42),
        ).frames[0]
        video_frames = np.asarray(video_frames, dtype=np.uint8)
        return Video.from_frames(video_frames, fps=16.0)


class ImageToVideo:
    """Generates videos from static images using local video diffusion."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._pipeline: Any = None
        self._device: str | None = None

    def _init_local(self) -> None:
        from diffusers import CogVideoXImageToVideoPipeline

        requested_device = self.device
        self._device, dtype = _get_torch_device_and_dtype(self.device)

        model_name = "THUDM/CogVideoX1.5-5B-I2V"
        self._pipeline = CogVideoXImageToVideoPipeline.from_pretrained(model_name, torch_dtype=dtype)
        self._pipeline.to(self._device)
        self.device = self._device
        log_device_initialization(
            "ImageToVideo",
            requested_device=requested_device,
            resolved_device=self._device,
        )

    def generate_video(
        self,
        image: Image,
        prompt: str = "",
        num_steps: int = 50,
        num_frames: int = 81,
        guidance_scale: float = 6.0,
    ) -> Video:
        """Generate video animation from a static image."""
        import torch

        if self._pipeline is None:
            self._init_local()

        video_frames = self._pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=num_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=self._device).manual_seed(42),
        ).frames[0]
        video_frames = np.asarray(video_frames, dtype=np.uint8)
        return Video.from_frames(video_frames, fps=16.0)
