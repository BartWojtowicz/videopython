"""Video generation with multi-backend support."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np

from videopython.ai.backends import (
    ImageToVideoBackend,
    TextToVideoBackend,
    UnsupportedBackendError,
)
from videopython.ai.config import get_default_backend
from videopython.base.video import Video

if TYPE_CHECKING:
    from PIL.Image import Image


class TextToVideo:
    """Generates videos from text descriptions using diffusion models."""

    SUPPORTED_BACKENDS: list[str] = ["local"]

    def __init__(
        self,
        backend: TextToVideoBackend | None = None,
    ):
        """Initialize text-to-video generator.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("text_to_video")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: TextToVideoBackend = resolved_backend  # type: ignore[assignment]
        self._pipeline: Any = None

    def _init_local(self) -> None:
        """Initialize local diffusion pipeline."""
        import torch
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but local TextToVideo requires CUDA.")

        model_name = "cerspense/zeroscope_v2_576w"
        self._pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self._pipeline.scheduler.config)
        self._pipeline.to("cuda")

    async def _generate_local(
        self,
        prompt: str,
        num_steps: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> Video:
        """Generate video using local diffusion model."""
        if self._pipeline is None:
            await asyncio.to_thread(self._init_local)

        def _run_pipeline() -> Video:
            video_frames = self._pipeline(
                prompt,
                num_inference_steps=num_steps,
                height=height,
                width=width,
                num_frames=num_frames,
            ).frames[0]
            video_frames = np.asarray(255 * video_frames, dtype=np.uint8)
            return Video.from_frames(video_frames, fps=24.0)

        return await asyncio.to_thread(_run_pipeline)

    async def generate_video(
        self,
        prompt: str,
        num_steps: int = 25,
        height: int = 320,
        width: int = 576,
        num_frames: int = 24,
    ) -> Video:
        """Generate video from text prompt.

        Args:
            prompt: Text description of desired video content.
            num_steps: Number of diffusion steps (local backend only).
            height: Video height in pixels (local backend only).
            width: Video width in pixels (local backend only).
            num_frames: Number of frames to generate.

        Returns:
            Generated video.
        """
        if self.backend == "local":
            return await self._generate_local(prompt, num_steps, height, width, num_frames)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)


class ImageToVideo:
    """Generates videos from static images using video diffusion."""

    SUPPORTED_BACKENDS: list[str] = ["local"]

    def __init__(
        self,
        backend: ImageToVideoBackend | None = None,
    ):
        """Initialize image-to-video generator.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("image_to_video")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: ImageToVideoBackend = resolved_backend  # type: ignore[assignment]
        self._pipeline: Any = None

    def _init_local(self) -> None:
        """Initialize local diffusion pipeline."""
        import torch
        from diffusers import DiffusionPipeline

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but local ImageToVideo requires CUDA.")

        model_name = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
        self._pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16").to(
            "cuda"
        )

    async def _generate_local(self, image: Image, fps: int) -> Video:
        """Generate video using local diffusion model."""
        if self._pipeline is None:
            await asyncio.to_thread(self._init_local)

        def _run_pipeline() -> Video:
            video_frames = self._pipeline(image=image, fps=fps, output_type="np").frames[0]
            video_frames = np.asarray(255 * video_frames, dtype=np.uint8)
            return Video.from_frames(video_frames, fps=float(fps))

        return await asyncio.to_thread(_run_pipeline)

    async def generate_video(self, image: Image, fps: int = 24) -> Video:
        """Generate video animation from a static image.

        Args:
            image: Input PIL image to animate.
            fps: Target frames per second (local backend only).

        Returns:
            Generated animated video.
        """
        if self.backend == "local":
            return await self._generate_local(image, fps)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)
