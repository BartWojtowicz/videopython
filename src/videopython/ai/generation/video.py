"""Video generation with multi-backend support."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np

from videopython.ai.backends import (
    ImageToVideoBackend,
    TextToVideoBackend,
    UnsupportedBackendError,
    VideoUpscalerBackend,
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
        from diffusers import CogVideoXPipeline

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but local TextToVideo requires CUDA.")

        model_name = "THUDM/CogVideoX1.5-5B"
        self._pipeline = CogVideoXPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self._pipeline.enable_sequential_cpu_offload()
        self._pipeline.vae.enable_tiling()
        self._pipeline.vae.enable_slicing()

    async def _generate_local(
        self,
        prompt: str,
        num_steps: int,
        num_frames: int,
        guidance_scale: float,
    ) -> Video:
        """Generate video using local CogVideoX diffusion model."""
        if self._pipeline is None:
            await asyncio.to_thread(self._init_local)

        def _run_pipeline() -> Video:
            import torch

            video_frames = self._pipeline(
                prompt=prompt,
                num_inference_steps=num_steps,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]
            video_frames = np.asarray(video_frames, dtype=np.uint8)
            return Video.from_frames(video_frames, fps=16.0)

        return await asyncio.to_thread(_run_pipeline)

    async def generate_video(
        self,
        prompt: str,
        num_steps: int = 50,
        num_frames: int = 81,
        guidance_scale: float = 6.0,
    ) -> Video:
        """Generate video from text prompt using CogVideoX1.5-5B.

        Args:
            prompt: Text description of desired video content (English, max 224 tokens).
            num_steps: Number of diffusion steps. Default 50.
            num_frames: Number of frames to generate. Should be 16N + 1 where N <= 10.
                        Default 81 (~5 seconds at 16fps). Use 161 for ~10 seconds.
            guidance_scale: Prompt guidance strength. Default 6.0.

        Returns:
            Generated video at 1360x768 resolution, 16fps.
        """
        if self.backend == "local":
            return await self._generate_local(prompt, num_steps, num_frames, guidance_scale)
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
        from diffusers import CogVideoXImageToVideoPipeline

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but local ImageToVideo requires CUDA.")

        model_name = "THUDM/CogVideoX1.5-5B-I2V"
        self._pipeline = CogVideoXImageToVideoPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self._pipeline.enable_sequential_cpu_offload()
        self._pipeline.vae.enable_tiling()
        self._pipeline.vae.enable_slicing()

    async def _generate_local(
        self,
        image: Image,
        prompt: str,
        num_steps: int,
        num_frames: int,
        guidance_scale: float,
    ) -> Video:
        """Generate video using local CogVideoX I2V diffusion model."""
        if self._pipeline is None:
            await asyncio.to_thread(self._init_local)

        def _run_pipeline() -> Video:
            import torch

            video_frames = self._pipeline(
                prompt=prompt,
                image=image,
                num_inference_steps=num_steps,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]
            video_frames = np.asarray(video_frames, dtype=np.uint8)
            return Video.from_frames(video_frames, fps=16.0)

        return await asyncio.to_thread(_run_pipeline)

    async def generate_video(
        self,
        image: Image,
        prompt: str = "",
        num_steps: int = 50,
        num_frames: int = 81,
        guidance_scale: float = 6.0,
    ) -> Video:
        """Generate video animation from a static image using CogVideoX1.5-5B-I2V.

        Args:
            image: Input PIL image to animate.
            prompt: Text description to guide the animation (English, max 224 tokens).
            num_steps: Number of diffusion steps. Default 50.
            num_frames: Number of frames to generate. Should be 16N + 1 where N <= 10.
                        Default 81 (~5 seconds at 16fps). Use 161 for ~10 seconds.
            guidance_scale: Prompt guidance strength. Default 6.0.

        Returns:
            Generated animated video at 16fps.
        """
        if self.backend == "local":
            return await self._generate_local(image, prompt, num_steps, num_frames, guidance_scale)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)


class VideoUpscaler:
    """Upscales video resolution using AI super-resolution models.

    Uses RealBasicVSR for 4x upscaling with temporal consistency.
    """

    SUPPORTED_BACKENDS: list[str] = ["local"]

    def __init__(
        self,
        backend: VideoUpscalerBackend | None = None,
    ):
        """Initialize video upscaler.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("video_upscaler")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: VideoUpscalerBackend = resolved_backend  # type: ignore[assignment]
        self._inferencer: Any = None

    def _init_local(self) -> None:
        """Initialize local RealBasicVSR model via MMagic."""
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but local VideoUpscaler requires CUDA.")

        from mmagic.apis import MMagicInferencer

        self._inferencer = MMagicInferencer(model_name="realbasicvsr")

    async def _upscale_local(self, video: Video) -> Video:
        """Upscale video using local RealBasicVSR model."""
        import tempfile
        from pathlib import Path

        if self._inferencer is None:
            await asyncio.to_thread(self._init_local)

        def _run_upscale() -> Video:
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = Path(tmpdir) / "input.mp4"
                output_path = Path(tmpdir) / "output.mp4"

                video.save(str(input_path))

                self._inferencer.infer(video=str(input_path), result_out_dir=str(output_path))

                return Video.from_path(str(output_path))

        return await asyncio.to_thread(_run_upscale)

    async def upscale(self, video: Video) -> Video:
        """Upscale video resolution by 4x.

        Args:
            video: Input video to upscale.

        Returns:
            Upscaled video with 4x resolution.
        """
        if self.backend == "local":
            return await self._upscale_local(video)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)
