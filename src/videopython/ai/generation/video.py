"""Video generation with multi-backend support."""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING

import numpy as np

from videopython.ai.backends import (
    ImageToVideoBackend,
    TextToVideoBackend,
    UnsupportedBackendError,
    get_api_key,
)
from videopython.ai.config import get_default_backend, get_replicate_model
from videopython.base.video import Video

if TYPE_CHECKING:
    from PIL.Image import Image


class TextToVideo:
    """Generates videos from text descriptions using diffusion models."""

    SUPPORTED_BACKENDS: list[str] = ["local", "runway", "luma", "replicate"]

    def __init__(
        self,
        backend: TextToVideoBackend | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize text-to-video generator.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            model: Model to use (only for 'replicate' backend).
            api_key: API key for cloud backends. If None, reads from environment.
        """
        if backend is None:
            backend = get_default_backend("text_to_video")  # type: ignore

        if backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(backend, self.SUPPORTED_BACKENDS)

        self.backend: TextToVideoBackend = backend  # type: ignore
        self.model = model
        self.api_key = api_key
        self._pipeline = None

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

    async def _generate_runway(
        self,
        prompt: str,
        num_frames: int,
    ) -> Video:
        """Generate video using Runway Gen-3."""
        # Runway SDK is not publicly available yet, using HTTP API
        import httpx

        api_key = get_api_key("runway", self.api_key)

        async with httpx.AsyncClient() as client:
            # Start generation
            response = await client.post(
                "https://api.runwayml.com/v1/generations",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "prompt": prompt,
                    "num_frames": num_frames,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            generation_id = response.json()["id"]

            # Poll for completion
            while True:
                status_response = await client.get(
                    f"https://api.runwayml.com/v1/generations/{generation_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30.0,
                )
                status_response.raise_for_status()
                status_data = status_response.json()

                if status_data["status"] == "completed":
                    video_url = status_data["output"]["video_url"]
                    break
                elif status_data["status"] == "failed":
                    raise RuntimeError(f"Runway generation failed: {status_data.get('error')}")

                await asyncio.sleep(2.0)

            # Download video
            video_response = await client.get(video_url, timeout=60.0)
            video_response.raise_for_status()

            return Video.from_bytes(video_response.content)

    async def _generate_luma(
        self,
        prompt: str,
    ) -> Video:
        """Generate video using Luma Dream Machine."""
        from lumaai import AsyncLumaAI

        api_key = get_api_key("luma", self.api_key)
        client = AsyncLumaAI(auth_token=api_key)

        # Start generation
        generation = await client.generations.create(prompt=prompt)

        # Poll for completion
        while True:
            generation = await client.generations.get(generation.id)

            if generation.state == "completed":
                video_url = generation.assets.video
                break
            elif generation.state == "failed":
                raise RuntimeError(f"Luma generation failed: {generation.failure_reason}")

            await asyncio.sleep(2.0)

        # Download video
        import httpx

        async with httpx.AsyncClient() as http_client:
            video_response = await http_client.get(video_url, timeout=60.0)
            video_response.raise_for_status()

            return Video.from_bytes(video_response.content)

    async def _generate_replicate(
        self,
        prompt: str,
        num_frames: int,
    ) -> Video:
        """Generate video using Replicate."""
        import replicate

        api_key = get_api_key("replicate", self.api_key)
        client = replicate.Client(api_token=api_key)

        model_name = self.model or get_replicate_model("text_to_video")

        # Run the model (replicate handles polling internally)
        def _run_replicate() -> str:
            output = client.run(
                model_name,
                input={"prompt": prompt, "num_frames": num_frames},
            )
            # Output is usually a URL or list of URLs
            if isinstance(output, list):
                return output[0]
            return output

        video_url = await asyncio.to_thread(_run_replicate)

        # Download video
        import httpx

        async with httpx.AsyncClient() as http_client:
            video_response = await http_client.get(video_url, timeout=60.0)
            video_response.raise_for_status()

            return Video.from_bytes(video_response.content)

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
        elif self.backend == "runway":
            return await self._generate_runway(prompt, num_frames)
        elif self.backend == "luma":
            return await self._generate_luma(prompt)
        elif self.backend == "replicate":
            return await self._generate_replicate(prompt, num_frames)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)


class ImageToVideo:
    """Generates videos from static images using video diffusion."""

    SUPPORTED_BACKENDS: list[str] = ["local", "runway", "luma", "replicate"]

    def __init__(
        self,
        backend: ImageToVideoBackend | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize image-to-video generator.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            model: Model to use (only for 'replicate' backend).
            api_key: API key for cloud backends. If None, reads from environment.
        """
        if backend is None:
            backend = get_default_backend("image_to_video")  # type: ignore

        if backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(backend, self.SUPPORTED_BACKENDS)

        self.backend: ImageToVideoBackend = backend  # type: ignore
        self.model = model
        self.api_key = api_key
        self._pipeline = None

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

    async def _generate_luma(self, image: Image) -> Video:
        """Generate video using Luma Dream Machine."""
        from lumaai import AsyncLumaAI

        api_key = get_api_key("luma", self.api_key)
        client = AsyncLumaAI(auth_token=api_key)

        # Convert image to base64
        import base64

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        image_url = f"data:image/png;base64,{image_base64}"

        # Start generation
        generation = await client.generations.create(
            prompt="",
            keyframes={"frame0": {"type": "image", "url": image_url}},
        )

        # Poll for completion
        while True:
            generation = await client.generations.get(generation.id)

            if generation.state == "completed":
                video_url = generation.assets.video
                break
            elif generation.state == "failed":
                raise RuntimeError(f"Luma generation failed: {generation.failure_reason}")

            await asyncio.sleep(2.0)

        # Download video
        import httpx

        async with httpx.AsyncClient() as http_client:
            video_response = await http_client.get(video_url, timeout=60.0)
            video_response.raise_for_status()

            return Video.from_bytes(video_response.content)

    async def _generate_runway(self, image: Image) -> Video:
        """Generate video using Runway Gen-3."""
        import base64

        import httpx

        api_key = get_api_key("runway", self.api_key)

        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        async with httpx.AsyncClient() as client:
            # Start generation
            response = await client.post(
                "https://api.runwayml.com/v1/generations",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "image": f"data:image/png;base64,{image_base64}",
                },
                timeout=120.0,
            )
            response.raise_for_status()
            generation_id = response.json()["id"]

            # Poll for completion
            while True:
                status_response = await client.get(
                    f"https://api.runwayml.com/v1/generations/{generation_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30.0,
                )
                status_response.raise_for_status()
                status_data = status_response.json()

                if status_data["status"] == "completed":
                    video_url = status_data["output"]["video_url"]
                    break
                elif status_data["status"] == "failed":
                    raise RuntimeError(f"Runway generation failed: {status_data.get('error')}")

                await asyncio.sleep(2.0)

            # Download video
            video_response = await client.get(video_url, timeout=60.0)
            video_response.raise_for_status()

            return Video.from_bytes(video_response.content)

    async def _generate_replicate(self, image: Image) -> Video:
        """Generate video using Replicate."""
        import base64

        import httpx
        import replicate

        api_key = get_api_key("replicate", self.api_key)
        client = replicate.Client(api_token=api_key)

        model_name = self.model or get_replicate_model("image_to_video")

        # Convert image to base64 data URI
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        image_uri = f"data:image/png;base64,{image_base64}"

        def _run_replicate() -> str:
            output = client.run(
                model_name,
                input={"image": image_uri},
            )
            if isinstance(output, list):
                return output[0]
            return output

        video_url = await asyncio.to_thread(_run_replicate)

        async with httpx.AsyncClient() as http_client:
            video_response = await http_client.get(video_url, timeout=60.0)
            video_response.raise_for_status()

            return Video.from_bytes(video_response.content)

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
        elif self.backend == "luma":
            return await self._generate_luma(image)
        elif self.backend == "runway":
            return await self._generate_runway(image)
        elif self.backend == "replicate":
            return await self._generate_replicate(image)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)
