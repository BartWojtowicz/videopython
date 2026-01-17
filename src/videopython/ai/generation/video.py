"""Video generation with multi-backend support."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from videopython.ai.backends import (
    ImageToVideoBackend,
    TextToVideoBackend,
    UnsupportedBackendError,
    get_api_key,
)
from videopython.ai.config import get_default_backend
from videopython.ai.exceptions import LumaGenerationError, RunwayGenerationError
from videopython.base.video import Video

if TYPE_CHECKING:
    from PIL.Image import Image


def _get_torch_device_and_dtype() -> tuple[str, Any]:
    """Get the best available torch device and appropriate dtype for CogVideoX.

    CogVideoX requires CUDA - MPS is not supported due to memory requirements
    (the model needs 364+ GB for attention computations on MPS).

    Returns:
        Tuple of (device_name, dtype) - e.g. ("cuda", torch.bfloat16)

    Raises:
        ValueError: If CUDA is not available.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    else:
        raise ValueError(
            "CUDA is required for local video generation. "
            "CogVideoX models are not supported on MPS due to memory requirements. "
            "Use a cloud backend (luma, runway) or run on a CUDA-enabled GPU."
        )


class TextToVideo:
    """Generates videos from text descriptions using diffusion models."""

    SUPPORTED_BACKENDS: list[str] = ["local", "luma"]

    def __init__(
        self,
        backend: TextToVideoBackend | None = None,
        api_key: str | None = None,
    ):
        """Initialize text-to-video generator.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            api_key: API key for cloud backends. If None, uses environment variable.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("text_to_video")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: TextToVideoBackend = resolved_backend  # type: ignore[assignment]
        self._api_key = api_key
        self._pipeline: Any = None
        self._device: str | None = None

    def _init_local(self) -> None:
        """Initialize local diffusion pipeline."""
        from diffusers import CogVideoXPipeline

        self._device, dtype = _get_torch_device_and_dtype()

        model_name = "THUDM/CogVideoX1.5-5B"
        self._pipeline = CogVideoXPipeline.from_pretrained(model_name, torch_dtype=dtype)
        self._pipeline.to(self._device)

    def _generate_local(
        self,
        prompt: str,
        num_steps: int,
        num_frames: int,
        guidance_scale: float,
    ) -> Video:
        """Generate video using local CogVideoX diffusion model."""
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

    def _generate_luma(self, prompt: str) -> Video:
        """Generate video using Luma AI Dream Machine API."""
        import tempfile
        from pathlib import Path

        import httpx
        from lumaai import LumaAI

        client = LumaAI(auth_token=get_api_key("luma", self._api_key))

        # Create generation request
        generation = client.generations.create(prompt=prompt, model="ray-2")

        # Poll for completion
        while generation.state not in ["completed", "failed"]:
            time.sleep(3)
            assert generation.id is not None
            generation = client.generations.get(generation.id)

        if generation.state == "failed":
            raise LumaGenerationError(f"Luma generation failed: {generation.failure_reason}")

        # Download the video
        if generation.assets is None:
            raise LumaGenerationError("Luma generation completed but no assets returned")
        video_url = generation.assets.video
        if not video_url:
            raise LumaGenerationError("Luma generation completed but no video URL returned")

        with httpx.Client() as http_client:
            response = http_client.get(video_url)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(response.content)
                temp_path = Path(f.name)

        video = Video.from_path(str(temp_path))
        temp_path.unlink()
        return video

    def generate_video(
        self,
        prompt: str,
        num_steps: int = 50,
        num_frames: int = 81,
        guidance_scale: float = 6.0,
    ) -> Video:
        """Generate video from text prompt.

        Args:
            prompt: Text description of desired video content.
            num_steps: Number of diffusion steps (local backend only). Default 50.
            num_frames: Number of frames to generate (local backend only). Default 81.
            guidance_scale: Prompt guidance strength (local backend only). Default 6.0.

        Returns:
            Generated video.
        """
        if self.backend == "local":
            return self._generate_local(prompt, num_steps, num_frames, guidance_scale)
        elif self.backend == "luma":
            return self._generate_luma(prompt)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)


class ImageToVideo:
    """Generates videos from static images using video diffusion."""

    SUPPORTED_BACKENDS: list[str] = ["local", "luma", "runway"]

    def __init__(
        self,
        backend: ImageToVideoBackend | None = None,
        api_key: str | None = None,
    ):
        """Initialize image-to-video generator.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            api_key: API key for cloud backends. If None, uses environment variable.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("image_to_video")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: ImageToVideoBackend = resolved_backend  # type: ignore[assignment]
        self._api_key = api_key
        self._pipeline: Any = None
        self._device: str | None = None

    def _init_local(self) -> None:
        """Initialize local diffusion pipeline."""
        from diffusers import CogVideoXImageToVideoPipeline

        self._device, dtype = _get_torch_device_and_dtype()

        model_name = "THUDM/CogVideoX1.5-5B-I2V"
        self._pipeline = CogVideoXImageToVideoPipeline.from_pretrained(model_name, torch_dtype=dtype)
        self._pipeline.to(self._device)

    def _generate_local(
        self,
        image: Image,
        prompt: str,
        num_steps: int,
        num_frames: int,
        guidance_scale: float,
    ) -> Video:
        """Generate video using local CogVideoX I2V diffusion model."""
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

    def _generate_runway(self, image: Image, prompt: str) -> Video:
        """Generate video using Runway Gen-4 Turbo API."""
        import base64
        import io
        import tempfile
        from pathlib import Path

        import httpx
        from runwayml import RunwayML

        client = RunwayML(api_key=get_api_key("runway", self._api_key))

        # Convert PIL image to base64 data URI
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_uri = f"data:image/png;base64,{image_base64}"

        # Create image-to-video task
        task_response = client.image_to_video.create(
            model="gen4_turbo",
            prompt_image=image_uri,
            prompt_text=prompt if prompt else "",
            ratio="1280:720",
        )

        # Poll for completion
        task = client.tasks.retrieve(task_response.id)
        while task.status not in ["SUCCEEDED", "FAILED"]:
            time.sleep(5)
            task = client.tasks.retrieve(task_response.id)

        if task.status == "FAILED":
            failure_msg = getattr(task, "failure", "Unknown error")
            raise RunwayGenerationError(f"Runway generation failed: {failure_msg}")

        # Download the video - task.status is "SUCCEEDED" at this point
        output = getattr(task, "output", None)
        if not output:
            raise RunwayGenerationError("Runway generation completed but no video URL returned")
        video_url: str = output[0]

        with httpx.Client() as http_client:
            response = http_client.get(video_url)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(response.content)
                temp_path = Path(f.name)

        video = Video.from_path(str(temp_path))
        temp_path.unlink()
        return video

    def _generate_luma(self, image: Image, prompt: str) -> Video:
        """Generate video using Luma AI Dream Machine API."""
        import base64
        import io
        import tempfile
        from pathlib import Path

        import httpx
        from lumaai import LumaAI

        client = LumaAI(auth_token=get_api_key("luma", self._api_key))

        # Convert PIL image to base64 data URI
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_uri = f"data:image/png;base64,{image_base64}"

        # Create generation request with image
        generation = client.generations.create(
            prompt=prompt if prompt else "Animate this image",
            model="ray-2",
            keyframes={"frame0": {"type": "image", "url": image_uri}},
        )

        # Poll for completion
        while generation.state not in ["completed", "failed"]:
            time.sleep(3)
            assert generation.id is not None
            generation = client.generations.get(generation.id)

        if generation.state == "failed":
            raise LumaGenerationError(f"Luma generation failed: {generation.failure_reason}")

        # Download the video
        if generation.assets is None:
            raise LumaGenerationError("Luma generation completed but no assets returned")
        video_url = generation.assets.video
        if not video_url:
            raise LumaGenerationError("Luma generation completed but no video URL returned")

        with httpx.Client() as http_client:
            response = http_client.get(video_url)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(response.content)
                temp_path = Path(f.name)

        video = Video.from_path(str(temp_path))
        temp_path.unlink()
        return video

    def generate_video(
        self,
        image: Image,
        prompt: str = "",
        num_steps: int = 50,
        num_frames: int = 81,
        guidance_scale: float = 6.0,
    ) -> Video:
        """Generate video animation from a static image.

        Args:
            image: Input PIL image to animate.
            prompt: Text description to guide the animation.
            num_steps: Number of diffusion steps (local backend only). Default 50.
            num_frames: Number of frames to generate (local backend only). Default 81.
            guidance_scale: Prompt guidance strength (local backend only). Default 6.0.

        Returns:
            Generated animated video.
        """
        if self.backend == "local":
            return self._generate_local(image, prompt, num_steps, num_frames, guidance_scale)
        elif self.backend == "runway":
            return self._generate_runway(image, prompt)
        elif self.backend == "luma":
            return self._generate_luma(image, prompt)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)
