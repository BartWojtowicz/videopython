"""Video generation using local diffusion models (Wan2.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from videopython.ai._device import log_device_initialization, select_device
from videopython.ai._predictor import ManagedPredictor
from videopython.ai._revisions import pinned
from videopython.base.video import Video

if TYPE_CHECKING:
    from PIL.Image import Image

# Canonical Wan2.2 negative prompt (from the diffusers WanPipeline docstring example).
_WAN_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def _get_torch_device_and_dtype(device: str | None) -> tuple[str, Any]:
    """Get the best available torch device and transformer dtype for Wan2.2.

    The Wan VAE is always loaded in float32 for decode quality; this dtype applies
    to the transformer/pipeline only (bf16 on CUDA, float32 otherwise).
    """
    import torch

    selected_device = select_device(device, mps_allowed=False)
    if selected_device == "cuda":
        return selected_device, torch.bfloat16
    return selected_device, torch.float32


class TextToVideo(ManagedPredictor):
    """Generates videos from text descriptions using Wan2.2-T2V (Apache-2.0)."""

    _model_attrs = ("_pipeline",)

    def __init__(self, device: str | None = None):
        self.device = device
        self._pipeline: Any = None

    def _init_local(self) -> None:
        import torch

        from videopython.ai._optional import require

        diffusers = require("diffusers", feature="TextToVideo")
        WanPipeline = diffusers.WanPipeline
        AutoencoderKLWan = diffusers.AutoencoderKLWan

        requested_device = self.device
        device, dtype = _get_torch_device_and_dtype(self.device)

        model_name = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        revision = pinned(model_name)
        # VAE stays float32 for decode quality; the transformer/pipeline use ``dtype``.
        vae = AutoencoderKLWan.from_pretrained(
            model_name, subfolder="vae", revision=revision, torch_dtype=torch.float32
        )
        self._pipeline = WanPipeline.from_pretrained(model_name, vae=vae, revision=revision, torch_dtype=dtype)

        if device == "cuda":
            # A14B is a MoE (high+low-noise experts), too large for a single-GPU
            # .to("cuda"); offload submodules on demand (offload manages placement,
            # so do NOT also call .to("cuda")).
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline.to(device)

        self.device = device
        log_device_initialization(
            "TextToVideo",
            requested_device=requested_device,
            resolved_device=device,
        )

    def generate_video(
        self,
        prompt: str,
        num_steps: int = 40,
        num_frames: int = 81,
        guidance_scale: float = 4.0,
    ) -> Video:
        """Generate video from text prompt."""
        import torch

        if self._pipeline is None:
            self._init_local()

        # output_type="pil" is required: the "np" default returns float32 in [0, 1],
        # which the uint8 cast below would floor to an all-black video.
        video_frames = self._pipeline(
            prompt=prompt,
            negative_prompt=_WAN_NEGATIVE_PROMPT,
            height=720,
            width=1280,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=3.0,  # low-noise MoE expert; None would reuse guidance_scale
            output_type="pil",
            generator=torch.Generator(device=self.device).manual_seed(42),
        ).frames[0]
        video_frames = np.asarray(video_frames, dtype=np.uint8)
        return Video.from_frames(video_frames, fps=16.0)


class ImageToVideo(ManagedPredictor):
    """Generates videos from static images using Wan2.2-I2V (Apache-2.0)."""

    _model_attrs = ("_pipeline",)

    def __init__(self, device: str | None = None):
        self.device = device
        self._pipeline: Any = None

    def _init_local(self) -> None:
        import torch

        from videopython.ai._optional import require

        diffusers = require("diffusers", feature="ImageToVideo")
        WanImageToVideoPipeline = diffusers.WanImageToVideoPipeline
        AutoencoderKLWan = diffusers.AutoencoderKLWan

        requested_device = self.device
        device, dtype = _get_torch_device_and_dtype(self.device)

        model_name = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        revision = pinned(model_name)
        vae = AutoencoderKLWan.from_pretrained(
            model_name, subfolder="vae", revision=revision, torch_dtype=torch.float32
        )
        self._pipeline = WanImageToVideoPipeline.from_pretrained(
            model_name, vae=vae, revision=revision, torch_dtype=dtype
        )

        if device == "cuda":
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline.to(device)

        self.device = device
        log_device_initialization(
            "ImageToVideo",
            requested_device=requested_device,
            resolved_device=device,
        )

    def _resize_to_model_grid(self, image: Image) -> tuple[Image, int, int]:
        """Resize ``image`` to Wan's area budget, snapped to the model's spatial grid.

        Returns the resized image plus the ``(height, width)`` to request, derived
        from the input aspect ratio against Wan's 480x832 area (per the model card).
        """
        max_area = 480 * 832
        aspect_ratio = image.height / image.width
        mod_value = self._pipeline.vae_scale_factor_spatial * self._pipeline.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        return image.resize((width, height)), height, width

    def generate_video(
        self,
        image: Image,
        prompt: str = "",
        num_steps: int = 40,
        num_frames: int = 81,
        guidance_scale: float = 3.5,
    ) -> Video:
        """Generate video animation from a static image."""
        import torch

        if self._pipeline is None:
            self._init_local()

        image, height, width = self._resize_to_model_grid(image)
        video_frames = self._pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=_WAN_NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            # No guidance_scale_2: the low-noise MoE expert reuses guidance_scale (its None default).
            output_type="pil",
            generator=torch.Generator(device=self.device).manual_seed(42),
        ).frames[0]
        video_frames = np.asarray(video_frames, dtype=np.uint8)
        return Video.from_frames(video_frames, fps=16.0)
