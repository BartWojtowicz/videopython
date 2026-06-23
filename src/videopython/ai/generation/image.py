"""Image generation using local diffusion models (Qwen-Image)."""

from __future__ import annotations

from typing import Any

from PIL import Image

from videopython.ai._device import log_device_initialization, select_device
from videopython.ai._predictor import ManagedPredictor
from videopython.ai._revisions import pinned

_MODEL_NAME = "Qwen/Qwen-Image-2512"

# Qwen-Image recommends appending a quality "magic" suffix to the prompt
# (verbatim from the model card).
_POSITIVE_MAGIC = ", Ultra HD, 4K, cinematic composition."


class TextToImage(ManagedPredictor):
    """Generates images from text descriptions using local models (Qwen-Image, Apache-2.0)."""

    _model_attrs = ("_pipeline",)

    def __init__(self, device: str | None = None):
        self.device = device
        self._pipeline: Any = None

    def _init_local(self) -> None:
        """Initialize the local Qwen-Image diffusion pipeline (CUDA-only)."""
        import torch

        from videopython.ai._optional import require

        requested_device = self.device
        device = select_device(self.device, mps_allowed=False)
        if device != "cuda":
            raise RuntimeError("TextToImage requires a CUDA GPU; Qwen-Image (~20B) is impractical on CPU/MPS.")

        QwenImagePipeline = require("diffusers", feature="TextToImage").QwenImagePipeline
        self._pipeline = QwenImagePipeline.from_pretrained(
            _MODEL_NAME,
            revision=pinned(_MODEL_NAME),
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        # ~20B params (Qwen2.5-VL text encoder + transformer + VAE). Offload submodules
        # to the GPU on demand so it fits a single GPU; offload manages device placement,
        # so we must NOT also call .to("cuda").
        self._pipeline.enable_model_cpu_offload()
        self._pipeline.enable_vae_tiling()

        self.device = device
        log_device_initialization(
            "TextToImage",
            requested_device=requested_device,
            resolved_device=device,
        )

    def generate_image(
        self,
        prompt: str,
        *,
        negative_prompt: str = " ",
        true_cfg_scale: float = 4.0,
        num_inference_steps: int = 50,
        width: int = 1328,
        height: int = 1328,
        add_magic: bool = True,
        seed: int = 42,
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Qwen-Image uses ``true_cfg_scale`` (not ``guidance_scale``) for
        classifier-free guidance; a non-empty ``negative_prompt`` (default a single
        space) is required to enable it. ``add_magic`` appends the model's
        recommended quality suffix to ``prompt``.
        """
        import torch

        if self._pipeline is None:
            self._init_local()

        full_prompt = prompt + _POSITIVE_MAGIC if add_magic else prompt
        return self._pipeline(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images[0]
