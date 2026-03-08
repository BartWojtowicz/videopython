"""Visual understanding using local Qwen VLM models."""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import numpy as np
from PIL import Image

from videopython.ai._device import log_device_initialization, select_device

logger = logging.getLogger(__name__)

SCENE_VLM_MODEL_IDS: dict[str, str] = {
    "2b": "Qwen/Qwen3-VL-2B-Instruct",
    "4b": "Qwen/Qwen3-VL-4B-Instruct",
}
DEFAULT_SCENE_VLM_MODEL_SIZE: Literal["2b", "4b"] = "4b"

_DEFAULT_PROMPT = (
    "Describe the visual content of these frames in 1-3 concise sentences. "
    "Focus on the main subjects, actions, and setting. Be specific and factual."
)


class SceneVLM:
    """Generates scene captions with local Qwen3-VL."""

    # Default pixel budget per image for scene captioning. Qwen3-VL tiles
    # images into 28x28 patches; fewer pixels = fewer vision tokens = faster
    # inference.  384x384 = 147456 is plenty for scene-level captioning.
    DEFAULT_MAX_IMAGE_PIXELS: int = 384 * 384

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        model_size: Literal["2b", "4b"] = DEFAULT_SCENE_VLM_MODEL_SIZE,
        max_image_pixels: int | None = None,
    ):
        if model_size not in SCENE_VLM_MODEL_IDS:
            supported = ", ".join(sorted(SCENE_VLM_MODEL_IDS))
            raise ValueError(f"model_size must be one of: {supported}")

        self.model_size = model_size
        self.model_name = model_name or SCENE_VLM_MODEL_IDS[model_size]
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_image_pixels = max_image_pixels if max_image_pixels is not None else self.DEFAULT_MAX_IMAGE_PIXELS
        self._processor: Any = None
        self._model: Any = None

    def _init_local(self) -> None:
        """Initialize local Qwen3-VL model."""
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore[attr-defined]

        t0 = time.perf_counter()
        requested_device = self.device
        resolved_device = select_device(self.device, mps_allowed=True)

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(self.model_name, dtype="auto")
        self._model.to(resolved_device)
        self._model.eval()
        self.device = resolved_device

        log_device_initialization(
            "SceneVLM",
            requested_device=requested_device,
            resolved_device=resolved_device,
        )
        logger.info("SceneVLM model weights loaded in %.2fs", time.perf_counter() - t0)

    def _downscale_image(self, img: Image.Image) -> Image.Image:
        """Downscale image to fit within max_image_pixels budget, preserving aspect ratio."""
        w, h = img.size
        pixels = w * h
        if pixels <= self.max_image_pixels:
            return img
        scale = (self.max_image_pixels / pixels) ** 0.5
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return img.resize((new_w, new_h), Image.LANCZOS)

    def _generation_config_for_run(self) -> Any | None:
        base_config = getattr(self._model, "generation_config", None)
        if base_config is None or not hasattr(base_config, "to_dict"):
            return None

        config = base_config.__class__.from_dict(base_config.to_dict())
        if self.temperature > 0:
            config.do_sample = True
            config.temperature = self.temperature
            return config

        config.do_sample = False
        for name, value in (("temperature", 1.0), ("top_p", 1.0), ("top_k", 50)):
            if hasattr(config, name):
                setattr(config, name, value)
        return config

    def analyze_frame(
        self,
        image: np.ndarray | Image.Image,
        prompt: str | None = None,
    ) -> str:
        """Analyze one frame and return a plain-text caption."""
        frame = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        return self.analyze_scene([frame], prompt=prompt)

    def analyze_scene(
        self,
        images: list[np.ndarray | Image.Image],
        prompt: str | None = None,
    ) -> str:
        """Analyze a scene with multiple frames and return a plain-text caption."""
        if not images:
            raise ValueError("`images` must contain at least one frame")

        pil_images = [
            self._downscale_image(Image.fromarray(img) if isinstance(img, np.ndarray) else img) for img in images
        ]
        user_prompt = prompt or _DEFAULT_PROMPT
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": user_prompt})
        messages = [{"role": "user", "content": content}]
        outputs = self._generate_from_message_batch([messages])
        caption = " ".join(outputs[0].split()).strip()
        return caption or "No scene description"

    def _generate_from_message_batch(self, messages_batch: list[list[dict[str, Any]]]) -> list[str]:
        """Run batch generation for one or more multimodal chat messages."""
        import torch

        if self._model is None:
            self._init_local()

        texts = [
            self._processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for msg in messages_batch
        ]

        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }

        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except ImportError:
            image_inputs = [
                [
                    item["image"]
                    for part in message
                    for item in part.get("content", [])
                    if isinstance(item, dict) and item.get("type") == "image" and "image" in item
                ]
                for message in messages_batch
            ]
            if all(len(items) == 1 for items in image_inputs):
                processor_kwargs["images"] = [items[0] for items in image_inputs]
            else:
                processor_kwargs["images"] = image_inputs
        else:
            image_inputs, video_inputs = process_vision_info(messages_batch)
            processor_kwargs["images"] = image_inputs
            if video_inputs is not None:
                processor_kwargs["videos"] = video_inputs

        num_images = sum(
            len(items) if isinstance(items, list) else 1 for items in (processor_kwargs.get("images") or [])
        )

        inputs = self._processor(**processor_kwargs)
        inputs = inputs.to(self.device) if hasattr(inputs, "to") else {k: v.to(self.device) for k, v in inputs.items()}

        generation_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        generation_config = self._generation_config_for_run()
        if generation_config is not None:
            generation_kwargs["generation_config"] = generation_config
        elif self.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": self.temperature})
        else:
            generation_kwargs["do_sample"] = False

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **generation_kwargs)
        logger.info(
            "SceneVLM inference: %.2fs, %d images, %d messages", time.perf_counter() - t0, num_images, len(texts)
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], output_ids, strict=False)
        ]
        output_texts = self._processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return [text.strip() for text in output_texts]
