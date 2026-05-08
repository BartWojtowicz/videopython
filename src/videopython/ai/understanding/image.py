"""Visual understanding using local Qwen 3.5 VLM models."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Literal

import numpy as np
from PIL import Image

from videopython.ai._device import log_device_initialization, select_device
from videopython.base.description import SceneDescription

logger = logging.getLogger(__name__)


SceneVLMModelSize = Literal["4b", "9b", "27b"]


SCENE_VLM_MODEL_IDS: dict[SceneVLMModelSize, str] = {
    "4b": "Qwen/Qwen3.5-4B",
    "9b": "Qwen/Qwen3.5-9B",
    "27b": "Qwen/Qwen3.5-27B",
}
DEFAULT_SCENE_VLM_MODEL_SIZE: SceneVLMModelSize = "4b"

# Free-VRAM threshold below which the 27b model size logs a loud warning.
# 45 GB is generous: 27B FP16 weights are ~54 GB, but users with 48 GB
# cards routinely run with their own quantization layer, so we warn
# instead of raising.
_LARGE_MODEL_VRAM_WARN_GB = 45.0

# Closed enum for shot_type. Constrained to keep parse rate stable on
# small models; "other" is the fallback when the scene doesn't match.
_VALID_SHOT_TYPES: tuple[str, ...] = (
    "wide",
    "medium",
    "close-up",
    "extreme close-up",
    "establishing",
    "other",
)


_STRUCTURED_PROMPT = (
    "Describe the visual content of these frames as a JSON object with "
    "exactly these three keys:\n"
    '  - "caption": one concise sentence summarizing the scene.\n'
    '  - "subjects": a list of the main on-screen subjects (people, '
    "objects, named entities). Open list, lowercase noun phrases.\n"
    '  - "shot_type": one of "wide", "medium", "close-up", '
    '"extreme close-up", "establishing", "other".\n\n'
    "Examples:\n"
    '{"caption": "A woman in a red coat walks along a snowy street '
    'at dusk.", "subjects": ["woman", "snowy street"], '
    '"shot_type": "wide"}\n'
    '{"caption": "Two men sit across a desk in conversation.", '
    '"subjects": ["two men", "desk"], "shot_type": "medium"}\n'
    '{"caption": "Close view of hands assembling a circuit board.", '
    '"subjects": ["hands", "circuit board"], "shot_type": "close-up"}\n\n'
    "Respond with ONLY the JSON object. No prose, no markdown fence, "
    "no preamble."
)

_RETRY_PROMPT = (
    "Your previous reply was not valid JSON. Reply with ONLY a single "
    'JSON object with keys "caption" (string), "subjects" (list of '
    'strings), and "shot_type" (one of "wide", "medium", "close-up", '
    '"extreme close-up", "establishing", "other"). No prose, no '
    "markdown fence, nothing outside the braces."
)


class SceneVLM:
    """Generates structured scene descriptions with local Qwen3.5.

    ``model_size`` maps to Qwen3.5 dense vision-capable variants:

        4b   -> Qwen/Qwen3.5-4B  (~8 GB FP16; default)
        9b   -> Qwen/Qwen3.5-9B  (~18 GB FP16; 24 GB GPU when solo)
        27b  -> Qwen/Qwen3.5-27B (~54 GB FP16; needs ≥48 GB)

    All sizes return a fully-populated ``SceneDescription``. JSON parse
    failures fall back to raw-text-as-caption with empty subjects and
    None shot_type; that path is the rare exception, not a tier.
    """

    DEFAULT_MAX_IMAGE_PIXELS: int = 384 * 384

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        model_size: SceneVLMModelSize = DEFAULT_SCENE_VLM_MODEL_SIZE,
        max_image_pixels: int | None = None,
    ):
        if model_size not in SCENE_VLM_MODEL_IDS:
            supported = ", ".join(SCENE_VLM_MODEL_IDS)
            raise ValueError(f"model_size must be one of: {supported}")

        self.model_size: SceneVLMModelSize = model_size
        self.model_name = model_name or SCENE_VLM_MODEL_IDS[model_size]
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_image_pixels = max_image_pixels if max_image_pixels is not None else self.DEFAULT_MAX_IMAGE_PIXELS
        self._processor: Any = None
        self._model: Any = None

        if model_size == "27b":
            self._warn_if_vram_under_large_model_floor()

    @staticmethod
    def _warn_if_vram_under_large_model_floor() -> None:
        """Loud WARNING when ``model_size='27b'`` is requested on a small card.

        Does not raise -- a knowledgeable user may run the 27B model with
        their own quantization layer or accept device off-loading. The
        warning makes the eventual OOM (deep inside ``from_pretrained``)
        easier to diagnose.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning(
                    "SceneVLM model_size='27b' requested but CUDA is not "
                    "available. 27B FP16 weights are ~54 GB; running on "
                    "CPU/MPS is likely to OOM."
                )
                return

            free_bytes, _total = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
            if free_gb < _LARGE_MODEL_VRAM_WARN_GB:
                logger.warning(
                    "SceneVLM model_size='27b' requested with %.1f GB free VRAM. "
                    "Qwen3.5-27B FP16 needs ~54 GB for weights alone -- expect "
                    "OOM during from_pretrained unless you wired up "
                    "quantization or device offloading.",
                    free_gb,
                )
        except ImportError:
            pass

    def _init_local(self) -> None:
        """Initialize local Qwen3.5 model."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore[attr-defined]

        t0 = time.perf_counter()
        requested_device = self.device
        resolved_device = select_device(self.device, mps_allowed=True)

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        # Save and restore default dtype -- transformers torch_dtype="auto" can
        # mutate torch.get_default_dtype(), which breaks concurrent models
        # (e.g. Whisper) that expect float32.
        saved_dtype = torch.get_default_dtype()
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(self.model_name, torch_dtype="auto")
        finally:
            torch.set_default_dtype(saved_dtype)
        self._model.to(resolved_device)
        self._model.eval()
        self.device = resolved_device

        log_device_initialization(
            "SceneVLM",
            requested_device=requested_device,
            resolved_device=resolved_device,
        )
        logger.info(
            "SceneVLM(%s, model_size=%s) model weights loaded in %.2fs",
            self.model_name,
            self.model_size,
            time.perf_counter() - t0,
        )

    def unload(self) -> None:
        """Release model + processor for ``low_memory`` parity with other stages.

        Mirrors ``MarianTranslator.unload`` / ``Qwen3Translator.unload``. Safe
        to call before ``_init_local`` -- it just clears already-None fields.
        """
        self._model = None
        self._processor = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

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
    ) -> SceneDescription:
        """Analyze one frame and return a structured scene description."""
        frame = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        return self.analyze_scene([frame], prompt=prompt)

    def analyze_scene(
        self,
        images: list[np.ndarray | Image.Image],
        prompt: str | None = None,
    ) -> SceneDescription:
        """Analyze a scene with multiple frames and return a structured description.

        Uses few-shot JSON prompting with one parse-retry. If both attempts
        fail to produce valid JSON, falls back to a raw-text caption with
        empty subjects and ``shot_type=None``.
        """
        if not images:
            raise ValueError("`images` must contain at least one frame")

        pil_images = [
            self._downscale_image(Image.fromarray(img) if isinstance(img, np.ndarray) else img) for img in images
        ]

        user_prompt = prompt or _STRUCTURED_PROMPT
        raw_first = self._generate_one(pil_images, user_prompt)
        parsed = _try_parse_scene_json(raw_first)
        if parsed is not None:
            return parsed

        logger.info("SceneVLM JSON parse failed on first attempt; retrying with tightened prompt")
        raw_retry = self._generate_one(pil_images, _RETRY_PROMPT)
        parsed = _try_parse_scene_json(raw_retry)
        if parsed is not None:
            return parsed

        # Final fallback: surface the raw text as a caption so the scene
        # still gets *something* useful, just without structured fields.
        fallback_text = " ".join(raw_first.split()).strip() or "No scene description"
        logger.warning("SceneVLM JSON parse failed after retry; using raw-text fallback")
        return SceneDescription(caption=fallback_text, subjects=[], shot_type=None)

    def _generate_one(self, pil_images: list[Image.Image], user_prompt: str) -> str:
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": user_prompt})
        messages = [{"role": "user", "content": content}]
        outputs = self._generate_from_message_batch([messages])
        return outputs[0]

    def _generate_from_message_batch(self, messages_batch: list[list[dict[str, Any]]]) -> list[str]:
        """Run batch generation for one or more multimodal chat messages."""
        import torch
        from qwen_vl_utils import process_vision_info  # type: ignore

        if self._model is None:
            self._init_local()

        texts = [
            self._processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for msg in messages_batch
        ]

        image_inputs, video_inputs = process_vision_info(messages_batch)
        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
            "images": image_inputs,
        }
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs

        num_images = sum(
            len(items) if isinstance(items, list) else 1 for items in (processor_kwargs.get("images") or [])
        )

        inputs = self._processor(**processor_kwargs)
        inputs = inputs.to(self.device) if hasattr(inputs, "to") else {k: v.to(self.device) for k, v in inputs.items()}

        generation_config = self._generation_config_for_run()
        if generation_config is not None:
            generation_config.max_new_tokens = self.max_new_tokens
            generation_kwargs: dict[str, Any] = {"generation_config": generation_config}
        elif self.temperature > 0:
            generation_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": self.temperature,
            }
        else:
            generation_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": False}

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


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_scene_json(raw: str) -> SceneDescription | None:
    """Tolerant JSON parse for ``SceneDescription``.

    Strategy mirrors ``Qwen3Translator``: try ``json.loads`` directly, then
    extract the first ``{...}`` block and re-parse. Returns ``None`` on
    failure so the caller can decide whether to retry or fall back.
    Validates ``shot_type`` against the closed enum and drops it (sets to
    None) if the model invented a value.
    """
    if not raw:
        return None

    obj: Any | None = None
    text = raw.strip()
    for candidate in (text, _extract_json_block(text)):
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            obj = parsed
            break

    if obj is None:
        return None

    caption_raw = obj.get("caption")
    if not isinstance(caption_raw, str) or not caption_raw.strip():
        return None
    caption = " ".join(caption_raw.split()).strip()

    subjects_raw = obj.get("subjects", [])
    if isinstance(subjects_raw, list):
        subjects = [str(s).strip() for s in subjects_raw if isinstance(s, str | int | float) and str(s).strip()]
    else:
        subjects = []

    shot_type_raw = obj.get("shot_type")
    if isinstance(shot_type_raw, str):
        normalized = shot_type_raw.strip().lower()
        shot_type: str | None = normalized if normalized in _VALID_SHOT_TYPES else None
    else:
        shot_type = None

    return SceneDescription(caption=caption, subjects=subjects, shot_type=shot_type)


def _extract_json_block(text: str) -> str | None:
    match = _JSON_BLOCK_RE.search(text)
    if match is None:
        return None
    return match.group(0)
